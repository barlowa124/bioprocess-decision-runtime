from __future__ import annotations

from collections import Counter
import hashlib
import itertools
import re
from typing import Any

from .gemma_ir import verify_gemma_ir
from .gemma_reduction_backend import _profile_call, verify_gemma_reduction_backend_binding
from .gemma_reduction_semantics import exact_sum_then_bfloat16
from .serialization import canonical_json

try:
    import torch
except ImportError:
    torch = None


PROBE_COUNT = 1024


def _require_cuda() -> None:
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError("WMMA position probing requires CUDA PyTorch")


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def build_probe_specifications(inner_dimension: int = 640) -> list[dict[str, Any]]:
    if (
        not isinstance(inner_dimension, int)
        or isinstance(inner_dimension, bool)
        or inner_dimension < 640
        or inner_dimension % 16 != 0
    ):
        raise ValueError("WMMA probe inner dimension must be a multiple of 16 and at least 640")
    tiles = inner_dimension // 16
    base_values = (0x4E80, 0x3F80, 0xCE80, 0x3F80)
    permutations = sorted(set(itertools.permutations(base_values)))
    probes = []
    for tile in range(min(tiles, 40)):
        for permutation_index, values in enumerate(permutations):
            positions = [tile * 16 + index for index in range(4)]
            probes.append(
                {
                    "family": "within_fragment_permutation",
                    "tile": tile,
                    "lane_start": 0,
                    "permutation_index": permutation_index,
                    "positions": positions,
                    "value_bits": [f"0x{value:04x}" for value in values],
                }
            )
    for tile in range(min(tiles, 40)):
        for lane_start in range(13):
            positions = [tile * 16 + lane_start + index for index in range(4)]
            probes.append(
                {
                    "family": "within_fragment_lane_shift",
                    "tile": tile,
                    "lane_start": lane_start,
                    "permutation_index": 0,
                    "positions": positions,
                    "value_bits": [f"0x{value:04x}" for value in base_values],
                }
            )
    for offset in range(PROBE_COUNT - len(probes)):
        selected_tiles = [
            (offset + stride * 7) % min(tiles, 40) for stride in range(4)
        ]
        lane = offset % 16
        positions = [tile * 16 + lane for tile in selected_tiles]
        probes.append(
            {
                "family": "cross_fragment",
                "tile": None,
                "lane_start": lane,
                "permutation_index": offset,
                "positions": positions,
                "value_bits": [f"0x{value:04x}" for value in base_values],
            }
        )
    if len(probes) != PROBE_COUNT:
        raise RuntimeError("Unexpected WMMA probe count")
    for index, probe in enumerate(probes):
        probe["probe_index"] = index
        left = [0] * inner_dimension
        for position, value in zip(probe["positions"], probe["value_bits"]):
            left[position] = int(value, 16)
        exact = exact_sum_then_bfloat16(left, [0x3F80] * inner_dimension)
        probe["exact_result_bits"] = f"0x{exact:04x}"
        probe["specification_sha256"] = _sha256(
            {
                key: value
                for key, value in probe.items()
                if key != "specification_sha256"
            }
        )
    return probes


def build_wmma_probe_certificate(
    program: dict[str, Any],
    reduction_certificate: dict[str, Any],
    nsight_suite: dict[str, Any],
    backend_binding: dict[str, Any],
) -> dict[str, Any]:
    _require_cuda()
    if not verify_gemma_ir(program)["valid"]:
        raise ValueError("Gemma IR is invalid")
    backend_verification = verify_gemma_reduction_backend_binding(
        program, reduction_certificate, nsight_suite, backend_binding
    )
    if not backend_verification["valid"]:
        raise ValueError("Reduction backend binding is invalid")
    configuration = program["configuration"]
    inner = configuration["hidden_size"]
    outputs = configuration["attention_heads"] * configuration["head_dimension"]
    sequence = backend_binding["sequence_length"]
    if outputs != PROBE_COUNT:
        raise ValueError("The current probe layout requires exactly 1,024 query outputs")
    probes = build_probe_specifications(inner)
    weights_bits = torch.zeros((PROBE_COUNT, inner), dtype=torch.uint16)
    for probe in probes:
        for position, value in zip(probe["positions"], probe["value_bits"]):
            weights_bits[probe["probe_index"], position] = int(value, 16)
    weights = weights_bits.view(torch.bfloat16).cuda()
    inputs = torch.ones((sequence, inner), dtype=torch.bfloat16, device="cuda")
    result, kernel_names = _profile_call(lambda: torch.nn.functional.linear(inputs, weights))
    input_rows_equal = all(
        torch.equal(inputs[0], inputs[row]) for row in range(1, inputs.shape[0])
    )
    repeated_rows_equal = all(
        torch.equal(result[0], result[row]) for row in range(1, result.shape[0])
    )
    result_bits = result[0].detach().cpu().view(torch.uint16).tolist()
    records = []
    for probe, actual in zip(probes, result_bits):
        body = {
            **probe,
            "actual_result_bits": f"0x{actual:04x}",
            "matches_exact_sum": actual == int(probe["exact_result_bits"], 16),
        }
        records.append({**body, "record_sha256": _sha256(body)})
    histogram = dict(sorted(Counter(record["actual_result_bits"] for record in records).items()))
    family_histograms = {
        family: dict(
            sorted(
                Counter(
                    record["actual_result_bits"]
                    for record in records
                    if record["family"] == family
                ).items()
            )
        )
        for family in sorted({record["family"] for record in records})
    }
    query_kernel_names = sorted(
        {
            kernel
            for record in backend_binding["records"]
            if record["role"] == "query_projection"
            for kernel in record["cuda_kernel_names"]
        }
    )
    body = {
        "schema_version": 1,
        "scope": "Integrity-bound controlled operand-position records through the exact Gemma query-projection shape and CUDA kernel; identical exact sums with differing outputs characterize hidden accumulation order, while cross-fragment coverage is limited and only explicit CUDA re-execution checks observed bits.",
        "program_sha256": program["program_sha256"],
        "reduction_backend_binding_sha256": backend_binding["binding_sha256"],
        "query_projection_shape": {
            "input": [sequence, inner],
            "weight": [PROBE_COUNT, inner],
            "output": [sequence, PROBE_COUNT],
        },
        "kernel_names": kernel_names,
        "expected_query_kernel_names": query_kernel_names,
        "kernel_identity_matches_backend_binding": sorted(kernel_names)
        == query_kernel_names,
        "input_rows_equal": input_rows_equal,
        "output_rows_equal": repeated_rows_equal,
        "probe_count": len(records),
        "cross_fragment_probe_count": sum(
            record["family"] == "cross_fragment" for record in records
        ),
        "cross_fragment_full_lane_tile_coverage": False,
        "records": records,
        "result_histogram": histogram,
        "family_histograms": family_histograms,
        "distinct_result_count": len(histogram),
        "all_exact_sums_identical": len(
            {record["exact_result_bits"] for record in records}
        )
        == 1,
        "all_results_match_exact_sum": all(
            record["matches_exact_sum"] for record in records
        ),
        "operand_position_invariance_established": len(histogram) == 1,
        "wmma_accumulator_mapping_identified": False,
        "reduction_order_semantics_established": False,
        "hardware_instruction_semantics_established": False,
    }
    return {**body, "certificate_sha256": _sha256(body)}


def verify_wmma_probe_certificate(
    program: dict[str, Any],
    reduction_certificate: dict[str, Any],
    nsight_suite: dict[str, Any],
    backend_binding: dict[str, Any],
    certificate: dict[str, Any],
    reexecute: bool = False,
) -> dict[str, Any]:
    if not isinstance(certificate, dict):
        return {"valid": False}
    body = {
        key: value for key, value in certificate.items() if key != "certificate_sha256"
    }
    try:
        certificate_hash_valid = _sha256(body) == certificate.get("certificate_sha256")
    except (TypeError, ValueError):
        return {"valid": False}
    backend_valid = verify_gemma_reduction_backend_binding(
        program, reduction_certificate, nsight_suite, backend_binding
    )["valid"]
    inner = program.get("configuration", {}).get("hidden_size")
    try:
        expected_probes = build_probe_specifications(inner)
    except (TypeError, ValueError, RuntimeError):
        expected_probes = []
    records = certificate.get("records")

    def record_valid(record: Any, expected: dict[str, Any]) -> bool:
        if not isinstance(record, dict):
            return False
        try:
            record_body = {
                key: value for key, value in record.items() if key != "record_sha256"
            }
            return bool(
                all(record.get(key) == value for key, value in expected.items())
                and re.fullmatch(
                    r"0x[0-9a-f]{4}", str(record.get("actual_result_bits", ""))
                )
                is not None
                and record.get("matches_exact_sum")
                is (
                    record.get("actual_result_bits")
                    == record.get("exact_result_bits")
                )
                and record.get("record_sha256") == _sha256(record_body)
            )
        except (TypeError, ValueError, AttributeError):
            return False

    records_valid = bool(
        isinstance(records, list)
        and len(records) == len(expected_probes) == PROBE_COUNT
        and all(
            record_valid(record, expected)
            for record, expected in zip(records, expected_probes)
        )
    )
    histogram = (
        dict(sorted(Counter(record["actual_result_bits"] for record in records).items()))
        if records_valid
        else {}
    )
    family_histograms = (
        {
            family: dict(
                sorted(
                    Counter(
                        record["actual_result_bits"]
                        for record in records
                        if record["family"] == family
                    ).items()
                )
            )
            for family in sorted({record["family"] for record in records})
        }
        if records_valid
        else {}
    )
    query_kernel_names = sorted(
        {
            kernel
            for record in backend_binding.get("records", [])
            if isinstance(record, dict) and record.get("role") == "query_projection"
            for kernel in record.get("cuda_kernel_names", [])
        }
    )
    kernels = certificate.get("kernel_names")
    claims_consistent = bool(
        certificate.get("schema_version") == 1
        and certificate.get("program_sha256") == program.get("program_sha256")
        and certificate.get("reduction_backend_binding_sha256")
        == backend_binding.get("binding_sha256")
        and certificate.get("probe_count") == PROBE_COUNT
        and certificate.get("cross_fragment_probe_count") == 24
        and certificate.get("cross_fragment_full_lane_tile_coverage") is False
        and certificate.get("result_histogram") == histogram
        and certificate.get("family_histograms") == family_histograms
        and certificate.get("distinct_result_count") == len(histogram)
        and certificate.get("expected_query_kernel_names") == query_kernel_names
        and isinstance(kernels, list)
        and certificate.get("kernel_identity_matches_backend_binding")
        is (sorted(kernels) == query_kernel_names)
        and certificate.get("input_rows_equal") is True
        and certificate.get("output_rows_equal") is True
        and certificate.get("all_exact_sums_identical") is True
        and certificate.get("all_results_match_exact_sum")
        is all(record["matches_exact_sum"] for record in records)
        and certificate.get("operand_position_invariance_established")
        is (len(histogram) == 1)
        and certificate.get("wmma_accumulator_mapping_identified") is False
        and certificate.get("reduction_order_semantics_established") is False
        and certificate.get("hardware_instruction_semantics_established") is False
    ) if records_valid else False
    reexecution_exact = True
    if reexecute:
        try:
            recomputed = build_wmma_probe_certificate(
                program, reduction_certificate, nsight_suite, backend_binding
            )
            reexecution_exact = recomputed == certificate
        except (RuntimeError, TypeError, ValueError):
            reexecution_exact = False
    valid = all(
        (
            certificate_hash_valid,
            backend_valid,
            records_valid,
            claims_consistent,
            certificate.get("kernel_identity_matches_backend_binding") is True,
            reexecution_exact,
        )
    )
    return {
        "valid": valid,
        "certificate_hash_valid": certificate_hash_valid,
        "backend_binding_valid": backend_valid,
        "records_valid": records_valid,
        "claims_consistent": claims_consistent,
        "reexecution_performed": reexecute,
        "reexecution_exact": reexecution_exact,
        "probe_count": len(records) if isinstance(records, list) else 0,
        "distinct_result_count": len(histogram),
        "all_results_match_exact_sum": certificate.get(
            "all_results_match_exact_sum"
        ),
        "operand_position_invariance_established": certificate.get(
            "operand_position_invariance_established"
        ),
        "wmma_accumulator_mapping_identified": certificate.get(
            "wmma_accumulator_mapping_identified"
        ),
    }
