from __future__ import annotations

from collections import Counter
import hashlib
import re
from typing import Any

from .gemma_float_semantics import decode_finite_bfloat16
from .gemma_ir import verify_gemma_ir
from .gemma_reduction_backend import _profile_call, verify_gemma_reduction_backend_binding
from .gemma_reduction_semantics import exact_sum_then_bfloat16
from .serialization import canonical_json

try:
    import torch
except ImportError:
    torch = None


MAGNITUDES_PER_SIGN = 64
PLACEMENTS = (
    ("lower_cancel_small_upper_8", 0, 1, 8),
    ("lower_cancel_small_upper_15", 6, 7, 15),
    ("upper_cancel_small_lower_0", 8, 9, 0),
    ("upper_cancel_small_lower_7", 14, 15, 7),
    ("split_positive_lower_small_upper", 0, 8, 9),
    ("split_negative_lower_small_upper", 8, 0, 9),
    ("all_lower", 0, 1, 2),
    ("all_upper", 8, 9, 10),
)
PROBE_COUNT = len(PLACEMENTS) * MAGNITUDES_PER_SIGN * 2


def _require_cuda() -> None:
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError("WMMA magnitude probing requires CUDA PyTorch")


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _magnitude_patterns() -> list[dict[str, Any]]:
    patterns = []
    for magnitude_index in range(MAGNITUDES_PER_SIGN):
        exponent_field = 1 + (magnitude_index * 155) // (MAGNITUDES_PER_SIGN - 1)
        fraction_field = (magnitude_index * 37) & 0x7F
        positive = (exponent_field << 7) | fraction_field
        for negative in (False, True):
            bits = positive | (0x8000 if negative else 0)
            value, _ = decode_finite_bfloat16(bits)
            patterns.append(
                {
                    "magnitude_index": magnitude_index,
                    "negative": negative,
                    "exponent_field": exponent_field,
                    "fraction_field": fraction_field,
                    "small_value_bits": f"0x{bits:04x}",
                    "exact_value_numerator": value.numerator,
                    "exact_value_denominator": value.denominator,
                }
            )
    return patterns


def build_magnitude_probe_specifications(inner_dimension: int = 640) -> list[dict[str, Any]]:
    if (
        not isinstance(inner_dimension, int)
        or isinstance(inner_dimension, bool)
        or inner_dimension < 16
        or inner_dimension % 16 != 0
    ):
        raise ValueError("Magnitude probe inner dimension must be a positive multiple of 16")
    probes = []
    for placement, positive_lane, negative_lane, small_lane in PLACEMENTS:
        for magnitude in _magnitude_patterns():
            small_bits = int(magnitude["small_value_bits"], 16)
            exact = exact_sum_then_bfloat16(
                [0x4E80, 0xCE80, small_bits], [0x3F80] * 3
            )
            body = {
                "probe_index": len(probes),
                "placement": placement,
                "positive_lane": positive_lane,
                "negative_lane": negative_lane,
                "small_lane": small_lane,
                "positions": [positive_lane, negative_lane, small_lane],
                "large_value_bits": ["0x4e80", "0xce80"],
                **magnitude,
                "exact_result_bits": f"0x{exact:04x}",
            }
            probes.append({**body, "specification_sha256": _sha256(body)})
    if len(probes) != PROBE_COUNT:
        raise RuntimeError("Unexpected WMMA magnitude probe count")
    return probes


def _summaries(records: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, int], int]:
    placement_statistics = {}
    for placement, _, _, _ in PLACEMENTS:
        selected = [record for record in records if record["placement"] == placement]
        exact = [record for record in selected if record["matches_exact_small"]]
        zero = [
            record
            for record in selected
            if record["actual_result_bits"] in {"0x0000", "0x8000"}
        ]
        exact_exponents = sorted({record["exponent_field"] for record in exact})
        other_results = dict(
            sorted(
                Counter(
                    record["actual_result_bits"]
                    for record in selected
                    if not record["matches_exact_small"]
                    and record["actual_result_bits"] not in {"0x0000", "0x8000"}
                ).items()
            )
        )
        placement_statistics[placement] = {
            "probes": len(selected),
            "exact_small": len(exact),
            "zero": len(zero),
            "other": len(selected) - len(exact) - len(zero),
            "other_result_histogram": other_results,
            "exact_exponent_fields": exact_exponents,
            "minimum_exact_exponent_field": min(exact_exponents) if exact_exponents else None,
            "maximum_nonexact_exponent_field": max(
                (
                    record["exponent_field"]
                    for record in selected
                    if not record["matches_exact_small"]
                ),
                default=None,
            ),
        }
    histogram = dict(sorted(Counter(record["actual_result_bits"] for record in records).items()))
    by_key = {
        (record["placement"], record["magnitude_index"], record["negative"]): record
        for record in records
    }
    sign_symmetry_matches = 0
    for placement, _, _, _ in PLACEMENTS:
        for magnitude_index in range(MAGNITUDES_PER_SIGN):
            positive = int(
                by_key[(placement, magnitude_index, False)]["actual_result_bits"], 16
            )
            negative = int(
                by_key[(placement, magnitude_index, True)]["actual_result_bits"], 16
            )
            if negative == (positive ^ 0x8000):
                sign_symmetry_matches += 1
    return placement_statistics, histogram, sign_symmetry_matches


def build_wmma_magnitude_probe_certificate(
    program: dict[str, Any],
    reduction_certificate: dict[str, Any],
    nsight_suite: dict[str, Any],
    backend_binding: dict[str, Any],
) -> dict[str, Any]:
    _require_cuda()
    if not verify_gemma_ir(program)["valid"]:
        raise ValueError("Gemma IR is invalid")
    if not verify_gemma_reduction_backend_binding(
        program, reduction_certificate, nsight_suite, backend_binding
    )["valid"]:
        raise ValueError("Reduction backend binding is invalid")
    configuration = program["configuration"]
    inner = configuration["hidden_size"]
    outputs = configuration["attention_heads"] * configuration["head_dimension"]
    sequence = backend_binding["sequence_length"]
    if outputs != PROBE_COUNT:
        raise ValueError("The current magnitude probe requires exactly 1,024 query outputs")
    probes = build_magnitude_probe_specifications(inner)
    weights_bits = torch.zeros((PROBE_COUNT, inner), dtype=torch.uint16)
    for probe in probes:
        values = ["0x4e80", "0xce80", probe["small_value_bits"]]
        for position, value in zip(probe["positions"], values):
            weights_bits[probe["probe_index"], position] = int(value, 16)
    weights = weights_bits.view(torch.bfloat16).cuda()
    inputs = torch.ones((sequence, inner), dtype=torch.bfloat16, device="cuda")
    result, kernel_names = _profile_call(lambda: torch.nn.functional.linear(inputs, weights))
    input_rows_equal = all(
        torch.equal(inputs[0], inputs[row]) for row in range(1, inputs.shape[0])
    )
    output_rows_equal = all(
        torch.equal(result[0], result[row]) for row in range(1, result.shape[0])
    )
    result_bits = result[0].detach().cpu().view(torch.uint16).tolist()
    records = []
    for probe, actual in zip(probes, result_bits):
        body = {
            **probe,
            "actual_result_bits": f"0x{actual:04x}",
            "matches_exact_small": actual == int(probe["exact_result_bits"], 16),
        }
        records.append({**body, "record_sha256": _sha256(body)})
    placement_statistics, histogram, sign_symmetry_matches = _summaries(records)
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
        "scope": "Integrity-bound signed-magnitude probes through eight K16 lower/upper/split placements in a controlled linear using the query-projection-shaped CUTLASS symbol attested by the Gemma backend binding; optional CUDA replay may re-verify observations, while thresholds and symmetries do not establish model-tensor, accumulator, or hardware semantics.",
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
        "output_rows_equal": output_rows_equal,
        "placement_classes": [placement for placement, _, _, _ in PLACEMENTS],
        "magnitude_patterns_per_sign": MAGNITUDES_PER_SIGN,
        "probe_count": len(records),
        "records": records,
        "result_histogram": histogram,
        "distinct_result_count": len(histogram),
        "placement_statistics": placement_statistics,
        "sign_pair_count": len(PLACEMENTS) * MAGNITUDES_PER_SIGN,
        "sign_symmetry_match_count": sign_symmetry_matches,
        "sign_symmetry_established_for_all_pairs": sign_symmetry_matches
        == len(PLACEMENTS) * MAGNITUDES_PER_SIGN,
        "magnitude_generalization_established": False,
        "wmma_accumulator_mapping_identified": False,
        "reduction_order_semantics_established": False,
        "hardware_instruction_semantics_established": False,
    }
    return {**body, "certificate_sha256": _sha256(body)}


def verify_wmma_magnitude_probe_certificate(
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
        expected = build_magnitude_probe_specifications(inner)
    except (TypeError, ValueError, RuntimeError):
        expected = []
    records = certificate.get("records")

    def record_valid(record: Any, specification: dict[str, Any]) -> bool:
        if not isinstance(record, dict):
            return False
        try:
            record_body = {
                key: value for key, value in record.items() if key != "record_sha256"
            }
            return bool(
                all(record.get(key) == value for key, value in specification.items())
                and re.fullmatch(
                    r"0x[0-9a-f]{4}", str(record.get("actual_result_bits", ""))
                )
                is not None
                and record.get("matches_exact_small")
                == (
                    record.get("actual_result_bits")
                    == record.get("exact_result_bits")
                )
                and record.get("record_sha256") == _sha256(record_body)
            )
        except (TypeError, ValueError, AttributeError):
            return False

    records_valid = bool(
        isinstance(records, list)
        and len(records) == len(expected) == PROBE_COUNT
        and all(
            record_valid(record, specification)
            for record, specification in zip(records, expected)
        )
    )
    if records_valid:
        placement_statistics, histogram, sign_symmetry_matches = _summaries(records)
    else:
        placement_statistics, histogram, sign_symmetry_matches = {}, {}, -1
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
        and certificate.get("placement_classes")
        == [placement for placement, _, _, _ in PLACEMENTS]
        and certificate.get("magnitude_patterns_per_sign") == MAGNITUDES_PER_SIGN
        and certificate.get("probe_count") == PROBE_COUNT
        and certificate.get("result_histogram") == histogram
        and certificate.get("distinct_result_count") == len(histogram)
        and certificate.get("placement_statistics") == placement_statistics
        and certificate.get("sign_pair_count")
        == len(PLACEMENTS) * MAGNITUDES_PER_SIGN
        and certificate.get("sign_symmetry_match_count") == sign_symmetry_matches
        and certificate.get("sign_symmetry_established_for_all_pairs")
        == (sign_symmetry_matches == len(PLACEMENTS) * MAGNITUDES_PER_SIGN)
        and certificate.get("expected_query_kernel_names") == query_kernel_names
        and isinstance(kernels, list)
        and certificate.get("kernel_identity_matches_backend_binding")
        == (sorted(kernels) == query_kernel_names)
        and certificate.get("input_rows_equal") is True
        and certificate.get("output_rows_equal") is True
        and certificate.get("magnitude_generalization_established") is False
        and certificate.get("wmma_accumulator_mapping_identified") is False
        and certificate.get("reduction_order_semantics_established") is False
        and certificate.get("hardware_instruction_semantics_established") is False
    ) if records_valid else False
    reexecution_exact = None
    if reexecute:
        try:
            recomputed = build_wmma_magnitude_probe_certificate(
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
            not reexecute or reexecution_exact is True,
        )
    )
    return {
        "valid": valid,
        "certificate_hash_valid": certificate_hash_valid,
        "backend_binding_valid": backend_valid,
        "records_valid": records_valid,
        "claims_consistent": claims_consistent,
        "mode": "replayed" if reexecute else "integrity_only",
        "reexecution_performed": reexecute,
        "reexecution_exact": reexecution_exact,
        "probe_count": len(records) if isinstance(records, list) else 0,
        "distinct_result_count": len(histogram),
        "sign_symmetry_match_count": sign_symmetry_matches,
        "sign_symmetry_established_for_all_pairs": certificate.get(
            "sign_symmetry_established_for_all_pairs"
        ),
        "magnitude_generalization_established": certificate.get(
            "magnitude_generalization_established"
        ),
        "wmma_accumulator_mapping_identified": certificate.get(
            "wmma_accumulator_mapping_identified"
        ),
    }
