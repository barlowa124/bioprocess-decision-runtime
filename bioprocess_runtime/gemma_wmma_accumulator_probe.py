from __future__ import annotations

from collections import Counter
import hashlib
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


PROBE_COUNT = 3360
ROWS_PER_LAUNCH = 1024


def _require_cuda() -> None:
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError("WMMA accumulator probing requires CUDA PyTorch")


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def build_accumulator_probe_specifications(
    inner_dimension: int = 640,
) -> list[dict[str, Any]]:
    if (
        not isinstance(inner_dimension, int)
        or isinstance(inner_dimension, bool)
        or inner_dimension < 640
        or inner_dimension % 16 != 0
    ):
        raise ValueError("Accumulator probe inner dimension must be a multiple of 16 and at least 640")
    tiles = inner_dimension // 16
    triplets = [
        (positive_lane, negative_lane, small_lane)
        for positive_lane in range(16)
        for negative_lane in range(16)
        for small_lane in range(16)
        if len({positive_lane, negative_lane, small_lane}) == 3
    ]
    if len(triplets) != PROBE_COUNT:
        raise RuntimeError("Unexpected K16 ordered-triplet domain size")
    value_bits = [0x4E80, 0xCE80, 0x3F80]
    exact_result = exact_sum_then_bfloat16(value_bits, [0x3F80] * 3)
    probes = []
    for probe_index, (positive_lane, negative_lane, small_lane) in enumerate(
        triplets
    ):
        tile = 0
        body = {
            "probe_index": probe_index,
            "tile": tile,
            "positive_lane": positive_lane,
            "negative_lane": negative_lane,
            "small_lane": small_lane,
            "positions": [
                tile * 16 + positive_lane,
                tile * 16 + negative_lane,
                tile * 16 + small_lane,
            ],
            "value_bits": [f"0x{value:04x}" for value in value_bits],
            "exact_result_bits": f"0x{exact_result:04x}",
        }
        probes.append({**body, "specification_sha256": _sha256(body)})
    return probes


def _lower_then_upper_retention_rule(record: dict[str, Any]) -> bool:
    return bool(
        record["positive_lane"] < 8
        and record["negative_lane"] < 8
        and record["small_lane"] >= 8
    )


def build_wmma_accumulator_probe_certificate(
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
    if outputs != ROWS_PER_LAUNCH:
        raise ValueError("The current probe layout requires exactly 1,024 query outputs per launch")
    probes = build_accumulator_probe_specifications(inner)
    inputs = torch.ones((sequence, inner), dtype=torch.bfloat16, device="cuda")
    input_rows_equal = all(
        torch.equal(inputs[0], inputs[row]) for row in range(1, inputs.shape[0])
    )
    records = []
    launch_kernel_names = []
    output_rows_equal = True
    for launch_index, start in enumerate(range(0, len(probes), ROWS_PER_LAUNCH)):
        selected_probes = probes[start : start + ROWS_PER_LAUNCH]
        weights_bits = torch.zeros((ROWS_PER_LAUNCH, inner), dtype=torch.uint16)
        for row, probe in enumerate(selected_probes):
            for position, value in zip(probe["positions"], probe["value_bits"]):
                weights_bits[row, position] = int(value, 16)
        weights = weights_bits.view(torch.bfloat16).cuda()
        result, kernel_names = _profile_call(
            lambda: torch.nn.functional.linear(inputs, weights)
        )
        launch_kernel_names.append(kernel_names)
        output_rows_equal &= all(
            torch.equal(result[0], result[row]) for row in range(1, result.shape[0])
        )
        result_bits = result[0].detach().cpu().view(torch.uint16).tolist()
        for row, probe in enumerate(selected_probes):
            actual = result_bits[row]
            body = {
                **probe,
                "launch_index": launch_index,
                "launch_row": row,
                "actual_result_bits": f"0x{actual:04x}",
                "small_value_retained": actual == 0x3F80,
            }
            records.append({**body, "record_sha256": _sha256(body)})
    kernel_names = sorted(
        {name for names in launch_kernel_names for name in names}
    )
    histogram = dict(sorted(Counter(record["actual_result_bits"] for record in records).items()))
    lane_statistics = {}
    for small_lane in range(16):
        selected = [record for record in records if record["small_lane"] == small_lane]
        lane_statistics[str(small_lane)] = {
            "probes": len(selected),
            "retained": sum(record["small_value_retained"] for record in selected),
        }
    relation_statistics = {}
    for relation, predicate in {
        "small_matches_positive_mod4": lambda record: record["small_lane"] % 4
        == record["positive_lane"] % 4,
        "small_matches_negative_mod4": lambda record: record["small_lane"] % 4
        == record["negative_lane"] % 4,
        "large_terms_share_mod4": lambda record: record["positive_lane"] % 4
        == record["negative_lane"] % 4,
        "large_terms_share_half": lambda record: record["positive_lane"] // 8
        == record["negative_lane"] // 8,
    }.items():
        selected = [record for record in records if predicate(record)]
        relation_statistics[relation] = {
            "probes": len(selected),
            "retained": sum(record["small_value_retained"] for record in selected),
        }
    exact_results = sorted(
        {record["exact_result_bits"] for record in records}
    )
    half_order_rule_match_count = sum(
        record["small_value_retained"]
        is _lower_then_upper_retention_rule(record)
        for record in records
    )
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
        "scope": "Integrity-bound three-term probes mapping whether one small addend survives cancellation inside the exact Gemma query-projection CUDA kernel; CUDA re-execution verifies observations, but retention correlations do not establish accumulator or hardware semantics.",
        "program_sha256": program["program_sha256"],
        "reduction_backend_binding_sha256": backend_binding["binding_sha256"],
        "query_projection_shape": {
            "input": [sequence, inner],
            "weight": [ROWS_PER_LAUNCH, inner],
            "output": [sequence, ROWS_PER_LAUNCH],
        },
        "kernel_names": kernel_names,
        "expected_query_kernel_names": query_kernel_names,
        "kernel_identity_matches_backend_binding": sorted(kernel_names)
        == query_kernel_names,
        "input_rows_equal": input_rows_equal,
        "output_rows_equal": output_rows_equal,
        "probe_count": len(records),
        "rows_per_launch": ROWS_PER_LAUNCH,
        "acquisition_launch_count": len(launch_kernel_names),
        "launch_kernel_names": launch_kernel_names,
        "triplet_domain_size": 16 * 15 * 14,
        "triplet_domain_exhausted": len(records) == 16 * 15 * 14,
        "records": records,
        "result_histogram": histogram,
        "distinct_result_count": len(histogram),
        "retained_count": sum(record["small_value_retained"] for record in records),
        "lane_statistics": lane_statistics,
        "relation_statistics": relation_statistics,
        "exact_result_values": exact_results,
        "all_exact_sums_identical": len(exact_results) == 1,
        "small_retention_position_dependent": len(
            {record["small_value_retained"] for record in records}
        )
        > 1,
        "candidate_retention_rule": "retain iff positive_lane < 8 and negative_lane < 8 and small_lane >= 8",
        "candidate_retention_rule_match_count": half_order_rule_match_count,
        "candidate_retention_rule_matches_exhaustive_triplet_domain": (
            half_order_rule_match_count == len(records) == 3360
        ),
        "candidate_k16_half_order": "lanes 0-7 before lanes 8-15",
        "candidate_k16_half_order_full_numeric_semantics_established": False,
        "accumulator_equivalence_classes_identified": False,
        "wmma_accumulator_mapping_identified": False,
        "reduction_order_semantics_established": False,
        "hardware_instruction_semantics_established": False,
    }
    return {**body, "certificate_sha256": _sha256(body)}


def verify_wmma_accumulator_probe_certificate(
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
        expected = build_accumulator_probe_specifications(inner)
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
                and record.get("launch_index")
                == specification["probe_index"] // ROWS_PER_LAUNCH
                and record.get("launch_row")
                == specification["probe_index"] % ROWS_PER_LAUNCH
                and re.fullmatch(
                    r"0x[0-9a-f]{4}", str(record.get("actual_result_bits", ""))
                )
                is not None
                and record.get("small_value_retained")
                is (record.get("actual_result_bits") == "0x3f80")
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
    histogram = (
        dict(sorted(Counter(record["actual_result_bits"] for record in records).items()))
        if records_valid
        else {}
    )
    retained_count = (
        sum(record["small_value_retained"] for record in records)
        if records_valid
        else -1
    )
    exact_results = (
        sorted({record["exact_result_bits"] for record in records})
        if records_valid
        else []
    )
    lane_statistics = {}
    if records_valid:
        for small_lane in range(16):
            selected = [record for record in records if record["small_lane"] == small_lane]
            lane_statistics[str(small_lane)] = {
                "probes": len(selected),
                "retained": sum(record["small_value_retained"] for record in selected),
            }
    relation_statistics = {}
    if records_valid:
        for relation, predicate in {
            "small_matches_positive_mod4": lambda record: record["small_lane"] % 4
            == record["positive_lane"] % 4,
            "small_matches_negative_mod4": lambda record: record["small_lane"] % 4
            == record["negative_lane"] % 4,
            "large_terms_share_mod4": lambda record: record["positive_lane"] % 4
            == record["negative_lane"] % 4,
            "large_terms_share_half": lambda record: record["positive_lane"] // 8
            == record["negative_lane"] // 8,
        }.items():
            selected = [record for record in records if predicate(record)]
            relation_statistics[relation] = {
                "probes": len(selected),
                "retained": sum(record["small_value_retained"] for record in selected),
            }
    half_order_rule_match_count = (
        sum(
            record["small_value_retained"]
            is _lower_then_upper_retention_rule(record)
            for record in records
        )
        if records_valid
        else -1
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
        and certificate.get("query_projection_shape")
        == {
            "input": [backend_binding.get("sequence_length"), inner],
            "weight": [ROWS_PER_LAUNCH, inner],
            "output": [backend_binding.get("sequence_length"), ROWS_PER_LAUNCH],
        }
        and certificate.get("probe_count") == PROBE_COUNT
        and certificate.get("rows_per_launch") == ROWS_PER_LAUNCH
        and certificate.get("acquisition_launch_count") == 4
        and isinstance(certificate.get("launch_kernel_names"), list)
        and len(certificate["launch_kernel_names"]) == 4
        and all(
            sorted(names) == query_kernel_names
            for names in certificate["launch_kernel_names"]
        )
        and certificate.get("triplet_domain_size") == 3360
        and certificate.get("triplet_domain_exhausted") is True
        and certificate.get("result_histogram") == histogram
        and certificate.get("distinct_result_count") == len(histogram)
        and certificate.get("retained_count") == retained_count
        and certificate.get("lane_statistics") == lane_statistics
        and certificate.get("relation_statistics") == relation_statistics
        and certificate.get("expected_query_kernel_names") == query_kernel_names
        and isinstance(kernels, list)
        and certificate.get("kernel_identity_matches_backend_binding")
        is (sorted(kernels) == query_kernel_names)
        and certificate.get("input_rows_equal") is True
        and certificate.get("output_rows_equal") is True
        and certificate.get("exact_result_values") == exact_results
        and certificate.get("all_exact_sums_identical")
        is (len(exact_results) == 1)
        and certificate.get("small_retention_position_dependent")
        is (len({record["small_value_retained"] for record in records}) > 1)
        and certificate.get("candidate_retention_rule")
        == "retain iff positive_lane < 8 and negative_lane < 8 and small_lane >= 8"
        and certificate.get("candidate_retention_rule_match_count")
        == half_order_rule_match_count
        and certificate.get("candidate_retention_rule_matches_exhaustive_triplet_domain")
        is (half_order_rule_match_count == len(records) == 3360)
        and certificate.get("candidate_k16_half_order")
        == "lanes 0-7 before lanes 8-15"
        and certificate.get("candidate_k16_half_order_full_numeric_semantics_established")
        is False
        and certificate.get("accumulator_equivalence_classes_identified") is False
        and certificate.get("wmma_accumulator_mapping_identified") is False
        and certificate.get("reduction_order_semantics_established") is False
        and certificate.get("hardware_instruction_semantics_established") is False
    ) if records_valid else False
    reexecution_exact = None
    if reexecute:
        try:
            recomputed = build_wmma_accumulator_probe_certificate(
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
        "reexecution_performed": reexecute,
        "reexecution_exact": reexecution_exact,
        "probe_count": len(records) if isinstance(records, list) else 0,
        "distinct_result_count": len(histogram),
        "retained_count": retained_count,
        "small_retention_position_dependent": certificate.get(
            "small_retention_position_dependent"
        ),
        "candidate_retention_rule_match_count": half_order_rule_match_count,
        "candidate_retention_rule_matches_exhaustive_triplet_domain": certificate.get(
            "candidate_retention_rule_matches_exhaustive_triplet_domain"
        ),
        "candidate_k16_half_order_full_numeric_semantics_established": certificate.get(
            "candidate_k16_half_order_full_numeric_semantics_established"
        ),
        "accumulator_equivalence_classes_identified": certificate.get(
            "accumulator_equivalence_classes_identified"
        ),
        "wmma_accumulator_mapping_identified": certificate.get(
            "wmma_accumulator_mapping_identified"
        ),
    }
