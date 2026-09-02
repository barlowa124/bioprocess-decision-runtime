from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from .serialization import canonical_json

_SOURCE_FILES = (
    "Makefile",
    "common.h",
    "inject_funcs.cu",
    "record_reg_vals.cu",
    "tool_func/flush_channel.cu",
    "test_apps/Makefile",
    "test_apps/carry_test.cu",
    "test_apps/lea_test.cu",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _source_hashes(directory: Path) -> dict[str, str]:
    return {
        relative: _sha256_bytes((directory / relative).read_bytes())
        for relative in _SOURCE_FILES
    }


def _report_hash(report: dict[str, Any]) -> str:
    return _sha256_bytes(canonical_json(report).encode("utf-8"))


def _validated_results(
    report: dict[str, Any], aggregate_flag: str, expected_count: int
) -> list[dict[str, Any]]:
    results = report.get("results")
    if report.get(aggregate_flag) is not True or not isinstance(results, list):
        raise ValueError(f"Report does not establish {aggregate_flag}")
    if len(results) != expected_count or not all(
        isinstance(result, dict) and result.get("valid") is True
        for result in results
    ):
        raise ValueError(f"Invalid result records for {aggregate_flag}")
    return results


def _require_flags(record: dict[str, Any], fields: tuple[str, ...]) -> None:
    if not all(record.get(field) is True for field in fields):
        raise ValueError(f"Required observational flags are incomplete: {fields}")


def _require_false_boundaries(report: dict[str, Any]) -> None:
    for field in (
        "exact_cubin_evidence_established",
        "independent_reproduction_complete",
        "full_gemma_forward_register_writes_performed",
        "authoritative_p2r_mask_semantics_established",
        "carry_semantics_qualified",
        "carry_equation_activation_allowed",
    ):
        if field in report and report[field] is not False:
            raise ValueError(f"Report boundary must remain false: {field}")


def build_sass_dynamic_summary(
    reports: dict[str, dict[str, Any]],
    tool_directory: Path,
    acquisition_tool_binary_sha256: str,
) -> dict[str, Any]:
    if not isinstance(reports, dict):
        raise ValueError("Dynamic reports must be a mapping")
    if not re.fullmatch(r"[0-9a-f]{64}", acquisition_tool_binary_sha256):
        raise ValueError("Invalid acquisition tool binary SHA-256")
    required = {
        "encoding",
        "low",
        "high",
        "query_pair",
        "key_pair",
        "value_triple",
        "p2r_encoding",
        "combined",
    }
    if set(reports) != required:
        raise ValueError(f"Expected reports {sorted(required)}")
    if not all(isinstance(report, dict) for report in reports.values()):
        raise ValueError("Each dynamic report must be a mapping")
    for report in reports.values():
        _require_false_boundaries(report)
    encoding_records = reports["encoding"].get("runtime_instruction_comparisons")
    if reports["encoding"].get("all_runtime_selected_encodings_match") is not True:
        raise ValueError("Runtime instruction encoding correspondence is incomplete")
    if not isinstance(encoding_records, list) or len(encoding_records) != 10:
        raise ValueError("Expected ten selected runtime instruction encodings")
    if reports["encoding"].get("torch") != "2.7.1+cu128" or reports[
        "encoding"
    ].get("torch_cuda") != "12.8":
        raise ValueError("Unexpected PyTorch/CUDA runtime identity")
    if not all(
        record.get("encoding_matches") is True and record.get("sass_matches") is True
        for record in encoding_records
    ):
        raise ValueError("Selected runtime instruction encoding mismatch")
    low_results = _validated_results(reports["low"], "all_qkv_classes_valid", 6)
    high_results = _validated_results(
        reports["high"], "all_qkv_high_classes_valid", 6
    )
    for result in (*low_results, *high_results):
        if result.get("controlled_launch_repetitions") != 3:
            raise ValueError("Dynamic carry observations require three repetitions")
        if result.get("controlled_active_lane_count") != 1536:
            raise ValueError("Dynamic carry observation lane coverage mismatch")
        if result.get("repeated_attention_outputs_identical") is not True:
            raise ValueError("Attention restoration evidence is incomplete")
    expected_classes = {
        (field, carry_class)
        for field in ("query_ptr", "key_ptr", "value_ptr")
        for carry_class in (0, 1)
    }
    if {
        (result.get("field"), result.get("carry_class"))
        for result in low_results
    } != expected_classes:
        raise ValueError("Low observation field/class coverage mismatch")
    if {
        (result.get("field"), result.get("carry_class"))
        for result in high_results
    } != expected_classes:
        raise ValueError("High observation field/class coverage mismatch")
    for result in low_results:
        _require_flags(
            result,
            (
                "controlled_coverage_valid",
                "controlled_gpr_valid",
                "controlled_ureg_valid",
                "controlled_low_result_valid",
                "controlled_predicate_valid",
                "runtime_encoding_matches_attested_instruction",
                "repeated_attention_outputs_identical",
            ),
        )
    for result in high_results:
        _require_flags(
            result,
            (
                "controlled_coverage_valid",
                "controlled_zero_high_sources_valid",
                "controlled_predicate_input_valid",
                "controlled_high_result_valid",
                "runtime_encoding_matches_attested_instruction",
                "repeated_attention_outputs_identical",
            ),
        )
    for name, field in (("query_pair", "query"), ("key_pair", "key")):
        report = reports[name]
        if report.get("field") != field:
            raise ValueError(f"Unexpected field in {name}")
        pair_results = _validated_results(
            report, "same_launch_sequential_pair_recomposition_established", 2
        )
        if {result.get("carry_class") for result in pair_results} != {0, 1}:
            raise ValueError(f"Pair class coverage mismatch for {field}")
        for result in pair_results:
            if result.get("controlled_launch_repetitions") != 3:
                raise ValueError(f"Pair repetition mismatch for {field}")
            if result.get("controlled_low_active_lane_count") != 1536 or result.get(
                "controlled_high_active_lane_count"
            ) != 1536:
                raise ValueError(f"Pair lane coverage mismatch for {field}")
            _require_flags(
                result,
                (
                    "controlled_coverage_valid",
                    "controlled_low_state_valid",
                    "controlled_high_state_valid",
                    "predicate_flows_low_to_high_valid",
                    "same_launch_recomposition_valid",
                    "low_runtime_encoding_matches",
                    "high_runtime_encoding_matches",
                    "repeated_attention_outputs_identical",
                ),
            )
    value_results = _validated_results(
        reports["value_triple"],
        "same_launch_sequential_value_recomposition_established",
        2,
    )
    if reports["value_triple"].get(
        "p2r_output_invariant_to_p6_classes_for_observed_vectors"
    ) is not True:
        raise ValueError("Observed P2R class-invariance evidence is incomplete")
    if {result.get("carry_class") for result in value_results} != {0, 1}:
        raise ValueError("Value pair class coverage mismatch")
    for result in value_results:
        if result.get("controlled_launch_repetitions") != 3:
            raise ValueError("Value pair repetitions are incomplete")
        if result.get("controlled_records_per_role") != {
            "low": 48,
            "middle": 48,
            "high": 48,
        }:
            raise ValueError("Value low/P2R/high record coverage mismatch")
        _require_flags(
            result,
            (
                "all_pair_slots_ready",
                "controlled_low_state_valid",
                "controlled_high_state_valid",
                "p6_state_valid_at_p2r",
                "p2r_repetitions_stable",
                "same_launch_recomposition_valid",
                "low_runtime_encoding_matches",
                "high_runtime_encoding_matches",
                "repeated_attention_outputs_identical",
            ),
        )
    if reports["p2r_encoding"].get("valid") is not True:
        raise ValueError("P2R runtime instruction encoding correspondence is invalid")
    _require_flags(
        reports["p2r_encoding"],
        (
            "attested_cubin_hash_valid",
            "instruction_text_matches",
            "instruction_encoding_matches",
        ),
    )
    if reports["p2r_encoding"].get("instruction_offset") != 0x3370:
        raise ValueError("Unexpected P2R instruction offset")
    if reports["combined"].get(
        "all_isolated_qkv_sequential_observations_valid"
    ) is not True:
        raise ValueError("Combined Q/K/V sequential observations are incomplete")
    source_hashes = _source_hashes(tool_directory)
    body = {
        "schema_version": 1,
        "scope": "Compact commitments to isolated NVBit carry observations; raw traces are omitted and no hardware-semantic, exact-cubin, independent-reproduction, memory-safety, or qualification claim is established.",
        "acquisition_tool_binary_sha256": acquisition_tool_binary_sha256,
        "source_report_sha256": {
            name: _report_hash(report) for name, report in sorted(reports.items())
        },
        "tool_source_sha256": source_hashes,
        "tool_source_commitment_sha256": _sha256_bytes(
            canonical_json(source_hashes).encode("utf-8")
        ),
        "nvbit_version": "1.8",
        "torch_version": reports["encoding"].get("torch"),
        "torch_cuda_version": reports["encoding"].get("torch_cuda"),
        "compute_capability": "8.9",
        "selected_runtime_instruction_encoding_match_count": 10,
        "low_observation_record_count": len(low_results),
        "high_observation_record_count": len(high_results),
        "controlled_launch_repetitions_per_class": 3,
        "controlled_active_lane_observations_per_field_class": 1536,
        "same_launch_sequential_fields": ["query_ptr", "key_ptr", "value_ptr"],
        "p2r_output_invariant_to_p6_for_observed_vectors": True,
        "attention_outputs_restored": True,
        "raw_traces_committed": False,
        "source_reports_committed": False,
        "source_reports_independently_replayed": False,
        "whole_cubin_equivalence_established": False,
        "exact_cubin_dynamic_evidence_established": False,
        "authoritative_p2r_mask_semantics_established": False,
        "independent_reproduction_complete": False,
        "hardware_instruction_semantics_established": False,
        "kernel_memory_safety_established": False,
        "carry_semantics_qualified": False,
        "carry_equation_activation_allowed": False,
    }
    return {
        **body,
        "summary_sha256": _sha256_bytes(canonical_json(body).encode("utf-8")),
    }


def verify_sass_dynamic_summary(
    summary: dict[str, Any], tool_directory: Path | None = None
) -> dict[str, Any]:
    if not isinstance(summary, dict):
        return {"valid": False}
    body = {
        key: value for key, value in summary.items() if key != "summary_sha256"
    }
    try:
        summary_hash_valid = _sha256_bytes(
            canonical_json(body).encode("utf-8")
        ) == summary.get("summary_sha256")
    except (TypeError, ValueError):
        return {"valid": False}
    source_hashes = summary.get("tool_source_sha256")
    source_commitment_valid = bool(
        isinstance(source_hashes, dict)
        and set(source_hashes) == set(_SOURCE_FILES)
        and _sha256_bytes(canonical_json(source_hashes).encode("utf-8"))
        == summary.get("tool_source_commitment_sha256")
    )
    local_source_hashes_valid = bool(
        tool_directory is not None
        and source_hashes == _source_hashes(tool_directory)
    )
    source_report_hashes = summary.get("source_report_sha256")
    source_report_commitments_valid = bool(
        isinstance(source_report_hashes, dict)
        and set(source_report_hashes)
        == {
            "encoding",
            "low",
            "high",
            "query_pair",
            "key_pair",
            "value_triple",
            "p2r_encoding",
            "combined",
        }
        and all(
            isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
            for value in source_report_hashes.values()
        )
    )
    observations_consistent = bool(
        summary.get("schema_version") == 1
        and bool(
            re.fullmatch(
                r"[0-9a-f]{64}",
                summary.get("acquisition_tool_binary_sha256", ""),
            )
        )
        and summary.get("nvbit_version") == "1.8"
        and summary.get("torch_version") == "2.7.1+cu128"
        and summary.get("torch_cuda_version") == "12.8"
        and summary.get("compute_capability") == "8.9"
        and summary.get("selected_runtime_instruction_encoding_match_count") == 10
        and summary.get("low_observation_record_count") == 6
        and summary.get("high_observation_record_count") == 6
        and summary.get("controlled_launch_repetitions_per_class") == 3
        and summary.get("controlled_active_lane_observations_per_field_class")
        == 1536
        and summary.get("same_launch_sequential_fields")
        == ["query_ptr", "key_ptr", "value_ptr"]
        and summary.get("p2r_output_invariant_to_p6_for_observed_vectors")
        is True
        and summary.get("attention_outputs_restored") is True
    )
    boundaries_preserved = all(
        summary.get(field) is False
        for field in (
            "raw_traces_committed",
            "source_reports_committed",
            "source_reports_independently_replayed",
            "whole_cubin_equivalence_established",
            "exact_cubin_dynamic_evidence_established",
            "authoritative_p2r_mask_semantics_established",
            "independent_reproduction_complete",
            "hardware_instruction_semantics_established",
            "kernel_memory_safety_established",
            "carry_semantics_qualified",
            "carry_equation_activation_allowed",
        )
    )
    valid = all(
        (
            summary_hash_valid,
            source_commitment_valid,
            local_source_hashes_valid,
            source_report_commitments_valid,
            observations_consistent,
            boundaries_preserved,
        )
    )
    return {
        "valid": valid,
        "summary_hash_valid": summary_hash_valid,
        "source_commitment_valid": source_commitment_valid,
        "local_source_hashes_valid": local_source_hashes_valid,
        "source_report_commitments_valid": source_report_commitments_valid,
        "source_reports_independently_replayed": False,
        "observations_consistent": observations_consistent,
        "boundaries_preserved": boundaries_preserved,
    }
