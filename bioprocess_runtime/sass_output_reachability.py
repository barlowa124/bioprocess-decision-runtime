from __future__ import annotations

import hashlib
import re
from typing import Any

from .serialization import canonical_json


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def build_sass_output_reachability_summary(
    report: dict[str, Any], acquisition_tool_binary_sha256: str
) -> dict[str, Any]:
    if not isinstance(report, dict):
        raise ValueError("Reachability report must be a mapping")
    if not re.fullmatch(r"[0-9a-f]{64}", acquisition_tool_binary_sha256):
        raise ValueError("Invalid acquisition tool binary SHA-256")
    records = report.get("records")
    if not isinstance(records, list) or len(records) != 8:
        raise ValueError("Expected eight reachability records")
    expected_lengths = [30, 64, 128, 129, 256, 257, 512, 1024]
    if [record.get("length") for record in records] != expected_lengths:
        raise ValueError("Unexpected reachability length grid")
    if report.get("all_outputs_finite") is not True:
        raise ValueError("Reachability outputs are not all finite")
    if report.get("long_sweep_repeat_deterministic") is not True:
        raise ValueError("Long reachability sweep was not deterministic")
    for record in records:
        if record.get("block") != [32, 4, 1] or record.get("finite") is not True:
            raise ValueError("Invalid launch or output-finiteness record")
        counts = record.get("instruction_after_record_counts")
        if not isinstance(counts, dict) or set(counts) != {
            "output_low",
            "output_high",
            "output_accum_low",
            "output_accum_high",
        }:
            raise ValueError("Invalid output instruction-count record")
        if not all(isinstance(value, int) and value >= 0 for value in counts.values()):
            raise ValueError("Invalid output instruction count")
        if record.get("output_pair_reached") != (
            counts["output_low"] > 0 and counts["output_high"] > 0
        ):
            raise ValueError("Output-pair reachability inconsistency")
        if record.get("output_accum_pair_reached") != (
            counts["output_accum_low"] > 0
            and counts["output_accum_high"] > 0
        ):
            raise ValueError("Output-accumulator reachability inconsistency")
    minimum_output = min(
        record["length"] for record in records if record["output_pair_reached"]
    )
    minimum_accum = min(
        record["length"]
        for record in records
        if record["output_accum_pair_reached"]
    )
    both_record = min(
        (
            record
            for record in records
            if record["output_pair_reached"]
            and record["output_accum_pair_reached"]
        ),
        key=lambda record: record["length"],
    )
    minimum_both = both_record["length"]
    minimum_both_threads = (
        both_record["grid"][0]
        * both_record["grid"][1]
        * both_record["grid"][2]
        * both_record["block"][0]
        * both_record["block"][1]
        * both_record["block"][2]
    )
    if report.get("minimum_output_pair_length_observed") != minimum_output:
        raise ValueError("Output-pair reachability threshold mismatch")
    if report.get("minimum_output_accum_pair_length_observed") != minimum_accum:
        raise ValueError("Output-accumulator reachability threshold mismatch")
    if any(
        report.get(field) is not False
        for field in (
            "register_writes_performed",
            "output_pair_semantics_established",
            "kernel_memory_safety_established",
            "carry_semantics_qualified",
            "carry_equation_activation_allowed",
        )
    ):
        raise ValueError("Reachability report boundaries were weakened")
    retained_records = [
        {
            "length": record["length"],
            "grid": record["grid"],
            "block": record["block"],
            "instruction_after_record_counts": record[
                "instruction_after_record_counts"
            ],
            "output_pair_reached": record["output_pair_reached"],
            "output_accum_pair_reached": record["output_accum_pair_reached"],
        }
        for record in records
    ]
    body = {
        "schema_version": 1,
        "scope": "Compact non-writing output-pair reachability commitments; no output-pair semantics, intervention safety, memory safety, qualification, or activation is established.",
        "source_report_sha256": _sha256(report),
        "acquisition_tool_binary_sha256": acquisition_tool_binary_sha256,
        "records": retained_records,
        "minimum_output_pair_length_observed": minimum_output,
        "minimum_output_accum_pair_length_observed": minimum_accum,
        "minimum_both_pairs_length_observed": minimum_both,
        "minimum_both_pairs_thread_count": minimum_both_threads,
        "long_sweep_repeat_deterministic": True,
        "all_outputs_finite": True,
        "register_writes_performed": False,
        "dynamic_occurrence_indexing_established": False,
        "output_pair_intervention_allowed": False,
        "output_pair_semantics_established": False,
        "kernel_memory_safety_established": False,
        "carry_semantics_qualified": False,
        "carry_equation_activation_allowed": False,
    }
    return {**body, "summary_sha256": _sha256(body)}


def verify_sass_output_reachability_summary(
    summary: dict[str, Any], acquisition_tool_binary_sha256: str | None = None
) -> dict[str, Any]:
    if not isinstance(summary, dict):
        return {"valid": False}
    body = {
        key: value for key, value in summary.items() if key != "summary_sha256"
    }
    try:
        summary_hash_valid = _sha256(body) == summary.get("summary_sha256")
    except (TypeError, ValueError):
        return {"valid": False}
    acquisition_tool_valid = bool(
        re.fullmatch(
            r"[0-9a-f]{64}", summary.get("acquisition_tool_binary_sha256", "")
        )
        and (
            acquisition_tool_binary_sha256 is None
            or summary.get("acquisition_tool_binary_sha256")
            == acquisition_tool_binary_sha256
        )
    )
    records = summary.get("records")
    expected_grids = [1, 2, 4, 5, 8, 9, 16, 32]

    def record_valid(record: dict[str, Any], grid_x: int) -> bool:
        counts = record.get("instruction_after_record_counts")
        return bool(
            record.get("grid") == [grid_x, 4, 1]
            and record.get("block") == [32, 4, 1]
            and isinstance(counts, dict)
            and set(counts)
            == {
                "output_low",
                "output_high",
                "output_accum_low",
                "output_accum_high",
            }
            and all(isinstance(value, int) and value >= 0 for value in counts.values())
            and record.get("output_pair_reached")
            == (counts["output_low"] > 0 and counts["output_high"] > 0)
            and record.get("output_accum_pair_reached")
            == (
                counts["output_accum_low"] > 0
                and counts["output_accum_high"] > 0
            )
        )

    records_valid = bool(
        isinstance(records, list)
        and len(records) == len(expected_grids)
        and [record.get("length") for record in records]
        == [30, 64, 128, 129, 256, 257, 512, 1024]
        and all(
            isinstance(record, dict) and record_valid(record, grid_x)
            for record, grid_x in zip(records, expected_grids)
        )
    )
    derived_thresholds_valid = False
    if (
        records_valid
        and any(record["output_pair_reached"] for record in records)
        and any(record["output_accum_pair_reached"] for record in records)
    ):
        minimum_output = min(
            record["length"] for record in records if record["output_pair_reached"]
        )
        minimum_accum = min(
            record["length"]
            for record in records
            if record["output_accum_pair_reached"]
        )
        both_record = min(
            (
                record
                for record in records
                if record["output_pair_reached"]
                and record["output_accum_pair_reached"]
            ),
            key=lambda record: record["length"],
        )
        thread_count = 1
        for value in (*both_record["grid"], *both_record["block"]):
            thread_count *= value
        derived_thresholds_valid = bool(
            summary.get("minimum_output_pair_length_observed") == minimum_output
            and summary.get("minimum_output_accum_pair_length_observed")
            == minimum_accum
            and summary.get("minimum_both_pairs_length_observed")
            == both_record["length"]
            and summary.get("minimum_both_pairs_thread_count") == thread_count
        )
    reachability_consistent = bool(
        records_valid
        and derived_thresholds_valid
        and summary.get("long_sweep_repeat_deterministic") is True
        and summary.get("all_outputs_finite") is True
    )
    boundaries_preserved = all(
        summary.get(field) is False
        for field in (
            "register_writes_performed",
            "dynamic_occurrence_indexing_established",
            "output_pair_intervention_allowed",
            "output_pair_semantics_established",
            "kernel_memory_safety_established",
            "carry_semantics_qualified",
            "carry_equation_activation_allowed",
        )
    )
    valid = all(
        (
            summary_hash_valid,
            acquisition_tool_valid,
            reachability_consistent,
            boundaries_preserved,
        )
    )
    return {
        "valid": valid,
        "summary_hash_valid": summary_hash_valid,
        "acquisition_tool_valid": acquisition_tool_valid,
        "reachability_consistent": reachability_consistent,
        "boundaries_preserved": boundaries_preserved,
    }
