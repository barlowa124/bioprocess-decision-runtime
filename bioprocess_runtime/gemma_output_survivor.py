from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .gemma_attention_output import OutputSources, acquire_output, output_report, verify_output
from .gemma_attention_entry import _descriptor, _state_array
from .gemma_output_split import _base
from .gemma_k1024_probes import verify_probes, candidates, _code_sha as probe_code_sha
from .gemma_wmma_candidate import _operand_aligned_accumulator, _merge_split_partials
from .gemma_float_semantics import decode_finite_bfloat16, encode_bfloat16_rne
from .gemma_rotary_slice import _sha, _seal, _check_hash
from .serialization import canonical_json

SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
GRID = candidates()
SCOPE = "Fixed-model regression of the unique controlled K1024 split/merge-grid survivor; original model outputs were already observed, so not a fresh model holdout or hardware partition proof."


def _code_sha() -> str:
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Output-survivor source changed after import")
    return _sha({"module": SOURCE_SHA256, "controlled_probe": probe_code_sha()})


def selected_candidate(source_summary: dict[str, Any], probe_plan: dict[str, Any], probe_bundle: dict[str, Any], probe_report: dict[str, Any]) -> dict[str, Any]:
    checked = verify_probes(source_summary, probe_plan, probe_bundle, probe_report)
    if not checked["valid"] or not checked["survivors_supported_in_declared_scope"] or not checked["unique_survivor_in_frozen_grid"]:
        raise ValueError("Original-model regression requires one supported controlled-grid survivor")
    identifier = checked["surviving_candidates"][0]
    return next(item for item in candidates() if item["id"] == identifier)


def survivor_dot(left: list[int], right: list[int], candidate: dict[str, Any]) -> int:
    if len(left) != 1024 or len(right) != 1024 or candidate not in GRID:
        raise ValueError("Unsupported controlled-survivor K1024 profile")
    partials = [_operand_aligned_accumulator(left[start:stop], right[start:stop]) for start, stop in candidate["partitions"]]
    if candidate["partial_format"] == "bfloat16_rne":
        partials = [decode_finite_bfloat16(encode_bfloat16_rne(value))[0] for value in partials]
    return _merge_split_partials(partials, candidate["merge"])


def _material(sources: OutputSources, base_plan: dict[str, Any], base_bundle: dict[str, Any], base_report: dict[str, Any], source_summary: dict[str, Any], probe_plan: dict[str, Any], probe_bundle: dict[str, Any], probe_report: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    probability, values, predictions = _base(sources, base_plan, base_bundle, base_report)
    if source_summary["source_report_sha256"] != base_report["report_sha256"] or source_summary["plan_sha256"] != base_plan["plan_sha256"]:
        raise ValueError("Controlled probe provenance differs from the model case")
    candidate = selected_candidate(source_summary, probe_plan, probe_bundle, probe_report)
    return probability, values, predictions, candidate


def _plan_body(sources: OutputSources, base_plan: dict[str, Any], base_report: dict[str, Any], probe_plan: dict[str, Any], probe_report: dict[str, Any], candidate: dict[str, Any], projected: np.ndarray, bundle: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": 1, "scope": SCOPE, "sources": sources.commitments(), "base_plan_sha256": base_plan["plan_sha256"],
            "base_bundle_sha256": base_plan["bundle_sha256"], "base_report_sha256": base_report["report_sha256"],
            "probe_plan_sha256": probe_plan["plan_sha256"], "probe_report_sha256": probe_report["report_sha256"],
            "candidate": candidate, "prediction": _descriptor(projected), "bundle_sha256": _sha(bundle), "code_sha256": _code_sha(),
            "runtime": base_plan["runtime"], "value_count": 19200, "model_case_previously_observed": True,
            "numerical_refitting_allowed": False, "fresh_model_holdout": False, "hardware_partitioning_established": False,
            "full_first_layer_qualified": False, "global_exactness_activation_allowed": False}


def build_survivor_plan(sources: OutputSources, base_plan: dict[str, Any], base_bundle: dict[str, Any], base_report: dict[str, Any], source_summary: dict[str, Any], probe_plan: dict[str, Any], probe_bundle: dict[str, Any], probe_report: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    code = _code_sha()
    _, _, predictions, candidate = _material(sources, base_plan, base_bundle, base_report, source_summary, probe_plan, probe_bundle, probe_report)
    weights = _state_array(base_bundle["weight_bits"], [640, 1024]).tolist()
    projected = np.asarray([[[survivor_dot(row, weight, candidate) for weight in weights] for row in predictions["concatenated"][0].tolist()]], dtype=np.uint16)
    bundle = {"projected_bits": projected.tolist()}
    if _code_sha() != code:
        raise ValueError("Output-survivor source changed during prediction")
    return _seal(_plan_body(sources, base_plan, base_report, probe_plan, probe_report, candidate, projected, bundle), "plan_sha256"), bundle


def check_survivor_plan(sources: OutputSources, base_plan: dict[str, Any], base_bundle: dict[str, Any], base_report: dict[str, Any], source_summary: dict[str, Any], probe_plan: dict[str, Any], probe_bundle: dict[str, Any], probe_report: dict[str, Any], plan: dict[str, Any], bundle: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    _check_hash(plan, "plan_sha256")
    probability, values, predictions, candidate = _material(sources, base_plan, base_bundle, base_report, source_summary, probe_plan, probe_bundle, probe_report)
    if set(bundle) != {"projected_bits"}:
        raise ValueError("Unexpected survivor prediction payload")
    projected = _state_array(bundle["projected_bits"], [1, 30, 640])
    expected = _seal(_plan_body(sources, base_plan, base_report, probe_plan, probe_report, candidate, projected, bundle), "plan_sha256")
    if canonical_json(expected) != canonical_json(plan):
        raise ValueError("Survivor model-case plan commitment/scope mismatch")
    return probability, values, {**predictions, "projected": projected}


def _report(plan: dict[str, Any], base_plan: dict[str, Any], probability: np.ndarray, values: np.ndarray, predictions: dict[str, np.ndarray], native_report: dict[str, Any]) -> dict[str, Any]:
    comparison = output_report({**base_plan, "plan_sha256": plan["plan_sha256"]}, probability, values, predictions, native_report["observations"], native_report["runtime"])
    return _seal({"schema_version": 1, "scope": SCOPE, "plan_sha256": plan["plan_sha256"], "native_report": native_report,
                  "comparison": {key: value for key, value in comparison.items() if key not in ("observations", "scope", "report_sha256")},
                  "survivor_reproduces_model_case": comparison["candidate_passes"], "fresh_model_holdout": False,
                  "hardware_partitioning_established": False, "full_first_layer_qualified": False, "global_exactness_activation_allowed": False}, "report_sha256")


def acquire_survivor(context: tuple[Any, ...], plan: dict[str, Any], bundle: dict[str, Any], model: Any) -> dict[str, Any]:
    probability, values, predictions = check_survivor_plan(*context, plan, bundle)
    sources, base_plan, base_bundle = context[:3]
    native = acquire_output(sources, model, base_plan, base_bundle)
    _code_sha()
    return _report(plan, base_plan, probability, values, predictions, native)


def verify_survivor(context: tuple[Any, ...], plan: dict[str, Any], bundle: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    try:
        probability, values, predictions = check_survivor_plan(*context, plan, bundle)
        sources, base_plan, base_bundle = context[:3]
        if not verify_output(sources, base_plan, base_bundle, report["native_report"])["valid"]:
            raise ValueError("Native output regression evidence is inconsistent")
        expected = _report(plan, base_plan, probability, values, predictions, report["native_report"])
        return {"valid": canonical_json(report) == canonical_json(expected), "mode": "integrity_only_with_source_and_coordinate_checks",
                "survivor_reproduces_model_case": expected["survivor_reproduces_model_case"], "mismatch_counts": expected["comparison"]["mismatch_counts"],
                "fresh_model_holdout": False, "global_exactness_activation_allowed": False}
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError, StopIteration) as error:
        return {"valid": False, "reason": str(error)}


def survivor_summary(plan: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in report.items() if key not in ("native_report", "comparison", "report_sha256")}
    body.update({"source_report_sha256": report["report_sha256"], "native_report_sha256": report["native_report"]["report_sha256"],
                 "candidate": plan["candidate"], "probe_plan_sha256": plan["probe_plan_sha256"], "probe_report_sha256": plan["probe_report_sha256"],
                 "base_plan_sha256": plan["base_plan_sha256"], "base_report_sha256": plan["base_report_sha256"],
                 "comparison": {key: value for key, value in report["comparison"].items() if key != "mismatches"}})
    return _seal(body, "summary_sha256")
