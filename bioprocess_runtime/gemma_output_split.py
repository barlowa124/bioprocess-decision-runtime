from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .gemma_attention_output import OutputSources, check_output_plan, verify_output, acquire_output, output_report, _code_sha as output_code_sha
from .gemma_attention_entry import _descriptor, _state_array
from .gemma_wmma_candidate import DENSE_SPLIT_PROFILE, operand_aligned_product_bits, _merge_split_partials
from .gemma_float_semantics import decode_finite_bfloat16
from .gemma_rotary_slice import _sha, _seal, _check_hash
from .serialization import canonical_json

SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
SCOPE = "Development-only K1024 output-projection transfer of the prior K64/BF16-partial/sequential-FP32 merge rule, selected after split-K kernel observation on the same fixed model case; not a fresh holdout or hardware partition proof."


def _code_sha() -> str:
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Output split-K source changed after process import")
    return _sha({"module": SOURCE_SHA256, "base_output": output_code_sha()})


def k1024_split_bits(left: list[int], right: list[int]) -> int:
    if len(left) != 1024 or len(right) != 1024 or DENSE_SPLIT_PROFILE != (64, "bfloat16_rne", "sequential_float32_rne"):
        raise ValueError("Output split-K hypothesis is restricted to the frozen K1024/K64 recipe")
    partials = [decode_finite_bfloat16(operand_aligned_product_bits(left[start:start + 64], right[start:start + 64]))[0] for start in range(0, 1024, 64)]
    return _merge_split_partials(partials, DENSE_SPLIT_PROFILE[2])


def predict_split_projection(inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if inputs.dtype != np.uint16 or weights.dtype != np.uint16 or inputs.shape != (1, 30, 1024) or weights.shape != (640, 1024):
        raise ValueError("Unsupported split output-projection geometry")
    rows = weights.tolist()
    return np.asarray([[[k1024_split_bits(row, weight) for weight in rows] for row in inputs[0].tolist()]], dtype=np.uint16)


def _base(sources: OutputSources, base_plan: dict[str, Any], base_bundle: dict[str, Any], base_report: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    checked = verify_output(sources, base_plan, base_bundle, base_report)
    if not checked["valid"] or any(base_report["mismatch_counts"][name] != [0, 0, 0] for name in ("head_output", "concatenated")) or not all(all(item.values()) for item in base_report["checks"]):
        raise ValueError("Split projection requires intact, exactly matching upstream aggregation/input evidence")
    if not all(any("splitKreduce" in name for name in pair["traced"]["projection_cuda_events"]) for pair in base_report["observations"]):
        raise ValueError("Missing original split-K projection kernel evidence")
    return check_output_plan(sources, base_plan, base_bundle)


def build_split_output_plan(sources: OutputSources, base_plan: dict[str, Any], base_bundle: dict[str, Any], base_report: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    code = _code_sha()
    _, _, base = _base(sources, base_plan, base_bundle, base_report)
    projected = predict_split_projection(base["concatenated"], _state_array(base_bundle["weight_bits"], [640, 1024]))
    bundle = {"projected_bits": projected.tolist()}
    if _code_sha() != code:
        raise ValueError("Split-projection source changed during prediction")
    body = {"schema_version": 1, "scope": SCOPE, "sources": sources.commitments(), "base_plan_sha256": base_plan["plan_sha256"],
            "base_bundle_sha256": base_plan["bundle_sha256"], "base_report_sha256": base_report["report_sha256"],
            "profile": list(DENSE_SPLIT_PROFILE), "reduction_length": 1024, "partial_count": 16,
            "input": base_plan["predictions"]["concatenated"], "weight": base_plan["weight"], "prediction": _descriptor(projected),
            "bundle_sha256": _sha(bundle), "code_sha256": code, "runtime": copy.deepcopy(base_plan["runtime"]),
            "value_count": 19200, "selected_after_kernel_observation": True, "same_case_observations_already_available": True,
            "numerical_parameters_refitted": False, "fresh_holdout_validation": False, "hardware_partitioning_established": False,
            "full_first_layer_qualified": False, "global_exactness_activation_allowed": False}
    return _seal(body, "plan_sha256"), bundle


def check_split_output_plan(sources: OutputSources, base_plan: dict[str, Any], base_bundle: dict[str, Any], base_report: dict[str, Any], plan: dict[str, Any], bundle: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    _check_hash(plan, "plan_sha256")
    probability, values, base = _base(sources, base_plan, base_bundle, base_report)
    expected = {"schema_version": 1, "scope": SCOPE, "sources": sources.commitments(), "base_plan_sha256": base_plan["plan_sha256"],
                "base_bundle_sha256": base_plan["bundle_sha256"], "base_report_sha256": base_report["report_sha256"],
                "profile": list(DENSE_SPLIT_PROFILE), "reduction_length": 1024, "partial_count": 16,
                "input": base_plan["predictions"]["concatenated"], "weight": base_plan["weight"], "bundle_sha256": _sha(bundle),
                "code_sha256": _code_sha(), "runtime": base_plan["runtime"], "value_count": 19200,
                "selected_after_kernel_observation": True, "same_case_observations_already_available": True,
                "numerical_parameters_refitted": False, "fresh_holdout_validation": False, "hardware_partitioning_established": False,
                "full_first_layer_qualified": False, "global_exactness_activation_allowed": False}
    if any(canonical_json(plan.get(key)) != canonical_json(value) for key, value in expected.items()) or set(bundle) != {"projected_bits"}:
        raise ValueError("Split output profile/source/scope mismatch")
    projected = _state_array(bundle["projected_bits"], [1, 30, 640])
    if canonical_json(plan["prediction"]) != canonical_json(_descriptor(projected)):
        raise ValueError("Split output prediction commitment mismatch")
    return probability, values, {**base, "projected": projected}


def _report(plan: dict[str, Any], base_plan: dict[str, Any], probability: np.ndarray, values: np.ndarray, predictions: dict[str, np.ndarray], native_report: dict[str, Any]) -> dict[str, Any]:
    comparison = output_report({**base_plan, "plan_sha256": plan["plan_sha256"]}, probability, values, predictions, native_report["observations"], native_report["runtime"])
    return _seal({"schema_version": 1, "scope": SCOPE, "plan_sha256": plan["plan_sha256"], "native_report": native_report,
                  "comparison": {key: value for key, value in comparison.items() if key not in ("observations", "scope", "report_sha256")},
                  "split_projection_matches_case": comparison["candidate_passes"], "selected_after_kernel_observation": True,
                  "fresh_holdout_validation": False, "hardware_partitioning_established": False,
                  "full_first_layer_qualified": False, "global_exactness_activation_allowed": False}, "report_sha256")


def acquire_split_output(sources: OutputSources, base_plan: dict[str, Any], base_bundle: dict[str, Any], base_report: dict[str, Any], plan: dict[str, Any], bundle: dict[str, Any], model: Any) -> dict[str, Any]:
    probability, values, predictions = check_split_output_plan(sources, base_plan, base_bundle, base_report, plan, bundle)
    native = acquire_output(sources, model, base_plan, base_bundle)
    _code_sha()
    return _report(plan, base_plan, probability, values, predictions, native)


def verify_split_output(sources: OutputSources, base_plan: dict[str, Any], base_bundle: dict[str, Any], base_report: dict[str, Any], plan: dict[str, Any], bundle: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    try:
        probability, values, predictions = check_split_output_plan(sources, base_plan, base_bundle, base_report, plan, bundle)
        if not verify_output(sources, base_plan, base_bundle, report["native_report"])["valid"]:
            raise ValueError("Split output native evidence is invalid")
        expected = _report(plan, base_plan, probability, values, predictions, report["native_report"])
        return {"valid": canonical_json(report) == canonical_json(expected), "mode": "integrity_only_with_upstream_coordinate_checks",
                "split_projection_matches_case": expected["split_projection_matches_case"], "mismatch_counts": expected["comparison"]["mismatch_counts"],
                "fresh_holdout_validation": False, "global_exactness_activation_allowed": False}
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError) as error:
        return {"valid": False, "reason": str(error)}


def split_output_summary(plan: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in report.items() if key not in ("native_report", "comparison", "report_sha256")}
    body.update({"source_report_sha256": report["report_sha256"], "base_plan_sha256": plan["base_plan_sha256"], "base_report_sha256": plan["base_report_sha256"],
                 "native_report_sha256": report["native_report"]["report_sha256"], "profile": plan["profile"], "code_sha256": plan["code_sha256"],
                 "comparison": {key: value for key, value in report["comparison"].items() if key != "mismatches"}})
    return _seal(body, "summary_sha256")
