from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

from .gemma_exp_lookup import CheckedExpLookup, _code_sha as exp_code_sha
from .gemma_softmax_slice import SoftmaxSources, OFFSETS, _maximum, _code_sha as source_code_sha, verify_softmax, acquire_softmax, softmax_report
from .gemma_float_semantics import encode_bfloat16_rne
from .gemma_reduction_semantics import decode_finite_float32
from .reference_gemma import _rms_f32_add, _rms_f32_round, _rms_f32_value
from .gemma_rotary_slice import _sha, _seal, _check_hash
from .serialization import canonical_json

SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
SCOPE = "Explicit-exponential-lookup softmax on verified serialized score inputs; independent FP32 max/subtract/XOR sum/division with runtime-bound empirical exp; not native exponential reconstruction or full-layer qualification."


def _code_sha() -> str:
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Lookup-softmax source changed after process import")
    return _sha({"module": SOURCE_SHA256, "original_softmax": source_code_sha(), "exponential_lookup": exp_code_sha()})


def lookup_softmax_row(bits: list[int], lookup: CheckedExpLookup, runtime: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(bits, list) or len(bits) != 30 or any(type(value) is not int or not 0 <= value <= 0xFFFFFFFF for value in bits):
        raise ValueError("Lookup softmax requires thirty float32 encodings")
    for value in bits:
        decode_finite_float32(value)
    maximum = [*bits, 0xFF800000, 0xFF800000]
    for offset in OFFSETS:
        maximum = [_maximum(maximum[lane], maximum[lane ^ offset]) for lane in range(32)]
    shifted = [_rms_f32_round(_rms_f32_value(value) - _rms_f32_value(maximum[lane])) for lane, value in enumerate(bits)]
    exponential = [lookup.predict_bits(value, runtime) for value in shifted]
    sums, tree = [*exponential, lookup.predict_bits(0xFF800000, runtime), lookup.predict_bits(0xFF800000, runtime)], []
    for offset in OFFSETS:
        sums = [_rms_f32_add(sums[lane], sums[lane ^ offset]) for lane in range(32)]
        tree.append(sums)
    output = [_rms_f32_round(_rms_f32_value(value) / _rms_f32_value(sums[lane])) for lane, value in enumerate(exponential)]
    return {"maximum_bits": maximum[:30], "shifted_bits": shifted, "exponential_bits": exponential, "xor_sum_stages": tree,
            "denominator_bits": sums[:30], "output_f32_bits": output, "output_bf16_bits": [encode_bfloat16_rne(*decode_finite_float32(value)) for value in output]}


def _check_provider(lookup: CheckedExpLookup) -> None:
    if type(lookup) is not CheckedExpLookup or getattr(lookup.predict_bits, "__func__", None) is not CheckedExpLookup.predict_bits:
        raise ValueError("Lookup softmax requires the registered checked exponential implementation")


def build_lookup_softmax_plan(sources: SoftmaxSources, original_plan: dict[str, Any], original_bundle: dict[str, Any], original_report: dict[str, Any], lookup: CheckedExpLookup) -> tuple[dict[str, Any], dict[str, Any]]:
    _check_provider(lookup)
    code = _code_sha()
    checked = verify_softmax(sources, original_plan, original_bundle, original_report)
    if not checked["valid"]:
        raise ValueError("Original softmax evidence is not intact")
    inputs = sources.inputs()
    runtime = original_plan["runtime"]
    rows = [lookup_softmax_row(row, lookup, runtime) for row in inputs.reshape(120, 30).tolist()]
    bundle = {"input_bits": inputs.tolist(), "rows": rows}
    if _code_sha() != code:
        raise ValueError("Lookup-softmax source changed during prediction")
    body = {"schema_version": 1, "scope": SCOPE, "sources": sources.commitments(), "original_plan_sha256": original_plan["plan_sha256"],
            "original_report_sha256": original_report["report_sha256"], "lookup_evidence": lookup.evidence,
            "code_sha256": code, "runtime": copy.deepcopy(runtime), "row_count": 120, "value_count": 3600,
            "bundle_sha256": _sha(bundle), "input_bits_sha256": _sha(inputs.tolist()),
            "profile": {"maximum_subtract": "source warp32 FP32", "exponential": "checked full nonpositive float32 lookup", "xor_offsets": list(OFFSETS), "sum_divide": "float32 RNE", "cast": "bfloat16 RNE"},
            "prefix_boundary_reused": True, "native_exponential_reconstructed": False, "fused_internal_stages_observed": False,
            "value_aggregation_executed": False, "full_first_layer_qualified": False, "global_exactness_activation_allowed": False}
    return _seal(body, "plan_sha256"), bundle


def check_lookup_softmax_plan(sources: SoftmaxSources, original_plan: dict[str, Any], original_bundle: dict[str, Any], original_report: dict[str, Any], lookup: CheckedExpLookup, plan: dict[str, Any], bundle: dict[str, Any]) -> None:
    _check_hash(plan, "plan_sha256")
    expected_plan, expected_bundle = build_lookup_softmax_plan(sources, original_plan, original_bundle, original_report, lookup)
    if canonical_json(plan) != canonical_json(expected_plan) or canonical_json(bundle) != canonical_json(expected_bundle):
        raise ValueError("Lookup-softmax plan does not reproduce independently")


def _report(plan: dict[str, Any], bundle: dict[str, Any], native_report: dict[str, Any]) -> dict[str, Any]:
    comparison = softmax_report(plan, bundle, native_report["observations"], native_report["runtime"])
    agreement = comparison["candidate_passes"] and all(all(count == 0 for count in diagnostic["candidate_vs_staged_aten"].values()) and diagnostic["staged_vs_fused_fp32"] == 0 and diagnostic["staged_vs_fused_bf16"] == 0 for diagnostic in comparison["diagnostics"])
    body = {"schema_version": 1, "scope": SCOPE, "plan_sha256": plan["plan_sha256"], "lookup_evidence": plan["lookup_evidence"],
            "native_report": native_report, "comparison": {key: value for key, value in comparison.items() if key not in ("observations", "scope", "report_sha256")},
            "lookup_softmax_passes": agreement, "native_exponential_reconstructed": False, "fused_internal_stages_observed": False,
            "prefix_independently_recomputed": False, "value_aggregation_executed": False,
            "full_first_layer_qualified": False, "hardware_semantics_established": False, "global_exactness_activation_allowed": False}
    return _seal(body, "report_sha256")


def acquire_lookup_softmax(sources: SoftmaxSources, original_plan: dict[str, Any], original_bundle: dict[str, Any], original_report: dict[str, Any], lookup: CheckedExpLookup, plan: dict[str, Any], bundle: dict[str, Any], model: Any) -> dict[str, Any]:
    check_lookup_softmax_plan(sources, original_plan, original_bundle, original_report, lookup, plan, bundle)
    native_report = acquire_softmax(sources, model, original_plan, original_bundle)
    _code_sha()
    return _report(plan, bundle, native_report)


def verify_lookup_softmax(sources: SoftmaxSources, original_plan: dict[str, Any], original_bundle: dict[str, Any], original_report: dict[str, Any], lookup: CheckedExpLookup, plan: dict[str, Any], bundle: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    try:
        check_lookup_softmax_plan(sources, original_plan, original_bundle, original_report, lookup, plan, bundle)
        if not verify_softmax(sources, original_plan, original_bundle, report["native_report"])["valid"]:
            raise ValueError("Native softmax acquisition is inconsistent")
        expected = _report(plan, bundle, report["native_report"])
        return {"valid": canonical_json(expected) == canonical_json(report), "mode": "integrity_and_lookup_softmax_recomputation",
                "lookup_softmax_passes": expected["lookup_softmax_passes"], "mismatch_counts": expected["comparison"]["mismatch_counts"],
                "global_exactness_activation_allowed": False}
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError) as error:
        return {"valid": False, "reason": str(error)}


def lookup_softmax_summary(plan: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in report.items() if key not in ("native_report", "comparison", "report_sha256")}
    body.update({"source_report_sha256": report["report_sha256"], "native_report_sha256": report["native_report"]["report_sha256"],
                 "original_plan_sha256": plan["original_plan_sha256"], "original_report_sha256": plan["original_report_sha256"],
                 "comparison": {key: value for key, value in report["comparison"].items() if key != "mismatches"},
                 "code_sha256": plan["code_sha256"], "row_count": 120, "value_count": 3600})
    return _seal(body, "summary_sha256")
