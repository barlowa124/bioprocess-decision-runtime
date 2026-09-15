from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from .gemma_attention_entry import _bits, _descriptor, _state_array, _model_parameters
from .gemma_k128_dense import DenseSources, verify_dense
from .gemma_output_survivor import verify_survivor
from .gemma_rsqrt_lookup import CheckedRsqrtLookup, _rms_lookup_row, _runtime
from .gemma_float_semantics import bfloat16_add_bits
from .gemma_ir_interpreter import bind_model_tensors
from .reference_gemma import _observe_rms_module, _rms_code_commitment
from .gemma_rotary_slice import _sha, _seal, _check_hash
from .serialization import canonical_json

SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
WEIGHT = "model.layers.0.post_attention_layernorm.weight"
SCOPE = "Fixed-input post-attention lookup RMS and BF16 residual addition from verified serialized projection/hidden boundaries; native scalar stages and residual compared before pre-feedforward normalization, not complete-layer qualification."


def _code_sha() -> str:
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Post-attention source changed after import")
    from .gemma_attention_entry import _entry_code_sha
    return _sha({"module": SOURCE_SHA256, "rms_arithmetic": _rms_code_commitment(), "lookup_and_scalar_dependencies": _entry_code_sha()})


@dataclass(frozen=True)
class PostAttentionSources:
    survivor_context: tuple[Any, ...]
    survivor_plan: dict[str, Any]
    survivor_bundle: dict[str, Any]
    survivor_report: dict[str, Any]
    dense_sources: DenseSources
    dense_plan: dict[str, Any]
    dense_bundle: dict[str, Any]
    dense_report: dict[str, Any]

    @property
    def program(self) -> dict[str, Any]:
        return self.survivor_context[0].program

    @property
    def lookup(self) -> CheckedRsqrtLookup:
        return self.survivor_context[0].softmax.scores.lookup

    @property
    def runtime(self) -> dict[str, Any]:
        return self.survivor_plan["runtime"]

    def boundaries(self) -> tuple[np.ndarray, np.ndarray]:
        verified = verify_survivor(self.survivor_context, self.survivor_plan, self.survivor_bundle, self.survivor_report)
        dense = verify_dense(self.dense_sources, self.dense_plan, self.dense_bundle, self.dense_report)
        if not verified["valid"] or not verified["survivor_reproduces_model_case"] or not dense["valid"] or not dense["candidate_passes_dense_holdout"]:
            raise ValueError("Post-attention requires passing projection and dense holdout evidence")
        if canonical_json(self.dense_plan["candidate"]) != canonical_json(self.survivor_plan["candidate"]) or canonical_json(self.dense_plan["runtime"]) != canonical_json(self.runtime) or self.dense_sources.model_plan["plan_sha256"] != self.survivor_context[1]["plan_sha256"]:
            raise ValueError("Post-attention upstream profile/runtime sources disagree")
        if type(self.lookup) is not CheckedRsqrtLookup or getattr(self.lookup.predict_bits, "__func__", None) is not CheckedRsqrtLookup.predict_bits:
            raise ValueError("Post-attention requires the registered checked rsqrt provider")
        entry = self.survivor_context[0].softmax.scores.rotary_bundle["entry_bundle"]
        projected = _state_array(self.survivor_bundle["projected_bits"], [1, 30, 640])
        residual = _state_array(entry["state_bits"]["hidden.0"], [1, 30, 640])
        return projected, residual

    def commitments(self) -> dict[str, Any]:
        return {"program_sha256": self.program["program_sha256"], "survivor_plan_sha256": self.survivor_plan["plan_sha256"],
                "survivor_report_sha256": self.survivor_report["report_sha256"], "survivor_bundle_sha256": self.survivor_plan["bundle_sha256"],
                "dense_plan_sha256": self.dense_plan["plan_sha256"], "dense_report_sha256": self.dense_report["report_sha256"], "lookup_evidence": self.lookup.evidence}


def post_instructions(program: dict[str, Any]) -> list[dict[str, Any]]:
    specifications = (("layer.0.attention.post_normalized", "RMS_NORM", ["layer.0.attention.projected"], [WEIGHT]),
                      ("layer.0.post_attention_residual", "ADD", ["hidden.0", "layer.0.attention.post_normalized"], []))
    result = []
    for output, opcode, inputs, parameters in specifications:
        found = [item for item in program["instructions"] if item["outputs"] == [output]]
        if len(found) != 1 or found[0]["opcode"] != opcode or found[0]["inputs"] != inputs or found[0]["parameter_refs"] != parameters:
            raise ValueError("Unsupported post-attention instruction binding")
        result.append(found[0])
    if set(result[0]["attributes"]) != {"epsilon"} or type(result[0]["attributes"]["epsilon"]) not in (float, int) or not 0 < result[0]["attributes"]["epsilon"] < 1 or result[1]["attributes"] != {"output_dtype": "torch.bfloat16"}:
        raise ValueError("Unsupported RMS epsilon or residual dtype")
    return result


def predict_post_attention(projected: np.ndarray, residual: np.ndarray, weights: list[int], epsilon: float, lookup: CheckedRsqrtLookup, runtime: dict[str, Any]) -> dict[str, Any]:
    if projected.dtype != np.uint16 or residual.dtype != np.uint16 or projected.shape != (1, 30, 640) or residual.shape != projected.shape or len(weights) != 640:
        raise ValueError("Post-attention prediction requires [1,30,640] BF16 boundaries")
    rows = [_rms_lookup_row(row, weights, epsilon, lookup, runtime) for row in projected[0].tolist()]
    normalized = np.asarray([row["output_bits"] for row in rows], dtype=np.uint16)[None, ...]
    combined = np.asarray([bfloat16_add_bits(int(left), int(right)) for left, right in zip(residual.reshape(-1), normalized.reshape(-1))], dtype=np.uint16).reshape(1, 30, 640)
    return {"normalized_bits": normalized.tolist(), "residual_bits": combined.tolist(),
            "stages": {key: [row[key] for row in rows] for key in ("mean_bits", "denominator_bits", "rsqrt_bits")}}


def _model_context(sources: PostAttentionSources, model: Any) -> tuple[Any, dict[str, Any]]:
    import torch
    parameters = _model_parameters(sources.program, model)
    module = model.model.layers[0].post_attention_layernorm
    instruction = post_instructions(sources.program)[0]
    weight = parameters[WEIGHT]
    if module.weight is not weight or weight.dtype != torch.bfloat16 or list(weight.shape) != [640] or weight.device != model.model.embed_tokens.weight.device or module.eps != instruction["attributes"]["epsilon"]:
        raise ValueError("Post-attention RMS module/weight binding mismatch")
    binding = {"class": type(model).__module__ + "." + type(model).__name__, "config_sha256": _sha(model.config.to_dict()), "parameter_commitments_sha256": _sha(sources.program["parameter_commitments"])}
    if canonical_json(binding) != canonical_json(sources.survivor_context[1]["model_binding"]) or canonical_json(_runtime()) != canonical_json(sources.runtime):
        raise ValueError("Post-attention model/runtime differs from source evidence")
    return weight, binding


def _plan_body(sources: PostAttentionSources, projected: np.ndarray, residual: np.ndarray, weights: np.ndarray, binding: dict[str, Any], predictions: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": 1, "scope": SCOPE, "sources": sources.commitments(), "instructions": post_instructions(sources.program),
            "code_sha256": _code_sha(), "runtime": sources.runtime, "model_binding": binding,
            "inputs": {"projected": _descriptor(projected), "residual_base": _descriptor(residual)}, "weight": _descriptor(weights),
            "predictions": {name: _descriptor(_state_array(predictions[key], [1, 30, 640])) for name, key in (("normalized", "normalized_bits"), ("residual", "residual_bits"))},
            "scalar_stages_sha256": _sha(predictions["stages"]), "bundle_sha256": _sha(bundle), "value_count": 38400, "scalar_stage_positions": 90,
            "repetitions": 3, "prefix_boundary_reused": True, "pre_feedforward_norm_executed": False, "mlp_executed": False,
            "native_arithmetic_reconstructed": False, "full_first_layer_qualified": False, "hardware_semantics_established": False,
            "global_exactness_activation_allowed": False}


def build_post_plan(sources: PostAttentionSources, model: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    code = _code_sha()
    projected, residual = sources.boundaries()
    weight, binding = _model_context(sources, model)
    weights = _bits(weight)
    predictions = predict_post_attention(projected, residual, weights.tolist(), post_instructions(sources.program)[0]["attributes"]["epsilon"], sources.lookup, sources.runtime)
    bundle = {"weight_bits": weights.tolist(), "predictions": predictions}
    bind_model_tensors(sources.program, model, verify_hashes=True)
    if _code_sha() != code:
        raise ValueError("Post-attention code changed during prediction")
    return _seal(_plan_body(sources, projected, residual, weights, binding, predictions, bundle), "plan_sha256"), bundle


def check_post_plan(sources: PostAttentionSources, plan: dict[str, Any], bundle: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    _check_hash(plan, "plan_sha256")
    projected, residual = sources.boundaries()
    weights = _state_array(bundle["weight_bits"], [640])
    if _descriptor(weights)["sha256"] != sources.program["parameter_commitments"][WEIGHT]["sha256"]:
        raise ValueError("Post-attention weight commitment mismatch")
    prediction = predict_post_attention(projected, residual, weights.tolist(), post_instructions(sources.program)[0]["attributes"]["epsilon"], sources.lookup, sources.runtime)
    expected_bundle = {"weight_bits": weights.tolist(), "predictions": prediction}
    expected_plan = _seal(_plan_body(sources, projected, residual, weights, sources.survivor_context[1]["model_binding"], prediction, expected_bundle), "plan_sha256")
    if canonical_json(bundle) != canonical_json(expected_bundle) or canonical_json(plan) != canonical_json(expected_plan):
        raise ValueError("Post-attention plan does not reproduce independently")
    return projected, residual, prediction


def _capture_post(model: Any, token_ids: list[list[int]], traced: bool) -> dict[str, Any]:
    import torch
    from torch.profiler import profile, ProfilerActivity

    layer = model.model.layers[0]
    norm = layer.post_attention_layernorm
    original = norm.forward
    record, handles = {}, []

    class Complete(Exception):
        pass

    def layer_input(module: Any, args: Any, kwargs: Any) -> None:
        if "residual_base" in record:
            raise ValueError("Repeated first-layer input")
        record["residual_base"] = _bits(args[0] if args else kwargs["hidden_states"]).tolist()

    def capture(name: str):
        def hook(module: Any, args: Any, output: Any) -> None:
            if name in record or list(output.shape) != [1, 30, 640]:
                raise ValueError("Unexpected post-attention boundary shape/occurrence")
            record[name] = _bits(output).tolist()
        return hook

    def observe(value: Any) -> Any:
        if "stages" in record or "projected" not in record or canonical_json(_bits(value).tolist()) != canonical_json(record["projected"]):
            raise ValueError("RMS input is not linked to original output projection")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiled:
            output, stages = _observe_rms_module(original, value)
            torch.cuda.synchronize()
        record["stages"] = stages
        record["cuda_events"] = sorted({event.name for event in profiled.events() if event.device_type == torch.autograd.DeviceType.CUDA})
        return output

    def stop(module: Any, args: Any) -> None:
        if "residual" in record or list(args[0].shape) != [1, 30, 640]:
            raise ValueError("Unexpected residual/pre-feedforward boundary")
        record["residual"] = _bits(args[0]).tolist()
        raise Complete()

    stopped = False
    try:
        handles.append(layer.register_forward_pre_hook(layer_input, with_kwargs=True))
        handles.append(layer.self_attn.o_proj.register_forward_hook(capture("projected")))
        handles.append(norm.register_forward_hook(capture("normalized")))
        handles.append(layer.pre_feedforward_layernorm.register_forward_pre_hook(stop))
        ids = torch.tensor(token_ids, dtype=torch.int64, device=norm.weight.device)
        with torch.no_grad():
            if traced:
                with patch.object(norm, "forward", side_effect=observe):
                    model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1)
            else:
                model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1)
    except Complete:
        stopped = True
    finally:
        for handle in handles:
            handle.remove()
    expected = {"projected", "residual_base", "normalized", "residual"} | ({"stages", "cuda_events"} if traced else set())
    if not stopped or set(record) != expected:
        raise ValueError("Original forward did not stop at the complete residual boundary")
    return record


def post_report(plan: dict[str, Any], projected: np.ndarray, residual: np.ndarray, prediction: dict[str, Any], observations: list[dict[str, Any]], runtime: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(observations, list) or len(observations) != 3:
        raise ValueError("Expected three post-attention repetitions")
    outputs = {"normalized": _state_array(prediction["normalized_bits"], [1, 30, 640]), "residual": _state_array(prediction["residual_bits"], [1, 30, 640])}
    mismatches, checks = [], []
    geometry = {"input_shape": [1, 30, 640], "input_strides": [19200, 640, 1], "alignment_mod16": 0, "input_dtype": "torch.float32", "axes": [-1], "keepdim": True}
    for repetition, pair in enumerate(observations):
        record, plain = pair["traced"], pair["untraced"]
        inputs_match = all(np.array_equal(_state_array(item[name], [1, 30, 640]), expected) for item in (record, plain) for name, expected in (("projected", projected), ("residual_base", residual)))
        observed = {name: _state_array(record[name], [1, 30, 640]) for name in outputs}
        for name, expected in outputs.items():
            for coordinate in np.argwhere(observed[name] != expected):
                mismatches.append({"repetition": repetition, "stage": name, "coordinate": coordinate.tolist(), "predicted_bits": int(expected[tuple(coordinate)]), "observed_bits": int(observed[name][tuple(coordinate)])})
        for name, expected in prediction["stages"].items():
            values = record["stages"][name]
            if not isinstance(values, list) or len(values) != 30 or any(type(value) is not int or not 0 <= value <= 0xFFFFFFFF for value in values):
                raise ValueError("Malformed post-attention FP32 stage")
            for row, (wanted, value) in enumerate(zip(expected, values)):
                if wanted != value:
                    mismatches.append({"repetition": repetition, "stage": name, "coordinate": [0, row, 0], "predicted_bits": wanted, "observed_bits": value})
        events = record["cuda_events"]
        if not isinstance(events, list) or not events or any(not isinstance(event, str) or not event for event in events):
            raise ValueError("Missing post-attention CUDA provenance")
        checks.append({"source_inputs_match": inputs_match, "mean_geometry_matches": canonical_json(record["stages"]["mean_input_metadata"]) == canonical_json(geometry),
                       "untraced_outputs_match": all(np.array_equal(_state_array(plain[name], [1, 30, 640]), observed[name]) for name in outputs)})
    stage_order = {name: index for index, name in enumerate(("mean_bits", "denominator_bits", "rsqrt_bits", "normalized", "residual"))}
    mismatches.sort(key=lambda item: (item["repetition"], stage_order[item["stage"]], item["coordinate"]))
    counts = {name: [sum(item["repetition"] == repetition and item["stage"] == name for item in mismatches) for repetition in range(3)] for name in (*prediction["stages"], *outputs)}
    runtime_match = canonical_json(runtime) == canonical_json(plan["runtime"])
    first = next((key for key in checks[0] if not all(item[key] for item in checks)), None) or (mismatches[0] if mismatches else None) or (None if runtime_match else "runtime")
    return _seal({"schema_version": 1, "scope": SCOPE, "plan_sha256": plan["plan_sha256"], "runtime": runtime, "runtime_matches_plan": runtime_match,
                  "observations": observations, "checks": checks, "mismatches": mismatches, "mismatch_counts": counts, "first_divergence": first,
                  "post_attention_matches": not mismatches and runtime_match and all(all(item.values()) for item in checks), "value_count": 38400,
                  "scalar_stage_positions": 90, "prefix_independently_recomputed": False, "pre_feedforward_norm_executed": False, "mlp_executed": False,
                  "native_arithmetic_reconstructed": False, "full_first_layer_qualified": False, "hardware_semantics_established": False,
                  "global_exactness_activation_allowed": False}, "report_sha256")


def acquire_post(sources: PostAttentionSources, model: Any, plan: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    projected, residual, prediction = check_post_plan(sources, plan, bundle)
    _model_context(sources, model)
    ids = sources.survivor_context[0].softmax.scores.rotary_bundle["entry_plan"]["input_token_ids"]
    observations = [{"traced": _capture_post(model, ids, True), "untraced": _capture_post(model, ids, False)} for _ in range(3)]
    bind_model_tensors(sources.program, model, verify_hashes=True)
    _code_sha()
    return post_report(plan, projected, residual, prediction, observations, _runtime())


def verify_post(sources: PostAttentionSources, plan: dict[str, Any], bundle: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    try:
        projected, residual, prediction = check_post_plan(sources, plan, bundle)
        expected = post_report(plan, projected, residual, prediction, report["observations"], report["runtime"])
        return {"valid": canonical_json(report) == canonical_json(expected), "mode": "independent_post_attention_recomputation_and_integrity",
                "post_attention_matches": expected["post_attention_matches"], "mismatch_counts": expected["mismatch_counts"],
                "first_divergence": expected["first_divergence"], "global_exactness_activation_allowed": False}
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError) as error:
        return {"valid": False, "reason": str(error)}


def post_summary(plan: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in report.items() if key not in ("observations", "mismatches", "report_sha256")}
    body.update({"source_report_sha256": report["report_sha256"], "sources": plan["sources"], "code_sha256": plan["code_sha256"],
                 "cuda_events": [pair["traced"]["cuda_events"] for pair in report["observations"]],
                 "observed_output_hashes": {name: [_descriptor(_state_array(pair["traced"][name], [1, 30, 640]))["sha256"] for pair in report["observations"]] for name in ("normalized", "residual")}})
    return _seal(body, "summary_sha256")
