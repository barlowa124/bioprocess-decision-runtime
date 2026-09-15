from __future__ import annotations

import hashlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from .gemma_attention_entry import _bits, _descriptor, _state_array, _model_parameters
from .gemma_post_attention import PostAttentionSources, verify_post
from .gemma_rsqrt_lookup import CheckedRsqrtLookup, _rms_lookup_row, _runtime
from .gemma_ir_interpreter import bind_model_tensors, _projection_arithmetic_commitment
from .gemma_wmma_candidate import operand_aligned_product_bits, OPERAND_ALIGNMENT_PROFILE
from .reference_gemma import _observe_rms_module, _rms_code_commitment
from .gemma_rotary_slice import _sha, _seal, _check_hash
from .serialization import canonical_json

SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
NORM_WEIGHT = "model.layers.0.pre_feedforward_layernorm.weight"
ROLES = ("gate", "up")
SCOPE = "Fixed-input pre-feedforward lookup RMS and independent K640 serial operand-aligned gate/up predictions from verified residual boundaries; original gate activation executes but is not qualified; stops before activation/up multiplication and down projection."
_WORKER_WEIGHTS: dict[str, list[list[int]]] = {}


def _code_sha() -> str:
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("MLP-entry source changed after import")
    return _sha({"module": SOURCE_SHA256, "projection": _projection_arithmetic_commitment(), "rms": _rms_code_commitment()})


def _worker_init(weights: dict[str, list[list[int]]], code: str) -> None:
    global _WORKER_WEIGHTS
    if _code_sha() != code:
        raise ValueError("MLP prediction worker source mismatch")
    _WORKER_WEIGHTS = weights


def _worker_row(task: tuple[str, int, list[int]]) -> tuple[str, int, list[int]]:
    role, index, values = task
    return role, index, [operand_aligned_product_bits(values, weight) for weight in _WORKER_WEIGHTS[role]]


def project_mlp_bits(normalized: np.ndarray, weights: dict[str, np.ndarray], workers: int = 1) -> dict[str, np.ndarray]:
    if type(workers) is not int or not 1 <= workers <= 4 or normalized.dtype != np.uint16 or normalized.shape != (1, 30, 640) or set(weights) != set(ROLES) or any(value.dtype != np.uint16 or value.shape != (2048, 640) for value in weights.values()):
        raise ValueError("Unsupported MLP-entry projection geometry or worker count")
    weight_rows = {role: weights[role].tolist() for role in ROLES}
    tasks = [(role, index, row) for role in ROLES for index, row in enumerate(normalized[0].tolist())]
    output = {role: np.empty((1, 30, 2048), dtype=np.uint16) for role in ROLES}
    if workers == 1:
        for role, index, row in tasks:
            output[role][0, index] = [operand_aligned_product_bits(row, weight) for weight in weight_rows[role]]
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init, initargs=(weight_rows, _code_sha())) as pool:
            for role, index, values in pool.map(_worker_row, tasks):
                output[role][0, index] = values
    return output


@dataclass(frozen=True)
class MlpEntrySources:
    post: PostAttentionSources
    post_plan: dict[str, Any]
    post_bundle: dict[str, Any]
    post_report: dict[str, Any]

    def input(self) -> np.ndarray:
        checked = verify_post(self.post, self.post_plan, self.post_bundle, self.post_report)
        if not checked["valid"] or not checked["post_attention_matches"]:
            raise ValueError("MLP entry requires passing residual evidence")
        lookup = self.post.lookup
        if type(lookup) is not CheckedRsqrtLookup or getattr(lookup.predict_bits, "__func__", None) is not CheckedRsqrtLookup.predict_bits:
            raise ValueError("MLP entry requires the registered rsqrt lookup")
        return _state_array(self.post_bundle["predictions"]["residual_bits"], [1, 30, 640])

    def commitments(self) -> dict[str, Any]:
        return {"program_sha256": self.post.program["program_sha256"], "post_plan_sha256": self.post_plan["plan_sha256"],
                "post_report_sha256": self.post_report["report_sha256"], "post_bundle_sha256": self.post_plan["bundle_sha256"], "lookup_evidence": self.post.lookup.evidence}


def mlp_instructions(program: dict[str, Any]) -> list[dict[str, Any]]:
    specs = [("layer.0.mlp.normalized", "RMS_NORM", ["layer.0.post_attention_residual"], [NORM_WEIGHT])]
    specs += [("layer.0.mlp." + role, "LINEAR", ["layer.0.mlp.normalized"], [f"model.layers.0.mlp.{role}_proj.weight"]) for role in ROLES]
    result = []
    for output, opcode, inputs, parameters in specs:
        found = [item for item in program["instructions"] if item["outputs"] == [output]]
        if len(found) != 1 or found[0]["opcode"] != opcode or found[0]["inputs"] != inputs or found[0]["parameter_refs"] != parameters:
            raise ValueError("Unsupported MLP-entry IR binding")
        if opcode == "LINEAR" and found[0]["attributes"]:
            raise ValueError("Unsupported MLP projection attributes")
        result.append(found[0])
    if program["configuration"]["intermediate_size"] != 2048 or set(result[0]["attributes"]) != {"epsilon"} or not 0 < result[0]["attributes"]["epsilon"] < 1:
        raise ValueError("Unsupported MLP width or normalization epsilon")
    return result


def normalize_mlp(inputs: np.ndarray, weights: list[int], epsilon: float, lookup: CheckedRsqrtLookup, runtime: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    if inputs.dtype != np.uint16 or inputs.shape != (1, 30, 640) or len(weights) != 640:
        raise ValueError("MLP normalization requires BF16 [1,30,640]")
    rows = [_rms_lookup_row(row, weights, epsilon, lookup, runtime) for row in inputs[0].tolist()]
    return np.asarray([row["output_bits"] for row in rows], dtype=np.uint16)[None, ...], {key: [row[key] for row in rows] for key in ("mean_bits", "denominator_bits", "rsqrt_bits")}


def _model_context(sources: MlpEntrySources, model: Any) -> dict[str, Any]:
    import torch
    parameters = _model_parameters(sources.post.program, model)
    layer = model.model.layers[0]
    instruction = mlp_instructions(sources.post.program)[0]
    norm = layer.pre_feedforward_layernorm
    if norm.weight is not parameters[NORM_WEIGHT] or norm.eps != instruction["attributes"]["epsilon"] or list(norm.weight.shape) != [640]:
        raise ValueError("MLP input norm is not bound to the IR")
    for role in ROLES:
        module = getattr(layer.mlp, role + "_proj")
        name = f"model.layers.0.mlp.{role}_proj.weight"
        if type(module) is not torch.nn.Linear or module.bias is not None or module.weight is not parameters[name] or module.weight.shape != (2048, 640) or module.weight.stride() != (640, 1):
            raise ValueError("Unsupported original MLP projection")
    selected = [NORM_WEIGHT, *(f"model.layers.0.mlp.{role}_proj.weight" for role in ROLES)]
    if any(parameters[name].dtype != torch.bfloat16 or parameters[name].device != model.model.embed_tokens.weight.device for name in selected) or not isinstance(layer.mlp.act_fn, torch.nn.Module):
        raise ValueError("Unsupported MLP parameter/activation binding")
    binding = {"class": type(model).__module__ + "." + type(model).__name__, "config_sha256": _sha(model.config.to_dict()), "parameter_commitments_sha256": _sha(sources.post.program["parameter_commitments"])}
    if canonical_json(binding) != canonical_json(sources.post_plan["model_binding"]) or canonical_json(_runtime()) != canonical_json(sources.post_plan["runtime"]):
        raise ValueError("MLP model/runtime differs from source evidence")
    return parameters


def build_mlp_plan(sources: MlpEntrySources, model: Any, workers: int = 1) -> tuple[dict[str, Any], dict[str, Any]]:
    code = _code_sha()
    inputs = sources.input()
    instructions = mlp_instructions(sources.post.program)
    parameters = _model_context(sources, model)
    norm_weights = _bits(parameters[NORM_WEIGHT])
    weights = {role: _bits(parameters[f"model.layers.0.mlp.{role}_proj.weight"]) for role in ROLES}
    normalized, stages = normalize_mlp(inputs, norm_weights.tolist(), instructions[0]["attributes"]["epsilon"], sources.post.lookup, sources.post_plan["runtime"])
    projections = project_mlp_bits(normalized, weights, workers)
    predictions = {"normalized": normalized, **projections}
    bundle = {"norm_weight_bits": norm_weights.tolist(), "projection_weight_bits": {role: value.tolist() for role, value in weights.items()},
              "predictions": {role: value.tolist() for role, value in predictions.items()}, "scalar_stages": stages}
    bind_model_tensors(sources.post.program, model, verify_hashes=True)
    if _code_sha() != code:
        raise ValueError("MLP-entry source changed during prediction")
    body = {"schema_version": 1, "scope": SCOPE, "sources": sources.commitments(), "instructions": instructions, "code_sha256": code,
            "runtime": sources.post_plan["runtime"], "model_binding": sources.post_plan["model_binding"], "input": _descriptor(inputs), "norm_weight": _descriptor(norm_weights),
            "projection_weights": {role: _descriptor(value) for role, value in weights.items()}, "profile": dict(OPERAND_ALIGNMENT_PROFILE),
            "predictions": {role: _descriptor(value) for role, value in predictions.items()}, "scalar_stages_sha256": _sha(stages), "bundle_sha256": _sha(bundle),
            "projection_value_count": 122880, "normalization_value_count": 19200, "scalar_stage_positions": 90, "repetitions": 3,
            "prefix_boundary_reused": True, "candidate_refitting_allowed": False, "shape_transfer_prequalified": False,
            "native_gate_activation_executes_before_up": True, "gate_activation_qualified": False, "activation_values_used_by_predictor": False,
            "product_executed": False, "down_projection_executed": False, "full_first_layer_qualified": False, "global_exactness_activation_allowed": False}
    return _seal(body, "plan_sha256"), bundle


def check_mlp_plan(sources: MlpEntrySources, plan: dict[str, Any], bundle: dict[str, Any]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    _check_hash(plan, "plan_sha256")
    inputs = sources.input()
    instructions = mlp_instructions(sources.post.program)
    if plan["scope"] != SCOPE or plan["code_sha256"] != _code_sha() or plan["bundle_sha256"] != _sha(bundle) or canonical_json(plan["sources"]) != canonical_json(sources.commitments()) or canonical_json(plan["instructions"]) != canonical_json(instructions) or canonical_json(plan["profile"]) != canonical_json(OPERAND_ALIGNMENT_PROFILE):
        raise ValueError("MLP-entry source/profile commitment mismatch")
    if canonical_json(plan["input"]) != canonical_json(_descriptor(inputs)) or canonical_json(plan["runtime"]) != canonical_json(sources.post_plan["runtime"]) or canonical_json(plan["model_binding"]) != canonical_json(sources.post_plan["model_binding"]):
        raise ValueError("MLP-entry input/model/runtime mismatch")
    false_fields = ("candidate_refitting_allowed", "shape_transfer_prequalified", "gate_activation_qualified", "activation_values_used_by_predictor", "product_executed", "down_projection_executed", "full_first_layer_qualified", "global_exactness_activation_allowed")
    if any(plan.get(key) is not False for key in false_fields) or any(plan.get(key) is not True for key in ("prefix_boundary_reused", "native_gate_activation_executes_before_up")) or any(type(plan[key]) is not int or plan[key] != value for key, value in (("projection_value_count", 122880), ("normalization_value_count", 19200), ("scalar_stage_positions", 90), ("repetitions", 3))):
        raise ValueError("MLP-entry scope overclaim")
    norm_weights = _state_array(bundle["norm_weight_bits"], [640])
    if _descriptor(norm_weights)["sha256"] != sources.post.program["parameter_commitments"][NORM_WEIGHT]["sha256"] or canonical_json(plan["norm_weight"]) != canonical_json(_descriptor(norm_weights)):
        raise ValueError("MLP norm weight commitment mismatch")
    if set(bundle["projection_weight_bits"]) != set(ROLES) or set(plan["projection_weights"]) != set(ROLES):
        raise ValueError("MLP projection weight coverage mismatch")
    for role in ROLES:
        weight = _state_array(bundle["projection_weight_bits"][role], [2048, 640])
        if _descriptor(weight)["sha256"] != sources.post.program["parameter_commitments"][f"model.layers.0.mlp.{role}_proj.weight"]["sha256"] or canonical_json(plan["projection_weights"][role]) != canonical_json(_descriptor(weight)):
            raise ValueError("MLP projection weight commitment mismatch")
    shapes = {"normalized": [1, 30, 640], "gate": [1, 30, 2048], "up": [1, 30, 2048]}
    if set(bundle["predictions"]) != set(shapes) or set(plan["predictions"]) != set(shapes):
        raise ValueError("MLP-entry prediction coverage mismatch")
    predictions = {role: _state_array(bundle["predictions"][role], shape) for role, shape in shapes.items()}
    if canonical_json(plan["predictions"]) != canonical_json({role: _descriptor(value) for role, value in predictions.items()}):
        raise ValueError("MLP-entry prediction descriptor mismatch")
    normalized, stages = normalize_mlp(inputs, norm_weights.tolist(), instructions[0]["attributes"]["epsilon"], sources.post.lookup, sources.post_plan["runtime"])
    if not np.array_equal(normalized, predictions["normalized"]) or canonical_json(stages) != canonical_json(bundle["scalar_stages"]) or plan["scalar_stages_sha256"] != _sha(stages):
        raise ValueError("MLP normalization does not reproduce independently")
    return inputs, predictions


def _capture_mlp(model: Any, ids: list[list[int]], traced: bool) -> dict[str, Any]:
    import torch
    from torch.profiler import profile, ProfilerActivity

    layer, record = model.model.layers[0], {"order": []}
    norm, mlp = layer.pre_feedforward_layernorm, layer.mlp
    norm_forward, original_linear = norm.forward, torch.nn.functional.linear
    handles = []

    class Complete(Exception):
        pass

    def observe_norm(value: Any) -> Any:
        if "scalar_stages" in record:
            raise ValueError("Repeated MLP input normalization")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as captured:
            output, stages = _observe_rms_module(norm_forward, value)
            torch.cuda.synchronize()
        record["scalar_stages"] = stages
        record["norm_cuda_events"] = sorted({event.name for event in captured.events() if event.device_type == torch.autograd.DeviceType.CUDA})
        return output

    def input_hook(module: Any, args: Any) -> None:
        if "input" in record:
            raise ValueError("Repeated MLP input boundary")
        record["input"] = _bits(args[0]).tolist()

    def output_hook(role: str):
        def capture(module: Any, args: Any, output: Any) -> None:
            if role in record:
                raise ValueError("Repeated MLP output boundary")
            record["order"].append(role)
            record[role] = _bits(output).tolist()
            if role in ROLES:
                record[role + "_input"] = _bits(args[0]).tolist()
            if role == "up":
                raise Complete()
        return capture

    def activation(module: Any, args: Any, output: Any) -> None:
        if record["order"] != ["normalized", "gate"] or canonical_json(_bits(args[0]).tolist()) != canonical_json(record["gate"]):
            raise ValueError("Unexpected native gate-activation order/input")
        record["order"].append("activation_unverified")

    def linear(input: Any, weight: Any, bias: Any = None) -> Any:
        role = next((role for role in ROLES if weight is getattr(mlp, role + "_proj").weight), None)
        if role is None:
            return original_linear(input, weight, bias)
        key = role + "_cuda_events"
        if key in record or bias is not None:
            raise ValueError("Unexpected original MLP projection occurrence/bias")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as captured:
            result = original_linear(input, weight, bias)
            torch.cuda.synchronize()
        record[key] = sorted({event.name for event in captured.events() if event.device_type == torch.autograd.DeviceType.CUDA})
        record[role + "_strides"] = {"input": list(input.stride()), "weight": list(weight.stride())}
        return result

    stopped = False
    try:
        handles.append(norm.register_forward_pre_hook(input_hook))
        handles.append(norm.register_forward_hook(output_hook("normalized")))
        handles.append(mlp.gate_proj.register_forward_hook(output_hook("gate")))
        handles.append(mlp.act_fn.register_forward_hook(activation))
        handles.append(mlp.up_proj.register_forward_hook(output_hook("up")))
        tokens = torch.tensor(ids, dtype=torch.int64, device=norm.weight.device)
        with torch.no_grad():
            if traced:
                with patch.object(norm, "forward", side_effect=observe_norm), patch.object(torch.nn.functional, "linear", side_effect=linear):
                    model(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False, logits_to_keep=1)
            else:
                model(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False, logits_to_keep=1)
    except Complete:
        stopped = True
    finally:
        for handle in handles:
            handle.remove()
    if not stopped or record["order"] != ["normalized", "gate", "activation_unverified", "up"]:
        raise ValueError("Native MLP did not stop at the declared up-projection boundary")
    return record


def mlp_report(plan: dict[str, Any], bundle: dict[str, Any], inputs: np.ndarray, predictions: dict[str, np.ndarray], observations: list[dict[str, Any]], runtime: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(observations, list) or len(observations) != 3:
        raise ValueError("Expected three MLP-entry observations")
    mismatches, checks = [], []
    geometry = {"input_shape": [1, 30, 640], "input_strides": [19200, 640, 1], "alignment_mod16": 0, "input_dtype": "torch.float32", "axes": [-1], "keepdim": True}
    for repetition, pair in enumerate(observations):
        traced, plain = pair["traced"], pair["untraced"]
        actual = {role: _state_array(traced[role], list(value.shape)) for role, value in predictions.items()}
        for key, expected in bundle["scalar_stages"].items():
            values = traced["scalar_stages"][key]
            if not isinstance(values, list) or len(values) != 30 or any(type(bits) is not int or not 0 <= bits <= 0xFFFFFFFF for bits in values):
                raise ValueError("Malformed MLP normalization scalar stage")
            for row, (wanted, observed) in enumerate(zip(expected, values)):
                if wanted != observed:
                    mismatches.append({"repetition": repetition, "stage": key, "coordinate": [0, row, 0], "predicted_bits": wanted, "observed_bits": observed})
        for role, expected in predictions.items():
            for coordinate in np.argwhere(actual[role] != expected):
                mismatches.append({"repetition": repetition, "stage": role, "coordinate": coordinate.tolist(), "predicted_bits": int(expected[tuple(coordinate)]), "observed_bits": int(actual[role][tuple(coordinate)])})
        checks.append({"source_input_matches": all(np.array_equal(_state_array(record["input"], [1, 30, 640]), inputs) for record in (traced, plain)),
                       "projection_inputs_match_normalized_prediction": all(np.array_equal(_state_array(record[role + "_input"], [1, 30, 640]), predictions["normalized"]) for record in (traced, plain) for role in ROLES),
                       "untraced_outputs_match": all(np.array_equal(_state_array(plain[role], list(value.shape)), actual[role]) for role, value in predictions.items()),
                       "mean_geometry_matches": canonical_json(traced["scalar_stages"]["mean_input_metadata"]) == canonical_json(geometry),
                       "declared_native_order_matches": all(record["order"] == ["normalized", "gate", "activation_unverified", "up"] for record in (traced, plain))})
        for role in ROLES:
            if traced[role + "_strides"] != {"input": [19200, 640, 1], "weight": [640, 1]}:
                raise ValueError("MLP projection geometry changed")
        for key in ("norm_cuda_events", "gate_cuda_events", "up_cuda_events"):
            if not isinstance(traced[key], list) or not traced[key] or any(not isinstance(event, str) or not event for event in traced[key]):
                raise ValueError("Missing native MLP-entry CUDA provenance")
    counts = {stage: [sum(item["stage"] == stage and item["repetition"] == repetition for item in mismatches) for repetition in range(3)] for stage in (*bundle["scalar_stages"], "normalized", *ROLES)}
    runtime_match = canonical_json(runtime) == canonical_json(plan["runtime"])
    first = next((key for key in checks[0] if not all(item[key] for item in checks)), None) or (mismatches[0] if mismatches else None) or (None if runtime_match else "runtime")
    return _seal({"schema_version": 1, "scope": SCOPE, "plan_sha256": plan["plan_sha256"], "runtime": runtime, "runtime_matches_plan": runtime_match,
                  "observations": observations, "checks": checks, "mismatches": mismatches, "mismatch_counts": counts, "first_divergence": first,
                  "mlp_entry_matches": not mismatches and runtime_match and all(all(item.values()) for item in checks), "projection_value_count": 122880,
                  "normalization_value_count": 19200, "scalar_stage_positions": 90, "prefix_independently_recomputed": False,
                  "native_gate_activation_executed_unverified": True, "gate_activation_qualified": False, "product_executed": False,
                  "down_projection_executed": False, "full_first_layer_qualified": False, "hardware_semantics_established": False, "global_exactness_activation_allowed": False}, "report_sha256")


def acquire_mlp(sources: MlpEntrySources, model: Any, plan: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    inputs, predictions = check_mlp_plan(sources, plan, bundle)
    _model_context(sources, model)
    ids = sources.post.survivor_context[0].softmax.scores.rotary_bundle["entry_plan"]["input_token_ids"]
    observations = [{"traced": _capture_mlp(model, ids, True), "untraced": _capture_mlp(model, ids, False)} for _ in range(3)]
    bind_model_tensors(sources.post.program, model, verify_hashes=True)
    _code_sha()
    return mlp_report(plan, bundle, inputs, predictions, observations, _runtime())


def verify_mlp(sources: MlpEntrySources, plan: dict[str, Any], bundle: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    try:
        inputs, predictions = check_mlp_plan(sources, plan, bundle)
        expected = mlp_report(plan, bundle, inputs, predictions, report["observations"], report["runtime"])
        return {"valid": canonical_json(report) == canonical_json(expected), "mode": "integrity_and_normalization_recomputation", "mlp_entry_matches": expected["mlp_entry_matches"],
                "mismatch_counts": expected["mismatch_counts"], "first_divergence": expected["first_divergence"], "global_exactness_activation_allowed": False}
    except (KeyError, ValueError, TypeError, RuntimeError, AttributeError) as error:
        return {"valid": False, "reason": str(error)}


def mlp_summary(plan: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in report.items() if key not in ("observations", "mismatches", "report_sha256")}
    body.update({"source_report_sha256": report["report_sha256"], "sources": plan["sources"], "profile": plan["profile"], "code_sha256": plan["code_sha256"],
                 "cuda_events": {role: [pair["traced"][role + "_cuda_events"] for pair in report["observations"]] for role in ("norm", *ROLES)},
                 "observed_output_hashes": {role: [_descriptor(_state_array(pair["traced"][role], plan["predictions"][role]["shape"]))["sha256"] for pair in report["observations"]] for role in plan["predictions"]}})
    return _seal(body, "summary_sha256")
