from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from . import gemma_k2048_dense as dense
from . import gemma_mlp_down as down
from . import gemma_post_attention as post
from .gemma_attention_entry import _bits, _descriptor, _state_array
from .gemma_ir import verify_gemma_ir
from .gemma_rsqrt_lookup import CheckedRsqrtLookup, _runtime
from .gemma_rotary_slice import _sha, _seal, _check_hash
from .reference_gemma import _observe_rms_module, _rms_code_commitment
from .serialization import canonical_json

SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
WEIGHT = "model.layers.0.post_feedforward_layernorm.weight"
SHAPE = [1, 30, 640]
STAGES = ("mean_bits", "denominator_bits", "rsqrt_bits")
ORDER = ["residual_base", "down", "norm_input", "normalized", "residual"]
MEAN_GEOMETRY = {"input_shape": SHAPE, "input_strides": [19200, 640, 1], "alignment_mod16": 0, "input_dtype": "torch.float32", "axes": [-1], "keepdim": True}
SCOPE = "Fixed-case post-feedforward RMS640 with registered empirical rsqrt and independent BF16 residual ADD to hidden.1 from verified reused branches. Observed ATen mean/denominator/rsqrt stages and original ADD boundaries, not all FP32 vector internals, fused registers, complete native semantics, independently connected first-layer execution, or hardware qualification. Distinct CUDA symbol sets do not establish launch order. No later layer, final normalization, or logits execution."
FALSE_FLAGS = ("qualified", "completeFirstLayerQualified", "full_first_layer_qualified", "connected_first_layer_independently_recomputed", "prefix_independently_recomputed", "native_arithmetic_reconstructed", "all_fp32_vector_internals_observed", "native_fused_registers_observed", "hardware_semantics_established", "global_exactness_activation_allowed", "qualification_promotion_allowed", "candidate_refitting_allowed", "later_layers_executed", "final_model_norm_executed", "logits_executed")


def _code_sha() -> str:
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Post-feedforward source changed after import")
    return _sha({"module": SOURCE_SHA256, "unchanged_post_attention": post._code_sha(), "unchanged_dense_k2048": dense._code_sha(), "rms_implementation": _rms_code_commitment()})


@dataclass(frozen=True)
class PostFeedforwardSources:
    dense_sources: dense.DenseSources
    dense_plan: dict[str, Any]
    dense_bundle: dict[str, Any]
    dense_report: dict[str, Any]

    @property
    def entry(self) -> Any:
        return self.dense_sources.down.product.entry

    @property
    def program(self) -> dict[str, Any]:
        return self.dense_sources.down.program

    @property
    def lookup(self) -> CheckedRsqrtLookup:
        return self.entry.post.lookup

    @property
    def runtime(self) -> dict[str, Any]:
        return self.dense_sources.model_plan["runtime"]

    def boundaries(self) -> tuple[np.ndarray, np.ndarray]:
        code = _code_sha()
        if type(self.lookup) is not CheckedRsqrtLookup or getattr(self.lookup.predict_bits, "__func__", None) is not CheckedRsqrtLookup.predict_bits:
            raise ValueError("Post-feedforward requires the registered checked rsqrt provider")
        checked = dense.verify_dense(self.dense_sources, self.dense_plan, self.dense_bundle, self.dense_report)
        if checked.get("valid") is not True or checked.get("candidate_passes_dense_holdout") is not True:
            raise ValueError("Post-feedforward requires passing dense K2048 and v2 model-down full lineage")
        post_instructions(self.program)
        if any(canonical_json(value) != canonical_json(self.runtime) for value in (self.dense_plan["runtime"], self.entry.post_plan["runtime"], self.entry.post.runtime)):
            raise ValueError("Post-feedforward branches have different runtimes")
        if canonical_json(self.dense_plan["candidate"]) != canonical_json(self.dense_sources.model_plan["candidate"]):
            raise ValueError("Post-feedforward dense/model candidate mismatch")
        if self.program["program_sha256"] != self.entry.post.program["program_sha256"] or canonical_json(self.dense_sources.model_plan["model_binding"]) != canonical_json(self.entry.post_plan["model_binding"]):
            raise ValueError("Post-feedforward branches have different program/model bindings")
        values = _state_array(self.dense_sources.model_bundle["down_bits"], SHAPE)
        residual = _state_array(self.entry.post_bundle["predictions"]["residual_bits"], SHAPE)
        if _code_sha() != code:
            raise ValueError("Post-feedforward source changed during validation")
        return values, residual

    def commitments(self) -> dict[str, Any]:
        result = {"program_sha256": self.program["program_sha256"], "lookup_evidence": self.lookup.evidence,
                  "runtime": self.runtime, "candidate": self.dense_sources.model_plan["candidate"], "dense_sources": self.dense_sources.commitments()}
        for name, plan, bundle, report in (("model_down", self.dense_sources.model_plan, self.dense_sources.model_bundle, self.dense_sources.model_report),
                                         ("post_attention", self.entry.post_plan, self.entry.post_bundle, self.entry.post_report),
                                         ("dense_k2048", self.dense_plan, self.dense_bundle, self.dense_report)):
            result[name] = {"plan_sha256": plan["plan_sha256"], "plan_payload_sha256": _sha(plan), "bundle_sha256": _sha(bundle), "report_sha256": report["report_sha256"], "report_payload_sha256": _sha(report)}
        return result


def post_instructions(program: dict[str, Any]) -> list[dict[str, Any]]:
    if not verify_gemma_ir(program)["valid"]:
        raise ValueError("Post-feedforward requires intact typed IR")
    specs = (("i0034", "layer.0.mlp.post_normalized", "RMS_NORM", ["layer.0.mlp.down"], [WEIGHT], {"epsilon": 1e-6}),
             ("i0035", "hidden.1", "ADD", ["layer.0.post_attention_residual", "layer.0.mlp.post_normalized"], [], {"output_dtype": "torch.bfloat16"}))
    result = []
    for identifier, output, opcode, inputs, parameters, attributes in specs:
        found = [node for node in program["instructions"] if node["outputs"] == [output]]
        if len(found) != 1:
            raise ValueError("Expected unique post-feedforward IR node")
        node = found[0]
        _check_hash(node, "instruction_sha256")
        if node["id"] != identifier or node["opcode"] != opcode or node["inputs"] != inputs or node["parameter_refs"] != parameters or node["attributes"] != attributes or type(node["layer"]) is not int or node["layer"] != 0:
            raise ValueError("Unsupported post-feedforward instruction binding")
        result.append(node)
    for name, producer in (("layer.0.mlp.down", "i0033"), ("layer.0.post_attention_residual", "i0027"), ("layer.0.mlp.post_normalized", "i0034"), ("hidden.1", "i0035")):
        declaration = program["tensors"][name]
        resolved = [{"B": 1, "S": 30}.get(value, value) for value in declaration["shape"]]
        if resolved != SHAPE or declaration["dtype"] != "torch.bfloat16" or declaration["producer"] != producer:
            raise ValueError("Unsupported post-feedforward shape/dtype/producer")
    weight = program["parameter_commitments"][WEIGHT]
    if weight["shape"] != [640] or weight["dtype"] != "torch.bfloat16" or weight["numel"] != 640 or program["configuration"]["rms_norm_epsilon"] != 1e-6 or program["configuration"]["hidden_size"] != 640:
        raise ValueError("Unsupported post-feedforward weight/configuration")
    return result


def predict_post_feedforward(values: np.ndarray, residual: np.ndarray, weights: list[int], epsilon: float, lookup: CheckedRsqrtLookup, runtime: dict[str, Any]) -> dict[str, Any]:
    if type(epsilon) not in (float, int) or epsilon != 1e-6:
        raise ValueError("Post-feedforward epsilon must be 1e-6")
    return post.predict_post_attention(values, residual, weights, epsilon, lookup, runtime)


def _model_context(sources: PostFeedforwardSources, model: Any) -> Any:
    import torch

    parameters = down._model_context(sources.dense_sources.down, model)
    norm = model.model.layers[0].post_feedforward_layernorm
    weight = parameters[WEIGHT]
    post_instructions(sources.program)
    binding = {"class": type(model).__module__ + "." + type(model).__name__, "config_sha256": _sha(model.config.to_dict()), "parameter_commitments_sha256": _sha(sources.program["parameter_commitments"])}
    if norm.weight is not weight or weight.dtype != torch.bfloat16 or list(weight.shape) != [640] or weight.stride() != (1,) or weight.device != model.model.embed_tokens.weight.device or norm.eps != 1e-6 or model.config.rms_norm_eps != 1e-6:
        raise ValueError("Post-feedforward RMS weight/epsilon identity mismatch")
    if canonical_json(binding) != canonical_json(sources.dense_sources.model_plan["model_binding"]) or canonical_json(_runtime()) != canonical_json(sources.runtime):
        raise ValueError("Post-feedforward model/runtime differs from frozen source")
    return weight


def _plan_body(sources: PostFeedforwardSources, values: np.ndarray, residual: np.ndarray, weights: np.ndarray, bundle: dict[str, Any]) -> dict[str, Any]:
    prediction = bundle["predictions"]
    symbols = [down._kernel_symbols(pair["traced"]["cuda_events"]) for pair in sources.entry.post_report["observations"]]
    if len(symbols) != 3 or any(item != symbols[0] for item in symbols):
        raise ValueError("Post-attention RMS provenance is not repeat stable")
    return {"schema_version": 1, "scope": SCOPE, "sources": sources.commitments(), "instructions": post_instructions(sources.program),
            "code_sha256": _code_sha(), "implementation_commitment": down._implementation_commitment(), "runtime": sources.runtime,
            "model_binding": sources.dense_sources.model_plan["model_binding"], "input": _descriptor(values), "residual_base": _descriptor(residual), "weight": _descriptor(weights),
            "predictions": {name: _descriptor(_state_array(prediction[name + "_bits"], SHAPE)) for name in ("normalized", "residual")},
            "scalar_stages_sha256": _sha(prediction["stages"]), "bundle_sha256": _sha(bundle), "expected_rms_symbols": symbols[0],
            "kernel_provenance": down.KERNEL_PROVENANCE, "value_count": 38400, "normalized_value_count": 19200, "residual_value_count": 19200, "scalar_stage_positions": 90,
            "repetitions": 3, "original_forward_count": 6, "prefix_boundary_reused": True, "epoch_scope": "fixed_source_case_only", **{key: False for key in FALSE_FLAGS}}


def build_post_plan(sources: PostFeedforwardSources, model: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    code = _code_sha()
    values, residual = sources.boundaries()
    commitments = canonical_json(sources.commitments())
    weights = _bits(_model_context(sources, model))
    prediction = predict_post_feedforward(values, residual, weights.tolist(), 1e-6, sources.lookup, sources.runtime)
    bundle = {"weight_bits": weights.tolist(), "predictions": prediction}
    plan = _seal(_plan_body(sources, values, residual, weights, bundle), "plan_sha256")
    _model_context(sources, model)
    if _code_sha() != code or canonical_json(sources.commitments()) != commitments:
        raise ValueError("Post-feedforward source changed during prediction")
    return plan, bundle


def check_post_plan(sources: PostFeedforwardSources, plan: dict[str, Any], bundle: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    code = _code_sha()
    _check_hash(plan, "plan_sha256")
    values, residual = sources.boundaries()
    commitments = canonical_json(sources.commitments())
    weights = _state_array(bundle["weight_bits"], [640])
    if _descriptor(weights)["sha256"] != sources.program["parameter_commitments"][WEIGHT]["sha256"]:
        raise ValueError("Post-feedforward checkpoint weight commitment mismatch")
    prediction = predict_post_feedforward(values, residual, weights.tolist(), 1e-6, sources.lookup, sources.runtime)
    expected_bundle = {"weight_bits": weights.tolist(), "predictions": prediction}
    expected_plan = _seal(_plan_body(sources, values, residual, weights, expected_bundle), "plan_sha256")
    if canonical_json(bundle) != canonical_json(expected_bundle) or canonical_json(plan) != canonical_json(expected_plan):
        raise ValueError("Post-feedforward plan/bundle does not reproduce independently; no refit")
    if _code_sha() != code or canonical_json(sources.commitments()) != commitments:
        raise ValueError("Post-feedforward source changed during plan validation")
    return values, residual, prediction


def _tensor_record(value: Any) -> dict[str, Any]:
    import torch

    with torch._C._DisableTorchDispatch():
        if value.dtype != torch.bfloat16 or list(value.shape) != SHAPE or list(value.stride()) != [19200, 640, 1]:
            raise ValueError("Unexpected post-feedforward BF16 boundary geometry")
        return {"bits": _bits(value).tolist(), "geometry": dense._snapshot(value)}


def _same_tensor(value: Any, original: Any, recorded: dict[str, Any]) -> bool:
    import torch

    with torch._C._DisableTorchDispatch():
        return value is original and value.data_ptr() == original.data_ptr() and canonical_json(_tensor_record(value)) == canonical_json(recorded)


def _check_add_operands(args: Any, kwargs: Any, tensors: dict[str, Any], record: dict[str, Any]) -> None:
    if "add" in record or len(args) != 2 or set(kwargs) - {"alpha"} or type(kwargs.get("alpha", 1)) not in (int, float) or kwargs.get("alpha", 1) != 1:
        raise ValueError("Unexpected final ADD occurrence/signature/alpha")
    if not _same_tensor(args[0], tensors["residual_base"], record["boundaries"]["residual_base"]) or not _same_tensor(args[1], tensors["normalized"], record["boundaries"]["normalized"]):
        raise ValueError("Final ADD operands do not have original ordered branch lineage")


def _capture_post(model: Any, token_ids: list[list[int]], traced: bool) -> dict[str, Any]:
    import torch
    from torch.profiler import profile, ProfilerActivity
    from torch.utils._python_dispatch import TorchDispatchMode

    layer = model.model.layers[0]
    norm = layer.post_feedforward_layernorm
    original_norm, original_layer = norm.forward, layer.forward
    record, tensors, handles = {"order": [], "boundaries": {}}, {}, []
    code, runtime = _code_sha(), _runtime()
    with torch._C._DisableTorchDispatch():
        weight_before = dense._snapshot(norm.weight)
        weight_identity = (id(norm.weight), norm.weight.data_ptr(), norm.weight._version)

    class Complete(Exception):
        pass

    def capture(name: str, value: Any) -> None:
        if name in tensors or ORDER[len(record["order"])] != name:
            raise ValueError("Unexpected post-feedforward boundary order/occurrence")
        record["boundaries"][name] = _tensor_record(value)
        tensors[name] = value
        record["order"].append(name)

    def residual_input(module: Any, args: Any) -> None:
        capture("residual_base", args[0])

    def down_output(module: Any, args: Any, output: Any) -> None:
        capture("down", output)

    def norm_input(module: Any, args: Any) -> None:
        if not _same_tensor(args[0], tensors["down"], record["boundaries"]["down"]):
            raise ValueError("Post-feedforward norm input is not original down output")
        capture("norm_input", args[0])

    def norm_output(module: Any, args: Any, output: Any) -> None:
        capture("normalized", output)

    def observe(value: Any) -> Any:
        if "stages" in record or not _same_tensor(value, tensors["norm_input"], record["boundaries"]["norm_input"]):
            raise ValueError("Repeated or unbound post-feedforward RMS observer")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiled:
            output, stages = _observe_rms_module(original_norm, value)
            torch.cuda.synchronize()
        record["stages"] = stages
        record["cuda_events"] = down._kernel_symbols(sorted({event.name for event in profiled.events() if event.device_type == torch.autograd.DeviceType.CUDA}))
        return output

    class CaptureAdd(TorchDispatchMode):
        def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
            kwargs = kwargs or {}
            if func is not torch.ops.aten.add.Tensor or "normalized" not in tensors:
                return func(*args, **kwargs)
            _check_add_operands(args, kwargs, tensors, record)
            before = [_tensor_record(value) for value in args]
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiled:
                output = func(*args, **kwargs)
                torch.cuda.synchronize()
            symbols = down._kernel_symbols(sorted({event.name for event in profiled.events() if event.device_type == torch.autograd.DeviceType.CUDA}))
            after = [_tensor_record(value) for value in args]
            tensors["add_output"] = output
            record["add"] = {"operator": "aten.add.Tensor", "alpha": kwargs.get("alpha", 1), "operand_roles": ["residual_base", "normalized"],
                             "before": before, "after": after, "output": _tensor_record(output), "kernel_names": symbols,
                             "ordered_operand_identity_matches": True, "operands_unchanged": canonical_json(before) == canonical_json(after)}
            return output

    def wrapped_layer(*args: Any, **kwargs: Any) -> Any:
        with CaptureAdd():
            return original_layer(*args, **kwargs)

    def layer_output(module: Any, args: Any, output: Any) -> None:
        if type(output) is not tuple or len(output) != 1:
            raise ValueError("Original decoder must return a one-element hidden-state tuple")
        if traced and ("add" not in record or not _same_tensor(output[0], tensors["add_output"], record["add"]["output"])):
            raise ValueError("Original final ADD did not reach decoder tuple output")
        capture("residual", output[0])
        if traced:
            record["add"]["reaches_layer_output"] = True
        raise Complete()

    def forbidden(module: Any, args: Any) -> None:
        raise ValueError("Later layer or final model norm must not execute")

    stopped = False
    try:
        handles.append(layer.pre_feedforward_layernorm.register_forward_pre_hook(residual_input))
        handles.append(layer.mlp.down_proj.register_forward_hook(down_output))
        handles.append(norm.register_forward_pre_hook(norm_input))
        handles.append(norm.register_forward_hook(norm_output))
        handles.append(layer.register_forward_hook(layer_output))
        handles.append(model.model.layers[1].register_forward_pre_hook(forbidden))
        handles.append(model.model.norm.register_forward_pre_hook(forbidden))
        ids = torch.tensor(token_ids, dtype=torch.int64, device=norm.weight.device)
        with torch.no_grad():
            if traced:
                with patch.object(norm, "forward", side_effect=observe), patch.object(layer, "forward", side_effect=wrapped_layer):
                    model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, output_attentions=False, logits_to_keep=1)
            else:
                model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, output_attentions=False, logits_to_keep=1)
    except Complete:
        stopped = True
    finally:
        for handle in handles:
            handle.remove()
    if not stopped or record["order"] != ORDER or set(record) != ({"order", "boundaries", "stages", "cuda_events", "add"} if traced else {"order", "boundaries"}):
        raise ValueError("Original forward did not stop at hidden.1")
    with torch._C._DisableTorchDispatch():
        record["weight_before"], record["weight_after"] = weight_before, dense._snapshot(norm.weight)
        record["weight_identity_unchanged"] = weight_identity == (id(norm.weight), norm.weight.data_ptr(), norm.weight._version)
    record.update({"code_before": code, "code_after": _code_sha(), "runtime_before": runtime, "runtime_after": _runtime(), "stopped_at_decoder_tuple": True})
    return record


def _boundary_matches(record: dict[str, Any], expected: np.ndarray) -> bool:
    return np.array_equal(_state_array(record["bits"], SHAPE), expected) and dense._snapshot_matches(record["geometry"], _descriptor(expected), "output")


def post_report(plan: dict[str, Any], values: np.ndarray, residual: np.ndarray, prediction: dict[str, Any], observations: list[dict[str, Any]], runtime: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(observations, list) or len(observations) != 3:
        raise ValueError("Expected three original post-feedforward traced/untraced pairs")
    outputs = {name: _state_array(prediction[name + "_bits"], SHAPE) for name in ("normalized", "residual")}
    mismatches, checks, add_symbols = [], [], []
    for repetition, pair in enumerate(observations):
        traced, plain = pair["traced"], pair["untraced"]
        current = {}
        for mode, record in (("traced", traced), ("untraced", plain)):
            boundaries = record["boundaries"]
            if set(boundaries) != set(ORDER):
                raise ValueError("Incomplete post-feedforward boundary coverage")
            current[mode + "_source_inputs_match"] = all(_boundary_matches(boundaries[name], expected) for name, expected in (("down", values), ("norm_input", values), ("residual_base", residual)))
            current[mode + "_boundary_order_matches"] = record["order"] == ORDER and record["stopped_at_decoder_tuple"] is True
            current[mode + "_code_matches"] = record["code_before"] == record["code_after"] == plan["code_sha256"]
            current[mode + "_runtime_matches"] = all(canonical_json(record[key]) == canonical_json(plan["runtime"]) for key in ("runtime_before", "runtime_after"))
            current[mode + "_weight_binding_matches"] = record["weight_identity_unchanged"] is True and all(down._geometry_matches(record[key], (640,), [1]) and record[key].get("contiguous") is True and canonical_json(record[key]["tensor"]) == canonical_json(plan["weight"]) for key in ("weight_before", "weight_after"))
            devices = {item["geometry"]["device"] for item in boundaries.values()} | {record[key]["device"] for key in ("weight_before", "weight_after")}
            current[mode + "_same_cuda_device"] = len(devices) == 1
            for name, expected in outputs.items():
                actual = _state_array(boundaries[name]["bits"], SHAPE)
                current[mode + "_" + name + "_geometry_matches"] = dense._snapshot_matches(boundaries[name]["geometry"], _descriptor(actual), "output")
                for coordinate in np.argwhere(actual != expected):
                    index = tuple(coordinate)
                    mismatches.append({"repetition": repetition, "mode": mode, "stage": name, "coordinate": coordinate.tolist(), "predicted_bits": int(expected[index]), "observed_bits": int(actual[index])})
        current["untraced_matches_traced"] = canonical_json(plain["boundaries"]) == canonical_json(traced["boundaries"])
        if any(key in plain for key in ("stages", "cuda_events", "add")):
            raise ValueError("Plain control must not claim observer/profile/dispatch evidence")
        stages = traced["stages"]
        if set(stages) != {*STAGES, "mean_input_metadata"}:
            raise ValueError("Unexpected RMS stage coverage")
        for name in STAGES:
            actual = stages[name]
            if not isinstance(actual, list) or len(actual) != 30 or any(type(value) is not int or not 0 <= value <= 0xFFFFFFFF for value in actual):
                raise ValueError("Malformed post-feedforward FP32 scalar stage")
            for row, (wanted, observed) in enumerate(zip(prediction["stages"][name], actual)):
                if wanted != observed:
                    mismatches.append({"repetition": repetition, "mode": "traced", "stage": name, "coordinate": [0, row, 0], "predicted_bits": wanted, "observed_bits": observed})
        current["mean_geometry_matches"] = canonical_json(stages["mean_input_metadata"]) == canonical_json(MEAN_GEOMETRY)
        current["rms_symbols_match"] = down._kernel_symbols(traced["cuda_events"]) == down._kernel_symbols(plan["expected_rms_symbols"])
        add = traced["add"]
        add_symbols.append(down._kernel_symbols(add["kernel_names"]))
        current["native_add_signature_matches"] = add["operator"] == "aten.add.Tensor" and type(add["alpha"]) in (int, float) and add["alpha"] == 1 and add["operand_roles"] == ["residual_base", "normalized"]
        expected_operands = [traced["boundaries"][name] for name in ("residual_base", "normalized")]
        current["native_add_operand_lineage_matches"] = add["ordered_operand_identity_matches"] is True and add["operands_unchanged"] is True and all(canonical_json(add[key]) == canonical_json(expected_operands) for key in ("before", "after"))
        current["native_add_reaches_layer_output"] = add["reaches_layer_output"] is True and canonical_json(add["output"]) == canonical_json(traced["boundaries"]["residual"])
        checks.append(current)
    symbol_stability = all(item == add_symbols[0] for item in add_symbols)
    stage_order = {name: index for index, name in enumerate((*STAGES, "normalized", "residual"))}
    mismatches.sort(key=lambda item: (item["repetition"], stage_order[item["stage"]], item["mode"], item["coordinate"]))
    counts = {name: [sum(item["repetition"] == repetition and item["stage"] == name and item["mode"] == "traced" for item in mismatches) for repetition in range(3)] for name in stage_order}
    plain_counts = {name: [sum(item["repetition"] == repetition and item["stage"] == name and item["mode"] == "untraced" for item in mismatches) for repetition in range(3)] for name in outputs}
    runtime_match = canonical_json(runtime) == canonical_json(plan["runtime"])
    first = next((key for key in checks[0] if not all(item[key] for item in checks)), None) or (mismatches[0] if mismatches else None) or (None if runtime_match and symbol_stability else "runtime_or_add_symbol_stability")
    passed = not mismatches and runtime_match and symbol_stability and all(all(item.values()) for item in checks)
    return _seal({"schema_version": 1, "scope": SCOPE, "plan_sha256": plan["plan_sha256"], "runtime": runtime, "runtime_matches_plan": runtime_match,
                  "observations": observations, "checks": checks, "mismatches": mismatches, "mismatch_counts": counts, "untraced_mismatch_counts": plain_counts, "first_divergence": first,
                  "post_feedforward_matches": passed, "status": "passed" if passed else "failed", "native_add_symbols_repeat_stable": symbol_stability,
                  "kernel_provenance": down.KERNEL_PROVENANCE, "value_count": 38400, "normalized_value_count": 19200, "residual_value_count": 19200, "scalar_stage_positions": 90,
                  "original_forward_count": 6, "prefix_boundary_reused": True, "epoch_scope": "fixed_source_case_only", **{key: False for key in FALSE_FLAGS}}, "report_sha256")


def acquire_post(sources: PostFeedforwardSources, model: Any, plan: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    code = _code_sha()
    values, residual, prediction = check_post_plan(sources, plan, bundle)
    commitments = canonical_json(sources.commitments())
    _model_context(sources, model)
    ids = sources.entry.post.survivor_context[0].softmax.scores.rotary_bundle["entry_plan"]["input_token_ids"]
    observations = [{"traced": _capture_post(model, ids, True), "untraced": _capture_post(model, ids, False)} for _ in range(3)]
    _model_context(sources, model)
    if _code_sha() != code or canonical_json(sources.commitments()) != commitments:
        raise ValueError("Post-feedforward source changed during acquisition")
    return post_report(plan, values, residual, prediction, observations, _runtime())


def verify_post(sources: PostFeedforwardSources, plan: dict[str, Any], bundle: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    try:
        code = _code_sha()
        values, residual, prediction = check_post_plan(sources, plan, bundle)
        _check_hash(report, "report_sha256")
        expected = post_report(plan, values, residual, prediction, report["observations"], report["runtime"])
        return {"valid": canonical_json(report) == canonical_json(expected) and _code_sha() == code, "mode": "independent_post_feedforward_rms_add_recomputation_and_full_source_integrity_no_prior_down_dot_recomputation",
                "post_feedforward_matches": expected["post_feedforward_matches"], "mismatch_counts": expected["mismatch_counts"], "first_divergence": expected["first_divergence"],
                "prefix_boundary_reused": True, **{key: False for key in FALSE_FLAGS}}
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError, IndexError, StopIteration) as error:
        return {"valid": False, "reason": str(error)}


def post_summary(plan: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in report.items() if key not in ("observations", "mismatches", "report_sha256")}
    body.update({"source_report_sha256": report["report_sha256"], "sources": plan["sources"], "code_sha256": plan["code_sha256"],
                 "rms_cuda_symbols": [down._kernel_symbols(pair["traced"]["cuda_events"]) for pair in report["observations"]],
                 "add_cuda_symbols": [down._kernel_symbols(pair["traced"]["add"]["kernel_names"]) for pair in report["observations"]],
                 "observed_output_hashes": {name: [_descriptor(_state_array(pair["traced"]["boundaries"][name]["bits"], SHAPE))["sha256"] for pair in report["observations"]] for name in ("normalized", "residual")}})
    return _seal(body, "summary_sha256")
