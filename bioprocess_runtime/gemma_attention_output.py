from __future__ import annotations

import copy
import hashlib
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from .gemma_attention_entry import _bits, _descriptor, _state_array, _model_parameters
from .gemma_exp_lookup import CheckedExpLookup
from .gemma_softmax_slice import SoftmaxSources
from .gemma_softmax_lookup import verify_lookup_softmax
from .gemma_rotary_slice import _sha, _seal, _check_hash
from .gemma_ir_interpreter import _projection_arithmetic_commitment, bind_model_tensors
from .gemma_wmma_candidate import OPERAND_ALIGNMENT_PROFILE, operand_aligned_product_bits
from .gemma_rsqrt_lookup import _runtime
from .operational_semantics import tensor_descriptor
from .serialization import canonical_json

SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
WEIGHT = "model.layers.0.self_attn.o_proj.weight"
SCOPE = "Frozen fixed-input value aggregation and attention output projection on verified serialized probability/V boundaries; K30-to-K32 zero-padding and K1024 dot candidates, not prequalified shape transfer or complete-layer equivalence."


def _code_sha() -> str:
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Attention-output source changed after process import")
    return _sha({"module": SOURCE_SHA256, "arithmetic": _projection_arithmetic_commitment()})


@dataclass(frozen=True)
class OutputSources:
    softmax: SoftmaxSources
    original_plan: dict[str, Any]
    original_bundle: dict[str, Any]
    original_report: dict[str, Any]
    lookup: CheckedExpLookup
    lookup_plan: dict[str, Any]
    lookup_bundle: dict[str, Any]
    lookup_report: dict[str, Any]

    @property
    def program(self) -> dict[str, Any]:
        return self.softmax.scores.program

    def inputs(self) -> tuple[np.ndarray, np.ndarray]:
        checked = verify_lookup_softmax(self.softmax, self.original_plan, self.original_bundle, self.original_report, self.lookup, self.lookup_plan, self.lookup_bundle, self.lookup_report)
        if not checked["valid"] or not checked["lookup_softmax_passes"]:
            raise ValueError("Attention output requires passing lookup-softmax evidence")
        probabilities = _state_array([row["output_bf16_bits"] for row in self.lookup_bundle["rows"]], [120, 30]).reshape(1, 4, 30, 30)
        values = self.softmax.scores.states()["layer.0.value.heads"]
        return probabilities, values

    def commitments(self) -> dict[str, Any]:
        return {"program_sha256": self.program["program_sha256"], "lookup_softmax_plan_sha256": self.lookup_plan["plan_sha256"],
                "lookup_softmax_report_sha256": self.lookup_report["report_sha256"], "lookup_softmax_bundle_sha256": self.lookup_plan["bundle_sha256"],
                "upstream": self.softmax.commitments()}


def output_instructions(program: dict[str, Any]) -> list[dict[str, Any]]:
    specs = (
        ("layer.0.value.repeated", "REPEAT_KV", ["layer.0.value.heads"], [], {"repetitions": 4}),
        ("layer.0.attention.head_output", "MATMUL_AV", ["layer.0.attention.probability", "layer.0.value.repeated"], [], {}),
        ("layer.0.attention.concatenated", "TRANSPOSE_RESHAPE_HEADS", ["layer.0.attention.head_output"], [], {"heads": 4, "head_dimension": 256}),
        ("layer.0.attention.projected", "LINEAR", ["layer.0.attention.concatenated"], [WEIGHT], {}),
    )
    result = []
    for output, opcode, inputs, parameters, attributes in specs:
        found = [item for item in program["instructions"] if item["outputs"] == [output]]
        if len(found) != 1 or found[0]["opcode"] != opcode or found[0]["inputs"] != inputs or found[0]["parameter_refs"] != parameters or found[0]["attributes"] != attributes:
            raise ValueError("Unsupported attention-output IR binding")
        result.append(found[0])
    return result


def predict_value_aggregation(probabilities: np.ndarray, values: np.ndarray) -> np.ndarray:
    if probabilities.dtype != np.uint16 or values.dtype != np.uint16 or probabilities.shape != (1, 4, 30, 30) or values.shape != (1, 1, 30, 256):
        raise ValueError("Unsupported value-aggregation geometry")
    columns = [column + [0, 0] for column in values[0, 0].T.tolist()]
    return np.asarray([[[[operand_aligned_product_bits(row + [0, 0], column) for column in columns] for row in probabilities[0, head].tolist()] for head in range(4)]], dtype=np.uint16)


def concatenate_heads(values: np.ndarray) -> np.ndarray:
    if values.dtype != np.uint16 or values.shape != (1, 4, 30, 256):
        raise ValueError("Unsupported attention head-coordinate mapping")
    return values.transpose(0, 2, 1, 3).reshape(1, 30, 1024)


def predict_output_projection(inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if inputs.dtype != np.uint16 or weights.dtype != np.uint16 or inputs.shape != (1, 30, 1024) or weights.shape != (640, 1024):
        raise ValueError("Unsupported output-projection geometry")
    rows = weights.tolist()
    return np.asarray([[[operand_aligned_product_bits(row, weight) for weight in rows] for row in inputs[0].tolist()]], dtype=np.uint16)


def _profile() -> dict[str, Any]:
    return {"dot": dict(OPERAND_ALIGNMENT_PROFILE), "aggregation_reduction_length": 30, "aggregation_candidate_length": 32,
            "aggregation_padding": "two appended positive-zero operands on each side; experimental, not asserted hardware padding",
            "projection_reduction_length": 1024, "coordinate_mapping": "transpose(0,2,1,3), reshape(1,30,1024)"}


def _model_context(sources: OutputSources, model: Any) -> tuple[Any, dict[str, Any]]:
    import torch

    parameters = _model_parameters(sources.program, model)
    module = model.model.layers[0].self_attn.o_proj
    weight = parameters[WEIGHT]
    if type(module) is not torch.nn.Linear or module.bias is not None or module.weight is not weight or weight.dtype != torch.bfloat16 or list(weight.shape) != [640, 1024] or weight.stride() != (1024, 1) or weight.device != model.model.embed_tokens.weight.device:
        raise ValueError("Original output projection does not match the bound model")
    binding = {"class": type(model).__module__ + "." + type(model).__name__, "config_sha256": _sha(model.config.to_dict()), "parameter_commitments_sha256": _sha(sources.program["parameter_commitments"])}
    expected = sources.softmax.scores.rotary_bundle["entry_plan"]["model_binding"]
    if canonical_json(binding) != canonical_json(expected) or canonical_json(_runtime()) != canonical_json(sources.lookup_plan["runtime"]):
        raise ValueError("Original output model/runtime differs from source evidence")
    return weight, binding


def build_output_plan(sources: OutputSources, model: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    code = _code_sha()
    probabilities, values = sources.inputs()
    instructions = output_instructions(sources.program)
    weight, binding = _model_context(sources, model)
    weight_bits = _bits(weight)
    head_output = predict_value_aggregation(probabilities, values)
    concatenated = concatenate_heads(head_output)
    projected = predict_output_projection(concatenated, weight_bits)
    predictions = {"head_output": head_output, "concatenated": concatenated, "projected": projected}
    bundle = {"weight_bits": weight_bits.tolist(), "predictions": {name: value.tolist() for name, value in predictions.items()}}
    bind_model_tensors(sources.program, model, verify_hashes=True)
    if _code_sha() != code:
        raise ValueError("Attention-output source changed during prediction")
    body = {"schema_version": 1, "scope": SCOPE, "sources": sources.commitments(), "instructions": instructions,
            "profile": _profile(), "model_binding": binding, "runtime": copy.deepcopy(sources.lookup_plan["runtime"]),
            "inputs": {"probability": _descriptor(probabilities), "value": _descriptor(values)}, "weight": _descriptor(weight_bits),
            "predictions": {name: _descriptor(value) for name, value in predictions.items()}, "bundle_sha256": _sha(bundle), "code_sha256": code,
            "aggregation_value_count": 30720, "projection_value_count": 19200, "repetitions": 3,
            "prefix_boundary_reused": True, "candidate_refitting_allowed": False, "shape_transfer_prequalified": False,
            "post_attention_norm_executed": False, "full_first_layer_qualified": False,
            "hardware_semantics_established": False, "global_exactness_activation_allowed": False}
    return _seal(body, "plan_sha256"), bundle


def check_output_plan(sources: OutputSources, plan: dict[str, Any], bundle: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    _check_hash(plan, "plan_sha256")
    probabilities, values = sources.inputs()
    if plan["code_sha256"] != _code_sha() or plan["bundle_sha256"] != _sha(bundle) or canonical_json(plan["sources"]) != canonical_json(sources.commitments()):
        raise ValueError("Attention-output source/bundle commitment mismatch")
    if plan["scope"] != SCOPE or canonical_json(plan["profile"]) != canonical_json(_profile()) or canonical_json(plan["instructions"]) != canonical_json(output_instructions(sources.program)) or canonical_json(plan["runtime"]) != canonical_json(sources.lookup_plan["runtime"]):
        raise ValueError("Attention-output profile/runtime mismatch")
    if canonical_json(plan["model_binding"]) != canonical_json(sources.softmax.scores.rotary_bundle["entry_plan"]["model_binding"]):
        raise ValueError("Output model binding mismatch")
    false_fields = ("candidate_refitting_allowed", "shape_transfer_prequalified", "post_attention_norm_executed", "full_first_layer_qualified", "hardware_semantics_established", "global_exactness_activation_allowed")
    if any(plan.get(key) is not False for key in false_fields) or plan["prefix_boundary_reused"] is not True or any(type(plan[key]) is not int or plan[key] != value for key, value in (("aggregation_value_count", 30720), ("projection_value_count", 19200), ("repetitions", 3))):
        raise ValueError("Attention-output scope overclaim")
    if canonical_json(plan["inputs"]) != canonical_json({"probability": _descriptor(probabilities), "value": _descriptor(values)}):
        raise ValueError("Output inputs disagree with verified source boundaries")
    weights = _state_array(bundle["weight_bits"], [640, 1024])
    if canonical_json(_descriptor(weights)) != canonical_json(plan["weight"]) or plan["weight"]["sha256"] != sources.program["parameter_commitments"][WEIGHT]["sha256"]:
        raise ValueError("Output weight bytes do not match the checkpoint")
    shapes = {"head_output": [1, 4, 30, 256], "concatenated": [1, 30, 1024], "projected": [1, 30, 640]}
    if set(bundle["predictions"]) != set(shapes) or set(plan["predictions"]) != set(shapes):
        raise ValueError("Attention-output prediction coverage mismatch")
    predicted = {name: _state_array(bundle["predictions"][name], shape) for name, shape in shapes.items()}
    if canonical_json(plan["predictions"]) != canonical_json({name: _descriptor(value) for name, value in predicted.items()}) or not np.array_equal(concatenate_heads(predicted["head_output"]), predicted["concatenated"]):
        raise ValueError("Output prediction descriptors or coordinate mapping disagree")
    return probabilities, values, predicted


def _tensor_bits(value: Any) -> list[Any]:
    import torch
    with torch._C._DisableTorchDispatch():
        return _bits(value).tolist()


def _geometry(value: Any) -> dict[str, Any]:
    import torch
    with torch._C._DisableTorchDispatch():
        return {"tensor": tensor_descriptor(value), "strides": list(value.stride())}


def _capture_forward(sources: OutputSources, model: Any, traced: bool) -> dict[str, Any]:
    import torch
    from torch.profiler import profile, ProfilerActivity
    from torch.utils._python_dispatch import TorchDispatchMode

    target = model.model.layers[0].self_attn
    namespace = importlib.import_module(type(target).__module__)
    original_attention, original_linear = namespace.eager_attention_forward, torch.nn.functional.linear
    record = {}

    class Complete(Exception):
        pass

    def profiled(call: Any) -> tuple[Any, list[str]]:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as capture:
            result = call()
            torch.cuda.synchronize()
        names = sorted({event.name for event in capture.events() if event.device_type == torch.autograd.DeviceType.CUDA})
        if not names:
            raise ValueError("No CUDA operation recorded at the declared output boundary")
        return result, names

    class Capture(TorchDispatchMode):
        def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
            if str(func) == "aten.bmm.default" and [list(arg.shape) for arg in args[:2]] == [[4, 30, 30], [4, 30, 256]]:
                if "head_output" in record:
                    raise ValueError("Repeated value-aggregation occurrence")
                record["pv_operands"] = {"probability": _tensor_bits(args[0]), "value": _tensor_bits(args[1])}
                record["pv_geometry"] = [_geometry(arg) for arg in args[:2]]
                output, names = profiled(lambda: func(*args, **(kwargs or {})))
                record["pv_cuda_events"] = names
                record["head_output"] = [_tensor_bits(output)]
                return output
            return func(*args, **(kwargs or {}))

    def attention(module: Any, query: Any, key: Any, value: Any, mask: Any, **kwargs: Any) -> Any:
        if module is not target:
            return original_attention(module, query, key, value, mask, **kwargs)
        if "value_input" in record or kwargs.get("dropout", 0.0) != 0.0 or kwargs.get("scaling") != 0.0625 or kwargs.get("softcap") is not None:
            raise ValueError("Unexpected original attention-output invocation")
        record["value_input"] = _tensor_bits(value)
        if traced:
            with Capture():
                result = original_attention(module, query, key, value, mask, **kwargs)
        else:
            result = original_attention(module, query, key, value, mask, **kwargs)
        if list(result[0].shape) != [1, 30, 4, 256] or list(result[1].shape) != [1, 4, 30, 30]:
            raise ValueError("Unexpected original eager-attention return shape")
        record["eager_output"] = _tensor_bits(result[0])
        record["probability"] = _tensor_bits(result[1])
        return result

    def linear(input: Any, weight: Any, bias: Any = None) -> Any:
        if weight is not target.o_proj.weight:
            return original_linear(input, weight, bias)
        if "projection_geometry" in record or bias is not None:
            raise ValueError("Unexpected output projection occurrence/bias")
        record["projection_geometry"] = [_geometry(input), _geometry(weight)]
        output, events = profiled(lambda: original_linear(input, weight, bias))
        record["projection_cuda_events"] = events
        return output

    def stop(module: Any, args: Any, output: Any) -> None:
        if "projected" in record or list(args[0].shape) != [1, 30, 1024] or list(output.shape) != [1, 30, 640]:
            raise ValueError("Unexpected original projection output boundary")
        record["concatenated"], record["projected"] = _tensor_bits(args[0]), _tensor_bits(output)
        raise Complete()

    ids = torch.tensor(sources.softmax.scores.rotary_bundle["entry_plan"]["input_token_ids"], dtype=torch.int64, device=target.o_proj.weight.device)
    hook, stopped = target.o_proj.register_forward_hook(stop), False
    try:
        with patch.object(namespace, "eager_attention_forward", side_effect=attention), torch.no_grad():
            if traced:
                with patch.object(torch.nn.functional, "linear", side_effect=linear):
                    model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1)
            else:
                model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1)
    except Complete:
        stopped = True
    finally:
        hook.remove()
    required = {"value_input", "eager_output", "probability", "concatenated", "projected"}
    if traced:
        required.update({"head_output", "pv_operands", "pv_geometry", "pv_cuda_events", "projection_geometry", "projection_cuda_events"})
    if not stopped or set(record) != required:
        raise ValueError("Incomplete original output capture or stop boundary")
    return record


def _check_geometry(geometry: dict[str, Any], value: np.ndarray) -> None:
    descriptor = _descriptor(value)
    if any(geometry["tensor"].get(key) != descriptor[key] for key in ("kind", "shape", "dtype", "layout", "numel", "sha256")) or not str(geometry["tensor"].get("device", "")).startswith("cuda:") or not isinstance(geometry["strides"], list) or len(geometry["strides"]) != value.ndim or any(type(stride) is not int or stride < 0 for stride in geometry["strides"]):
        raise ValueError("Output operand geometry is inconsistent")


def output_report(plan: dict[str, Any], probabilities: np.ndarray, values: np.ndarray, predicted: dict[str, np.ndarray], observations: list[dict[str, Any]], runtime: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(observations, list) or len(observations) != 3:
        raise ValueError("Expected three original output repetitions and controls")
    mismatches, checks = [], []
    for repetition, pair in enumerate(observations):
        traced, plain = pair["traced"], pair["untraced"]
        actual = {name: _state_array(traced[name], list(value.shape)) for name, value in predicted.items()}
        inputs_match = all(np.array_equal(_state_array(record["probability"], [1, 4, 30, 30]), probabilities) and np.array_equal(_state_array(record["value_input"], [1, 1, 30, 256]), values) for record in (traced, plain))
        left = _state_array(traced["pv_operands"]["probability"], [4, 30, 30])
        right = _state_array(traced["pv_operands"]["value"], [4, 30, 256])
        if len(traced["pv_geometry"]) != 2 or len(traced["projection_geometry"]) != 2:
            raise ValueError("Incomplete output operand geometry")
        for geometry, operand in zip(traced["pv_geometry"], (left, right)):
            _check_geometry(geometry, operand)
        _check_geometry(traced["projection_geometry"][0], actual["concatenated"])
        weight_descriptor = traced["projection_geometry"][1]["tensor"]
        if any(weight_descriptor.get(key) != plan["weight"][key] for key in ("kind", "shape", "dtype", "layout", "numel", "sha256")) or traced["projection_geometry"][1]["strides"] != [1024, 1]:
            raise ValueError("Original output weight binding/geometry changed")
        for name in ("head_output", "concatenated", "projected"):
            for coordinate in np.argwhere(actual[name] != predicted[name]):
                mismatches.append({"repetition": repetition, "stage": name, "coordinate": coordinate.tolist(),
                                   "predicted_bits": int(predicted[name][tuple(coordinate)]), "observed_bits": int(actual[name][tuple(coordinate)])})
        returned = _state_array(traced["eager_output"], [1, 30, 4, 256]).transpose(0, 2, 1, 3)
        control_head = _state_array(plain["eager_output"], [1, 30, 4, 256]).transpose(0, 2, 1, 3)
        checks.append({"source_inputs_match": inputs_match,
                       "pv_operands_match": np.array_equal(left, probabilities.reshape(4, 30, 30)) and np.array_equal(right, np.repeat(values, 4, axis=1).reshape(4, 30, 256)),
                       "returned_head_coordinates_match": np.array_equal(returned, actual["head_output"]),
                       "original_concatenation_matches": np.array_equal(concatenate_heads(actual["head_output"]), actual["concatenated"]),
                       "untraced_head_coordinates_match": np.array_equal(control_head, actual["head_output"]),
                       "untraced_concatenation_matches": np.array_equal(_state_array(plain["concatenated"], [1, 30, 1024]), actual["concatenated"]),
                       "untraced_projection_matches": np.array_equal(_state_array(plain["projected"], [1, 30, 640]), actual["projected"]),
                       "projection_input_matches_prediction": np.array_equal(actual["concatenated"], predicted["concatenated"])})
        for key in ("pv_cuda_events", "projection_cuda_events"):
            if not isinstance(traced[key], list) or not traced[key] or any(not isinstance(event, str) or not event for event in traced[key]):
                raise ValueError("Missing output-stage CUDA provenance")
    counts = {name: [sum(item["stage"] == name and item["repetition"] == repetition for item in mismatches) for repetition in range(3)] for name in predicted}
    runtime_match = canonical_json(runtime) == canonical_json(plan["runtime"])
    first = next((key for key in checks[0] if key != "projection_input_matches_prediction" and not all(item[key] for item in checks)), None) or (mismatches[0] if mismatches else None) or (None if runtime_match else "runtime")
    body = {"schema_version": 1, "scope": SCOPE, "plan_sha256": plan["plan_sha256"], "runtime": runtime, "runtime_matches_plan": runtime_match,
            "observations": observations, "checks": checks, "mismatch_counts": counts, "mismatch_count": len(mismatches), "mismatches": mismatches,
            "first_divergence": first, "candidate_passes": not mismatches and runtime_match and all(all(item.values()) for item in checks),
            "aggregation_value_count": 30720, "projection_value_count": 19200, "prefix_independently_recomputed": False,
            "post_attention_norm_executed": False, "full_first_layer_qualified": False, "hardware_semantics_established": False,
            "global_exactness_activation_allowed": False}
    return _seal(body, "report_sha256")


def acquire_output(sources: OutputSources, model: Any, plan: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    probabilities, values, predicted = check_output_plan(sources, plan, bundle)
    _model_context(sources, model)
    observations = [{"traced": _capture_forward(sources, model, True), "untraced": _capture_forward(sources, model, False)} for _ in range(3)]
    bind_model_tensors(sources.program, model, verify_hashes=True)
    _code_sha()
    return output_report(plan, probabilities, values, predicted, observations, _runtime())


def verify_output(sources: OutputSources, plan: dict[str, Any], bundle: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    try:
        probabilities, values, predicted = check_output_plan(sources, plan, bundle)
        expected = output_report(plan, probabilities, values, predicted, report["observations"], report["runtime"])
        return {"valid": canonical_json(report) == canonical_json(expected), "mode": "integrity_and_coordinate_checks_only",
                "candidate_passes": expected["candidate_passes"], "mismatch_counts": expected["mismatch_counts"], "first_divergence": expected["first_divergence"],
                "global_exactness_activation_allowed": False}
    except (KeyError, ValueError, TypeError, RuntimeError, AttributeError) as error:
        return {"valid": False, "reason": str(error)}


def output_summary(plan: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in report.items() if key not in ("observations", "mismatches", "report_sha256")}
    body.update({"source_report_sha256": report["report_sha256"], "sources": plan["sources"], "profile": plan["profile"], "code_sha256": plan["code_sha256"],
                 "pv_cuda_events": [pair["traced"]["pv_cuda_events"] for pair in report["observations"]],
                 "projection_cuda_events": [pair["traced"]["projection_cuda_events"] for pair in report["observations"]],
                 "operand_geometry": [{key: pair["traced"][key] for key in ("pv_geometry", "projection_geometry")} for pair in report["observations"]],
                 "observed_hashes": {name: [_descriptor(_state_array(pair["traced"][name], plan["predictions"][name]["shape"]))["sha256"] for pair in report["observations"]] for name in plan["predictions"]}})
    return _seal(body, "summary_sha256")
