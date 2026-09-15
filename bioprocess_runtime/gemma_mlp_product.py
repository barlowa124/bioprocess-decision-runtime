from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from .gemma_attention_entry import _bits, _descriptor, _state_array
from .gemma_mlp_entry import MlpEntrySources, verify_mlp, _model_context as entry_model_context
from .gemma_gelu_lookup import CheckedGeluLookup, _code_sha as gelu_code_sha
from .gemma_float_semantics import bfloat16_multiply_bits
from .gemma_ir_interpreter import bind_model_tensors, _projection_arithmetic_commitment
from .gemma_rsqrt_lookup import _runtime
from .gemma_rotary_slice import _sha, _seal, _check_hash
from .serialization import canonical_json

SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
SCOPE = "Fixed-case empirical GELU-tanh lookup and independent finite-BF16 activation/up multiplication from verified gate/up boundaries; original forward stopped before down projection, not native activation arithmetic or full-layer qualification."


def _code_sha() -> str:
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("MLP-product source changed after import")
    return _sha({"module": SOURCE_SHA256, "gelu_lookup": gelu_code_sha(), "scalar_arithmetic": _projection_arithmetic_commitment()})


@dataclass(frozen=True)
class ProductSources:
    entry: MlpEntrySources
    entry_plan: dict[str, Any]
    entry_bundle: dict[str, Any]
    entry_report: dict[str, Any]
    gelu: CheckedGeluLookup

    def inputs(self) -> tuple[np.ndarray, np.ndarray]:
        checked = verify_mlp(self.entry, self.entry_plan, self.entry_bundle, self.entry_report)
        if not checked["valid"] or not checked["mlp_entry_matches"]:
            raise ValueError("MLP product requires passing gate/up evidence")
        if type(self.gelu) is not CheckedGeluLookup or getattr(self.gelu.predict_bits, "__func__", None) is not CheckedGeluLookup.predict_bits:
            raise ValueError("MLP product requires the registered checked GELU specification")
        return tuple(_state_array(self.entry_bundle["predictions"][role], [1, 30, 2048]) for role in ("gate", "up"))

    def commitments(self) -> dict[str, Any]:
        return {"program_sha256": self.entry.post.program["program_sha256"], "entry_plan_sha256": self.entry_plan["plan_sha256"],
                "entry_report_sha256": self.entry_report["report_sha256"], "entry_bundle_sha256": self.entry_plan["bundle_sha256"], "gelu_evidence": self.gelu.evidence}


def product_instructions(program: dict[str, Any]) -> list[dict[str, Any]]:
    specs = (("layer.0.mlp.activated_gate", "GELU_TANH", ["layer.0.mlp.gate"], {}),
             ("layer.0.mlp.product", "MUL", ["layer.0.mlp.activated_gate", "layer.0.mlp.up"], {"output_dtype": "torch.bfloat16"}))
    selected = []
    for output, opcode, inputs, attributes in specs:
        found = [item for item in program["instructions"] if item["outputs"] == [output]]
        if len(found) != 1 or found[0]["opcode"] != opcode or found[0]["inputs"] != inputs or found[0]["attributes"] != attributes or found[0]["parameter_refs"]:
            raise ValueError("Unsupported activation/product IR binding")
        selected.append(found[0])
    return selected


def predict_product(gate: np.ndarray, up: np.ndarray, lookup: CheckedGeluLookup, runtime: dict[str, Any]) -> dict[str, Any]:
    if gate.dtype != np.uint16 or up.dtype != np.uint16 or gate.shape != (1, 30, 2048) or up.shape != gate.shape:
        raise ValueError("MLP product requires BF16 gate/up [1,30,2048]")
    activation = np.asarray([lookup.predict_bits(int(value), runtime) for value in gate.reshape(-1)], dtype=np.uint16).reshape(gate.shape)
    product = np.asarray([bfloat16_multiply_bits(int(left), int(right)) for left, right in zip(activation.reshape(-1), up.reshape(-1))], dtype=np.uint16).reshape(gate.shape)
    return {"activation": activation.tolist(), "product": product.tolist()}


def _plan_body(sources: ProductSources, gate: np.ndarray, up: np.ndarray, bundle: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": 1, "scope": SCOPE, "sources": sources.commitments(), "instructions": product_instructions(sources.entry.post.program),
            "code_sha256": _code_sha(), "runtime": sources.entry_plan["runtime"], "model_binding": sources.entry_plan["model_binding"],
            "inputs": {"gate": _descriptor(gate), "up": _descriptor(up)},
            "predictions": {name: _descriptor(_state_array(value, [1, 30, 2048])) for name, value in bundle.items()},
            "bundle_sha256": _sha(bundle), "activation_value_count": 61440, "product_value_count": 61440, "repetitions": 3,
            "prefix_boundary_reused": True, "uses_empirical_gelu_specification": True, "native_activation_arithmetic_reconstructed": False,
            "down_projection_executed": False, "full_first_layer_qualified": False, "global_exactness_activation_allowed": False}


def build_product_plan(sources: ProductSources) -> tuple[dict[str, Any], dict[str, Any]]:
    code = _code_sha()
    gate, up = sources.inputs()
    bundle = predict_product(gate, up, sources.gelu, sources.entry_plan["runtime"])
    if _code_sha() != code:
        raise ValueError("MLP-product source changed during prediction")
    return _seal(_plan_body(sources, gate, up, bundle), "plan_sha256"), bundle


def check_product_plan(sources: ProductSources, plan: dict[str, Any], bundle: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    _check_hash(plan, "plan_sha256")
    gate, up = sources.inputs()
    expected_bundle = predict_product(gate, up, sources.gelu, sources.entry_plan["runtime"])
    expected_plan = _seal(_plan_body(sources, gate, up, expected_bundle), "plan_sha256")
    if canonical_json(plan) != canonical_json(expected_plan) or canonical_json(bundle) != canonical_json(expected_bundle):
        raise ValueError("Activation/product predictions do not reproduce independently")
    return gate, up, {name: _state_array(value, [1, 30, 2048]) for name, value in expected_bundle.items()}


def _model_context(sources: ProductSources, model: Any) -> None:
    from transformers.activations import PytorchGELUTanh
    entry_model_context(sources.entry, model)
    activation = model.model.layers[0].mlp.act_fn
    if type(activation) is not PytorchGELUTanh or getattr(activation.forward, "__func__", None) is not PytorchGELUTanh.forward or hashlib.sha256(inspect.getsource(PytorchGELUTanh.forward).encode("utf-8")).hexdigest() != sources.gelu.activation_source_sha256:
        raise ValueError("Original model activation differs from the GELU specification")
    if canonical_json(_runtime()) != canonical_json(sources.entry_plan["runtime"]):
        raise ValueError("Original activation runtime differs from the prediction plan")


def _capture_product(model: Any, token_ids: list[list[int]], traced: bool) -> dict[str, Any]:
    import torch
    from torch.profiler import profile, ProfilerActivity
    from torch.utils._python_dispatch import TorchDispatchMode

    mlp = model.model.layers[0].mlp
    original_mlp, original_activation = mlp.forward, mlp.act_fn.forward
    record, handles = {"order": []}, []

    class Complete(Exception):
        pass

    def bits(value: Any) -> list[Any]:
        with torch._C._DisableTorchDispatch():
            return _bits(value).tolist()

    def profiled(call: Any) -> tuple[Any, list[str]]:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as captured:
            output = call()
            torch.cuda.synchronize()
        names = sorted({event.name for event in captured.events() if event.device_type == torch.autograd.DeviceType.CUDA})
        return output, names

    class Capture(TorchDispatchMode):
        def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
            if str(func) == "aten.mul.Tensor" and "up" in record and list(args[0].shape) == [1, 30, 2048]:
                if "multiply_output" in record or canonical_json(bits(args[0])) != canonical_json(record["activation"]) or canonical_json(bits(args[1])) != canonical_json(record["up"]):
                    raise ValueError("Original multiplication operands/occurrence do not match gate/up outputs")
                output, names = profiled(lambda: func(*args, **(kwargs or {})))
                record["multiply_output"], record["product_cuda_events"] = bits(output), names
                record["multiply_strides"] = [list(value.stride()) for value in args[:2]]
                return output
            return func(*args, **(kwargs or {}))

    def mlp_forward(value: Any) -> Any:
        with Capture():
            return original_mlp(value)

    def activation_forward(value: Any) -> Any:
        if "activation_cuda_events" in record:
            raise ValueError("Repeated GELU invocation")
        output, names = profiled(lambda: original_activation(value))
        record["activation_cuda_events"] = names
        record["activation_input_strides"] = list(value.stride())
        return output

    def capture(role: str):
        def hook(module: Any, args: Any, output: Any) -> None:
            if role in record or list(output.shape) != [1, 30, 2048]:
                raise ValueError("Unexpected activation/product source boundary")
            if role == "activation" and (record["order"] != ["gate"] or canonical_json(bits(args[0])) != canonical_json(record["gate"])):
                raise ValueError("Native GELU input/order differs from the gate output")
            record[role] = bits(output)
            record["order"].append(role)
        return hook

    def stop(module: Any, args: Any) -> None:
        if record["order"] != ["gate", "activation", "up"] or list(args[0].shape) != [1, 30, 2048]:
            raise ValueError("Unexpected down-projection input/order")
        record["product"] = bits(args[0])
        record["order"].append("product")
        raise Complete()

    stopped = False
    try:
        handles.append(mlp.gate_proj.register_forward_hook(capture("gate")))
        handles.append(mlp.act_fn.register_forward_hook(capture("activation")))
        handles.append(mlp.up_proj.register_forward_hook(capture("up")))
        handles.append(mlp.down_proj.register_forward_pre_hook(stop))
        ids = torch.tensor(token_ids, dtype=torch.int64, device=mlp.gate_proj.weight.device)
        with torch.no_grad():
            if traced:
                with patch.object(mlp, "forward", side_effect=mlp_forward), patch.object(mlp.act_fn, "forward", side_effect=activation_forward):
                    model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1)
            else:
                model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1)
    except Complete:
        stopped = True
    finally:
        for handle in handles:
            handle.remove()
    if not stopped or record["order"] != ["gate", "activation", "up", "product"] or (traced and "multiply_output" not in record):
        raise ValueError("Original forward did not stop at the complete product boundary")
    return record


def product_report(plan: dict[str, Any], gate: np.ndarray, up: np.ndarray, predictions: dict[str, np.ndarray], observations: list[dict[str, Any]], runtime: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(observations, list) or len(observations) != 3:
        raise ValueError("Expected three activation/product observations")
    mismatches, checks = [], []
    for repetition, pair in enumerate(observations):
        traced, plain = pair["traced"], pair["untraced"]
        actual = {role: _state_array(traced[role], [1, 30, 2048]) for role in ("activation", "product")}
        for role in ("activation", "product"):
            for coordinate in np.argwhere(actual[role] != predictions[role]):
                mismatches.append({"repetition": repetition, "stage": role, "coordinate": coordinate.tolist(), "predicted_bits": int(predictions[role][tuple(coordinate)]), "observed_bits": int(actual[role][tuple(coordinate)])})
        checks.append({"source_inputs_match": all(np.array_equal(_state_array(record[role], [1, 30, 2048]), expected) for record in (traced, plain) for role, expected in (("gate", gate), ("up", up))),
                       "untraced_outputs_match": all(np.array_equal(_state_array(plain[role], [1, 30, 2048]), actual[role]) for role in ("activation", "product")),
                       "multiply_reaches_down_input": np.array_equal(_state_array(traced["multiply_output"], [1, 30, 2048]), actual["product"]),
                       "declared_order_matches": all(record["order"] == ["gate", "activation", "up", "product"] for record in (traced, plain)),
                       "declared_contiguous_layout_matches": traced["activation_input_strides"] == [61440, 2048, 1] and traced["multiply_strides"] == [[61440, 2048, 1]] * 2})
        for key in ("activation_cuda_events", "product_cuda_events"):
            if not isinstance(traced[key], list) or not traced[key] or any(not isinstance(name, str) or not name for name in traced[key]):
                raise ValueError("Missing activation/product CUDA provenance")
    runtime_match = canonical_json(runtime) == canonical_json(plan["runtime"])
    first = next((key for key in checks[0] if not all(item[key] for item in checks)), None) or (mismatches[0] if mismatches else None) or (None if runtime_match else "runtime")
    counts = {role: [sum(item["stage"] == role and item["repetition"] == repetition for item in mismatches) for repetition in range(3)] for role in ("activation", "product")}
    return _seal({"schema_version": 1, "scope": SCOPE, "plan_sha256": plan["plan_sha256"], "runtime": runtime, "runtime_matches_plan": runtime_match,
                  "observations": observations, "checks": checks, "mismatches": mismatches, "mismatch_counts": counts, "first_divergence": first,
                  "activation_and_product_match": not mismatches and runtime_match and all(all(item.values()) for item in checks),
                  "activation_value_count": 61440, "product_value_count": 61440, "prefix_independently_recomputed": False,
                  "uses_empirical_gelu_specification": True, "native_activation_arithmetic_reconstructed": False, "down_projection_executed": False,
                  "full_first_layer_qualified": False, "hardware_semantics_established": False, "global_exactness_activation_allowed": False}, "report_sha256")


def acquire_product(sources: ProductSources, model: Any, plan: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    gate, up, predictions = check_product_plan(sources, plan, bundle)
    _model_context(sources, model)
    ids = sources.entry.post.survivor_context[0].softmax.scores.rotary_bundle["entry_plan"]["input_token_ids"]
    observations = [{"traced": _capture_product(model, ids, True), "untraced": _capture_product(model, ids, False)} for _ in range(3)]
    bind_model_tensors(sources.entry.post.program, model, verify_hashes=True)
    _code_sha()
    return product_report(plan, gate, up, predictions, observations, _runtime())


def verify_product(sources: ProductSources, plan: dict[str, Any], bundle: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    try:
        gate, up, predictions = check_product_plan(sources, plan, bundle)
        expected = product_report(plan, gate, up, predictions, report["observations"], report["runtime"])
        return {"valid": canonical_json(expected) == canonical_json(report), "mode": "independent_activation_product_recomputation_and_integrity",
                "activation_and_product_match": expected["activation_and_product_match"], "mismatch_counts": expected["mismatch_counts"],
                "global_exactness_activation_allowed": False}
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError) as error:
        return {"valid": False, "reason": str(error)}


def product_summary(plan: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in report.items() if key not in ("observations", "mismatches", "report_sha256")}
    body.update({"source_report_sha256": report["report_sha256"], "sources": plan["sources"], "code_sha256": plan["code_sha256"],
                 "cuda_events": {role: [pair["traced"][role + "_cuda_events"] for pair in report["observations"]] for role in ("activation", "product")},
                 "observed_output_hashes": {role: [_descriptor(_state_array(pair["traced"][role], [1, 30, 2048]))["sha256"] for pair in report["observations"]] for role in ("activation", "product")}})
    return _seal(body, "summary_sha256")
