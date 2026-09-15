from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from .gemma_attention_entry import _bits, _descriptor, _state_array
from .gemma_ir_interpreter import _projection_arithmetic_commitment, bind_model_tensors
from .gemma_rotary_slice import ROTATED, _sha, _seal, _check_hash, _rotary_context, check_rotary_slice, verify_rotary_slice
from .gemma_rsqrt_lookup import CheckedRsqrtLookup, _runtime
from .gemma_wmma_candidate import OPERAND_ALIGNMENT_PROFILE, operand_aligned_product_bits
from .gemma_float_semantics import bfloat16_multiply_bits, bfloat16_add_bits
from .operational_semantics import tensor_descriptor
from .serialization import canonical_json

SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
SCOPE = "Frozen first-layer attention-score candidate on verified serialized RoPE boundaries; independent K256 dot, BF16 scale and causal-mask prediction; original forward stopped before softmax; not generic batched-GEMM or full-layer qualification."
STAGES = ("unscaled", "scaled", "masked")


def _code_sha() -> str:
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Attention-score source changed after process import")
    return _sha({"score_module": SOURCE_SHA256, "arithmetic": _projection_arithmetic_commitment()})


@dataclass(frozen=True)
class ScoreSources:
    program: dict[str, Any]
    rotary_plan: dict[str, Any]
    rotary_bundle: dict[str, Any]
    rotary_report: dict[str, Any]
    lookup: CheckedRsqrtLookup
    table_plan: dict[str, Any]
    table_manifest: dict[str, Any]
    table_bundle: dict[str, Any]

    def states(self) -> dict[str, np.ndarray]:
        evidence = (self.lookup, self.table_plan, self.table_manifest, self.table_bundle)
        checked = verify_rotary_slice(self.program, self.rotary_plan, self.rotary_bundle, self.rotary_report, *evidence)
        if not checked["valid"] or not checked["rotated_attention_inputs_bit_exact"]:
            raise ValueError("Attention scores require intact passing rotary-boundary evidence")
        return check_rotary_slice(self.program, self.rotary_plan, self.rotary_bundle, *evidence)

    def commitments(self) -> dict[str, Any]:
        return {"program_sha256": self.program["program_sha256"], "rotary_plan_sha256": self.rotary_plan["plan_sha256"],
                "rotary_report_sha256": self.rotary_report["report_sha256"], "rotary_execution_root": self.rotary_plan["execution_root"],
                "rotary_bundle_sha256": self.rotary_plan["bundle_sha256"], "lookup_evidence": self.lookup.evidence,
                "table_manifest_sha256": self.table_manifest["manifest_sha256"]}


def score_instructions(program: dict[str, Any]) -> list[dict[str, Any]]:
    specifications = (
        ("layer.0.key.repeated", "REPEAT_KV", [ROTATED[1]], {"repetitions": 4}),
        ("layer.0.attention.unscaled_scores", "MATMUL_QK", [ROTATED[0], "layer.0.key.repeated"], {}),
        ("layer.0.attention.scaled_scores", "SCALE", ["layer.0.attention.unscaled_scores"], {"scalar": 0.0625}),
        ("mask.sliding", "CAUSAL_MASK", ["input_ids", "hidden.0"], {"sequence_length": "S", "sliding_window": 512}),
        ("layer.0.attention.masked_scores", "ADD", ["layer.0.attention.scaled_scores", "mask.sliding"], {"output_dtype": "torch.bfloat16"}),
    )
    selected = []
    for output, opcode, inputs, attributes in specifications:
        found = [item for item in program["instructions"] if item["outputs"] == [output]]
        if len(found) != 1 or found[0]["opcode"] != opcode or found[0]["inputs"] != inputs or found[0]["attributes"] != attributes or found[0]["parameter_refs"]:
            raise ValueError("Unsupported attention score/mask IR binding")
        selected.append(found[0])
    return selected


def causal_mask_bits() -> np.ndarray:
    return np.asarray([[[[0 if column <= row else 0xFF7F for column in range(30)] for row in range(30)]]], dtype=np.uint16)


def scale_score_bits(raw: np.ndarray) -> np.ndarray:
    return np.asarray([bfloat16_multiply_bits(int(bits), 0x3D80) for bits in raw.reshape(-1)], dtype=np.uint16).reshape(1, 4, 30, 30)


def mask_score_bits(scaled: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if scaled.shape != (1, 4, 30, 30) or mask.shape != (1, 1, 30, 30):
        raise ValueError("Unsupported score mask geometry")
    return np.asarray([[[[bfloat16_add_bits(int(scaled[0, head, row, column]), int(mask[0, 0, row, column])) for column in range(30)] for row in range(30)] for head in range(4)]], dtype=np.uint16)


def predict_score_bits(query: np.ndarray, key: np.ndarray) -> dict[str, np.ndarray]:
    if query.dtype != np.uint16 or key.dtype != np.uint16 or query.shape != (1, 4, 30, 256) or key.shape != (1, 1, 30, 256):
        raise ValueError("Score candidate requires Q[1,4,30,256] and K[1,1,30,256] BF16 encodings")
    keys = key[0, 0].tolist()
    raw = np.asarray([[[[operand_aligned_product_bits(row, column) for column in keys] for row in query[0, head].tolist()] for head in range(4)]], dtype=np.uint16)
    scaled, mask = scale_score_bits(raw), causal_mask_bits()
    return {"unscaled": raw, "scaled": scaled, "mask": mask, "masked": mask_score_bits(scaled, mask)}


def build_score_plan(sources: ScoreSources) -> tuple[dict[str, Any], dict[str, Any]]:
    code = _code_sha()
    states = sources.states()
    instructions = score_instructions(sources.program)
    predictions = predict_score_bits(states[ROTATED[0]], states[ROTATED[1]])
    bundle = {name: value.tolist() for name, value in predictions.items()}
    if _code_sha() != code:
        raise ValueError("Score numerical source changed during prediction")
    body = {"schema_version": 1, "scope": SCOPE, "sources": sources.commitments(), "runtime": sources.rotary_plan["runtime"],
            "instructions": instructions, "profile": {"dot": dict(OPERAND_ALIGNMENT_PROFILE), "reduction_length": 256, "scale_bits": 0x3D80, "mask_bits": 0xFF7F},
            "inputs": {name: _descriptor(states[name]) for name in (*ROTATED, "layer.0.value.heads")},
            "predictions": {name: _descriptor(value) for name, value in predictions.items()}, "bundle_sha256": _sha(bundle), "code_sha256": code,
            "score_count": 3600, "repetitions": 3, "prefix_boundary_reused": True, "candidate_refitting_allowed": False,
            "scope_transfer_prequalified": False, "softmax_executed": False, "full_first_layer_qualified": False,
            "hardware_semantics_established": False, "global_exactness_activation_allowed": False}
    return _seal(body, "plan_sha256"), bundle


def check_score_plan(sources: ScoreSources, plan: dict[str, Any], bundle: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    _check_hash(plan, "plan_sha256")
    states = sources.states()
    if canonical_json(plan["sources"]) != canonical_json(sources.commitments()) or plan["bundle_sha256"] != _sha(bundle) or plan["code_sha256"] != _code_sha():
        raise ValueError("Attention-score source/bundle commitment mismatch")
    expected_profile = {"dot": dict(OPERAND_ALIGNMENT_PROFILE), "reduction_length": 256, "scale_bits": 0x3D80, "mask_bits": 0xFF7F}
    if canonical_json(plan["profile"]) != canonical_json(expected_profile) or canonical_json(plan["instructions"]) != canonical_json(score_instructions(sources.program)) or canonical_json(plan["runtime"]) != canonical_json(sources.rotary_plan["runtime"]):
        raise ValueError("Attention-score numerical profile or runtime changed")
    false_fields = ("candidate_refitting_allowed", "scope_transfer_prequalified", "softmax_executed", "full_first_layer_qualified", "hardware_semantics_established", "global_exactness_activation_allowed")
    if plan["scope"] != SCOPE or type(plan["score_count"]) is not int or plan["score_count"] != 3600 or type(plan["repetitions"]) is not int or plan["repetitions"] != 3 or plan["prefix_boundary_reused"] is not True or any(plan.get(key) is not False for key in false_fields):
        raise ValueError("Attention-score scope overclaim")
    if canonical_json(plan["inputs"]) != canonical_json({name: _descriptor(states[name]) for name in (*ROTATED, "layer.0.value.heads")}):
        raise ValueError("Score inputs do not match the verified rotary boundary")
    if set(bundle) != {*STAGES, "mask"} or set(plan["predictions"]) != set(bundle):
        raise ValueError("Attention-score prediction coverage mismatch")
    predictions = {name: _state_array(bits, [1, 1 if name == "mask" else 4, 30, 30]) for name, bits in bundle.items()}
    if canonical_json(plan["predictions"]) != canonical_json({name: _descriptor(value) for name, value in predictions.items()}):
        raise ValueError("Score prediction descriptor mismatch")
    if not np.array_equal(predictions["mask"], causal_mask_bits()) or not np.array_equal(predictions["scaled"], scale_score_bits(predictions["unscaled"])) or not np.array_equal(predictions["masked"], mask_score_bits(predictions["scaled"], predictions["mask"])):
        raise ValueError("Scale/mask predictions violate the frozen arithmetic")
    return states, predictions


def _read_bits(value: Any) -> list[Any]:
    import torch

    with torch._C._DisableTorchDispatch():
        return _bits(value).tolist()


def _observe_scores(original: Any, module: Any, query: Any, key: Any, value: Any, mask: Any, kwargs: dict[str, Any], traced: bool) -> dict[str, Any]:
    import torch
    from torch.profiler import profile, ProfilerActivity
    from torch.utils._python_dispatch import TorchDispatchMode

    if kwargs.get("scaling") != 0.0625 or kwargs.get("softcap") is not None or kwargs.get("dropout", 0.0) != 0.0:
        raise ValueError("Unsupported original score scaling/softcap/dropout")
    if mask is None or list(mask.shape) != [1, 1, 30, 30] or mask.dtype != torch.bfloat16:
        raise ValueError("Unsupported original causal mask")
    record = {"inputs": {name: _read_bits(tensor) for name, tensor in zip((*ROTATED, "layer.0.value.heads"), (query, key, value))}, "mask": _read_bits(mask)}
    stopped = False

    class SoftmaxBoundary(Exception):
        pass

    class Capture(TorchDispatchMode):
        def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kw: Any = None) -> Any:
            name = str(func)
            if name == "aten._softmax.default":
                if set(STAGES) - set(record) or args[0].dtype != torch.float32 or list(args[0].shape) != [1, 4, 30, 30] or args[1] not in (-1, 3) or args[2] is not False:
                    raise ValueError("Unexpected original softmax boundary")
                with torch._C._DisableTorchDispatch():
                    record["softmax_input_f32"] = args[0].detach().contiguous().cpu().view(torch.int32).numpy().astype(np.uint32).tolist()
                raise SoftmaxBoundary()
            if name == "aten.bmm.default":
                if "unscaled" in record or [list(arg.shape) for arg in args[:2]] != [[4, 30, 256], [4, 256, 30]]:
                    raise ValueError("Unexpected score matrix multiplication occurrence/shape")
                record["bmm_inputs"] = {"left": _read_bits(args[0]), "right": _read_bits(args[1])}
                with torch._C._DisableTorchDispatch():
                    record["bmm_geometry"] = [{"tensor": tensor_descriptor(arg), "strides": list(arg.stride())} for arg in args[:2]]
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as captured:
                    output = func(*args, **(kw or {}))
                    torch.cuda.synchronize()
                record["bmm_cuda_events"] = sorted({event.name for event in captured.events() if event.device_type == torch.autograd.DeviceType.CUDA})
                if not record["bmm_cuda_events"]:
                    raise ValueError("No score GEMM CUDA activity was captured")
                record["unscaled"] = [_read_bits(output)]
                return output
            output = func(*args, **(kw or {}))
            if name in ("aten.mul.Tensor", "aten.add.Tensor") and isinstance(output, torch.Tensor) and output.dtype == torch.bfloat16 and list(output.shape) == [1, 4, 30, 30]:
                stage = "scaled" if name == "aten.mul.Tensor" else "masked"
                previous = "unscaled" if stage == "scaled" else "scaled"
                if previous not in record or stage in record or canonical_json(_read_bits(args[0])) != canonical_json(record[previous]):
                    raise ValueError("Original score stage lineage mismatch")
                if stage == "scaled" and args[1] != 0.0625:
                    raise ValueError("Original score scale differs from the declared constant")
                if stage == "masked" and canonical_json(_read_bits(args[1])) != canonical_json(record["mask"]):
                    raise ValueError("Original mask-add operand changed")
                record[stage] = _read_bits(output)
            return output

    def stop_softmax(input: Any, dim: Any = None, _stacklevel: int = 3, dtype: Any = None) -> Any:
        if dim != -1 or dtype != torch.float32:
            raise ValueError("Unexpected untraced softmax boundary")
        record["masked"] = _read_bits(input)
        raise SoftmaxBoundary()

    try:
        if traced:
            with Capture():
                original(module, query, key, value, mask, **kwargs)
        else:
            with patch.object(torch.nn.functional, "softmax", side_effect=stop_softmax):
                original(module, query, key, value, mask, **kwargs)
    except SoftmaxBoundary:
        stopped = True
    if not stopped:
        raise ValueError("Original score execution did not stop before softmax")
    return record


def score_report(plan: dict[str, Any], states: dict[str, np.ndarray], predictions: dict[str, np.ndarray], observations: list[dict[str, Any]], runtime: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(observations, list) or len(observations) != 3:
        raise ValueError("Expected three score observations and controls")
    mismatches, conditions, events = [], [], []
    for repetition, observation in enumerate(observations):
        record, control = observation["traced"], observation["untraced"]
        observed = {name: _state_array(record[name], [1, 1 if name == "mask" else 4, 30, 30]) for name in (*STAGES, "mask")}
        input_match = all(np.array_equal(_state_array(source["inputs"][name], list(states[name].shape)), states[name]) for source in (record, control) for name in (*ROTATED, "layer.0.value.heads"))
        left = _state_array(record["bmm_inputs"]["left"], [4, 30, 256])
        right = _state_array(record["bmm_inputs"]["right"], [4, 256, 30])
        operand_match = np.array_equal(left, states[ROTATED[0]].reshape(4, 30, 256)) and np.array_equal(right, np.repeat(states[ROTATED[1]], 4, axis=1).reshape(4, 30, 256).transpose(0, 2, 1))
        if not isinstance(record["bmm_geometry"], list) or len(record["bmm_geometry"]) != 2:
            raise ValueError("Missing batched-GEMM geometry")
        for geometry, values in zip(record["bmm_geometry"], (left, right)):
            descriptor = _descriptor(values)
            if any(geometry["tensor"].get(key) != descriptor[key] for key in ("kind", "shape", "dtype", "layout", "numel", "sha256")) or not str(geometry["tensor"].get("device", "")).startswith("cuda:") or not isinstance(geometry["strides"], list) or len(geometry["strides"]) != 3 or any(type(stride) is not int or stride < 0 for stride in geometry["strides"]):
                raise ValueError("Batched-GEMM operand geometry/value mismatch")
        for name in ("mask", *STAGES):
            expected = predictions[name]
            for coordinate in np.argwhere(observed[name] != expected):
                mismatches.append({"repetition": repetition, "stage": name, "coordinate": coordinate.tolist(), "predicted_bits": int(expected[tuple(coordinate)]), "observed_bits": int(observed[name][tuple(coordinate)])})
        cast = np.asarray(record["softmax_input_f32"], dtype=object)
        if cast.shape != (1, 4, 30, 30) or any(type(bits) is not int or not 0 <= bits <= 0xFFFFFFFF for bits in cast.reshape(-1)):
            raise ValueError("Malformed softmax input capture")
        conditions.append({"source_inputs_match": input_match, "bmm_operands_match": operand_match,
                           "untraced_mask_matches": np.array_equal(_state_array(control["mask"], [1, 1, 30, 30]), observed["mask"]),
                           "untraced_masked_output_matches": np.array_equal(_state_array(control["masked"], [1, 4, 30, 30]), observed["masked"]),
                           "scale_rule_on_observed_input_matches": np.array_equal(scale_score_bits(observed["unscaled"]), observed["scaled"]),
                           "mask_rule_on_observed_input_matches": np.array_equal(mask_score_bits(observed["scaled"], observed["mask"]), observed["masked"]),
                           "softmax_input_cast_matches": np.array_equal(observed["masked"].astype(np.uint32) << 16, cast)})
        if not isinstance(record["bmm_cuda_events"], list) or not record["bmm_cuda_events"] or any(not isinstance(event, str) or not event for event in record["bmm_cuda_events"]):
            raise ValueError("Missing attention-score CUDA provenance")
        events.append(record["bmm_cuda_events"])
    stage_counts = {name: [sum(item["stage"] == name and item["repetition"] == repetition for item in mismatches) for repetition in range(3)] for name in (*STAGES, "mask")}
    runtime_match = canonical_json(runtime) == canonical_json(plan["runtime"])
    first_condition_failure = next((key for key in conditions[0] if not all(item[key] for item in conditions)), None)
    first_divergence = next((key for key in ("source_inputs_match", "bmm_operands_match") if not all(item[key] for item in conditions)), None) or (mismatches[0] if mismatches else first_condition_failure) or (None if runtime_match else "runtime")
    body = {"schema_version": 1, "scope": SCOPE, "plan_sha256": plan["plan_sha256"], "runtime": runtime, "runtime_matches_plan": runtime_match,
            "observations": observations, "stage_mismatch_counts": stage_counts, "mismatch_count": len(mismatches), "mismatches": mismatches,
            "first_divergence": first_divergence,
            "conditions": conditions, "bmm_cuda_events": events,
            "candidate_passes": not mismatches and runtime_match and all(all(item.values()) for item in conditions),
            "diagnostics_conditioned_on_observed_inputs": True, "score_count": 3600,
            "repeated_traced_outputs_identical": all(canonical_json({name: item["traced"][name] for name in STAGES}) == canonical_json({name: observations[0]["traced"][name] for name in STAGES}) for item in observations),
            "prefix_recomputed_in_this_experiment": False, "softmax_executed": False, "full_first_layer_qualified": False,
            "hardware_semantics_established": False, "global_exactness_activation_allowed": False}
    return _seal(body, "report_sha256")


def acquire_scores(sources: ScoreSources, model: Any, plan: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    import torch

    states, predictions = check_score_plan(sources, plan, bundle)
    _, _, binding = _rotary_context(sources.program, model)
    if canonical_json(binding) != canonical_json(sources.rotary_bundle["entry_plan"]["model_binding"]) or canonical_json(_runtime()) != canonical_json(plan["runtime"]):
        raise ValueError("Original score model/runtime differs from the plan")
    target = model.model.layers[0].self_attn
    namespace = importlib.import_module(type(target).__module__)
    original = namespace.eager_attention_forward
    ids = torch.tensor(sources.rotary_bundle["entry_plan"]["input_token_ids"], dtype=torch.int64, device=model.model.embed_tokens.weight.device)
    observations = []

    class ScoresComplete(Exception):
        pass

    for _ in range(3):
        pair = {}
        for traced, label in ((True, "traced"), (False, "untraced")):
            stopped = False

            def capture(module: Any, query: Any, key: Any, value: Any, mask: Any, **kwargs: Any) -> Any:
                if module is not target:
                    return original(module, query, key, value, mask, **kwargs)
                if label in pair:
                    raise ValueError("Repeated first-layer score boundary")
                pair[label] = _observe_scores(original, module, query, key, value, mask, kwargs, traced)
                raise ScoresComplete()

            try:
                with patch.object(namespace, "eager_attention_forward", side_effect=capture), torch.no_grad():
                    model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1)
            except ScoresComplete:
                stopped = True
            if not stopped or label not in pair:
                raise ValueError("Missing original score boundary")
        observations.append(pair)
    bind_model_tensors(sources.program, model, verify_hashes=True)
    _code_sha()
    return score_report(plan, states, predictions, observations, _runtime())


def verify_scores(sources: ScoreSources, plan: dict[str, Any], bundle: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    try:
        states, predictions = check_score_plan(sources, plan, bundle)
        expected = score_report(plan, states, predictions, report["observations"], report["runtime"])
        return {"valid": canonical_json(expected) == canonical_json(report), "mode": "integrity_only_with_scale_mask_recomputation",
                "candidate_passes": expected["candidate_passes"], "stage_mismatch_counts": expected["stage_mismatch_counts"],
                "first_divergence": expected["first_divergence"], "global_exactness_activation_allowed": False}
    except (KeyError, ValueError, TypeError, RuntimeError, AttributeError, StopIteration) as error:
        return {"valid": False, "reason": str(error)}


def score_summary(plan: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in report.items() if key not in ("observations", "mismatches", "report_sha256")}
    body.update({"source_report_sha256": report["report_sha256"], "sources": plan["sources"], "profile": plan["profile"], "code_sha256": plan["code_sha256"],
                 "bmm_operand_geometry": [pair["traced"]["bmm_geometry"] for pair in report["observations"]],
                 "observed_stage_hashes": {name: [_descriptor(_state_array(pair["traced"][name], [1, 1 if name == "mask" else 4, 30, 30]))["sha256"] for pair in report["observations"]] for name in (*STAGES, "mask")}})
    return _seal(body, "summary_sha256")
