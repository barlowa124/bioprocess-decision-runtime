from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from .gemma_attention_entry import _bits, _descriptor, _state_array
from .gemma_attention_scores import ScoreSources, verify_scores, check_score_plan
from .gemma_float_semantics import encode_bfloat16_rne
from .gemma_reduction_semantics import decode_finite_float32, encode_float32_rne
from .gemma_rotary_slice import _sha, _seal, _check_hash, _rotary_context, _f32_bits
from .gemma_rsqrt_lookup import _runtime
from .gemma_ir_interpreter import bind_model_tensors
from .reference_gemma import _rms_f32_add, _rms_f32_round, _rms_f32_value, _rms_code_commitment
from .serialization import canonical_json

SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
OFFSETS = (16, 8, 4, 2, 1)
SCOPE = "Fixed-input warp32 softmax candidate on verified serialized first-layer scores; independent FP32 exponential, XOR sum and division, original FP32/BF16 outputs compared; staged ATen diagnostics are not fused-kernel register observations."
PROFILE = {"lanes": 32, "elements": 30, "warp_batch": 2, "xor_offsets": list(OFFSETS), "padding": "negative infinity then positive-zero exponential",
           "subtract": "float32 RNE", "exponential": "Decimal RN interval enclosing exp, unique float32 RNE encoding", "sum": "per-lane float32 RNE XOR butterfly",
           "divide": "float32 RNE", "output_cast": "bfloat16 RNE"}
STAGES = ("maximum_bits", "shifted_bits", "exponential_bits", "denominator_bits", "output_f32_bits", "output_bf16_bits")
DISPATCH_SOURCE = "pytorch v2.7.1 aten/src/ATen/native/cuda/SoftMax.cu: inner_size=1, dim_size=30, float input selects persistent forward"


def _code_sha() -> str:
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Softmax source changed after process import")
    return _sha({"module": SOURCE_SHA256, "finite_arithmetic": _rms_code_commitment()})


def softmax_header_sha() -> str:
    import torch
    header = Path(torch.__file__).parent / "include/ATen/native/cuda/PersistentSoftmax.cuh"
    return hashlib.sha256(header.read_bytes()).hexdigest()


@lru_cache(maxsize=16384)
def exp_float32_rne(bits: int) -> int:
    if type(bits) is not int or not 0 <= bits <= 0xFFFFFFFF:
        raise ValueError("Exponential input must be a float32 bit encoding")
    if bits == 0xFF800000:
        return 0
    value = _rms_f32_value(bits)
    if value > 0:
        raise ValueError("Softmax exponential requires a nonpositive argument")
    if value == 0:
        return 0x3F800000
    if value <= -128:
        if Fraction(5, 2) ** 128 <= 2 ** 150:
            raise ValueError("Invalid exponential underflow bound")
        return 0
    with localcontext(Context(prec=200, rounding=ROUND_HALF_EVEN, Emin=-999999, Emax=999999)):
        argument = Decimal(value.numerator) / Decimal(value.denominator)
        if Fraction(argument) != value:
            raise ValueError("Decimal argument is not exact")
    for precision in (80, 160, 320):
        with localcontext(Context(prec=precision, rounding=ROUND_HALF_EVEN, Emin=-999999, Emax=999999)):
            rounded = argument.exp()
            unit = Decimal(1).scaleb(rounded.adjusted() - precision + 1)
        lower, upper = Fraction(rounded) - Fraction(unit), Fraction(rounded) + Fraction(unit)
        low_bits, high_bits = encode_float32_rne(lower), encode_float32_rne(upper)
        if low_bits == high_bits:
            return low_bits
    raise ValueError("Exponential interval did not resolve float32 rounding")


def _maximum(left: int, right: int) -> int:
    if left == 0xFF800000:
        return right
    if right == 0xFF800000:
        return left
    return right if _rms_f32_value(left) < _rms_f32_value(right) else left


def predict_softmax_row(bits: list[int]) -> dict[str, Any]:
    if not isinstance(bits, list) or len(bits) != 30 or any(type(value) is not int or not 0 <= value <= 0xFFFFFFFF for value in bits):
        raise ValueError("Softmax candidate requires thirty float32 encodings")
    for value in bits:
        decode_finite_float32(value)
    maximum = [*bits, 0xFF800000, 0xFF800000]
    for offset in OFFSETS:
        maximum = [_maximum(maximum[lane], maximum[lane ^ offset]) for lane in range(32)]
    shifted = [_rms_f32_round(_rms_f32_value(value) - _rms_f32_value(maximum[lane])) for lane, value in enumerate(bits)]
    exponentials = [exp_float32_rne(value) for value in shifted]
    sums, tree = [*exponentials, 0, 0], []
    for offset in OFFSETS:
        sums = [_rms_f32_add(sums[lane], sums[lane ^ offset]) for lane in range(32)]
        tree.append(sums)
    outputs = [_rms_f32_round(_rms_f32_value(value) / _rms_f32_value(sums[lane])) for lane, value in enumerate(exponentials)]
    return {"maximum_bits": maximum[:30], "shifted_bits": shifted, "exponential_bits": exponentials, "xor_sum_stages": tree,
            "denominator_bits": sums[:30], "output_f32_bits": outputs,
            "output_bf16_bits": [encode_bfloat16_rne(*decode_finite_float32(value)) for value in outputs]}


@dataclass(frozen=True)
class SoftmaxSources:
    scores: ScoreSources
    score_plan: dict[str, Any]
    score_bundle: dict[str, Any]
    score_report: dict[str, Any]

    def inputs(self) -> np.ndarray:
        checked = verify_scores(self.scores, self.score_plan, self.score_bundle, self.score_report)
        if not checked["valid"] or not checked["candidate_passes"]:
            raise ValueError("Softmax requires passing verified score evidence")
        _, predictions = check_score_plan(self.scores, self.score_plan, self.score_bundle)
        return predictions["masked"].astype(np.uint32) << 16

    def commitments(self) -> dict[str, Any]:
        return {"score_plan_sha256": self.score_plan["plan_sha256"], "score_report_sha256": self.score_report["report_sha256"],
                "score_bundle_sha256": self.score_plan["bundle_sha256"], "upstream": self.scores.commitments()}


def _instruction(sources: SoftmaxSources) -> dict[str, Any]:
    found = [item for item in sources.scores.program["instructions"] if item["outputs"] == ["layer.0.attention.probability"]]
    if len(found) != 1 or found[0]["opcode"] != "SOFTMAX" or found[0]["inputs"] != ["layer.0.attention.masked_scores"] or found[0]["attributes"] != {"axis": -1, "accumulation_dtype": "torch.float32", "output_dtype": "torch.bfloat16"}:
        raise ValueError("Unsupported softmax IR declaration")
    return found[0]


def build_softmax_plan(sources: SoftmaxSources) -> tuple[dict[str, Any], dict[str, Any]]:
    code = _code_sha()
    values = sources.inputs()
    rows = [predict_softmax_row(row) for row in values.reshape(120, 30).tolist()]
    bundle = {"input_bits": values.tolist(), "rows": rows}
    if _code_sha() != code:
        raise ValueError("Softmax source changed during prediction")
    body = {"schema_version": 1, "scope": SCOPE, "sources": sources.commitments(), "instruction": _instruction(sources),
            "profile": PROFILE, "runtime": sources.score_plan["runtime"], "code_sha256": code,
            "persistent_header_sha256": softmax_header_sha(), "dispatch_source": DISPATCH_SOURCE,
            "shape": [1, 4, 30, 30], "row_count": 120, "value_count": 3600, "repetitions": 3,
            "input_bits_sha256": _sha(values.tolist()), "bundle_sha256": _sha(bundle),
            "expected_kernel": "softmax_warp_forward<float,float,float,5,false,false>",
            "prefix_boundary_reused": True, "candidate_refitting_allowed": False, "fused_internal_stages_observed": False,
            "native_exponential_reconstructed": False, "value_aggregation_executed": False,
            "full_first_layer_qualified": False, "hardware_semantics_established": False, "global_exactness_activation_allowed": False}
    return _seal(body, "plan_sha256"), bundle


def check_softmax_plan(sources: SoftmaxSources, plan: dict[str, Any], bundle: dict[str, Any]) -> np.ndarray:
    _check_hash(plan, "plan_sha256")
    inputs = sources.inputs()
    if plan["code_sha256"] != _code_sha() or plan["persistent_header_sha256"] != softmax_header_sha() or canonical_json(plan["sources"]) != canonical_json(sources.commitments()) or canonical_json(plan["instruction"]) != canonical_json(_instruction(sources)):
        raise ValueError("Softmax implementation/source commitment mismatch")
    if plan["scope"] != SCOPE or canonical_json(plan["profile"]) != canonical_json(PROFILE) or plan["shape"] != [1, 4, 30, 30] or plan["expected_kernel"] != "softmax_warp_forward<float,float,float,5,false,false>" or any(type(plan[key]) is not int or plan[key] != value for key, value in (("row_count", 120), ("value_count", 3600), ("repetitions", 3))):
        raise ValueError("Softmax scope/profile mismatch")
    false_fields = ("candidate_refitting_allowed", "fused_internal_stages_observed", "native_exponential_reconstructed", "value_aggregation_executed", "full_first_layer_qualified", "hardware_semantics_established", "global_exactness_activation_allowed")
    if any(plan.get(key) is not False for key in false_fields) or plan["prefix_boundary_reused"] is not True or plan["dispatch_source"] != DISPATCH_SOURCE or canonical_json(plan["runtime"]) != canonical_json(sources.score_plan["runtime"]):
        raise ValueError("Softmax qualification/runtime mismatch")
    expected = {"input_bits": inputs.tolist(), "rows": [predict_softmax_row(row) for row in inputs.reshape(120, 30).tolist()]}
    if plan["input_bits_sha256"] != _sha(inputs.tolist()) or plan["bundle_sha256"] != _sha(bundle) or canonical_json(expected) != canonical_json(bundle):
        raise ValueError("Softmax predictions do not reproduce independently")
    return inputs


def _native_staged(values: np.ndarray) -> dict[str, Any]:
    import torch

    source = torch.tensor(values.view(np.int32).tolist(), dtype=torch.int32, device="cuda").view(torch.float32)
    maximum = source.max(dim=-1, keepdim=True).values.expand_as(source)
    shifted = source - maximum
    exponential = shifted.exp()
    sums = torch.cat((exponential, torch.zeros((1, 4, 30, 2), dtype=torch.float32, device=source.device)), dim=-1)
    indices = torch.arange(32, device=source.device)
    for offset in OFFSETS:
        sums = sums + sums.index_select(-1, indices ^ offset)
    denominator = sums[..., :30]
    output = exponential / denominator
    tensors = (maximum, shifted, exponential, denominator, output)
    result = {key: _f32_bits(value).reshape(120, 30).tolist() for key, value in zip(STAGES[:-1], tensors)}
    result["output_bf16_bits"] = _bits(output.to(torch.bfloat16)).reshape(120, 30).tolist()
    return result


def _observe_softmax(original: Any, module: Any, query: Any, key: Any, value: Any, mask: Any, kwargs: dict[str, Any], traced: bool) -> dict[str, Any]:
    import torch
    from torch.profiler import profile, ProfilerActivity
    from torch.utils._python_dispatch import TorchDispatchMode

    record, stopped = {}, False

    class ProbabilityBoundary(Exception):
        pass

    class Capture(TorchDispatchMode):
        def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kw: Any = None) -> Any:
            name = str(func)
            if name == "aten._softmax.default":
                if record or args[0].dtype != torch.float32 or list(args[0].shape) != [1, 4, 30, 30] or args[1] not in (-1, 3) or args[2] is not False:
                    raise ValueError("Unexpected original softmax invocation")
                record["input_bits"] = _f32_bits(args[0]).reshape(120, 30).tolist()
                record["input_strides"] = list(args[0].stride())
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as captured:
                    output = func(*args, **(kw or {}))
                    torch.cuda.synchronize()
                record["cuda_events"] = sorted({event.name for event in captured.events() if event.device_type == torch.autograd.DeviceType.CUDA})
                record["output_f32_bits"] = _f32_bits(output).reshape(120, 30).tolist()
                return output
            if name == "aten.bmm.default" and "output_f32_bits" in record:
                raise ValueError("Value aggregation reached before the BF16 probability capture")
            output = func(*args, **(kw or {}))
            if name == "aten._to_copy.default" and "output_f32_bits" in record and output.dtype == torch.bfloat16 and list(output.shape) == [1, 4, 30, 30]:
                if canonical_json(_f32_bits(args[0]).reshape(120, 30).tolist()) != canonical_json(record["output_f32_bits"]):
                    raise ValueError("Probability cast is not linked to native softmax")
                with torch._C._DisableTorchDispatch():
                    record["output_bf16_bits"] = _bits(output).reshape(120, 30).tolist()
                raise ProbabilityBoundary()
            return output

    original_softmax, original_matmul = torch.nn.functional.softmax, torch.matmul

    def softmax(input: Any, dim: Any = None, _stacklevel: int = 3, dtype: Any = None) -> Any:
        if record or dim != -1 or dtype != torch.float32 or input.dtype != torch.bfloat16:
            raise ValueError("Unexpected untraced softmax invocation")
        record["input_bits"] = (_bits(input).astype(np.uint32) << 16).reshape(120, 30).tolist()
        output = original_softmax(input, dim=dim, _stacklevel=_stacklevel, dtype=dtype)
        record["output_f32_bits"] = _f32_bits(output).reshape(120, 30).tolist()
        return output

    def matmul(left: Any, right: Any, *args: Any, **kw: Any) -> Any:
        if "output_f32_bits" not in record:
            return original_matmul(left, right, *args, **kw)
        if left.dtype != torch.bfloat16 or list(left.shape) != [1, 4, 30, 30]:
            raise ValueError("Unexpected untraced value-aggregation boundary")
        record["output_bf16_bits"] = _bits(left).reshape(120, 30).tolist()
        raise ProbabilityBoundary()

    try:
        if traced:
            with Capture():
                original(module, query, key, value, mask, **kwargs)
        else:
            with patch.object(torch.nn.functional, "softmax", side_effect=softmax), patch.object(torch, "matmul", side_effect=matmul):
                original(module, query, key, value, mask, **kwargs)
    except ProbabilityBoundary:
        stopped = True
    if not stopped or "output_bf16_bits" not in record:
        raise ValueError("Original forward did not stop before value aggregation")
    return record


def _u32_rows(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=object)
    if array.shape != (120, 30) or any(type(bits) is not int or not 0 <= bits <= 0xFFFFFFFF for bits in array.reshape(-1)):
        raise ValueError("Malformed FP32 softmax observation")
    for bits in array.reshape(-1):
        decode_finite_float32(bits)
    return array.astype(np.uint32)


def softmax_report(plan: dict[str, Any], bundle: dict[str, Any], observations: list[dict[str, Any]], runtime: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(observations, list) or len(observations) != 3:
        raise ValueError("Expected three native softmax repetitions")
    predicted = {key: np.asarray([row[key] for row in bundle["rows"]], dtype=np.uint16 if key == "output_bf16_bits" else np.uint32) for key in STAGES}
    expected_input = np.asarray(bundle["input_bits"], dtype=np.uint32).reshape(120, 30)
    mismatches, diagnostics, events, conditions = [], [], [], []
    for repetition, observation in enumerate(observations):
        traced, plain, staged = observation["traced"], observation["untraced"], observation["staged_aten"]
        native = {"output_f32_bits": _u32_rows(traced["output_f32_bits"]), "output_bf16_bits": _state_array(traced["output_bf16_bits"], [120, 30])}
        for key in native:
            for row, column in np.argwhere(native[key] != predicted[key]):
                mismatches.append({"repetition": repetition, "stage": key, "coordinate": [int(row // 30), int(row % 30), int(column)],
                                   "predicted_bits": int(predicted[key][row, column]), "observed_bits": int(native[key][row, column])})
        stage_values = {key: _state_array(staged[key], [120, 30]) if key == "output_bf16_bits" else _u32_rows(staged[key]) for key in STAGES}
        diagnostics.append({"candidate_vs_staged_aten": {key: int(np.count_nonzero(predicted[key] != stage_values[key])) for key in STAGES},
                            "staged_vs_fused_fp32": int(np.count_nonzero(stage_values["output_f32_bits"] != native["output_f32_bits"])),
                            "staged_vs_fused_bf16": int(np.count_nonzero(stage_values["output_bf16_bits"] != native["output_bf16_bits"]))})
        names = traced["cuda_events"]
        if not isinstance(names, list) or not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("Missing softmax CUDA provenance")
        kernel_matches = any("softmax_warp_forward" in name and ("IfffLi5ELb0ELb0E" in name or "<float, float, float, 5, false, false>" in name) for name in names)
        conditions.append({"input_matches": np.array_equal(_u32_rows(traced["input_bits"]), expected_input) and np.array_equal(_u32_rows(plain["input_bits"]), expected_input),
                           "untraced_fp32_matches": np.array_equal(_u32_rows(plain["output_f32_bits"]), native["output_f32_bits"]),
                           "untraced_bf16_matches": np.array_equal(_state_array(plain["output_bf16_bits"], [120, 30]), native["output_bf16_bits"]),
                           "persistent_kernel_profile_matches": kernel_matches,
                           "native_bfloat16_cast_matches": np.array_equal(native["output_bf16_bits"], np.asarray([encode_bfloat16_rne(*decode_finite_float32(int(bits))) for bits in native["output_f32_bits"].reshape(-1)], dtype=np.uint16).reshape(120, 30))})
        if not isinstance(traced["input_strides"], list) or len(traced["input_strides"]) != 4 or any(type(stride) is not int or stride < 0 for stride in traced["input_strides"]):
            raise ValueError("Malformed softmax input strides")
        events.append(names)
    counts = {key: [sum(item["stage"] == key and item["repetition"] == repetition for item in mismatches) for repetition in range(3)] for key in ("output_f32_bits", "output_bf16_bits")}
    runtime_matches = canonical_json(runtime) == canonical_json(plan["runtime"])
    body = {"schema_version": 1, "scope": SCOPE, "plan_sha256": plan["plan_sha256"], "runtime": runtime, "runtime_matches_plan": runtime_matches,
            "observations": observations, "mismatch_count": len(mismatches), "mismatches": mismatches, "mismatch_counts": counts,
            "conditions": conditions, "diagnostics": diagnostics, "cuda_events": events,
            "candidate_passes": not mismatches and runtime_matches and all(all(item.values()) for item in conditions),
            "fused_internal_stages_observed": False, "staged_aten_is_diagnostic": True,
            "prefix_independently_recomputed": False, "native_exponential_reconstructed": False, "value_aggregation_executed": False,
            "full_first_layer_qualified": False, "hardware_semantics_established": False, "global_exactness_activation_allowed": False}
    return _seal(body, "report_sha256")


def acquire_softmax(sources: SoftmaxSources, model: Any, plan: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    import torch

    inputs = check_softmax_plan(sources, plan, bundle)
    _, _, binding = _rotary_context(sources.scores.program, model)
    if canonical_json(binding) != canonical_json(sources.scores.rotary_bundle["entry_plan"]["model_binding"]) or canonical_json(_runtime()) != canonical_json(plan["runtime"]):
        raise ValueError("Original softmax model/runtime differs from the plan")
    target = model.model.layers[0].self_attn
    namespace = importlib.import_module(type(target).__module__)
    original = namespace.eager_attention_forward
    ids = torch.tensor(sources.scores.rotary_bundle["entry_plan"]["input_token_ids"], dtype=torch.int64, device=model.model.embed_tokens.weight.device)
    observations = []

    class Complete(Exception):
        pass

    for _ in range(3):
        pair = {}
        for traced, label in ((True, "traced"), (False, "untraced")):
            stopped = False

            def capture(module: Any, query: Any, key: Any, value: Any, mask: Any, **kwargs: Any) -> Any:
                if module is not target:
                    return original(module, query, key, value, mask, **kwargs)
                if label in pair or kwargs.get("scaling") != 0.0625 or kwargs.get("dropout", 0.0) != 0.0 or kwargs.get("softcap") is not None:
                    raise ValueError("Unexpected original attention invocation")
                pair[label] = _observe_softmax(original, module, query, key, value, mask, kwargs, traced)
                raise Complete()

            try:
                with patch.object(namespace, "eager_attention_forward", side_effect=capture), torch.no_grad():
                    model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1)
            except Complete:
                stopped = True
            if not stopped:
                raise ValueError("Missing softmax boundary")
        with torch.no_grad():
            pair["staged_aten"] = _native_staged(inputs)
        observations.append(pair)
    bind_model_tensors(sources.scores.program, model, verify_hashes=True)
    _code_sha()
    return softmax_report(plan, bundle, observations, _runtime())


def verify_softmax(sources: SoftmaxSources, plan: dict[str, Any], bundle: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    try:
        check_softmax_plan(sources, plan, bundle)
        expected = softmax_report(plan, bundle, report["observations"], report["runtime"])
        return {"valid": canonical_json(expected) == canonical_json(report), "mode": "integrity_and_independent_softmax_recomputation",
                "candidate_passes": expected["candidate_passes"], "mismatch_counts": expected["mismatch_counts"],
                "diagnostics": expected["diagnostics"], "global_exactness_activation_allowed": False}
    except (KeyError, ValueError, TypeError, RuntimeError, AttributeError, StopIteration) as error:
        return {"valid": False, "reason": str(error)}


def softmax_summary(plan: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in report.items() if key not in ("observations", "mismatches", "report_sha256")}
    body.update({"source_report_sha256": report["report_sha256"], "sources": plan["sources"], "profile": plan["profile"], "code_sha256": plan["code_sha256"],
                 "row_count": 120, "value_count": 3600, "input_strides": [pair["traced"]["input_strides"] for pair in report["observations"]],
                 "observed_fp32_hashes": [_sha(pair["traced"]["output_f32_bits"]) for pair in report["observations"]],
                 "observed_bf16_hashes": [_sha(pair["traced"]["output_bf16_bits"]) for pair in report["observations"]]})
    return _seal(body, "summary_sha256")
