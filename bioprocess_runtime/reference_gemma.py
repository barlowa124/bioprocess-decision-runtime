from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from .interpretability import _model_device, _tokenize
from .operational_semantics import append_chain_record, tensor_descriptor, verify_trace_chain
from .serialization import canonical_json

try:
    import torch
    import torch.nn.functional as functional
except ModuleNotFoundError:
    torch = None
    functional = None


@dataclass(frozen=True)
class ReferenceOutput:
    logits: Any
    hidden_states: tuple[Any, ...]
    records: tuple[dict[str, Any], ...]
    attention_kernel_comparisons: tuple[dict[str, Any], ...]


def _require_torch() -> None:
    if torch is None or functional is None:
        raise RuntimeError("PyTorch is required; install the gemma extra")


def model_state_sha256(model: Any) -> str:
    cached = getattr(model, "_bioprocess_state_sha256", None)
    if cached:
        return cached
    tensors = []
    for name, parameter in model.named_parameters(remove_duplicate=False):
        tensors.append({"kind": "parameter", "name": name, "sha256": tensor_descriptor(parameter)["sha256"]})
    for name, buffer in model.named_buffers(remove_duplicate=False):
        tensors.append({"kind": "buffer", "name": name, "sha256": tensor_descriptor(buffer)["sha256"]})
    body = {
        "model_class": type(model).__name__,
        "config": model.config.to_dict(),
        "tensors": tensors,
    }
    fingerprint = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    setattr(model, "_bioprocess_state_sha256", fingerprint)
    return fingerprint


def reference_rms_norm(value: Any, weight: Any, epsilon: float) -> Any:
    normalized = value.float() * torch.rsqrt(value.float().pow(2).mean(-1, keepdim=True) + epsilon)
    return (normalized * (1.0 + weight.float())).type_as(value)


RMS_REDUCTIONS = ("source_vec4_warp32", "sequential_float32", "exact_sum_float32")
RMS_ROOTS = ("rsqrt_rne", "sqrt_rne_then_reciprocal_rne")


def _rms_sha(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _rms_f32_value(bits: int) -> Fraction:
    from .gemma_reduction_semantics import decode_finite_float32
    return decode_finite_float32(bits)[0]


def _rms_f32_round(value: Fraction, negative_zero: bool = False) -> int:
    from .gemma_reduction_semantics import encode_float32_rne, decode_finite_float32
    bits = encode_float32_rne(value, negative_zero)
    decode_finite_float32(bits)
    return bits


def _rms_f32_add(left: int, right: int) -> int:
    return _rms_f32_round(_rms_f32_value(left) + _rms_f32_value(right), left == right == 0x80000000)


def _rms_f32_mul(left: int, right: int) -> int:
    value = _rms_f32_value(left) * _rms_f32_value(right)
    return _rms_f32_round(value, value == 0 and bool((left ^ right) & 0x80000000))


def _rms_bfloat_to_float(bits: int) -> int:
    from .gemma_float_semantics import decode_finite_bfloat16
    value, negative_zero = decode_finite_bfloat16(bits)
    return _rms_f32_round(value, negative_zero)


def rms_root_bits(bits: int, mode: str) -> int:
    value = _rms_f32_value(bits)
    if value <= 0 or mode not in RMS_ROOTS:
        raise ValueError("RMS root requires a positive finite input and a declared profile")
    target = 1 / value if mode == "rsqrt_rne" else value
    low, high = 0, 0x7F7FFFFF
    while low < high:
        middle = (low + high + 1) // 2
        if _rms_f32_value(middle) ** 2 <= target:
            low = middle
        else:
            high = middle - 1
    if low == 0x7F7FFFFF:
        raise ValueError("RMS root is outside the supported finite result domain")
    midpoint = (_rms_f32_value(low) + _rms_f32_value(low + 1)) / 2
    squared = midpoint ** 2
    result = low + int(target > squared or (target == squared and low & 1))
    return result if mode == "rsqrt_rne" else _rms_f32_round(1 / _rms_f32_value(result))


def rms_sum_bits(values: list[int], mode: str) -> int:
    if not values or mode not in RMS_REDUCTIONS:
        raise ValueError("RMS sum requires inputs and a declared profile")
    if mode == "exact_sum_float32":
        return _rms_f32_round(sum((_rms_f32_value(bits) for bits in values), Fraction(0)))
    if mode == "sequential_float32":
        result = 0
        for bits in values:
            result = _rms_f32_add(result, bits)
        return result
    if len(values) not in (256, 640):
        raise ValueError("Source-derived RMS reduction is restricted to aligned K256/K640 rows")
    lanes = []
    for lane in range(32):
        accumulators = [0] * 4
        for start in range(lane * 4, len(values), 128):
            for offset in range(4):
                accumulators[offset] = _rms_f32_add(accumulators[offset], values[start + offset])
        result = accumulators[0]
        for partial in accumulators[1:]:
            result = _rms_f32_add(result, partial)
        lanes.append(result)
    while len(lanes) > 1:
        lanes = [_rms_f32_add(lanes[k], lanes[k + 1]) for k in range(0, len(lanes), 2)]
    return lanes[0]


def rms_row_candidate(input_bits: list[int], weight_bits: list[int], epsilon: float, reduction: str, root: str) -> dict[str, Any]:
    from .gemma_float_semantics import encode_bfloat16_rne
    from .gemma_reduction_semantics import decode_finite_float32

    if len(input_bits) != len(weight_bits) or len(input_bits) not in (256, 640) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Unsupported RMS row or epsilon")
    values = [_rms_bfloat_to_float(bits) for bits in input_bits]
    squared = [_rms_f32_mul(bits, bits) for bits in values]
    mean = _rms_f32_mul(rms_sum_bits(squared, reduction), _rms_f32_round(Fraction(1, len(values))))
    denominator = _rms_f32_add(mean, _rms_f32_round(Fraction.from_float(float(epsilon))))
    inverse = rms_root_bits(denominator, root)
    outputs = []
    for value, weight in zip(values, weight_bits):
        scaled = _rms_f32_mul(_rms_f32_mul(value, inverse), _rms_f32_add(0x3F800000, _rms_bfloat_to_float(weight)))
        result, negative_zero = decode_finite_float32(scaled)
        outputs.append(encode_bfloat16_rne(result, negative_zero))
    return {"mean_bits": mean, "denominator_bits": denominator, "rsqrt_bits": inverse, "output_bits": outputs}


RMS_SLICE_SCOPE = "Fixed-input actual layer-0 input/Q/K RMS characterization with independently specified finite arithmetic candidates and observed ATen stages; not fresh-prompt validation, linked-binary proof, or first-layer qualification."
RMS_SLICE_ROLES = (("input_norm", "model.layers.0.input_layernorm.weight", "layer.0.attention.normalized"),
                   ("query_norm", "model.layers.0.self_attn.q_norm.weight", "layer.0.query.normalized"),
                   ("key_norm", "model.layers.0.self_attn.k_norm.weight", "layer.0.key.normalized"))


def _rms_code_commitment() -> str:
    import inspect
    from . import gemma_float_semantics, gemma_reduction_semantics

    functions = (_rms_f32_value, _rms_f32_round, _rms_f32_add, _rms_f32_mul, _rms_bfloat_to_float, rms_root_bits, rms_sum_bits, rms_row_candidate)
    return _rms_sha({"functions": {fn.__name__: inspect.getsource(fn) for fn in functions},
                     "bfloat16": inspect.getsource(gemma_float_semantics), "float32": inspect.getsource(gemma_reduction_semantics)})


def _rms_rows_tensor(rows: Any, shape: list[int], device: Any) -> Any:
    if not isinstance(rows, list) or len(rows) != math.prod(shape[:-1]) or any(not isinstance(row, list) or len(row) != shape[-1] or any(type(bits) is not int or not 0 <= bits <= 65535 for bits in row) for row in rows):
        raise ValueError("Malformed RMS bit matrix")
    return torch.tensor(rows, dtype=torch.uint16).view(torch.bfloat16).reshape(shape).to(device)


def check_rms_projection_source(program: dict[str, Any], projection_plan: dict[str, Any], projection_bundle: dict[str, Any], summary: dict[str, Any], plan: dict[str, Any] | None = None) -> None:
    from .gemma_ir_interpreter import _check_projection_plan, _projection_bits_tensor

    _check_projection_plan(program, projection_plan, projection_bundle)
    if _rms_sha({k: v for k, v in summary.items() if k != "summary_sha256"}) != summary.get("summary_sha256") or summary.get("plan_sha256") != projection_plan["plan_sha256"] or summary.get("slice_passes_declared_comparison") is not True:
        raise ValueError("RMS slice requires intact passing projection evidence")
    for role in ("query", "key", "value"):
        if summary["observed_output_hashes"][role] != [projection_plan["projection_records"][role]["prediction"]["sha256"]] * 3:
            raise ValueError("RMS source projection output binding mismatch")
    if plan is not None:
        if plan["source_projection_plan_sha256"] != projection_plan["plan_sha256"] or plan["source_projection_summary_sha256"] != summary["summary_sha256"]:
            raise ValueError("RMS projection source reference mismatch")
        hidden = next(record["payload"]["outputs"]["hidden.0"] for record in projection_plan["shared_prefix_records"] if "hidden.0" in record["payload"]["outputs"])
        if canonical_json(hidden) != canonical_json(plan["roles"]["input_norm"]["input"]):
            raise ValueError("RMS embedding/scale input differs from its projection source")
        for role, heads in (("query", 4), ("key", 1)):
            flat = _projection_bits_tensor(projection_bundle["prediction_bits"][role], [1, 30, heads * 256], "cpu")
            descriptor = tensor_descriptor(flat.view(1, 30, heads, 256).transpose(1, 2))
            descriptor["device"] = projection_plan["projection_records"][role]["prediction"]["device"]
            if canonical_json(descriptor) != canonical_json(plan["roles"][role + "_norm"]["input"]):
                raise ValueError("RMS head input differs from its bound projection output")


def _rms_slice_inputs(program: dict[str, Any], model: Any, projection_plan: dict[str, Any], projection_bundle: dict[str, Any], summary: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    from .gemma_ir_interpreter import bind_model_tensors, _projection_bits_tensor
    from .gemma_float_semantics import bfloat16_multiply_bits

    check_rms_projection_source(program, projection_plan, projection_bundle, summary)
    parameters = bind_model_tensors(program, model, verify_hashes=True)
    if model.training or model.config._attn_implementation != "eager" or type(model).__module__ + "." + type(model).__name__ != projection_plan["model_binding"]["class"] or _rms_sha(model.config.to_dict()) != projection_plan["model_binding"]["config_sha256"]:
        raise ValueError("RMS model configuration differs from the projection source")
    device = parameters["model.embed_tokens.weight"].device
    if device.type != "cuda" or device.index != torch.cuda.current_device():
        raise ValueError("RMS slice requires the current CUDA device")
    scale = parameters["model.embed_tokens.embed_scale"]
    if scale.dtype != torch.bfloat16 or scale.numel() != 1:
        raise ValueError("Independent embedding scale is restricted to the bound bfloat16 scalar")
    scale_bits = int(scale.detach().cpu().view(torch.uint16).item())
    embeddings = [parameters["model.embed_tokens.weight"][token].detach().cpu().view(torch.uint16).tolist() for token in projection_plan["input_token_ids"][0]]
    scaled = [[bfloat16_multiply_bits(bits, scale_bits) for bits in row] for row in embeddings]
    values = {"input_norm": _rms_rows_tensor(scaled, [1, 30, 640], device)}
    expected_hidden = next(record["payload"]["outputs"]["hidden.0"] for record in projection_plan["shared_prefix_records"] if "hidden.0" in record["payload"]["outputs"])
    if tensor_descriptor(values["input_norm"])["sha256"] != expected_hidden["sha256"]:
        raise ValueError("Independent embedding/scale differs from the source IR prefix")
    for role, heads in (("query", 4), ("key", 1)):
        flat = _projection_bits_tensor(projection_bundle["prediction_bits"][role], [1, 30, heads * 256], device)
        values[role + "_norm"] = flat.view(1, 30, heads, 256).transpose(1, 2)
    modules = {"input_norm": model.model.layers[0].input_layernorm, "query_norm": model.model.layers[0].self_attn.q_norm, "key_norm": model.model.layers[0].self_attn.k_norm}
    instructions = {}
    for role, weight_name, output in RMS_SLICE_ROLES:
        module = modules[role]
        instruction = next(item for item in program["instructions"] if item["outputs"] == [output])
        if instruction["opcode"] != "RMS_NORM" or instruction["parameter_refs"] != [weight_name] or module.eps != instruction["attributes"]["epsilon"] or module.weight is not parameters[weight_name] or module.weight.dtype != torch.bfloat16 or module.weight.device != device:
            raise ValueError("Original RMS module differs from its IR binding")
        instructions[role] = instruction
    return values, modules, instructions


def build_rms_slice(program: dict[str, Any], model: Any, projection_plan: dict[str, Any], projection_bundle: dict[str, Any], summary: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    from pathlib import Path
    from .gemma_ir_interpreter import _projection_environment

    values, modules, instructions = _rms_slice_inputs(program, model, projection_plan, projection_bundle, summary)
    profiles = [{"id": reduction + ":" + root, "reduction": reduction, "root": root} for reduction in RMS_REDUCTIONS for root in RMS_ROOTS]
    records, bundles = {}, {}
    for role, weight_name, output_name in RMS_SLICE_ROLES:
        value = values[role]
        width = value.shape[-1]
        rows = value.detach().cpu().reshape(-1, width).view(torch.uint16).tolist()
        weights = modules[role].weight.detach().cpu().view(torch.uint16).tolist()
        predicted, commitments = {}, {}
        for profile in profiles:
            calculations = [rms_row_candidate(row, weights, modules[role].eps, profile["reduction"], profile["root"]) for row in rows]
            result = {key: [item[key] for item in calculations] for key in ("mean_bits", "denominator_bits", "rsqrt_bits", "output_bits")}
            predicted[profile["id"]] = result
            output = _rms_rows_tensor(result["output_bits"], list(value.shape), value.device)
            commitments[profile["id"]] = {"prediction_sha256": _rms_sha(result), "output": tensor_descriptor(output)}
        bundles[role] = {"input_bits": rows, "weight_bits": weights, "predictions": predicted}
        records[role] = {"input": tensor_descriptor(value), "input_strides": list(value.stride()), "weight_name": weight_name,
                         "weight": tensor_descriptor(modules[role].weight), "epsilon": modules[role].eps,
                         "instruction_id": instructions[role]["id"], "instruction_sha256": instructions[role]["instruction_sha256"],
                         "output_tensor": output_name, "profiles": commitments}
    include = Path(torch.__file__).resolve().parent / "include/ATen/native"
    headers = {name: hashlib.sha256((include / name).read_bytes()).hexdigest() for name in ("cuda/Reduce.cuh", "SharedReduceOps.h")}
    bundle = {"program_sha256": program["program_sha256"], "roles": bundles}
    body = {"schema_version": 1, "scope": RMS_SLICE_SCOPE, "program_sha256": program["program_sha256"],
            "source_projection_plan_sha256": projection_plan["plan_sha256"], "source_projection_summary_sha256": summary["summary_sha256"],
            "source_validation": "projection bundle and summary integrity; use projection-slice-verify --reexecute for full source replay",
            "independent_embedding_scale_matches_source": True, "roles": records, "profiles": profiles,
            "arithmetic_implementation_sha256": _rms_code_commitment(), "installed_header_sha256": headers,
            "source_reference": "PyTorch v2.7.1 ReduceMomentKernel.cu mean factor is float32(num_outputs)/numel",
            "template_assumptions": {"input_vector_size": 4, "block_width": 32, "block_height": 16, "shuffle_offsets": [1, 2, 4, 8, 16], "cross_warp_or_global_reduction": False},
            "runtime": _projection_environment(), "bundle_sha256": _rms_sha(bundle), "compared_values_per_profile": 57600,
            "candidate_refitting_allowed": False, "linked_binary_correspondence_established": False,
            "full_first_layer_qualified": False, "global_exactness_activation_allowed": False}
    return {**body, "plan_sha256": _rms_sha(body)}, bundle


def _rms_tensor_f32_bits(value: Any) -> list[int]:
    if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
        raise ValueError("Observed RMS stage must be float32")
    with torch._C._DisableTorchDispatch():
        return [int(bits) & 0xFFFFFFFF for bits in value.detach().contiguous().view(torch.int32).cpu().reshape(-1).tolist()]


def _observe_rms_module(module: Any, value: Any) -> tuple[Any, dict[str, Any]]:
    from torch.utils._python_dispatch import TorchDispatchMode

    records = {"mean": [], "rsqrt": []}

    class Capture(TorchDispatchMode):
        def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
            output = func(*args, **(kwargs or {}))
            if str(func) == "aten.mean.dim":
                records["mean"].append({"bits": _rms_tensor_f32_bits(output), "input_shape": list(args[0].shape),
                                         "input_strides": list(args[0].stride()), "alignment_mod16": args[0].data_ptr() % 16,
                                         "input_dtype": str(args[0].dtype), "axes": list(args[1]), "keepdim": args[2]})
            elif str(func) == "aten.rsqrt.default":
                records["rsqrt"].append({"input_bits": _rms_tensor_f32_bits(args[0]), "output_bits": _rms_tensor_f32_bits(output)})
            return output

    with Capture():
        output = module(value)
    if len(records["mean"]) != 1 or len(records["rsqrt"]) != 1:
        raise ValueError("Original RMS module did not expose one mean and one rsqrt operation")
    mean, root = records["mean"][0], records["rsqrt"][0]
    return output, {"mean_bits": mean.pop("bits"), "denominator_bits": root["input_bits"], "rsqrt_bits": root["output_bits"], "mean_input_metadata": mean}


def _check_rms_plan(program: dict[str, Any], plan: dict[str, Any], bundle: dict[str, Any]) -> None:
    from .gemma_ir import verify_gemma_ir

    if not verify_gemma_ir(program)["valid"] or plan["program_sha256"] != program["program_sha256"] or bundle["program_sha256"] != program["program_sha256"]:
        raise ValueError("RMS program identity mismatch")
    if _rms_sha({k: v for k, v in plan.items() if k != "plan_sha256"}) != plan["plan_sha256"] or _rms_sha(bundle) != plan["bundle_sha256"] or plan["arithmetic_implementation_sha256"] != _rms_code_commitment():
        raise ValueError("RMS plan, bundle, or arithmetic commitment mismatch")
    profiles = [{"id": reduction + ":" + root, "reduction": reduction, "root": root} for reduction in RMS_REDUCTIONS for root in RMS_ROOTS]
    if canonical_json(plan["profiles"]) != canonical_json(profiles) or set(plan["roles"]) != {role for role, _, _ in RMS_SLICE_ROLES} or set(bundle["roles"]) != set(plan["roles"]):
        raise ValueError("RMS profile or role coverage mismatch")
    assumptions = {"input_vector_size": 4, "block_width": 32, "block_height": 16, "shuffle_offsets": [1, 2, 4, 8, 16], "cross_warp_or_global_reduction": False}
    if plan["scope"] != RMS_SLICE_SCOPE or plan["compared_values_per_profile"] != 57600 or canonical_json(plan["template_assumptions"]) != canonical_json(assumptions) or any(plan.get(key) is not False for key in ("candidate_refitting_allowed", "linked_binary_correspondence_established", "full_first_layer_qualified", "global_exactness_activation_allowed")):
        raise ValueError("RMS scope overclaim")
    for role, weight_name, output in RMS_SLICE_ROLES:
        record, data = plan["roles"][role], bundle["roles"][role]
        instruction = next(item for item in program["instructions"] if item["outputs"] == [output])
        if record["instruction_sha256"] != instruction["instruction_sha256"] or record["weight_name"] != weight_name or record["weight"]["sha256"] != program["parameter_commitments"][weight_name]["sha256"]:
            raise ValueError("RMS instruction or weight binding mismatch")
        shape = record["input"]["shape"]
        expected_shape = {"input_norm": [1, 30, 640], "query_norm": [1, 4, 30, 256], "key_norm": [1, 1, 30, 256]}[role]
        if canonical_json(shape) != canonical_json(expected_shape) or record["input"]["dtype"] != "torch.bfloat16" or record["epsilon"] != instruction["attributes"]["epsilon"] or record["instruction_id"] != instruction["id"]:
            raise ValueError("RMS shape, dtype, epsilon, or instruction identity mismatch")
        input_descriptor = tensor_descriptor(_rms_rows_tensor(data["input_bits"], shape, "cpu"))
        if any(record["input"].get(key) != value for key, value in input_descriptor.items() if key != "device"):
            raise ValueError("RMS input tensor or metadata mismatch")
        if not isinstance(data["weight_bits"], list) or len(data["weight_bits"]) != shape[-1] or any(type(bits) is not int or not 0 <= bits <= 65535 for bits in data["weight_bits"]):
            raise ValueError("Malformed RMS weight bits")
        weight_descriptor = tensor_descriptor(torch.tensor(data["weight_bits"], dtype=torch.uint16).view(torch.bfloat16))
        if any(record["weight"].get(key) != value for key, value in weight_descriptor.items() if key != "device"):
            raise ValueError("RMS weight bits or metadata mismatch")
        if set(data["predictions"]) != {profile["id"] for profile in profiles}:
            raise ValueError("Missing RMS candidate predictions")
        for profile in profiles:
            predicted = data["predictions"][profile["id"]]
            if any(not isinstance(predicted.get(key), list) or len(predicted[key]) != math.prod(shape[:-1]) or any(type(bits) is not int or not 0 <= bits <= 0xFFFFFFFF for bits in predicted[key]) for key in ("mean_bits", "denominator_bits", "rsqrt_bits")):
                raise ValueError("Malformed RMS predicted float32 stages")
            descriptor = tensor_descriptor(_rms_rows_tensor(predicted["output_bits"], shape, "cpu"))
            if _rms_sha(predicted) != record["profiles"][profile["id"]]["prediction_sha256"] or any(record["profiles"][profile["id"]]["output"].get(key) != value for key, value in descriptor.items() if key != "device"):
                raise ValueError("RMS prediction commitment or metadata mismatch")


def _rms_slice_report(plan: dict[str, Any], bundle: dict[str, Any], observations: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    if set(observations) != set(plan["roles"]):
        raise ValueError("Missing RMS observations")
    comparisons = {}
    inputs_match, geometry_match = True, True
    for role, _, _ in RMS_SLICE_ROLES:
        record, actual = plan["roles"][role], observations[role]
        inputs_match = inputs_match and canonical_json(record["input"]) == canonical_json(actual["input"])
        if not isinstance(actual["repetitions"], list) or len(actual["repetitions"]) != 3:
            raise ValueError("Expected three original RMS module repetitions")
        rows, width = math.prod(record["input"]["shape"][:-1]), record["input"]["shape"][-1]
        for observed in actual["repetitions"]:
            metadata = observed["mean_input_metadata"]
            geometry_match = geometry_match and metadata["input_shape"] == record["input"]["shape"] and metadata["input_dtype"] == "torch.float32" and metadata["alignment_mod16"] == 0 and metadata["input_strides"][-1] == 1 and all(stride % 4 == 0 for stride in metadata["input_strides"][:-1]) and metadata["axes"] in ([-1], [len(record["input"]["shape"]) - 1]) and metadata["keepdim"] is True
            if not isinstance(observed.get("profiled_cuda_event_names"), list) or not observed["profiled_cuda_event_names"] or any(not isinstance(name, str) or not name for name in observed["profiled_cuda_event_names"]):
                raise ValueError("Missing RMS CUDA activity provenance")
        result = {}
        for profile in plan["profiles"]:
            prediction = bundle["roles"][role]["predictions"][profile["id"]]
            failures, stage_counts = [], {key: [] for key in ("mean_bits", "denominator_bits", "rsqrt_bits")}
            for repetition, observed in enumerate(actual["repetitions"]):
                _rms_rows_tensor(observed["output_bits"], record["input"]["shape"], "cpu")
                for key in stage_counts:
                    bits = observed[key]
                    if not isinstance(bits, list) or len(bits) != rows or any(type(value) is not int or not 0 <= value <= 0xFFFFFFFF for value in bits):
                        raise ValueError("Malformed observed RMS float32 stage")
                    stage_counts[key].append(sum(a != b for a, b in zip(bits, prediction[key])))
                failures.extend({"repetition": repetition, "flat_row": row, "column": column, "predicted_bits": predicted, "observed_bits": observed["output_bits"][row][column]}
                                for row, values in enumerate(prediction["output_bits"]) for column, predicted in enumerate(values) if predicted != observed["output_bits"][row][column])
            result[profile["id"]] = {"output_mismatch_count": len(failures), "output_mismatches": failures, "stage_mismatch_counts": stage_counts,
                                      "first_differing_observed_stage": next((key for key in stage_counts if any(stage_counts[key])), "output_bits" if failures else None)}
        comparisons[role] = result
    output_matching = [profile["id"] for profile in plan["profiles"] if all(comparisons[role][profile["id"]]["output_mismatch_count"] == 0 for role in comparisons)]
    fully_matching = [name for name in output_matching if all(not any(counts) for role in comparisons for counts in comparisons[role][name]["stage_mismatch_counts"].values())]
    body = {"schema_version": 1, "scope": RMS_SLICE_SCOPE, "plan_sha256": plan["plan_sha256"], "bundle_sha256": plan["bundle_sha256"],
            "observations": observations, "runtime": runtime, "actual_inputs_match_plan": inputs_match,
            "runtime_matches_plan": canonical_json(runtime) == canonical_json(plan["runtime"]), "comparisons": comparisons,
            "output_matching_profiles": output_matching, "fully_matching_profiles": fully_matching,
            "source_template_geometry_matches": geometry_match,
            "all_stages_candidate_passes": bool(fully_matching) and inputs_match and geometry_match and canonical_json(runtime) == canonical_json(plan["runtime"]),
            "observed_stage_boundary": "actual module ATen mean output and rsqrt input/output, not instrumented hardware registers",
            "fresh_holdout_validation_established": False, "full_first_layer_qualified": False,
            "hardware_semantics_established": False, "global_exactness_activation_allowed": False}
    return {**body, "report_sha256": _rms_sha(body)}


def acquire_rms_slice(program: dict[str, Any], model: Any, projection_plan: dict[str, Any], projection_bundle: dict[str, Any], summary: dict[str, Any], plan: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    from .gemma_ir_interpreter import _projection_environment, bind_model_tensors
    from .gemma_reduction_backend import _profile_call

    _check_rms_plan(program, plan, bundle)
    values, modules, _ = _rms_slice_inputs(program, model, projection_plan, projection_bundle, summary)
    if plan["source_projection_plan_sha256"] != projection_plan["plan_sha256"] or plan["source_projection_summary_sha256"] != summary["summary_sha256"]:
        raise ValueError("RMS projection source mismatch")
    for role in values:
        if tensor_descriptor(values[role]) != plan["roles"][role]["input"]:
            raise ValueError("RMS regenerated input differs from the prediction plan")
    observations, normalized = {}, None
    with torch.no_grad():
        ids = torch.tensor(projection_plan["input_token_ids"], dtype=torch.int64, device=values["input_norm"].device)
        actual_inputs = {"input_norm": model.model.embed_tokens(ids)}
        for role, _, _ in RMS_SLICE_ROLES:
            if role != "input_norm":
                short, heads = ("q", 4) if role == "query_norm" else ("k", 1)
                flat = getattr(model.model.layers[0].self_attn, short + "_proj")(normalized)
                actual_inputs[role] = flat.view(1, 30, heads, 256).transpose(1, 2)
            value, module = actual_inputs[role], modules[role]
            captured, repetitions = [], []

            def invoke() -> Any:
                output, stages = _observe_rms_module(module, value)
                captured[:] = [stages]
                return output

            for _ in range(3):
                output, events = _profile_call(invoke)
                if output.dtype != torch.bfloat16 or output.shape != value.shape:
                    raise ValueError("Original RMS output type mismatch")
                rows = output.detach().cpu().reshape(-1, output.shape[-1]).view(torch.uint16).tolist()
                repetitions.append({**captured[0], "output_bits": rows, "profiled_cuda_event_names": events})
                if role == "input_norm" and normalized is None:
                    normalized = output.detach()
            observations[role] = {"input": tensor_descriptor(value), "repetitions": repetitions}
    bind_model_tensors(program, model, verify_hashes=True)
    return _rms_slice_report(plan, bundle, observations, _projection_environment())


def rms_observer_controls(program: dict[str, Any], model: Any, projection_plan: dict[str, Any], projection_bundle: dict[str, Any], summary: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    values, modules, _ = _rms_slice_inputs(program, model, projection_plan, projection_bundle, summary)
    checks = {}
    with torch.no_grad():
        for role, value in values.items():
            output = modules[role](value)
            rows = output.detach().cpu().reshape(-1, output.shape[-1]).view(torch.uint16).tolist()
            mean = value.float().pow(2).mean(-1, keepdim=True)
            denominator = mean + modules[role].eps
            inverse = denominator.rsqrt()
            stages = {"mean_bits": _rms_tensor_f32_bits(mean), "denominator_bits": _rms_tensor_f32_bits(denominator), "rsqrt_bits": _rms_tensor_f32_bits(inverse)}
            checks[role] = {"untraced_module_outputs_match": all(rows == item["output_bits"] for item in report["observations"][role]["repetitions"]),
                            "untraced_aten_stage_reproduction_matches": all(stages[key] == item[key] for item in report["observations"][role]["repetitions"] for key in stages)}
    return checks


def verify_rms_slice(program: dict[str, Any], plan: dict[str, Any], bundle: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    try:
        _check_rms_plan(program, plan, bundle)
        expected = _rms_slice_report(plan, bundle, report["observations"], report["runtime"])
        return {"valid": canonical_json(expected) == canonical_json(report), "mode": "integrity_only",
                "output_matching_profiles": expected["output_matching_profiles"], "fully_matching_profiles": expected["fully_matching_profiles"],
                "all_stages_candidate_passes": expected["all_stages_candidate_passes"], "actual_inputs_match_plan": expected["actual_inputs_match_plan"],
                "global_exactness_activation_allowed": False}
    except (KeyError, IndexError, TypeError, ValueError, RuntimeError, AttributeError, StopIteration) as error:
        return {"valid": False, "reason": str(error)}


def rms_slice_summary(plan: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in report.items() if key not in ("observations", "report_sha256", "comparisons")}
    body.update({"source_report_sha256": report["report_sha256"], "program_sha256": plan["program_sha256"],
                 "profiles": plan["profiles"], "roles": plan["roles"], "installed_header_sha256": plan["installed_header_sha256"],
                 "comparisons": {role: {name: {key: value for key, value in result.items() if key != "output_mismatches"} for name, result in candidates.items()} for role, candidates in report["comparisons"].items()},
                 "profiled_cuda_event_names": {role: [item["profiled_cuda_event_names"] for item in actual["repetitions"]] for role, actual in report["observations"].items()}})
    return {**body, "summary_sha256": _rms_sha(body)}


def reference_rotate_half(value: Any) -> Any:
    half = value.shape[-1] // 2
    return torch.cat((-value[..., half:], value[..., :half]), dim=-1)


def reference_rotary(value: Any, position_ids: Any, inverse_frequency: Any, scaling: float = 1.0) -> tuple[Any, Any]:
    expanded_frequency = inverse_frequency[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(value.device)
    expanded_position = position_ids[:, None, :].float()
    frequency = (expanded_frequency @ expanded_position).transpose(1, 2)
    embedding = torch.cat((frequency, frequency), dim=-1)
    return (embedding.cos() * scaling).to(value.dtype), (embedding.sin() * scaling).to(value.dtype)


def reference_apply_rotary(query: Any, key: Any, cosine: Any, sine: Any) -> tuple[Any, Any]:
    cosine = cosine.unsqueeze(1)
    sine = sine.unsqueeze(1)
    return query * cosine + reference_rotate_half(query) * sine, key * cosine + reference_rotate_half(key) * sine


def reference_repeat_kv(value: Any, repetitions: int) -> Any:
    if repetitions == 1:
        return value
    batch, heads, sequence, dimension = value.shape
    expanded = value[:, :, None, :, :].expand(batch, heads, repetitions, sequence, dimension)
    return expanded.reshape(batch, heads * repetitions, sequence, dimension)


def reference_causal_mask(sequence_length: int, dtype: Any, device: Any, sliding_window: int | None = None) -> Any:
    row = torch.arange(sequence_length, device=device)[:, None]
    column = torch.arange(sequence_length, device=device)[None, :]
    allowed = column <= row
    if sliding_window is not None:
        allowed = allowed & (column > row - sliding_window)
    zero = torch.zeros((sequence_length, sequence_length), dtype=dtype, device=device)
    blocked = torch.full_like(zero, torch.finfo(dtype).min)
    return torch.where(allowed, zero, blocked)[None, None, :, :]


def _record(records: list[dict[str, Any]] | None, name: str, tensor: Any, layer: int | None = None) -> None:
    if records is None:
        return
    append_chain_record(
        records,
        {
            "stage": name,
            "layer": layer,
            "tensor": tensor_descriptor(tensor),
        },
    )


def reference_gemma_forward(
    model: Any,
    input_ids: Any,
    record_provenance: bool = True,
    compare_attention_kernels: bool = True,
    capture_boundaries: bool = True,
    last_token_only: bool = False,
) -> ReferenceOutput:
    _require_torch()
    config = model.config
    text_model = model.model
    if compare_attention_kernels and any(
        layer.self_attn.attn_logit_softcapping is not None for layer in text_model.layers
    ):
        raise ValueError("Direct SDPA comparison is undefined for attention-logit-softcapped configurations")
    records: list[dict[str, Any]] | None = [] if record_provenance else None
    attention_kernel_comparisons = []
    hidden_states = functional.embedding(input_ids, text_model.embed_tokens.weight)
    hidden_states = hidden_states * text_model.embed_tokens.embed_scale.to(text_model.embed_tokens.weight.dtype)
    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    global_cosine, global_sine = reference_rotary(
        hidden_states,
        position_ids,
        text_model.rotary_emb.inv_freq,
        float(text_model.rotary_emb.attention_scaling),
    )
    local_cosine, local_sine = reference_rotary(
        hidden_states,
        position_ids,
        text_model.rotary_emb_local.inv_freq,
        float(text_model.rotary_emb_local.attention_scaling),
    )
    full_mask = reference_causal_mask(input_ids.shape[1], hidden_states.dtype, hidden_states.device)
    sliding_mask = reference_causal_mask(
        input_ids.shape[1], hidden_states.dtype, hidden_states.device, config.sliding_window
    )
    boundaries = [hidden_states] if capture_boundaries else []
    _record(records, "embedding", hidden_states)
    _record(records, "global_rotary_cosine", global_cosine)
    _record(records, "global_rotary_sine", global_sine)
    _record(records, "local_rotary_cosine", local_cosine)
    _record(records, "local_rotary_sine", local_sine)

    for layer_index, layer in enumerate(text_model.layers):
        residual = hidden_states
        normalized = reference_rms_norm(hidden_states, layer.input_layernorm.weight, layer.input_layernorm.eps)
        _record(records, "attention_input_norm", normalized, layer_index)
        batch, sequence, _ = normalized.shape
        head_dimension = layer.self_attn.head_dim
        query = functional.linear(normalized, layer.self_attn.q_proj.weight)
        key = functional.linear(normalized, layer.self_attn.k_proj.weight)
        value = functional.linear(normalized, layer.self_attn.v_proj.weight)
        query = query.view(batch, sequence, config.num_attention_heads, head_dimension).transpose(1, 2)
        key = key.view(batch, sequence, config.num_key_value_heads, head_dimension).transpose(1, 2)
        value = value.view(batch, sequence, config.num_key_value_heads, head_dimension).transpose(1, 2)
        query = reference_rms_norm(query, layer.self_attn.q_norm.weight, layer.self_attn.q_norm.eps)
        key = reference_rms_norm(key, layer.self_attn.k_norm.weight, layer.self_attn.k_norm.eps)
        cosine, sine = (local_cosine, local_sine) if layer.self_attn.is_sliding else (global_cosine, global_sine)
        query, key = reference_apply_rotary(query, key, cosine, sine)
        _record(records, "query", query, layer_index)
        _record(records, "key", key, layer_index)
        _record(records, "value", value, layer_index)
        repeated_key = reference_repeat_kv(key, layer.self_attn.num_key_value_groups)
        repeated_value = reference_repeat_kv(value, layer.self_attn.num_key_value_groups)
        scores = torch.matmul(query, repeated_key.transpose(2, 3)) * layer.self_attn.scaling
        if layer.self_attn.attn_logit_softcapping is not None:
            cap = layer.self_attn.attn_logit_softcapping
            scores = torch.tanh(scores / cap) * cap
        mask = sliding_mask if layer.self_attn.is_sliding else full_mask
        scores = scores + mask
        probability = functional.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        eager_attention_heads = torch.matmul(probability, repeated_value)
        if compare_attention_kernels:
            sdpa_attention_heads = functional.scaled_dot_product_attention(
                query.contiguous(),
                repeated_key.contiguous(),
                repeated_value.contiguous(),
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=False,
                scale=layer.self_attn.scaling,
            )
            kernel_comparison = _difference(eager_attention_heads, sdpa_attention_heads)
            kernel_comparison.update({"layer": layer_index, "attention_type": layer.attention_type})
            attention_kernel_comparisons.append(kernel_comparison)
        attention_heads = eager_attention_heads.transpose(1, 2).contiguous()
        attention_concatenated = attention_heads.reshape(batch, sequence, -1).contiguous()
        attention_output = functional.linear(attention_concatenated, layer.self_attn.o_proj.weight)
        _record(records, "attention_scores", scores, layer_index)
        _record(records, "attention_probability", probability, layer_index)
        _record(records, "attention_concatenated_heads", attention_concatenated, layer_index)
        _record(records, "attention_output_projection", attention_output, layer_index)
        attention_output = reference_rms_norm(
            attention_output, layer.post_attention_layernorm.weight, layer.post_attention_layernorm.eps
        )
        hidden_states = residual + attention_output
        _record(records, "post_attention_residual", hidden_states, layer_index)
        residual = hidden_states
        mlp_input = reference_rms_norm(
            hidden_states, layer.pre_feedforward_layernorm.weight, layer.pre_feedforward_layernorm.eps
        )
        gate = functional.linear(mlp_input, layer.mlp.gate_proj.weight)
        up = functional.linear(mlp_input, layer.mlp.up_proj.weight)
        activated_gate = functional.gelu(gate, approximate="tanh")
        intermediate = activated_gate * up
        mlp_output = functional.linear(intermediate, layer.mlp.down_proj.weight)
        _record(records, "mlp_input_norm", mlp_input, layer_index)
        _record(records, "mlp_gate", gate, layer_index)
        _record(records, "mlp_up", up, layer_index)
        _record(records, "mlp_activated_gate", activated_gate, layer_index)
        _record(records, "mlp_intermediate_product", intermediate, layer_index)
        _record(records, "mlp_down_projection", mlp_output, layer_index)
        mlp_output = reference_rms_norm(
            mlp_output, layer.post_feedforward_layernorm.weight, layer.post_feedforward_layernorm.eps
        )
        hidden_states = residual + mlp_output
        _record(records, "post_mlp_residual", hidden_states, layer_index)
        if capture_boundaries and layer_index < len(text_model.layers) - 1:
            boundaries.append(hidden_states)

    hidden_states = reference_rms_norm(hidden_states, text_model.norm.weight, text_model.norm.eps)
    if capture_boundaries:
        boundaries.append(hidden_states)
    _record(records, "final_norm", hidden_states)
    logits_input = hidden_states[:, -1:, :] if last_token_only else hidden_states
    logits = functional.linear(logits_input, model.lm_head.weight)
    if config.final_logit_softcapping is not None:
        cap = config.final_logit_softcapping
        logits = torch.tanh(logits / cap) * cap
    _record(records, "vocabulary_logits", logits)
    return ReferenceOutput(
        logits=logits,
        hidden_states=tuple(boundaries),
        records=tuple(records or ()),
        attention_kernel_comparisons=tuple(attention_kernel_comparisons),
    )


def _difference(reference: Any, implementation: Any) -> dict[str, Any]:
    difference = (reference.float() - implementation.float()).abs()
    return {
        "reference": tensor_descriptor(reference),
        "implementation": tensor_descriptor(implementation),
        "exact_equal": bool(torch.equal(reference, implementation)),
        "max_absolute_error": float(difference.max()),
        "mean_absolute_error": float(difference.mean()),
    }


def fixed_input_equivalence_certificate(model: Any, input_ids: Any, absolute_tolerance: float = 0.0) -> dict[str, Any]:
    _require_torch()
    original_attention = model.config._attn_implementation
    model.config._attn_implementation = "eager"
    try:
        with torch.no_grad():
            implementation = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                output_hidden_states=True,
            )
            reference = reference_gemma_forward(model, input_ids)
    finally:
        model.config._attn_implementation = original_attention
    with torch.no_grad():
        deployed = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
            output_hidden_states=True,
        )
    implementation_boundaries = implementation.hidden_states
    if len(reference.hidden_states) != len(implementation_boundaries):
        raise RuntimeError("Reference and implementation boundary counts differ")
    records = []
    boundary_results = []
    for index, (reference_state, implementation_state) in enumerate(zip(reference.hidden_states, implementation_boundaries)):
        result = _difference(reference_state, implementation_state)
        result["boundary"] = "embedding" if index == 0 else ("final_norm" if index == len(reference.hidden_states) - 1 else f"layer_{index - 1}")
        append_chain_record(records, result)
        boundary_results.append(result)
    logits_result = _difference(reference.logits, implementation.logits)
    append_chain_record(records, {"boundary": "logits", **logits_result})
    deployed_boundaries = [
        {"boundary": boundary_results[index]["boundary"], **_difference(reference_state, deployed_state)}
        for index, (reference_state, deployed_state) in enumerate(zip(reference.hidden_states, deployed.hidden_states))
    ]
    deployed_logits = _difference(reference.logits, deployed.logits)
    reference_token = int(torch.argmax(reference.logits[0, -1]))
    implementation_token = int(torch.argmax(implementation.logits[0, -1]))
    deployed_token = int(torch.argmax(deployed.logits[0, -1]))
    all_within_tolerance = all(result["max_absolute_error"] <= absolute_tolerance for result in boundary_results)
    all_within_tolerance = all_within_tolerance and logits_result["max_absolute_error"] <= absolute_tolerance
    root = records[-1]["record_hash"] if records else "GENESIS"
    body = {
        "scope": "Machine-checked numerical equivalence for one fixed input between independent Python orchestration and Hugging Face eager execution; not a universal proof.",
        "absolute_tolerance": absolute_tolerance,
        "model_state_sha256": model_state_sha256(model),
        "input_ids": input_ids[0].tolist(),
        "input_ids_sha256": hashlib.sha256(canonical_json(input_ids[0].tolist()).encode("utf-8")).hexdigest(),
        "reference_trace_root_sha256": reference.records[-1]["record_hash"] if reference.records else "GENESIS",
        "reference_trace_records": list(reference.records),
        "reference_trace_chain_verification": verify_trace_chain(
            list(reference.records), reference.records[-1]["record_hash"] if reference.records else "GENESIS"
        ),
        "attention_kernel_comparisons": list(reference.attention_kernel_comparisons),
        "boundaries": boundary_results,
        "logits": logits_result,
        "reference_selected_token_id": reference_token,
        "implementation_selected_token_id": implementation_token,
        "selected_token_exact_match": reference_token == implementation_token,
        "all_boundaries_within_tolerance": all_within_tolerance,
        "deployed_path": {
            "attention_implementation": original_attention,
            "boundaries": deployed_boundaries,
            "logits": deployed_logits,
            "selected_token_id": deployed_token,
            "selected_token_matches_reference": deployed_token == reference_token,
        },
        "certificate_records": records,
        "certificate_root_sha256": root,
        "certificate_chain_verification": verify_trace_chain(records, root),
    }
    body["certificate_sha256"] = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    return body


def verify_fixed_input_certificate(certificate: dict[str, Any]) -> dict[str, Any]:
    records = certificate.get("certificate_records", [])
    boundaries = certificate.get("boundaries", [])
    logits = certificate.get("logits", {})
    chain = verify_trace_chain(records, certificate.get("certificate_root_sha256"))
    reference_chain = verify_trace_chain(
        certificate.get("reference_trace_records", []), certificate.get("reference_trace_root_sha256")
    )
    original_hash = certificate.get("certificate_sha256")
    body = {key: value for key, value in certificate.items() if key != "certificate_sha256"}
    calculated_hash = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    tolerance = certificate.get("absolute_tolerance")
    tolerance_valid = isinstance(tolerance, (int, float)) and math.isfinite(tolerance) and tolerance >= 0
    calculated_boundaries_within = bool(
        tolerance_valid
        and boundaries
        and all(isinstance(item.get("max_absolute_error"), (int, float)) and item["max_absolute_error"] <= tolerance for item in boundaries)
        and isinstance(logits.get("max_absolute_error"), (int, float))
        and logits["max_absolute_error"] <= tolerance
    )
    calculated_token_match = certificate.get("reference_selected_token_id") == certificate.get("implementation_selected_token_id")
    deployed = certificate.get("deployed_path", {})
    calculated_deployed_token_match = certificate.get("reference_selected_token_id") == deployed.get("selected_token_id")
    deployed_claims_consistent = bool(
        deployed.get("selected_token_matches_reference") == calculated_deployed_token_match
        and deployed.get("logits", {}).get("exact_equal")
        == (
            deployed.get("logits", {}).get("reference", {}).get("sha256")
            == deployed.get("logits", {}).get("implementation", {}).get("sha256")
            and deployed.get("logits", {}).get("max_absolute_error") == 0.0
        )
        and all(
            item.get("exact_equal")
            == (
                item.get("reference", {}).get("sha256") == item.get("implementation", {}).get("sha256")
                and item.get("max_absolute_error") == 0.0
            )
            for item in deployed.get("boundaries", [])
        )
    )
    expected_payloads = [*boundaries, {"boundary": "logits", **logits}]
    chain_payloads_match = len(records) == len(expected_payloads) and all(
        record.get("payload") == payload for record, payload in zip(records, expected_payloads)
    )
    claims_consistent = bool(
        certificate.get("selected_token_exact_match") == calculated_token_match
        and certificate.get("all_boundaries_within_tolerance") == calculated_boundaries_within
    )
    valid = bool(
        chain["valid"]
        and reference_chain["valid"]
        and original_hash == calculated_hash
        and chain_payloads_match
        and claims_consistent
        and deployed_claims_consistent
        and calculated_token_match
        and calculated_deployed_token_match
        and calculated_boundaries_within
    )
    return {
        "valid": valid,
        "certificate_hash_match": original_hash == calculated_hash,
        "chain": chain,
        "reference_trace_chain": reference_chain,
        "chain_payloads_match": chain_payloads_match,
        "claims_consistent": claims_consistent,
        "deployed_claims_consistent": deployed_claims_consistent,
        "selected_token_exact_match": calculated_token_match,
        "deployed_selected_token_match": calculated_deployed_token_match,
        "all_boundaries_within_tolerance": calculated_boundaries_within,
    }


def recompute_fixed_input_certificate(model: Any, certificate: dict[str, Any]) -> dict[str, Any]:
    integrity = verify_fixed_input_certificate(certificate)
    supplied_model_hash = model_state_sha256(model)
    model_matches = supplied_model_hash == certificate.get("model_state_sha256")
    if not model_matches:
        return {
            "valid": False,
            "integrity": integrity,
            "model_matches": False,
            "supplied_model_state_sha256": supplied_model_hash,
            "reason": "Supplied model does not match the certificate model commitment",
        }
    input_ids = torch.tensor(certificate["input_ids"], dtype=torch.long, device=next(model.parameters()).device).unsqueeze(0)
    recomputed = fixed_input_equivalence_certificate(model, input_ids, certificate["absolute_tolerance"])
    recomputed_integrity = verify_fixed_input_certificate(recomputed)
    claims_match = bool(
        certificate.get("input_ids_sha256") == recomputed.get("input_ids_sha256")
        and certificate.get("reference_selected_token_id") == recomputed.get("reference_selected_token_id")
        and certificate.get("implementation_selected_token_id") == recomputed.get("implementation_selected_token_id")
        and certificate.get("selected_token_exact_match") == recomputed.get("selected_token_exact_match")
        and certificate.get("all_boundaries_within_tolerance") == recomputed.get("all_boundaries_within_tolerance")
        and certificate.get("logits", {}).get("exact_equal") == recomputed.get("logits", {}).get("exact_equal")
        and certificate.get("deployed_path", {}).get("selected_token_id")
        == recomputed.get("deployed_path", {}).get("selected_token_id")
        and certificate.get("deployed_path", {}).get("selected_token_matches_reference")
        == recomputed.get("deployed_path", {}).get("selected_token_matches_reference")
    )
    exact_match = canonical_json(recomputed) == canonical_json(certificate)
    return {
        "valid": bool(integrity["valid"] and recomputed_integrity["valid"] and model_matches and claims_match),
        "integrity": integrity,
        "recomputed_integrity": recomputed_integrity,
        "model_matches": model_matches,
        "claims_match": claims_match,
        "reexecution_exact_match": exact_match,
        "recomputed_certificate_sha256": recomputed["certificate_sha256"],
    }


CANONICAL_OXYGEN_TEMPLATE = (
    "Dissolved oxygen: {oxygen_percent} percent. "
    "Slope: {slope_percent_per_minute} percent per minute. "
    "Sensor agreement: {sensor_agreement}. Does this require review? Answer Yes or No."
)


def canonical_oxygen_prompt(oxygen_percent: float, slope_percent_per_minute: float, sensor_agreement: bool) -> str:
    return CANONICAL_OXYGEN_TEMPLATE.format(
        oxygen_percent=format(oxygen_percent, "g"),
        slope_percent_per_minute=format(slope_percent_per_minute, "g"),
        sensor_agreement="true" if sensor_agreement else "false",
    )


def bounded_domain_equivalence_certificate(
    model: Any,
    tokenizer: Any,
    oxygen_values: tuple[float, ...],
    slope_values: tuple[float, ...],
    sensor_agreement_values: tuple[bool, ...] = (False, True),
) -> dict[str, Any]:
    _require_torch()
    if not oxygen_values or not slope_values or not sensor_agreement_values:
        raise ValueError("Every bounded-domain axis must contain at least one value")
    numeric_values = (*oxygen_values, *slope_values)
    if any(not math.isfinite(value) for value in numeric_values):
        raise ValueError("Bounded-domain numeric values must be finite")
    if len(set(oxygen_values)) != len(oxygen_values) or len(set(slope_values)) != len(slope_values):
        raise ValueError("Bounded-domain axes cannot contain duplicate values")
    if any(not isinstance(value, bool) for value in sensor_agreement_values) or len(set(sensor_agreement_values)) != len(sensor_agreement_values):
        raise ValueError("Sensor-agreement axis must contain unique boolean values")
    original_attention = model.config._attn_implementation
    records = []
    states = []
    try:
        for oxygen in oxygen_values:
            for slope in slope_values:
                for agreement in sensor_agreement_values:
                    prompt = canonical_oxygen_prompt(oxygen, slope, agreement)
                    input_ids = _tokenize(tokenizer, prompt, _model_device(model))["input_ids"]
                    with torch.no_grad():
                        reference = reference_gemma_forward(
                            model,
                            input_ids,
                            record_provenance=False,
                            compare_attention_kernels=False,
                            capture_boundaries=False,
                            last_token_only=True,
                        )
                        model.config._attn_implementation = "eager"
                        eager = model(
                            input_ids=input_ids,
                            attention_mask=torch.ones_like(input_ids),
                            use_cache=False,
                            logits_to_keep=1,
                        ).logits
                        model.config._attn_implementation = original_attention
                        deployed = model(
                            input_ids=input_ids,
                            attention_mask=torch.ones_like(input_ids),
                            use_cache=False,
                            logits_to_keep=1,
                        ).logits
                    eager_difference = (reference.logits.float() - eager.float()).abs()
                    deployed_difference = (reference.logits.float() - deployed.float()).abs()
                    reference_token = int(torch.argmax(reference.logits[0, -1]))
                    eager_token = int(torch.argmax(eager[0, -1]))
                    deployed_token = int(torch.argmax(deployed[0, -1]))
                    state = {
                        "oxygen_percent": oxygen,
                        "slope_percent_per_minute": slope,
                        "sensor_agreement": agreement,
                        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                        "input_ids_sha256": hashlib.sha256(canonical_json(input_ids[0].tolist()).encode("utf-8")).hexdigest(),
                        "reference_logits_sha256": tensor_descriptor(reference.logits)["sha256"],
                        "eager_logits_sha256": tensor_descriptor(eager)["sha256"],
                        "deployed_logits_sha256": tensor_descriptor(deployed)["sha256"],
                        "reference_vs_eager_exact": bool(torch.equal(reference.logits, eager)),
                        "reference_vs_eager_maximum_error": float(eager_difference.max()),
                        "reference_token_id": reference_token,
                        "eager_token_id": eager_token,
                        "deployed_token_id": deployed_token,
                        "reference_eager_token_match": reference_token == eager_token,
                        "reference_deployed_token_match": reference_token == deployed_token,
                        "reference_vs_deployed_maximum_error": float(deployed_difference.max()),
                    }
                    append_chain_record(records, state)
                    states.append(state)
    finally:
        model.config._attn_implementation = original_attention
    root = records[-1]["record_hash"] if records else "GENESIS"
    body = {
        "scope": "Exhaustive fixed-template verification over the explicitly enumerated finite state grid; not unrestricted natural-language equivalence.",
        "model_state_sha256": model_state_sha256(model),
        "domain": {
            "oxygen_percent": list(oxygen_values),
            "slope_percent_per_minute": list(slope_values),
            "sensor_agreement": list(sensor_agreement_values),
            "canonical_prompt_template": CANONICAL_OXYGEN_TEMPLATE,
            "numeric_format": "Python format(value, 'g')",
            "state_count": len(states),
        },
        "attention_implementations": {"reference": "explicit_eager", "eager_comparator": "eager", "deployed": original_attention},
        "states": states,
        "summary": {
            "reference_eager_exact_states": sum(state["reference_vs_eager_exact"] for state in states),
            "reference_eager_token_match_states": sum(state["reference_eager_token_match"] for state in states),
            "reference_deployed_token_match_states": sum(state["reference_deployed_token_match"] for state in states),
            "maximum_reference_eager_error": max(state["reference_vs_eager_maximum_error"] for state in states),
            "maximum_reference_deployed_error": max(state["reference_vs_deployed_maximum_error"] for state in states),
        },
        "records": records,
        "root_sha256": root,
        "chain_verification": verify_trace_chain(records, root),
    }
    body["certificate_sha256"] = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    return body


def verify_bounded_domain_certificate(certificate: dict[str, Any]) -> dict[str, Any]:
    records = certificate.get("records", [])
    states = certificate.get("states", [])
    summary = certificate.get("summary", {})
    chain = verify_trace_chain(records, certificate.get("root_sha256"))
    original_hash = certificate.get("certificate_sha256")
    body = {key: value for key, value in certificate.items() if key != "certificate_sha256"}
    calculated_hash = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    payloads_match = len(records) == len(states) and all(record.get("payload") == state for record, state in zip(records, states))
    domain = certificate.get("domain", {})
    oxygen_axis = domain.get("oxygen_percent", [])
    slope_axis = domain.get("slope_percent_per_minute", [])
    agreement_axis = domain.get("sensor_agreement", [])
    expected_states = [
        (oxygen, slope, agreement)
        for oxygen in oxygen_axis
        for slope in slope_axis
        for agreement in agreement_axis
    ]
    observed_states = [
        (state.get("oxygen_percent"), state.get("slope_percent_per_minute"), state.get("sensor_agreement"))
        for state in states
    ]
    domain_complete = bool(expected_states and len(observed_states) == len(expected_states) and set(observed_states) == set(expected_states))
    def canonical_hash_matches(state: dict[str, Any]) -> bool:
        try:
            prompt = canonical_oxygen_prompt(
                float(state["oxygen_percent"]),
                float(state["slope_percent_per_minute"]),
                state["sensor_agreement"],
            )
        except (KeyError, TypeError, ValueError):
            return False
        return state.get("prompt_sha256") == hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    canonical_prompt_hashes_match = all(canonical_hash_matches(state) for state in states)
    state_claims_consistent = all(
        state.get("reference_vs_eager_exact")
        == (
            state.get("reference_logits_sha256") == state.get("eager_logits_sha256")
            and state.get("reference_vs_eager_maximum_error") == 0.0
        )
        and state.get("reference_eager_token_match")
        == (state.get("reference_token_id") == state.get("eager_token_id"))
        and state.get("reference_deployed_token_match")
        == (state.get("reference_token_id") == state.get("deployed_token_id"))
        for state in states
    )
    calculated_summary = {
        "reference_eager_exact_states": sum(state.get("reference_vs_eager_exact") is True for state in states),
        "reference_eager_token_match_states": sum(state.get("reference_eager_token_match") is True for state in states),
        "reference_deployed_token_match_states": sum(state.get("reference_deployed_token_match") is True for state in states),
        "maximum_reference_eager_error": max((state.get("reference_vs_eager_maximum_error", math.inf) for state in states), default=math.inf),
        "maximum_reference_deployed_error": max((state.get("reference_vs_deployed_maximum_error", math.inf) for state in states), default=math.inf),
    }
    state_count_matches = domain.get("state_count") == len(states) == len(expected_states)
    valid = bool(
        chain["valid"]
        and original_hash == calculated_hash
        and payloads_match
        and calculated_summary == summary
        and state_count_matches
        and domain_complete
        and canonical_prompt_hashes_match
        and state_claims_consistent
        and calculated_summary["reference_eager_exact_states"] == len(states)
        and calculated_summary["reference_eager_token_match_states"] == len(states)
        and calculated_summary["reference_deployed_token_match_states"] == len(states)
    )
    return {
        "valid": valid,
        "certificate_hash_match": original_hash == calculated_hash,
        "chain": chain,
        "payloads_match": payloads_match,
        "summary_matches": calculated_summary == summary,
        "state_count_matches": state_count_matches,
        "domain_complete": domain_complete,
        "canonical_prompt_hashes_match": canonical_prompt_hashes_match,
        "state_claims_consistent": state_claims_consistent,
        "verified_state_count": len(states),
        "reference_eager_exact_states": calculated_summary["reference_eager_exact_states"],
        "reference_deployed_token_match_states": calculated_summary["reference_deployed_token_match_states"],
    }


def recompute_bounded_domain_certificate(model: Any, tokenizer: Any, certificate: dict[str, Any]) -> dict[str, Any]:
    integrity = verify_bounded_domain_certificate(certificate)
    supplied_model_hash = model_state_sha256(model)
    model_matches = supplied_model_hash == certificate.get("model_state_sha256")
    if not model_matches:
        return {
            "valid": False,
            "integrity": integrity,
            "model_matches": False,
            "supplied_model_state_sha256": supplied_model_hash,
            "reason": "Supplied model does not match the certificate model commitment",
        }
    domain = certificate["domain"]
    recomputed = bounded_domain_equivalence_certificate(
        model,
        tokenizer,
        tuple(domain["oxygen_percent"]),
        tuple(domain["slope_percent_per_minute"]),
        tuple(domain["sensor_agreement"]),
    )
    recomputed_integrity = verify_bounded_domain_certificate(recomputed)
    key = lambda state: (
        state["oxygen_percent"],
        state["slope_percent_per_minute"],
        state["sensor_agreement"],
    )
    original_states = {key(state): state for state in certificate["states"]}
    recomputed_states = {key(state): state for state in recomputed["states"]}
    claims_match = original_states.keys() == recomputed_states.keys() and all(
        original_states[state_key][claim] == recomputed_states[state_key][claim]
        for state_key in original_states
        for claim in (
            "reference_vs_eager_exact",
            "reference_token_id",
            "eager_token_id",
            "deployed_token_id",
            "reference_eager_token_match",
            "reference_deployed_token_match",
        )
    )
    exact_match = canonical_json(recomputed) == canonical_json(certificate)
    return {
        "valid": bool(integrity["valid"] and recomputed_integrity["valid"] and model_matches and claims_match),
        "integrity": integrity,
        "recomputed_integrity": recomputed_integrity,
        "model_matches": model_matches,
        "claims_match": claims_match,
        "reexecution_exact_match": exact_match,
        "recomputed_certificate_sha256": recomputed["certificate_sha256"],
    }


def summarize_bounded_domain(certificate: dict[str, Any]) -> dict[str, Any]:
    states = certificate["states"]
    token_counts: dict[str, int] = {}
    for state in states:
        key = str(state["reference_token_id"])
        token_counts[key] = token_counts.get(key, 0) + 1
    return {
        "scope": certificate["scope"],
        "model_state_sha256": certificate["model_state_sha256"],
        "domain": certificate["domain"],
        "attention_implementations": certificate["attention_implementations"],
        "summary": certificate["summary"],
        "reference_output_token_counts": token_counts,
        "certificate_sha256": certificate["certificate_sha256"],
        "root_sha256": certificate["root_sha256"],
        "chain_verified": certificate["chain_verification"]["valid"],
        "interpretation": "The independent reference and Hugging Face eager paths were exactly equal for every enumerated canonical state. The deployed SDPA path selected the same token for every state despite nonzero logit differences.",
        "limitations": [
            "Coverage is exhaustive only for the explicitly listed values and one canonical prompt template.",
            "The certificate does not cover paraphrases, intermediate numeric values, or unrestricted natural language.",
            "Token agreement does not imply equality of logits or internal states.",
            "The finite grid is expandable through CLI axes but is not a universal symbolic proof.",
            "No GMP, biological, clinical, product-quality, or patient-safety conclusion is supported.",
        ],
    }


def semantic_coordinate_registry(model: Any, tokenizer: Any) -> dict[str, Any]:
    config = model.config
    body = {
        "scope": "Architecture-defined coordinate roles. Learned latent coordinates retain numeric identity but have no preassigned biological meaning.",
        "coordinates": {
            "input_ids[position]": "Tokenizer vocabulary identifier at an ordered context position.",
            "embedding[token_id, hidden_coordinate]": "Learned token-vector value; token_id maps to tokenizer text, hidden_coordinate is an unlabeled learned basis coordinate.",
            "query[layer, query_head, position, head_coordinate]": "Projected and normalized query channel used in attention compatibility calculations.",
            "key[layer, key_value_head, position, head_coordinate]": "Projected and normalized key channel; shared across query heads under grouped-query attention.",
            "value[layer, key_value_head, position, head_coordinate]": "Projected value channel; shared across query heads under grouped-query attention.",
            "attention_probability[layer, query_head, query_position, key_position]": "Normalized weight multiplying a repeated value vector for one query/key position pair.",
            "residual[layer, position, hidden_coordinate]": "Running learned hidden-state coordinate after the named residual boundary; hidden_coordinate has no intrinsic domain label.",
            "mlp_intermediate[layer, position, neuron]": "Product of GELU-tanh gate and up-projection values before down projection; neuron is a learned intermediate coordinate.",
            "logits[position, token_id]": "Unnormalized vocabulary score whose token_id has exact tokenizer text semantics.",
        },
        "dimensions": {
            "layers": config.num_hidden_layers,
            "hidden_coordinates": config.hidden_size,
            "query_heads": config.num_attention_heads,
            "key_value_heads": config.num_key_value_heads,
            "head_coordinates": config.head_dim,
            "mlp_neurons": config.intermediate_size,
            "vocabulary_coordinates": config.vocab_size,
        },
        "semantic_boundary": {
            "exact_by_construction": ["tensor role", "axis role", "token ID to tokenizer text", "operator dependency"],
            "requires_empirical_or_formal_domain_evidence": ["biological concept assigned to a hidden coordinate", "causal sufficiency of a coordinate set", "scientific correctness of a learned association"],
        },
    }
    body["sha256"] = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    return body


def summarize_reference_evidence(
    certificate: dict[str, Any],
    conformance: dict[str, Any],
    coordinate_registry: dict[str, Any],
) -> dict[str, Any]:
    deployed = certificate["deployed_path"]
    deployed_boundaries = deployed["boundaries"]
    first_divergence = next((item["boundary"] for item in deployed_boundaries if not item["exact_equal"]), None)
    return {
        "scope": "Fixed-input machine-checked equivalence evidence for an independent Python orchestration; not a universal proof over all inputs or a proof of CUDA kernel internals.",
        "model_state_sha256": certificate["model_state_sha256"],
        "input_ids_sha256": certificate["input_ids_sha256"],
        "reference_vs_huggingface_eager": {
            "checked_boundaries": len(certificate["boundaries"]),
            "all_boundaries_exact": all(item["exact_equal"] for item in certificate["boundaries"]),
            "logits_exact": certificate["logits"]["exact_equal"],
            "maximum_logit_error": certificate["logits"]["max_absolute_error"],
            "selected_token_exact_match": certificate["selected_token_exact_match"],
        },
        "reference_certificate": {
            "sha256": certificate["certificate_sha256"],
            "root_sha256": certificate["certificate_root_sha256"],
            "record_count": len(certificate["certificate_records"]),
            "chain_verified": certificate["certificate_chain_verification"]["valid"],
            "reference_execution_trace_root_sha256": certificate["reference_trace_root_sha256"],
            "reference_execution_trace_record_count": len(certificate["reference_trace_records"]),
            "reference_execution_trace_chain_verified": certificate["reference_trace_chain_verification"]["valid"],
        },
        "explicit_attention_vs_sdpa_same_inputs": {
            "layer_count": len(certificate["attention_kernel_comparisons"]),
            "exact_layers": sum(item["exact_equal"] for item in certificate["attention_kernel_comparisons"]),
            "maximum_absolute_error": max(item["max_absolute_error"] for item in certificate["attention_kernel_comparisons"]),
            "mean_of_layer_mean_absolute_errors": sum(item["mean_absolute_error"] for item in certificate["attention_kernel_comparisons"]) / len(certificate["attention_kernel_comparisons"]),
            "layers": [
                {
                    "layer": item["layer"],
                    "attention_type": item["attention_type"],
                    "exact_equal": item["exact_equal"],
                    "max_absolute_error": item["max_absolute_error"],
                    "mean_absolute_error": item["mean_absolute_error"],
                }
                for item in certificate["attention_kernel_comparisons"]
            ],
        },
        "reference_vs_deployed_path": {
            "attention_implementation": deployed["attention_implementation"],
            "first_diverging_boundary": first_divergence,
            "maximum_boundary_error": max(item["max_absolute_error"] for item in deployed_boundaries),
            "final_norm_maximum_error": deployed_boundaries[-1]["max_absolute_error"],
            "full_sequence_logits_exact": deployed["logits"]["exact_equal"],
            "full_sequence_maximum_logit_error": deployed["logits"]["max_absolute_error"],
            "full_sequence_mean_logit_error": deployed["logits"]["mean_absolute_error"],
            "selected_token_matches_reference": deployed["selected_token_matches_reference"],
        },
        "operator_equation_conformance": {
            "sha256": conformance["sha256"],
            "case_count": len(conformance["cases"]),
            "maximum_absolute_error": conformance["maximum_absolute_error"],
            "scope": conformance["scope"],
        },
        "coordinate_semantics": {
            "sha256": coordinate_registry["sha256"],
            "exact_by_construction": coordinate_registry["semantic_boundary"]["exact_by_construction"],
            "requires_additional_evidence": coordinate_registry["semantic_boundary"]["requires_empirical_or_formal_domain_evidence"],
        },
        "interpretation": "The independently orchestrated eager equations exactly reproduce every checked Hugging Face eager boundary and vocabulary logit for this input. The deployed SDPA path selects the same token but diverges numerically beginning at layer 0, showing that implementation choice is part of the model's operational rationale.",
        "limitations": [
            "The equivalence certificate covers one fixed input and does not prove equality for all token sequences.",
            "Both implementations ultimately use PyTorch tensor kernels; orchestration is independent, primitive kernels are not.",
            "Fixed-vector scalar equation checks are conformance examples rather than universal kernel proofs.",
            "SDPA may contain fused CUDA arithmetic not exposed by the eager reference trace.",
            "Architecture-defined coordinate roles do not assign biological meaning to learned hidden coordinates.",
            "No GMP, biological, clinical, product-quality, or patient-safety conclusion is supported.",
        ],
    }


def operator_equation_conformance() -> dict[str, Any]:
    _require_torch()
    cases = []

    vector = [1.25, -0.5, 2.0]
    weight = [0.1, -0.2, 0.3]
    python_linear = sum(value * coefficient for value, coefficient in zip(vector, weight))
    torch_linear = float(functional.linear(torch.tensor(vector, dtype=torch.float64), torch.tensor([weight], dtype=torch.float64))[0])
    cases.append({"operator": "linear", "python": python_linear, "torch": torch_linear, "absolute_error": abs(python_linear - torch_linear)})

    epsilon = 1e-6
    scale = [0.2, -0.1, 0.3]
    mean_square = sum(value * value for value in vector) / len(vector)
    python_norm = [value / math.sqrt(mean_square + epsilon) * (1.0 + gain) for value, gain in zip(vector, scale)]
    torch_norm = reference_rms_norm(
        torch.tensor(vector, dtype=torch.float64), torch.tensor(scale, dtype=torch.float64), epsilon
    ).tolist()
    cases.append({
        "operator": "rms_norm",
        "python": python_norm,
        "torch": torch_norm,
        "max_absolute_error": max(abs(left - right) for left, right in zip(python_norm, torch_norm)),
    })

    python_gelu = [0.5 * value * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (value + 0.044715 * value**3))) for value in vector]
    torch_gelu = functional.gelu(torch.tensor(vector, dtype=torch.float64), approximate="tanh").tolist()
    cases.append({
        "operator": "gelu_tanh",
        "python": python_gelu,
        "torch": torch_gelu,
        "max_absolute_error": max(abs(left - right) for left, right in zip(python_gelu, torch_gelu)),
    })

    logits = [2.0, 1.0, -0.5]
    maximum = max(logits)
    exponentials = [math.exp(value - maximum) for value in logits]
    total = sum(exponentials)
    python_softmax = [value / total for value in exponentials]
    torch_softmax = functional.softmax(torch.tensor(logits, dtype=torch.float64), dim=-1).tolist()
    cases.append({
        "operator": "stable_softmax",
        "python": python_softmax,
        "torch": torch_softmax,
        "max_absolute_error": max(abs(left - right) for left, right in zip(python_softmax, torch_softmax)),
    })

    if torch.cuda.is_available():
        cuda = torch.device("cuda")
        cuda_linear = float(functional.linear(torch.tensor(vector, dtype=torch.float64, device=cuda), torch.tensor([weight], dtype=torch.float64, device=cuda))[0].cpu())
        cases[0]["cuda"] = cuda_linear
        cases[0]["cuda_absolute_error"] = abs(python_linear - cuda_linear)
        cuda_norm = reference_rms_norm(
            torch.tensor(vector, dtype=torch.float64, device=cuda),
            torch.tensor(scale, dtype=torch.float64, device=cuda),
            epsilon,
        ).cpu().tolist()
        cases[1]["cuda"] = cuda_norm
        cases[1]["cuda_max_absolute_error"] = max(abs(left - right) for left, right in zip(python_norm, cuda_norm))
        cuda_gelu = functional.gelu(torch.tensor(vector, dtype=torch.float64, device=cuda), approximate="tanh").cpu().tolist()
        cases[2]["cuda"] = cuda_gelu
        cases[2]["cuda_max_absolute_error"] = max(abs(left - right) for left, right in zip(python_gelu, cuda_gelu))
        cuda_softmax = functional.softmax(torch.tensor(logits, dtype=torch.float64, device=cuda), dim=-1).cpu().tolist()
        cases[3]["cuda"] = cuda_softmax
        cases[3]["cuda_max_absolute_error"] = max(abs(left - right) for left, right in zip(python_softmax, cuda_softmax))
    maximum_error = max(
        max(
            case.get("max_absolute_error", case.get("absolute_error", 0.0)),
            case.get("cuda_max_absolute_error", case.get("cuda_absolute_error", 0.0)),
        )
        for case in cases
    )
    body = {
        "scope": "Fixed-vector CPU and available CUDA conformance checks against separate Python scalar equations; not a proof for all values or kernel instructions.",
        "cases": cases,
        "maximum_absolute_error": maximum_error,
    }
    body["sha256"] = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    return body
