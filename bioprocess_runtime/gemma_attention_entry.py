from __future__ import annotations

import hashlib
import inspect
from typing import Any

import numpy as np

from .gemma_ir import verify_gemma_ir
from .gemma_ir_interpreter import bind_model_tensors, _rectangular_integer_rows, _projection_arithmetic_commitment
from .gemma_rsqrt_lookup import CheckedRsqrtLookup, _rms_lookup_row, _runtime, _code_sha
from .operational_semantics import tensor_descriptor, append_chain_record, verify_trace_chain
from .serialization import canonical_json

TARGETS = ("layer.0.query.normalized", "layer.0.key.normalized", "layer.0.value.heads")
OBSERVED = ("hidden.0", "layer.0.attention.normalized", "layer.0.query.flat", "layer.0.key.flat", "layer.0.value.flat", *TARGETS[:2])
SCOPE = "Connected fixed-input first-layer dependency slice through pre-RoPE normalized Q/K and V heads; independent numerical execution with an explicit empirical rsqrt lookup; not complete attention/layer/model or native-arithmetic qualification."
VALUE_HEADS_COMPARISON = "coordinate reinterpretation of captured original v_proj output; no floating arithmetic"


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def entry_instructions(program: dict[str, Any]) -> list[dict[str, Any]]:
    if not verify_gemma_ir(program)["valid"]:
        raise ValueError("Attention entry requires a valid typed IR")
    needed, selected = set(TARGETS), []
    for instruction in reversed(program["instructions"]):
        if needed.intersection(instruction["outputs"]):
            selected.append(instruction)
            needed.difference_update(instruction["outputs"])
            needed.update(instruction["inputs"])
    selected.reverse()
    expected = {"embedding_unscaled": ("EMBEDDING", ["input_ids"], ["model.embed_tokens.weight"]),
                "hidden.0": ("SCALE", ["embedding_unscaled"], ["model.embed_tokens.embed_scale"]),
                "layer.0.attention.normalized": ("RMS_NORM", ["hidden.0"], ["model.layers.0.input_layernorm.weight"])}
    for role, short, heads in (("query", "q", 4), ("key", "k", 1), ("value", "v", 1)):
        expected[f"layer.0.{role}.flat"] = ("LINEAR", ["layer.0.attention.normalized"], [f"model.layers.0.self_attn.{short}_proj.weight"])
        expected[f"layer.0.{role}.heads"] = ("RESHAPE_TRANSPOSE_HEADS", [f"layer.0.{role}.flat"], [])
        if role != "value":
            expected[f"layer.0.{role}.normalized"] = ("RMS_NORM", [f"layer.0.{role}.heads"], [f"model.layers.0.self_attn.{short}_norm.weight"])
    if needed != {"input_ids"} or len(selected) != 11 or {name for instruction in selected for name in instruction["outputs"]} != set(expected):
        raise ValueError("Attention entry dependency coverage mismatch")
    for instruction in selected:
        if len(instruction["outputs"]) != 1:
            raise ValueError("Unexpected multi-output entry instruction")
        opcode, inputs, parameters = expected[instruction["outputs"][0]]
        if instruction["opcode"] != opcode or instruction["inputs"] != inputs or instruction["parameter_refs"] != parameters:
            raise ValueError("Entry instruction is not bound to its declared model role")
        if opcode == "RESHAPE_TRANSPOSE_HEADS":
            heads = 4 if instruction["outputs"][0] == "layer.0.query.heads" else 1
            if instruction["attributes"] != {"heads": heads, "head_dimension": 256}:
                raise ValueError("Unsupported head-coordinate mapping")
    return selected


def _bits(parameter: Any) -> np.ndarray:
    import torch

    if parameter.dtype != torch.bfloat16:
        raise ValueError("Entry parameters must be bfloat16")
    return parameter.detach().contiguous().cpu().view(torch.uint16).numpy().copy()


def _descriptor(values: np.ndarray) -> dict[str, Any]:
    import torch

    tensor = torch.from_numpy(np.ascontiguousarray(values)).view(torch.bfloat16)
    return tensor_descriptor(tensor)


def execute_attention_entry(program: dict[str, Any], parameters: dict[str, Any], token_ids: list[list[int]], lookup: CheckedRsqrtLookup, runtime: dict[str, Any]) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    from .gemma_float_semantics import bfloat16_multiply_bits
    from .gemma_wmma_candidate import operand_aligned_product_bits, split_k_candidate_bits, DENSE_SPLIT_PROFILE

    instructions = entry_instructions(program)
    if not _rectangular_integer_rows(token_ids) or len(token_ids) != 1 or len(token_ids[0]) != 30 or any(not 0 <= token < program["configuration"]["vocabulary_size"] for token in token_ids[0]):
        raise ValueError("Entry input is restricted to one valid 30-token sequence")
    states, records, scalar_stages = {}, [], {}
    for instruction in instructions:
        output_name = instruction["outputs"][0]
        opcode = instruction["opcode"]
        if opcode == "EMBEDDING":
            table = parameters[instruction["parameter_refs"][0]]
            value = np.stack([_bits(table[token]) for token in token_ids[0]])[None, ...]
            provider = "independent_coordinate_load_v1"
        elif opcode == "SCALE":
            source = states[instruction["inputs"][0]]
            scale = _bits(parameters[instruction["parameter_refs"][0]])
            if scale.size != 1:
                raise ValueError("Expected scalar embedding scale")
            value = np.asarray([bfloat16_multiply_bits(int(bits), int(scale.item())) for bits in source.reshape(-1)], dtype=np.uint16).reshape(source.shape)
            provider = "finite_bfloat16_multiply_rne_v1"
        elif opcode == "RMS_NORM":
            source = states[instruction["inputs"][0]]
            weights = _bits(parameters[instruction["parameter_refs"][0]]).tolist()
            rows = [_rms_lookup_row(row, weights, instruction["attributes"]["epsilon"], lookup, runtime) for row in source.reshape(-1, source.shape[-1]).tolist()]
            value = np.asarray([row["output_bits"] for row in rows], dtype=np.uint16).reshape(source.shape)
            scalar_stages[output_name] = {key: [row[key] for row in rows] for key in ("mean_bits", "denominator_bits", "rsqrt_bits")}
            provider = "source_vec4_warp32_and_explicit_rsqrt_lookup_v1"
        elif opcode == "LINEAR":
            source = states[instruction["inputs"][0]]
            weights = _bits(parameters[instruction["parameter_refs"][0]]).tolist()
            query = output_name == "layer.0.query.flat"
            width = 1024 if query else 256
            if source.shape != (1, 30, 640) or len(weights) != width or any(len(row) != 640 for row in weights):
                raise ValueError("Unsupported entry projection geometry")
            oracle = operand_aligned_product_bits if query else lambda left, right: split_k_candidate_bits(left, right, *DENSE_SPLIT_PROFILE)
            value = np.asarray([[oracle(row, weight) for weight in weights] for row in source[0].tolist()], dtype=np.uint16)[None, ...]
            provider = "operand_alignment_v1" if query else "split_k64_bf16_serial_fp32_v1"
        else:
            source = states[instruction["inputs"][0]]
            heads = instruction["attributes"]["heads"]
            value = source.reshape(1, 30, heads, 256).transpose(0, 2, 1, 3)
            provider = "independent_head_coordinate_map_v1"
        declaration = program["tensors"][output_name]
        expected_shape = [{"B": 1, "S": 30}.get(dimension, dimension) for dimension in declaration["shape"]]
        if list(value.shape) != expected_shape or declaration["dtype"] != "torch.bfloat16":
            raise ValueError("Entry result violates an IR tensor declaration")
        states[output_name] = value
        payload = {"instruction_id": instruction["id"], "instruction_sha256": instruction["instruction_sha256"], "opcode": opcode,
                   "inputs": instruction["inputs"], "parameter_refs": instruction["parameter_refs"], "output_name": output_name,
                   "provider": provider, "output": _descriptor(value)}
        if output_name in scalar_stages:
            payload["scalar_stages_sha256"] = _sha(scalar_stages[output_name])
        append_chain_record(records, payload)
    return states, records, scalar_stages


def _entry_code_sha() -> str:
    from .reference_gemma import _rms_code_commitment

    return _sha({"functions": {fn.__name__: inspect.getsource(fn) for fn in (entry_instructions, _bits, _descriptor, execute_attention_entry, _rms_lookup_row)},
                 "projection_arithmetic": _projection_arithmetic_commitment(), "rms_arithmetic": _rms_code_commitment(), "lookup_mapping": _code_sha()})


def _model_parameters(program: dict[str, Any], model: Any) -> dict[str, Any]:
    import torch

    entry_instructions(program)
    parameters = bind_model_tensors(program, model, verify_hashes=True)
    device = parameters["model.embed_tokens.weight"].device
    if model.training or model.config._attn_implementation != "eager" or device.type != "cuda" or device.index != torch.cuda.current_device():
        raise ValueError("Entry comparison requires evaluation/eager on the current CUDA device")
    embedding = model.model.embed_tokens
    if not isinstance(embedding, torch.nn.Embedding) or embedding.weight is not parameters["model.embed_tokens.weight"] or embedding.embed_scale is not parameters["model.embed_tokens.embed_scale"] or embedding.weight.dtype != torch.bfloat16 or embedding.embed_scale.dtype != torch.bfloat16 or embedding.embed_scale.device != device:
        raise ValueError("Original embedding/scale module binding mismatch")
    for role, short, width in (("query", "q", 1024), ("key", "k", 256), ("value", "v", 256)):
        module = getattr(model.model.layers[0].self_attn, short + "_proj")
        name = f"model.layers.0.self_attn.{short}_proj.weight"
        if type(module) is not torch.nn.Linear or module.bias is not None or module.weight is not parameters[name] or module.weight.shape != (width, 640) or module.weight.dtype != torch.bfloat16 or module.weight.device != device:
            raise ValueError("Unsupported original entry projection")
    norms = {"layer.0.attention.normalized": model.model.layers[0].input_layernorm,
             "layer.0.query.normalized": model.model.layers[0].self_attn.q_norm,
             "layer.0.key.normalized": model.model.layers[0].self_attn.k_norm}
    for instruction in entry_instructions(program):
        if instruction["opcode"] == "RMS_NORM":
            module = norms[instruction["outputs"][0]]
            if module.eps != instruction["attributes"]["epsilon"] or module.weight is not parameters[instruction["parameter_refs"][0]] or module.weight.device != device or module.weight.dtype != torch.bfloat16:
                raise ValueError("Original entry RMS module binding mismatch")
    return parameters


def build_attention_entry_plan(program: dict[str, Any], model: Any, fixture: dict[str, Any], lookup: CheckedRsqrtLookup) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    if type(lookup) is not CheckedRsqrtLookup or getattr(lookup.predict_bits, "__func__", None) is not CheckedRsqrtLookup.predict_bits:
        raise ValueError("Entry certificates require the checked lookup implementation")
    if fixture["program_sha256"] != program["program_sha256"]:
        raise ValueError("Entry fixture belongs to another IR")
    token_ids = fixture["input_token_ids"]
    if not _rectangular_integer_rows(token_ids) or len(token_ids) != 1 or len(token_ids[0]) != 30:
        raise ValueError("Entry fixture requires one 30-token sequence")
    ids_descriptor = tensor_descriptor(torch.tensor(token_ids, dtype=torch.int64))
    if ids_descriptor["sha256"] != fixture["input_ids"]["sha256"]:
        raise ValueError("Entry token commitment mismatch")
    parameters = _model_parameters(program, model)
    runtime = _runtime()
    states, records, stages = execute_attention_entry(program, parameters, token_ids, lookup, runtime)
    bind_model_tensors(program, model, verify_hashes=True)
    bundle = {"state_bits": {name: value.tolist() for name, value in states.items()}, "scalar_stages": stages}
    body = {"schema_version": 1, "scope": SCOPE, "program_sha256": program["program_sha256"], "fixture_sha256": _sha(fixture),
            "input_token_ids": token_ids, "input_ids": ids_descriptor,
            "model_binding": {"class": type(model).__module__ + "." + type(model).__name__, "config_sha256": _sha(model.config.to_dict()),
                              "parameter_commitments_sha256": _sha(program["parameter_commitments"])},
            "runtime": runtime, "lookup_evidence": lookup.evidence, "execution_code_sha256": _entry_code_sha(),
            "records": records, "execution_root": records[-1]["record_hash"], "bundle_sha256": _sha(bundle),
            "target_names": list(TARGETS), "observed_module_boundaries": list(OBSERVED), "target_value_count": 46080,
            "value_heads_comparison": VALUE_HEADS_COMPARISON,
            "uses_framework_floating_arithmetic_in_slice": False, "uses_empirical_rsqrt_specification": True,
            "candidate_refitting_allowed": False, "rope_rotation_compared": False, "attention_compared": False,
            "full_first_layer_qualified": False, "hardware_semantics_established": False, "global_exactness_activation_allowed": False}
    return {**body, "plan_sha256": _sha(body)}, bundle


def _state_array(bits: Any, shape: list[int]) -> np.ndarray:
    def flatten(value: Any, dimensions: list[int]) -> list[int]:
        if not dimensions:
            if type(value) is not int or not 0 <= value <= 65535:
                raise ValueError("Entry values must be uint16 bit encodings")
            return [value]
        if not isinstance(value, list) or len(value) != dimensions[0]:
            raise ValueError("Entry tensor shape mismatch")
        return [item for child in value for item in flatten(child, dimensions[1:])]
    return np.asarray(flatten(bits, shape), dtype=np.uint16).reshape(shape)


def check_attention_entry_plan(program: dict[str, Any], plan: dict[str, Any], bundle: dict[str, Any], lookup: CheckedRsqrtLookup) -> dict[str, np.ndarray]:
    import torch

    if type(lookup) is not CheckedRsqrtLookup or getattr(lookup.predict_bits, "__func__", None) is not CheckedRsqrtLookup.predict_bits:
        raise ValueError("Entry verification requires the checked lookup implementation")
    instructions = entry_instructions(program)
    if plan.get("value_heads_comparison") != VALUE_HEADS_COMPARISON:
        raise ValueError("Entry value-head comparison boundary mismatch")
    if plan["program_sha256"] != program["program_sha256"] or _sha({k: v for k, v in plan.items() if k != "plan_sha256"}) != plan["plan_sha256"] or _sha(bundle) != plan["bundle_sha256"]:
        raise ValueError("Entry plan or bundle commitment mismatch")
    if plan["execution_code_sha256"] != _entry_code_sha() or canonical_json(plan["lookup_evidence"]) != canonical_json(lookup.evidence):
        raise ValueError("Entry numerical implementation or lookup specification mismatch")
    false_fields = ("uses_framework_floating_arithmetic_in_slice", "candidate_refitting_allowed", "rope_rotation_compared", "attention_compared", "full_first_layer_qualified", "hardware_semantics_established", "global_exactness_activation_allowed")
    if plan["scope"] != SCOPE or any(plan.get(key) is not False for key in false_fields) or plan.get("uses_empirical_rsqrt_specification") is not True or plan["target_names"] != list(TARGETS) or plan["observed_module_boundaries"] != list(OBSERVED) or plan["target_value_count"] != 46080:
        raise ValueError("Entry scope boundary mismatch")
    ids = plan["input_token_ids"]
    if not _rectangular_integer_rows(ids) or len(ids) != 1 or len(ids[0]) != 30 or any(not 0 <= token < program["configuration"]["vocabulary_size"] for token in ids[0]) or tensor_descriptor(torch.tensor(ids, dtype=torch.int64)) != plan["input_ids"]:
        raise ValueError("Entry token identity mismatch")
    if plan["model_binding"]["parameter_commitments_sha256"] != _sha(program["parameter_commitments"]):
        raise ValueError("Entry model parameter commitment mismatch")
    if not verify_trace_chain(plan["records"], plan["execution_root"])["valid"] or len(plan["records"]) != len(instructions):
        raise ValueError("Entry witness chain or coverage mismatch")
    expected_names = {item["outputs"][0] for item in instructions}
    if set(bundle["state_bits"]) != expected_names:
        raise ValueError("Entry state coverage mismatch")
    rms_names = {item["outputs"][0] for item in instructions if item["opcode"] == "RMS_NORM"}
    if set(bundle["scalar_stages"]) != rms_names:
        raise ValueError("Entry RMS scalar-stage coverage mismatch")
    arrays = {}
    for instruction, record in zip(instructions, plan["records"]):
        name = instruction["outputs"][0]
        shape = [{"B": 1, "S": 30}.get(dimension, dimension) for dimension in program["tensors"][name]["shape"]]
        values = _state_array(bundle["state_bits"][name], shape)
        payload = record["payload"]
        providers = {"EMBEDDING": "independent_coordinate_load_v1", "SCALE": "finite_bfloat16_multiply_rne_v1", "RMS_NORM": "source_vec4_warp32_and_explicit_rsqrt_lookup_v1", "RESHAPE_TRANSPOSE_HEADS": "independent_head_coordinate_map_v1"}
        provider = ("operand_alignment_v1" if name == "layer.0.query.flat" else "split_k64_bf16_serial_fp32_v1") if instruction["opcode"] == "LINEAR" else providers[instruction["opcode"]]
        if payload.get("provider") != provider or payload.get("opcode") != instruction["opcode"]:
            raise ValueError("Unregistered entry numerical provider")
        if payload["instruction_id"] != instruction["id"] or payload["instruction_sha256"] != instruction["instruction_sha256"] or payload["inputs"] != instruction["inputs"] or payload["parameter_refs"] != instruction["parameter_refs"] or payload["output_name"] != name or canonical_json(payload["output"]) != canonical_json(_descriptor(values)):
            raise ValueError("Entry witness is not bound to its IR instruction/state")
        if instruction["opcode"] == "RMS_NORM":
            stages = bundle["scalar_stages"][name]
            if set(stages) != {"mean_bits", "denominator_bits", "rsqrt_bits"} or any(not isinstance(bits, list) or len(bits) != int(np.prod(shape[:-1])) or any(type(value) is not int or not 0 <= value <= 0xFFFFFFFF for value in bits) for bits in stages.values()):
                raise ValueError("Malformed entry RMS scalar stages")
            if payload.get("scalar_stages_sha256") != _sha(stages):
                raise ValueError("Entry RMS scalar-stage commitment mismatch")
        arrays[name] = values
    return arrays


def _entry_report(plan: dict[str, Any], predicted: dict[str, np.ndarray], observations: list[dict[str, Any]], runtime: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(observations, list) or len(observations) != 3:
        raise ValueError("Entry comparison requires three forward-prefix repetitions")
    mismatches, summaries = [], {}
    for repetition, observed in enumerate(observations):
        if set(observed) != set(OBSERVED):
            raise ValueError("Original forward did not expose every declared module boundary")
        for name in OBSERVED:
            values = _state_array(observed[name], list(predicted[name].shape))
            indices = np.argwhere(values != predicted[name])
            mismatches.extend({"repetition": repetition, "tensor": name, "coordinate": coordinate.tolist(),
                               "predicted_bits": int(predicted[name][tuple(coordinate)]), "observed_bits": int(values[tuple(coordinate)])} for coordinate in indices)
        value_heads = _state_array(observed["layer.0.value.flat"], [1, 30, 256]).reshape(1, 30, 1, 256).transpose(0, 2, 1, 3)
        for coordinate in np.argwhere(value_heads != predicted["layer.0.value.heads"]):
            mismatches.append({"repetition": repetition, "tensor": "layer.0.value.heads", "coordinate": coordinate.tolist(),
                               "predicted_bits": int(predicted["layer.0.value.heads"][tuple(coordinate)]), "observed_bits": int(value_heads[tuple(coordinate)])})
    for name in (*OBSERVED, "layer.0.value.heads"):
        summaries[name] = {"shape": list(predicted[name].shape), "value_count": predicted[name].size,
                           "prediction_sha256": _descriptor(predicted[name])["sha256"],
                           "mismatch_count": sum(item["tensor"] == name for item in mismatches)}
    body = {"schema_version": 1, "scope": SCOPE, "plan_sha256": plan["plan_sha256"], "execution_root": plan["execution_root"],
            "observations": observations, "runtime": runtime, "runtime_matches_plan": canonical_json(runtime) == canonical_json(plan["runtime"]),
            "tensor_summaries": summaries, "mismatch_count": len(mismatches), "mismatches": mismatches,
            "first_divergence": mismatches[0] if mismatches else None,
            "connected_slice_bit_exact": not mismatches and canonical_json(runtime) == canonical_json(plan["runtime"]),
            "repeated_outputs_identical": all(canonical_json(item) == canonical_json(observations[0]) for item in observations),
            "target_value_count": 46080, "uses_framework_floating_arithmetic_in_slice": False,
            "uses_empirical_rsqrt_specification": True, "original_forward_stopped_before_rope_rotation": True,
            "full_first_layer_qualified": False, "full_model_independently_qualified": False,
            "native_arithmetic_reconstructed": False, "hardware_semantics_established": False, "global_exactness_activation_allowed": False}
    return {**body, "report_sha256": _sha(body)}


def acquire_attention_entry(program: dict[str, Any], model: Any, plan: dict[str, Any], bundle: dict[str, Any], lookup: CheckedRsqrtLookup) -> dict[str, Any]:
    import torch

    predicted = check_attention_entry_plan(program, plan, bundle, lookup)
    parameters = _model_parameters(program, model)
    binding = {"class": type(model).__module__ + "." + type(model).__name__, "config_sha256": _sha(model.config.to_dict()), "parameter_commitments_sha256": _sha(program["parameter_commitments"])}
    if canonical_json(binding) != canonical_json(plan["model_binding"]) or canonical_json(_runtime()) != canonical_json(plan["runtime"]):
        raise ValueError("Original entry model or runtime differs from the prediction plan")
    layer = model.model.layers[0]
    modules = {"hidden.0": model.model.embed_tokens, "layer.0.attention.normalized": layer.input_layernorm,
               "layer.0.query.flat": layer.self_attn.q_proj, "layer.0.key.flat": layer.self_attn.k_proj, "layer.0.value.flat": layer.self_attn.v_proj,
               "layer.0.query.normalized": layer.self_attn.q_norm, "layer.0.key.normalized": layer.self_attn.k_norm}
    ids = torch.tensor(plan["input_token_ids"], dtype=torch.int64, device=parameters["model.embed_tokens.weight"].device)
    observations = []

    class EntryComplete(Exception):
        pass

    for _ in range(3):
        captured, handles = {}, []
        stopped = False

        def hook(name: str):
            def capture(module: Any, args: Any, output: Any) -> None:
                if name in captured or output.dtype != torch.bfloat16 or list(output.shape) != list(predicted[name].shape):
                    raise ValueError("Original entry boundary occurrence or tensor type mismatch")
                captured[name] = _bits(output).tolist()
                if name == "layer.0.key.normalized":
                    raise EntryComplete()
            return capture

        try:
            for name, module in modules.items():
                handles.append(module.register_forward_hook(hook(name)))
            with torch.no_grad():
                model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1)
        except EntryComplete:
            stopped = True
        finally:
            for handle in handles:
                handle.remove()
        if not stopped or set(captured) != set(OBSERVED):
            raise ValueError("Original model did not stop at the complete pre-RoPE boundary")
        observations.append(captured)
    bind_model_tensors(program, model, verify_hashes=True)
    return _entry_report(plan, predicted, observations, _runtime())


def verify_attention_entry(program: dict[str, Any], plan: dict[str, Any], bundle: dict[str, Any], report: dict[str, Any], lookup: CheckedRsqrtLookup) -> dict[str, Any]:
    try:
        predicted = check_attention_entry_plan(program, plan, bundle, lookup)
        expected = _entry_report(plan, predicted, report["observations"], report["runtime"])
        return {"valid": canonical_json(report) == canonical_json(expected), "mode": "integrity_only",
                "connected_slice_bit_exact": expected["connected_slice_bit_exact"], "mismatch_count": expected["mismatch_count"],
                "first_divergence": expected["first_divergence"], "global_exactness_activation_allowed": False}
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError) as error:
        return {"valid": False, "reason": str(error)}


def attention_entry_summary(plan: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in report.items() if key not in ("observations", "mismatches", "report_sha256")}
    body.update({"source_report_sha256": report["report_sha256"], "program_sha256": plan["program_sha256"],
                 "lookup_evidence": plan["lookup_evidence"], "model_binding": plan["model_binding"],
                 "execution_code_sha256": plan["execution_code_sha256"], "independent_instruction_count": len(plan["records"]),
                 "original_boundary_hashes": {name: [_descriptor(_state_array(observed[name], record["shape"]))["sha256"] for observed in report["observations"]] for name, record in report["tensor_summaries"].items() if name in OBSERVED}})
    body["derived_value_head_hashes"] = [_descriptor(_state_array(observed["layer.0.value.flat"], [1, 30, 256]).reshape(1, 30, 1, 256).transpose(0, 2, 1, 3))["sha256"] for observed in report["observations"]]
    body["value_heads_comparison"] = plan["value_heads_comparison"]
    return {**body, "summary_sha256": _sha(body)}
