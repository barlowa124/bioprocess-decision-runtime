from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping

from .gemma_ir import HASH_PATTERN, verify_gemma_ir
from .operational_semantics import append_chain_record, tensor_descriptor, tensor_sha256, verify_trace_chain
from .serialization import canonical_json

try:
    import torch
    from torch.nn import functional
except ImportError:
    torch = None
    functional = None


@dataclass(frozen=True)
class GemmaIrExecution:
    logits: Any
    selected_token_id: Any
    records: tuple[dict[str, Any], ...]
    root_sha256: str


def _require_torch() -> None:
    if torch is None or functional is None:
        raise RuntimeError("Gemma IR numerical execution requires PyTorch")


def bind_model_tensors(
    program: dict[str, Any], model: Any, verify_hashes: bool = True
) -> dict[str, Any]:
    _require_torch()
    verification = verify_gemma_ir(program)
    if not verification["valid"]:
        raise ValueError("Cannot bind model tensors to an invalid Gemma IR")
    bound = dict(model.named_parameters(remove_duplicate=False))
    bound.update(dict(model.named_buffers(remove_duplicate=False)))
    expected = program["parameter_commitments"]
    if set(bound) != set(expected):
        missing = sorted(set(expected) - set(bound))
        unexpected = sorted(set(bound) - set(expected))
        raise ValueError(f"Model tensor binding mismatch; missing={missing}, unexpected={unexpected}")
    for name, descriptor in expected.items():
        tensor = bound[name]
        if list(tensor.shape) != descriptor["shape"] or str(tensor.dtype) != descriptor["dtype"]:
            raise ValueError(f"Model tensor metadata mismatch for {name}")
        if verify_hashes and tensor_sha256(tensor) != descriptor["sha256"]:
            raise ValueError(f"Model tensor hash mismatch for {name}")
    return bound


def _rotate_half(value: Any) -> Any:
    half = value.shape[-1] // 2
    return torch.cat((-value[..., half:], value[..., :half]), dim=-1)


def _execute_instruction(
    instruction: dict[str, Any], states: dict[str, Any], parameters: Mapping[str, Any]
) -> dict[str, Any]:
    opcode = instruction["opcode"]
    inputs = [states[name] for name in instruction["inputs"]]
    parameter_values = [parameters[name] for name in instruction["parameter_refs"]]
    attributes = instruction["attributes"]
    output_names = instruction["outputs"]

    if opcode == "ARANGE":
        values = (torch.arange(inputs[0].shape[1], device=inputs[0].device).unsqueeze(0),)
    elif opcode == "EMBEDDING":
        values = (functional.embedding(inputs[0], parameter_values[0]),)
    elif opcode == "SCALE":
        scalar = parameter_values[0].to(inputs[0].dtype) if parameter_values else attributes["scalar"]
        values = (inputs[0] * scalar,)
    elif opcode == "ROTARY_TABLE":
        value, position_ids = inputs
        inverse_frequency = parameter_values[0]
        expanded_frequency = inverse_frequency[None, :, None].float().expand(value.shape[0], -1, 1).to(value.device)
        expanded_position = position_ids[:, None, :].float()
        position_scaling = attributes.get("position_scaling")
        if position_scaling is not None:
            expanded_position = expanded_position / float(position_scaling)
        frequency = (expanded_frequency @ expanded_position).transpose(1, 2)
        embedding = torch.cat((frequency, frequency), dim=-1)
        scaling = attributes["attention_scaling"]
        values = (
            (embedding.cos() * scaling).to(value.dtype),
            (embedding.sin() * scaling).to(value.dtype),
        )
    elif opcode == "CAUSAL_MASK":
        mask_input_ids, hidden_states = inputs
        sequence = mask_input_ids.shape[1]
        dtype_name = hidden_states.dtype
        mask_device = hidden_states.device
        row = torch.arange(sequence, device=mask_device)[:, None]
        column = torch.arange(sequence, device=mask_device)[None, :]
        allowed = column <= row
        window = attributes["sliding_window"]
        if window is not None:
            allowed = allowed & (column > row - window)
        zero = torch.zeros((sequence, sequence), dtype=dtype_name, device=mask_device)
        blocked = torch.full_like(zero, torch.finfo(dtype_name).min)
        values = (torch.where(allowed, zero, blocked)[None, None, :, :],)
    elif opcode == "RMS_NORM":
        value = inputs[0]
        normalized = value.float() * torch.rsqrt(
            value.float().pow(2).mean(-1, keepdim=True) + attributes["epsilon"]
        )
        values = ((normalized * (1.0 + parameter_values[0].float())).type_as(value),)
    elif opcode == "RMS_NORM_PLAIN":
        # Qwen/Llama RMSNorm order: cast the normalized stream to input dtype
        # first, then apply the weight in input dtype — distinct from the
        # Gemma (1 + weight) multiply-in-fp32 ordering.
        value = inputs[0]
        normalized = value.float() * torch.rsqrt(
            value.float().pow(2).mean(-1, keepdim=True) + attributes["epsilon"]
        )
        values = (parameter_values[0] * normalized.type_as(value),)
    elif opcode == "LINEAR":
        values = (functional.linear(inputs[0], parameter_values[0]),)
    elif opcode == "LINEAR_BIAS":
        values = (
            functional.linear(inputs[0], parameter_values[0], parameter_values[1]),
        )
    elif opcode == "RESHAPE_TRANSPOSE_HEADS":
        value = inputs[0]
        batch, sequence, _ = value.shape
        values = (
            value.view(
                batch,
                sequence,
                attributes["heads"],
                attributes["head_dimension"],
            ).transpose(1, 2),
        )
    elif opcode == "ROTARY_APPLY_PAIR":
        query, key, cosine, sine = inputs
        cosine = cosine.unsqueeze(1)
        sine = sine.unsqueeze(1)
        values = (
            query * cosine + _rotate_half(query) * sine,
            key * cosine + _rotate_half(key) * sine,
        )
    elif opcode == "REPEAT_KV":
        value = inputs[0]
        repetitions = attributes["repetitions"]
        if repetitions == 1:
            values = (value,)
        else:
            batch, heads, sequence, dimension = value.shape
            expanded = value[:, :, None, :, :].expand(
                batch, heads, repetitions, sequence, dimension
            )
            values = (expanded.reshape(batch, heads * repetitions, sequence, dimension),)
    elif opcode == "MATMUL_QK":
        values = (torch.matmul(inputs[0], inputs[1].transpose(2, 3)),)
    elif opcode == "SOFTCAP":
        values = (torch.tanh(inputs[0] / attributes["cap"]) * attributes["cap"],)
    elif opcode == "ADD":
        values = (inputs[0] + inputs[1],)
    elif opcode == "SOFTMAX":
        values = (functional.softmax(inputs[0], dim=attributes["axis"], dtype=torch.float32).to(inputs[0].dtype),)
    elif opcode == "MATMUL_AV":
        values = (torch.matmul(inputs[0], inputs[1]),)
    elif opcode == "TRANSPOSE_RESHAPE_HEADS":
        value = inputs[0].transpose(1, 2).contiguous()
        values = (value.reshape(value.shape[0], value.shape[1], -1).contiguous(),)
    elif opcode == "GELU_TANH":
        values = (functional.gelu(inputs[0], approximate="tanh"),)
    elif opcode == "SILU":
        values = (functional.silu(inputs[0]),)
    elif opcode == "MUL":
        values = (inputs[0] * inputs[1],)
    elif opcode == "SLICE_LAST_TOKEN":
        values = (inputs[0][:, -1:, :],)
    elif opcode == "ARGMAX":
        values = (torch.argmax(inputs[0][:, -1, :], dim=-1),)
    else:
        raise ValueError(f"No numerical interpreter for opcode {opcode}")
    if len(values) != len(output_names):
        raise RuntimeError(f"Interpreter output arity mismatch for {instruction['id']}")
    return dict(zip(output_names, values))


def execute_gemma_ir(
    program: dict[str, Any],
    parameters: Mapping[str, Any],
    input_ids: Any,
) -> GemmaIrExecution:
    _require_torch()
    verification = verify_gemma_ir(program)
    if not verification["valid"]:
        raise ValueError("Cannot execute an invalid Gemma IR")
    if not isinstance(input_ids, torch.Tensor) or input_ids.dtype != torch.int64 or input_ids.ndim != 2:
        raise ValueError("Gemma IR input_ids must be a rank-two torch.int64 tensor")
    vocabulary_size = program["configuration"]["vocabulary_size"]
    if input_ids.numel() == 0 or bool(
        torch.any((input_ids < 0) | (input_ids >= vocabulary_size)).item()
    ):
        raise ValueError("Gemma IR input_ids contain values outside the declared vocabulary")
    if set(parameters) != set(program["parameter_commitments"]):
        raise ValueError("Interpreter parameter names do not match the Gemma IR")
    states: dict[str, Any] = {"input_ids": input_ids}
    records: list[dict[str, Any]] = []
    symbols = {"B": input_ids.shape[0], "S": input_ids.shape[1]}
    for instruction in program["instructions"]:
        outputs = _execute_instruction(instruction, states, parameters)
        for name, value in outputs.items():
            declaration = program["tensors"][name]
            expected_shape = [symbols.get(dimension, dimension) for dimension in declaration["shape"]]
            if list(value.shape) != expected_shape or str(value.dtype) != declaration["dtype"]:
                raise RuntimeError(
                    f"Interpreter result violates IR tensor declaration for {name}: "
                    f"shape={list(value.shape)}, dtype={value.dtype}"
                )
        states.update(outputs)
        append_chain_record(
            records,
            {
                "instruction_id": instruction["id"],
                "instruction_sha256": instruction["instruction_sha256"],
                "opcode": instruction["opcode"],
                "layer": instruction["layer"],
                "inputs": instruction["inputs"],
                "parameter_refs": instruction["parameter_refs"],
                "outputs": {
                    name: tensor_descriptor(value) for name, value in outputs.items()
                },
            },
        )
    root = records[-1]["record_hash"]
    chain = verify_trace_chain(records, root)
    if not chain["valid"]:
        raise RuntimeError("Gemma IR execution witness failed internal chain verification")
    logits_name = program["declared_outputs"][0]
    return GemmaIrExecution(
        logits=states[logits_name],
        selected_token_id=states["selected_token_id"],
        records=tuple(records),
        root_sha256=root,
    )


PROJECTION_SLICE_SCOPE = "Fixed-input actual Gemma layer-0 Q/K/V projection comparison; shared PyTorch embedding/scale/input-RMS prefix, independent experimental projection arithmetic, original modules compared only after predictions are saved; not full-layer/model or hardware qualification."
PROJECTION_SLICE_ROLES = (("query", "q", 1024), ("key", "k", 256), ("value", "v", 256))


def _projection_prefix(program: dict[str, Any], parameters: Mapping[str, Any], input_ids: Any) -> tuple[Any, list[dict[str, Any]]]:
    if not verify_gemma_ir(program)["valid"]:
        raise ValueError("Projection prefix requires a valid IR")
    if not isinstance(input_ids, torch.Tensor) or input_ids.dtype != torch.int64 or input_ids.ndim != 2 or input_ids.numel() == 0:
        raise ValueError("Projection prefix requires rank-two integer token IDs")
    if bool(torch.any((input_ids < 0) | (input_ids >= program["configuration"]["vocabulary_size"])).item()):
        raise ValueError("Projection input IDs are outside the vocabulary")
    target = "layer.0.attention.normalized"
    needed, selected = {target}, []
    for instruction in reversed(program["instructions"]):
        if needed.intersection(instruction["outputs"]):
            selected.append(instruction)
            needed.difference_update(instruction["outputs"])
            needed.update(instruction["inputs"])
    if needed != {"input_ids"}:
        raise ValueError("Projection prefix dependency boundary mismatch")
    states, records = {"input_ids": input_ids}, []
    symbols = {"B": input_ids.shape[0], "S": input_ids.shape[1]}
    for instruction in reversed(selected):
        if instruction["opcode"] not in ("EMBEDDING", "SCALE", "RMS_NORM"):
            raise ValueError("Unsupported shared-prefix operation")
        outputs = _execute_instruction(instruction, states, parameters)
        for name, value in outputs.items():
            declaration = program["tensors"][name]
            shape = [symbols.get(dimension, dimension) for dimension in declaration["shape"]]
            if list(value.shape) != shape or str(value.dtype) != declaration["dtype"]:
                raise ValueError("Projection prefix violates an IR tensor declaration")
        states.update(outputs)
        append_chain_record(records, {"provider": "shared_pytorch_ir_prefix", "instruction_id": instruction["id"],
                                      "instruction_sha256": instruction["instruction_sha256"], "opcode": instruction["opcode"],
                                      "inputs": instruction["inputs"], "parameter_refs": instruction["parameter_refs"],
                                      "outputs": {name: tensor_descriptor(value) for name, value in outputs.items()}})
    return states[target], records


def _projection_arithmetic_commitment() -> str:
    import inspect
    from . import gemma_wmma_candidate as numerical, gemma_float_semantics as scalar, gemma_reduction_semantics as reductions

    names = ("operand_aligned_product_bits", "_operand_aligned_accumulator", "_float32_carry", "_floor_log2",
             "_at_least_power_of_two", "_scale_power_of_two", "_round_units", "split_k_candidate_bits", "_split_partitions", "_merge_split_partials")
    return _sha256({"functions": {name: inspect.getsource(getattr(numerical, name)) for name in names},
                    "scalar_module": inspect.getsource(scalar), "reduction_module": inspect.getsource(reductions)})


def _projection_environment() -> dict[str, Any]:
    import os
    import platform
    import transformers

    return {"torch": str(torch.__version__), "cuda": torch.version.cuda, "transformers": transformers.__version__,
            "python": platform.python_version(), "device": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32, "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_autocast_enabled": torch.is_autocast_enabled(), "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def _projection_context(program: dict[str, Any], model: Any, token_ids: list[list[int]]) -> tuple[dict[str, Any], Any, list[dict[str, Any]], dict[str, Any]]:
    _require_torch()
    if not _rectangular_integer_rows(token_ids) or len(token_ids) != 1 or len(token_ids[0]) != 30:
        raise ValueError("Projection slice is restricted to one 30-token input")
    if model.training or model.config._attn_implementation != "eager":
        raise ValueError("Projection slice requires evaluation mode and declared eager attention")
    parameters = bind_model_tensors(program, model, verify_hashes=True)
    device = parameters["model.embed_tokens.weight"].device
    if device.type != "cuda" or device.index != torch.cuda.current_device():
        raise ValueError("Projection slice requires the current CUDA device")
    ids = torch.tensor(token_ids, dtype=torch.int64, device=device)
    with torch.no_grad():
        normalized, records = _projection_prefix(program, parameters, ids)
    if list(normalized.shape) != [1, 30, 640] or normalized.dtype != torch.bfloat16:
        raise ValueError("Unsupported projection input shape or dtype")
    instructions = {}
    for role, short, width in PROJECTION_SLICE_ROLES:
        module = getattr(model.model.layers[0].self_attn, f"{short}_proj")
        name = f"model.layers.0.self_attn.{short}_proj.weight"
        if type(module) is not torch.nn.Linear or module.bias is not None or list(module.weight.shape) != [width, 640] or module.weight.dtype != torch.bfloat16 or module.weight.device != device:
            raise ValueError("Unsupported actual projection module")
        if module.weight is not parameters[name]:
            raise ValueError("Original projection does not use the bound parameter")
        found = [item for item in program["instructions"] if item["outputs"] == [f"layer.0.{role}.flat"]]
        if len(found) != 1 or found[0]["opcode"] != "LINEAR" or found[0]["inputs"] != ["layer.0.attention.normalized"] or found[0]["parameter_refs"] != [name]:
            raise ValueError("Actual projection instruction binding mismatch")
        instructions[role] = found[0]
    return parameters, normalized, records, instructions


def _projection_bits_tensor(bits: Any, shape: list[int], device: Any) -> Any:
    if not isinstance(bits, list) or len(bits) != shape[0] or any(
        not isinstance(rows, list) or len(rows) != shape[1] or any(not isinstance(row, list) or len(row) != shape[2] or
        any(type(value) is not int or not 0 <= value <= 65535 for value in row) for row in rows) for rows in bits
    ):
        raise ValueError("Malformed projection bit tensor")
    return torch.tensor(bits, dtype=torch.uint16).view(torch.bfloat16).to(device)


def build_projection_slice(program: dict[str, Any], model: Any, fixture: dict[str, Any], query_evidence: dict[str, Any], split_evidence: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    from .gemma_wmma_candidate import operand_aligned_product_bits, split_k_candidate_bits, OPERAND_ALIGNMENT_PROFILE, DENSE_SPLIT_PROFILE

    for evidence, count in ((query_evidence, 4096), (split_evidence, 1024)):
        if _sha256({k: v for k, v in evidence.items() if k != "report_sha256"}) != evidence.get("report_sha256") or evidence.get("tested_case_count") != count or evidence.get("candidate_passes_within_declared_scope") is not True:
            raise ValueError("Projection slice requires intact passing controlled evidence")
    if fixture.get("program_sha256") != program["program_sha256"]:
        raise ValueError("Input fixture belongs to another IR")
    parameters, normalized, prefix, instructions = _projection_context(program, model, fixture["input_token_ids"])
    input_ids = torch.tensor(fixture["input_token_ids"], dtype=torch.int64, device=normalized.device)
    if tensor_sha256(input_ids) != fixture["input_ids"]["sha256"]:
        raise ValueError("Fixture token commitment mismatch")
    input_bits = normalized.detach().cpu().view(torch.uint16).tolist()
    outputs, predictions = {}, {}
    for role, short, width in PROJECTION_SLICE_ROLES:
        name = f"model.layers.0.self_attn.{short}_proj.weight"
        weights = parameters[name].detach().cpu().view(torch.uint16).tolist()
        oracle = operand_aligned_product_bits if role == "query" else lambda left, right: split_k_candidate_bits(left, right, *DENSE_SPLIT_PROFILE)
        bits = [[[oracle(row, weight) for weight in weights] for row in input_bits[0]]]
        predictions[role] = bits
        predicted = _projection_bits_tensor(bits, [1, 30, width], normalized.device)
        evidence = query_evidence if role == "query" else split_evidence
        outputs[role] = {"instruction_id": instructions[role]["id"], "instruction_sha256": instructions[role]["instruction_sha256"],
                         "input_tensor": "layer.0.attention.normalized", "output_tensor": f"layer.0.{role}.flat",
                         "weight_name": name, "weight": tensor_descriptor(parameters[name]), "prediction": tensor_descriptor(predicted),
                         "numerical_provider": "operand_alignment_v1" if role == "query" else "split_k64_bf16_serial_fp32_v1",
                         "expected_kernel_names": evidence["cuda_kernel_names"][0], "expected_environment": evidence["environment"],
                         "source_evidence_sha256": evidence["report_sha256"]}
    bundle = {"program_sha256": program["program_sha256"], "input_token_ids": fixture["input_token_ids"],
              "normalized_input_bits": input_bits, "prediction_bits": predictions}
    body = {"schema_version": 1, "scope": PROJECTION_SLICE_SCOPE, "program_sha256": program["program_sha256"],
            "input_fixture_sha256": _sha256(fixture), "input_token_ids": fixture["input_token_ids"], "input_ids": tensor_descriptor(input_ids),
            "model_binding": {"class": type(model).__module__ + "." + type(model).__name__, "config_sha256": _sha256(model.config.to_dict()),
                              "parameter_commitments_sha256": _sha256(program["parameter_commitments"])},
            "normalized_input": tensor_descriptor(normalized), "shared_prefix_records": prefix, "shared_prefix_root": prefix[-1]["record_hash"],
            "projection_records": outputs, "numeric_profile": {"partial": dict(OPERAND_ALIGNMENT_PROFILE), "split": list(DENSE_SPLIT_PROFILE)},
            "arithmetic_implementation_sha256": _projection_arithmetic_commitment(),
            "runtime": _projection_environment(), "bundle_sha256": _sha256(bundle), "predicted_value_count": 46080,
            "source_evidence_verification": "commitments_only; use the source experiment verifiers for source replay",
            "shared_prefix_independently_qualified": False, "candidate_refitting_allowed": False,
            "full_first_layer_qualified": False, "global_exactness_activation_allowed": False}
    return {**body, "plan_sha256": _sha256(body)}, bundle


def _check_projection_plan(program: dict[str, Any], plan: dict[str, Any], bundle: dict[str, Any]) -> None:
    from .gemma_wmma_candidate import OPERAND_ALIGNMENT_PROFILE, DENSE_SPLIT_PROFILE

    if plan.get("scope") != PROJECTION_SLICE_SCOPE or plan.get("predicted_value_count") != 46080 or any(plan.get(key) is not False for key in ("shared_prefix_independently_qualified", "candidate_refitting_allowed", "full_first_layer_qualified", "global_exactness_activation_allowed")):
        raise ValueError("Projection scope or qualification boundary mismatch")
    if plan.get("arithmetic_implementation_sha256") != _projection_arithmetic_commitment() or canonical_json(plan["numeric_profile"]) != canonical_json({"partial": OPERAND_ALIGNMENT_PROFILE, "split": list(DENSE_SPLIT_PROFILE)}):
        raise ValueError("Projection arithmetic implementation or profile mismatch")
    if not verify_gemma_ir(program)["valid"] or plan["program_sha256"] != program["program_sha256"]:
        raise ValueError("Projection program commitment mismatch")
    if _sha256({k: v for k, v in plan.items() if k != "plan_sha256"}) != plan["plan_sha256"] or _sha256(bundle) != plan["bundle_sha256"]:
        raise ValueError("Projection plan or tensor-bundle commitment mismatch")
    if bundle["program_sha256"] != plan["program_sha256"] or canonical_json(bundle["input_token_ids"]) != canonical_json(plan["input_token_ids"]):
        raise ValueError("Projection bundle identity mismatch")
    ids = plan["input_token_ids"]
    if not _rectangular_integer_rows(ids) or len(ids) != 1 or len(ids[0]) != 30 or any(not 0 <= value < program["configuration"]["vocabulary_size"] for value in ids[0]):
        raise ValueError("Projection token domain mismatch")
    if any(plan["input_ids"].get(key) != value for key, value in tensor_descriptor(torch.tensor(ids, dtype=torch.int64)).items() if key != "device") or plan["model_binding"]["parameter_commitments_sha256"] != _sha256(program["parameter_commitments"]):
        raise ValueError("Projection token or model commitment mismatch")
    if not verify_trace_chain(plan["shared_prefix_records"], plan["shared_prefix_root"])["valid"]:
        raise ValueError("Projection shared-prefix witness mismatch")
    needed, selected = {"layer.0.attention.normalized"}, []
    for instruction in reversed(program["instructions"]):
        if needed.intersection(instruction["outputs"]):
            selected.append(instruction)
            needed.difference_update(instruction["outputs"])
            needed.update(instruction["inputs"])
    if len(selected) != len(plan["shared_prefix_records"]) or needed != {"input_ids"}:
        raise ValueError("Projection prefix coverage mismatch")
    for instruction, record in zip(reversed(selected), plan["shared_prefix_records"]):
        payload = record["payload"]
        if payload.get("provider") != "shared_pytorch_ir_prefix" or any(payload.get(key) != instruction[key] for key in ("opcode", "inputs", "parameter_refs")) or payload.get("instruction_id") != instruction["id"] or payload.get("instruction_sha256") != instruction["instruction_sha256"] or set(payload.get("outputs", {})) != set(instruction["outputs"]):
            raise ValueError("Projection prefix is not bound to the declared IR")
    if canonical_json(plan["shared_prefix_records"][-1]["payload"]["outputs"]["layer.0.attention.normalized"]) != canonical_json(plan["normalized_input"]):
        raise ValueError("Projection boundary is not the shared-prefix output")
    if set(plan["projection_records"]) != {role for role, _, _ in PROJECTION_SLICE_ROLES} or set(bundle["prediction_bits"]) != set(plan["projection_records"]):
        raise ValueError("Missing or substituted projection roles")
    normalized = _projection_bits_tensor(bundle["normalized_input_bits"], [1, 30, 640], "cpu")
    if any(plan["normalized_input"].get(key) != value for key, value in tensor_descriptor(normalized).items() if key != "device"):
        raise ValueError("Projection normalized input bits or metadata mismatch")
    for role, short, width in PROJECTION_SLICE_ROLES:
        record = plan["projection_records"][role]
        name = f"model.layers.0.self_attn.{short}_proj.weight"
        instruction = next(item for item in program["instructions"] if item["outputs"] == [f"layer.0.{role}.flat"])
        prediction = _projection_bits_tensor(bundle["prediction_bits"][role], [1, 30, width], "cpu")
        provider = "operand_alignment_v1" if role == "query" else "split_k64_bf16_serial_fp32_v1"
        if any(record["prediction"].get(key) != value for key, value in tensor_descriptor(prediction).items() if key != "device") or any(record["weight"].get(key) != program["parameter_commitments"][name][key] for key in ("shape", "dtype")):
            raise ValueError("Projection tensor metadata mismatch")
        if record["weight_name"] != name or record["weight"]["sha256"] != program["parameter_commitments"][name]["sha256"] or record["instruction_sha256"] != instruction["instruction_sha256"] or record["instruction_id"] != instruction["id"] or record["numerical_provider"] != provider or record["input_tensor"] != "layer.0.attention.normalized" or record["output_tensor"] != instruction["outputs"][0] or tensor_sha256(prediction) != record["prediction"]["sha256"]:
            raise ValueError("Projection instruction, weight, or prediction binding mismatch")


def _projection_slice_report(plan: dict[str, Any], bundle: dict[str, Any], observations: dict[str, Any], actual_prefix: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(observations, dict) or set(observations) != set(plan["projection_records"]):
        raise ValueError("Projection comparison is missing roles")
    if not isinstance(actual_prefix, dict) or not isinstance(runtime, dict):
        raise ValueError("Missing projection prefix or runtime metadata")
    matches, differences = {}, []
    for role, _, width in PROJECTION_SLICE_ROLES:
        item = observations[role]
        if not isinstance(item["bits"], list) or len(item["bits"]) != 3 or not isinstance(item["kernel_names"], list) or len(item["kernel_names"]) != 3 or any(not isinstance(names, list) or not names or any(not isinstance(name, str) or not name for name in names) for names in item["kernel_names"]):
            raise ValueError("Missing projection repetitions or kernel names")
        predicted = _projection_bits_tensor(bundle["prediction_bits"][role], [1, 30, width], "cpu").view(torch.int16)
        for repetition, bits in enumerate(item["bits"]):
            actual = _projection_bits_tensor(bits, [1, 30, width], "cpu").view(torch.int16)
            for batch, row, column in (actual != predicted).nonzero().tolist():
                differences.append({"role": role, "repetition": repetition, "row": row, "column": column,
                                    "predicted_bits": int(predicted[batch, row, column]) & 65535, "observed_bits": int(actual[batch, row, column]) & 65535})
        expected = plan["projection_records"][role]
        matches[role] = {"value_count": 30 * width,
                         "repeated_outputs_identical": all(bits == item["bits"][0] for bits in item["bits"]),
                         "kernel_names_match": all(names == expected["expected_kernel_names"] for names in item["kernel_names"]),
                         "source_environment_match": all(canonical_json(runtime.get(key)) == canonical_json(value) for key, value in expected["expected_environment"].items())}
    prefix_match = canonical_json(actual_prefix) == canonical_json(plan["normalized_input"])
    scope_match = canonical_json(runtime) == canonical_json(plan["runtime"]) and all(item["kernel_names_match"] and item["source_environment_match"] for item in matches.values())
    body = {"schema_version": 1, "scope": PROJECTION_SLICE_SCOPE, "plan_sha256": plan["plan_sha256"], "bundle_sha256": plan["bundle_sha256"],
            "original_prefix_input": actual_prefix, "original_prefix_matches_shared_ir": prefix_match,
            "observations": observations, "runtime": runtime, "projection_summaries": matches,
            "mismatch_count": len(differences), "mismatches": differences, "projection_values_bit_exact": not differences,
            "source_kernel_environment_match": scope_match, "slice_passes_declared_comparison": prefix_match and not differences and scope_match,
            "first_divergence": ("shared_prefix" if not prefix_match else differences[0] if differences else None),
            "compared_values_per_repetition": 46080, "shared_prefix_independently_qualified": False,
            "full_first_layer_qualified": False, "full_model_independently_qualified": False,
            "hardware_semantics_established": False, "global_exactness_activation_allowed": False}
    return {**body, "report_sha256": _sha256(body)}


def acquire_projection_slice(program: dict[str, Any], model: Any, plan: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    from .gemma_reduction_backend import _profile_call

    _check_projection_plan(program, plan, bundle)
    parameters, normalized, prefix, _ = _projection_context(program, model, plan["input_token_ids"])
    binding = {"class": type(model).__module__ + "." + type(model).__name__, "config_sha256": _sha256(model.config.to_dict()),
               "parameter_commitments_sha256": _sha256(program["parameter_commitments"])}
    if binding != plan["model_binding"] or canonical_json(prefix) != canonical_json(plan["shared_prefix_records"]) or tensor_descriptor(normalized) != plan["normalized_input"]:
        raise ValueError("Model or shared prefix no longer matches the prediction plan")
    captured = []

    class BoundaryReached(Exception):
        pass

    def capture(module: Any, args: Any) -> None:
        captured.append(args[0].detach().clone())
        raise BoundaryReached()

    handle = model.model.layers[0].self_attn.q_proj.register_forward_pre_hook(capture)
    try:
        with torch.no_grad():
            model(input_ids=torch.tensor(plan["input_token_ids"], dtype=torch.int64, device=normalized.device),
                  attention_mask=torch.ones((1, 30), dtype=torch.int64, device=normalized.device), use_cache=False, logits_to_keep=1)
    except BoundaryReached:
        pass
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError("Original model did not reach exactly one projection boundary")
    observations = {}
    with torch.no_grad():
        for role, short, width in PROJECTION_SLICE_ROLES:
            module = getattr(model.model.layers[0].self_attn, f"{short}_proj")
            bits, kernels = [], []
            for _ in range(3):
                output, names = _profile_call(lambda: module(captured[0]))
                if output.dtype != torch.bfloat16 or list(output.shape) != [1, 30, width] or output.device != normalized.device:
                    raise ValueError("Original projection output violates the declared tensor type")
                bits.append(output.detach().cpu().view(torch.uint16).tolist())
                kernels.append(names)
            observations[role] = {"bits": bits, "kernel_names": kernels}
    bind_model_tensors(program, model, verify_hashes=True)
    return _projection_slice_report(plan, bundle, observations, tensor_descriptor(captured[0]), _projection_environment())


def verify_projection_slice(program: dict[str, Any], plan: dict[str, Any], bundle: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    try:
        _check_projection_plan(program, plan, bundle)
        expected = _projection_slice_report(plan, bundle, report["observations"], report["original_prefix_input"], report["runtime"])
        valid = canonical_json(expected) == canonical_json(report)
        return {"valid": valid, "mode": "integrity_only", "slice_passes_declared_comparison": expected["slice_passes_declared_comparison"],
                "projection_summaries": expected["projection_summaries"], "mismatch_count": expected["mismatch_count"],
                "first_divergence": expected["first_divergence"], "global_exactness_activation_allowed": False}
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError, StopIteration) as error:
        return {"valid": False, "reason": str(error)}


def projection_slice_summary(plan: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in report.items() if key not in ("observations", "mismatches", "report_sha256")}
    body.update({"source_report_sha256": report["report_sha256"], "program_sha256": plan["program_sha256"],
                 "model_binding": plan["model_binding"], "input_token_ids": plan["input_token_ids"],
                 "projection_records": plan["projection_records"], "arithmetic_implementation_sha256": plan["arithmetic_implementation_sha256"],
                 "observed_output_hashes": {role: [tensor_sha256(_projection_bits_tensor(bits, [1, 30, width], "cpu")) for bits in report["observations"][role]["bits"]] for role, _, width in PROJECTION_SLICE_ROLES}})
    return {**body, "summary_sha256": _sha256(body)}


def compare_ir_execution(
    program: dict[str, Any], execution: GemmaIrExecution, expected_records: list[dict[str, Any]]
) -> dict[str, Any]:
    actual = list(execution.records)
    first_divergence = None
    compared = min(len(actual), len(expected_records))
    for index in range(compared):
        actual_payload = actual[index].get("payload", {})
        expected_payload = expected_records[index].get("payload", {})
        if (
            actual_payload.get("instruction_id") != expected_payload.get("instruction_id")
            or actual_payload.get("outputs") != expected_payload.get("outputs")
        ):
            first_divergence = {
                "index": index,
                "instruction_id": actual_payload.get("instruction_id"),
                "actual_outputs": actual_payload.get("outputs"),
                "expected_outputs": expected_payload.get("outputs"),
            }
            break
    if first_divergence is None and len(actual) != len(expected_records):
        first_divergence = {
            "index": compared,
            "instruction_id": None,
            "actual_record_count": len(actual),
            "expected_record_count": len(expected_records),
        }
    return {
        "exact_match": first_divergence is None,
        "first_divergence": first_divergence,
        "actual_record_count": len(actual),
        "expected_record_count": len(expected_records),
        "program_sha256": program["program_sha256"],
    }


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _rectangular_integer_rows(value: Any) -> bool:
    return bool(
        isinstance(value, list)
        and value
        and all(
            isinstance(row, list)
            and row
            and all(isinstance(token, int) and not isinstance(token, bool) for token in row)
            for row in value
        )
        and len({len(row) for row in value}) == 1
    )


def _selection_proofs(logits: Any, selected_token_ids: Any) -> list[dict[str, Any]]:
    proofs = []
    for batch_index in range(logits.shape[0]):
        values = logits[batch_index, -1, :]
        selected_id = int(selected_token_ids[batch_index].item())
        candidate_ids = torch.arange(values.shape[0], device=values.device)
        candidate_ids = candidate_ids[candidate_ids != selected_id]
        if candidate_ids.numel() == 0:
            raise ValueError("Token-selection proof requires at least two vocabulary entries")
        competitor_values = values[candidate_ids]
        runner_up_position = int(torch.argmax(competitor_values).item())
        runner_up_id = int(candidate_ids[runner_up_position].item())
        selected_value = values[selected_id]
        runner_up_value = values[runner_up_id]
        selected_float = float(selected_value.float().item())
        runner_up_float = float(runner_up_value.float().item())
        proofs.append(
            {
                "batch_index": batch_index,
                "selection_rule": "lowest token index attaining the maximum logit",
                "selected_token_id": selected_id,
                "selected_logit": selected_float,
                "selected_logit_descriptor": tensor_descriptor(
                    selected_value.reshape(1)
                ),
                "runner_up_token_id": runner_up_id,
                "runner_up_logit": runner_up_float,
                "runner_up_logit_descriptor": tensor_descriptor(
                    runner_up_value.reshape(1)
                ),
                "winning_margin_float32": selected_float - runner_up_float,
                "strict_winner": selected_float > runner_up_float,
                "argmax_recomputed": int(torch.argmax(values).item()) == selected_id,
                "all_competitors_not_greater": bool(
                    torch.all(competitor_values <= selected_value).item()
                ),
            }
        )
    return proofs


def _scalar_descriptor_matches(value: float, descriptor: Any) -> bool:
    if not isinstance(descriptor, dict):
        return False
    dtype_name = descriptor.get("dtype")
    if not isinstance(dtype_name, str) or not dtype_name.startswith("torch."):
        return False
    dtype = getattr(torch, dtype_name.removeprefix("torch."), None)
    if dtype is None:
        return False
    reconstructed = tensor_descriptor(torch.tensor([value], dtype=dtype))
    return bool(
        descriptor.get("sha256") == reconstructed.get("sha256")
        and descriptor.get("shape") == [1]
        and descriptor.get("dtype") == reconstructed.get("dtype")
    )


def _selection_proofs_valid(
    proofs: Any, predicted_token_ids: Any, vocabulary_size: int
) -> bool:
    return bool(
        isinstance(proofs, list)
        and isinstance(predicted_token_ids, list)
        and len(proofs) == len(predicted_token_ids)
        and all(
            isinstance(proof, dict)
            and isinstance(proof.get("batch_index"), int)
            and not isinstance(proof.get("batch_index"), bool)
            and proof.get("batch_index") == index
            and isinstance(proof.get("selected_token_id"), int)
            and not isinstance(proof.get("selected_token_id"), bool)
            and proof.get("selected_token_id") == predicted_token_ids[index]
            and 0 <= proof["selected_token_id"] < vocabulary_size
            and isinstance(proof.get("runner_up_token_id"), int)
            and not isinstance(proof.get("runner_up_token_id"), bool)
            and 0 <= proof["runner_up_token_id"] < vocabulary_size
            and proof.get("runner_up_token_id") != predicted_token_ids[index]
            and isinstance(proof.get("selected_logit"), (int, float))
            and not isinstance(proof.get("selected_logit"), bool)
            and math.isfinite(proof["selected_logit"])
            and isinstance(proof.get("runner_up_logit"), (int, float))
            and not isinstance(proof.get("runner_up_logit"), bool)
            and math.isfinite(proof["runner_up_logit"])
            and proof.get("winning_margin_float32")
            == proof["selected_logit"] - proof["runner_up_logit"]
            and proof.get("strict_winner")
            is (proof["selected_logit"] > proof["runner_up_logit"])
            and proof.get("argmax_recomputed") is True
            and proof.get("all_competitors_not_greater") is True
            and _scalar_descriptor_matches(
                proof["selected_logit"], proof.get("selected_logit_descriptor")
            )
            and _scalar_descriptor_matches(
                proof["runner_up_logit"], proof.get("runner_up_logit_descriptor")
            )
            for index, proof in enumerate(proofs)
        )
    )


def build_ir_execution_certificate(
    program: dict[str, Any],
    input_ids: Any,
    execution: GemmaIrExecution,
    observed_logits: Any,
    execution_profile: str,
    model_state_sha256: str,
) -> dict[str, Any]:
    _require_torch()
    if execution_profile != "huggingface_eager":
        raise ValueError("The current IR correspondence profile is restricted to huggingface_eager")
    if HASH_PATTERN.fullmatch(model_state_sha256) is None:
        raise ValueError("Invalid model-state SHA-256")
    if len(execution.records) != len(program["instructions"]):
        raise ValueError("IR execution witness does not cover every program instruction")
    predicted_descriptor = tensor_descriptor(execution.logits)
    observed_descriptor = tensor_descriptor(observed_logits)
    predicted_token_descriptor = tensor_descriptor(execution.selected_token_id)
    predicted_tokens = execution.selected_token_id.detach().cpu().tolist()
    observed_tokens = torch.argmax(observed_logits[:, -1, :], dim=-1).detach().cpu().tolist()
    logits_exact = bool(torch.equal(execution.logits, observed_logits))
    token_exact = predicted_tokens == observed_tokens
    body = {
        "schema_version": 1,
        "scope": "Fixed-input exact prediction certificate for the canonical typed Gemma IR against a pinned Hugging Face eager execution; no SDPA, fused-kernel, hardware-semantic, scientific, or regulatory equivalence is established.",
        "program_sha256": program["program_sha256"],
        "model_state_sha256": model_state_sha256,
        "execution_profile": execution_profile,
        "input_ids": tensor_descriptor(input_ids),
        "input_token_ids": input_ids.detach().cpu().tolist(),
        "predicted_logits": predicted_descriptor,
        "observed_logits": observed_descriptor,
        "predicted_token_ids": predicted_tokens,
        "predicted_token_ids_descriptor": predicted_token_descriptor,
        "observed_token_ids": observed_tokens,
        "logits_bit_exact": logits_exact,
        "selected_tokens_exact": token_exact,
        "selection_proofs": _selection_proofs(
            execution.logits, execution.selected_token_id
        ),
        "first_divergence": None if logits_exact else "final_logits",
        "execution_records": list(execution.records),
        "execution_root_sha256": execution.root_sha256,
        "instruction_coverage_count": len(execution.records),
        "instruction_coverage_complete": len(execution.records) == len(program["instructions"]),
        "fixed_input_canonical_eager_prediction_established": logits_exact and token_exact,
        "deployed_sdpa_correspondence_established": False,
        "hardware_instruction_semantics_established": False,
    }
    return {**body, "certificate_sha256": _sha256(body)}


def verify_ir_execution_certificate(
    program: dict[str, Any], certificate: dict[str, Any]
) -> dict[str, Any]:
    _require_torch()
    program_verification = verify_gemma_ir(program)
    if not program_verification["valid"] or not isinstance(certificate, dict):
        return {"valid": False}
    body = {
        key: value for key, value in certificate.items() if key != "certificate_sha256"
    }
    try:
        certificate_hash_valid = _sha256(body) == certificate.get("certificate_sha256")
    except (TypeError, ValueError):
        return {"valid": False}
    records = certificate.get("execution_records")
    chain = (
        verify_trace_chain(records, certificate.get("execution_root_sha256"))
        if isinstance(records, list)
        else {"valid": False}
    )
    def instruction_record_matches(
        record: Any, instruction: dict[str, Any]
    ) -> bool:
        return bool(
            isinstance(record, dict)
            and isinstance(record.get("payload"), dict)
            and record["payload"].get("instruction_id") == instruction.get("id")
            and record["payload"].get("instruction_sha256")
            == instruction.get("instruction_sha256")
            and record["payload"].get("opcode") == instruction.get("opcode")
            and record["payload"].get("layer") == instruction.get("layer")
            and record["payload"].get("inputs") == instruction.get("inputs")
            and record["payload"].get("parameter_refs")
            == instruction.get("parameter_refs")
            and isinstance(record["payload"].get("outputs"), dict)
            and set(record["payload"]["outputs"])
            == set(instruction.get("outputs", []))
        )

    binding_mismatches = [
        index
        for index, (record, instruction) in enumerate(
            zip(records if isinstance(records, list) else [], program.get("instructions", []))
        )
        if not instruction_record_matches(record, instruction)
    ]
    instruction_binding_valid = bool(
        isinstance(records, list)
        and len(records) == len(program.get("instructions", []))
        and not binding_mismatches
    )
    predicted = certificate.get("predicted_logits")
    predicted_token_descriptor = certificate.get("predicted_token_ids_descriptor")
    logits_output = program.get("declared_outputs", [None])[0]
    logits_record_descriptor = None
    token_record_descriptor = None
    if isinstance(records, list) and instruction_binding_valid:
        for record in records:
            outputs = record["payload"].get("outputs", {})
            if logits_output in outputs:
                logits_record_descriptor = outputs[logits_output]
            if "selected_token_id" in outputs:
                token_record_descriptor = outputs["selected_token_id"]
    record_output_binding_valid = bool(
        isinstance(predicted, dict)
        and predicted == logits_record_descriptor
        and isinstance(predicted_token_descriptor, dict)
        and predicted_token_descriptor == token_record_descriptor
    )
    observed = certificate.get("observed_logits")
    calculated_logits_exact = bool(
        isinstance(predicted, dict)
        and isinstance(observed, dict)
        and isinstance(predicted.get("shape"), list)
        and isinstance(predicted.get("dtype"), str)
        and HASH_PATTERN.fullmatch(str(predicted.get("sha256", ""))) is not None
        and predicted.get("sha256") == observed.get("sha256")
        and predicted.get("shape") == observed.get("shape")
        and predicted.get("dtype") == observed.get("dtype")
    )
    predicted_tokens = certificate.get("predicted_token_ids")
    observed_tokens = certificate.get("observed_token_ids")
    calculated_tokens_exact = bool(
        isinstance(predicted_tokens, list)
        and predicted_tokens
        and predicted_tokens == observed_tokens
        and all(isinstance(token, int) and not isinstance(token, bool) for token in predicted_tokens)
    )
    reconstructed_token_descriptor = (
        tensor_descriptor(torch.tensor(predicted_tokens, dtype=torch.int64))
        if calculated_tokens_exact
        else {}
    )
    token_value_binding_valid = bool(
        isinstance(predicted_token_descriptor, dict)
        and predicted_token_descriptor.get("sha256")
        == reconstructed_token_descriptor.get("sha256")
        and predicted_token_descriptor.get("shape")
        == reconstructed_token_descriptor.get("shape")
        and predicted_token_descriptor.get("dtype")
        == reconstructed_token_descriptor.get("dtype")
    )
    input_values = certificate.get("input_token_ids")
    input_descriptor = certificate.get("input_ids")
    input_value_binding_valid = False
    if (
        _rectangular_integer_rows(input_values)
        and isinstance(input_descriptor, dict)
    ):
        reconstructed_input = tensor_descriptor(
            torch.tensor(input_values, dtype=torch.int64)
        )
        input_value_binding_valid = bool(
            input_descriptor.get("sha256") == reconstructed_input.get("sha256")
            and input_descriptor.get("shape") == reconstructed_input.get("shape")
            and input_descriptor.get("dtype") == reconstructed_input.get("dtype")
        )
    selection_proofs_valid = _selection_proofs_valid(
        certificate.get("selection_proofs"),
        predicted_tokens,
        program["configuration"]["vocabulary_size"],
    )
    claims_consistent = bool(
        certificate.get("program_sha256") == program.get("program_sha256")
        and HASH_PATTERN.fullmatch(str(certificate.get("model_state_sha256", "")))
        is not None
        and certificate.get("execution_profile") == "huggingface_eager"
        and certificate.get("logits_bit_exact") is calculated_logits_exact
        and certificate.get("selected_tokens_exact") is calculated_tokens_exact
        and certificate.get("first_divergence")
        == (None if calculated_logits_exact else "final_logits")
        and certificate.get("instruction_coverage_count")
        == len(program.get("instructions", []))
        and certificate.get("instruction_coverage_complete") is True
        and certificate.get("fixed_input_canonical_eager_prediction_established")
        is (calculated_logits_exact and calculated_tokens_exact)
        and certificate.get("deployed_sdpa_correspondence_established") is False
        and certificate.get("hardware_instruction_semantics_established") is False
    )
    valid = all(
        (
            certificate_hash_valid,
            chain.get("valid", False),
            instruction_binding_valid,
            record_output_binding_valid,
            token_value_binding_valid,
            input_value_binding_valid,
            selection_proofs_valid,
            claims_consistent,
            calculated_logits_exact,
            calculated_tokens_exact,
        )
    )
    return {
        "valid": valid,
        "certificate_hash_valid": certificate_hash_valid,
        "execution_chain_valid": chain.get("valid", False),
        "instruction_binding_valid": instruction_binding_valid,
        "first_instruction_binding_mismatch": (
            binding_mismatches[0] if binding_mismatches else None
        ),
        "record_output_binding_valid": record_output_binding_valid,
        "token_value_binding_valid": token_value_binding_valid,
        "input_value_binding_valid": input_value_binding_valid,
        "selection_proofs_valid": selection_proofs_valid,
        "claims_consistent": claims_consistent,
        "logits_bit_exact": calculated_logits_exact,
        "selected_tokens_exact": calculated_tokens_exact,
        "instruction_coverage_count": len(records) if isinstance(records, list) else 0,
    }


def recompute_ir_execution_certificate(
    program: dict[str, Any], certificate: dict[str, Any], model: Any
) -> dict[str, Any]:
    _require_torch()
    integrity = verify_ir_execution_certificate(program, certificate)
    if not integrity["valid"]:
        return {
            "valid": False,
            "reexecution_performed": False,
            "reason": "Certificate integrity verification failed",
            "integrity": integrity,
        }
    from .reference_gemma import model_state_sha256

    current_model_state = model_state_sha256(model)
    if current_model_state != certificate.get("model_state_sha256"):
        return {
            "valid": False,
            "reexecution_performed": False,
            "reason": "Model-state SHA-256 mismatch",
            "integrity": integrity,
        }
    device = next(model.parameters()).device
    input_ids = torch.tensor(
        certificate["input_token_ids"], dtype=torch.int64, device=device
    )
    parameters = bind_model_tensors(program, model)
    original_attention = model.config._attn_implementation
    model.config._attn_implementation = "eager"
    try:
        with torch.no_grad():
            execution = execute_gemma_ir(program, parameters, input_ids)
            observed = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                logits_to_keep=1,
            )
    finally:
        model.config._attn_implementation = original_attention
    recomputed = build_ir_execution_certificate(
        program,
        input_ids,
        execution,
        observed.logits,
        "huggingface_eager",
        current_model_state,
    )
    exact_match = recomputed == certificate
    return {
        "valid": exact_match,
        "reexecution_performed": True,
        "certificate_exact_match": exact_match,
        "recomputed_certificate_sha256": recomputed["certificate_sha256"],
        "record_count": len(recomputed["execution_records"]),
        "logits_bit_exact": recomputed["logits_bit_exact"],
        "selected_tokens_exact": recomputed["selected_tokens_exact"],
        "integrity": integrity,
    }


def summarize_ir_execution(
    program: dict[str, Any], certificate: dict[str, Any]
) -> dict[str, Any]:
    verification = verify_ir_execution_certificate(program, certificate)
    if not verification["valid"]:
        raise ValueError("Cannot summarize an invalid Gemma IR execution certificate")
    body = {
        "schema_version": 1,
        "scope": "Compact fixed-input commitment to a bit-exact typed-IR prediction against pinned Hugging Face eager execution; full instruction records are omitted and no SDPA, fused-kernel, hardware-semantic, scientific, or regulatory equivalence is established.",
        "program_sha256": program["program_sha256"],
        "model_state_sha256": certificate["model_state_sha256"],
        "source_certificate_sha256": certificate["certificate_sha256"],
        "execution_profile": certificate["execution_profile"],
        "input_ids": certificate["input_ids"],
        "input_token_ids": certificate["input_token_ids"],
        "predicted_logits": certificate["predicted_logits"],
        "observed_logits": certificate["observed_logits"],
        "predicted_token_ids": certificate["predicted_token_ids"],
        "predicted_token_ids_descriptor": certificate[
            "predicted_token_ids_descriptor"
        ],
        "observed_token_ids": certificate["observed_token_ids"],
        "logits_bit_exact": certificate["logits_bit_exact"],
        "selected_tokens_exact": certificate["selected_tokens_exact"],
        "selection_proofs": certificate["selection_proofs"],
        "first_divergence": None,
        "execution_root_sha256": certificate["execution_root_sha256"],
        "instruction_coverage_count": len(program["instructions"]),
        "instruction_coverage_complete": True,
        "full_execution_records_committed": False,
        "fixed_input_canonical_eager_prediction_established": True,
        "deployed_sdpa_correspondence_established": False,
        "hardware_instruction_semantics_established": False,
    }
    return {**body, "summary_sha256": _sha256(body)}


def verify_ir_execution_summary(
    program: dict[str, Any],
    summary: dict[str, Any],
    source_certificate: dict[str, Any],
) -> dict[str, Any]:
    _require_torch()
    if not verify_gemma_ir(program)["valid"] or not isinstance(summary, dict):
        return {"valid": False}
    source_verification = verify_ir_execution_certificate(
        program, source_certificate
    )
    body = {key: value for key, value in summary.items() if key != "summary_sha256"}
    try:
        summary_hash_valid = _sha256(body) == summary.get("summary_sha256")
    except (TypeError, ValueError):
        return {"valid": False}
    predicted = summary.get("predicted_logits")
    observed = summary.get("observed_logits")
    logits_exact = bool(
        isinstance(predicted, dict)
        and isinstance(observed, dict)
        and isinstance(predicted.get("shape"), list)
        and isinstance(predicted.get("dtype"), str)
        and HASH_PATTERN.fullmatch(str(predicted.get("sha256", ""))) is not None
        and predicted.get("sha256") == observed.get("sha256")
        and predicted.get("shape") == observed.get("shape")
        and predicted.get("dtype") == observed.get("dtype")
    )
    predicted_tokens = summary.get("predicted_token_ids")
    observed_tokens = summary.get("observed_token_ids")
    tokens_exact = bool(
        isinstance(predicted_tokens, list)
        and predicted_tokens
        and predicted_tokens == observed_tokens
        and all(isinstance(token, int) and not isinstance(token, bool) for token in predicted_tokens)
    )
    selection_proofs_valid = _selection_proofs_valid(
        summary.get("selection_proofs"),
        predicted_tokens,
        program["configuration"]["vocabulary_size"],
    )
    token_descriptor = summary.get("predicted_token_ids_descriptor")
    reconstructed_token_descriptor = (
        tensor_descriptor(torch.tensor(predicted_tokens, dtype=torch.int64))
        if tokens_exact
        else {}
    )
    token_descriptor_valid = bool(
        isinstance(token_descriptor, dict)
        and token_descriptor.get("sha256")
        == reconstructed_token_descriptor.get("sha256")
        and token_descriptor.get("shape")
        == reconstructed_token_descriptor.get("shape")
        and token_descriptor.get("dtype")
        == reconstructed_token_descriptor.get("dtype")
    )
    input_values = summary.get("input_token_ids")
    input_descriptor = summary.get("input_ids")
    input_descriptor_valid = False
    if (
        _rectangular_integer_rows(input_values)
        and isinstance(input_descriptor, dict)
    ):
        reconstructed_input = tensor_descriptor(
            torch.tensor(input_values, dtype=torch.int64)
        )
        input_descriptor_valid = bool(
            input_descriptor.get("sha256") == reconstructed_input.get("sha256")
            and input_descriptor.get("shape") == reconstructed_input.get("shape")
            and input_descriptor.get("dtype") == reconstructed_input.get("dtype")
        )
    source_fields = (
        "program_sha256",
        "model_state_sha256",
        "execution_profile",
        "input_ids",
        "input_token_ids",
        "predicted_logits",
        "observed_logits",
        "predicted_token_ids",
        "predicted_token_ids_descriptor",
        "observed_token_ids",
        "logits_bit_exact",
        "selected_tokens_exact",
        "selection_proofs",
        "first_divergence",
        "execution_root_sha256",
        "instruction_coverage_count",
        "instruction_coverage_complete",
        "fixed_input_canonical_eager_prediction_established",
        "deployed_sdpa_correspondence_established",
        "hardware_instruction_semantics_established",
    )
    source_certificate_binding_valid = bool(
        source_verification.get("valid")
        and summary.get("source_certificate_sha256")
        == source_certificate.get("certificate_sha256")
        and all(summary.get(field) == source_certificate.get(field) for field in source_fields)
    )
    claims_consistent = bool(
        summary.get("schema_version") == 1
        and summary.get("program_sha256") == program.get("program_sha256")
        and HASH_PATTERN.fullmatch(str(summary.get("model_state_sha256", "")))
        is not None
        and summary.get("execution_profile") == "huggingface_eager"
        and summary.get("logits_bit_exact") is logits_exact
        and summary.get("selected_tokens_exact") is tokens_exact
        and summary.get("first_divergence") is None
        and summary.get("instruction_coverage_count")
        == len(program.get("instructions", []))
        and summary.get("instruction_coverage_complete") is True
        and summary.get("full_execution_records_committed") is False
        and summary.get("fixed_input_canonical_eager_prediction_established") is True
        and summary.get("deployed_sdpa_correspondence_established") is False
        and summary.get("hardware_instruction_semantics_established") is False
    )
    valid = all(
        (
            summary_hash_valid,
            source_certificate_binding_valid,
            logits_exact,
            tokens_exact,
            token_descriptor_valid,
            input_descriptor_valid,
            selection_proofs_valid,
            claims_consistent,
        )
    )
    return {
        "valid": valid,
        "summary_hash_valid": summary_hash_valid,
        "source_certificate_binding_valid": source_certificate_binding_valid,
        "logits_bit_exact": logits_exact,
        "selected_tokens_exact": tokens_exact,
        "token_descriptor_valid": token_descriptor_valid,
        "input_descriptor_valid": input_descriptor_valid,
        "selection_proofs_valid": selection_proofs_valid,
        "claims_consistent": claims_consistent,
    }
