from __future__ import annotations

import copy
import hashlib
import re
from typing import Any

from .serialization import canonical_json


SCHEMA_VERSION = 1
SEMANTICS_VERSION = "gemma-ir-semantics-v1"
HASH_PATTERN = re.compile(r"[0-9a-f]{64}")


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


SEMANTICS: dict[str, dict[str, Any]] = {
    "ARANGE": {"equation": "output[0,s] = s for 0 <= s < sequence_length", "rounding": "none"},
    "EMBEDDING": {"equation": "output[b,s,h] = weight[input_ids[b,s],h]", "rounding": "exact indexed load"},
    "SCALE": {"equation": "output[i] = cast(input[i] * scalar, output_dtype)", "rounding": "execution-profile dtype conversion required"},
    "ROTARY_TABLE": {"equation": "angles[b,s,d] = float32(position_ids[b,s]) * inverse_frequency[d]; cosine and sine are duplicated across head_dimension then cast to output_dtype", "rounding": "execution-profile transcendental and cast semantics required"},
    "CAUSAL_MASK": {"equation": "output[b,1,q,k] = 0 when k <= q and, if sliding_window is set, k > q - sliding_window; otherwise output is the minimum finite output_dtype value", "rounding": "exact construction in output_dtype"},
    "RMS_NORM": {"equation": "output = cast(input * rsqrt(mean(float32(input)^2, last_axis) + epsilon) * (1 + float32(weight)), input_dtype)", "rounding": "execution-profile reduction, rsqrt, multiplication, and cast semantics required"},
    "LINEAR": {"equation": "output[...,o] = sum_i input[...,i] * weight[o,i]", "rounding": "execution-profile multiply-accumulate and reduction order required"},
    "RESHAPE_TRANSPOSE_HEADS": {"equation": "output[b,head,s,d] = input[b,s,head * head_dimension + d]", "rounding": "none"},
    "ROTARY_APPLY_PAIR": {"equation": "query_out = query * cosine + rotate_half(query) * sine; key_out = key * cosine + rotate_half(key) * sine", "rounding": "execution-profile elementwise operation order required"},
    "REPEAT_KV": {"equation": "output[b,h,s,d] = input[b,floor(h / repetitions),s,d]", "rounding": "none"},
    "MATMUL_QK": {"equation": "output[b,h,q,k] = sum_d query[b,h,q,d] * key[b,h,k,d]", "rounding": "execution-profile multiply-accumulate and reduction order required"},
    "SOFTCAP": {"equation": "output = tanh(input / cap) * cap", "rounding": "execution-profile transcendental and operation order required"},
    "ADD": {"equation": "output[i] = cast(left[i] + right[i], output_dtype)", "rounding": "execution-profile addition and cast semantics required"},
    "SOFTMAX": {"equation": "output[i] = cast(exp(float32(input[i]) - max(float32(input))) / sum(exp(float32(input) - max(float32(input)))), output_dtype)", "rounding": "execution-profile maximum, exponential, reduction order, division, and cast semantics required"},
    "MATMUL_AV": {"equation": "output[b,h,q,d] = sum_k probability[b,h,q,k] * value[b,h,k,d]", "rounding": "execution-profile multiply-accumulate and reduction order required"},
    "TRANSPOSE_RESHAPE_HEADS": {"equation": "output[b,s,h * head_dimension + d] = input[b,h,s,d]", "rounding": "none"},
    "GELU_TANH": {"equation": "output = 0.5 * input * (1 + tanh(sqrt(2/pi) * (input + 0.044715 * input^3)))", "rounding": "execution-profile constants, operation order, transcendental, and casts required"},
    "MUL": {"equation": "output[i] = cast(left[i] * right[i], output_dtype)", "rounding": "execution-profile multiplication and cast semantics required"},
    "SLICE_LAST_TOKEN": {"equation": "output[b,0,h] = input[b,sequence_length-1,h]", "rounding": "none"},
    "ARGMAX": {"equation": "output[b] is the lowest token index whose value equals max_token input[b,0,token]", "rounding": "none"},
}


ALLOWED_OPCODES = frozenset(SEMANTICS)


def _shape(*dimensions: int | str) -> list[int | str]:
    return list(dimensions)


def _parameter_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    model = manifest.get("model", {})
    tensors = model.get("parameter_tensors", []) + model.get("buffer_tensors", [])
    parameters: dict[str, dict[str, Any]] = {}
    for tensor in tensors:
        name = tensor.get("name")
        if not isinstance(name, str) or name in parameters:
            raise ValueError("Architecture manifest has invalid or duplicate tensor names")
        descriptor = {
            "shape": tensor.get("shape"),
            "dtype": tensor.get("dtype"),
            "numel": tensor.get("numel"),
            "sha256": tensor.get("sha256"),
        }
        if not isinstance(descriptor["shape"], list) or not HASH_PATTERN.fullmatch(str(descriptor["sha256"])):
            raise ValueError(f"Invalid tensor descriptor for {name}")
        parameters[name] = descriptor
    return parameters


def _require_config(config: dict[str, Any], field: str, expected_type: type) -> Any:
    value = config.get(field)
    if not isinstance(value, expected_type) or isinstance(value, bool) != (expected_type is bool):
        raise ValueError(f"Invalid or missing Gemma configuration field {field}")
    return value


def compile_gemma_ir(manifest: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(manifest, dict) or not isinstance(manifest.get("model"), dict):
        raise ValueError("Architecture manifest must contain a model mapping")
    model = manifest["model"]
    config = model.get("config")
    if not isinstance(config, dict):
        raise ValueError("Architecture manifest must contain model configuration")
    layers = _require_config(config, "num_hidden_layers", int)
    hidden = _require_config(config, "hidden_size", int)
    intermediate = _require_config(config, "intermediate_size", int)
    heads = _require_config(config, "num_attention_heads", int)
    kv_heads = _require_config(config, "num_key_value_heads", int)
    head_dimension = _require_config(config, "head_dim", int)
    vocabulary = _require_config(config, "vocab_size", int)
    sliding_window = _require_config(config, "sliding_window", int)
    epsilon = config.get("rms_norm_eps")
    layer_types = config.get("layer_types")
    if not isinstance(epsilon, (int, float)) or epsilon <= 0:
        raise ValueError("Invalid RMS normalization epsilon")
    if config.get("rope_scaling") is not None:
        raise ValueError(
            "Scaled RoPE requires manifest-bound global and local attention-scaling constants"
        )
    if not isinstance(layer_types, list) or len(layer_types) != layers or any(
        layer_type not in {"sliding_attention", "full_attention"} for layer_type in layer_types
    ):
        raise ValueError("Invalid Gemma layer types")
    if heads <= 0 or kv_heads <= 0 or heads % kv_heads != 0 or hidden <= 0:
        raise ValueError("Invalid Gemma attention dimensions")
    parameters = _parameter_map(manifest)
    required_parameters = {
        "model.embed_tokens.weight",
        "model.embed_tokens.embed_scale",
        "model.rotary_emb.inv_freq",
        "model.rotary_emb_local.inv_freq",
        "model.norm.weight",
        "lm_head.weight",
    }
    for layer in range(layers):
        prefix = f"model.layers.{layer}"
        required_parameters.update(
            {
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.self_attn.q_norm.weight",
                f"{prefix}.self_attn.k_norm.weight",
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.post_attention_layernorm.weight",
                f"{prefix}.pre_feedforward_layernorm.weight",
                f"{prefix}.post_feedforward_layernorm.weight",
            }
        )
    missing = sorted(required_parameters - parameters.keys())
    if missing:
        raise ValueError(f"Architecture manifest is missing required tensors: {missing}")

    tensors: dict[str, dict[str, Any]] = {
        "input_ids": {"shape": _shape("B", "S"), "dtype": "torch.int64", "producer": "EXTERNAL"}
    }
    instructions: list[dict[str, Any]] = []

    def emit(
        opcode: str,
        inputs: list[str],
        outputs: dict[str, tuple[list[int | str], str]],
        *,
        layer: int | None = None,
        parameters_used: list[str] | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        if opcode not in ALLOWED_OPCODES:
            raise ValueError(f"Opaque or unsupported opcode {opcode}")
        identifier = f"i{len(instructions):04d}"
        output_names = list(outputs)
        body = {
            "id": identifier,
            "opcode": opcode,
            "semantics": f"{SEMANTICS_VERSION}:{opcode}",
            "layer": layer,
            "inputs": inputs,
            "parameter_refs": parameters_used or [],
            "outputs": output_names,
            "attributes": attributes or {},
        }
        instruction = {**body, "instruction_sha256": _sha256(body)}
        instructions.append(instruction)
        for output_name, (output_shape, output_dtype) in outputs.items():
            if output_name in tensors:
                raise ValueError(f"Duplicate IR tensor {output_name}")
            tensors[output_name] = {
                "shape": output_shape,
                "dtype": output_dtype,
                "producer": identifier,
            }

    configured_dtype = config.get("torch_dtype")
    dtype = (
        parameters["model.embed_tokens.weight"]["dtype"]
        if configured_dtype is None
        else str(configured_dtype)
    )
    if not dtype.startswith("torch."):
        dtype = f"torch.{dtype}"
    emit("ARANGE", ["input_ids"], {"position_ids": (_shape(1, "S"), "torch.int64")}, attributes={"sequence_length": "S"})
    emit(
        "EMBEDDING",
        ["input_ids"],
        {"embedding_unscaled": (_shape("B", "S", hidden), dtype)},
        parameters_used=["model.embed_tokens.weight"],
    )
    emit(
        "SCALE",
        ["embedding_unscaled"],
        {"hidden.0": (_shape("B", "S", hidden), dtype)},
        parameters_used=["model.embed_tokens.embed_scale"],
    )
    emit(
        "ROTARY_TABLE",
        ["hidden.0", "position_ids"],
        {
            "rotary.global.cosine": (_shape("B", "S", head_dimension), dtype),
            "rotary.global.sine": (_shape("B", "S", head_dimension), dtype),
        },
        parameters_used=["model.rotary_emb.inv_freq"],
        attributes={"attention_scaling": 1.0},
    )
    emit(
        "ROTARY_TABLE",
        ["hidden.0", "position_ids"],
        {
            "rotary.local.cosine": (_shape("B", "S", head_dimension), dtype),
            "rotary.local.sine": (_shape("B", "S", head_dimension), dtype),
        },
        parameters_used=["model.rotary_emb_local.inv_freq"],
        attributes={"attention_scaling": 1.0},
    )
    emit(
        "CAUSAL_MASK",
        ["input_ids", "hidden.0"],
        {"mask.full": (_shape(1, 1, "S", "S"), dtype)},
        attributes={"sequence_length": "S", "sliding_window": None},
    )
    emit(
        "CAUSAL_MASK",
        ["input_ids", "hidden.0"],
        {"mask.sliding": (_shape(1, 1, "S", "S"), dtype)},
        attributes={"sequence_length": "S", "sliding_window": sliding_window},
    )

    kv_repetitions = heads // kv_heads
    attention_scale = float(config.get("query_pre_attn_scalar", head_dimension)) ** -0.5
    for layer in range(layers):
        prefix = f"model.layers.{layer}"
        input_hidden = f"hidden.{layer}"
        emit(
            "RMS_NORM",
            [input_hidden],
            {f"layer.{layer}.attention.normalized": (_shape("B", "S", hidden), dtype)},
            layer=layer,
            parameters_used=[f"{prefix}.input_layernorm.weight"],
            attributes={"epsilon": epsilon},
        )
        normalized = f"layer.{layer}.attention.normalized"
        for projection, count in (("query", heads), ("key", kv_heads), ("value", kv_heads)):
            parameter = projection[0] if projection != "value" else "v"
            emit(
                "LINEAR",
                [normalized],
                {f"layer.{layer}.{projection}.flat": (_shape("B", "S", count * head_dimension), dtype)},
                layer=layer,
                parameters_used=[f"{prefix}.self_attn.{parameter}_proj.weight"],
            )
            emit(
                "RESHAPE_TRANSPOSE_HEADS",
                [f"layer.{layer}.{projection}.flat"],
                {f"layer.{layer}.{projection}.heads": (_shape("B", count, "S", head_dimension), dtype)},
                layer=layer,
                attributes={"heads": count, "head_dimension": head_dimension},
            )
        emit(
            "RMS_NORM",
            [f"layer.{layer}.query.heads"],
            {f"layer.{layer}.query.normalized": (_shape("B", heads, "S", head_dimension), dtype)},
            layer=layer,
            parameters_used=[f"{prefix}.self_attn.q_norm.weight"],
            attributes={"epsilon": epsilon},
        )
        emit(
            "RMS_NORM",
            [f"layer.{layer}.key.heads"],
            {f"layer.{layer}.key.normalized": (_shape("B", kv_heads, "S", head_dimension), dtype)},
            layer=layer,
            parameters_used=[f"{prefix}.self_attn.k_norm.weight"],
            attributes={"epsilon": epsilon},
        )
        rotary = "local" if layer_types[layer] == "sliding_attention" else "global"
        emit(
            "ROTARY_APPLY_PAIR",
            [
                f"layer.{layer}.query.normalized",
                f"layer.{layer}.key.normalized",
                f"rotary.{rotary}.cosine",
                f"rotary.{rotary}.sine",
            ],
            {
                f"layer.{layer}.query.rotary": (_shape("B", heads, "S", head_dimension), dtype),
                f"layer.{layer}.key.rotary": (_shape("B", kv_heads, "S", head_dimension), dtype),
            },
            layer=layer,
            attributes={"rotary_profile": rotary},
        )
        emit(
            "REPEAT_KV",
            [f"layer.{layer}.key.rotary"],
            {f"layer.{layer}.key.repeated": (_shape("B", heads, "S", head_dimension), dtype)},
            layer=layer,
            attributes={"repetitions": kv_repetitions},
        )
        emit(
            "REPEAT_KV",
            [f"layer.{layer}.value.heads"],
            {f"layer.{layer}.value.repeated": (_shape("B", heads, "S", head_dimension), dtype)},
            layer=layer,
            attributes={"repetitions": kv_repetitions},
        )
        emit(
            "MATMUL_QK",
            [f"layer.{layer}.query.rotary", f"layer.{layer}.key.repeated"],
            {f"layer.{layer}.attention.unscaled_scores": (_shape("B", heads, "S", "S"), dtype)},
            layer=layer,
        )
        emit(
            "SCALE",
            [f"layer.{layer}.attention.unscaled_scores"],
            {f"layer.{layer}.attention.scaled_scores": (_shape("B", heads, "S", "S"), dtype)},
            layer=layer,
            attributes={"scalar": attention_scale},
        )
        score_name = f"layer.{layer}.attention.scaled_scores"
        cap = config.get("attn_logit_softcapping")
        if cap is not None:
            emit(
                "SOFTCAP",
                [score_name],
                {f"layer.{layer}.attention.softcapped_scores": (_shape("B", heads, "S", "S"), dtype)},
                layer=layer,
                attributes={"cap": cap},
            )
            score_name = f"layer.{layer}.attention.softcapped_scores"
        mask = "mask.sliding" if layer_types[layer] == "sliding_attention" else "mask.full"
        emit(
            "ADD",
            [score_name, mask],
            {f"layer.{layer}.attention.masked_scores": (_shape("B", heads, "S", "S"), dtype)},
            layer=layer,
            attributes={"output_dtype": dtype},
        )
        emit(
            "SOFTMAX",
            [f"layer.{layer}.attention.masked_scores"],
            {f"layer.{layer}.attention.probability": (_shape("B", heads, "S", "S"), dtype)},
            layer=layer,
            attributes={"axis": -1, "accumulation_dtype": "torch.float32", "output_dtype": dtype},
        )
        emit(
            "MATMUL_AV",
            [f"layer.{layer}.attention.probability", f"layer.{layer}.value.repeated"],
            {f"layer.{layer}.attention.head_output": (_shape("B", heads, "S", head_dimension), dtype)},
            layer=layer,
        )
        emit(
            "TRANSPOSE_RESHAPE_HEADS",
            [f"layer.{layer}.attention.head_output"],
            {f"layer.{layer}.attention.concatenated": (_shape("B", "S", heads * head_dimension), dtype)},
            layer=layer,
            attributes={"heads": heads, "head_dimension": head_dimension},
        )
        emit(
            "LINEAR",
            [f"layer.{layer}.attention.concatenated"],
            {f"layer.{layer}.attention.projected": (_shape("B", "S", hidden), dtype)},
            layer=layer,
            parameters_used=[f"{prefix}.self_attn.o_proj.weight"],
        )
        emit(
            "RMS_NORM",
            [f"layer.{layer}.attention.projected"],
            {f"layer.{layer}.attention.post_normalized": (_shape("B", "S", hidden), dtype)},
            layer=layer,
            parameters_used=[f"{prefix}.post_attention_layernorm.weight"],
            attributes={"epsilon": epsilon},
        )
        emit(
            "ADD",
            [input_hidden, f"layer.{layer}.attention.post_normalized"],
            {f"layer.{layer}.post_attention_residual": (_shape("B", "S", hidden), dtype)},
            layer=layer,
            attributes={"output_dtype": dtype},
        )
        emit(
            "RMS_NORM",
            [f"layer.{layer}.post_attention_residual"],
            {f"layer.{layer}.mlp.normalized": (_shape("B", "S", hidden), dtype)},
            layer=layer,
            parameters_used=[f"{prefix}.pre_feedforward_layernorm.weight"],
            attributes={"epsilon": epsilon},
        )
        for projection in ("gate", "up"):
            emit(
                "LINEAR",
                [f"layer.{layer}.mlp.normalized"],
                {f"layer.{layer}.mlp.{projection}": (_shape("B", "S", intermediate), dtype)},
                layer=layer,
                parameters_used=[f"{prefix}.mlp.{projection}_proj.weight"],
            )
        emit(
            "GELU_TANH",
            [f"layer.{layer}.mlp.gate"],
            {f"layer.{layer}.mlp.activated_gate": (_shape("B", "S", intermediate), dtype)},
            layer=layer,
        )
        emit(
            "MUL",
            [f"layer.{layer}.mlp.activated_gate", f"layer.{layer}.mlp.up"],
            {f"layer.{layer}.mlp.product": (_shape("B", "S", intermediate), dtype)},
            layer=layer,
            attributes={"output_dtype": dtype},
        )
        emit(
            "LINEAR",
            [f"layer.{layer}.mlp.product"],
            {f"layer.{layer}.mlp.down": (_shape("B", "S", hidden), dtype)},
            layer=layer,
            parameters_used=[f"{prefix}.mlp.down_proj.weight"],
        )
        emit(
            "RMS_NORM",
            [f"layer.{layer}.mlp.down"],
            {f"layer.{layer}.mlp.post_normalized": (_shape("B", "S", hidden), dtype)},
            layer=layer,
            parameters_used=[f"{prefix}.post_feedforward_layernorm.weight"],
            attributes={"epsilon": epsilon},
        )
        emit(
            "ADD",
            [f"layer.{layer}.post_attention_residual", f"layer.{layer}.mlp.post_normalized"],
            {f"hidden.{layer + 1}": (_shape("B", "S", hidden), dtype)},
            layer=layer,
            attributes={"output_dtype": dtype},
        )

    emit(
        "RMS_NORM",
        [f"hidden.{layers}"],
        {"hidden.final": (_shape("B", "S", hidden), dtype)},
        parameters_used=["model.norm.weight"],
        attributes={"epsilon": epsilon},
    )
    emit(
        "SLICE_LAST_TOKEN",
        ["hidden.final"],
        {"hidden.last": (_shape("B", 1, hidden), dtype)},
    )
    emit(
        "LINEAR",
        ["hidden.last"],
        {"logits.last": (_shape("B", 1, vocabulary), dtype)},
        parameters_used=["lm_head.weight"],
    )
    final_cap = config.get("final_logit_softcapping")
    logits_output = "logits.last"
    if final_cap is not None:
        emit(
            "SOFTCAP",
            [logits_output],
            {"logits.final": (_shape("B", 1, vocabulary), dtype)},
            attributes={"cap": final_cap},
        )
        logits_output = "logits.final"
    emit("ARGMAX", [logits_output], {"selected_token_id": (_shape("B"), "torch.int64")}, attributes={"axis": -1})

    body = {
        "schema_version": SCHEMA_VERSION,
        "scope": "Canonical structural Gemma execution IR compiled from a frozen architecture manifest; primitive equations are declared, but bit-exact numerical qualification requires a separate execution-profile certificate.",
        "semantics_version": SEMANTICS_VERSION,
        "source_manifest_sha256": manifest.get("manifest_sha256"),
        "source_manifest_commitment_sha256": _sha256(manifest),
        "model_class": model.get("class"),
        "configuration": {
            "layers": layers,
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "attention_heads": heads,
            "key_value_heads": kv_heads,
            "head_dimension": head_dimension,
            "vocabulary_size": vocabulary,
            "layer_types": layer_types,
            "dtype": dtype,
            "rms_norm_epsilon": epsilon,
            "sliding_window": sliding_window,
            "attention_scale": attention_scale,
            "rotary_attention_scaling": 1.0,
        },
        "external_inputs": ["input_ids"],
        "declared_outputs": [logits_output, "selected_token_id"],
        "parameter_commitments": parameters,
        "semantics": copy.deepcopy(SEMANTICS),
        "instructions": instructions,
        "tensors": tensors,
        "opaque_composite_operations": [],
        "bit_exact_numerical_execution_qualified": False,
        "deployed_backend_correspondence_established": False,
    }
    return {**body, "program_sha256": _sha256(body)}


def verify_gemma_ir(program: dict[str, Any], manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    if not isinstance(program, dict):
        return {"valid": False}
    body = {key: value for key, value in program.items() if key != "program_sha256"}
    try:
        program_hash_valid = _sha256(body) == program.get("program_sha256")
    except (TypeError, ValueError):
        return {"valid": False}
    instructions = program.get("instructions")
    tensors = program.get("tensors")
    parameters = program.get("parameter_commitments")
    semantics = program.get("semantics")
    configuration = program.get("configuration")
    layer_count = (
        configuration.get("layers")
        if isinstance(configuration, dict)
        and isinstance(configuration.get("layers"), int)
        and not isinstance(configuration.get("layers"), bool)
        and configuration.get("layers") > 0
        else 0
    )
    structure_valid = bool(
        program.get("schema_version") == SCHEMA_VERSION
        and program.get("semantics_version") == SEMANTICS_VERSION
        and isinstance(instructions, list)
        and instructions
        and isinstance(tensors, dict)
        and isinstance(parameters, dict)
        and isinstance(semantics, dict)
        and layer_count > 0
        and program.get("external_inputs") == ["input_ids"]
        and isinstance(program.get("declared_outputs"), list)
        and program.get("declared_outputs")
        and all(isinstance(output, str) for output in program.get("declared_outputs"))
        and program.get("opaque_composite_operations") == []
        and program.get("bit_exact_numerical_execution_qualified") is False
        and program.get("deployed_backend_correspondence_established") is False
    )
    dependency_valid = structure_valid
    instruction_hashes_valid = structure_valid
    semantics_coverage_valid = structure_valid
    parameter_references_valid = structure_valid
    layer_coverage_valid = structure_valid
    produced = set(program.get("external_inputs", []))
    output_producers: dict[str, str] = {}
    layer_counts = {layer: 0 for layer in range(layer_count)}
    if structure_valid:
        for index, instruction in enumerate(instructions):
            if not isinstance(instruction, dict):
                dependency_valid = False
                instruction_hashes_valid = False
                semantics_coverage_valid = False
                parameter_references_valid = False
                layer_coverage_valid = False
                continue
            instruction_body = {
                key: value for key, value in instruction.items() if key != "instruction_sha256"
            }
            instruction_hashes_valid &= bool(
                instruction.get("id") == f"i{index:04d}"
                and _sha256(instruction_body) == instruction.get("instruction_sha256")
            )
            opcode = instruction.get("opcode")
            semantics_coverage_valid &= bool(
                opcode in ALLOWED_OPCODES
                and instruction.get("semantics") == f"{SEMANTICS_VERSION}:{opcode}"
                and semantics.get(opcode) == SEMANTICS.get(opcode)
            )
            inputs = instruction.get("inputs")
            outputs = instruction.get("outputs")
            refs = instruction.get("parameter_refs")
            inputs_valid = isinstance(inputs, list) and all(
                isinstance(name, str) for name in inputs
            )
            outputs_valid = bool(
                isinstance(outputs, list)
                and outputs
                and all(isinstance(name, str) for name in outputs)
            )
            refs_valid = isinstance(refs, list) and all(
                isinstance(reference, str) for reference in refs
            )
            dependency_valid &= bool(
                inputs_valid
                and outputs_valid
                and all(name in produced for name in inputs)
                and all(name not in produced for name in outputs)
            )
            parameter_references_valid &= bool(
                refs_valid and all(reference in parameters for reference in refs)
            )
            if outputs_valid:
                for output in outputs:
                    produced.add(output)
                    output_producers[output] = instruction.get("id")
            layer = instruction.get("layer")
            if isinstance(layer, int) and layer in layer_counts:
                layer_counts[layer] += 1
            elif layer is not None:
                layer_coverage_valid = False
        dependency_valid &= all(output in produced for output in program["declared_outputs"])
        dependency_valid &= set(tensors) == produced
        dependency_valid &= all(
            isinstance(name, str)
            and isinstance(descriptor, dict)
            and isinstance(descriptor.get("shape"), list)
            and all(
                (isinstance(dimension, int) and not isinstance(dimension, bool) and dimension >= 0)
                or (isinstance(dimension, str) and dimension in {"B", "S"})
                for dimension in descriptor["shape"]
            )
            and isinstance(descriptor.get("dtype"), str)
            and descriptor.get("producer")
            == ("EXTERNAL" if name in program["external_inputs"] else output_producers.get(name))
            for name, descriptor in tensors.items()
        )
        layer_coverage_valid &= bool(
            layer_counts
            and len(set(layer_counts.values())) == 1
            and min(layer_counts.values()) > 0
        )
        semantics_coverage_valid &= set(semantics) == set(ALLOWED_OPCODES)
        parameter_references_valid &= all(
            isinstance(descriptor, dict)
            and isinstance(descriptor.get("shape"), list)
            and HASH_PATTERN.fullmatch(str(descriptor.get("sha256", ""))) is not None
            for descriptor in parameters.values()
        )
    manifest_binding_valid = manifest is None
    if manifest is not None:
        try:
            manifest_binding_valid = bool(
                program.get("source_manifest_sha256") == manifest.get("manifest_sha256")
                and program.get("source_manifest_commitment_sha256") == _sha256(manifest)
                and program.get("parameter_commitments") == _parameter_map(manifest)
            )
        except (TypeError, ValueError):
            manifest_binding_valid = False
    valid = all(
        (
            program_hash_valid,
            structure_valid,
            dependency_valid,
            instruction_hashes_valid,
            semantics_coverage_valid,
            parameter_references_valid,
            layer_coverage_valid,
            manifest_binding_valid,
        )
    )
    return {
        "valid": valid,
        "program_hash_valid": program_hash_valid,
        "structure_valid": structure_valid,
        "dependency_graph_complete": dependency_valid,
        "instruction_hashes_valid": instruction_hashes_valid,
        "semantics_coverage_valid": semantics_coverage_valid,
        "parameter_references_valid": parameter_references_valid,
        "layer_coverage_valid": layer_coverage_valid,
        "manifest_binding_valid": manifest_binding_valid,
        "instruction_count": len(instructions) if isinstance(instructions, list) else 0,
        "tensor_count": len(tensors) if isinstance(tensors, dict) else 0,
        "unqualified_numerical_semantics": sorted(
            opcode for opcode, declaration in SEMANTICS.items() if "required" in declaration["rounding"]
        ),
    }


def build_rationale_slice(program: dict[str, Any], output: str = "selected_token_id") -> dict[str, Any]:
    verification = verify_gemma_ir(program)
    if not verification["valid"]:
        raise ValueError("Cannot derive a rationale slice from an invalid Gemma IR")
    if output not in program["declared_outputs"]:
        raise ValueError(f"Output {output!r} is not a declared program output")
    by_output = {
        tensor: instruction
        for instruction in program["instructions"]
        for tensor in instruction["outputs"]
    }
    required_tensors: set[str] = set()
    required_instructions: set[str] = set()
    required_parameters: set[str] = set()
    external_inputs: set[str] = set()

    def visit(tensor: str) -> None:
        if tensor in required_tensors:
            return
        required_tensors.add(tensor)
        if tensor in program["external_inputs"]:
            external_inputs.add(tensor)
            return
        instruction = by_output.get(tensor)
        if instruction is None:
            raise ValueError(f"No producer for rationale tensor {tensor}")
        required_instructions.add(instruction["id"])
        required_parameters.update(instruction["parameter_refs"])
        for input_tensor in instruction["inputs"]:
            visit(input_tensor)

    visit(output)
    ordered_instructions = [
        instruction["id"]
        for instruction in program["instructions"]
        if instruction["id"] in required_instructions
    ]
    body = {
        "schema_version": 1,
        "scope": "Complete structural backward slice from a declared Gemma IR output to external inputs and frozen parameter commitments; numerical causal sufficiency requires an exact execution witness.",
        "program_sha256": program["program_sha256"],
        "output": output,
        "instruction_ids": ordered_instructions,
        "tensor_names": sorted(required_tensors),
        "external_inputs": sorted(external_inputs),
        "parameter_refs": sorted(required_parameters),
        "dependency_slice_complete": True,
        "numerical_execution_witness_bound": False,
        "causal_semantic_interpretation_established": False,
    }
    return {**body, "rationale_sha256": _sha256(body)}


def verify_rationale_slice(program: dict[str, Any], rationale: dict[str, Any]) -> dict[str, Any]:
    try:
        expected = build_rationale_slice(program, rationale.get("output", ""))
    except (TypeError, ValueError):
        return {"valid": False}
    rationale_hash_valid = _sha256(
        {key: value for key, value in rationale.items() if key != "rationale_sha256"}
    ) == rationale.get("rationale_sha256")
    exact_match = rationale == expected
    return {
        "valid": rationale_hash_valid and exact_match,
        "rationale_hash_valid": rationale_hash_valid,
        "derived_slice_exact_match": exact_match,
        "instruction_count": len(expected["instruction_ids"]),
        "parameter_count": len(expected["parameter_refs"]),
        "tensor_count": len(expected["tensor_names"]),
    }
