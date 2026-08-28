from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

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


def _require_torch() -> None:
    if torch is None or functional is None:
        raise RuntimeError("PyTorch is required; install the gemma extra")


def reference_rms_norm(value: Any, weight: Any, epsilon: float) -> Any:
    normalized = value.float() * torch.rsqrt(value.float().pow(2).mean(-1, keepdim=True) + epsilon)
    return (normalized * (1.0 + weight.float())).type_as(value)


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


def _record(records: list[dict[str, Any]], name: str, tensor: Any, layer: int | None = None) -> None:
    append_chain_record(
        records,
        {
            "stage": name,
            "layer": layer,
            "tensor": tensor_descriptor(tensor),
        },
    )


def reference_gemma_forward(model: Any, input_ids: Any) -> ReferenceOutput:
    _require_torch()
    config = model.config
    text_model = model.model
    records: list[dict[str, Any]] = []
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
    boundaries = [hidden_states]
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
        attention_heads = torch.matmul(probability, repeated_value).transpose(1, 2).contiguous()
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
        if layer_index < len(text_model.layers) - 1:
            boundaries.append(hidden_states)

    hidden_states = reference_rms_norm(hidden_states, text_model.norm.weight, text_model.norm.eps)
    boundaries.append(hidden_states)
    _record(records, "final_norm", hidden_states)
    logits = functional.linear(hidden_states, model.lm_head.weight)
    if config.final_logit_softcapping is not None:
        cap = config.final_logit_softcapping
        logits = torch.tanh(logits / cap) * cap
    _record(records, "vocabulary_logits", logits)
    return ReferenceOutput(logits=logits, hidden_states=tuple(boundaries), records=tuple(records))


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
        "input_ids": input_ids[0].tolist(),
        "input_ids_sha256": hashlib.sha256(canonical_json(input_ids[0].tolist()).encode("utf-8")).hexdigest(),
        "reference_trace_root_sha256": reference.records[-1]["record_hash"] if reference.records else "GENESIS",
        "reference_trace_records": list(reference.records),
        "reference_trace_chain_verification": verify_trace_chain(
            list(reference.records), reference.records[-1]["record_hash"] if reference.records else "GENESIS"
        ),
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
        and calculated_token_match
        and calculated_boundaries_within
    )
    return {
        "valid": valid,
        "certificate_hash_match": original_hash == calculated_hash,
        "chain": chain,
        "reference_trace_chain": reference_chain,
        "chain_payloads_match": chain_payloads_match,
        "claims_consistent": claims_consistent,
        "selected_token_exact_match": calculated_token_match,
        "all_boundaries_within_tolerance": calculated_boundaries_within,
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

    maximum_error = max(case.get("max_absolute_error", case.get("absolute_error", 0.0)) for case in cases)
    body = {
        "scope": "Fixed-vector equation conformance checks against separate Python scalar equations; not a proof for all values or CUDA kernels.",
        "cases": cases,
        "maximum_absolute_error": maximum_error,
    }
    body["sha256"] = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    return body
