from __future__ import annotations

import hashlib
import math
import platform
import sys
from collections import Counter
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from .interpretability import _model_device, _tokenize, hash_model_snapshot
from .serialization import canonical_json

try:
    import torch
    import transformers
    from torch.utils._python_dispatch import TorchDispatchMode
except ModuleNotFoundError:
    torch = None
    transformers = None

    class TorchDispatchMode:
        pass


GENESIS = "GENESIS"


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required; install the gemma extra")


def _require_gemma_dependencies() -> None:
    _require_torch()
    if transformers is None:
        raise RuntimeError("Transformers is required; install the gemma extra")


def _model_dtype(model: Any) -> str:
    return str(next(model.parameters()).dtype)


def tensor_sha256(tensor: Any) -> str:
    _require_torch()
    with torch._C._DisableTorchDispatch():
        detached = tensor.detach().contiguous()
        raw = detached.reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
        metadata = f"{tuple(detached.shape)}|{detached.dtype}|{detached.layout}".encode("utf-8")
    digest = hashlib.sha256()
    digest.update(metadata)
    digest.update(raw)
    return digest.hexdigest()


def tensor_descriptor(tensor: Any, cache: dict[tuple[int, int], dict[str, Any]] | None = None) -> dict[str, Any]:
    _require_torch()
    version = int(getattr(tensor, "_version", 0))
    key = (id(tensor), version)
    cacheable = bool(getattr(tensor, "requires_grad", False) and getattr(tensor, "is_leaf", False))
    if cacheable and cache is not None and key in cache:
        return cache[key]
    descriptor = {
        "kind": "tensor",
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "layout": str(tensor.layout),
        "numel": int(tensor.numel()),
        "sha256": tensor_sha256(tensor),
    }
    if cacheable and cache is not None:
        cache[key] = descriptor
    return descriptor


def _value_descriptor(value: Any, cache: dict[tuple[int, int], dict[str, Any]]) -> Any:
    if torch is not None and isinstance(value, torch.Tensor):
        return tensor_descriptor(value, cache)
    if isinstance(value, dict):
        return {str(key): _value_descriptor(item, cache) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list)):
        return [_value_descriptor(item, cache) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        label = "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
        return {"kind": "non_finite_float", "value": label}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"kind": "python", "type": type(value).__name__, "repr": repr(value)[:500]}


def append_chain_record(records: list[dict[str, Any]], payload: dict[str, Any]) -> dict[str, Any]:
    previous_hash = records[-1]["record_hash"] if records else GENESIS
    material = f"{previous_hash}\n{canonical_json(payload)}".encode("utf-8")
    record = {
        "index": len(records),
        "previous_hash": previous_hash,
        "payload": payload,
        "record_hash": hashlib.sha256(material).hexdigest(),
    }
    records.append(record)
    return record


def verify_trace_chain(records: list[dict[str, Any]], expected_root: str | None = None) -> dict[str, Any]:
    previous_hash = GENESIS
    for index, record in enumerate(records):
        if record.get("index") != index:
            return {"valid": False, "records": index, "reason": f"Unexpected index at record {index}"}
        if record.get("previous_hash") != previous_hash:
            return {"valid": False, "records": index, "reason": f"Broken previous hash at record {index}"}
        payload = record.get("payload")
        if not isinstance(payload, dict):
            return {"valid": False, "records": index, "reason": f"Missing payload at record {index}"}
        calculated = hashlib.sha256(f"{previous_hash}\n{canonical_json(payload)}".encode("utf-8")).hexdigest()
        if calculated != record.get("record_hash"):
            return {"valid": False, "records": index, "reason": f"Hash mismatch at record {index}"}
        previous_hash = calculated
    if expected_root is not None and previous_hash != expected_root:
        return {"valid": False, "records": len(records), "reason": "Trace root mismatch"}
    return {"valid": True, "records": len(records), "root": previous_hash}


def operator_semantics_catalog() -> dict[str, Any]:
    return {
        "scope": "Mathematical operator descriptions; deployed behavior also depends on dtype, rounding, kernel, and reduction order.",
        "embedding": "E[token_id] * embedding_scale",
        "linear": "y = x @ transpose(weight)",
        "rms_norm": "y = x / sqrt(mean(x^2) + epsilon) * (1 + weight)",
        "rotary_position": "q_p = R_p q; k_p = R_p k",
        "attention_scores": "scores = q @ transpose(k) / sqrt(head_dimension)",
        "softmax": "p_i = exp(z_i - max(z)) / sum_j exp(z_j - max(z))",
        "attention_value": "head = softmax(scores) @ value",
        "gated_mlp": "down_proj(gelu_tanh(gate_proj(x)) * up_proj(x))",
        "residual_addition": "residual_next = residual + component_output",
        "vocabulary_projection": "logits = final_hidden @ transpose(token_embedding_weight)",
        "greedy_decode": "next_token = first index attaining max(logits)",
    }


def build_architecture_manifest(model: Any, tokenizer: Any, model_path: str | Path) -> dict[str, Any]:
    _require_gemma_dependencies()
    parameter_entries = []
    unique_hashes: dict[int, str] = {}
    unique_numels: dict[int, int] = {}
    aliases: dict[int, list[str]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
        if id(parameter) not in unique_hashes:
            unique_hashes[id(parameter)] = tensor_sha256(parameter)
            unique_numels[id(parameter)] = int(parameter.numel())
        parameter_entries.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "numel": int(parameter.numel()),
                "requires_grad": bool(parameter.requires_grad),
                "sha256": unique_hashes[id(parameter)],
            }
        )
    buffer_entries = [
        {
            "name": name,
            "shape": list(buffer.shape),
            "dtype": str(buffer.dtype),
            "numel": int(buffer.numel()),
            "sha256": tensor_sha256(buffer),
        }
        for name, buffer in model.named_buffers(remove_duplicate=False)
    ]
    modules = []
    for name, module in model.named_modules():
        modules.append(
            {
                "name": name or "<root>",
                "class": type(module).__name__,
                "direct_parameters": [parameter_name for parameter_name, _ in module.named_parameters(recurse=False)],
                "direct_buffers": [buffer_name for buffer_name, _ in module.named_buffers(recurse=False)],
            }
        )
    tied_parameters = [names for names in aliases.values() if len(names) > 1]
    model_device = _model_device(model)
    runtime = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__ if transformers is not None else None,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(model_device),
        "gpu": torch.cuda.get_device_name(model_device) if model_device.type == "cuda" else None,
    }
    body = {
        "scope": "Static architecture and deterministic logical-value tensor hashes after contiguous CPU canonicalization; not original-storage hashes or a formal proof.",
        "model": {
            "class": type(model).__name__,
            "snapshot_sha256": hash_model_snapshot(model_path),
            "config": model.config.to_dict(),
            "unique_parameter_count": sum(unique_numels.values()),
            "unique_parameter_tensors": len(unique_hashes),
            "listed_parameter_bindings": len(parameter_entries),
            "parameter_tensors": parameter_entries,
            "buffer_tensors": buffer_entries,
            "tied_parameter_names": tied_parameters,
        },
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "vocabulary_size": int(tokenizer.vocab_size),
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        },
        "modules": modules,
        "operator_semantics": operator_semantics_catalog(),
        "runtime": runtime,
    }
    body["manifest_sha256"] = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    return body


class ModuleExecutionTracer(AbstractContextManager):
    def __init__(self, model: Any):
        _require_torch()
        self.model = model
        self.records: list[dict[str, Any]] = []
        self.handles = []
        self.tensor_cache: dict[tuple[int, int], dict[str, Any]] = {}
        self.generation_step = 0

    def _hook(self, name: str, module: Any):
        def hook(_module: Any, arguments: Any, keyword_arguments: Any, output: Any) -> None:
            payload = {
                "trace_level": "module",
                "generation_step": self.generation_step,
                "module": name,
                "module_class": type(module).__name__,
                "inputs": _value_descriptor({"args": arguments, "kwargs": keyword_arguments}, self.tensor_cache),
                "outputs": _value_descriptor(output, self.tensor_cache),
            }
            append_chain_record(self.records, payload)

        return hook

    def __enter__(self):
        for name, module in self.model.named_modules():
            if name and not tuple(module.children()):
                self.handles.append(module.register_forward_hook(self._hook(name, module), with_kwargs=True))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False


class ATenExecutionTracer(TorchDispatchMode):
    def __init__(self):
        _require_torch()
        super().__init__()
        self.records: list[dict[str, Any]] = []
        self.tensor_cache: dict[tuple[int, int], dict[str, Any]] = {}
        self.generation_step = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        keyword_arguments = kwargs or {}
        with torch._C._DisableTorchDispatch():
            inputs = _value_descriptor({"args": args, "kwargs": keyword_arguments}, self.tensor_cache)
            output = func(*args, **keyword_arguments)
            outputs = _value_descriptor(output, self.tensor_cache)
        payload = {
            "trace_level": "aten",
            "generation_step": self.generation_step,
            "operator": str(func),
            "inputs": inputs,
            "outputs": outputs,
        }
        append_chain_record(self.records, payload)
        return output


def _trace_context(model: Any, trace_level: str):
    if trace_level == "module":
        return ModuleExecutionTracer(model)
    if trace_level == "aten":
        return ATenExecutionTracer()
    raise ValueError("trace_level must be module or aten")


def _top_two(logits: Any, tokenizer: Any) -> list[dict[str, Any]]:
    values, indices = torch.topk(logits.float(), k=2)
    return [
        {
            "token_id": int(indices[index]),
            "token_text": tokenizer.decode([int(indices[index])]),
            "logit": float(values[index]),
        }
        for index in range(2)
    ]


def summarize_operational_evidence(
    manifest: dict[str, Any],
    module_report: dict[str, Any],
    aten_report: dict[str, Any],
    model_label: str,
) -> dict[str, Any]:
    module_prediction = module_report["prediction"]
    aten_prediction = aten_report["prediction"]
    if module_prediction["output_commitment_sha256"] != aten_prediction["output_commitment_sha256"]:
        raise ValueError("Module and ATen reports do not commit to the same output")
    if module_report["input"]["token_ids_sha256"] != aten_report["input"]["token_ids_sha256"]:
        raise ValueError("Module and ATen reports do not use the same input tokens")
    step = aten_prediction["steps"][0]
    top_two = step["top_two"]
    operator_counts = Counter(record["payload"]["operator"] for record in aten_report["trace"]["records"])
    config = manifest["model"]["config"]
    runtime = manifest["runtime"]
    return {
        "scope": "Operational-semantics execution-recording prototype using deterministic logical-value hashes after contiguous CPU canonicalization; not original-storage hashing or an independently formalized proof.",
        "model": {
            "repository": model_label,
            "class": manifest["model"]["class"],
            "snapshot_sha256": manifest["model"]["snapshot_sha256"],
            "manifest_sha256": manifest["manifest_sha256"],
            "unique_parameters": manifest["model"]["unique_parameter_count"],
            "unique_parameter_tensors": manifest["model"]["unique_parameter_tensors"],
            "listed_parameter_bindings": manifest["model"]["listed_parameter_bindings"],
            "buffer_tensors": len(manifest["model"]["buffer_tensors"]),
            "module_count": len(manifest["modules"]),
            "tied_parameters": manifest["model"]["tied_parameter_names"][0],
            "layers": config["num_hidden_layers"],
            "hidden_size": config["hidden_size"],
            "attention_heads": config["num_attention_heads"],
            "head_dimension": config["head_dim"],
            "mlp_intermediate_size": config["intermediate_size"],
            "vocabulary_size": config["vocab_size"],
            "dtype": config["torch_dtype"],
        },
        "runtime": {
            "torch": runtime["torch"],
            "transformers": runtime["transformers"],
            "device": runtime["gpu"] or runtime["device"],
            "manual_decode": "full-context recomputation with use_cache=False and torch.argmax",
        },
        "input": {
            "prompt": aten_report["input"]["prompt"],
            "prompt_utf8_sha256": aten_report["input"]["prompt_utf8_sha256"],
            "token_count": len(aten_report["input"]["token_ids"]),
            "token_ids_sha256": aten_report["input"]["token_ids_sha256"],
        },
        "prediction": {
            "requested_output_prefix_tokens": aten_prediction["requested_prefix_length"],
            "selected_token_id": step["selected_token_id"],
            "selected_text": step["selected_token_text"],
            "selected_logit": top_two[0]["logit"],
            "runner_up_token_id": top_two[1]["token_id"],
            "runner_up_text": top_two[1].get("token_text"),
            "runner_up_logit": top_two[1]["logit"],
            "winning_margin": step["winning_margin"],
            "logits_tensor_sha256": step["logits"]["sha256"],
            "output_commitment_sha256": aten_prediction["output_commitment_sha256"],
            "transformers_generate_exact_match": aten_report["reference_generate_comparison"]["exact_match"],
        },
        "module_trace": {
            "record_count": module_report["trace"]["record_count"],
            "root_sha256": module_report["trace"]["root_sha256"],
            "chain_verified": module_report["trace"]["chain_verification"]["valid"],
        },
        "aten_trace": {
            "record_count": aten_report["trace"]["record_count"],
            "root_sha256": aten_report["trace"]["root_sha256"],
            "chain_verified": aten_report["trace"]["chain_verification"]["valid"],
            "most_frequent_operators": dict(operator_counts.most_common(10)),
        },
        "interpretation": "The prototype predicts a finite output prefix by directly executing the frozen model, commits logits and intermediate logical tensor values, and hash-chains module or ATen records. Activation-direction experiments remain separate evidence rather than the complete rationale.",
        "limitations": [
            "The predictor and transformers.generate comparison use the same model implementation and are not independent implementations.",
            "The trace records observed PyTorch execution; a separate formally specified reference interpreter and proof of equivalence do not yet exist.",
            "ATen tracing may represent a fused kernel as one dispatched operation rather than exposing its internal arithmetic instructions.",
            "Hash-chain verification detects record modification but does not prove that each operator implemented an independently specified equation.",
            "Logical-value tensor hashes do not preserve original strides, storage offsets, device byte order, or semantic meaning.",
            "Results are scoped to the recorded checkpoint, prompt, runtime, dtype, device, and decoding path.",
            "No GMP, biological, clinical, product-quality, or patient-safety conclusion is supported.",
        ],
    }


def predict_with_provenance(
    model: Any,
    tokenizer: Any,
    model_path: str | Path,
    prompt: str,
    max_new_tokens: int = 1,
    trace_level: str = "module",
) -> dict[str, Any]:
    _require_gemma_dependencies()
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    original_inputs = _tokenize(tokenizer, prompt, _model_device(model))
    original_ids = original_inputs["input_ids"].clone()
    generated_ids = original_ids.clone()
    attention_mask = original_inputs.get("attention_mask", torch.ones_like(generated_ids)).clone()
    predicted_ids = []
    token_steps = []
    tracer = _trace_context(model, trace_level)

    with tracer, torch.no_grad():
        for step in range(max_new_tokens):
            tracer.generation_step = step
            outputs = model(input_ids=generated_ids, attention_mask=attention_mask, use_cache=False)
            logits = outputs.logits[0, -1]
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError(f"Non-finite logits at generation step {step}")
            top_two = _top_two(logits, tokenizer)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            token_id = int(next_token)
            predicted_ids.append(token_id)
            token_steps.append(
                {
                    "step": step,
                    "context_token_count": int(generated_ids.shape[1]),
                    "logits": tensor_descriptor(logits),
                    "top_two": top_two,
                    "winning_margin": top_two[0]["logit"] - top_two[1]["logit"],
                    "selected_token_id": token_id,
                    "selected_token_text": tokenizer.decode([token_id]),
                    "selection_rule": "torch.argmax; first maximal index under the deployed implementation",
                }
            )
            generated_ids = torch.cat((generated_ids, next_token.reshape(1, 1)), dim=1)
            attention_mask = torch.cat(
                (attention_mask, torch.ones((attention_mask.shape[0], 1), device=attention_mask.device, dtype=attention_mask.dtype)),
                dim=1,
            )
            if token_id == tokenizer.eos_token_id:
                break

    generation_config = transformers.GenerationConfig(
        do_sample=False,
        use_cache=False,
        top_p=None,
        top_k=None,
        cache_implementation=None,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    with torch.no_grad():
        reference = model.generate(
            input_ids=original_ids,
            attention_mask=original_inputs.get("attention_mask", torch.ones_like(original_ids)),
            max_new_tokens=len(predicted_ids),
            generation_config=generation_config,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    reference_new_ids = reference[0, original_ids.shape[1]:].tolist()
    output_text = tokenizer.decode(predicted_ids, skip_special_tokens=False)
    output_commitment = hashlib.sha256(
        canonical_json({"token_ids": predicted_ids, "text_utf8_sha256": hashlib.sha256(output_text.encode("utf-8")).hexdigest()}).encode("utf-8")
    ).hexdigest()
    trace_root = tracer.records[-1]["record_hash"] if tracer.records else GENESIS
    return {
        "scope": "Deterministic logical-value execution recording for one configured PyTorch runtime; not an original-storage trace, independent formal proof, or semantic explanation.",
        "identity": {
            "model_class": type(model).__name__,
            "model_snapshot_sha256": hash_model_snapshot(model_path),
            "tokenizer_class": type(tokenizer).__name__,
            "runtime": {
                "torch": torch.__version__,
                "transformers": transformers.__version__ if transformers is not None else None,
                "device": str(_model_device(model)),
                "dtype": _model_dtype(model),
            },
        },
        "input": {
            "prompt": prompt,
            "prompt_utf8_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "token_ids": original_ids[0].tolist(),
            "token_ids_sha256": hashlib.sha256(canonical_json(original_ids[0].tolist()).encode("utf-8")).hexdigest(),
        },
        "prediction": {
            "requested_prefix_length": max_new_tokens,
            "generated_token_ids": predicted_ids,
            "generated_text": output_text,
            "output_commitment_sha256": output_commitment,
            "steps": token_steps,
        },
        "trace": {
            "level": trace_level,
            "record_count": len(tracer.records),
            "root_sha256": trace_root,
            "chain_verification": verify_trace_chain(tracer.records, trace_root),
            "records": tracer.records,
        },
        "reference_generate_comparison": {
            "generated_token_ids": reference_new_ids,
            "exact_match": predicted_ids == reference_new_ids,
            "method": "transformers.generate with do_sample=False and use_cache=False",
        },
        "limitations": [
            "The trace records the deployed PyTorch execution rather than proving equivalence to an independently formalized Gemma interpreter.",
            "Module traces omit functional operations inside modules; ATen traces expose dispatched primitives but may treat fused kernels as one operation.",
            "Tensor hashes establish deterministic logical-value identity after contiguous CPU canonicalization but do not assign human semantic meaning.",
            "Token-level equality is scoped to the recorded checkpoint, runtime, dtype, device, prompt, and decoding configuration.",
        ],
    }
