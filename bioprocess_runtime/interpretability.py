from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class PromptSet:
    positive_training: tuple[str, ...]
    negative_training: tuple[str, ...]
    positive_held_out: tuple[str, ...]
    negative_held_out: tuple[str, ...]


class ActivationCapture:
    def __init__(self, model: Any):
        self.model = model
        self.layers = self._resolve_layers(model)

    @staticmethod
    def _resolve_layers(model: Any) -> Any:
        candidate = getattr(model, "model", None)
        layers = getattr(candidate, "layers", None)
        if layers is None:
            raise ValueError("Expected a causal language model exposing model.layers")
        return layers

    @contextmanager
    def capture(self, layer_indices: tuple[int, ...]) -> Iterator[dict[int, Any]]:
        activations: dict[int, Any] = {}
        handles = []

        def make_hook(index: int):
            def hook(_module: Any, _arguments: Any, output: Any) -> None:
                tensor = output[0] if isinstance(output, tuple) else output
                activations[index] = tensor[0, -1].detach().float().cpu().clone()

            return hook

        try:
            for index in layer_indices:
                handles.append(self.layers[index].register_forward_hook(make_hook(index)))
            yield activations
        finally:
            for handle in handles:
                handle.remove()

    @contextmanager
    def intervene(self, layer_index: int, direction: Any, magnitude: float) -> Iterator[None]:
        import torch

        layer = self.layers[layer_index]

        def hook(_module: Any, _arguments: Any, output: Any) -> Any:
            tensor = output[0] if isinstance(output, tuple) else output
            modified = tensor.clone()
            vector = direction.to(device=modified.device, dtype=modified.dtype)
            modified[:, -1, :] = modified[:, -1, :] + magnitude * vector
            if isinstance(output, tuple):
                return (modified, *output[1:])
            return modified

        handle = layer.register_forward_hook(hook)
        try:
            with torch.no_grad():
                yield
        finally:
            handle.remove()


def default_low_oxygen_prompts() -> PromptSet:
    question = " Does this record describe a low-oxygen condition requiring review? Answer Yes or No."
    return PromptSet(
        positive_training=(
            "Dissolved oxygen is 28 percent and declining rapidly." + question,
            "The oxygen reading is 30 percent with a negative one percent-per-minute trend." + question,
            "The vessel reports reduced dissolved oxygen and a continuing downward trend." + question,
            "Oxygen availability is low and still falling during the observation window." + question,
            "The measured dissolved oxygen has dropped below the usual region and continues to decrease." + question,
            "A sustained oxygen decline has reached a low observed value." + question,
        ),
        negative_training=(
            "Dissolved oxygen is 48 percent and stable." + question,
            "The oxygen reading is 45 percent with a zero percent-per-minute trend." + question,
            "The vessel reports adequate dissolved oxygen without a downward trend." + question,
            "Oxygen availability remains stable during the observation window." + question,
            "The measured dissolved oxygen remains in its usual region without decreasing." + question,
            "The oxygen value is steady and has not entered a low region." + question,
        ),
        positive_held_out=(
            "The latest oxygen measurement is diminished and falling." + question,
            "A low dissolved-oxygen value is accompanied by continued decline." + question,
            "Oxygen has become limited and the trajectory remains negative." + question,
            "The record shows an oxygen deficit that is worsening." + question,
        ),
        negative_held_out=(
            "The latest oxygen measurement is adequate and unchanged." + question,
            "A normal dissolved-oxygen value is accompanied by a flat trend." + question,
            "Oxygen remains sufficient and the trajectory is stable." + question,
            "The record shows no oxygen deficit or deterioration." + question,
        ),
    )


def _model_device(model: Any) -> Any:
    return next(model.parameters()).device


def _tokenize(tokenizer: Any, prompt: str, device: Any) -> dict[str, Any]:
    import torch

    if getattr(tokenizer, "chat_template", None):
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
    return {name: tensor.to(device) for name, tensor in tokenizer(prompt, return_tensors="pt").items()}


def capture_prompt_activations(model: Any, tokenizer: Any, prompt: str, capture: ActivationCapture) -> dict[int, np.ndarray]:
    import torch

    indices = tuple(range(len(capture.layers)))
    with capture.capture(indices) as activations, torch.no_grad():
        model(**_tokenize(tokenizer, prompt, _model_device(model)), use_cache=False)
    return {index: tensor.numpy() for index, tensor in activations.items()}


def _activation_matrix(model: Any, tokenizer: Any, prompts: tuple[str, ...], capture: ActivationCapture) -> dict[int, np.ndarray]:
    collected: dict[int, list[np.ndarray]] = {index: [] for index in range(len(capture.layers))}
    for prompt in prompts:
        for index, activation in capture_prompt_activations(model, tokenizer, prompt, capture).items():
            collected[index].append(activation)
    return {index: np.stack(values) for index, values in collected.items()}


def _normalized(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError("Concept direction has zero norm")
    return vector / norm


def _effect_size(positive: np.ndarray, negative: np.ndarray) -> float:
    combined = np.concatenate((positive, negative))
    deviation = float(np.std(combined))
    return float((np.mean(positive) - np.mean(negative)) / deviation) if deviation else 0.0


def _next_token_margin(model: Any, tokenizer: Any, prompt: str, positive_token_id: int, negative_token_id: int) -> float:
    import torch

    with torch.no_grad():
        logits = model(**_tokenize(tokenizer, prompt, _model_device(model)), use_cache=False).logits[0, -1]
    return float((logits[positive_token_id] - logits[negative_token_id]).float().cpu())


def _single_token_id(tokenizer: Any, text: str) -> int:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"Target {text!r} must encode to exactly one token, received {token_ids}")
    return int(token_ids[0])


def hash_model_snapshot(model_path: str | Path) -> str:
    digest = hashlib.sha256()
    root = Path(model_path)
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def run_concept_experiment(model: Any, tokenizer: Any, model_path: str | Path, prompts: PromptSet | None = None) -> dict[str, Any]:
    prompt_set = prompts or default_low_oxygen_prompts()
    capture = ActivationCapture(model)
    positive_training = _activation_matrix(model, tokenizer, prompt_set.positive_training, capture)
    negative_training = _activation_matrix(model, tokenizer, prompt_set.negative_training, capture)
    positive_held_out = _activation_matrix(model, tokenizer, prompt_set.positive_held_out, capture)
    negative_held_out = _activation_matrix(model, tokenizer, prompt_set.negative_held_out, capture)
    positive_token_id = _single_token_id(tokenizer, "Yes")
    negative_token_id = _single_token_id(tokenizer, "No")
    intervention_prompts = prompt_set.negative_held_out
    baseline_margins = [
        _next_token_margin(model, tokenizer, prompt, positive_token_id, negative_token_id)
        for prompt in intervention_prompts
    ]
    layer_results = []

    for index in range(len(capture.layers)):
        positive = positive_training[index]
        negative = negative_training[index]
        direction = _normalized(np.mean(positive, axis=0) - np.mean(negative, axis=0))
        positive_projection = positive @ direction
        negative_projection = negative @ direction
        held_positive_projection = positive_held_out[index] @ direction
        held_negative_projection = negative_held_out[index] @ direction
        labels = np.concatenate((np.ones(len(held_positive_projection)), np.zeros(len(held_negative_projection))))
        scores = np.concatenate((held_positive_projection, held_negative_projection))
        gap = float(np.mean(positive_projection) - np.mean(negative_projection))

        import torch

        tensor_direction = torch.from_numpy(direction)
        with capture.intervene(index, tensor_direction, gap):
            intervened_margins = [
                _next_token_margin(model, tokenizer, prompt, positive_token_id, negative_token_id)
                for prompt in intervention_prompts
            ]
        margin_deltas = np.array(intervened_margins) - np.array(baseline_margins)
        layer_results.append(
            {
                "layer": index,
                "direction_norm": float(np.linalg.norm(direction)),
                "training_projection_gap": gap,
                "held_out_roc_auc": float(roc_auc_score(labels, scores)),
                "held_out_effect_size": _effect_size(held_positive_projection, held_negative_projection),
                "intervention_prompt_count": len(intervention_prompts),
                "baseline_yes_minus_no_logit_mean": float(np.mean(baseline_margins)),
                "intervened_yes_minus_no_logit_mean": float(np.mean(intervened_margins)),
                "intervention_logit_margin_delta_mean": float(np.mean(margin_deltas)),
                "intervention_logit_margin_delta_minimum": float(np.min(margin_deltas)),
                "intervention_logit_margin_delta_maximum": float(np.max(margin_deltas)),
                "positive_intervention_effect_fraction": float(np.mean(margin_deltas > 0)),
                "intervention_magnitude": gap,
            }
        )

    prompt_manifest = {
        "positive_training": prompt_set.positive_training,
        "negative_training": prompt_set.negative_training,
        "positive_held_out": prompt_set.positive_held_out,
        "negative_held_out": prompt_set.negative_held_out,
    }
    best_separation = max(layer_results, key=lambda item: item["held_out_roc_auc"])
    largest_intervention = max(layer_results, key=lambda item: abs(item["intervention_logit_margin_delta_mean"]))
    return {
        "scope": "Exploratory activation analysis on synthetic language prompts; not a complete interpretation or biological validation.",
        "model": {
            "path": str(Path(model_path)),
            "class": type(model).__name__,
            "snapshot_sha256": hash_model_snapshot(model_path),
            "layers": len(capture.layers),
            "hidden_size": int(model.config.hidden_size),
            "dtype": str(model.dtype),
            "device": str(_model_device(model)),
        },
        "concept": {
            "name": "low_oxygen_language_contrast",
            "method": "mean residual-stream difference at the final prompt token",
            "prompt_manifest_sha256": hashlib.sha256(json.dumps(prompt_manifest, sort_keys=True).encode("utf-8")).hexdigest(),
            "positive_target_token": {"text": "Yes", "id": positive_token_id},
            "negative_target_token": {"text": "No", "id": negative_token_id},
            "intervention_prompts": intervention_prompts,
        },
        "best_held_out_separation": best_separation,
        "largest_absolute_intervention_effect": largest_intervention,
        "layers": layer_results,
        "limitations": [
            "A linear direction can encode lexical or formatting correlations rather than a stable human concept.",
            "Held-out prompts are synthetic and drawn from the same authored prompt family.",
            "Residual intervention demonstrates sensitivity, not a complete causal circuit or semantic proof.",
            "Results on Gemma 3 270M do not establish behavior of Gemma 4 31B or another checkpoint.",
            "No GMP, process-control, product-quality, clinical, or patient-safety conclusion is supported.",
        ],
    }


def load_local_gemma(model_path: str | Path) -> tuple[Any, Any]:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = str(Path(model_path))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the configured local Gemma experiment")
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    ).to("cuda").eval()
    return model, tokenizer
