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
    positive_validation: tuple[str, ...]
    negative_validation: tuple[str, ...]
    positive_test: tuple[str, ...]
    negative_test: tuple[str, ...]


class ActivationCapture:
    def __init__(self, model: Any, component: str = "residual"):
        self.model = model
        self.layers = self._resolve_layers(model)
        if component not in {"residual", "attention", "mlp"}:
            raise ValueError(f"Unsupported component {component!r}")
        self.component = component

    @staticmethod
    def _resolve_layers(model: Any) -> Any:
        candidate = getattr(model, "model", None)
        layers = getattr(candidate, "layers", None)
        if layers is None:
            raise ValueError("Expected a causal language model exposing model.layers")
        return layers

    def _module(self, layer_index: int) -> Any:
        layer = self.layers[layer_index]
        if self.component == "attention":
            return layer.self_attn
        if self.component == "mlp":
            return layer.mlp
        return layer

    @staticmethod
    def _position_index(sequence_length: int, position: str | int) -> int:
        if isinstance(position, int):
            index = position if position >= 0 else sequence_length + position
        elif position == "quarter":
            index = sequence_length // 4
        elif position == "middle":
            index = sequence_length // 2
        elif position == "last":
            index = sequence_length - 1
        else:
            raise ValueError(f"Unsupported token position {position!r}")
        if index < 0 or index >= sequence_length:
            raise ValueError(f"Token position {position!r} is outside sequence length {sequence_length}")
        return index

    @staticmethod
    def _tensor(output: Any) -> Any:
        tensor = output[0] if isinstance(output, tuple) else output
        if not hasattr(tensor, "shape") or len(tensor.shape) != 3:
            raise ValueError("Expected component output with shape [batch, sequence, hidden]")
        return tensor

    @contextmanager
    def capture(self, layer_indices: tuple[int, ...], position: str | int = "last") -> Iterator[dict[int, Any]]:
        activations: dict[int, Any] = {}
        handles = []

        def make_hook(index: int):
            def hook(_module: Any, _arguments: Any, output: Any) -> None:
                tensor = self._tensor(output)
                token_index = self._position_index(tensor.shape[1], position)
                activations[index] = tensor[0, token_index].detach().float().cpu().clone()

            return hook

        try:
            for index in layer_indices:
                handles.append(self._module(index).register_forward_hook(make_hook(index)))
            yield activations
        finally:
            for handle in handles:
                handle.remove()

    @contextmanager
    def intervene(
        self,
        layer_index: int,
        direction: Any,
        magnitude: float,
        position: str | int = "last",
    ) -> Iterator[None]:
        import torch

        def hook(_module: Any, _arguments: Any, output: Any) -> Any:
            tensor = self._tensor(output)
            modified = tensor.clone()
            token_index = self._position_index(modified.shape[1], position)
            vector = direction.to(device=modified.device, dtype=modified.dtype)
            modified[:, token_index, :] = modified[:, token_index, :] + magnitude * vector
            if isinstance(output, tuple):
                return (modified, *output[1:])
            return modified

        handle = self._module(layer_index).register_forward_hook(hook)
        try:
            with torch.no_grad():
                yield
        finally:
            handle.remove()


def default_low_oxygen_prompts() -> PromptSet:
    question = " Does this record describe a low-oxygen condition requiring review? Answer Yes or No."
    test_question = "\nREVIEW_QUERY: Is low oxygen review required? Respond Yes or No."
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
        positive_validation=(
            "The latest oxygen measurement is diminished and falling." + question,
            "A low dissolved-oxygen value is accompanied by continued decline." + question,
            "Oxygen has become limited and the trajectory remains negative." + question,
            "The record shows an oxygen deficit that is worsening." + question,
        ),
        negative_validation=(
            "The latest oxygen measurement is adequate and unchanged." + question,
            "A normal dissolved-oxygen value is accompanied by a flat trend." + question,
            "Oxygen remains sufficient and the trajectory is stable." + question,
            "The record shows no oxygen deficit or deterioration." + question,
        ),
        positive_test=(
            "INPUT_RECORD|do_pct=27|do_slope_pct_per_min=-1.2|state=falling" + test_question,
            "INPUT_RECORD|do_pct=31|do_slope_pct_per_min=-0.9|state=declining" + test_question,
            "TABLE_ROW: oxygen=29 percent; trend=downward; status=limited" + test_question,
            "TELEMETRY: DO low; derivative negative; observation worsening" + test_question,
        ),
        negative_test=(
            "INPUT_RECORD|do_pct=47|do_slope_pct_per_min=0.0|state=stable" + test_question,
            "INPUT_RECORD|do_pct=44|do_slope_pct_per_min=0.1|state=steady" + test_question,
            "TABLE_ROW: oxygen=49 percent; trend=flat; status=adequate" + test_question,
            "TELEMETRY: DO normal; derivative zero; observation unchanged" + test_question,
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


def capture_prompt_activations(
    model: Any,
    tokenizer: Any,
    prompt: str,
    capture: ActivationCapture,
    layer_indices: tuple[int, ...] | None = None,
    position: str | int = "last",
) -> dict[int, np.ndarray]:
    import torch

    indices = layer_indices or tuple(range(len(capture.layers)))
    with capture.capture(indices, position) as activations, torch.no_grad():
        model(**_tokenize(tokenizer, prompt, _model_device(model)), use_cache=False)
    return {index: tensor.numpy() for index, tensor in activations.items()}


def _activation_matrix(
    model: Any,
    tokenizer: Any,
    prompts: tuple[str, ...],
    capture: ActivationCapture,
    layer_indices: tuple[int, ...] | None = None,
    position: str | int = "last",
) -> dict[int, np.ndarray]:
    indices = layer_indices or tuple(range(len(capture.layers)))
    collected: dict[int, list[np.ndarray]] = {index: [] for index in indices}
    for prompt in prompts:
        for index, activation in capture_prompt_activations(model, tokenizer, prompt, capture, indices, position).items():
            collected[index].append(activation)
    return {index: np.stack(values) for index, values in collected.items()}


def _normalized(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError("Concept direction has zero norm")
    return vector / norm


def _direction(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    return _normalized(np.mean(positive, axis=0) - np.mean(negative, axis=0))


def _effect_size(positive: np.ndarray, negative: np.ndarray) -> float:
    combined = np.concatenate((positive, negative))
    deviation = float(np.std(combined))
    return float((np.mean(positive) - np.mean(negative)) / deviation) if deviation else 0.0


def _direction_metrics(direction: np.ndarray, positive: np.ndarray, negative: np.ndarray) -> dict[str, float]:
    positive_projection = positive @ direction
    negative_projection = negative @ direction
    labels = np.concatenate((np.ones(len(positive_projection)), np.zeros(len(negative_projection))))
    scores = np.concatenate((positive_projection, negative_projection))
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "effect_size": _effect_size(positive_projection, negative_projection),
        "projection_gap": float(np.mean(positive_projection) - np.mean(negative_projection)),
    }


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


def _intervention_metrics(
    model: Any,
    tokenizer: Any,
    capture: ActivationCapture,
    layer_index: int,
    direction: np.ndarray,
    magnitude: float,
    prompts: tuple[str, ...],
    positive_token_id: int,
    negative_token_id: int,
    position: str | int = "last",
) -> dict[str, Any]:
    import torch

    baseline = [_next_token_margin(model, tokenizer, prompt, positive_token_id, negative_token_id) for prompt in prompts]
    with capture.intervene(layer_index, torch.from_numpy(direction), magnitude, position):
        intervened = [_next_token_margin(model, tokenizer, prompt, positive_token_id, negative_token_id) for prompt in prompts]
    deltas = np.array(intervened) - np.array(baseline)
    return {
        "prompt_count": len(prompts),
        "magnitude": magnitude,
        "baseline_yes_minus_no_logit_mean": float(np.mean(baseline)),
        "intervened_yes_minus_no_logit_mean": float(np.mean(intervened)),
        "logit_margin_delta_mean": float(np.mean(deltas)),
        "logit_margin_delta_minimum": float(np.min(deltas)),
        "logit_margin_delta_maximum": float(np.max(deltas)),
        "positive_effect_fraction": float(np.mean(deltas > 0)),
    }


def _null_summary(values: list[float], observed: float) -> dict[str, Any]:
    array = np.array(values)
    return {
        "iterations": len(values),
        "mean_roc_auc": float(np.mean(array)),
        "standard_deviation": float(np.std(array)),
        "95th_percentile": float(np.quantile(array, 0.95)),
        "maximum": float(np.max(array)),
        "empirical_right_tail_p_value": float((1 + np.sum(array >= observed)) / (len(array) + 1)),
    }


def _null_controls(
    positive_training: dict[int, np.ndarray],
    negative_training: dict[int, np.ndarray],
    positive_validation: dict[int, np.ndarray],
    negative_validation: dict[int, np.ndarray],
    positive_test: dict[int, np.ndarray],
    negative_test: dict[int, np.ndarray],
    observed_auc: float,
    iterations: int = 200,
    seed: int = 17,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    layers = sorted(positive_training)
    positive_count = len(positive_training[layers[0]])
    shuffled_aucs = []
    random_aucs = []
    shuffled_selected_layers = []
    random_selected_layers = []
    for _ in range(iterations):
        permutation = rng.permutation(positive_count + len(negative_training[layers[0]]))
        shuffled_candidates = []
        random_candidates = []
        for layer in layers:
            combined = np.concatenate((positive_training[layer], negative_training[layer]))
            shuffled_direction = _direction(combined[permutation[:positive_count]], combined[permutation[positive_count:]])
            shuffled_validation = _direction_metrics(shuffled_direction, positive_validation[layer], negative_validation[layer])
            shuffled_candidates.append((shuffled_validation["roc_auc"], shuffled_validation["effect_size"], layer, shuffled_direction))
            random_direction = _normalized(rng.normal(size=combined.shape[1]))
            random_validation = _direction_metrics(random_direction, positive_validation[layer], negative_validation[layer])
            random_candidates.append((random_validation["roc_auc"], random_validation["effect_size"], layer, random_direction))
        shuffled_selected = max(shuffled_candidates, key=lambda item: (item[0], item[1]))
        random_selected = max(random_candidates, key=lambda item: (item[0], item[1]))
        shuffled_selected_layers.append(shuffled_selected[2])
        random_selected_layers.append(random_selected[2])
        shuffled_aucs.append(
            _direction_metrics(shuffled_selected[3], positive_test[shuffled_selected[2]], negative_test[shuffled_selected[2]])["roc_auc"]
        )
        random_aucs.append(
            _direction_metrics(random_selected[3], positive_test[random_selected[2]], negative_test[random_selected[2]])["roc_auc"]
        )
    return {
        "selection_protocol": "Every null iteration independently selects a layer on validation data before test evaluation.",
        "observed_test_roc_auc": observed_auc,
        "shuffled_training_labels": {
            **_null_summary(shuffled_aucs, observed_auc),
            "selected_layer_counts": {str(layer): shuffled_selected_layers.count(layer) for layer in sorted(set(shuffled_selected_layers))},
        },
        "random_directions": {
            **_null_summary(random_aucs, observed_auc),
            "selected_layer_counts": {str(layer): random_selected_layers.count(layer) for layer in sorted(set(random_selected_layers))},
        },
    }


def hash_model_snapshot(model_path: str | Path) -> str:
    digest = hashlib.sha256()
    root = Path(model_path)
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _component_analysis(
    model: Any,
    tokenizer: Any,
    prompt_set: PromptSet,
    layer_index: int,
    component: str,
    position: str,
    positive_token_id: int,
    negative_token_id: int,
) -> dict[str, Any]:
    capture = ActivationCapture(model, component)
    indices = (layer_index,)
    positive_training = _activation_matrix(model, tokenizer, prompt_set.positive_training, capture, indices, position)[layer_index]
    negative_training = _activation_matrix(model, tokenizer, prompt_set.negative_training, capture, indices, position)[layer_index]
    positive_test = _activation_matrix(model, tokenizer, prompt_set.positive_test, capture, indices, position)[layer_index]
    negative_test = _activation_matrix(model, tokenizer, prompt_set.negative_test, capture, indices, position)[layer_index]
    direction = _direction(positive_training, negative_training)
    training_metrics = _direction_metrics(direction, positive_training, negative_training)
    test_metrics = _direction_metrics(direction, positive_test, negative_test)
    intervention = _intervention_metrics(
        model,
        tokenizer,
        capture,
        layer_index,
        direction,
        training_metrics["projection_gap"],
        prompt_set.negative_test,
        positive_token_id,
        negative_token_id,
        position,
    )
    return {
        "component": component,
        "position": position,
        "test": test_metrics,
        "intervention": intervention,
    }


def run_concept_experiment(model: Any, tokenizer: Any, model_path: str | Path, prompts: PromptSet | None = None) -> dict[str, Any]:
    prompt_set = prompts or default_low_oxygen_prompts()
    residual = ActivationCapture(model, "residual")
    positive_training = _activation_matrix(model, tokenizer, prompt_set.positive_training, residual)
    negative_training = _activation_matrix(model, tokenizer, prompt_set.negative_training, residual)
    positive_validation = _activation_matrix(model, tokenizer, prompt_set.positive_validation, residual)
    negative_validation = _activation_matrix(model, tokenizer, prompt_set.negative_validation, residual)
    positive_test = _activation_matrix(model, tokenizer, prompt_set.positive_test, residual)
    negative_test = _activation_matrix(model, tokenizer, prompt_set.negative_test, residual)
    directions = {}
    layer_selection = []

    for index in range(len(residual.layers)):
        direction = _direction(positive_training[index], negative_training[index])
        directions[index] = direction
        validation = _direction_metrics(direction, positive_validation[index], negative_validation[index])
        layer_selection.append({"layer": index, "validation": validation})

    selected = max(layer_selection, key=lambda item: (item["validation"]["roc_auc"], item["validation"]["effect_size"]))
    selected_layer = int(selected["layer"])
    selected_direction = directions[selected_layer]
    final_test = _direction_metrics(selected_direction, positive_test[selected_layer], negative_test[selected_layer])
    positive_token_id = _single_token_id(tokenizer, "Yes")
    negative_token_id = _single_token_id(tokenizer, "No")
    intervention = _intervention_metrics(
        model,
        tokenizer,
        residual,
        selected_layer,
        selected_direction,
        _direction_metrics(selected_direction, positive_training[selected_layer], negative_training[selected_layer])["projection_gap"],
        prompt_set.negative_test,
        positive_token_id,
        negative_token_id,
    )
    null_controls = _null_controls(
        positive_training,
        negative_training,
        positive_validation,
        negative_validation,
        positive_test,
        negative_test,
        final_test["roc_auc"],
    )
    position_localization = [
        _component_analysis(model, tokenizer, prompt_set, selected_layer, "residual", position, positive_token_id, negative_token_id)
        for position in ("quarter", "middle", "last")
    ]
    component_localization = [
        _component_analysis(model, tokenizer, prompt_set, selected_layer, component, "last", positive_token_id, negative_token_id)
        for component in ("attention", "mlp")
    ]
    layer_types = getattr(model.config, "layer_types", None)
    prompt_manifest = {
        "positive_training": prompt_set.positive_training,
        "negative_training": prompt_set.negative_training,
        "positive_validation": prompt_set.positive_validation,
        "negative_validation": prompt_set.negative_validation,
        "positive_test": prompt_set.positive_test,
        "negative_test": prompt_set.negative_test,
    }
    return {
        "scope": "Exploratory activation analysis on synthetic language prompts; not a complete interpretation or biological validation.",
        "model": {
            "path": str(Path(model_path)),
            "class": type(model).__name__,
            "snapshot_sha256": hash_model_snapshot(model_path),
            "layers": len(residual.layers),
            "hidden_size": int(model.config.hidden_size),
            "dtype": str(model.dtype),
            "device": str(_model_device(model)),
        },
        "concept": {
            "name": "low_oxygen_language_contrast",
            "method": "mean activation difference with separate training, validation, and format-shifted test prompts",
            "prompt_manifest_sha256": hashlib.sha256(json.dumps(prompt_manifest, sort_keys=True).encode("utf-8")).hexdigest(),
            "positive_target_token": {"text": "Yes", "id": positive_token_id},
            "negative_target_token": {"text": "No", "id": negative_token_id},
            "split_sizes_per_class": {"training": 6, "validation": 4, "format_shift_test": 4},
        },
        "selection_protocol": "Layer selected only by validation ROC AUC, then effect size; final metrics and interventions use format-shifted test prompts.",
        "selected_layer": {
            "layer": selected_layer,
            "attention_type": layer_types[selected_layer] if layer_types else "not_reported",
            "validation": selected["validation"],
        },
        "final_format_shift_test": final_test,
        "selected_layer_intervention": intervention,
        "null_controls": null_controls,
        "token_position_localization": position_localization,
        "component_localization": component_localization,
        "layer_selection": layer_selection,
        "limitations": [
            "A linear direction can encode lexical, formatting, or authored-dataset correlations rather than a stable human concept.",
            "The format-shift test set is separately formatted but still authored for this experiment and contains only eight prompts.",
            "Empirical null p-values are descriptive controls for this finite prompt set, not population-level significance claims.",
            "Residual and component interventions demonstrate sensitivity, not a complete causal circuit or semantic proof.",
            "Attention and MLP outputs are aggregate component outputs; individual heads and neurons are not localized.",
            "Results on Gemma 3 270M do not establish behavior of Gemma 4 31B or another checkpoint.",
            "The bfloat16 results are not claimed to be numerically identical across hardware, drivers, or library versions.",
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
