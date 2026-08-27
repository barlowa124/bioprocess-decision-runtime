from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = ("dissolved_oxygen_pct", "dissolved_oxygen_slope")


@dataclass(frozen=True)
class TrainingResult:
    policy_path: Path
    report: dict[str, Any]


def generate_synthetic_training_data(samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if samples < 100:
        raise ValueError("At least 100 synthetic samples are required")
    rng = np.random.default_rng(seed)
    oxygen = rng.uniform(20.0, 60.0, samples)
    slope = np.clip(rng.normal(-0.15, 0.75, samples), -3.0, 3.0)
    latent_logit = 5.0 - 0.12 * oxygen - 1.5 * slope + rng.normal(0.0, 0.45, samples)
    probability = 1.0 / (1.0 + np.exp(-latent_logit))
    labels = rng.binomial(1, probability)
    return np.column_stack((oxygen, slope)), labels


def _raw_coefficients(pipeline: Pipeline) -> tuple[float, dict[str, float]]:
    scaler = pipeline.named_steps["scale"]
    model = pipeline.named_steps["model"]
    scaled_weights = model.coef_[0]
    raw_weights = scaled_weights / scaler.scale_
    raw_bias = float(model.intercept_[0] - np.sum(scaled_weights * scaler.mean_ / scaler.scale_))
    return raw_bias, {name: float(value) for name, value in zip(FEATURES, raw_weights, strict=True)}


def _render_policy(template: str, version: str, bias: float, weights: dict[str, float]) -> str:
    rendered = re.sub(r"^(POLICY\s+\S+\s+VERSION\s+)\S+$", rf"\g<1>{version}", template, flags=re.MULTILINE)
    rendered = re.sub(r"^BIAS\s+[-+0-9.eE]+$", f"BIAS {bias:.17g}", rendered, flags=re.MULTILINE)
    for name, value in weights.items():
        pattern = rf"^WEIGHT\s+{re.escape(name)}\s+[-+0-9.eE]+$"
        rendered, count = re.subn(pattern, f"WEIGHT {name} {value:.17g}", rendered, flags=re.MULTILINE)
        if count != 1:
            raise ValueError(f"Template must contain exactly one weight for {name}")
    return rendered


def train_transparent_policy(
    template_path: str | Path,
    output_policy_path: str | Path,
    samples: int = 2000,
    seed: int = 17,
    version: str = "synthetic-1.0.0",
) -> TrainingResult:
    x, y = generate_synthetic_training_data(samples, seed)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=seed,
        stratify=y,
    )
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=seed)),
        ]
    )
    cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cross_validation_auc = cross_val_score(pipeline, x_train, y_train, cv=cross_validation, scoring="roc_auc")
    pipeline.fit(x_train, y_train)
    probability = pipeline.predict_proba(x_test)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    bias, weights = _raw_coefficients(pipeline)
    raw_logits = bias + x_test @ np.array([weights[name] for name in FEATURES])
    sklearn_logits = pipeline.decision_function(x_test)

    report: dict[str, Any] = {
        "scope": "Synthetic software demonstration; metrics do not establish biological, clinical, or GMP performance.",
        "data_generator": {
            "samples": samples,
            "seed": seed,
            "features": list(FEATURES),
            "label_process": "Bernoulli draw from a noisy illustrative logistic relationship",
        },
        "split": {
            "training_samples": int(len(x_train)),
            "test_samples": int(len(x_test)),
            "positive_fraction_training": float(np.mean(y_train)),
            "positive_fraction_test": float(np.mean(y_test)),
        },
        "evaluation": {
            "test_roc_auc": float(roc_auc_score(y_test, probability)),
            "test_accuracy_at_0_5": float(accuracy_score(y_test, prediction)),
            "test_brier_score": float(brier_score_loss(y_test, probability)),
            "cross_validation_roc_auc_mean": float(np.mean(cross_validation_auc)),
            "cross_validation_roc_auc_standard_deviation": float(np.std(cross_validation_auc)),
            "maximum_policy_logit_translation_error": float(np.max(np.abs(raw_logits - sklearn_logits))),
        },
        "interpretable_model": {
            "link": "LOGISTIC",
            "bias": bias,
            "weights_in_declared_input_units": weights,
            "runtime_formula": "sigmoid(bias + sum(weight * input))",
        },
        "limitations": [
            "The generator is illustrative rather than a validated mechanistic bioprocess model.",
            "The held-out data come from the same synthetic generator as the training data.",
            "No process, product-quality, clinical, or patient-safety conclusion can be drawn.",
        ],
    }

    output_path = Path(output_policy_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    template = Path(template_path).read_text(encoding="utf-8")
    output_path.write_text(_render_policy(template, version, bias, weights), encoding="utf-8")
    return TrainingResult(output_path, report)


def write_training_report(path: str | Path, report: dict[str, Any]) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
