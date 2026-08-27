from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class Observation:
    value: float | bool
    unit: str
    observed_at: datetime
    source: str
    quality: str = "valid"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if isinstance(self.value, float) and not math.isfinite(self.value):
            result["value"] = "NaN" if math.isnan(self.value) else ("Infinity" if self.value > 0 else "-Infinity")
        result["observed_at"] = self.observed_at.isoformat()
        return result


@dataclass(frozen=True)
class InputSpec:
    name: str
    value_type: str
    unit: str
    minimum: float | None = None
    maximum: float | None = None
    max_age_seconds: int | None = None


@dataclass(frozen=True)
class ModelSpec:
    name: str
    link: str
    bias: float
    weights: dict[str, float]


@dataclass(frozen=True)
class Comparison:
    field: str
    operator: str
    expected: float | bool


@dataclass(frozen=True)
class Recommendation:
    field: str
    delta: float
    maximum: float


@dataclass(frozen=True)
class RuleSpec:
    name: str
    condition: Comparison
    requirements: tuple[Comparison, ...]
    recommendation: Recommendation
    abstain_reason: str


@dataclass(frozen=True)
class Policy:
    name: str
    version: str
    intended_use: str
    mode: str
    inputs: dict[str, InputSpec]
    model: ModelSpec
    rules: tuple[RuleSpec, ...]
    default_reason: str
    source_sha256: str


@dataclass(frozen=True)
class TraceStep:
    stage: str
    result: str
    detail: str
    values: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    decision_id: str
    status: str
    reason: str
    policy_name: str
    policy_version: str
    policy_sha256: str
    evaluated_at: datetime
    inputs: dict[str, Observation]
    model_output: dict[str, Any] | None
    recommendation: dict[str, Any] | None
    human_review_required: bool
    trace: tuple[TraceStep, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "status": self.status,
            "reason": self.reason,
            "policy": {
                "name": self.policy_name,
                "version": self.policy_version,
                "sha256": self.policy_sha256,
            },
            "evaluated_at": self.evaluated_at.isoformat(),
            "inputs": {name: observation.to_dict() for name, observation in sorted(self.inputs.items())},
            "model_output": self.model_output,
            "recommendation": self.recommendation,
            "human_review_required": self.human_review_required,
            "trace": [asdict(step) for step in self.trace],
        }
