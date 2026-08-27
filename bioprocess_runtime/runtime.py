from __future__ import annotations

import hashlib
import math
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from .domain import Comparison, Decision, Observation, Policy, TraceStep
from .serialization import canonical_json


_COMPARATORS = {
    "<": lambda actual, expected: actual < expected,
    "<=": lambda actual, expected: actual <= expected,
    "==": lambda actual, expected: actual == expected,
    ">=": lambda actual, expected: actual >= expected,
    ">": lambda actual, expected: actual > expected,
}


def _compare(comparison: Comparison, values: dict[str, float | bool]) -> tuple[bool, float | bool | None]:
    actual = values.get(comparison.field)
    if actual is None:
        return False, None
    return _COMPARATORS[comparison.operator](actual, comparison.expected), actual


def _finalize(
    policy: Policy,
    observations: dict[str, Observation],
    evaluated_at: datetime,
    status: str,
    reason: str,
    model_output: dict[str, Any] | None,
    recommendation: dict[str, Any] | None,
    trace: list[TraceStep],
) -> Decision:
    core = {
        "status": status,
        "reason": reason,
        "policy": {"name": policy.name, "version": policy.version, "sha256": policy.source_sha256},
        "evaluated_at": evaluated_at.isoformat(),
        "inputs": {name: value.to_dict() for name, value in sorted(observations.items())},
        "model_output": model_output,
        "recommendation": recommendation,
        "trace": [asdict(step) for step in trace],
    }
    decision_id = hashlib.sha256(canonical_json(core).encode("utf-8")).hexdigest()
    return Decision(
        decision_id=decision_id,
        status=status,
        reason=reason,
        policy_name=policy.name,
        policy_version=policy.version,
        policy_sha256=policy.source_sha256,
        evaluated_at=evaluated_at,
        inputs=observations,
        model_output=model_output,
        recommendation=recommendation,
        human_review_required=status in {"RECOMMENDATION", "ABSTAIN"},
        trace=tuple(trace),
    )


def evaluate(policy: Policy, observations: dict[str, Observation], evaluated_at: datetime | None = None) -> Decision:
    evaluated_at = evaluated_at or datetime.now(timezone.utc)
    if evaluated_at.tzinfo is None:
        raise ValueError("evaluated_at must be timezone-aware")

    trace: list[TraceStep] = [
        TraceStep("policy", "PASS", "Loaded bounded advisory policy", {"name": policy.name, "version": policy.version, "mode": policy.mode})
    ]
    values: dict[str, float | bool] = {}
    unexpected_inputs = set(observations).difference(policy.inputs)
    if unexpected_inputs:
        names = ", ".join(sorted(unexpected_inputs))
        trace.append(TraceStep("input_validation", "FAIL", "Unexpected observations are outside the declared interface", {"inputs": sorted(unexpected_inputs)}))
        return _finalize(policy, observations, evaluated_at, "ABSTAIN", f"Unexpected inputs: {names}", None, None, trace)

    for name, spec in policy.inputs.items():
        observation = observations.get(name)
        if observation is None:
            trace.append(TraceStep("input_validation", "FAIL", "Required observation is missing", {"input": name}))
            return _finalize(policy, observations, evaluated_at, "ABSTAIN", f"Missing required input: {name}", None, None, trace)
        if observation.unit != spec.unit:
            trace.append(TraceStep("input_validation", "FAIL", "Unit does not match policy declaration", {"input": name, "expected": spec.unit, "actual": observation.unit}))
            return _finalize(policy, observations, evaluated_at, "ABSTAIN", f"Unit mismatch for {name}", None, None, trace)
        if observation.quality != "valid":
            trace.append(TraceStep("input_validation", "FAIL", "Observation quality is not valid", {"input": name, "quality": observation.quality}))
            return _finalize(policy, observations, evaluated_at, "ABSTAIN", f"Invalid quality for {name}", None, None, trace)
        if observation.observed_at.tzinfo is None:
            trace.append(TraceStep("input_validation", "FAIL", "Observation timestamp is not timezone-aware", {"input": name}))
            return _finalize(policy, observations, evaluated_at, "ABSTAIN", f"Naive timestamp for {name}", None, None, trace)
        age_seconds = (evaluated_at - observation.observed_at).total_seconds()
        if age_seconds < 0 or (spec.max_age_seconds is not None and age_seconds > spec.max_age_seconds):
            trace.append(TraceStep("input_validation", "FAIL", "Observation is stale or future-dated", {"input": name, "age_seconds": age_seconds, "maximum": spec.max_age_seconds}))
            return _finalize(policy, observations, evaluated_at, "ABSTAIN", f"Stale or future-dated input: {name}", None, None, trace)
        if spec.value_type == "NUMBER" and (isinstance(observation.value, bool) or not isinstance(observation.value, (int, float))):
            trace.append(TraceStep("input_validation", "FAIL", "Observation does not have the declared numeric type", {"input": name}))
            return _finalize(policy, observations, evaluated_at, "ABSTAIN", f"Invalid numeric type for {name}", None, None, trace)
        if spec.value_type == "BOOLEAN" and not isinstance(observation.value, bool):
            trace.append(TraceStep("input_validation", "FAIL", "Observation does not have the declared boolean type", {"input": name}))
            return _finalize(policy, observations, evaluated_at, "ABSTAIN", f"Invalid boolean type for {name}", None, None, trace)
        numeric_value = float(observation.value)
        if not math.isfinite(numeric_value):
            trace.append(TraceStep("input_validation", "FAIL", "Observation is not finite", {"input": name}))
            return _finalize(policy, observations, evaluated_at, "ABSTAIN", f"Non-finite input: {name}", None, None, trace)
        if (spec.minimum is not None and numeric_value < spec.minimum) or (spec.maximum is not None and numeric_value > spec.maximum):
            trace.append(TraceStep("input_validation", "FAIL", "Observation is outside the declared demonstration envelope", {"input": name, "value": observation.value, "minimum": spec.minimum, "maximum": spec.maximum}))
            return _finalize(policy, observations, evaluated_at, "ABSTAIN", f"Input outside demonstration envelope: {name}", None, None, trace)
        if not observation.source.strip():
            trace.append(TraceStep("input_validation", "FAIL", "Observation provenance source is empty", {"input": name}))
            return _finalize(policy, observations, evaluated_at, "ABSTAIN", f"Missing provenance source for {name}", None, None, trace)
        values[name] = observation.value
        trace.append(TraceStep("input_validation", "PASS", "Observation accepted", {"input": name, "value": observation.value, "unit": observation.unit, "source": observation.source, "age_seconds": age_seconds}))

    contributions: dict[str, float] = {}
    logit = policy.model.bias
    for name, weight in policy.model.weights.items():
        contribution = weight * float(values[name])
        contributions[name] = contribution
        logit += contribution
    if logit >= 0:
        score = 1.0 / (1.0 + math.exp(-logit))
    else:
        exp_logit = math.exp(logit)
        score = exp_logit / (1.0 + exp_logit)
    model_output = {
        "name": policy.model.name,
        "link": policy.model.link,
        "formula": "sigmoid(bias + sum(weight * input))",
        "bias": policy.model.bias,
        "contributions": contributions,
        "logit": logit,
        "score": score,
    }
    values[policy.model.name] = score
    trace.append(TraceStep("model", "PASS", "Computed transparent model score", model_output))

    for rule in policy.rules:
        matched, actual = _compare(rule.condition, values)
        trace.append(TraceStep("rule_condition", "MATCH" if matched else "NO_MATCH", f"Evaluated rule {rule.name}", {"rule": rule.name, "field": rule.condition.field, "actual": actual, "operator": rule.condition.operator, "expected": rule.condition.expected}))
        if not matched:
            continue
        for requirement in rule.requirements:
            passed, requirement_actual = _compare(requirement, values)
            trace.append(TraceStep("rule_requirement", "PASS" if passed else "FAIL", f"Evaluated requirement for {rule.name}", {"field": requirement.field, "actual": requirement_actual, "operator": requirement.operator, "expected": requirement.expected}))
            if not passed:
                return _finalize(policy, observations, evaluated_at, "ABSTAIN", rule.abstain_reason, model_output, None, trace)
        recommendation = rule.recommendation
        current = float(values[recommendation.field])
        proposed = current + recommendation.delta
        minimum = policy.inputs[recommendation.field].minimum
        if (minimum is not None and proposed < minimum) or proposed > recommendation.maximum:
            trace.append(TraceStep("recommendation_constraint", "FAIL", "Proposed value is outside the demonstration policy interval", {"field": recommendation.field, "current": current, "delta": recommendation.delta, "proposed": proposed, "minimum": minimum, "maximum": recommendation.maximum}))
            return _finalize(policy, observations, evaluated_at, "ABSTAIN", f"Recommendation is outside the demonstration policy interval for {recommendation.field}", model_output, None, trace)
        output = {
            "field": recommendation.field,
            "current": current,
            "delta": recommendation.delta,
            "proposed": proposed,
            "unit": policy.inputs[recommendation.field].unit,
            "authority": "advisory_only",
        }
        trace.append(TraceStep("recommendation", "PASS", "Generated advisory recommendation; no process write was performed", output))
        return _finalize(policy, observations, evaluated_at, "RECOMMENDATION", f"Rule {rule.name} matched", model_output, output, trace)

    trace.append(TraceStep("default", "PASS", "No decision rule matched", {"result": "NO_ACTION"}))
    return _finalize(policy, observations, evaluated_at, "NO_ACTION", policy.default_reason, model_output, None, trace)
