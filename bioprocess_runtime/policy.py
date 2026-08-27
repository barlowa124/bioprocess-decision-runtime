from __future__ import annotations

import hashlib
import math
import shlex
from pathlib import Path

from .domain import Comparison, InputSpec, ModelSpec, Policy, Recommendation, RuleSpec


class PolicySyntaxError(ValueError):
    pass


def _scalar(token: str) -> float | bool:
    if token.lower() in {"true", "false"}:
        return token.lower() == "true"
    try:
        return float(token)
    except ValueError as exc:
        raise PolicySyntaxError(f"Expected a number or boolean, received {token!r}") from exc


def _comparison(tokens: list[str]) -> Comparison:
    if len(tokens) != 3 or tokens[1] not in {"<", "<=", "==", ">=", ">"}:
        raise PolicySyntaxError(f"Invalid comparison: {' '.join(tokens)}")
    return Comparison(tokens[0], tokens[1], _scalar(tokens[2]))


def parse_policy_text(source: str) -> Policy:
    source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    lines: list[tuple[int, list[str]]] = []
    for number, raw_line in enumerate(source.splitlines(), start=1):
        tokens = shlex.split(raw_line, comments=True)
        if tokens:
            lines.append((number, tokens))

    name = version = intended_use = mode = default_reason = None
    inputs: dict[str, InputSpec] = {}
    model_name = model_link = None
    model_bias: float | None = None
    model_weights: dict[str, float] = {}
    rules: list[RuleSpec] = []
    current_rule: dict[str, object] | None = None
    in_model = False

    for number, tokens in lines:
        command = tokens[0]
        try:
            if command == "POLICY" and len(tokens) == 4 and tokens[2] == "VERSION":
                if name is not None:
                    raise PolicySyntaxError("Duplicate POLICY declaration")
                name, version = tokens[1], tokens[3]
            elif command == "INTENDED_USE" and len(tokens) == 2:
                if intended_use is not None:
                    raise PolicySyntaxError("Duplicate INTENDED_USE declaration")
                intended_use = tokens[1]
            elif command == "MODE" and len(tokens) == 2:
                if mode is not None:
                    raise PolicySyntaxError("Duplicate MODE declaration")
                mode = tokens[1]
            elif command == "INPUT":
                if len(tokens) != 11 or tokens[3] != "UNIT" or tokens[5] != "MIN" or tokens[7] != "MAX" or tokens[9] != "MAX_AGE_SECONDS":
                    raise PolicySyntaxError("INPUT syntax is INPUT <name> <NUMBER|BOOLEAN> UNIT <unit> MIN <n> MAX <n> MAX_AGE_SECONDS <n>")
                spec = InputSpec(
                    name=tokens[1],
                    value_type=tokens[2],
                    unit=tokens[4],
                    minimum=float(tokens[6]),
                    maximum=float(tokens[8]),
                    max_age_seconds=int(tokens[10]),
                )
                if spec.name in inputs:
                    raise PolicySyntaxError(f"Duplicate input {spec.name!r}")
                inputs[spec.name] = spec
            elif command == "MODEL" and len(tokens) == 3:
                if in_model or model_name is not None:
                    raise PolicySyntaxError("Exactly one MODEL block is allowed")
                model_name, model_link, in_model = tokens[1], tokens[2], True
            elif command == "BIAS" and in_model and len(tokens) == 2:
                if model_bias is not None:
                    raise PolicySyntaxError("Duplicate BIAS declaration")
                model_bias = float(tokens[1])
            elif command == "WEIGHT" and in_model and len(tokens) == 3:
                if tokens[1] in model_weights:
                    raise PolicySyntaxError(f"Duplicate weight for {tokens[1]!r}")
                model_weights[tokens[1]] = float(tokens[2])
            elif command == "END_MODEL" and len(tokens) == 1 and in_model:
                in_model = False
            elif command == "RULE" and len(tokens) == 2:
                if current_rule is not None:
                    raise PolicySyntaxError("Nested RULE blocks are not allowed")
                current_rule = {"name": tokens[1], "requirements": []}
            elif command == "WHEN" and current_rule is not None:
                current_rule["condition"] = _comparison(tokens[1:])
            elif command == "REQUIRE" and current_rule is not None:
                requirements = current_rule["requirements"]
                if not isinstance(requirements, list):
                    raise PolicySyntaxError("Rule requirements have invalid parser state")
                requirements.append(_comparison(tokens[1:]))
            elif command == "RECOMMEND" and current_rule is not None:
                if len(tokens) != 7 or tokens[2] != "DELTA" or tokens[4] != "MAX":
                    raise PolicySyntaxError("RECOMMEND syntax is RECOMMEND <field> DELTA <n> MAX <n> <unit>")
                current_rule["recommendation"] = Recommendation(tokens[1], float(tokens[3]), float(tokens[5]))
                current_rule["recommendation_unit"] = tokens[6]
            elif command == "ELSE_ABSTAIN" and current_rule is not None and len(tokens) == 2:
                current_rule["abstain_reason"] = tokens[1]
            elif command == "END_RULE" and current_rule is not None:
                required = {"name", "condition", "requirements", "recommendation", "abstain_reason"}
                missing = required.difference(current_rule)
                if missing:
                    raise PolicySyntaxError(f"Incomplete rule; missing {', '.join(sorted(missing))}")
                recommendation = current_rule["recommendation"]
                if not isinstance(recommendation, Recommendation):
                    raise PolicySyntaxError("Rule recommendation has invalid parser state")
                recommendation_unit = current_rule["recommendation_unit"]
                if recommendation.field not in inputs or inputs[recommendation.field].unit != recommendation_unit:
                    raise PolicySyntaxError("Recommendation field or unit does not match a declared input")
                rules.append(
                    RuleSpec(
                        name=str(current_rule["name"]),
                        condition=current_rule["condition"],
                        requirements=tuple(current_rule["requirements"]),
                        recommendation=recommendation,
                        abstain_reason=str(current_rule["abstain_reason"]),
                    )
                )
                current_rule = None
            elif command == "DEFAULT" and len(tokens) == 3 and tokens[1] == "NO_ACTION":
                if default_reason is not None:
                    raise PolicySyntaxError("Duplicate DEFAULT declaration")
                default_reason = tokens[2]
            else:
                raise PolicySyntaxError(f"Unknown or misplaced statement {command!r}")
        except (ValueError, IndexError) as exc:
            raise PolicySyntaxError(f"Line {number}: {exc}") from exc

    if in_model or current_rule is not None:
        raise PolicySyntaxError("Unclosed MODEL or RULE block")
    required_headers = {"name": name, "version": version, "intended use": intended_use, "mode": mode, "default": default_reason}
    missing_headers = [key for key, value in required_headers.items() if value is None]
    if missing_headers:
        raise PolicySyntaxError(f"Missing required fields: {', '.join(missing_headers)}")
    if model_name is None or model_link is None or model_bias is None:
        raise PolicySyntaxError("Exactly one complete MODEL is required")
    if model_link != "LOGISTIC":
        raise PolicySyntaxError("Only LOGISTIC models are supported")
    if not inputs or not model_weights or not rules:
        raise PolicySyntaxError("At least one input, model weight, and rule are required")
    unsupported_types = {spec.value_type for spec in inputs.values()}.difference({"NUMBER", "BOOLEAN"})
    if unsupported_types:
        raise PolicySyntaxError(f"Unsupported input types: {', '.join(sorted(unsupported_types))}")
    for spec in inputs.values():
        if spec.minimum is None or spec.maximum is None or not math.isfinite(spec.minimum) or not math.isfinite(spec.maximum):
            raise PolicySyntaxError(f"Input {spec.name!r} requires finite bounds")
        if spec.minimum > spec.maximum or spec.max_age_seconds is None or spec.max_age_seconds <= 0:
            raise PolicySyntaxError(f"Input {spec.name!r} has an invalid range or maximum age")
        if spec.value_type == "BOOLEAN" and (spec.unit != "boolean" or spec.minimum != 0 or spec.maximum != 1):
            raise PolicySyntaxError(f"Boolean input {spec.name!r} must use unit boolean and range 0..1")
    if not math.isfinite(model_bias) or any(not math.isfinite(weight) for weight in model_weights.values()):
        raise PolicySyntaxError("Model bias and weights must be finite")
    unknown_weights = set(model_weights).difference(inputs)
    if unknown_weights:
        raise PolicySyntaxError(f"Model weights reference unknown inputs: {', '.join(sorted(unknown_weights))}")
    boolean_weights = {name for name in model_weights if inputs[name].value_type == "BOOLEAN"}
    if boolean_weights:
        raise PolicySyntaxError(f"Model weights must reference numeric inputs: {', '.join(sorted(boolean_weights))}")
    available_values = set(inputs) | {model_name}
    for rule in rules:
        comparisons = (rule.condition, *rule.requirements)
        referenced = {comparison.field for comparison in comparisons}
        unknown_references = referenced.difference(available_values)
        if unknown_references:
            raise PolicySyntaxError(f"Rule {rule.name!r} references unknown values: {', '.join(sorted(unknown_references))}")
        for comparison in comparisons:
            value_type = inputs[comparison.field].value_type if comparison.field in inputs else "NUMBER"
            if value_type == "BOOLEAN" and (comparison.operator != "==" or not isinstance(comparison.expected, bool)):
                raise PolicySyntaxError(f"Boolean comparison for {comparison.field!r} must use == true or == false")
            if value_type == "NUMBER" and (isinstance(comparison.expected, bool) or not math.isfinite(comparison.expected)):
                raise PolicySyntaxError(f"Numeric comparison for {comparison.field!r} requires a finite number")
        if inputs[rule.recommendation.field].value_type != "NUMBER":
            raise PolicySyntaxError(f"Recommendation field {rule.recommendation.field!r} must be numeric")
        if not math.isfinite(rule.recommendation.delta) or not math.isfinite(rule.recommendation.maximum):
            raise PolicySyntaxError(f"Recommendation in rule {rule.name!r} must be finite")
    if mode != "ADVISORY":
        raise PolicySyntaxError("This prototype supports ADVISORY mode only")

    return Policy(
        name=name,
        version=version,
        intended_use=intended_use,
        mode=mode,
        inputs=inputs,
        model=ModelSpec(model_name, model_link, model_bias, model_weights),
        rules=tuple(rules),
        default_reason=default_reason,
        source_sha256=source_sha256,
    )


def load_policy(path: str | Path) -> Policy:
    return parse_policy_text(Path(path).read_text(encoding="utf-8"))
