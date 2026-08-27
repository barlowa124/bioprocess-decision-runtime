from __future__ import annotations

import hashlib
import math
import shlex
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .domain import Decision, Observation, Policy
from .runtime import evaluate


class DecisionProgramError(ValueError):
    pass


@dataclass(frozen=True)
class ProposedAction:
    field: str
    delta: float
    unit: str


@dataclass(frozen=True)
class DecisionProgram:
    version: str
    policy_name: str
    policy_sha256: str
    bindings: dict[str, str]
    model_name: str
    rule_name: str | None
    expected_status: str
    proposed_action: ProposedAction | None
    source_sha256: str


@dataclass(frozen=True)
class ProgramExecution:
    accepted: bool
    reason: str
    program_sha256: str
    calculated_decision: Decision | None
    authorized_recommendation: dict[str, Any] | None
    validation_trace: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "program_sha256": self.program_sha256,
            "calculated_decision": self.calculated_decision.to_dict() if self.calculated_decision else None,
            "authorized_recommendation": self.authorized_recommendation,
            "validation_trace": list(self.validation_trace),
        }


def parse_decision_program(source: str) -> DecisionProgram:
    lines = []
    for number, raw_line in enumerate(source.splitlines(), start=1):
        tokens = shlex.split(raw_line, comments=False)
        if tokens:
            lines.append((number, tokens))
    if not lines or lines[0][1] != ["DECISION_PROGRAM", "VERSION", "1"]:
        raise DecisionProgramError("Program must begin with DECISION_PROGRAM VERSION 1")
    if lines[-1][1] != ["END"]:
        raise DecisionProgramError("Program must end with END")

    policy_name = policy_sha256 = model_name = expected_status = None
    rule_name: str | None = None
    proposed_action: ProposedAction | None = None
    bindings: dict[str, str] = {}
    seen_apply = False
    seen_propose = False

    for number, tokens in lines[1:-1]:
        command = tokens[0]
        try:
            if command == "POLICY" and len(tokens) == 4 and tokens[2] == "SHA256":
                if policy_name is not None:
                    raise DecisionProgramError("Duplicate POLICY statement")
                policy_name, policy_sha256 = tokens[1], tokens[3]
            elif command == "BIND" and len(tokens) == 4 and tokens[2] == "FROM":
                if tokens[1] in bindings:
                    raise DecisionProgramError(f"Duplicate binding for {tokens[1]!r}")
                bindings[tokens[1]] = tokens[3]
            elif command == "COMPUTE" and len(tokens) == 2:
                if model_name is not None:
                    raise DecisionProgramError("Duplicate COMPUTE statement")
                model_name = tokens[1]
            elif command == "APPLY" and len(tokens) == 2:
                if seen_apply:
                    raise DecisionProgramError("Duplicate APPLY statement")
                seen_apply = True
                rule_name = None if tokens[1] == "NONE" else tokens[1]
            elif command == "EXPECT" and len(tokens) == 2:
                if expected_status is not None:
                    raise DecisionProgramError("Duplicate EXPECT statement")
                expected_status = tokens[1]
            elif command == "PROPOSE" and tokens == ["PROPOSE", "NONE"]:
                if seen_propose:
                    raise DecisionProgramError("Duplicate PROPOSE statement")
                seen_propose = True
            elif command == "PROPOSE" and len(tokens) == 5 and tokens[2] == "DELTA":
                if seen_propose:
                    raise DecisionProgramError("Duplicate PROPOSE statement")
                delta = float(tokens[3])
                if not math.isfinite(delta):
                    raise DecisionProgramError("Proposed delta must be finite")
                seen_propose = True
                proposed_action = ProposedAction(tokens[1], delta, tokens[4])
            else:
                raise DecisionProgramError(f"Unknown or malformed statement {command!r}")
        except (IndexError, ValueError) as exc:
            if isinstance(exc, DecisionProgramError):
                raise
            raise DecisionProgramError(f"Line {number}: {exc}") from exc

    if not all((policy_name, policy_sha256, model_name, expected_status)) or not seen_apply or not seen_propose:
        raise DecisionProgramError("Program is missing POLICY, COMPUTE, APPLY, EXPECT, or PROPOSE")
    if expected_status not in {"RECOMMENDATION", "NO_ACTION", "ABSTAIN"}:
        raise DecisionProgramError(f"Unsupported expected status {expected_status!r}")
    if len(policy_sha256) != 64 or any(character not in "0123456789abcdef" for character in policy_sha256.lower()):
        raise DecisionProgramError("POLICY SHA256 must contain 64 hexadecimal characters")
    return DecisionProgram(
        version="1",
        policy_name=policy_name,
        policy_sha256=policy_sha256.lower(),
        bindings=bindings,
        model_name=model_name,
        rule_name=rule_name,
        expected_status=expected_status,
        proposed_action=proposed_action,
        source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
    )


def _matched_rule(decision: Decision) -> str | None:
    for step in decision.trace:
        if step.stage == "rule_condition" and step.result == "MATCH":
            rule = step.values.get("rule")
            return str(rule) if rule is not None else None
    return None


def execute_decision_program(
    program: DecisionProgram,
    policy: Policy,
    observations: dict[str, Observation],
    evaluated_at: datetime,
) -> ProgramExecution:
    trace: list[dict[str, Any]] = []

    def reject(reason: str, decision: Decision | None = None) -> ProgramExecution:
        trace.append({"stage": "program_validation", "result": "REJECT", "detail": reason})
        return ProgramExecution(False, reason, program.source_sha256, decision, None, tuple(trace))

    if program.policy_name != policy.name or program.policy_sha256 != policy.source_sha256:
        return reject("Program policy identity does not match the approved policy")
    trace.append({"stage": "policy_binding", "result": "PASS", "detail": "Approved policy name and source hash match"})
    if program.model_name != policy.model.name:
        return reject("Program COMPUTE target does not match the approved model")
    if set(program.bindings) != set(policy.inputs):
        return reject("Program must bind every and only declared policy input")
    if any(input_name != observation_name for input_name, observation_name in program.bindings.items()):
        return reject("This prototype permits only identity evidence bindings")
    if set(program.bindings.values()) != set(observations):
        return reject("Program evidence bindings do not match the supplied observations")
    trace.append({"stage": "evidence_binding", "result": "PASS", "detail": "Every declared input is bound exactly once to its named observation"})

    bound_observations = {input_name: observations[observation_name] for input_name, observation_name in program.bindings.items()}
    decision = evaluate(policy, bound_observations, evaluated_at)
    actual_rule = _matched_rule(decision)
    if program.rule_name != actual_rule:
        return reject(f"Program selected rule {program.rule_name!r}, but execution selected {actual_rule!r}", decision)
    if program.expected_status != decision.status:
        return reject(f"Program expected {program.expected_status}, but execution produced {decision.status}", decision)
    trace.append({"stage": "execution_claim", "result": "PASS", "detail": "Selected rule and expected status match independent policy execution"})

    recommendation = decision.recommendation
    if recommendation is None and program.proposed_action is not None:
        return reject("Program proposed an action when independent execution produced none", decision)
    if recommendation is not None:
        proposal = program.proposed_action
        if proposal is None:
            return reject("Program omitted the independently calculated recommendation", decision)
        if proposal.field != recommendation["field"] or proposal.unit != recommendation["unit"] or proposal.delta != recommendation["delta"]:
            return reject("Program proposal does not match the independently calculated recommendation", decision)
    trace.append({"stage": "proposal_check", "result": "PASS", "detail": "Program proposal matches the independently calculated advisory result"})
    authorized = recommendation if decision.status == "RECOMMENDATION" else None
    return ProgramExecution(True, "Program matches approved policy execution", program.source_sha256, decision, authorized, tuple(trace))


def render_program_template(policy: Policy, observations: dict[str, Observation]) -> str:
    bindings = "\n".join(f"BIND {name} FROM {name}" for name in sorted(observations))
    rules = ", ".join(rule.name for rule in policy.rules)
    return (
        "DECISION_PROGRAM VERSION 1\n"
        f"POLICY {policy.name} SHA256 {policy.source_sha256}\n"
        f"{bindings}\n"
        f"COMPUTE {policy.model.name}\n"
        f"APPLY <one of: {rules}, NONE>\n"
        "EXPECT <RECOMMENDATION, NO_ACTION, or ABSTAIN>\n"
        "PROPOSE <field DELTA number unit, or NONE>\n"
        "END"
    )
