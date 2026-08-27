from __future__ import annotations

import unittest
from pathlib import Path

from bioprocess_runtime.decision_program import DecisionProgramError, execute_decision_program, parse_decision_program
from bioprocess_runtime.gemma_api import GemmaAPIConfig, build_program_grammar, build_program_prompt
from bioprocess_runtime.policy import load_policy
from bioprocess_runtime.simulator import scenarios


ROOT = Path(__file__).resolve().parent.parent
POLICY = load_policy(ROOT / "policies" / "oxygen_advisory.bpr")


def program_source(status: str, rule: str, proposal: str, policy_hash: str | None = None) -> str:
    return f"""DECISION_PROGRAM VERSION 1
POLICY {POLICY.name} SHA256 {policy_hash or POLICY.source_sha256}
BIND agitation_rpm FROM agitation_rpm
BIND dissolved_oxygen_pct FROM dissolved_oxygen_pct
BIND dissolved_oxygen_slope FROM dissolved_oxygen_slope
BIND sensor_agreement FROM sensor_agreement
COMPUTE {POLICY.model.name}
APPLY {rule}
EXPECT {status}
PROPOSE {proposal}
END
"""


class DecisionProgramTests(unittest.TestCase):
    def test_accepts_matching_advisory_program(self) -> None:
        scenario = scenarios()["low_oxygen"]
        program = parse_decision_program(program_source("RECOMMENDATION", "low_oxygen_advisory", "agitation_rpm DELTA 5 rpm"))
        result = execute_decision_program(program, POLICY, scenario.observations, scenario.evaluated_at)
        self.assertTrue(result.accepted)
        self.assertEqual(result.authorized_recommendation["proposed"], 95.0)
        self.assertEqual(result.authorized_recommendation["authority"], "advisory_only")

    def test_accepts_no_action_program(self) -> None:
        scenario = scenarios()["normal"]
        program = parse_decision_program(program_source("NO_ACTION", "NONE", "NONE"))
        result = execute_decision_program(program, POLICY, scenario.observations, scenario.evaluated_at)
        self.assertTrue(result.accepted)
        self.assertIsNone(result.authorized_recommendation)

    def test_accepts_abstention_for_failed_requirement(self) -> None:
        scenario = scenarios()["sensor_disagreement"]
        program = parse_decision_program(program_source("ABSTAIN", "low_oxygen_advisory", "NONE"))
        result = execute_decision_program(program, POLICY, scenario.observations, scenario.evaluated_at)
        self.assertTrue(result.accepted)
        self.assertIsNone(result.authorized_recommendation)

    def test_rejects_wrong_policy_hash(self) -> None:
        scenario = scenarios()["low_oxygen"]
        program = parse_decision_program(program_source("RECOMMENDATION", "low_oxygen_advisory", "agitation_rpm DELTA 5 rpm", "0" * 64))
        result = execute_decision_program(program, POLICY, scenario.observations, scenario.evaluated_at)
        self.assertFalse(result.accepted)
        self.assertIn("policy identity", result.reason)

    def test_rejects_invented_rule(self) -> None:
        scenario = scenarios()["low_oxygen"]
        program = parse_decision_program(program_source("RECOMMENDATION", "invented_rule", "agitation_rpm DELTA 5 rpm"))
        result = execute_decision_program(program, POLICY, scenario.observations, scenario.evaluated_at)
        self.assertFalse(result.accepted)
        self.assertIn("selected rule", result.reason)

    def test_rejects_wrong_proposal(self) -> None:
        scenario = scenarios()["low_oxygen"]
        program = parse_decision_program(program_source("RECOMMENDATION", "low_oxygen_advisory", "agitation_rpm DELTA 50 rpm"))
        result = execute_decision_program(program, POLICY, scenario.observations, scenario.evaluated_at)
        self.assertFalse(result.accepted)
        self.assertIn("proposal", result.reason)

    def test_rejects_free_text(self) -> None:
        with self.assertRaises(DecisionProgramError):
            parse_decision_program("I recommend increasing agitation.")

    def test_prompt_contains_only_approved_contract_and_evidence(self) -> None:
        scenario = scenarios()["low_oxygen"]
        prompt = build_program_prompt(POLICY, scenario.observations, scenario.evaluated_at)
        self.assertIn(POLICY.source_sha256, prompt)
        self.assertIn("The policy is authoritative", prompt)
        self.assertIn("PROPOSE <field DELTA number unit, or NONE>", prompt)

    def test_grammar_constrains_policy_identity_and_vocabulary(self) -> None:
        scenario = scenarios()["low_oxygen"]
        grammar = build_program_grammar(POLICY, scenario.observations)
        self.assertIn(POLICY.source_sha256, grammar)
        self.assertIn("APPLY low_oxygen_advisory", grammar)
        self.assertIn("PROPOSE agitation_rpm DELTA", grammar)
        self.assertNotIn("invented_rule", grammar)

    def test_api_client_rejects_remote_endpoint(self) -> None:
        with self.assertRaises(ValueError):
            GemmaAPIConfig(base_url="https://example.com/v1")


if __name__ == "__main__":
    unittest.main()
