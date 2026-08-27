from __future__ import annotations

import io
import math
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

from bioprocess_runtime.audit import AuditIntegrityError, append_decision, read_records, verify_file
from bioprocess_runtime.cli import command_replay
from bioprocess_runtime.domain import Observation
from bioprocess_runtime.policy import PolicySyntaxError, load_policy, parse_policy_text
from bioprocess_runtime.runtime import evaluate
from bioprocess_runtime.serialization import canonical_json
from bioprocess_runtime.simulator import scenarios
from bioprocess_runtime.training import train_transparent_policy


ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = ROOT / "policies" / "oxygen_advisory.bpr"


class PolicyTests(unittest.TestCase):
    def test_loads_bounded_advisory_policy(self) -> None:
        policy = load_policy(POLICY_PATH)
        self.assertEqual(policy.name, "oxygen_advisory")
        self.assertEqual(policy.mode, "ADVISORY")
        self.assertEqual(policy.model.link, "LOGISTIC")
        self.assertEqual(len(policy.rules), 1)

    def test_rejects_unknown_rule_reference(self) -> None:
        source = POLICY_PATH.read_text(encoding="utf-8").replace("WHEN oxygen_risk", "WHEN unknown_score")
        with self.assertRaisesRegex(PolicySyntaxError, "unknown values"):
            parse_policy_text(source)

    def test_rejects_non_advisory_mode(self) -> None:
        source = POLICY_PATH.read_text(encoding="utf-8").replace("MODE ADVISORY", "MODE AUTONOMOUS")
        with self.assertRaisesRegex(PolicySyntaxError, "ADVISORY mode only"):
            parse_policy_text(source)

    def test_rejects_non_finite_model_values(self) -> None:
        source = POLICY_PATH.read_text(encoding="utf-8").replace("BIAS 5.0", "BIAS nan")
        with self.assertRaisesRegex(PolicySyntaxError, "must be finite"):
            parse_policy_text(source)

    def test_rejects_invalid_boolean_comparison(self) -> None:
        source = POLICY_PATH.read_text(encoding="utf-8").replace("sensor_agreement == true", "sensor_agreement > false")
        with self.assertRaisesRegex(PolicySyntaxError, "Boolean comparison"):
            parse_policy_text(source)


class RuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = load_policy(POLICY_PATH)
        self.scenarios = scenarios()

    def test_scenario_expected_outcomes(self) -> None:
        for scenario in self.scenarios.values():
            with self.subTest(scenario=scenario.name):
                decision = evaluate(self.policy, scenario.observations, scenario.evaluated_at)
                self.assertEqual(decision.status, scenario.expected_status)

    def test_model_trace_is_exact_and_reproducible(self) -> None:
        scenario = self.scenarios["low_oxygen"]
        first = evaluate(self.policy, scenario.observations, scenario.evaluated_at)
        second = evaluate(self.policy, scenario.observations, scenario.evaluated_at)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertAlmostEqual(first.model_output["contributions"]["dissolved_oxygen_pct"], -3.6)
        self.assertAlmostEqual(first.model_output["contributions"]["dissolved_oxygen_slope"], 1.5)

    def test_never_grants_execution_authority(self) -> None:
        scenario = self.scenarios["low_oxygen"]
        decision = evaluate(self.policy, scenario.observations, scenario.evaluated_at)
        self.assertEqual(decision.status, "RECOMMENDATION")
        self.assertEqual(decision.recommendation["authority"], "advisory_only")
        self.assertTrue(decision.human_review_required)

    def test_missing_provenance_causes_abstention(self) -> None:
        scenario = self.scenarios["normal"]
        observations = dict(scenario.observations)
        observations["agitation_rpm"] = replace(observations["agitation_rpm"], source="")
        decision = evaluate(self.policy, observations, scenario.evaluated_at)
        self.assertEqual(decision.status, "ABSTAIN")
        self.assertIn("provenance", decision.reason.lower())

    def test_future_dated_observation_causes_abstention(self) -> None:
        scenario = self.scenarios["normal"]
        observations = dict(scenario.observations)
        future = scenario.evaluated_at + timedelta(seconds=1)
        observations["agitation_rpm"] = replace(observations["agitation_rpm"], observed_at=future)
        decision = evaluate(self.policy, observations, scenario.evaluated_at)
        self.assertEqual(decision.status, "ABSTAIN")

    def test_wrong_boolean_type_causes_abstention(self) -> None:
        scenario = self.scenarios["normal"]
        observations = dict(scenario.observations)
        original = observations["sensor_agreement"]
        observations["sensor_agreement"] = Observation(1.0, original.unit, original.observed_at, original.source)
        decision = evaluate(self.policy, observations, scenario.evaluated_at)
        self.assertEqual(decision.status, "ABSTAIN")
        self.assertIn("boolean", decision.reason.lower())

    def test_non_finite_input_causes_abstention(self) -> None:
        scenario = self.scenarios["normal"]
        observations = dict(scenario.observations)
        observations["agitation_rpm"] = replace(observations["agitation_rpm"], value=math.nan)
        decision = evaluate(self.policy, observations, scenario.evaluated_at)
        self.assertEqual(decision.status, "ABSTAIN")
        self.assertIn("non-finite", decision.reason.lower())

    def test_unexpected_input_causes_abstention(self) -> None:
        scenario = self.scenarios["normal"]
        observations = dict(scenario.observations)
        observations["undeclared"] = Observation(1.0, "none", scenario.evaluated_at, "synthetic.extra")
        decision = evaluate(self.policy, observations, scenario.evaluated_at)
        self.assertEqual(decision.status, "ABSTAIN")
        self.assertIn("unexpected", decision.reason.lower())

    def test_recommendation_below_declared_input_minimum_causes_abstention(self) -> None:
        scenario = self.scenarios["low_oxygen"]
        rule = self.policy.rules[0]
        recommendation = replace(rule.recommendation, delta=-100.0)
        policy = replace(self.policy, rules=(replace(rule, recommendation=recommendation),))
        decision = evaluate(policy, scenario.observations, scenario.evaluated_at)
        self.assertEqual(decision.status, "ABSTAIN")
        self.assertIn("outside", decision.reason)

    def test_decision_ids_are_full_content_hashes(self) -> None:
        normal = self.scenarios["normal"]
        low = self.scenarios["low_oxygen"]
        normal_decision = evaluate(self.policy, normal.observations, normal.evaluated_at)
        repeated = evaluate(self.policy, normal.observations, normal.evaluated_at)
        low_decision = evaluate(self.policy, low.observations, low.evaluated_at)
        self.assertEqual(len(normal_decision.decision_id), 64)
        self.assertEqual(normal_decision.decision_id, repeated.decision_id)
        self.assertNotEqual(normal_decision.decision_id, low_decision.decision_id)


class TrainingTests(unittest.TestCase):
    def test_learned_model_is_translated_into_executable_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "learned.bpr"
            result = train_transparent_policy(POLICY_PATH, output, samples=500, seed=23)
            learned_policy = load_policy(output)
            self.assertEqual(learned_policy.version, "synthetic-1.0.0")
            self.assertEqual(set(learned_policy.model.weights), {"dissolved_oxygen_pct", "dissolved_oxygen_slope"})
            self.assertEqual(learned_policy.model.bias, result.report["interpretable_model"]["bias"])
            self.assertEqual(learned_policy.model.weights, result.report["interpretable_model"]["weights_in_declared_input_units"])
            self.assertLess(result.report["evaluation"]["maximum_policy_logit_translation_error"], 1e-10)
            self.assertIn("Synthetic software demonstration", result.report["scope"])


class AuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = load_policy(POLICY_PATH)
        self.scenario = scenarios()["low_oxygen"]
        self.decision = evaluate(self.policy, self.scenario.observations, self.scenario.evaluated_at)

    def test_canonical_serialization_rejects_non_finite_values(self) -> None:
        with self.assertRaises(ValueError):
            canonical_json({"value": math.nan})

    def test_hash_chain_verifies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.ndjson"
            append_decision(path, self.decision)
            append_decision(path, self.decision)
            self.assertEqual(verify_file(path), 2)

    def test_payload_tampering_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.ndjson"
            append_decision(path, self.decision)
            records = read_records(path)
            records[0]["payload"]["status"] = "NO_ACTION"
            path.write_text(canonical_json(records[0]) + "\n", encoding="utf-8")
            with self.assertRaises(AuditIntegrityError):
                verify_file(path)

    def test_refuses_to_append_to_damaged_chain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.ndjson"
            append_decision(path, self.decision)
            text = path.read_text(encoding="utf-8").replace("RECOMMENDATION", "NO_ACTION")
            path.write_text(text, encoding="utf-8")
            with self.assertRaises(AuditIntegrityError):
                append_decision(path, self.decision)

    def test_cli_replay_verifies_an_unchanged_decision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.ndjson"
            append_decision(path, self.decision)
            output = io.StringIO()
            with redirect_stdout(output):
                result = command_replay(Namespace(audit=path, decision_id=self.decision.decision_id, policy=POLICY_PATH))
            self.assertEqual(result, 0)
            self.assertIn("Replay verified", output.getvalue())

    def test_cli_replay_rejects_a_different_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            audit_path = directory_path / "audit.ndjson"
            changed_policy = directory_path / "changed.bpr"
            append_decision(audit_path, self.decision)
            changed_policy.write_text(POLICY_PATH.read_text(encoding="utf-8").replace("VERSION 1.0.0", "VERSION 1.0.1"), encoding="utf-8")
            error = io.StringIO()
            with redirect_stderr(error):
                result = command_replay(Namespace(audit=audit_path, decision_id=self.decision.decision_id, policy=changed_policy))
            self.assertEqual(result, 1)
            self.assertIn("policy does not match", error.getvalue())

    def test_cli_replay_rejects_a_damaged_audit_chain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.ndjson"
            append_decision(path, self.decision)
            path.write_text(path.read_text(encoding="utf-8").replace("RECOMMENDATION", "NO_ACTION"), encoding="utf-8")
            error = io.StringIO()
            with redirect_stderr(error):
                result = command_replay(Namespace(audit=path, decision_id=self.decision.decision_id, policy=POLICY_PATH))
            self.assertEqual(result, 1)
            self.assertIn("audit verification failed", error.getvalue())


if __name__ == "__main__":
    unittest.main()
