"""Robustness battery: policy-DSL rejects, runtime ABSTAIN boundaries,
advisory-only invariant, and audit-chain tamper detection."""

import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile

from bioprocess_runtime.audit import (AuditIntegrityError, append_decision,
                                    read_records, verify_records)
from bioprocess_runtime.decision_program import (DecisionProgramError,
                                                 parse_decision_program)
from bioprocess_runtime.domain import Observation
from bioprocess_runtime.policy import PolicySyntaxError, load_policy, parse_policy_text
from bioprocess_runtime.runtime import evaluate

POLICY_PATH = "policies/oxygen_advisory.bpr"
NOW = datetime(2025, 1, 1, tzinfo=timezone.utc)


def obs(value, unit, when=NOW, quality="valid", source="sim"):
    return Observation(value=value, unit=unit, observed_at=when,
                       source=source, quality=quality)


def _valid_inputs(**over):
    d = {
        "dissolved_oxygen_pct": obs(95.0, "percent"),
        "dissolved_oxygen_slope": obs(0.0, "percent_per_minute"),
        "agitation_rpm": obs(50.0, "rpm"),
        "sensor_agreement": obs(True, "boolean"),
    }
    d.update(over)
    return d


class TestPolicySyntaxRejects(unittest.TestCase):
    def _minimal(self, body=""):
        return ("POLICY p VERSION 1.0\n"
                'INTENDED_USE "demo"\nMODE ADVISORY\n'
                "INPUT x NUMBER UNIT u MIN 0 MAX 1 MAX_AGE_SECONDS 10\n"
                "MODEL m LOGISTIC\nBIAS 0.0\nWEIGHT x 1.0\nEND_MODEL\n"
                "RULE r\nWHEN m > 0.5\nRECOMMEND x DELTA 1 MAX 1 u\n"
                'ELSE_ABSTAIN "no"\nEND_RULE\n'
                'DEFAULT NO_ACTION "default"\n' + body)

    def test_empty_and_garbage(self):
        for bad in ["", "\n\n", "POLICY", "HELLO WORLD"]:
            with self.subTest(bad=bad), self.assertRaises(PolicySyntaxError):
                parse_policy_text(bad)

    def test_missing_headers(self):
        with self.assertRaises(PolicySyntaxError):
            parse_policy_text(
                "INPUT x NUMBER UNIT u MIN 0 MAX 1 MAX_AGE_SECONDS 10\n")

    def test_duplicate_policy_header(self):
        with self.assertRaises(PolicySyntaxError):
            parse_policy_text("POLICY a VERSION 1.0\nPOLICY b VERSION 1.0\n")

    def test_unclosed_model_block(self):
        with self.assertRaises(PolicySyntaxError):
            parse_policy_text(self._minimal().replace("END_MODEL\n", ""))

    def test_unknown_statement_rejected(self):
        with self.assertRaises(PolicySyntaxError):
            parse_policy_text(self._minimal("EXECUTE rm -rf /\n"))

    def test_incomplete_rule_missing_fields(self):
        src = ("POLICY p VERSION 1.0\nINTENDED_USE \"d\"\nMODE ADVISORY\n"
               "INPUT x NUMBER UNIT u MIN 0 MAX 1 MAX_AGE_SECONDS 10\n"
               "MODEL m LOGISTIC\nBIAS 0.0\nWEIGHT x 1.0\nEND_MODEL\n"
               "RULE r\nWHEN m > 0.5\nEND_RULE\n"
               "DEFAULT NO_ACTION \"d\"\n")
        with self.assertRaises(PolicySyntaxError):
            parse_policy_text(src)

    def test_min_gt_max_rejected(self):
        with self.assertRaises(PolicySyntaxError):
            parse_policy_text(
                self._minimal().replace("MIN 0 MAX 1", "MIN 5 MAX 1"))

    def test_bad_comparison_operator(self):
        with self.assertRaises(PolicySyntaxError):
            parse_policy_text(
                self._minimal().replace("WHEN m > 0.5", "WHEN m ~= 0.5"))

    def test_valid_policy_parses(self):
        p = parse_policy_text(self._minimal())
        self.assertEqual(p.name, "p")
        self.assertEqual(len(p.rules), 1)


class TestRuntimeAbstainBoundaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = load_policy(POLICY_PATH)

    def test_naive_evaluated_at_raises(self):
        with self.assertRaises(ValueError):
            evaluate(self.policy, _valid_inputs(),
                     evaluated_at=datetime(2025, 1, 1))

    def test_valid_inputs_no_abstain(self):
        d = evaluate(self.policy, _valid_inputs(), evaluated_at=NOW)
        self.assertNotEqual(d.status, "ABSTAIN")

    def test_each_boundary_abstains(self):
        cases = {
            "missing": lambda d: d.pop("sensor_agreement"),
            "wrong-unit": lambda d: d.update(
                dissolved_oxygen_pct=obs(95.0, "celsius")),
            "bad-quality": lambda d: d.update(
                sensor_agreement=obs(True, "boolean", quality="suspect")),
            "nan": lambda d: d.update(
                dissolved_oxygen_pct=obs(float("nan"), "percent")),
            "out-of-range": lambda d: d.update(
                dissolved_oxygen_pct=obs(150.0, "percent")),
            "no-source": lambda d: d.update(
                dissolved_oxygen_pct=obs(95.0, "percent", source=" ")),
            "stale": lambda d: d.update(dissolved_oxygen_pct=obs(
                95.0, "percent", when=NOW - timedelta(seconds=60))),
            "future": lambda d: d.update(dissolved_oxygen_pct=obs(
                95.0, "percent", when=NOW + timedelta(seconds=60))),
            "unexpected": lambda d: d.update(
                rogue_input=obs(1.0, "percent")),
        }
        for name, mutate in cases.items():
            with self.subTest(case=name):
                inputs = _valid_inputs()
                mutate(inputs)
                d = evaluate(self.policy, inputs, evaluated_at=NOW)
                self.assertEqual(d.status, "ABSTAIN")
                self.assertTrue(d.reason)
                self.assertTrue(d.human_review_required)

    def test_advisory_only_invariant(self):
        # a recommendation must always require human review
        inputs = _valid_inputs(
            dissolved_oxygen_pct=obs(0.0, "percent"),
            dissolved_oxygen_slope=obs(-5.0, "percent_per_minute"))
        d = evaluate(self.policy, inputs, evaluated_at=NOW)
        self.assertIn(d.status, {"RECOMMENDATION", "ABSTAIN"})
        self.assertTrue(d.human_review_required)
        if d.status == "RECOMMENDATION":
            self.assertIsNotNone(d.recommendation)

    def test_rule_requirement_failure_abstains(self):
        inputs = _valid_inputs(
            dissolved_oxygen_pct=obs(0.0, "percent"),
            dissolved_oxygen_slope=obs(-5.0, "percent_per_minute"),
            sensor_agreement=obs(False, "boolean"))
        d = evaluate(self.policy, inputs, evaluated_at=NOW)
        self.assertEqual(d.status, "ABSTAIN")

    def test_decision_deterministic(self):
        a = evaluate(self.policy, _valid_inputs(), evaluated_at=NOW)
        b = evaluate(self.policy, _valid_inputs(), evaluated_at=NOW)
        self.assertEqual(a.decision_id, b.decision_id)


class TestAuditChain(unittest.TestCase):
    def setUp(self):
        self.policy = load_policy(POLICY_PATH)
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "audit.log"

    def tearDown(self):
        self.tmp.cleanup()

    def test_tamper_detected(self):
        d1 = evaluate(self.policy, _valid_inputs(), evaluated_at=NOW)
        append_decision(self.path, d1)
        append_decision(self.path, evaluate(self.policy, _valid_inputs(),
                                            evaluated_at=NOW))
        self.assertEqual(verify_records(read_records(self.path)), 2)
        recs = read_records(self.path)
        recs[0]["payload"]["status"] = "FORGED"
        with self.assertRaises(AuditIntegrityError):
            verify_records(recs)

    def test_broken_link_detected(self):
        d = evaluate(self.policy, _valid_inputs(), evaluated_at=NOW)
        append_decision(self.path, d)
        append_decision(self.path, d)
        recs = read_records(self.path)
        recs[1]["previous_hash"] = "0" * 64
        with self.assertRaises(AuditIntegrityError):
            verify_records(recs)

    def test_invalid_json_line(self):
        self.path.write_text('{"a":1}\n{not json\n')
        with self.assertRaises(AuditIntegrityError):
            read_records(self.path)

    def test_empty_and_missing_file(self):
        self.assertEqual(read_records(self.tmp.name + "/none.log"), [])
        self.assertEqual(verify_records([]), 0)


class TestDecisionProgramSyntax(unittest.TestCase):
    def test_missing_version_header(self):
        with self.assertRaises(DecisionProgramError):
            parse_decision_program("POLICY p SHA256 abc\nEND\n")

    def test_missing_end(self):
        with self.assertRaises(DecisionProgramError):
            parse_decision_program("DECISION_PROGRAM VERSION 1\n")

    def test_whitespace_only(self):
        with self.assertRaises(DecisionProgramError):
            parse_decision_program("\n  \n")


if __name__ == "__main__":
    unittest.main()
