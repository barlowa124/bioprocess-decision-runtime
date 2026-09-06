from __future__ import annotations

import copy
import hashlib
import json
import unittest
from pathlib import Path

from bioprocess_runtime.serialization import canonical_json


ROOT = Path(__file__).resolve().parent.parent


def _load(name: str) -> dict:
    return json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class GemmaWmmaAccumulatorProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.program = _load("gemma3_270m_execution_ir.json")
        cls.reduction = _load("gemma3_270m_reduction_characterization.json")
        cls.suite = _load("gemma3_270m_nsight_kernel_suite.json")
        cls.backend = _load("gemma3_270m_reduction_backend_binding.json")
        cls.certificate = _load("gemma3_270m_wmma_accumulator_probe.json")

    def test_exhaustive_triplet_certificate_verifies(self) -> None:
        from bioprocess_runtime.gemma_wmma_accumulator_probe import (
            verify_wmma_accumulator_probe_certificate,
        )

        verification = verify_wmma_accumulator_probe_certificate(
            self.program,
            self.reduction,
            self.suite,
            self.backend,
            self.certificate,
        )
        self.assertTrue(verification["valid"], verification)
        self.assertEqual(verification["probe_count"], 3360)
        self.assertEqual(verification["retained_count"], 448)
        self.assertTrue(verification["small_retention_position_dependent"])
        self.assertTrue(
            verification[
                "candidate_retention_rule_matches_exhaustive_triplet_domain"
            ]
        )
        self.assertFalse(
            verification[
                "candidate_k16_half_order_full_numeric_semantics_established"
            ]
        )
        self.assertFalse(verification["wmma_accumulator_mapping_identified"])

    def test_retention_exactly_matches_lower_then_upper_rule(self) -> None:
        self.assertTrue(self.certificate["triplet_domain_exhausted"])
        self.assertEqual(self.certificate["triplet_domain_size"], 3360)
        self.assertEqual(self.certificate["result_histogram"], {
            "0x0000": 2912,
            "0x3f80": 448,
        })
        self.assertEqual(self.certificate["candidate_retention_rule_match_count"], 3360)
        for record in self.certificate["records"]:
            predicted = (
                record["positive_lane"] < 8
                and record["negative_lane"] < 8
                and record["small_lane"] >= 8
            )
            self.assertEqual(record["small_value_retained"], predicted)

    def test_lower_small_lanes_never_retain_and_upper_each_retain_56(self) -> None:
        statistics = self.certificate["lane_statistics"]
        for lane in range(8):
            self.assertEqual(statistics[str(lane)], {"probes": 210, "retained": 0})
        for lane in range(8, 16):
            self.assertEqual(statistics[str(lane)], {"probes": 210, "retained": 56})

    def test_accumulator_probe_rejects_record_and_claim_tampering(self) -> None:
        from bioprocess_runtime.gemma_wmma_accumulator_probe import (
            verify_wmma_accumulator_probe_certificate,
        )

        damaged = copy.deepcopy(self.certificate)
        damaged["records"][0]["small_lane"] = 15
        record_body = {
            key: value
            for key, value in damaged["records"][0].items()
            if key != "record_sha256"
        }
        damaged["records"][0]["record_sha256"] = _sha256(record_body)
        body = {
            key: value for key, value in damaged.items() if key != "certificate_sha256"
        }
        damaged["certificate_sha256"] = _sha256(body)
        verification = verify_wmma_accumulator_probe_certificate(
            self.program, self.reduction, self.suite, self.backend, damaged
        )
        self.assertFalse(verification["valid"])
        self.assertFalse(verification["records_valid"])
        activated = copy.deepcopy(self.certificate)
        activated["wmma_accumulator_mapping_identified"] = True
        body = {
            key: value for key, value in activated.items() if key != "certificate_sha256"
        }
        activated["certificate_sha256"] = _sha256(body)
        self.assertFalse(
            verify_wmma_accumulator_probe_certificate(
                self.program,
                self.reduction,
                self.suite,
                self.backend,
                activated,
            )["valid"]
        )


if __name__ == "__main__":
    unittest.main()
