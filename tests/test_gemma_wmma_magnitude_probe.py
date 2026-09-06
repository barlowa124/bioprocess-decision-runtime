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


class GemmaWmmaMagnitudeProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.program = _load("gemma3_270m_execution_ir.json")
        cls.reduction = _load("gemma3_270m_reduction_characterization.json")
        cls.suite = _load("gemma3_270m_nsight_kernel_suite.json")
        cls.backend = _load("gemma3_270m_reduction_backend_binding.json")
        cls.certificate = _load("gemma3_270m_wmma_magnitude_probe.json")

    def test_magnitude_certificate_verifies(self) -> None:
        from bioprocess_runtime.gemma_wmma_magnitude_probe import (
            verify_wmma_magnitude_probe_certificate,
        )

        verification = verify_wmma_magnitude_probe_certificate(
            self.program,
            self.reduction,
            self.suite,
            self.backend,
            self.certificate,
        )
        self.assertTrue(verification["valid"], verification)
        self.assertEqual(verification["probe_count"], 1024)
        self.assertEqual(verification["distinct_result_count"], 133)
        self.assertEqual(verification["sign_symmetry_match_count"], 188)
        self.assertFalse(verification["sign_symmetry_established_for_all_pairs"])
        self.assertFalse(verification["magnitude_generalization_established"])
        self.assertFalse(verification["wmma_accumulator_mapping_identified"])

    def test_lower_cancel_then_upper_small_preserves_every_magnitude(self) -> None:
        statistics = self.certificate["placement_statistics"]
        for placement in (
            "lower_cancel_small_upper_8",
            "lower_cancel_small_upper_15",
        ):
            self.assertEqual(statistics[placement]["probes"], 128)
            self.assertEqual(statistics[placement]["exact_small"], 128)
            self.assertEqual(statistics[placement]["zero"], 0)
            self.assertEqual(statistics[placement]["other"], 0)
            self.assertEqual(statistics[placement]["minimum_exact_exponent_field"], 1)
            self.assertIsNone(statistics[placement]["maximum_nonexact_exponent_field"])

    def test_other_placements_share_observed_recovery_threshold(self) -> None:
        statistics = self.certificate["placement_statistics"]
        excluded = {
            "lower_cancel_small_upper_8",
            "lower_cancel_small_upper_15",
        }
        for placement, result in statistics.items():
            if placement in excluded:
                continue
            self.assertEqual(result["probes"], 128)
            self.assertEqual(result["exact_small"], 16)
            self.assertEqual(result["zero"], 108)
            self.assertEqual(result["other"], 4)
            self.assertEqual(result["other_result_histogram"], {
                "0x4280": 1,
                "0x4470": 1,
                "0xc280": 1,
                "0xc470": 1,
            })
            self.assertEqual(result["minimum_exact_exponent_field"], 138)
            self.assertEqual(result["maximum_nonexact_exponent_field"], 136)

    def test_magnitude_probe_rejects_record_and_claim_tampering(self) -> None:
        from bioprocess_runtime.gemma_wmma_magnitude_probe import (
            verify_wmma_magnitude_probe_certificate,
        )

        damaged = copy.deepcopy(self.certificate)
        damaged["records"][0]["exponent_field"] = 2
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
        verification = verify_wmma_magnitude_probe_certificate(
            self.program, self.reduction, self.suite, self.backend, damaged
        )
        self.assertFalse(verification["valid"])
        self.assertFalse(verification["records_valid"])
        activated = copy.deepcopy(self.certificate)
        activated["magnitude_generalization_established"] = True
        body = {
            key: value for key, value in activated.items() if key != "certificate_sha256"
        }
        activated["certificate_sha256"] = _sha256(body)
        self.assertFalse(
            verify_wmma_magnitude_probe_certificate(
                self.program,
                self.reduction,
                self.suite,
                self.backend,
                activated,
            )["valid"]
        )


if __name__ == "__main__":
    unittest.main()
