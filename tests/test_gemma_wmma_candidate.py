from __future__ import annotations

import copy
from fractions import Fraction
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


class GemmaWmmaCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.program = _load("gemma3_270m_execution_ir.json")
        cls.reduction = _load("gemma3_270m_reduction_characterization.json")
        cls.suite = _load("gemma3_270m_nsight_kernel_suite.json")
        cls.backend = _load("gemma3_270m_reduction_backend_binding.json")
        cls.position = _load("gemma3_270m_wmma_probe.json")
        cls.accumulator = _load("gemma3_270m_wmma_accumulator_probe.json")
        cls.magnitude = _load("gemma3_270m_wmma_magnitude_probe.json")
        cls.search = _load("gemma3_270m_wmma_candidate_search.json")

    def test_unique_probe_consistent_candidate(self) -> None:
        from bioprocess_runtime.gemma_wmma_candidate import (
            verify_wmma_candidate_search,
        )

        verification = verify_wmma_candidate_search(
            self.program,
            self.reduction,
            self.suite,
            self.backend,
            self.position,
            self.accumulator,
            self.magnitude,
            self.search,
        )
        self.assertTrue(verification["valid"], verification)
        self.assertEqual(verification["profile_count"], 64)
        self.assertEqual(verification["total_records"], 5384)
        self.assertEqual(verification["best_total_matches"], 5384)
        self.assertEqual(verification["complete_matching_profile_count"], 1)
        self.assertTrue(verification["unique_all_matching_candidate_in_search_space"])
        self.assertFalse(verification["complete_numeric_transition_established"])
        self.assertEqual(self.search["complete_matching_profiles"], [
            {
                "precision_bits": 25,
                "rounding_mode": "toward_zero",
                "half_order": "lower_then_upper",
            }
        ])
        self.assertEqual(self.search["mismatch_frontier"], [])

    def test_shared_exponent_precision_and_rounding(self) -> None:
        from bioprocess_runtime.gemma_wmma_candidate import _shared_exponent_sum

        values = [Fraction(1 << 30), Fraction(103), Fraction(-(1 << 30))]
        self.assertEqual(
            _shared_exponent_sum(values, 25, "toward_zero"),
            Fraction(64),
        )
        self.assertEqual(
            _shared_exponent_sum(values, 25, "nearest_even"),
            Fraction(128),
        )
        self.assertEqual(_shared_exponent_sum([], 25, "toward_zero"), Fraction(0))

    def test_candidate_reproduces_position_sensitive_examples(self) -> None:
        from bioprocess_runtime.gemma_wmma_candidate import k16_half_transition_bits

        values = [0] * 16
        values[0:3] = [0x4E80, 0xCE80, 0x3F80]
        self.assertEqual(
            k16_half_transition_bits(values, 25, "toward_zero", "lower_then_upper"),
            0x0000,
        )
        values = [0] * 16
        values[0] = 0x4E80
        values[1] = 0xCE80
        values[8] = 0x3F80
        self.assertEqual(
            k16_half_transition_bits(values, 25, "toward_zero", "lower_then_upper"),
            0x3F80,
        )
        self.assertEqual(
            k16_half_transition_bits(values, 25, "toward_zero", "upper_then_lower"),
            0x0000,
        )

    def test_record_vector_rejects_modulo_lane_collision(self) -> None:
        from bioprocess_runtime.gemma_wmma_candidate import _record_vector

        with self.assertRaisesRegex(ValueError, "multiple values"):
            _record_vector({
                "positions": [0, 16],
                "value_bits": ["0x3f80", "0x4000"],
            })

    def test_candidate_search_rejects_tampering(self) -> None:
        from bioprocess_runtime.gemma_wmma_candidate import (
            verify_wmma_candidate_search,
        )

        damaged = copy.deepcopy(self.search)
        damaged["complete_matching_profiles"][0]["precision_bits"] = 24
        body = {
            key: value for key, value in damaged.items() if key != "search_sha256"
        }
        damaged["search_sha256"] = _sha256(body)
        verification = verify_wmma_candidate_search(
            self.program,
            self.reduction,
            self.suite,
            self.backend,
            self.position,
            self.accumulator,
            self.magnitude,
            damaged,
        )
        self.assertFalse(verification["valid"])
        self.assertFalse(verification["derived_search_exact_match"])
        activated = copy.deepcopy(self.search)
        activated["complete_numeric_transition_established"] = True
        body = {
            key: value for key, value in activated.items() if key != "search_sha256"
        }
        activated["search_sha256"] = _sha256(body)
        self.assertFalse(
            verify_wmma_candidate_search(
                self.program,
                self.reduction,
                self.suite,
                self.backend,
                self.position,
                self.accumulator,
                self.magnitude,
                activated,
            )["valid"]
        )


if __name__ == "__main__":
    unittest.main()
