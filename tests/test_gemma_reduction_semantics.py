from __future__ import annotations

import copy
from fractions import Fraction
import hashlib
import importlib.util
import unittest

from bioprocess_runtime.serialization import canonical_json


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch is required")
class GemmaReductionSemanticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from bioprocess_runtime.gemma_reduction_semantics import (
            build_reduction_characterization_certificate,
        )

        cls.certificate = build_reduction_characterization_certificate()

    def test_float32_rne_and_round_trip_examples(self) -> None:
        from bioprocess_runtime.gemma_reduction_semantics import (
            decode_finite_float32,
            encode_float32_rne,
        )

        for bits in (
            0x00000000,
            0x80000000,
            0x00000001,
            0x007FFFFF,
            0x00800000,
            0x3F800000,
            0xBF800000,
            0x7F7FFFFF,
            0xFF7FFFFF,
        ):
            value, negative_zero = decode_finite_float32(bits)
            self.assertEqual(encode_float32_rne(value, negative_zero), bits)
        self.assertEqual(
            encode_float32_rne(Fraction((1 << 24) + 1, 1 << 24)),
            0x3F800000,
        )
        self.assertEqual(
            encode_float32_rne(Fraction((1 << 24) + 3, 1 << 24)),
            0x3F800002,
        )
        self.assertEqual(
            encode_float32_rne(-Fraction((1 << 24) + 1, 1 << 24)),
            0xBF800000,
        )
        self.assertEqual(encode_float32_rne(Fraction(1 << 128)), 0x7F800000)
        self.assertEqual(encode_float32_rne(-Fraction(1 << 128)), 0xFF800000)

    def test_reduction_candidates_expose_order_dependence(self) -> None:
        from bioprocess_runtime.gemma_reduction_semantics import (
            block_float32_then_bfloat16,
            exact_sum_then_bfloat16,
            pairwise_float32_then_bfloat16,
            sequential_bfloat16,
            sequential_float32_then_bfloat16,
        )

        left = [0x4E80, 0x3F80, 0xCE80, 0x3F80]
        right = [0x3F80] * 4
        self.assertEqual(exact_sum_then_bfloat16(left, right), 0x4000)
        self.assertEqual(pairwise_float32_then_bfloat16(left, right), 0x0000)
        self.assertEqual(sequential_float32_then_bfloat16(left, right), 0x3F80)
        self.assertEqual(sequential_bfloat16(left, right), 0x3F80)
        odd_left = [0x3F80] * 5
        odd_right = [0x3F80] * 5
        self.assertEqual(
            pairwise_float32_then_bfloat16(odd_left, odd_right),
            exact_sum_then_bfloat16(odd_left, odd_right),
        )
        tail_left = [0x3F80] * 17
        tail_right = [0x3F80] * 17
        self.assertEqual(
            block_float32_then_bfloat16(tail_left, tail_right),
            exact_sum_then_bfloat16(tail_left, tail_right),
        )

    def test_characterization_reexecutes_without_selecting_profile(self) -> None:
        from bioprocess_runtime.gemma_reduction_semantics import (
            verify_reduction_characterization_certificate,
        )

        verification = verify_reduction_characterization_certificate(self.certificate)
        self.assertTrue(verification["valid"], verification)
        self.assertEqual(verification["record_count"], 81)
        self.assertGreater(verification["discriminating_record_count"], 0)
        self.assertFalse(verification["unique_reduction_profile_identified"])
        self.assertFalse(verification["reduction_order_semantics_established"])
        self.assertFalse(self.certificate["tensor_core_semantics_established"])
        self.assertFalse(self.certificate["hardware_instruction_semantics_established"])

    def test_characterization_rejects_tampering(self) -> None:
        from bioprocess_runtime.gemma_reduction_semantics import (
            verify_reduction_characterization_certificate,
        )

        damaged = copy.deepcopy(self.certificate)
        damaged["unique_reduction_profile_identified"] = True
        self.assertFalse(
            verify_reduction_characterization_certificate(
                damaged, reexecute=False
            )["valid"]
        )
        forged_counts = copy.deepcopy(self.certificate)
        forged_counts["match_counts"]["cpu"]["LINEAR"][
            "exact_sum_then_bfloat16"
        ] += 1
        forged_body = {
            key: value
            for key, value in forged_counts.items()
            if key != "certificate_sha256"
        }
        forged_counts["certificate_sha256"] = _sha256(forged_body)
        forged_verification = verify_reduction_characterization_certificate(
            forged_counts, reexecute=False
        )
        self.assertFalse(forged_verification["valid"])
        self.assertFalse(forged_verification["claims_consistent"])
        malformed = copy.deepcopy(self.certificate)
        malformed["records"][0] = []
        self.assertFalse(
            verify_reduction_characterization_certificate(
                malformed, reexecute=False
            )["valid"]
        )


if __name__ == "__main__":
    unittest.main()
