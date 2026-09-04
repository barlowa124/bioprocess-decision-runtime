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
class GemmaFloatSemanticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from bioprocess_runtime.gemma_float_semantics import (
            build_bfloat16_semantics_certificate,
        )

        cls.certificate = build_bfloat16_semantics_certificate()

    def test_exact_rational_bfloat16_operations(self) -> None:
        from bioprocess_runtime.gemma_float_semantics import (
            bfloat16_add_bits,
            bfloat16_multiply_bits,
            decode_finite_bfloat16,
            encode_bfloat16_rne,
        )

        self.assertEqual(bfloat16_add_bits(0x3F80, 0x3F80), 0x4000)
        self.assertEqual(bfloat16_multiply_bits(0x3FC0, 0x4000), 0x4040)
        self.assertEqual(bfloat16_multiply_bits(0x8000, 0x4000), 0x8000)
        self.assertEqual(bfloat16_add_bits(0x8000, 0x8000), 0x8000)
        self.assertEqual(bfloat16_add_bits(0x7F7F, 0x7F7F), 0x7F80)
        self.assertEqual(bfloat16_add_bits(0x007F, 0x0001), 0x0080)
        self.assertEqual(bfloat16_multiply_bits(0x0001, 0x3F00), 0x0000)
        self.assertEqual(bfloat16_multiply_bits(0x0003, 0x3F00), 0x0002)
        self.assertEqual(encode_bfloat16_rne(Fraction(257, 256)), 0x3F80)
        self.assertEqual(encode_bfloat16_rne(Fraction(259, 256)), 0x3F82)
        value, negative_zero = decode_finite_bfloat16(0x8080)
        self.assertEqual(encode_bfloat16_rne(value, negative_zero), 0x8080)

    def test_all_finite_patterns_round_trip(self) -> None:
        self.assertEqual(self.certificate["finite_pattern_roundtrip_count"], 65280)
        self.assertTrue(self.certificate["finite_pattern_roundtrip_complete"])

    def test_cpu_and_cuda_conformance_reexecute(self) -> None:
        from bioprocess_runtime.gemma_float_semantics import (
            verify_bfloat16_semantics_certificate,
        )

        verification = verify_bfloat16_semantics_certificate(self.certificate)
        self.assertTrue(verification["valid"], verification)
        self.assertEqual(verification["operation_counts"], {
            "ADD": 484,
            "MUL": 484,
            "SCALE_TENSOR": 110,
            "SCALE_PYTHON": 110,
        })
        if verification["cuda_available"]:
            self.assertTrue(verification["cuda_all_exact"])
        else:
            self.assertIsNone(verification["cuda_all_exact"])
        self.assertFalse(self.certificate["complete_binary_truth_tables_established"])
        self.assertFalse(self.certificate["reduction_order_semantics_established"])
        self.assertFalse(self.certificate["hardware_instruction_semantics_established"])

    def test_rejects_special_inputs_and_certificate_tampering(self) -> None:
        from bioprocess_runtime.gemma_float_semantics import (
            decode_finite_bfloat16,
            verify_bfloat16_semantics_certificate,
        )

        with self.assertRaisesRegex(ValueError, "outside finite"):
            decode_finite_bfloat16(0x7F80)
        with self.assertRaisesRegex(ValueError, "outside finite"):
            decode_finite_bfloat16(0x7FC1)
        damaged = copy.deepcopy(self.certificate)
        damaged["complete_binary_truth_tables_established"] = True
        self.assertFalse(
            verify_bfloat16_semantics_certificate(damaged, reexecute=False)["valid"]
        )
        forged = copy.deepcopy(self.certificate)
        forged_record = next(
            record
            for record in forged["records"]
            if record["oracle_result_bits"] != "0x0000"
        )
        forged_record["oracle_result_bits"] = "0x0000"
        forged_record["cpu_result_bits"] = "0x0000"
        if forged_record["cuda_result_bits"] is not None:
            forged_record["cuda_result_bits"] = "0x0000"
        record_body = {
            key: value
            for key, value in forged_record.items()
            if key != "record_sha256"
        }
        forged_record["record_sha256"] = _sha256(record_body)
        certificate_body = {
            key: value
            for key, value in forged.items()
            if key != "certificate_sha256"
        }
        forged["certificate_sha256"] = _sha256(certificate_body)
        forged_verification = verify_bfloat16_semantics_certificate(
            forged, reexecute=False
        )
        self.assertFalse(forged_verification["valid"])
        self.assertFalse(forged_verification["records_valid"])


if __name__ == "__main__":
    unittest.main()
