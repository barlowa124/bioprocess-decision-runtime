from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import unittest
from pathlib import Path

from bioprocess_runtime.serialization import canonical_json


ROOT = Path(__file__).resolve().parent.parent
PROGRAM = ROOT / "results" / "gemma3_270m_execution_ir.json"


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch is required")
class GemmaIrPrimitiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from bioprocess_runtime.gemma_float_semantics import (
            build_bfloat16_semantics_certificate,
        )
        from bioprocess_runtime.gemma_ir_primitives import (
            build_primitive_qualification_certificate,
        )
        from bioprocess_runtime.gemma_reduction_semantics import (
            build_reduction_characterization_certificate,
        )

        cls.program = json.loads(PROGRAM.read_text(encoding="utf-8"))
        cls.certificate = build_primitive_qualification_certificate()
        cls.bfloat16_certificate = build_bfloat16_semantics_certificate()
        cls.reduction_certificate = build_reduction_characterization_certificate()

    def test_independent_index_oracles_reexecute_exactly(self) -> None:
        from bioprocess_runtime.gemma_ir_primitives import (
            EXACT_INDEX_OPCODES,
            verify_primitive_qualification_certificate,
        )

        verification = verify_primitive_qualification_certificate(self.certificate)
        self.assertTrue(verification["valid"], verification)
        self.assertEqual(verification["record_count"], 610)
        self.assertEqual(set(verification["case_counts"]), EXACT_INDEX_OPCODES)
        self.assertTrue(all(count > 0 for count in verification["case_counts"].values()))
        self.assertFalse(
            self.certificate["floating_point_primitive_qualification_established"]
        )
        self.assertFalse(
            self.certificate["unrestricted_domain_qualification_established"]
        )

    def test_qualification_certificate_rejects_forged_case(self) -> None:
        from bioprocess_runtime.gemma_ir_primitives import (
            verify_primitive_qualification_certificate,
        )

        damaged = copy.deepcopy(self.certificate)
        damaged["records"][0]["expected_sha256"] = "0" * 64
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
        verification = verify_primitive_qualification_certificate(
            damaged, reexecute=False
        )
        self.assertFalse(verification["valid"])
        self.assertFalse(verification["records_valid"])
        malformed = copy.deepcopy(self.certificate)
        malformed["records"][0] = []
        body = {
            key: value
            for key, value in malformed.items()
            if key != "certificate_sha256"
        }
        malformed["certificate_sha256"] = _sha256(body)
        malformed_verification = verify_primitive_qualification_certificate(
            malformed, reexecute=False
        )
        self.assertFalse(malformed_verification["valid"])
        self.assertFalse(malformed_verification["records_valid"])

    def test_gate_blocks_unqualified_floating_point_primitives(self) -> None:
        from bioprocess_runtime.gemma_ir_primitives import (
            build_primitive_qualification_gate,
            verify_primitive_qualification_gate,
        )

        gate = build_primitive_qualification_gate(
            self.program,
            self.certificate,
            self.bfloat16_certificate,
            self.reduction_certificate,
        )
        verification = verify_primitive_qualification_gate(
            self.program,
            self.certificate,
            self.bfloat16_certificate,
            self.reduction_certificate,
            gate,
        )
        self.assertTrue(verification["valid"], verification)
        self.assertEqual(verification["independently_tested_opcode_count"], 8)
        self.assertEqual(
            verification["independently_specified_finite_bfloat16_opcode_count"],
            3,
        )
        self.assertEqual(verification["bounded_characterized_reduction_opcode_count"], 3)
        self.assertFalse(verification["unique_reduction_profile_identified"])
        self.assertFalse(verification["reduction_order_semantics_established"])
        self.assertEqual(verification["unrestricted_floating_point_opcode_count"], 11)
        self.assertFalse(verification["all_reached_primitive_semantics_qualified"])
        self.assertFalse(verification["global_exactness_activation_allowed"])
        self.assertIn("LINEAR", gate["unrestricted_floating_point_opcodes"])
        self.assertIn("SOFTMAX", gate["unrestricted_floating_point_opcodes"])

    def test_gate_rejects_forged_activation(self) -> None:
        from bioprocess_runtime.gemma_ir_primitives import (
            build_primitive_qualification_gate,
            verify_primitive_qualification_gate,
        )

        gate = build_primitive_qualification_gate(
            self.program,
            self.certificate,
            self.bfloat16_certificate,
            self.reduction_certificate,
        )
        damaged = copy.deepcopy(gate)
        damaged["global_exactness_activation_allowed"] = True
        body = {key: value for key, value in damaged.items() if key != "gate_sha256"}
        damaged["gate_sha256"] = _sha256(body)
        verification = verify_primitive_qualification_gate(
            self.program,
            self.certificate,
            self.bfloat16_certificate,
            self.reduction_certificate,
            damaged,
        )
        self.assertFalse(verification["valid"])
        self.assertFalse(verification["derived_gate_exact_match"])


if __name__ == "__main__":
    unittest.main()
