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


class GemmaReductionBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.program = _load("gemma3_270m_execution_ir.json")
        cls.reduction = _load("gemma3_270m_reduction_characterization.json")
        cls.suite = _load("gemma3_270m_nsight_kernel_suite.json")
        cls.binding = _load("gemma3_270m_reduction_backend_binding.json")

    def test_controlled_gemma_shapes_bind_linear_kernel_symbols(self) -> None:
        from bioprocess_runtime.gemma_reduction_backend import (
            verify_gemma_reduction_backend_binding,
        )

        verification = verify_gemma_reduction_backend_binding(
            self.program, self.reduction, self.suite, self.binding
        )
        self.assertTrue(verification["valid"], verification)
        self.assertEqual(verification["record_count"], 24)
        self.assertEqual(verification["exact_nsight_symbol_overlap_record_count"], 18)
        self.assertTrue(verification["all_linear_roles_have_attested_symbol_overlap"])
        self.assertFalse(
            verification["canonical_eager_attention_symbols_attested_in_deployed_suite"]
        )
        self.assertFalse(verification["reduction_order_semantics_established"])
        self.assertEqual(
            set(self.binding["vector_classes"]),
            {"cancellation", "grouped", "pseudo"},
        )

    def test_binding_preserves_controlled_value_and_semantic_boundaries(self) -> None:
        self.assertFalse(self.binding["controlled_values_equal_recorded_model_tensors"])
        self.assertFalse(self.binding["per_invocation_argument_binding_established"])
        self.assertFalse(self.binding["reduction_order_semantics_established"])
        self.assertFalse(self.binding["tensor_core_accumulator_semantics_established"])
        self.assertFalse(self.binding["hardware_instruction_semantics_established"])
        linear_records = [
            record for record in self.binding["records"] if record["primitive"] == "LINEAR"
        ]
        self.assertEqual(len(linear_records), 18)
        self.assertTrue(
            all(record["exact_nsight_suite_symbol_matches"] for record in linear_records)
        )

    def test_binding_rejects_symbol_and_claim_tampering(self) -> None:
        from bioprocess_runtime.gemma_reduction_backend import (
            verify_gemma_reduction_backend_binding,
        )

        damaged = copy.deepcopy(self.binding)
        damaged["records"][0]["cuda_kernel_names"] = ["forged_kernel"]
        record_body = {
            key: value
            for key, value in damaged["records"][0].items()
            if key != "record_sha256"
        }
        damaged["records"][0]["record_sha256"] = _sha256(record_body)
        body = {
            key: value for key, value in damaged.items() if key != "binding_sha256"
        }
        damaged["binding_sha256"] = _sha256(body)
        verification = verify_gemma_reduction_backend_binding(
            self.program, self.reduction, self.suite, damaged
        )
        self.assertFalse(verification["valid"])
        self.assertFalse(verification["records_valid"])
        forged_shape = copy.deepcopy(self.binding)
        forged_shape["records"][0]["input_shape"][0] += 1
        record_body = {
            key: value
            for key, value in forged_shape["records"][0].items()
            if key != "record_sha256"
        }
        forged_shape["records"][0]["record_sha256"] = _sha256(record_body)
        body = {
            key: value
            for key, value in forged_shape.items()
            if key != "binding_sha256"
        }
        forged_shape["binding_sha256"] = _sha256(body)
        self.assertFalse(
            verify_gemma_reduction_backend_binding(
                self.program, self.reduction, self.suite, forged_shape
            )["valid"]
        )
        forged_candidates = copy.deepcopy(self.binding)
        candidate_record = forged_candidates["records"][0]
        candidate_record["candidate_results"]["exact_sum_then_bfloat16"] = (
            candidate_record["actual_result_bits"]
        )
        candidate_record["matching_candidates"] = sorted(
            name
            for name, result in candidate_record["candidate_results"].items()
            if result == candidate_record["actual_result_bits"]
        )
        record_body = {
            key: value
            for key, value in candidate_record.items()
            if key != "record_sha256"
        }
        candidate_record["record_sha256"] = _sha256(record_body)
        body = {
            key: value
            for key, value in forged_candidates.items()
            if key != "binding_sha256"
        }
        forged_candidates["binding_sha256"] = _sha256(body)
        self.assertFalse(
            verify_gemma_reduction_backend_binding(
                self.program, self.reduction, self.suite, forged_candidates
            )["valid"]
        )
        activated = copy.deepcopy(self.binding)
        activated["reduction_order_semantics_established"] = True
        body = {
            key: value for key, value in activated.items() if key != "binding_sha256"
        }
        activated["binding_sha256"] = _sha256(body)
        self.assertFalse(
            verify_gemma_reduction_backend_binding(
                self.program, self.reduction, self.suite, activated
            )["valid"]
        )


if __name__ == "__main__":
    unittest.main()
