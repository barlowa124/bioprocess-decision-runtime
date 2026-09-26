from __future__ import annotations

import copy
import hashlib
import json
import unittest
from pathlib import Path

from bioprocess_runtime.serialization import canonical_json


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "artifacts" / "gemma270m_architecture_manifest.json"


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class GemmaIrTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from bioprocess_runtime.gemma_ir import compile_gemma_ir

        if not MANIFEST.is_file():
            raise unittest.SkipTest(
                "Original architecture manifest is intentionally outside Git"
            )
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        cls.program = compile_gemma_ir(cls.manifest)

    def test_compiles_complete_typed_gemma_program(self) -> None:
        from bioprocess_runtime.gemma_ir import verify_gemma_ir

        verification = verify_gemma_ir(self.program, self.manifest)
        self.assertTrue(verification["valid"], verification)
        self.assertEqual(verification["instruction_count"], 533)
        self.assertEqual(verification["tensor_count"], 554)
        self.assertTrue(verification["dependency_graph_complete"])
        self.assertEqual(self.program["configuration"]["layers"], 18)
        self.assertEqual(self.program["opaque_composite_operations"], [])
        self.assertFalse(self.program["bit_exact_numerical_execution_qualified"])
        self.assertFalse(self.program["deployed_backend_correspondence_established"])
        self.assertEqual(self.program["instructions"][-1]["opcode"], "ARGMAX")
        self.assertEqual(self.program["instructions"][-1]["outputs"], ["selected_token_id"])
        self.assertEqual(self.program["instructions"][0]["inputs"], ["input_ids"])
        mask_instructions = [
            instruction
            for instruction in self.program["instructions"]
            if instruction["opcode"] == "CAUSAL_MASK"
        ]
        self.assertEqual(len(mask_instructions), 2)
        self.assertTrue(
            all(
                instruction["inputs"] == ["input_ids", "hidden.0"]
                for instruction in mask_instructions
            )
        )

    def test_every_instruction_has_declared_semantics_and_hash(self) -> None:
        from bioprocess_runtime.gemma_ir import LEGACY_OPCODES, SEMANTICS_VERSION

        self.assertEqual(
            {instruction["opcode"] for instruction in self.program["instructions"]},
            LEGACY_OPCODES - {"SOFTCAP"},
        )
        for instruction in self.program["instructions"]:
            self.assertEqual(
                instruction["semantics"],
                f"{SEMANTICS_VERSION}:{instruction['opcode']}",
            )
            body = {
                key: value
                for key, value in instruction.items()
                if key != "instruction_sha256"
            }
            self.assertEqual(instruction["instruction_sha256"], _sha256(body))

    def test_verifier_rejects_opaque_operation_with_recomputed_hashes(self) -> None:
        from bioprocess_runtime.gemma_ir import verify_gemma_ir

        damaged = copy.deepcopy(self.program)
        damaged["instructions"][12]["opcode"] = "OPAQUE_GEMMA_ATTENTION"
        instruction_body = {
            key: value
            for key, value in damaged["instructions"][12].items()
            if key != "instruction_sha256"
        }
        damaged["instructions"][12]["instruction_sha256"] = _sha256(instruction_body)
        program_body = {
            key: value for key, value in damaged.items() if key != "program_sha256"
        }
        damaged["program_sha256"] = _sha256(program_body)
        verification = verify_gemma_ir(damaged, self.manifest)
        self.assertFalse(verification["valid"])
        self.assertFalse(verification["semantics_coverage_valid"])

    def test_verifier_rejects_dependency_and_parameter_tampering(self) -> None:
        from bioprocess_runtime.gemma_ir import verify_gemma_ir

        missing_input = copy.deepcopy(self.program)
        missing_input["instructions"][20]["inputs"] = ["unproduced.tensor"]
        self.assertFalse(verify_gemma_ir(missing_input, self.manifest)["valid"])
        missing_parameter = copy.deepcopy(self.program)
        missing_parameter["instructions"][8]["parameter_refs"] = ["missing.weight"]
        self.assertFalse(verify_gemma_ir(missing_parameter, self.manifest)["valid"])

    def test_compiler_rejects_unbound_scaled_rope(self) -> None:
        from bioprocess_runtime.gemma_ir import compile_gemma_ir

        scaled = copy.deepcopy(self.manifest)
        scaled["model"]["config"]["rope_scaling"] = {
            "rope_type": "yarn",
            "factor": 2.0,
        }
        with self.assertRaisesRegex(ValueError, "Unsupported rope_scaling type"):
            compile_gemma_ir(scaled)

    def test_verifier_rejects_malformed_structures_without_crashing(self) -> None:
        from bioprocess_runtime.gemma_ir import verify_gemma_ir

        malformed_layer_count = copy.deepcopy(self.program)
        malformed_layer_count["configuration"]["layers"] = "18"
        body = {
            key: value
            for key, value in malformed_layer_count.items()
            if key != "program_sha256"
        }
        malformed_layer_count["program_sha256"] = _sha256(body)
        self.assertFalse(verify_gemma_ir(malformed_layer_count)["valid"])
        malformed_instruction = copy.deepcopy(self.program)
        malformed_instruction["instructions"][0] = None
        body = {
            key: value
            for key, value in malformed_instruction.items()
            if key != "program_sha256"
        }
        malformed_instruction["program_sha256"] = _sha256(body)
        self.assertFalse(verify_gemma_ir(malformed_instruction)["valid"])

    def test_rationale_is_complete_structural_backward_slice(self) -> None:
        from bioprocess_runtime.gemma_ir import build_rationale_slice, verify_rationale_slice

        rationale = build_rationale_slice(self.program)
        verification = verify_rationale_slice(self.program, rationale)
        self.assertTrue(verification["valid"], verification)
        self.assertEqual(verification["instruction_count"], 533)
        self.assertEqual(rationale["external_inputs"], ["input_ids"])
        self.assertIn("model.embed_tokens.weight", rationale["parameter_refs"])
        self.assertIn("lm_head.weight", rationale["parameter_refs"])
        self.assertTrue(rationale["dependency_slice_complete"])
        self.assertFalse(rationale["numerical_execution_witness_bound"])
        self.assertFalse(rationale["causal_semantic_interpretation_established"])
        damaged = copy.deepcopy(rationale)
        damaged["instruction_ids"].pop()
        self.assertFalse(verify_rationale_slice(self.program, damaged)["valid"])


if __name__ == "__main__":
    unittest.main()
