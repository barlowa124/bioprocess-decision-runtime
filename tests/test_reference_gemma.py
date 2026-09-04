from __future__ import annotations

import copy
import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path


DEPENDENCIES_AVAILABLE = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None


@unittest.skipUnless(DEPENDENCIES_AVAILABLE, "Gemma optional dependencies are not installed")
class ReferenceGemmaTests(unittest.TestCase):
    def setUp(self) -> None:
        import torch
        from transformers import Gemma3ForCausalLM, Gemma3TextConfig

        config = Gemma3TextConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=64,
            sliding_window=8,
            layer_types=["sliding_attention", "full_attention"],
            query_pre_attn_scalar=8,
            rope_theta=10000.0,
            rope_local_base_freq=1000.0,
        )
        self.torch = torch
        self.model = Gemma3ForCausalLM(config).eval()
        self.model.config._attn_implementation = "eager"
        self.input_ids = torch.tensor([[2, 4, 5, 6]], dtype=torch.long)

    def test_independent_orchestration_exactly_matches_eager_boundaries(self) -> None:
        from bioprocess_runtime.reference_gemma import fixed_input_equivalence_certificate

        certificate = fixed_input_equivalence_certificate(self.model, self.input_ids, absolute_tolerance=0.0)
        self.assertTrue(certificate["all_boundaries_within_tolerance"])
        self.assertTrue(certificate["selected_token_exact_match"])
        self.assertTrue(certificate["logits"]["exact_equal"])
        self.assertTrue(all(boundary["exact_equal"] for boundary in certificate["boundaries"]))
        self.assertEqual(len(certificate["attention_kernel_comparisons"]), 2)

    def test_typed_ir_interpreter_predicts_eager_output_exactly(self) -> None:
        from bioprocess_runtime.gemma_ir import compile_gemma_ir, verify_gemma_ir
        from bioprocess_runtime.gemma_ir_interpreter import (
            bind_model_tensors,
            build_ir_execution_certificate,
            compare_ir_execution,
            execute_gemma_ir,
            recompute_ir_execution_certificate,
            summarize_ir_execution,
            verify_ir_execution_certificate,
            verify_ir_execution_summary,
        )
        from bioprocess_runtime.operational_semantics import (
            append_chain_record,
            build_architecture_manifest,
        )
        from bioprocess_runtime.reference_gemma import model_state_sha256
        from bioprocess_runtime.serialization import canonical_json

        tokenizer = type(
            "Tokenizer",
            (),
            {
                "vocab_size": 32,
                "bos_token_id": 2,
                "eos_token_id": 1,
                "pad_token_id": 0,
            },
        )()
        with tempfile.TemporaryDirectory() as directory:
            manifest = build_architecture_manifest(
                self.model, tokenizer, Path(directory)
            )
        program = compile_gemma_ir(manifest)
        self.assertTrue(verify_gemma_ir(program, manifest)["valid"])
        parameters = bind_model_tensors(program, self.model)
        ir_input_ids = (self.torch.arange(12, dtype=self.torch.long) % 30).unsqueeze(0)
        prediction = execute_gemma_ir(program, parameters, ir_input_ids)
        with self.torch.no_grad():
            observed = self.model(
                input_ids=ir_input_ids,
                attention_mask=self.torch.ones_like(ir_input_ids),
                use_cache=False,
                logits_to_keep=1,
            )
        self.assertTrue(self.torch.equal(prediction.logits, observed.logits))
        self.assertTrue(
            self.torch.equal(
                prediction.selected_token_id,
                self.torch.argmax(observed.logits[:, -1, :], dim=-1),
            )
        )
        certificate = build_ir_execution_certificate(
            program,
            ir_input_ids,
            prediction,
            observed.logits,
            "huggingface_eager",
            model_state_sha256(self.model),
        )
        certificate_verification = verify_ir_execution_certificate(
            program, certificate
        )
        self.assertTrue(certificate_verification["valid"], certificate_verification)
        replay = recompute_ir_execution_certificate(program, certificate, self.model)
        self.assertTrue(replay["valid"], replay)
        self.assertTrue(replay["reexecution_performed"])
        self.assertTrue(certificate["fixed_input_canonical_eager_prediction_established"])
        self.assertFalse(certificate["deployed_sdpa_correspondence_established"])
        self.assertEqual(len(certificate["selection_proofs"]), 1)
        self.assertEqual(
            certificate["selection_proofs"][0]["selected_token_id"],
            certificate["predicted_token_ids"][0],
        )
        self.assertTrue(certificate["selection_proofs"][0]["argmax_recomputed"])
        self.assertTrue(
            certificate["selection_proofs"][0]["all_competitors_not_greater"]
        )
        summary = summarize_ir_execution(program, certificate)
        self.assertTrue(
            verify_ir_execution_summary(program, summary, certificate)["valid"]
        )
        self.assertFalse(summary["full_execution_records_committed"])
        wrong_source = copy.deepcopy(certificate)
        wrong_source["certificate_sha256"] = "0" * 64
        wrong_source_verification = verify_ir_execution_summary(
            program, summary, wrong_source
        )
        self.assertFalse(wrong_source_verification["valid"])
        self.assertFalse(
            wrong_source_verification["source_certificate_binding_valid"]
        )
        damaged_summary = copy.deepcopy(summary)
        damaged_summary["predicted_token_ids"] = [
            (summary["observed_token_ids"][0] + 1) % 32
        ]
        self.assertFalse(
            verify_ir_execution_summary(program, damaged_summary, certificate)["valid"]
        )
        damaged_certificate = copy.deepcopy(certificate)
        damaged_certificate["observed_logits"]["sha256"] = "0" * 64
        self.assertFalse(
            verify_ir_execution_certificate(program, damaged_certificate)["valid"]
        )
        ragged_certificate = copy.deepcopy(certificate)
        ragged_certificate["input_token_ids"] = [[1, 2], [3]]
        ragged_body = {
            key: value
            for key, value in ragged_certificate.items()
            if key != "certificate_sha256"
        }
        ragged_certificate["certificate_sha256"] = hashlib.sha256(
            canonical_json(ragged_body).encode("utf-8")
        ).hexdigest()
        ragged_verification = verify_ir_execution_certificate(
            program, ragged_certificate
        )
        self.assertFalse(ragged_verification["valid"])
        self.assertFalse(ragged_verification["input_value_binding_valid"])
        forged_selection = copy.deepcopy(certificate)
        forged_selection["selection_proofs"][0]["runner_up_token_id"] = (
            forged_selection["predicted_token_ids"][0]
        )
        selection_body = {
            key: value
            for key, value in forged_selection.items()
            if key != "certificate_sha256"
        }
        forged_selection["certificate_sha256"] = hashlib.sha256(
            canonical_json(selection_body).encode("utf-8")
        ).hexdigest()
        selection_verification = verify_ir_execution_certificate(
            program, forged_selection
        )
        self.assertFalse(selection_verification["valid"])
        self.assertFalse(selection_verification["selection_proofs_valid"])
        forged_certificate = copy.deepcopy(certificate)
        logits_name = program["declared_outputs"][0]
        forged_payloads = [
            copy.deepcopy(record["payload"])
            for record in forged_certificate["execution_records"]
        ]
        logits_payload = next(
            payload for payload in forged_payloads if logits_name in payload["outputs"]
        )
        logits_payload["outputs"][logits_name]["sha256"] = "f" * 64
        forged_records = []
        for payload in forged_payloads:
            append_chain_record(forged_records, payload)
        forged_certificate["execution_records"] = forged_records
        forged_certificate["execution_root_sha256"] = forged_records[-1]["record_hash"]
        certificate_body = {
            key: value
            for key, value in forged_certificate.items()
            if key != "certificate_sha256"
        }
        forged_certificate["certificate_sha256"] = hashlib.sha256(
            canonical_json(certificate_body).encode("utf-8")
        ).hexdigest()
        forged_verification = verify_ir_execution_certificate(
            program, forged_certificate
        )
        self.assertFalse(forged_verification["valid"])
        self.assertFalse(forged_verification["record_output_binding_valid"])
        self.assertEqual(len(prediction.records), len(program["instructions"]))
        self.assertEqual(prediction.records[-1]["payload"]["opcode"], "ARGMAX")
        exact_comparison = compare_ir_execution(
            program, prediction, list(prediction.records)
        )
        self.assertTrue(exact_comparison["exact_match"])
        damaged_records = copy.deepcopy(list(prediction.records))
        first_output = next(iter(damaged_records[10]["payload"]["outputs"].values()))
        first_output["sha256"] = "0" * 64
        divergence = compare_ir_execution(program, prediction, damaged_records)
        self.assertFalse(divergence["exact_match"])
        self.assertEqual(divergence["first_divergence"]["index"], 10)
        batched_input = self.torch.cat((ir_input_ids, ir_input_ids.flip(1)), dim=0)
        batched_prediction = execute_gemma_ir(program, parameters, batched_input)
        with self.torch.no_grad():
            batched_observed = self.model(
                input_ids=batched_input,
                attention_mask=self.torch.ones_like(batched_input),
                use_cache=False,
                logits_to_keep=1,
            )
        self.assertTrue(
            self.torch.equal(batched_prediction.logits, batched_observed.logits)
        )
        invalid_input = ir_input_ids.clone()
        invalid_input[0, 0] = 32
        with self.assertRaisesRegex(ValueError, "declared vocabulary"):
            execute_gemma_ir(program, parameters, invalid_input)

    def test_reference_matches_eager_beyond_sliding_window(self) -> None:
        from bioprocess_runtime.reference_gemma import fixed_input_equivalence_certificate

        long_input = (self.torch.arange(12, dtype=self.torch.long) % 30).unsqueeze(0)
        certificate = fixed_input_equivalence_certificate(self.model, long_input, absolute_tolerance=0.0)
        self.assertTrue(certificate["all_boundaries_within_tolerance"])

    def test_sdpa_comparison_rejects_attention_softcapping(self) -> None:
        from bioprocess_runtime.reference_gemma import reference_gemma_forward

        for layer in self.model.model.layers:
            layer.self_attn.attn_logit_softcapping = 50.0
        with self.assertRaisesRegex(ValueError, "softcapped"):
            reference_gemma_forward(self.model, self.input_ids, compare_attention_kernels=True)
        output = reference_gemma_forward(self.model, self.input_ids, compare_attention_kernels=False)
        self.assertEqual(output.logits.shape[-1], 32)

    def test_reference_trace_names_intermediate_computations(self) -> None:
        from bioprocess_runtime.reference_gemma import reference_gemma_forward

        output = reference_gemma_forward(self.model, self.input_ids)
        stages = {record["payload"]["stage"] for record in output.records}
        self.assertTrue({"query", "key", "value", "attention_probability", "mlp_gate", "mlp_intermediate_product", "vocabulary_logits"}.issubset(stages))

    def test_certificate_verifier_detects_changes(self) -> None:
        from bioprocess_runtime.reference_gemma import (
            fixed_input_equivalence_certificate,
            recompute_fixed_input_certificate,
            verify_fixed_input_certificate,
        )

        certificate = fixed_input_equivalence_certificate(self.model, self.input_ids, absolute_tolerance=0.0)
        self.assertTrue(verify_fixed_input_certificate(certificate)["valid"])
        self.assertTrue(recompute_fixed_input_certificate(self.model, certificate)["valid"])
        damaged = copy.deepcopy(certificate)
        damaged["reference_selected_token_id"] += 1
        self.assertFalse(verify_fixed_input_certificate(damaged)["valid"])
        forged_claim = copy.deepcopy(certificate)
        forged_claim["all_boundaries_within_tolerance"] = False
        body = {key: value for key, value in forged_claim.items() if key != "certificate_sha256"}
        from bioprocess_runtime.serialization import canonical_json
        forged_claim["certificate_sha256"] = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
        verification = verify_fixed_input_certificate(forged_claim)
        self.assertFalse(verification["valid"])
        self.assertFalse(verification["claims_consistent"])

    def test_operator_equations_match_fixed_vector_implementations(self) -> None:
        from bioprocess_runtime.reference_gemma import operator_equation_conformance

        report = operator_equation_conformance()
        self.assertLess(report["maximum_absolute_error"], 1e-6)
        self.assertEqual({case["operator"] for case in report["cases"]}, {"linear", "rms_norm", "gelu_tanh", "stable_softmax"})
        if self.torch.cuda.is_available():
            self.assertTrue(all("cuda" in case for case in report["cases"]))

    def test_coordinate_registry_separates_structural_and_domain_meaning(self) -> None:
        from bioprocess_runtime.reference_gemma import semantic_coordinate_registry

        tokenizer = type("Tokenizer", (), {})()
        registry = semantic_coordinate_registry(self.model, tokenizer)
        self.assertEqual(registry["dimensions"]["hidden_coordinates"], 16)
        self.assertIn("tensor role", registry["semantic_boundary"]["exact_by_construction"])
        self.assertIn(
            "biological concept assigned to a hidden coordinate",
            registry["semantic_boundary"]["requires_empirical_or_formal_domain_evidence"],
        )

    def test_summary_reports_eager_equivalence_and_deployed_path(self) -> None:
        from bioprocess_runtime.reference_gemma import (
            fixed_input_equivalence_certificate,
            operator_equation_conformance,
            semantic_coordinate_registry,
            summarize_reference_evidence,
        )

        certificate = fixed_input_equivalence_certificate(self.model, self.input_ids, absolute_tolerance=0.0)
        registry = semantic_coordinate_registry(self.model, type("Tokenizer", (), {})())
        summary = summarize_reference_evidence(certificate, operator_equation_conformance(), registry)
        self.assertTrue(summary["reference_vs_huggingface_eager"]["all_boundaries_exact"])
        self.assertTrue(summary["reference_vs_deployed_path"]["selected_token_matches_reference"])

    def test_bounded_domain_certificate_exhausts_declared_grid(self) -> None:
        from bioprocess_runtime.reference_gemma import (
            bounded_domain_equivalence_certificate,
            recompute_bounded_domain_certificate,
            summarize_bounded_domain,
            verify_bounded_domain_certificate,
        )

        torch = self.torch

        class GridTokenizer:
            chat_template = None

            def __call__(self, prompt, return_tensors="pt"):
                checksum = sum(prompt.encode("utf-8")) % 29 + 3
                ids = torch.tensor([[2, checksum]], dtype=torch.long)
                return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

        tokenizer = GridTokenizer()
        certificate = bounded_domain_equivalence_certificate(
            self.model,
            tokenizer,
            oxygen_values=(25.0, 45.0),
            slope_values=(-1.0, 0.0),
            sensor_agreement_values=(False, True),
        )
        verification = verify_bounded_domain_certificate(certificate)
        self.assertTrue(verification["valid"])
        self.assertTrue(recompute_bounded_domain_certificate(self.model, tokenizer, certificate)["valid"])
        self.assertEqual(verification["verified_state_count"], 8)
        self.assertEqual(certificate["summary"]["reference_eager_exact_states"], 8)
        summary = summarize_bounded_domain(certificate)
        self.assertEqual(sum(summary["reference_output_token_counts"].values()), 8)
        incomplete = copy.deepcopy(certificate)
        incomplete["states"].pop()
        self.assertFalse(verify_bounded_domain_certificate(incomplete)["valid"])
        with self.assertRaises(ValueError):
            bounded_domain_equivalence_certificate(
                self.model,
                GridTokenizer(),
                oxygen_values=(25.0, 25.0),
                slope_values=(0.0,),
            )

    def test_sliding_mask_excludes_distant_and_future_positions(self) -> None:
        from bioprocess_runtime.reference_gemma import reference_causal_mask

        mask = reference_causal_mask(5, self.torch.float32, self.torch.device("cpu"), sliding_window=2)[0, 0]
        self.assertEqual(float(mask[4, 4]), 0.0)
        self.assertEqual(float(mask[4, 3]), 0.0)
        self.assertLess(float(mask[4, 2]), -1e30)
        self.assertLess(float(mask[1, 2]), -1e30)


if __name__ == "__main__":
    unittest.main()
