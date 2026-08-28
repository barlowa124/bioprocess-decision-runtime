from __future__ import annotations

import copy
import hashlib
import importlib.util
import unittest


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

    def test_reference_matches_eager_beyond_sliding_window(self) -> None:
        from bioprocess_runtime.reference_gemma import fixed_input_equivalence_certificate

        long_input = (self.torch.arange(12, dtype=self.torch.long) % 30).unsqueeze(0)
        certificate = fixed_input_equivalence_certificate(self.model, long_input, absolute_tolerance=0.0)
        self.assertTrue(certificate["all_boundaries_within_tolerance"])

    def test_reference_trace_names_intermediate_computations(self) -> None:
        from bioprocess_runtime.reference_gemma import reference_gemma_forward

        output = reference_gemma_forward(self.model, self.input_ids)
        stages = {record["payload"]["stage"] for record in output.records}
        self.assertTrue({"query", "key", "value", "attention_probability", "mlp_gate", "mlp_intermediate_product", "vocabulary_logits"}.issubset(stages))

    def test_certificate_verifier_detects_changes(self) -> None:
        from bioprocess_runtime.reference_gemma import fixed_input_equivalence_certificate, verify_fixed_input_certificate

        certificate = fixed_input_equivalence_certificate(self.model, self.input_ids, absolute_tolerance=0.0)
        self.assertTrue(verify_fixed_input_certificate(certificate)["valid"])
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

    def test_sliding_mask_excludes_distant_and_future_positions(self) -> None:
        from bioprocess_runtime.reference_gemma import reference_causal_mask

        mask = reference_causal_mask(5, self.torch.float32, self.torch.device("cpu"), sliding_window=2)[0, 0]
        self.assertEqual(float(mask[4, 4]), 0.0)
        self.assertEqual(float(mask[4, 3]), 0.0)
        self.assertLess(float(mask[4, 2]), -1e30)
        self.assertLess(float(mask[1, 2]), -1e30)


if __name__ == "__main__":
    unittest.main()
