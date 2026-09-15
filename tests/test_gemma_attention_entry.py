from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bioprocess_runtime.gemma_attention_entry import entry_instructions, _state_array, TARGETS

ROOT = Path(__file__).resolve().parent.parent


class AttentionEntryTests(unittest.TestCase):
    def test_exact_dependency_slice_excludes_rope_and_attention(self) -> None:
        program = json.loads((ROOT / "results/gemma3_270m_execution_ir.json").read_text(encoding="utf-8"))
        instructions = entry_instructions(program)
        self.assertEqual(len(instructions), 11)
        self.assertEqual([item["opcode"] for item in instructions].count("RMS_NORM"), 3)
        self.assertEqual([item["opcode"] for item in instructions].count("LINEAR"), 3)
        self.assertTrue(set(TARGETS).issubset({item["outputs"][0] for item in instructions}))
        self.assertFalse(any(item["opcode"] in ("ROTARY_TABLE", "ROTARY_APPLY_PAIR", "MATMUL_QK", "SOFTMAX") for item in instructions))
        damaged = copy.deepcopy(program)
        next(item for item in damaged["instructions"] if item["outputs"] == ["layer.0.key.flat"])["parameter_refs"] = ["model.layers.0.self_attn.v_proj.weight"]
        with self.assertRaises(ValueError):
            entry_instructions(damaged)

    def test_checked_summary_records_connected_original_forward_agreement(self) -> None:
        from bioprocess_runtime.serialization import canonical_json
        import hashlib

        plan = json.loads((ROOT / "results/gemma3_270m_attention_entry_plan.json").read_text(encoding="utf-8"))
        summary = json.loads((ROOT / "results/gemma3_270m_attention_entry_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["summary_sha256"], hashlib.sha256(canonical_json({key: value for key, value in summary.items() if key != "summary_sha256"}).encode("utf-8")).hexdigest())
        self.assertEqual(summary["plan_sha256"], plan["plan_sha256"])
        self.assertEqual(summary["independent_instruction_count"], 11)
        self.assertEqual(summary["target_value_count"], 46080)
        self.assertTrue(summary["connected_slice_bit_exact"])
        self.assertTrue(summary["original_forward_stopped_before_rope_rotation"])
        self.assertFalse(summary["uses_framework_floating_arithmetic_in_slice"])
        self.assertTrue(summary["uses_empirical_rsqrt_specification"])
        self.assertFalse(summary["full_first_layer_qualified"])
        self.assertFalse(summary["global_exactness_activation_allowed"])
        self.assertEqual(summary["derived_value_head_hashes"], [summary["tensor_summaries"]["layer.0.value.heads"]["prediction_sha256"]] * 3)
        for name, hashes in summary["original_boundary_hashes"].items():
            self.assertEqual(hashes, [summary["tensor_summaries"][name]["prediction_sha256"]] * 3)

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"), "Gemma optional dependencies are unavailable")
    def test_artifact_integrity_rejects_changed_values_and_unbound_providers(self) -> None:
        import hashlib
        from bioprocess_runtime.gemma_attention_entry import verify_attention_entry
        from bioprocess_runtime.gemma_rsqrt_lookup import CheckedRsqrtLookup
        from bioprocess_runtime.operational_semantics import append_chain_record
        from bioprocess_runtime.serialization import canonical_json

        private = ("artifacts/gemma3_270m_attention_entry_predictions.json", "artifacts/gemma3_270m_attention_entry_report.json", "artifacts/gemma3_270m_rsqrt_table.bin")
        if any(not (ROOT / path).exists() for path in private):
            self.skipTest("Tensor-rich entry and lookup artifacts are intentionally outside Git")
        load = lambda path: json.loads((ROOT / path).read_text(encoding="utf-8"))
        lookup = CheckedRsqrtLookup(load("results/gemma3_270m_rsqrt_table_plan.json"), load("results/gemma3_270m_rsqrt_table_manifest.json"), ROOT / private[2], load("results/gemma3_270m_rsqrt_domain_plan.json"), load("results/gemma3_270m_rsqrt_domain_report.json"), ROOT / "artifacts/rsqrt_mismatches")
        program, plan = load("results/gemma3_270m_execution_ir.json"), load("results/gemma3_270m_attention_entry_plan.json")
        bundle, report = load(private[0]), load(private[1])
        self.assertTrue(verify_attention_entry(program, plan, bundle, report, lookup)["valid"])
        damaged = copy.deepcopy(report)
        damaged["observations"][0]["layer.0.key.normalized"][0][0][29][255] ^= 1
        damaged["report_sha256"] = hashlib.sha256(canonical_json({k: v for k, v in damaged.items() if k != "report_sha256"}).encode("utf-8")).hexdigest()
        self.assertFalse(verify_attention_entry(program, plan, bundle, damaged, lookup)["valid"])
        damaged_plan, records = copy.deepcopy(plan), []
        for record in damaged_plan["records"]:
            payload = record["payload"]
            if payload["output_name"] == "layer.0.query.flat":
                payload["provider"] = "unrecorded_override"
            append_chain_record(records, payload)
        damaged_plan["records"], damaged_plan["execution_root"] = records, records[-1]["record_hash"]
        damaged_plan["plan_sha256"] = hashlib.sha256(canonical_json({k: v for k, v in damaged_plan.items() if k != "plan_sha256"}).encode("utf-8")).hexdigest()
        self.assertFalse(verify_attention_entry(program, damaged_plan, bundle, report, lookup)["valid"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is unavailable")
    def test_certificate_rejects_unregistered_lookup(self) -> None:
        from bioprocess_runtime.gemma_attention_entry import build_attention_entry_plan

        with self.assertRaises(ValueError):
            build_attention_entry_plan({}, None, {}, object())

    def test_cli_preserves_existing_outputs(self) -> None:
        from argparse import Namespace
        from bioprocess_runtime.cli import build_parser, command_attention_entry

        parser = build_parser()
        args = parser.parse_args(["gemma-attention-entry-verify", "program", "fixture", "plan", "report", "--bundle", "bundle", "--reexecute"])
        self.assertEqual(args.operation, "verify")
        self.assertTrue(args.reexecute)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing.json"
            output.write_text("preserve", encoding="utf-8")
            with self.assertRaises(ValueError):
                command_attention_entry(Namespace(operation="plan", output=output, bundle=Path(directory) / "bundle", summary=None))
            self.assertEqual(output.read_text(encoding="utf-8"), "preserve")

    def test_bit_tensor_input_validation(self) -> None:
        self.assertEqual(_state_array([[[0, 65535]]], [1, 1, 2]).tolist(), [[[0, 65535]]])
        for value in ([[[True, 0]]], [[[1.0, 0]]], [[[0]]], [[[65536, 0]]]):
            with self.assertRaises(ValueError):
                _state_array(value, [1, 1, 2])

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"), "Gemma optional dependencies are unavailable")
    def test_connected_dispatch_never_calls_framework_floating_operators(self) -> None:
        import torch
        from transformers import Gemma3ForCausalLM, Gemma3TextConfig
        from bioprocess_runtime.gemma_attention_entry import execute_attention_entry
        from bioprocess_runtime.gemma_ir import compile_gemma_ir
        from bioprocess_runtime.gemma_ir_interpreter import bind_model_tensors
        from bioprocess_runtime.operational_semantics import build_architecture_manifest, verify_trace_chain
        from bioprocess_runtime.reference_gemma import rms_root_bits

        config = Gemma3TextConfig(vocab_size=32, hidden_size=640, intermediate_size=16, num_hidden_layers=1,
                                 num_attention_heads=4, num_key_value_heads=1, head_dim=256, max_position_embeddings=64,
                                 sliding_window=32, layer_types=["sliding_attention"])
        model = Gemma3ForCausalLM(config).to(dtype=torch.bfloat16).eval()
        tokenizer = type("Tokenizer", (), {"vocab_size": 32, "bos_token_id": 2, "eos_token_id": 1, "pad_token_id": 0})()
        with tempfile.TemporaryDirectory() as directory:
            program = compile_gemma_ir(build_architecture_manifest(model, tokenizer, Path(directory)))
        parameters = bind_model_tensors(program, model)

        class SyntheticLookup:
            def predict_bits(self, bits, runtime):
                return rms_root_bits(bits, "rsqrt_rne")

        failure = AssertionError("Framework floating arithmetic entered the independent slice")
        with patch("torch.nn.functional.linear", side_effect=failure), patch("torch.nn.functional.embedding", side_effect=failure), patch("torch.rsqrt", side_effect=failure), patch.object(torch.Tensor, "mean", side_effect=failure), patch.object(torch.Tensor, "rsqrt", side_effect=failure), patch("bioprocess_runtime.gemma_wmma_candidate.operand_aligned_product_bits", return_value=0x3F80), patch("bioprocess_runtime.gemma_wmma_candidate.split_k_candidate_bits", return_value=0x3F80):
            states, records, stages = execute_attention_entry(program, parameters, [[2] * 30], SyntheticLookup(), {})
        self.assertEqual(states["layer.0.query.normalized"].shape, (1, 4, 30, 256))
        self.assertEqual(states["layer.0.key.normalized"].shape, (1, 1, 30, 256))
        self.assertEqual(states["layer.0.value.heads"].shape, (1, 1, 30, 256))
        self.assertEqual(len(records), 11)
        self.assertEqual(len(stages), 3)
        self.assertTrue(verify_trace_chain(records)["valid"])
        self.assertFalse(any("shared" in record["payload"]["provider"] for record in records))
