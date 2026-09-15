from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bioprocess_runtime.gemma_post_attention import predict_post_attention, post_instructions, post_report, _code_sha

ROOT = Path(__file__).resolve().parent.parent


class PostAttentionTests(unittest.TestCase):
    def test_independent_rms_and_residual_arithmetic(self) -> None:
        class SyntheticRoot:
            def predict_bits(self, value, runtime):
                return 0x3F800000

        projected = np.full((1, 30, 640), 0x3F80, dtype=np.uint16)
        residual = np.full_like(projected, 0x3F00)
        result = predict_post_attention(projected, residual, [0] * 640, 1e-6, SyntheticRoot(), {})
        self.assertTrue(np.all(np.asarray(result["normalized_bits"]) == 0x3F80))
        self.assertTrue(np.all(np.asarray(result["residual_bits"]) == 0x3FC0))
        self.assertEqual(len(result["stages"]["mean_bits"]), 30)
        with self.assertRaises(ValueError):
            predict_post_attention(projected[..., :639], residual, [0] * 640, 1e-6, SyntheticRoot(), {})

    def test_ir_stops_before_pre_feedforward_norm(self) -> None:
        program = json.loads((ROOT / "results/gemma3_270m_execution_ir.json").read_text(encoding="utf-8"))
        instructions = post_instructions(program)
        self.assertEqual([item["opcode"] for item in instructions], ["RMS_NORM", "ADD"])
        self.assertEqual(instructions[1]["inputs"], ["hidden.0", "layer.0.attention.post_normalized"])
        changed = copy.deepcopy(program)
        next(item for item in changed["instructions"] if item["outputs"] == ["layer.0.post_attention_residual"])["inputs"][0] = "layer.0.attention.projected"
        with self.assertRaises(ValueError):
            post_instructions(changed)

    def test_scalar_discrepancy_is_not_hidden_by_bfloat_outputs(self) -> None:
        values = np.zeros((1, 30, 640), dtype=np.uint16)
        stages = {name: [0x3F800000] * 30 for name in ("mean_bits", "denominator_bits", "rsqrt_bits")}
        prediction = {"normalized_bits": values.tolist(), "residual_bits": values.tolist(), "stages": stages}
        record = {"projected": values.tolist(), "residual_base": values.tolist(), "normalized": values.tolist(), "residual": values.tolist(),
                  "stages": copy.deepcopy(stages), "cuda_events": ["synthetic"]}
        record["stages"]["rsqrt_bits"][0] ^= 1
        record["stages"]["mean_input_metadata"] = {"input_shape": [1, 30, 640], "input_strides": [19200, 640, 1], "alignment_mod16": 0, "input_dtype": "torch.float32", "axes": [-1], "keepdim": True}
        report = post_report({"plan_sha256": "a" * 64, "runtime": {}}, values, values, prediction, [{"traced": record, "untraced": record}] * 3, {})
        self.assertFalse(report["post_attention_matches"])
        self.assertEqual(report["mismatch_counts"]["rsqrt_bits"], [1, 1, 1])
        self.assertEqual(report["mismatch_counts"]["normalized"], [0, 0, 0])
        self.assertEqual(report["first_divergence"]["stage"], "rsqrt_bits")
        self.assertFalse(report["mlp_executed"])

    def test_recorded_post_attention_boundary_is_exact_and_bounded(self) -> None:
        from bioprocess_runtime.gemma_rotary_slice import _sha

        summary = json.loads((ROOT / "results/gemma3_270m_post_attention_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["summary_sha256"], _sha({key: value for key, value in summary.items() if key != "summary_sha256"}))
        self.assertTrue(summary["post_attention_matches"])
        self.assertEqual(summary["value_count"], 38400)
        self.assertEqual(summary["scalar_stage_positions"], 90)
        self.assertFalse(summary["prefix_independently_recomputed"])
        self.assertFalse(summary["pre_feedforward_norm_executed"])
        self.assertFalse(summary["mlp_executed"])
        self.assertFalse(summary["global_exactness_activation_allowed"])
        for counts in summary["mismatch_counts"].values():
            self.assertEqual(counts, [0, 0, 0])
        for hashes in summary["observed_output_hashes"].values():
            self.assertEqual(len(set(hashes)), 1)

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"), "Gemma dependencies are unavailable")
    def test_actual_evidence_rejects_scalar_and_geometry_tampering(self) -> None:
        from bioprocess_runtime.cli import build_parser, _load_post_attention_sources
        from bioprocess_runtime.gemma_post_attention import verify_post
        from bioprocess_runtime.gemma_rotary_slice import _sha, _seal

        required = ("exp_table.bin", "post_attention_report.json", "post_attention_predictions.json", "k128_dense_inputs.json", "k128_dense_report.json")
        if any(not (ROOT / "artifacts" / ("gemma3_270m_" + name)).exists() for name in required):
            self.skipTest("Private post-attention/source artifacts intentionally remain outside Git")
        args = build_parser().parse_args(["gemma-post-attention-verify", "results/gemma3_270m_execution_ir.json", "--bundle", "artifacts/gemma3_270m_post_attention_predictions.json", "--report", "artifacts/gemma3_270m_post_attention_report.json"])
        for name, value in vars(args).items():
            if isinstance(value, Path) and not value.is_absolute():
                setattr(args, name, ROOT / value)
        load = lambda path: json.loads(path.read_text(encoding="utf-8"))
        sources = _load_post_attention_sources(args)
        plan, bundle, report = load(args.plan), load(args.bundle), load(args.report)
        self.assertTrue(verify_post(sources, plan, bundle, report)["valid"])
        changed_plan, changed_bundle = copy.deepcopy(plan), copy.deepcopy(bundle)
        changed_bundle["predictions"]["stages"]["rsqrt_bits"][0] ^= 1
        changed_plan["bundle_sha256"] = _sha(changed_bundle)
        changed_plan["scalar_stages_sha256"] = _sha(changed_bundle["predictions"]["stages"])
        changed_plan = _seal({key: value for key, value in changed_plan.items() if key != "plan_sha256"}, "plan_sha256")
        self.assertFalse(verify_post(sources, changed_plan, changed_bundle, report)["valid"])
        altered = copy.deepcopy(report)
        altered["observations"][0]["traced"]["stages"]["mean_input_metadata"]["input_strides"][1] += 1
        altered = _seal({key: value for key, value in altered.items() if key != "report_sha256"}, "report_sha256")
        self.assertFalse(verify_post(sources, plan, bundle, altered)["valid"])

    def test_cli_output_and_source_guards(self) -> None:
        from bioprocess_runtime.cli import build_parser, command_post_attention

        parser = build_parser()
        for operation in ("plan", "run", "verify"):
            argv = ["gemma-post-attention-" + operation, "program", "--bundle", "bundle"]
            argv += ["--report", "report", "--reexecute"] if operation == "verify" else ["--output", "output"]
            args = parser.parse_args(argv)
            self.assertEqual(args.operation, operation)
            self.assertEqual(args.dense_plan.name, "gemma3_270m_k128_dense_plan.json")
            self.assertEqual(args.survivor_plan.name, "gemma3_270m_output_survivor_plan.json")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "source.json"
            output.write_text("preserve", encoding="utf-8")
            with self.assertRaises(ValueError):
                command_post_attention(Namespace(operation="plan", output=output, bundle=Path(directory) / "bundle"))
            self.assertEqual(output.read_text(encoding="utf-8"), "preserve")
        with patch("bioprocess_runtime.gemma_post_attention.Path.read_bytes", return_value=b"changed"):
            with self.assertRaises(ValueError):
                _code_sha()
