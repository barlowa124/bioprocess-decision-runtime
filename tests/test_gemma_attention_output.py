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

from bioprocess_runtime.gemma_attention_output import predict_value_aggregation, concatenate_heads, predict_output_projection, output_instructions, output_report

ROOT = Path(__file__).resolve().parent.parent


class AttentionOutputTests(unittest.TestCase):
    def test_explicit_zero_padding_and_projection_dispatch(self) -> None:
        probability = np.full((1, 4, 30, 30), 0x3F80, dtype=np.uint16)
        values = np.full((1, 1, 30, 256), 0x4000, dtype=np.uint16)
        with patch("bioprocess_runtime.gemma_attention_output.operand_aligned_product_bits", return_value=0x4270) as dot:
            result = predict_value_aggregation(probability, values)
            self.assertEqual(dot.call_count, 30720)
            self.assertEqual(dot.call_args_list[0].args, ([0x3F80] * 30 + [0, 0], [0x4000] * 30 + [0, 0]))
        self.assertEqual(result.shape, (1, 4, 30, 256))
        inputs = concatenate_heads(result)
        with patch("bioprocess_runtime.gemma_attention_output.operand_aligned_product_bits", return_value=0) as dot:
            projected = predict_output_projection(inputs, np.zeros((640, 1024), dtype=np.uint16))
            self.assertEqual(dot.call_count, 19200)
            self.assertEqual(len(dot.call_args_list[0].args[0]), 1024)
        self.assertEqual(projected.shape, (1, 30, 640))
        with self.assertRaises(ValueError):
            predict_value_aggregation(probability[:, :, :29], values)

    def test_head_coordinate_order(self) -> None:
        heads = np.arange(30720, dtype=np.uint16).reshape(1, 4, 30, 256)
        actual = concatenate_heads(heads)
        for head, row, dimension in ((0, 0, 0), (3, 29, 255), (2, 7, 31)):
            self.assertEqual(int(actual[0, row, head * 256 + dimension]), int(heads[0, head, row, dimension]))
        program = json.loads((ROOT / "results/gemma3_270m_execution_ir.json").read_text(encoding="utf-8"))
        self.assertEqual([item["opcode"] for item in output_instructions(program)], ["REPEAT_KV", "MATMUL_AV", "TRANSPOSE_RESHAPE_HEADS", "LINEAR"])
        damaged = copy.deepcopy(program)
        next(item for item in damaged["instructions"] if item["outputs"] == ["layer.0.attention.projected"])["parameter_refs"] = ["model.layers.0.self_attn.q_proj.weight"]
        with self.assertRaises(ValueError):
            output_instructions(damaged)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is unavailable")
    def test_projection_agreement_cannot_hide_aggregation_error(self) -> None:
        from bioprocess_runtime.gemma_attention_entry import _descriptor

        probability, values = np.zeros((1, 4, 30, 30), dtype=np.uint16), np.zeros((1, 1, 30, 256), dtype=np.uint16)
        predicted = {"head_output": np.zeros((1, 4, 30, 256), dtype=np.uint16), "concatenated": np.zeros((1, 30, 1024), dtype=np.uint16), "projected": np.zeros((1, 30, 640), dtype=np.uint16)}
        actual_head = predicted["head_output"].copy()
        actual_head[0, 0, 0, 0] = 0x3F80
        concatenated = concatenate_heads(actual_head)
        left, right = probability.reshape(4, 30, 30), np.repeat(values, 4, axis=1).reshape(4, 30, 256)
        weights = np.zeros((640, 1024), dtype=np.uint16)
        geometry = lambda tensor: {"tensor": {**_descriptor(tensor), "device": "cuda:0"}, "strides": [value // 2 for value in tensor.strides]}
        record = {"head_output": actual_head.tolist(), "concatenated": concatenated.tolist(), "projected": predicted["projected"].tolist(),
                  "value_input": values.tolist(), "probability": probability.tolist(), "eager_output": actual_head.transpose(0, 2, 1, 3).tolist(),
                  "pv_operands": {"probability": left.tolist(), "value": right.tolist()}, "pv_geometry": [geometry(left), geometry(right)],
                  "projection_geometry": [geometry(concatenated), geometry(weights)], "pv_cuda_events": ["synthetic"], "projection_cuda_events": ["synthetic"]}
        report = output_report({"plan_sha256": "a" * 64, "runtime": {}, "weight": _descriptor(weights)}, probability, values, predicted, [{"traced": record, "untraced": record}] * 3, {})
        self.assertFalse(report["candidate_passes"])
        self.assertEqual(report["mismatch_counts"]["projected"], [0] * 3)
        self.assertEqual(report["mismatch_counts"]["head_output"], [1] * 3)
        self.assertEqual(report["first_divergence"]["stage"], "head_output")
        self.assertTrue(all(not item["projection_input_matches_prediction"] for item in report["checks"]))

    def test_separate_k1024_split_recipe_and_prior_k640_guard(self) -> None:
        from bioprocess_runtime.gemma_output_split import k1024_split_bits
        from bioprocess_runtime.gemma_wmma_candidate import split_k_candidate_bits, DENSE_SPLIT_PROFILE

        self.assertEqual(k1024_split_bits([0x3F80] * 1024, [0x3F80] * 1024), 0x4480)
        with self.assertRaises(ValueError):
            k1024_split_bits([0x3F80] * 640, [0x3F80] * 640)
        with self.assertRaises(ValueError):
            split_k_candidate_bits([0x3F80] * 1024, [0x3F80] * 1024, *DENSE_SPLIT_PROFILE)
        from bioprocess_runtime.cli import build_parser
        args = build_parser().parse_args(["gemma-attention-output-split-verify", "program", "--bundle", "bundle", "--report", "report", "--reexecute"])
        self.assertEqual(args.base_plan.name, "gemma3_270m_attention_output_plan.json")
        self.assertEqual(args.plan.name, "gemma3_270m_output_split_plan.json")

    def test_recorded_aggregation_pass_and_projection_failures(self) -> None:
        from bioprocess_runtime.gemma_rotary_slice import _sha

        load = lambda path: json.loads((ROOT / path).read_text(encoding="utf-8"))
        base = load("results/gemma3_270m_attention_output_summary.json")
        split = load("results/gemma3_270m_output_split_summary.json")
        for summary in (base, split):
            self.assertEqual(summary["summary_sha256"], _sha({key: value for key, value in summary.items() if key != "summary_sha256"}))
            self.assertFalse(summary["global_exactness_activation_allowed"])
        self.assertEqual(base["mismatch_counts"]["head_output"], [0] * 3)
        self.assertEqual(base["mismatch_counts"]["concatenated"], [0] * 3)
        self.assertEqual(base["mismatch_counts"]["projected"], [6013] * 3)
        self.assertFalse(base["candidate_passes"])
        self.assertFalse(split["split_projection_matches_case"])
        self.assertFalse(split["fresh_holdout_validation"])
        self.assertFalse(split["hardware_partitioning_established"])
        self.assertTrue(split["selected_after_kernel_observation"])
        self.assertEqual(split["comparison"]["mismatch_counts"]["projected"], [7446] * 3)
        self.assertEqual(split["native_report_sha256"], base["source_report_sha256"])
        for checks in base["checks"]:
            self.assertTrue(all(checks.values()))

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"), "Gemma dependencies are unavailable")
    def test_real_output_evidence_rejects_weight_and_holdout_tampering(self) -> None:
        from bioprocess_runtime.cli import build_parser, _load_output_sources
        from bioprocess_runtime.gemma_attention_output import verify_output
        from bioprocess_runtime.gemma_output_split import verify_split_output
        from bioprocess_runtime.gemma_rotary_slice import _sha, _seal

        if any(not (ROOT / "artifacts" / ("gemma3_270m_" + name)).exists() for name in ("exp_table.bin", "attention_output_predictions.json", "attention_output_report.json", "output_split_predictions.json", "output_split_report.json")):
            self.skipTest("Private source/output artifacts intentionally remain outside Git")
        args = build_parser().parse_args(["gemma-attention-output-split-verify", "results/gemma3_270m_execution_ir.json", "--bundle", "artifacts/gemma3_270m_output_split_predictions.json", "--report", "artifacts/gemma3_270m_output_split_report.json"])
        for name, value in vars(args).items():
            if isinstance(value, Path) and not value.is_absolute():
                setattr(args, name, ROOT / value)
        load = lambda path: json.loads(path.read_text(encoding="utf-8"))
        sources = _load_output_sources(args)
        base_plan, base_bundle, base_report = load(args.base_plan), load(args.base_bundle), load(args.base_report)
        plan, bundle, report = load(args.plan), load(args.bundle), load(args.report)
        self.assertTrue(verify_output(sources, base_plan, base_bundle, base_report)["valid"])
        self.assertTrue(verify_split_output(sources, base_plan, base_bundle, base_report, plan, bundle, report)["valid"])
        altered_bundle, altered_plan = copy.deepcopy(base_bundle), copy.deepcopy(base_plan)
        altered_bundle["weight_bits"][0][0] ^= 1
        altered_plan["bundle_sha256"] = _sha(altered_bundle)
        altered_plan = _seal({key: value for key, value in altered_plan.items() if key != "plan_sha256"}, "plan_sha256")
        self.assertFalse(verify_output(sources, altered_plan, altered_bundle, base_report)["valid"])
        promoted = copy.deepcopy(plan)
        promoted["fresh_holdout_validation"] = True
        promoted = _seal({key: value for key, value in promoted.items() if key != "plan_sha256"}, "plan_sha256")
        self.assertFalse(verify_split_output(sources, base_plan, base_bundle, base_report, promoted, bundle, report)["valid"])

    def test_cli_existing_output_and_source_guards(self) -> None:
        from bioprocess_runtime.cli import build_parser, command_attention_output
        from bioprocess_runtime.gemma_attention_output import _code_sha

        parser = build_parser()
        for operation in ("plan", "run", "verify"):
            argv = ["gemma-attention-output-" + operation, "program", "--bundle", "bundle"]
            argv += ["--report", "report", "--reexecute"] if operation == "verify" else ["--output", "output"]
            args = parser.parse_args(argv)
            self.assertEqual(args.operation, operation)
            self.assertEqual(args.lookup_plan.name, "gemma3_270m_softmax_lookup_plan.json")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "source.json"
            output.write_text("preserve", encoding="utf-8")
            with self.assertRaises(ValueError):
                command_attention_output(Namespace(operation="plan", output=output, bundle=Path(directory) / "bundle"))
            self.assertEqual(output.read_text(encoding="utf-8"), "preserve")
        with patch("bioprocess_runtime.gemma_attention_output.Path.read_bytes", return_value=b"changed"):
            with self.assertRaises(ValueError):
                _code_sha()
