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

from bioprocess_runtime.gemma_attention_scores import causal_mask_bits, scale_score_bits, mask_score_bits, predict_score_bits, score_instructions, score_report

ROOT = Path(__file__).resolve().parent.parent


class AttentionScoreTests(unittest.TestCase):
    def test_causal_mask_and_separate_scaling(self) -> None:
        raw = np.full((1, 4, 30, 30), 0x4380, dtype=np.uint16)
        scaled = scale_score_bits(raw)
        self.assertTrue(np.all(scaled == 0x4180))
        mask = causal_mask_bits()
        result = mask_score_bits(scaled, mask)
        for row in range(30):
            self.assertTrue(np.all(result[0, :, row, :row + 1] == 0x4180))
            self.assertTrue(np.all(result[0, :, row, row + 1:] == 0xFF7F))

    def test_score_instruction_scope(self) -> None:
        program = json.loads((ROOT / "results/gemma3_270m_execution_ir.json").read_text(encoding="utf-8"))
        instructions = score_instructions(program)
        self.assertEqual([item["opcode"] for item in instructions], ["REPEAT_KV", "MATMUL_QK", "SCALE", "CAUSAL_MASK", "ADD"])
        changed = copy.deepcopy(program)
        next(item for item in changed["instructions"] if item["outputs"] == ["layer.0.attention.scaled_scores"])["attributes"]["scalar"] = 0.125
        with self.assertRaises(ValueError):
            score_instructions(changed)

    def test_shape_rejection_and_frozen_dot_dispatch(self) -> None:
        query = np.full((1, 4, 30, 256), 0x3F80, dtype=np.uint16)
        key = np.full((1, 1, 30, 256), 0x3F80, dtype=np.uint16)
        with patch("bioprocess_runtime.gemma_attention_scores.operand_aligned_product_bits", return_value=0x4380) as dot:
            result = predict_score_bits(query, key)
        self.assertEqual(dot.call_count, 3600)
        self.assertTrue(np.all(result["unscaled"] == 0x4380))
        with self.assertRaises(ValueError):
            predict_score_bits(query[:, :, :29], key)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is unavailable")
    def test_masked_agreement_cannot_hide_raw_score_failure(self) -> None:
        from bioprocess_runtime.gemma_attention_entry import _descriptor
        from bioprocess_runtime.gemma_rotary_slice import ROTATED

        states = {ROTATED[0]: np.zeros((1, 4, 30, 256), dtype=np.uint16), ROTATED[1]: np.zeros((1, 1, 30, 256), dtype=np.uint16), "layer.0.value.heads": np.zeros((1, 1, 30, 256), dtype=np.uint16)}
        raw = np.zeros((1, 4, 30, 30), dtype=np.uint16)
        mask = causal_mask_bits()
        predicted = {"unscaled": raw, "scaled": scale_score_bits(raw), "mask": mask, "masked": mask_score_bits(scale_score_bits(raw), mask)}
        observed_raw = raw.copy()
        observed_raw[0, 0, 0, 1] = 0x3F80
        scaled = scale_score_bits(observed_raw)
        masked = mask_score_bits(scaled, mask)
        left = states[ROTATED[0]].reshape(4, 30, 256)
        right = np.repeat(states[ROTATED[1]], 4, axis=1).reshape(4, 30, 256).transpose(0, 2, 1)
        geometry = [{"tensor": {**_descriptor(value), "device": "cuda:0"}, "strides": [stride // 2 for stride in value.strides]} for value in (left, right)]
        inputs = {name: value.tolist() for name, value in states.items()}
        traced = {"inputs": inputs, "unscaled": observed_raw.tolist(), "scaled": scaled.tolist(), "masked": masked.tolist(), "mask": mask.tolist(),
                  "bmm_inputs": {"left": left.tolist(), "right": right.tolist()}, "bmm_geometry": geometry, "bmm_cuda_events": ["synthetic"],
                  "softmax_input_f32": (masked.astype(np.uint32) << 16).tolist()}
        observation = {"traced": traced, "untraced": {"inputs": inputs, "mask": mask.tolist(), "masked": masked.tolist()}}
        report = score_report({"plan_sha256": "a" * 64, "runtime": {}}, states, predicted, [observation] * 3, {})
        self.assertFalse(report["candidate_passes"])
        self.assertEqual(report["stage_mismatch_counts"]["unscaled"], [1, 1, 1])
        self.assertEqual(report["stage_mismatch_counts"]["masked"], [0, 0, 0])
        self.assertEqual(report["first_divergence"]["stage"], "unscaled")
        self.assertTrue(all(item["scale_rule_on_observed_input_matches"] and item["mask_rule_on_observed_input_matches"] for item in report["conditions"]))

    def test_source_byte_drift_is_rejected(self) -> None:
        from bioprocess_runtime.gemma_attention_scores import _code_sha

        with patch("bioprocess_runtime.gemma_attention_scores.Path.read_bytes", return_value=b"changed source"):
            with self.assertRaisesRegex(ValueError, "source changed after process import"):
                _code_sha()

    def test_recorded_score_summary_stays_bounded(self) -> None:
        from bioprocess_runtime.gemma_rotary_slice import _sha

        summary = json.loads((ROOT / "results/gemma3_270m_attention_scores_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["summary_sha256"], _sha({key: value for key, value in summary.items() if key != "summary_sha256"}))
        self.assertTrue(summary["candidate_passes"])
        self.assertEqual(summary["score_count"], 3600)
        self.assertFalse(summary["softmax_executed"])
        self.assertFalse(summary["prefix_recomputed_in_this_experiment"])
        self.assertFalse(summary["global_exactness_activation_allowed"])
        for counts in summary["stage_mismatch_counts"].values():
            self.assertEqual(counts, [0, 0, 0])
        for checks in summary["conditions"]:
            self.assertTrue(all(checks.values()))

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"), "Gemma dependencies are unavailable")
    def test_real_score_evidence_rejects_value_and_profile_tampering(self) -> None:
        from bioprocess_runtime.gemma_attention_scores import ScoreSources, verify_scores
        from bioprocess_runtime.gemma_rsqrt_lookup import CheckedRsqrtLookup
        from bioprocess_runtime.gemma_rotary_slice import _seal

        private = ("artifacts/gemma3_270m_attention_scores_report.json", "artifacts/gemma3_270m_attention_scores_predictions.json", "artifacts/gemma3_270m_rsqrt_table.bin", "artifacts/gemma3_270m_rotary_table.json", "artifacts/gemma3_270m_rotary_slice_predictions_v2.json", "artifacts/gemma3_270m_rotary_slice_report.json")
        if any(not (ROOT / name).exists() for name in private):
            self.skipTest("Tensor-rich score/source artifacts intentionally remain outside Git")
        load = lambda path: json.loads((ROOT / path).read_text(encoding="utf-8"))
        lookup = CheckedRsqrtLookup(load("results/gemma3_270m_rsqrt_table_plan.json"), load("results/gemma3_270m_rsqrt_table_manifest.json"), ROOT / private[2], load("results/gemma3_270m_rsqrt_domain_plan.json"), load("results/gemma3_270m_rsqrt_domain_report.json"), ROOT / "artifacts/rsqrt_mismatches")
        sources = ScoreSources(load("results/gemma3_270m_execution_ir.json"), load("results/gemma3_270m_rotary_slice_plan_v2.json"), load(private[4]), load(private[5]), lookup, load("results/gemma3_270m_rotary_table_plan.json"), load("results/gemma3_270m_rotary_table_manifest.json"), load(private[3]))
        plan, bundle, report = load("results/gemma3_270m_attention_scores_plan.json"), load(private[1]), load(private[0])
        self.assertTrue(verify_scores(sources, plan, bundle, report)["valid"])
        damaged = copy.deepcopy(report)
        damaged["observations"][0]["traced"]["unscaled"][0][0][0][0] ^= 1
        damaged = _seal({key: value for key, value in damaged.items() if key != "report_sha256"}, "report_sha256")
        self.assertFalse(verify_scores(sources, plan, bundle, damaged)["valid"])
        changed = copy.deepcopy(plan)
        changed["profile"]["dot"]["precision_bits"] = 24
        changed = _seal({key: value for key, value in changed.items() if key != "plan_sha256"}, "plan_sha256")
        self.assertFalse(verify_scores(sources, changed, bundle, report)["valid"])

    def test_cli_parser_and_existing_output_protection(self) -> None:
        from bioprocess_runtime.cli import build_parser, command_attention_scores

        parser = build_parser()
        args = parser.parse_args(["gemma-attention-scores-verify", "program", "--bundle", "bundle", "--report", "report", "--reexecute"])
        self.assertEqual(args.operation, "verify")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "source.json"
            output.write_text("preserve", encoding="utf-8")
            with self.assertRaises(ValueError):
                command_attention_scores(Namespace(operation="plan", output=output, bundle=Path(directory) / "bundle"))
            self.assertEqual(output.read_text(encoding="utf-8"), "preserve")
