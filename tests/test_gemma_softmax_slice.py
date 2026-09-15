from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from argparse import Namespace
from decimal import ROUND_DOWN, localcontext
from pathlib import Path
from unittest.mock import patch

from bioprocess_runtime.gemma_softmax_slice import exp_float32_rne, predict_softmax_row, softmax_report, STAGES


class SoftmaxSliceTests(unittest.TestCase):
    def test_exponential_rounding_and_domain(self) -> None:
        self.assertEqual(exp_float32_rne(0), 0x3F800000)
        self.assertEqual(exp_float32_rne(0x80000000), 0x3F800000)
        self.assertEqual(exp_float32_rne(0xBF800000), 0x3EBC5AB2)
        self.assertEqual(exp_float32_rne(0xC2C80000), 0x1B)
        self.assertEqual(exp_float32_rne(0xC3000000), 0)
        self.assertEqual(exp_float32_rne(0xFF800000), 0)
        for value in (0x3F800000, 0x7FC00000, True):
            with self.assertRaises(ValueError):
                exp_float32_rne(value)
        exp_float32_rne.cache_clear()
        with localcontext() as context:
            context.prec = 6
            context.rounding = ROUND_DOWN
            self.assertEqual(exp_float32_rne(0xBF800000), 0x3EBC5AB2)

    def test_warp_padding_and_division(self) -> None:
        row = predict_softmax_row([0x3F800000] * 30)
        self.assertEqual(row["maximum_bits"], [0x3F800000] * 30)
        self.assertEqual(row["shifted_bits"], [0] * 30)
        self.assertEqual(row["denominator_bits"], [0x41F00000] * 30)
        self.assertEqual(row["output_f32_bits"], [0x3D088889] * 30)
        self.assertEqual(row["output_bf16_bits"], [0x3D09] * 30)
        self.assertEqual(len(row["xor_sum_stages"]), 5)
        one_hot = predict_softmax_row([0] + [0xFF7F0000] * 29)
        self.assertEqual(one_hot["output_f32_bits"], [0x3F800000] + [0] * 29)
        for invalid in ([0] * 29, [0x7F800000] + [0] * 29, [False] + [0] * 29):
            with self.assertRaises(ValueError):
                predict_softmax_row(invalid)

    def test_bfloat16_agreement_does_not_hide_fp32_failure(self) -> None:
        row = predict_softmax_row([0x3F800000] * 30)
        rows = [copy.deepcopy(row) for _ in range(120)]
        input_bits = [[[[0x3F800000] * 30 for _ in range(30)] for _ in range(4)]]
        fp32 = [item["output_f32_bits"][:] for item in rows]
        fp32[0][0] += 1
        bf16 = [item["output_bf16_bits"][:] for item in rows]
        traced = {"input_bits": [[0x3F800000] * 30 for _ in range(120)], "input_strides": [3600, 900, 30, 1],
                  "output_f32_bits": fp32, "output_bf16_bits": bf16, "cuda_events": ["synthetic_softmax_warp_forwardIfffLi5ELb0ELb0E"]}
        observation = {"traced": traced, "untraced": copy.deepcopy(traced), "staged_aten": {key: [item[key] for item in rows] for key in STAGES}}
        report = softmax_report({"plan_sha256": "a" * 64, "runtime": {}}, {"rows": rows, "input_bits": input_bits}, [observation] * 3, {})
        self.assertFalse(report["candidate_passes"])
        self.assertEqual(report["mismatch_counts"]["output_f32_bits"], [1, 1, 1])
        self.assertEqual(report["mismatch_counts"]["output_bf16_bits"], [0, 0, 0])
        self.assertFalse(report["fused_internal_stages_observed"])
        self.assertTrue(report["staged_aten_is_diagnostic"])

    def test_preserved_failure_summary(self) -> None:
        from bioprocess_runtime.gemma_rotary_slice import _sha

        root = Path(__file__).resolve().parent.parent
        summary = json.loads((root / "results/gemma3_270m_softmax_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["summary_sha256"], _sha({key: value for key, value in summary.items() if key != "summary_sha256"}))
        self.assertFalse(summary["candidate_passes"])
        self.assertEqual(summary["mismatch_counts"]["output_f32_bits"], [630] * 3)
        self.assertEqual(summary["mismatch_counts"]["output_bf16_bits"], [0] * 3)
        for checks in summary["conditions"]:
            self.assertTrue(all(checks.values()))
        for diagnostic in summary["diagnostics"]:
            self.assertEqual(diagnostic["candidate_vs_staged_aten"]["exponential_bits"], 573)
            self.assertEqual(diagnostic["candidate_vs_staged_aten"]["maximum_bits"], 0)
            self.assertEqual(diagnostic["candidate_vs_staged_aten"]["shifted_bits"], 0)
            self.assertEqual(diagnostic["staged_vs_fused_fp32"], 0)
        self.assertFalse(summary["fused_internal_stages_observed"])
        self.assertFalse(summary["global_exactness_activation_allowed"])

    def test_conditional_reproduction_using_observed_exponentials(self) -> None:
        from bioprocess_runtime.reference_gemma import _rms_f32_add, _rms_f32_round, _rms_f32_value
        from bioprocess_runtime.gemma_softmax_slice import OFFSETS

        report_path = Path(__file__).resolve().parent.parent / "artifacts/gemma3_270m_softmax_report.json"
        if not report_path.exists():
            self.skipTest("Full softmax observations are intentionally outside Git")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for observation in report["observations"]:
            stage = observation["staged_aten"]
            for row, exponential in enumerate(stage["exponential_bits"]):
                sums = exponential + [0, 0]
                for offset in OFFSETS:
                    sums = [_rms_f32_add(sums[lane], sums[lane ^ offset]) for lane in range(32)]
                self.assertEqual(sums[:30], stage["denominator_bits"][row])
                probabilities = [_rms_f32_round(_rms_f32_value(bits) / _rms_f32_value(sums[lane])) for lane, bits in enumerate(exponential)]
                self.assertEqual(probabilities, observation["traced"]["output_f32_bits"][row])

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"), "Gemma dependencies are unavailable")
    def test_failed_report_verifies_and_rejects_promotion_and_refit(self) -> None:
        from bioprocess_runtime.gemma_attention_scores import ScoreSources
        from bioprocess_runtime.gemma_softmax_slice import SoftmaxSources, verify_softmax
        from bioprocess_runtime.gemma_rsqrt_lookup import CheckedRsqrtLookup
        from bioprocess_runtime.gemma_rotary_slice import _seal, _sha

        root = Path(__file__).resolve().parent.parent
        names = ("softmax_report.json", "softmax_predictions.json", "rsqrt_table.bin", "rotary_table.json", "rotary_slice_predictions_v2.json", "rotary_slice_report.json", "attention_scores_predictions.json", "attention_scores_report.json")
        if any(not (root / "artifacts" / ("gemma3_270m_" + name)).exists() for name in names):
            self.skipTest("Complete softmax/source artifacts intentionally remain outside Git")
        load = lambda path: json.loads((root / path).read_text(encoding="utf-8"))
        lookup = CheckedRsqrtLookup(load("results/gemma3_270m_rsqrt_table_plan.json"), load("results/gemma3_270m_rsqrt_table_manifest.json"), root / "artifacts/gemma3_270m_rsqrt_table.bin", load("results/gemma3_270m_rsqrt_domain_plan.json"), load("results/gemma3_270m_rsqrt_domain_report.json"), root / "artifacts/rsqrt_mismatches")
        scores = ScoreSources(load("results/gemma3_270m_execution_ir.json"), load("results/gemma3_270m_rotary_slice_plan_v2.json"), load("artifacts/gemma3_270m_rotary_slice_predictions_v2.json"), load("artifacts/gemma3_270m_rotary_slice_report.json"), lookup, load("results/gemma3_270m_rotary_table_plan.json"), load("results/gemma3_270m_rotary_table_manifest.json"), load("artifacts/gemma3_270m_rotary_table.json"))
        sources = SoftmaxSources(scores, load("results/gemma3_270m_attention_scores_plan.json"), load("artifacts/gemma3_270m_attention_scores_predictions.json"), load("artifacts/gemma3_270m_attention_scores_report.json"))
        plan, bundle, report = load("results/gemma3_270m_softmax_plan.json"), load("artifacts/gemma3_270m_softmax_predictions.json"), load("artifacts/gemma3_270m_softmax_report.json")
        self.assertTrue(verify_softmax(sources, plan, bundle, report)["valid"])
        promoted = copy.deepcopy(report)
        promoted["candidate_passes"] = True
        promoted = _seal({key: value for key, value in promoted.items() if key != "report_sha256"}, "report_sha256")
        self.assertFalse(verify_softmax(sources, plan, bundle, promoted)["valid"])
        altered, changed_plan = copy.deepcopy(bundle), copy.deepcopy(plan)
        altered["rows"][0]["exponential_bits"][0] ^= 1
        changed_plan["bundle_sha256"] = _sha(altered)
        changed_plan = _seal({key: value for key, value in changed_plan.items() if key != "plan_sha256"}, "plan_sha256")
        self.assertFalse(verify_softmax(sources, changed_plan, altered, report)["valid"])

    def test_cli_and_source_drift_guards(self) -> None:
        from bioprocess_runtime.cli import build_parser, command_softmax_slice
        from bioprocess_runtime.gemma_softmax_slice import _code_sha

        parser = build_parser()
        args = parser.parse_args(["gemma-softmax-verify", "program", "--bundle", "bundle", "--report", "report", "--reexecute"])
        self.assertTrue(args.reexecute)
        for operation in ("plan", "run"):
            self.assertEqual(parser.parse_args(["gemma-softmax-" + operation, "program", "--bundle", "bundle", "--output", "output"]).operation, operation)
        with patch("bioprocess_runtime.gemma_softmax_slice.Path.read_bytes", return_value=b"source changed"):
            with self.assertRaises(ValueError):
                _code_sha()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "source.json"
            output.write_text("preserve", encoding="utf-8")
            with self.assertRaises(ValueError):
                command_softmax_slice(Namespace(operation="plan", output=output, bundle=Path(directory) / "bundle"))
            self.assertEqual(output.read_text(encoding="utf-8"), "preserve")
