from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from bioprocess_runtime.gemma_softmax_lookup import lookup_softmax_row, _check_provider, _code_sha
from bioprocess_runtime.gemma_softmax_slice import predict_softmax_row, exp_float32_rne


class SoftmaxLookupTests(unittest.TestCase):
    def test_lookup_is_the_only_changed_primitive(self) -> None:
        class SyntheticExp:
            def predict_bits(self, bits, runtime):
                return exp_float32_rne(bits)

        for row in ([0x3F800000] * 30, [0] + [0xFF7F0000] * 29, [0, 0xBF800000] * 15):
            self.assertEqual(lookup_softmax_row(row, SyntheticExp(), {}), predict_softmax_row(row))
        for row in ([0] * 29, [True] + [0] * 29, [0x7FC00000] + [0] * 29):
            with self.assertRaises(ValueError):
                lookup_softmax_row(row, SyntheticExp(), {})

    def test_unregistered_provider_and_source_drift_reject(self) -> None:
        with self.assertRaises(ValueError):
            _check_provider(object())
        with patch("bioprocess_runtime.gemma_softmax_lookup.Path.read_bytes", return_value=b"changed"):
            with self.assertRaises(ValueError):
                _code_sha()

    def test_compact_result_preserves_original_failure(self) -> None:
        from bioprocess_runtime.gemma_rotary_slice import _sha

        root = Path(__file__).resolve().parent.parent
        result = json.loads((root / "results/gemma3_270m_softmax_lookup_summary.json").read_text(encoding="utf-8"))
        original = json.loads((root / "results/gemma3_270m_softmax_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(result["summary_sha256"], _sha({key: value for key, value in result.items() if key != "summary_sha256"}))
        self.assertTrue(result["lookup_softmax_passes"])
        self.assertFalse(result["native_exponential_reconstructed"])
        self.assertFalse(result["global_exactness_activation_allowed"])
        self.assertFalse(result["fused_internal_stages_observed"])
        self.assertEqual(result["comparison"]["mismatch_counts"]["output_f32_bits"], [0] * 3)
        self.assertFalse(original["candidate_passes"])
        self.assertEqual(original["mismatch_counts"]["output_f32_bits"], [630] * 3)
        self.assertEqual(result["native_report_sha256"], original["source_report_sha256"])

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"), "Gemma dependencies are unavailable")
    def test_checked_real_table_softmax_and_tamper_guards(self) -> None:
        from bioprocess_runtime.cli import build_parser, _load_softmax_sources
        from bioprocess_runtime.gemma_exp_lookup import CheckedExpLookup
        from bioprocess_runtime.gemma_softmax_lookup import verify_lookup_softmax
        from bioprocess_runtime.gemma_rotary_slice import _seal, _sha

        root = Path(__file__).resolve().parent.parent
        required = ("exp_table.bin", "rsqrt_table.bin", "softmax_lookup_predictions.json", "softmax_lookup_report.json", "softmax_report.json", "softmax_predictions.json")
        if any(not (root / "artifacts" / ("gemma3_270m_" + name)).exists() for name in required):
            self.skipTest("Large lookup and tensor-rich artifacts intentionally remain outside Git")
        args = build_parser().parse_args(["gemma-softmax-lookup-verify", "results/gemma3_270m_execution_ir.json", "--bundle", "artifacts/gemma3_270m_softmax_lookup_predictions.json", "--report", "artifacts/gemma3_270m_softmax_lookup_report.json"])
        for name, value in vars(args).items():
            if isinstance(value, Path) and not value.is_absolute():
                setattr(args, name, root / value)
        load = lambda path: json.loads(path.read_text(encoding="utf-8"))
        sources = _load_softmax_sources(args)
        lookup = CheckedExpLookup(load(args.exp_plan), load(args.exp_manifest), args.exp_table, args.exp_audit_dir)
        context = (sources, load(args.original_plan), load(args.original_bundle), load(args.original_report), lookup)
        plan, bundle, report = load(args.plan), load(args.bundle), load(args.report)
        self.assertTrue(verify_lookup_softmax(*context, plan, bundle, report)["valid"])
        promoted = copy.deepcopy(report)
        promoted["global_exactness_activation_allowed"] = True
        promoted = _seal({key: value for key, value in promoted.items() if key != "report_sha256"}, "report_sha256")
        self.assertFalse(verify_lookup_softmax(*context, plan, bundle, promoted)["valid"])
        changed, altered_plan = copy.deepcopy(bundle), copy.deepcopy(plan)
        changed["rows"][0]["exponential_bits"][0] ^= 1
        altered_plan["bundle_sha256"] = _sha(changed)
        altered_plan = _seal({key: value for key, value in altered_plan.items() if key != "plan_sha256"}, "plan_sha256")
        self.assertFalse(verify_lookup_softmax(*context, altered_plan, changed, report)["valid"])

    def test_cli_preserves_prior_evidence(self) -> None:
        from bioprocess_runtime.cli import build_parser, command_lookup_softmax

        parser = build_parser()
        for operation in ("plan", "run", "verify"):
            argv = ["gemma-softmax-lookup-" + operation, "program", "--bundle", "bundle"]
            argv += ["--report", "report", "--reexecute"] if operation == "verify" else ["--output", "output"]
            args = parser.parse_args(argv)
            self.assertEqual(args.operation, operation)
            self.assertEqual(args.original_plan.name, "gemma3_270m_softmax_plan.json")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "original.json"
            output.write_text("preserve", encoding="utf-8")
            with self.assertRaises(ValueError):
                command_lookup_softmax(Namespace(operation="plan", output=output, bundle=Path(directory) / "bundle"))
            self.assertEqual(output.read_text(encoding="utf-8"), "preserve")
