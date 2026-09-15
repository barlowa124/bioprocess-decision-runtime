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

from bioprocess_runtime.gemma_mlp_product import predict_product, product_instructions, product_report, _code_sha

ROOT = Path(__file__).resolve().parent.parent


class MlpProductTests(unittest.TestCase):
    def test_lookup_and_independent_signed_zero_product(self) -> None:
        class SyntheticLookup:
            def predict_bits(self, bits, runtime):
                return 0x8000 if bits & 0x8000 else 0x3F80

        gate = np.full((1, 30, 2048), 0x3F80, dtype=np.uint16)
        gate[0, 0, 0] = 0xBF80
        up = np.full_like(gate, 0x4000)
        predicted = predict_product(gate, up, SyntheticLookup(), {})
        self.assertEqual(predicted["activation"][0][0][0], 0x8000)
        self.assertEqual(predicted["product"][0][0][0], 0x8000)
        self.assertEqual(predicted["product"][0][29][2047], 0x4000)
        with self.assertRaises(ValueError):
            predict_product(gate[..., :2047], up, SyntheticLookup(), {})

    def test_ir_and_last_coordinate_coverage(self) -> None:
        program = json.loads((ROOT / "results/gemma3_270m_execution_ir.json").read_text(encoding="utf-8"))
        self.assertEqual([item["opcode"] for item in product_instructions(program)], ["GELU_TANH", "MUL"])
        values = np.zeros((1, 30, 2048), dtype=np.uint16)
        actual = values.copy()
        actual[0, 29, 2047] = 1
        record = {"gate": values.tolist(), "up": values.tolist(), "activation": values.tolist(), "product": actual.tolist(), "multiply_output": actual.tolist(),
                  "order": ["gate", "activation", "up", "product"], "activation_input_strides": [61440, 2048, 1],
                  "multiply_strides": [[61440, 2048, 1]] * 2, "activation_cuda_events": ["synthetic"], "product_cuda_events": ["synthetic"]}
        report = product_report({"plan_sha256": "a" * 64, "runtime": {}}, values, values, {"activation": values, "product": values}, [{"traced": record, "untraced": record}] * 3, {})
        self.assertFalse(report["activation_and_product_match"])
        self.assertEqual(report["mismatch_counts"]["activation"], [0, 0, 0])
        self.assertEqual(report["mismatch_counts"]["product"], [1, 1, 1])
        self.assertEqual(report["first_divergence"]["coordinate"], [0, 29, 2047])
        self.assertFalse(report["down_projection_executed"])

    def test_recorded_product_scope(self) -> None:
        from bioprocess_runtime.gemma_rotary_slice import _sha
        summary = json.loads((ROOT / "results/gemma3_270m_mlp_product_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["summary_sha256"], _sha({key: value for key, value in summary.items() if key != "summary_sha256"}))
        self.assertTrue(summary["activation_and_product_match"])
        self.assertTrue(summary["uses_empirical_gelu_specification"])
        self.assertEqual(summary["activation_value_count"], 61440)
        self.assertEqual(summary["product_value_count"], 61440)
        self.assertFalse(summary["native_activation_arithmetic_reconstructed"])
        self.assertFalse(summary["down_projection_executed"])
        self.assertFalse(summary["global_exactness_activation_allowed"])
        self.assertEqual(summary["mismatch_counts"], {"activation": [0] * 3, "product": [0] * 3})

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"), "Gemma dependencies are unavailable")
    def test_real_evidence_rejects_changed_product_and_scope(self) -> None:
        from bioprocess_runtime.cli import build_parser, _load_product_sources
        from bioprocess_runtime.gemma_mlp_product import verify_product
        from bioprocess_runtime.gemma_rotary_slice import _seal

        if any(not (ROOT / "artifacts" / ("gemma3_270m_" + name)).exists() for name in ("exp_table.bin", "gelu_table.bin", "mlp_product_predictions.json", "mlp_product_report.json", "mlp_entry_predictions.json", "mlp_entry_report.json")):
            self.skipTest("Private product/source tensors intentionally remain outside Git")
        args = build_parser().parse_args(["gemma-mlp-product-verify", "results/gemma3_270m_execution_ir.json", "--bundle", "artifacts/gemma3_270m_mlp_product_predictions.json", "--report", "artifacts/gemma3_270m_mlp_product_report.json"])
        for name, value in vars(args).items():
            if isinstance(value, Path) and not value.is_absolute():
                setattr(args, name, ROOT / value)
        load = lambda path: json.loads(path.read_text(encoding="utf-8"))
        sources = _load_product_sources(args)
        plan, bundle, report = load(args.plan), load(args.bundle), load(args.report)
        self.assertTrue(verify_product(sources, plan, bundle, report)["valid"])
        changed = copy.deepcopy(report)
        changed["observations"][0]["traced"]["product"][0][29][2047] ^= 1
        changed = _seal({key: value for key, value in changed.items() if key != "report_sha256"}, "report_sha256")
        self.assertFalse(verify_product(sources, plan, bundle, changed)["valid"])
        promoted = copy.deepcopy(plan)
        promoted["native_activation_arithmetic_reconstructed"] = True
        promoted = _seal({key: value for key, value in promoted.items() if key != "plan_sha256"}, "plan_sha256")
        self.assertFalse(verify_product(sources, promoted, bundle, report)["valid"])

    def test_cli_output_and_source_guards(self) -> None:
        from bioprocess_runtime.cli import build_parser, command_mlp_product
        parser = build_parser()
        for operation in ("plan", "run", "verify"):
            argv = ["gemma-mlp-product-" + operation, "program", "--bundle", "bundle"]
            argv += ["--report", "report", "--reexecute"] if operation == "verify" else ["--output", "output"]
            args = parser.parse_args(argv)
            self.assertEqual(args.operation, operation)
            self.assertEqual(args.entry_plan.name, "gemma3_270m_mlp_entry_plan.json")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.json"
            path.write_text("preserve", encoding="utf-8")
            with self.assertRaises(ValueError):
                command_mlp_product(Namespace(operation="plan", output=path, bundle=Path(directory) / "bundle"))
            self.assertEqual(path.read_text(encoding="utf-8"), "preserve")
        with patch("bioprocess_runtime.gemma_mlp_product.Path.read_bytes", return_value=b"changed"):
            with self.assertRaises(ValueError):
                _code_sha()
