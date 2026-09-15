from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bioprocess_runtime.gemma_mlp_entry import mlp_instructions, normalize_mlp, project_mlp_bits, mlp_report, _worker_init, _worker_row, _code_sha

ROOT = Path(__file__).resolve().parent.parent


class MlpEntryTests(unittest.TestCase):
    def test_instruction_scope_and_norm(self) -> None:
        program = json.loads((ROOT / "results/gemma3_270m_execution_ir.json").read_text(encoding="utf-8"))
        instructions = mlp_instructions(program)
        self.assertEqual([item["opcode"] for item in instructions], ["RMS_NORM", "LINEAR", "LINEAR"])
        changed = copy.deepcopy(program)
        next(item for item in changed["instructions"] if item["outputs"] == ["layer.0.mlp.up"])["parameter_refs"] = ["model.layers.0.mlp.gate_proj.weight"]
        with self.assertRaises(ValueError):
            mlp_instructions(changed)

        class SyntheticRoot:
            def predict_bits(self, bits, runtime):
                return 0x3F800000

        inputs = np.full((1, 30, 640), 0x3F80, dtype=np.uint16)
        normalized, stages = normalize_mlp(inputs, [0] * 640, 1e-6, SyntheticRoot(), {})
        self.assertTrue(np.array_equal(normalized, inputs))
        self.assertEqual(stages["mean_bits"], [0x3F800000] * 30)
        with self.assertRaises(ValueError):
            project_mlp_bits(inputs, {}, 5)

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"), "Gemma dependencies are unavailable")
    def test_cpu_workers_preserve_task_order_and_exact_arithmetic(self) -> None:
        weights = {"gate": [[0x3F80] * 16, [0x4000] * 16], "up": [[0x3F80] * 16]}
        tasks = [("gate", 7, [0x3F80] * 16), ("up", 3, [0x3F80] * 16)]
        with ProcessPoolExecutor(max_workers=2, initializer=_worker_init, initargs=(weights, _code_sha())) as pool:
            results = list(pool.map(_worker_row, tasks))
        self.assertEqual(results, [("gate", 7, [0x4180, 0x4200]), ("up", 3, [0x4180])])

    def test_projection_error_and_activation_scope(self) -> None:
        normalized = np.zeros((1, 30, 640), dtype=np.uint16)
        projection = np.zeros((1, 30, 2048), dtype=np.uint16)
        predictions = {"normalized": normalized, "gate": projection, "up": projection}
        stages = {key: [0x3F800000] * 30 for key in ("mean_bits", "denominator_bits", "rsqrt_bits")}
        record = {role: value.tolist() for role, value in predictions.items()}
        record.update({"input": normalized.tolist(), "gate_input": normalized.tolist(), "up_input": normalized.tolist(),
                       "order": ["normalized", "gate", "activation_unverified", "up"], "scalar_stages": copy.deepcopy(stages),
                       "gate_strides": {"input": [19200, 640, 1], "weight": [640, 1]}, "up_strides": {"input": [19200, 640, 1], "weight": [640, 1]},
                       "norm_cuda_events": ["synthetic"], "gate_cuda_events": ["synthetic"], "up_cuda_events": ["synthetic"]})
        record["gate"][0][29][2047] = 1
        record["scalar_stages"]["mean_input_metadata"] = {"input_shape": [1, 30, 640], "input_strides": [19200, 640, 1], "alignment_mod16": 0, "input_dtype": "torch.float32", "axes": [-1], "keepdim": True}
        report = mlp_report({"plan_sha256": "a" * 64, "runtime": {}}, {"scalar_stages": stages}, normalized, predictions, [{"traced": record, "untraced": record}] * 3, {})
        self.assertFalse(report["mlp_entry_matches"])
        self.assertEqual(report["mismatch_counts"]["gate"], [1, 1, 1])
        self.assertTrue(report["native_gate_activation_executed_unverified"])
        self.assertFalse(report["gate_activation_qualified"])
        self.assertFalse(report["product_executed"])
        self.assertFalse(report["down_projection_executed"])

    def test_recorded_mlp_entry_scope_and_exact_outputs(self) -> None:
        from bioprocess_runtime.gemma_rotary_slice import _sha
        summary = json.loads((ROOT / "results/gemma3_270m_mlp_entry_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["summary_sha256"], _sha({key: value for key, value in summary.items() if key != "summary_sha256"}))
        self.assertTrue(summary["mlp_entry_matches"])
        self.assertEqual(summary["projection_value_count"], 122880)
        self.assertEqual(summary["normalization_value_count"], 19200)
        self.assertTrue(summary["native_gate_activation_executed_unverified"])
        self.assertFalse(summary["gate_activation_qualified"])
        self.assertFalse(summary["product_executed"])
        self.assertFalse(summary["down_projection_executed"])
        self.assertFalse(summary["global_exactness_activation_allowed"])
        for counts in summary["mismatch_counts"].values():
            self.assertEqual(counts, [0, 0, 0])

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"), "Gemma dependencies are unavailable")
    def test_actual_evidence_rejects_output_tampering_and_activation_promotion(self) -> None:
        from bioprocess_runtime.cli import build_parser, _load_mlp_entry_sources
        from bioprocess_runtime.gemma_mlp_entry import verify_mlp
        from bioprocess_runtime.gemma_rotary_slice import _seal

        if any(not (ROOT / "artifacts" / ("gemma3_270m_" + name)).exists() for name in ("exp_table.bin", "mlp_entry_predictions.json", "mlp_entry_report.json", "post_attention_predictions.json", "post_attention_report.json")):
            self.skipTest("Private MLP/source tensors intentionally remain outside Git")
        args = build_parser().parse_args(["gemma-mlp-entry-verify", "results/gemma3_270m_execution_ir.json", "--bundle", "artifacts/gemma3_270m_mlp_entry_predictions.json", "--report", "artifacts/gemma3_270m_mlp_entry_report.json"])
        for name, value in vars(args).items():
            if isinstance(value, Path) and not value.is_absolute():
                setattr(args, name, ROOT / value)
        load = lambda path: json.loads(path.read_text(encoding="utf-8"))
        sources = _load_mlp_entry_sources(args)
        plan, bundle, report = load(args.plan), load(args.bundle), load(args.report)
        self.assertTrue(verify_mlp(sources, plan, bundle, report)["valid"])
        changed = copy.deepcopy(report)
        changed["observations"][0]["traced"]["gate"][0][29][2047] ^= 1
        changed = _seal({key: value for key, value in changed.items() if key != "report_sha256"}, "report_sha256")
        self.assertFalse(verify_mlp(sources, plan, bundle, changed)["valid"])
        promoted = copy.deepcopy(plan)
        promoted["gate_activation_qualified"] = True
        promoted = _seal({key: value for key, value in promoted.items() if key != "plan_sha256"}, "plan_sha256")
        self.assertFalse(verify_mlp(sources, promoted, bundle, report)["valid"])

    def test_cli_output_and_source_guards(self) -> None:
        from bioprocess_runtime.cli import build_parser, command_mlp_entry
        parser = build_parser()
        for operation in ("plan", "run", "verify"):
            argv = ["gemma-mlp-entry-" + operation, "program", "--bundle", "bundle"]
            argv += ["--report", "report", "--reexecute"] if operation == "verify" else ["--output", "output"]
            args = parser.parse_args(argv)
            self.assertEqual(args.operation, operation)
            self.assertEqual(args.workers, 4)
            self.assertEqual(args.post_plan.name, "gemma3_270m_post_attention_plan.json")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "source.json"
            output.write_text("preserve", encoding="utf-8")
            with self.assertRaises(ValueError):
                command_mlp_entry(Namespace(operation="plan", output=output, bundle=Path(directory) / "bundle"))
            self.assertEqual(output.read_text(encoding="utf-8"), "preserve")
        with patch("bioprocess_runtime.gemma_mlp_entry.Path.read_bytes", return_value=b"changed"):
            with self.assertRaises(ValueError):
                _code_sha()
