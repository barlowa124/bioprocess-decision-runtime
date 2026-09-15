from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from bioprocess_runtime.gemma_k1024_probes import candidates, candidate_predictions, probe_pool, select_probes, FAMILIES, CASE_COUNT, _code_sha


class K1024ProbeTests(unittest.TestCase):
    def test_candidate_grid_and_merge_discrimination(self) -> None:
        grid = candidates()
        self.assertEqual(len(grid), 128)
        self.assertEqual(len({item["id"] for item in grid}), 128)
        for candidate in grid:
            self.assertEqual(candidate["partitions"][0][0], 0)
            self.assertEqual(candidate["partitions"][-1][1], 1024)
            self.assertEqual(sum(stop - start for start, stop in candidate["partitions"]), 1024)
        right = [0] * 1024
        right[0], right[64], right[128] = 0x4E80, 0x3F80, 0xCE80
        predictions = dict(zip((item["id"] for item in grid), candidate_predictions([0x3F80] * 1024, right)))
        self.assertEqual(predictions["k64:bfloat16_rne:exact"], 0x3F80)
        self.assertEqual(predictions["k64:bfloat16_rne:sequential_float32_rne"], 0)
        with self.assertRaises(ValueError):
            candidate_predictions([0] * 640, [0] * 640)

    def test_deterministic_pool_and_prediction_only_selection(self) -> None:
        left, pool = probe_pool()
        self.assertEqual((left, pool), probe_pool())
        self.assertEqual(len(left), 1024)
        self.assertGreater(len(set(left)), 1)
        self.assertEqual(len(pool), 256)
        self.assertTrue(all(any(item["right_bits"]) for item in pool))
        predictions = [[(index + candidate) % 4 for candidate in range(128)] for index in range(len(pool))]
        selected, separated, groups = select_probes(pool, predictions)
        self.assertEqual(len(selected), CASE_COUNT)
        self.assertEqual(len(set(selected)), CASE_COUNT)
        for family in FAMILIES:
            self.assertEqual(sum(pool[index]["family"] == family for index in selected), 32)
        self.assertGreater(separated, 0)
        self.assertEqual(len(groups), 4)
        self.assertEqual((selected, separated, groups), select_probes(pool, predictions))

    def test_survivor_recipe_and_cli_binding(self) -> None:
        from bioprocess_runtime.gemma_output_survivor import survivor_dot
        from bioprocess_runtime.cli import build_parser

        candidate = next(item for item in candidates() if item["id"] == "k128:bfloat16_rne:sequential_float32_rne")
        self.assertEqual(survivor_dot([0x3F80] * 1024, [0x3F80] * 1024, candidate), 0x4480)
        changed = copy.deepcopy(candidate)
        changed["chunk_size"] = 127
        with self.assertRaises(ValueError):
            survivor_dot([0x3F80] * 1024, [0x3F80] * 1024, changed)
        args = build_parser().parse_args(["gemma-attention-output-survivor-verify", "program", "--bundle", "bundle", "--report", "report", "--reexecute"])
        self.assertEqual(args.base_plan.name, "gemma3_270m_attention_output_plan.json")
        self.assertEqual(args.probe_plan.name, "gemma3_270m_k1024_probes_plan.json")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is unavailable")
    def test_actual_probe_evidence_rejects_changed_survivors(self) -> None:
        from bioprocess_runtime.gemma_k1024_probes import verify_probes
        from bioprocess_runtime.gemma_rotary_slice import _seal

        root = Path(__file__).resolve().parent.parent
        if not (root / "artifacts/gemma3_270m_k1024_probes_report.json").exists():
            self.skipTest("Controlled tensors are intentionally outside Git")
        load = lambda path: json.loads((root / path).read_text(encoding="utf-8"))
        source = load("results/gemma3_270m_attention_output_summary.json")
        plan = load("results/gemma3_270m_k1024_probes_plan.json")
        bundle, report = load("artifacts/gemma3_270m_k1024_probes_inputs.json"), load("artifacts/gemma3_270m_k1024_probes_report.json")
        checked = verify_probes(source, plan, bundle, report)
        self.assertTrue(checked["valid"])
        self.assertTrue(checked["survivors_supported_in_declared_scope"])
        self.assertEqual(checked["surviving_candidates"], ["k128:bfloat16_rne:sequential_float32_rne"])
        altered = copy.deepcopy(report)
        altered["surviving_candidates"] = ["k64:bfloat16_rne:sequential_float32_rne"]
        altered = _seal({key: value for key, value in altered.items() if key != "report_sha256"}, "report_sha256")
        self.assertFalse(verify_probes(source, plan, bundle, altered)["valid"])
        altered_plan = copy.deepcopy(plan)
        altered_plan["global_exactness_activation_allowed"] = True
        altered_plan = _seal({key: value for key, value in altered_plan.items() if key != "plan_sha256"}, "plan_sha256")
        self.assertFalse(verify_probes(source, altered_plan, bundle, report)["valid"])

    def test_recorded_controlled_survivor_and_model_regression_scope(self) -> None:
        from bioprocess_runtime.gemma_rotary_slice import _sha

        root = Path(__file__).resolve().parent.parent
        load = lambda path: json.loads((root / path).read_text(encoding="utf-8"))
        controlled = load("results/gemma3_270m_k1024_probes_summary.json")
        regression = load("results/gemma3_270m_output_survivor_summary.json")
        for summary in (controlled, regression):
            self.assertEqual(summary["summary_sha256"], _sha({key: value for key, value in summary.items() if key != "summary_sha256"}))
            self.assertFalse(summary["global_exactness_activation_allowed"])
        self.assertEqual(controlled["candidate_count"], 128)
        self.assertEqual(controlled["candidate_pairs_separated"], 8024)
        self.assertEqual(controlled["surviving_candidates"], ["k128:bfloat16_rne:sequential_float32_rne"])
        self.assertTrue(regression["survivor_reproduces_model_case"])
        self.assertEqual(regression["comparison"]["mismatch_counts"]["projected"], [0] * 3)
        self.assertFalse(regression["fresh_model_holdout"])
        self.assertFalse(regression["hardware_partitioning_established"])

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"), "Gemma dependencies are unavailable")
    def test_model_regression_rejects_promoted_claims_and_changed_outputs(self) -> None:
        from bioprocess_runtime.cli import build_parser, _load_survivor_context
        from bioprocess_runtime.gemma_output_survivor import verify_survivor
        from bioprocess_runtime.gemma_rotary_slice import _seal

        root = Path(__file__).resolve().parent.parent
        if any(not (root / "artifacts" / ("gemma3_270m_" + name)).exists() for name in ("exp_table.bin", "output_survivor_report.json", "output_survivor_predictions.json")):
            self.skipTest("Private lookup/model artifacts intentionally remain outside Git")
        args = build_parser().parse_args(["gemma-attention-output-survivor-verify", "results/gemma3_270m_execution_ir.json", "--bundle", "artifacts/gemma3_270m_output_survivor_predictions.json", "--report", "artifacts/gemma3_270m_output_survivor_report.json"])
        for name, value in vars(args).items():
            if isinstance(value, Path) and not value.is_absolute():
                setattr(args, name, root / value)
        load = lambda path: json.loads(path.read_text(encoding="utf-8"))
        context = _load_survivor_context(args)
        plan, bundle, report = load(args.plan), load(args.bundle), load(args.report)
        self.assertTrue(verify_survivor(context, plan, bundle, report)["valid"])
        promoted = copy.deepcopy(plan)
        promoted["fresh_model_holdout"] = True
        promoted = _seal({key: value for key, value in promoted.items() if key != "plan_sha256"}, "plan_sha256")
        self.assertFalse(verify_survivor(context, promoted, bundle, report)["valid"])
        changed = copy.deepcopy(report)
        changed["native_report"]["observations"][0]["traced"]["projected"][0][0][0] ^= 1
        changed = _seal({key: value for key, value in changed.items() if key != "report_sha256"}, "report_sha256")
        self.assertFalse(verify_survivor(context, plan, bundle, changed)["valid"])

    def test_cli_output_and_source_guards(self) -> None:
        from bioprocess_runtime.cli import build_parser, command_k1024_probes

        parser = build_parser()
        for operation in ("plan", "run", "verify"):
            argv = ["gemma-k1024-probes-" + operation, "--bundle", "bundle"]
            argv += ["--report", "report", "--reexecute"] if operation == "verify" else ["--output", "output"]
            self.assertEqual(parser.parse_args(argv).operation, operation)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "source.json"
            output.write_text("preserve", encoding="utf-8")
            with self.assertRaises(ValueError):
                command_k1024_probes(Namespace(operation="plan", output=output, bundle=Path(directory) / "bundle"))
            self.assertEqual(output.read_text(encoding="utf-8"), "preserve")
        with patch("bioprocess_runtime.gemma_k1024_probes.Path.read_bytes", return_value=b"changed source"):
            with self.assertRaises(ValueError):
                _code_sha()
