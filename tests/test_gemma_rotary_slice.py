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

from bioprocess_runtime.gemma_rotary_slice import rotary_angle_bits, rotate_bfloat16_bits, _rotary_instructions

ROOT = Path(__file__).resolve().parent.parent


class RotarySliceTests(unittest.TestCase):
    def test_position_and_frequency_domain(self) -> None:
        angles = rotary_angle_bits([0x3F000000] * 128, list(range(30)))
        self.assertEqual(angles.shape, (1, 30, 256))
        self.assertTrue(np.all(angles[0, 0] == 0))
        self.assertTrue(np.all(angles[0, 1] == 0x3F000000))
        self.assertTrue(np.all(angles[0, 29] == 0x41680000))
        with self.assertRaises(ValueError):
            rotary_angle_bits([0x3F000000] * 128, list(range(1, 31)))
        with self.assertRaises(ValueError):
            rotary_angle_bits([0x3F000000] * 64, list(range(30)))

    def test_half_rotation_sign_and_broadcast(self) -> None:
        values = np.full((1, 4, 30, 256), 0x3F80, dtype=np.uint16)
        values[..., 128:] = 0x4000
        cosine = np.zeros((1, 30, 256), dtype=np.uint16)
        sine = np.full_like(cosine, 0x3F80)
        rotated = rotate_bfloat16_bits(values, cosine, sine)
        self.assertTrue(np.all(rotated[..., :128] == 0xC000))
        self.assertTrue(np.all(rotated[..., 128:] == 0x3F80))
        with self.assertRaises(ValueError):
            rotate_bfloat16_bits(values[:, :, :29], cosine, sine)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is unavailable")
    def test_separate_bfloat16_rounding_matches_cpu_eager_reference(self) -> None:
        import torch

        values = (np.arange(7680, dtype=np.uint16).reshape(1, 1, 30, 256) % 0x300 + 0x3D00).astype(np.uint16)
        values[..., 128:] |= 0x8000
        cosine = np.full((1, 30, 256), 0x3F35, dtype=np.uint16)
        sine = np.full_like(cosine, 0x3F35)
        result = rotate_bfloat16_bits(values, cosine, sine)
        tensor = torch.from_numpy(values).view(torch.bfloat16)
        cos, sin = [torch.from_numpy(item).view(torch.bfloat16).unsqueeze(1) for item in (cosine, sine)]
        rotated = torch.cat((-tensor[..., 128:], tensor[..., :128]), dim=-1)
        expected = ((tensor * cos) + (rotated * sin)).contiguous().view(torch.uint16).numpy()
        self.assertTrue(np.array_equal(result, expected))

    def test_first_layer_instruction_binding(self) -> None:
        program = json.loads((ROOT / "results/gemma3_270m_execution_ir.json").read_text(encoding="utf-8"))
        position, table, rotation = _rotary_instructions(program)
        self.assertEqual(position["opcode"], "ARANGE")
        self.assertEqual(table["parameter_refs"], ["model.rotary_emb_local.inv_freq"])
        self.assertEqual(rotation["attributes"], {"rotary_profile": "local"})

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"), "Gemma dependencies are unavailable")
    def test_real_table_rejects_changed_arguments_and_scope(self) -> None:
        from bioprocess_runtime.gemma_rotary_slice import check_rotary_table

        private = ROOT / "artifacts/gemma3_270m_rotary_table.json"
        if not private.exists():
            self.skipTest("Empirical table data is intentionally outside Git")
        load = lambda path: json.loads((ROOT / path).read_text(encoding="utf-8"))
        program = load("results/gemma3_270m_execution_ir.json")
        plan = load("results/gemma3_270m_rotary_table_plan.json")
        manifest = load("results/gemma3_270m_rotary_table_manifest.json")
        bundle = load("artifacts/gemma3_270m_rotary_table.json")
        check_rotary_table(program, plan, manifest, bundle)
        damaged = copy.deepcopy(bundle)
        damaged["repetitions"][0]["stages"]["cosine"]["input_bits"][0][1][0] ^= 1
        with self.assertRaises(ValueError):
            check_rotary_table(program, plan, manifest, damaged)
        damaged_plan = copy.deepcopy(plan)
        damaged_plan["positions"] = list(range(1, 31))
        with self.assertRaises(ValueError):
            check_rotary_table(program, damaged_plan, manifest, bundle)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is unavailable")
    def test_source_change_blocks_plan_freezing(self) -> None:
        from bioprocess_runtime.gemma_rotary_slice import build_rotary_slice

        load = lambda path: json.loads((ROOT / path).read_text(encoding="utf-8"))
        program = load("results/gemma3_270m_execution_ir.json")
        entry_plan = load("results/gemma3_270m_attention_entry_plan.json")
        tables = {"repetitions": [{"cosine": np.full((1, 30, 256), 0x3F80, dtype=np.uint16).tolist(), "sine": np.zeros((1, 30, 256), dtype=np.uint16).tolist()}]}
        states = {"layer.0.query.normalized": np.zeros((1, 4, 30, 256), dtype=np.uint16), "layer.0.key.normalized": np.zeros((1, 1, 30, 256), dtype=np.uint16)}
        with patch("bioprocess_runtime.gemma_rotary_slice._rotation_code_sha", side_effect=["a" * 64, "b" * 64]), patch("bioprocess_runtime.gemma_rotary_slice.check_rotary_table"), patch("bioprocess_runtime.gemma_rotary_slice.build_rotary_table_plan", return_value={}), patch("bioprocess_runtime.gemma_rotary_slice.build_attention_entry_plan", return_value=(entry_plan, {})), patch("bioprocess_runtime.gemma_rotary_slice.check_attention_entry_plan", return_value=states):
            with self.assertRaisesRegex(ValueError, "source changed during prediction"):
                build_rotary_slice(program, None, {}, None, {}, {"manifest_sha256": "c" * 64}, tables)

    def test_compact_rotary_result_and_rejected_source_hash_plan(self) -> None:
        from bioprocess_runtime.gemma_rotary_slice import _sha

        load = lambda path: json.loads((ROOT / path).read_text(encoding="utf-8"))
        summary = load("results/gemma3_270m_rotary_slice_summary.json")
        plan = load("results/gemma3_270m_rotary_slice_plan_v2.json")
        rejected = load("results/gemma3_270m_rotary_slice_plan.json")
        self.assertEqual(summary["summary_sha256"], _sha({key: value for key, value in summary.items() if key != "summary_sha256"}))
        self.assertEqual(summary["plan_sha256"], plan["plan_sha256"])
        self.assertEqual(summary["instruction_count"], 14)
        self.assertEqual(summary["target_value_count"], 46080)
        self.assertTrue(summary["rotated_attention_inputs_bit_exact"])
        self.assertTrue(summary["original_tables_match_specification"])
        self.assertFalse(summary["attention_arithmetic_executed"])
        self.assertFalse(summary["native_trigonometric_arithmetic_reconstructed"])
        self.assertFalse(summary["global_exactness_activation_allowed"])
        self.assertEqual(rejected["bundle_sha256"], plan["bundle_sha256"])
        self.assertNotEqual(rejected["rotation_code_sha256"], plan["rotation_code_sha256"])
        for hashes in summary["observed_target_hashes"].values():
            self.assertEqual(len(hashes), 3)
            self.assertEqual(len(set(hashes)), 1)

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"), "Gemma dependencies are unavailable")
    def test_connected_artifact_tamper_rejection(self) -> None:
        from bioprocess_runtime.gemma_rotary_slice import verify_rotary_slice, _seal
        from bioprocess_runtime.gemma_rsqrt_lookup import CheckedRsqrtLookup
        from bioprocess_runtime.operational_semantics import append_chain_record

        if not (ROOT / "artifacts/gemma3_270m_rotary_slice_report.json").exists():
            self.skipTest("Complete rotary observations are intentionally outside Git")
        load = lambda path: json.loads((ROOT / path).read_text(encoding="utf-8"))
        lookup = CheckedRsqrtLookup(load("results/gemma3_270m_rsqrt_table_plan.json"), load("results/gemma3_270m_rsqrt_table_manifest.json"), ROOT / "artifacts/gemma3_270m_rsqrt_table.bin", load("results/gemma3_270m_rsqrt_domain_plan.json"), load("results/gemma3_270m_rsqrt_domain_report.json"), ROOT / "artifacts/rsqrt_mismatches")
        program = load("results/gemma3_270m_execution_ir.json")
        plan = load("results/gemma3_270m_rotary_slice_plan_v2.json")
        bundle = load("artifacts/gemma3_270m_rotary_slice_predictions_v2.json")
        report = load("artifacts/gemma3_270m_rotary_slice_report.json")
        evidence = (lookup, load("results/gemma3_270m_rotary_table_plan.json"), load("results/gemma3_270m_rotary_table_manifest.json"), load("artifacts/gemma3_270m_rotary_table.json"))
        self.assertTrue(verify_rotary_slice(program, plan, bundle, report, *evidence)["valid"])
        damaged = copy.deepcopy(report)
        damaged["observations"][0]["attention_inputs"]["layer.0.query.rotary"][0][3][29][255] ^= 1
        damaged = _seal({key: value for key, value in damaged.items() if key != "report_sha256"}, "report_sha256")
        self.assertFalse(verify_rotary_slice(program, plan, bundle, damaged, *evidence)["valid"])
        changed, records = copy.deepcopy(plan), []
        for index, record in enumerate(changed["records"]):
            payload = record["payload"]
            if index == 13:
                payload["provider"] = "unregistered_override"
            append_chain_record(records, payload)
        changed["records"], changed["execution_root"] = records, records[-1]["record_hash"]
        changed = _seal({key: value for key, value in changed.items() if key != "plan_sha256"}, "plan_sha256")
        self.assertFalse(verify_rotary_slice(program, changed, bundle, report, *evidence)["valid"])
        rejected = load("results/gemma3_270m_rotary_slice_plan.json")
        self.assertFalse(verify_rotary_slice(program, rejected, bundle, report, *evidence)["valid"])

    def test_cli_modes_and_output_collision(self) -> None:
        from bioprocess_runtime.cli import build_parser, command_rotary_slice

        parser = build_parser()
        for operation in ("table-plan", "table-run", "table-verify", "plan", "run", "verify"):
            argv = ["gemma-rotary-" + operation, "program"]
            if operation in ("table-run", "plan", "run", "verify"):
                argv.extend(["--bundle", "bundle"])
            if operation.endswith("verify"):
                argv.append("--reexecute")
            else:
                argv.extend(["--output", "output"])
            if operation == "verify":
                argv.extend(["--report", "report"])
            self.assertEqual(parser.parse_args(argv).operation, operation)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "source.json"
            output.write_text("preserve", encoding="utf-8")
            with self.assertRaises(ValueError):
                command_rotary_slice(Namespace(operation="table-plan", output=output))
            self.assertEqual(output.read_text(encoding="utf-8"), "preserve")
