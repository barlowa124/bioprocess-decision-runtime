from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from argparse import Namespace
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bioprocess_runtime import gemma_exp_lookup as exponential
from bioprocess_runtime.gemma_rotary_slice import _seal


class ExpLookupTests(unittest.TestCase):
    def test_full_domain_accounting(self) -> None:
        self.assertEqual(exponential.FINITE_STOP, 2139095040)
        self.assertEqual(exponential.TABLE_BYTES, 1241513984)
        self.assertEqual(exponential.FINITE_STOP // exponential.CHUNK, 2040)
        self.assertEqual((exponential.STOP - exponential.START) // exponential.CHUNK, 296)

    def test_provider_domains_and_target_runtime(self) -> None:
        table = {0: 0x3F800000, exponential.STOP - exponential.START - 1: 0}
        plan = {"runtime": {"target": "synthetic"}, "plan_sha256": "a" * 64}
        manifest = {"table_usable": True, "manifest_sha256": "b" * 64, "table_sha256": "c" * 64}
        with patch.object(exponential, "load_exp_table", return_value=table):
            provider = exponential.CheckedExpLookup(plan, manifest, Path("unused"), Path("unused"))
        runtime = plan["runtime"]
        for bits in (0, 0x80000000, 0x80000001, 0x80000000 | (exponential.START - 1), 0x80000000 | exponential.START):
            self.assertEqual(provider.predict_bits(bits, runtime), 0x3F800000)
        for bits in (0x80000000 | (exponential.STOP - 1), 0x80000000 | exponential.STOP, 0xFF7FFFFF, 0xFF800000):
            self.assertEqual(provider.predict_bits(bits, runtime), 0)
        for bits in (1, 0x3F800000, 0x7F800000, 0x7FC00000, 0xFFC00000, True, -1, 0x100000000):
            with self.assertRaises(ValueError):
                provider.predict_bits(bits, runtime)
        with self.assertRaises(ValueError):
            provider.predict_bits(0, {"target": "different"})

    def test_acquisition_coverage_replay_and_table_tamper(self) -> None:
        runtime = {"device": "synthetic"}
        summary = _seal({"candidate_passes": False, "fused_internal_stages_observed": False, "runtime": runtime, "source_report_sha256": "d" * 64}, "summary_sha256")

        def native(bits):
            values = []
            for encoding in bits.tolist():
                magnitude = encoding & 0x7FFFFFFF
                values.append(0x3F800000 if magnitude < 4 else 0x3F000000 - magnitude if magnitude < 12 else 0)
            return np.asarray(values, dtype="<u4"), ["synthetic_exp_kernel_cuda"]

        with ExitStack() as stack, tempfile.TemporaryDirectory() as directory:
            for name, value in {"START": 4, "STOP": 12, "FINITE_STOP": 16, "CHUNK": 4, "TABLE_BYTES": 32}.items():
                stack.enter_context(patch.object(exponential, name, value))
            stack.enter_context(patch.object(exponential, "_runtime", return_value=runtime))
            stack.enter_context(patch.object(exponential, "_cuda_exp", side_effect=native))
            plan = exponential.build_exp_plan(summary)
            table_path, audit = Path(directory) / "table.bin", Path(directory) / "audit"
            progress = []
            manifest = exponential.acquire_exp_table(plan, table_path, audit, progress.append)
            self.assertTrue(manifest["table_usable"])
            self.assertTrue(progress[-1]["complete"])
            loaded = exponential.load_exp_table(plan, manifest, table_path, audit)
            self.assertEqual(loaded.tolist(), [0x3F000000 - index for index in range(4, 12)])
            self.assertFalse(loaded.flags.writeable)
            self.assertTrue(exponential.replay_exp_table(plan, manifest, table_path, audit)["reexecution_exact"])
            damaged = copy.deepcopy(manifest)
            damaged["records"].pop()
            damaged = _seal({key: value for key, value in damaged.items() if key != "manifest_sha256"}, "manifest_sha256")
            with self.assertRaises(ValueError):
                exponential.load_exp_table(plan, damaged, table_path, audit)
            raw = bytearray(table_path.read_bytes())
            raw[0] ^= 1
            table_path.write_bytes(raw)
            with self.assertRaises(ValueError):
                exponential.load_exp_table(plan, manifest, table_path, audit)

    def test_failed_constant_region_retains_auditable_payloads(self) -> None:
        runtime = {"device": "synthetic"}
        summary = _seal({"candidate_passes": False, "fused_internal_stages_observed": False, "runtime": runtime, "source_report_sha256": "d" * 64}, "summary_sha256")

        def native(bits):
            values = [0x3F800000 if (encoding & 0x7FFFFFFF) < 4 else 0 for encoding in bits.tolist()]
            if len(values) == 4 and (int(bits[0]) & 0x7FFFFFFF) == 0:
                values[0] = 0x3F7FFFFF
            return np.asarray(values, dtype="<u4"), ["synthetic_exp_kernel_cuda"]

        with ExitStack() as stack, tempfile.TemporaryDirectory() as directory:
            for name, value in {"START": 4, "STOP": 12, "FINITE_STOP": 16, "CHUNK": 4, "TABLE_BYTES": 32}.items():
                stack.enter_context(patch.object(exponential, name, value))
            stack.enter_context(patch.object(exponential, "_runtime", return_value=runtime))
            stack.enter_context(patch.object(exponential, "_cuda_exp", side_effect=native))
            plan = exponential.build_exp_plan(summary)
            table_path, audit = Path(directory) / "table.bin", Path(directory) / "audit"
            manifest = exponential.acquire_exp_table(plan, table_path, audit, lambda value: None)
            self.assertFalse(manifest["table_usable"])
            self.assertEqual(manifest["records"][0]["observations"][0]["mismatch_count"], 1)
            exponential.load_exp_table(plan, manifest, table_path, audit)
            with self.assertRaises(ValueError):
                exponential.CheckedExpLookup(plan, manifest, table_path, audit)
            payload = audit / (manifest["records"][0]["observations"][0]["mismatch_payload_sha256"] + ".bin")
            payload.write_bytes(b"corrupted")
            with self.assertRaises(ValueError):
                exponential.load_exp_table(plan, manifest, table_path, audit)

    def test_cli_output_and_source_guards(self) -> None:
        from bioprocess_runtime.cli import build_parser, command_exp_lookup

        parser = build_parser()
        self.assertTrue(parser.parse_args(["gemma-exp-verify", "--reexecute", "--replay-output", "replay"]).reexecute)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "source.json"
            output.write_text("preserve", encoding="utf-8")
            with self.assertRaises(ValueError):
                command_exp_lookup(Namespace(operation="plan", output=output))
            self.assertEqual(output.read_text(encoding="utf-8"), "preserve")
        with patch.object(exponential.Path, "read_bytes", return_value=b"changed"):
            with self.assertRaises(ValueError):
                exponential._code_sha()
