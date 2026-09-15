from __future__ import annotations

import copy
import importlib.util
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bioprocess_runtime import gemma_gelu_lookup as gelu
from bioprocess_runtime.gemma_rotary_slice import _seal


class GeluLookupTests(unittest.TestCase):
    def test_complete_finite_domain_and_layout_coverage(self) -> None:
        bits = gelu.finite_inputs().tolist()
        self.assertEqual(len(bits), 65280)
        self.assertEqual(gelu.TABLE_BYTES, 130560)
        self.assertEqual([gelu.finite_index(value) for value in bits], list(range(65280)))
        covered = {index for start in gelu.WINDOW_STARTS for index in range(start, start + gelu.WINDOW_COUNT)}
        self.assertEqual(covered, set(range(65280)))
        for value in (True, -1, 65536, 0x7F80, 0xFF80, 0x7FC0, 0xFFFF):
            with self.assertRaises(ValueError):
                gelu.finite_index(value)

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"), "Gemma dependencies are unavailable")
    def test_table_layout_runtime_and_tamper_guards(self) -> None:
        runtime = {"device": "synthetic"}
        source = _seal({"mlp_entry_matches": True, "runtime": runtime, "source_report_sha256": "a" * 64}, "summary_sha256")
        native = lambda bits, shape: (bits.astype("<u2").copy(), ["synthetic_GeluCUDAKernelImpl"])
        with tempfile.TemporaryDirectory() as directory, patch.object(gelu, "_runtime", return_value=runtime), patch.object(gelu, "_cuda_gelu", side_effect=native):
            path, audit = Path(directory) / "table.bin", Path(directory) / "audit"
            plan = gelu.build_gelu_plan(source)
            manifest = gelu.acquire_gelu_table(plan, path, audit)
            layout = gelu.acquire_gelu_layouts(plan, manifest, path, audit)
            provider = gelu.CheckedGeluLookup(plan, manifest, path, layout, audit)
            self.assertFalse(provider._table.flags.writeable)
            for bits in (0, 1, 0x7F7F, 0x8000, 0x8001, 0xFF7F):
                self.assertEqual(provider.predict_bits(bits, runtime), bits)
            with self.assertRaises(ValueError):
                provider.predict_bits(0, {"device": "other"})
            changed = copy.deepcopy(layout)
            changed["windows"].pop()
            changed = _seal({key: value for key, value in changed.items() if key != "report_sha256"}, "report_sha256")
            with self.assertRaises(ValueError):
                gelu.CheckedGeluLookup(plan, manifest, path, changed, audit)
            raw = bytearray(path.read_bytes())
            raw[0] ^= 1
            path.write_bytes(raw)
            with self.assertRaises(ValueError):
                gelu.load_gelu_table(plan, manifest, path, audit)

    def test_cli_existing_output_and_source_drift(self) -> None:
        from bioprocess_runtime.cli import build_parser, command_gelu_table
        parser = build_parser()
        for operation in ("plan", "run", "layouts", "verify"):
            argv = ["gemma-gelu-table-" + operation]
            argv += ["--reexecute", "--replay-output", "replay"] if operation == "verify" else ["--output", "output"]
            self.assertEqual(parser.parse_args(argv).operation, operation)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.json"
            path.write_text("preserve", encoding="utf-8")
            with self.assertRaises(ValueError):
                command_gelu_table(Namespace(operation="plan", output=path))
            self.assertEqual(path.read_text(encoding="utf-8"), "preserve")
        with patch.object(gelu.Path, "read_bytes", return_value=b"changed"):
            with self.assertRaises(ValueError):
                gelu._code_sha()
