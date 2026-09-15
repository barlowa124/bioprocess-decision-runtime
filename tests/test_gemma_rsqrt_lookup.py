from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bioprocess_runtime.gemma_rsqrt_lookup import (
    BASE_START, CHUNK, MANTISSAS, CheckedRsqrtLookup, _domain_report, _expected_chunk,
    _positive_normal, _raw_lookup,
)


class RsqrtLookupTests(unittest.TestCase):
    def test_exponent_parity_and_scaling(self) -> None:
        table = {0: 0x3F800000, MANTISSAS: 0x3F3504F3}
        self.assertEqual(_raw_lookup(0x3F800000, table), 0x3F800000)
        self.assertEqual(_raw_lookup(0x40800000, table), 0x3F000000)
        self.assertEqual(_raw_lookup(0x3E800000, table), 0x40000000)
        self.assertEqual(_raw_lookup(0x3F000000, table), 0x3FB504F3)
        self.assertEqual(_raw_lookup(0x00800000, table), 0x5F000000)
        for bits in (0, 1, 0x80000000, 0xBF800000, 0x7F800000, 0x7FC00000, True, 1.0, -1, 1 << 40):
            with self.assertRaises(ValueError):
                _positive_normal(bits)

    def test_vector_predictions_match_scalar_mapping(self) -> None:
        table = np.full(2 * MANTISSAS, 0x3F400000, dtype="<u4")
        for exponent in (1, 2, 126, 127, 128, 129, 254):
            expected = _expected_chunk(table, exponent, 0)
            for mantissa in (0, 1, 127, CHUNK - 1):
                self.assertEqual(int(expected[mantissa]), _raw_lookup((exponent << 23) | mantissa, table))

    def test_missing_exponent_coverage_and_runtime_are_rejected(self) -> None:
        plan = {"runtime": {"device": "synthetic"}, "plan_sha256": "synthetic", "expected_kernel_names": ["synthetic"]}
        manifest = {"table_sha256": "synthetic", "manifest_sha256": "synthetic"}
        with patch("bioprocess_runtime.gemma_rsqrt_lookup.load_table", return_value={0: 0x3F800000}):
            lookup = CheckedRsqrtLookup(plan, manifest, Path("unused"))
        self.assertEqual(lookup.predict_bits(BASE_START, plan["runtime"]), 0x3F800000)
        metadata = lookup.evidence
        metadata["table_sha256"] = "changed"
        self.assertEqual(lookup.evidence["table_sha256"], "synthetic")
        with self.assertRaises(AttributeError):
            lookup.table_sha256 = "changed"
        with self.assertRaises(ValueError):
            lookup.predict_bits(0x40800000, plan["runtime"])
        with self.assertRaises(ValueError):
            lookup.predict_bits(BASE_START, {"device": "different"})

    def _artifacts(self):
        root = Path(__file__).resolve().parent.parent
        table_path = root / "artifacts/gemma3_270m_rsqrt_table.bin"
        if not table_path.exists():
            self.skipTest("The empirical rsqrt table is intentionally outside Git")
        load = lambda name: json.loads((root / "results" / name).read_text(encoding="utf-8"))
        return root, table_path, load("gemma3_270m_rsqrt_table_plan.json"), load("gemma3_270m_rsqrt_table_manifest.json"), load("gemma3_270m_rsqrt_domain_plan.json"), load("gemma3_270m_rsqrt_domain_report.json")

    def test_real_table_domain_and_tamper_guards(self) -> None:
        from bioprocess_runtime.gemma_rsqrt_lookup import load_table, verify_domain, _seal

        root, path, table_plan, manifest, plan, report = self._artifacts()
        table = load_table(table_plan, manifest, path)
        self.assertFalse(table.flags.writeable)
        result = verify_domain(table_plan, manifest, path, plan, report, root / "artifacts/rsqrt_mismatches")
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["tested_value_count"], 2_130_706_432)
        self.assertEqual(result["covered_exponents"], list(range(1, 255)))
        self.assertFalse(report["native_arithmetic_reconstructed"])
        damaged = copy.deepcopy(manifest)
        damaged["table_sha256"] = "0" * 64
        damaged = _seal({k: v for k, v in damaged.items() if k != "manifest_sha256"}, "manifest_sha256")
        with self.assertRaises(ValueError):
            load_table(table_plan, damaged, path)
        damaged = copy.deepcopy(report)
        damaged["records"][0]["observed_sha256"] = "f" * 64
        damaged = _seal({k: v for k, v in damaged.items() if k != "report_sha256"}, "report_sha256")
        with patch("bioprocess_runtime.gemma_rsqrt_lookup.build_domain_plan", return_value=plan):
            self.assertFalse(verify_domain(table_plan, manifest, path, plan, damaged, root / "artifacts/rsqrt_mismatches")["valid"])

    def test_lookup_rms_matches_all_preserved_stages(self) -> None:
        from bioprocess_runtime.gemma_rsqrt_lookup import verify_rms_lookup, _seal

        root, path, table_plan, manifest, domain_plan, domain_report = self._artifacts()
        audit = root / "artifacts/rsqrt_mismatches"
        if any(not (root / name).exists() for name in ("artifacts/gemma3_270m_rms_slice_predictions.json", "artifacts/gemma3_270m_rms_slice_report.json", "artifacts/gemma3_270m_rsqrt_rms_predictions.json")):
            self.skipTest("Tensor-rich RMS regression artifacts are intentionally outside Git")
        lookup = CheckedRsqrtLookup(table_plan, manifest, path, domain_plan, domain_report, audit)
        with self.assertRaises(ValueError):
            lookup.predict_bits(BASE_START, {**table_plan["runtime"], "driver_version": "different"})
        load = lambda path: json.loads((root / path).read_text(encoding="utf-8"))
        sources = [load(name) for name in ("results/gemma3_270m_execution_ir.json", "results/gemma3_270m_rms_slice_plan.json", "artifacts/gemma3_270m_rms_slice_predictions.json", "artifacts/gemma3_270m_rms_slice_report.json")]
        plan = load("results/gemma3_270m_rsqrt_rms_plan.json")
        bundle = load("artifacts/gemma3_270m_rsqrt_rms_predictions.json")
        report = load("results/gemma3_270m_rsqrt_rms_report.json")
        with patch("bioprocess_runtime.gemma_rsqrt_lookup._runtime", return_value=table_plan["runtime"]):
            result = verify_rms_lookup(*sources, lookup, plan, bundle, report, audit)
            self.assertTrue(result["valid"], result)
            self.assertTrue(result["lookup_rms_check_passes"])
            damaged = copy.deepcopy(report)
            damaged["role_summaries"]["input_norm"]["stored_stage_mismatch_counts"]["rsqrt_bits"] = [1, 1, 1]
            damaged = _seal({k: v for k, v in damaged.items() if k != "report_sha256"}, "report_sha256")
            self.assertFalse(verify_rms_lookup(*sources, lookup, plan, bundle, damaged, audit)["valid"])

    def test_cli_modes_and_existing_output_protection(self) -> None:
        from bioprocess_runtime.cli import build_parser, command_rsqrt_lookup
        from bioprocess_runtime.gemma_rsqrt_lookup import _seal

        parser = build_parser()
        common = ["table-plan", "manifest", "--table", "table.bin"]
        cases = {
            "table-plan": ["summary", "--output", "plan"],
            "table-build": ["table-plan", "--table", "table.bin", "--audit-dir", "audit", "--output", "manifest"],
            "table-verify": common,
            "domain-plan": [*common, "--output", "domain-plan"],
            "domain-run": [*common, "domain-plan", "--audit-dir", "audit", "--journal", "journal", "--output", "domain-report"],
            "domain-verify": [*common, "domain-plan", "domain-report", "--audit-dir", "audit", "--reexecute"],
            "rms-plan": [*common, "domain-plan", "domain-report", "--audit-dir", "audit", "--bundle", "bundle", "--output", "rms-plan"],
            "rms-run": [*common, "domain-plan", "domain-report", "rms-plan", "--audit-dir", "audit", "--bundle", "bundle", "--output", "rms-report"],
            "rms-verify": [*common, "domain-plan", "domain-report", "rms-plan", "rms-report", "--audit-dir", "audit", "--bundle", "bundle", "--reexecute"],
        }
        for operation, arguments in cases.items():
            parsed = parser.parse_args(["gemma-rsqrt-" + operation, *arguments])
            self.assertEqual(parsed.operation, operation)
            self.assertIs(parsed.handler, command_rsqrt_lookup)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = _seal({"runtime": {"device": "synthetic"}, "actual_inputs_match_plan": True,
                             "profiled_cuda_event_names": {"input_norm": [["synthetic_rsqrt_kernel_cuda"]]}, "source_report_sha256": "a" * 64}, "summary_sha256")
            source, destination = root / "source.json", root / "existing.json"
            source.write_text(json.dumps(summary), encoding="utf-8")
            destination.write_text("preserve", encoding="utf-8")
            args = parser.parse_args(["gemma-rsqrt-table-plan", str(source), "--output", str(destination)])
            with patch("bioprocess_runtime.gemma_rsqrt_lookup._runtime", return_value=summary["runtime"]), self.assertRaises(FileExistsError):
                command_rsqrt_lookup(args)
            self.assertEqual(destination.read_text(encoding="utf-8"), "preserve")

    def test_complete_domain_counts_and_failed_exponent(self) -> None:
        chunks = [{"exponent_field": exponent, "count": CHUNK} for exponent in range(1, 255) for _ in range(MANTISSAS // CHUNK)]
        plan = {"chunks": chunks, "plan_sha256": "synthetic", "table_sha256": "synthetic", "runtime": {}, "expected_kernel_names": ["synthetic"], "launch_shape": [CHUNK]}
        records = [{**chunk, "mismatch_count": 0, "kernel_names": ["synthetic"]} for chunk in chunks]
        report = _domain_report(plan, records, {})
        self.assertEqual(report["tested_value_count"], 2_130_706_432)
        self.assertTrue(report["all_positive_normal_values_match"])
        records[0]["mismatch_count"] = 1
        report = _domain_report(plan, records, {})
        self.assertNotIn(1, report["covered_exponents"])
        self.assertEqual(report["mismatch_count"], 1)
        self.assertFalse(report["all_positive_normal_values_match"])
        self.assertEqual(_domain_report(plan, records, {"changed": True})["covered_exponents"], [])
        with self.assertRaises(ValueError):
            _domain_report(plan, records[:-1], {})
