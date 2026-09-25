"""Standard-library-only checks for the explicit S=30 execution profile."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from bioprocess_runtime import gemma_execution_profile as profile_module


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
PACKAGE = ROOT / "bioprocess_runtime"


class ExecutionProfileTests(unittest.TestCase):

    def test_profile_is_a_fresh_copy(self):
        copy_profile = profile_module.profile()
        copy_profile["input_domain"]["sequence_length"] = 999
        self.assertEqual(profile_module.PROFILE["input_domain"]["sequence_length"], 30)
        self.assertEqual(profile_module.profile()["input_domain"]["position_ids"],
                         list(range(30)))

    def test_frozen_artifacts_still_record_the_declared_boundary(self):
        checks = profile_module.validate_against_artifacts(RESULTS)
        self.assertTrue(checks)
        self.assertTrue(all(check["passed"] for check in checks),
                        [check for check in checks if not check["passed"]])

    def test_artifact_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for name in ("gemma3_270m_independent_holdout_declaration.json",
                         "gemma3_270m_independent_baseline_tokens.json",
                         "gemma3_270m_operational_semantics_summary.json",
                         "gemma3_270m_copy_drift_v1.json"):
                shutil.copy(RESULTS / name, tmp_path / name)
            declaration_path = tmp_path / "gemma3_270m_independent_holdout_declaration.json"
            declaration = json.loads(declaration_path.read_text(encoding="utf-8"))
            declaration["sequence_length"] = 31
            declaration_path.write_text(json.dumps(declaration), encoding="utf-8")
            checks = profile_module.validate_against_artifacts(tmp_path)
            failed = [check for check in checks if not check["passed"]]
            self.assertEqual([check["check"] for check in failed],
                             ["holdout_declaration.sequence_length"])

    def test_pinned_sources_still_contain_the_declared_literals(self):
        checks = profile_module.check_source_specializations(PACKAGE)
        self.assertEqual(len(checks), len(profile_module.SOURCE_SPECIALIZATIONS))
        self.assertTrue(all(check["passed"] for check in checks),
                        [check for check in checks if not check["passed"]])

    def test_source_drift_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for site in profile_module.SOURCE_SPECIALIZATIONS:
                source = PACKAGE / site["file"]
                if source.exists():
                    shutil.copy(source, tmp_path / site["file"])
            target = tmp_path / "gemma_independent.py"
            target.write_text(target.read_text(encoding="utf-8").replace(
                "sequence != 30", "sequence != 64"), encoding="utf-8")
            checks = profile_module.check_source_specializations(tmp_path)
            failed = [check for check in checks if not check["passed"]]
            self.assertTrue(any(check["file"] == "gemma_independent.py"
                                and "sequence != 30" in check["missing"]
                                for check in failed))

    def test_report_is_conformant_and_repeatable(self):
        first = profile_module.build_report(RESULTS, PACKAGE)
        second = profile_module.build_report(RESULTS, PACKAGE)
        self.assertTrue(first["conformant"])
        self.assertEqual(first["report_sha256"], second["report_sha256"])
        self.assertFalse(first["model_inference_performed"])
        self.assertFalse(
            first["profile"]["decoding_paths"]["decode_paths_established_equivalent"])

    def test_cli_refuses_to_overwrite_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "report.json"
            output.write_text("existing", encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, "-m", "bioprocess_runtime.gemma_execution_profile",
                 "--output", str(output)],
                cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertEqual(output.read_text(encoding="utf-8"), "existing")


if __name__ == "__main__":
    unittest.main()
