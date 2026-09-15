from __future__ import annotations

import copy
import hashlib
import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from bioprocess_runtime import demo_ui as ui
from bioprocess_runtime.serialization import canonical_json

ROOT = Path(__file__).resolve().parents[1]


class DemoDataTests(unittest.TestCase):
    def test_real_catalog_pinned_evidence_and_honest_coverage(self):
        catalog = ui.catalog(ROOT)
        self.assertEqual(catalog["mode"], "saved_evidence")
        self.assertEqual(len(catalog["layers"]), 18)
        self.assertEqual([layer["index"] for layer in catalog["layers"] if layer["coverage"] == "connected_recorded"], list(range(18)))
        self.assertTrue(catalog["full_target_replay_recorded"])
        self.assertTrue(all(layer["evidence_id"] == "full-target-holdouts" for layer in catalog["layers"]))
        for spec in ui.EVIDENCE:
            detail = ui.evidence_detail(ROOT, spec.key)
            self.assertEqual(detail["availability"], "available", detail)
            self.assertTrue(detail["recorded_match"])
            self.assertFalse(detail["fresh_execution"])
            self.assertFalse(detail["raw_evidence_revalidated"])
            self.assertEqual(detail["mismatches"], 0)
            self.assertTrue(any(key.startswith("trace_") for key in detail["hashes"]))
            self.assertEqual(len(detail["instructions"]), len(spec.instruction_indices), detail)
            self.assertTrue(all(value is False for value in detail["qualification"].values()))

    def test_full_target_cases_show_tokens_and_continuation_without_inference(self):
        value = ui.evidence_detail(ROOT, "full-target-holdouts")
        self.assertTrue(value["replay_recorded_match"])
        self.assertEqual([case["selected_token_id"] for case in value["cases"]], [76113, 75027])
        self.assertEqual([case["prediction_restored_prefix_count"] for case in value["cases"]], [0, 494])
        self.assertEqual([case["replay_restored_prefix_count"] for case in value["cases"]], [241, 0])
        self.assertEqual(len(value["instructions"]), 533)
        self.assertFalse(value["fresh_execution"])
        self.assertFalse(value["raw_evidence_revalidated"])

    def test_full_target_replay_binding_cannot_be_omitted(self):
        original = ui._json
        def load(root, path):
            value = original(root, path)
            if path.endswith("full_target_holdout_summary.json"):
                value = copy.deepcopy(value)
                value["source_replay_sha256"] = "wrong"
            return value
        with patch.object(ui, "_json", side_effect=load), patch.object(ui, "_sealed", return_value=True):
            result = ui.evidence_detail(ROOT, "full-target-holdouts")
        self.assertEqual(result["availability"], "invalid")
        self.assertFalse(result["recorded_match"])

    def test_missing_full_target_evidence_falls_back_to_prior_coverage(self):
        original = ui._json
        def load(root, path):
            if "full_target_holdout" in path:
                raise FileNotFoundError(path)
            return original(root, path)
        with patch.object(ui, "_json", side_effect=load):
            value = ui.catalog(ROOT)
        self.assertFalse(value["full_target_replay_recorded"])
        self.assertEqual([layer["index"] for layer in value["layers"] if layer["coverage"] == "connected_recorded"], [0, 1])
        self.assertEqual(value["layers"][2]["coverage"], "partial_recorded")

    def test_browser_open_is_opt_in_and_uses_bound_loopback_port(self):
        for enabled in (False, True):
            with patch.object(ui, 'DemoServer') as server_type, patch.object(ui.webbrowser, 'open') as open_browser, patch('sys.argv', ['demo', '--port', '0'] + (['--open-browser'] if enabled else [])):
                server = server_type.return_value.__enter__.return_value
                server.server_port = 12345
                server.serve_forever.side_effect = KeyboardInterrupt
                ui.main()
                if enabled:
                    open_browser.assert_called_once_with('http://127.0.0.1:12345')
                else:
                    open_browser.assert_not_called()

    def test_browser_failure_still_serves_the_demo(self):
        with patch.object(ui, 'DemoServer') as server_type, patch.object(ui.webbrowser, 'open', side_effect=ui.webbrowser.Error('no browser')), patch('sys.argv', ['demo', '--open-browser']):
            server = server_type.return_value.__enter__.return_value
            server.server_port = 12345
            server.serve_forever.side_effect = KeyboardInterrupt
            ui.main()
            server.serve_forever.assert_called_once()

    def test_missing_evidence_never_invents_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            result = ui.catalog(Path(directory))
        self.assertTrue(all(item["availability"] == "missing" for item in result["experiments"]))
        self.assertTrue(all(not item["recorded_match"] for item in result["experiments"]))
        self.assertTrue(all(layer["coverage"] == "not_shown" for layer in result["layers"]))

    def test_unknown_identifiers_and_paths_rejected(self):
        for value in ("../README.md", "/api/scenarios/normal", "two-layers/", "%2e%2e", ""):
            with self.subTest(value=value), self.assertRaises(KeyError):
                ui.evidence_detail(ROOT, value)
        with self.assertRaises(KeyError):
            ui.run_scenario(ROOT, "unknown")
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ui.DemoDataError):
            ui._file(Path(directory), "../outside")

    def test_corrupt_and_resealed_artifacts_fail_closed(self):
        original = ui._json
        spec = ui.EVIDENCE[-1]
        for field, value in (("scope", "<script>alert(1)</script>"), ("aggregate_mismatch_count", -1), ("rotary_scores_match", False), ("qualified", True)):
            def altered(root, path):
                result = original(root, path)
                if path.endswith(spec.stem + "_summary.json"):
                    result = copy.deepcopy(result)
                    result[field] = value
                    result["summary_sha256"] = hashlib.sha256(canonical_json({key: item for key, item in result.items() if key != "summary_sha256"}).encode()).hexdigest()
                return result
            with self.subTest(field=field), patch.object(ui, "_json", side_effect=altered):
                result = ui.evidence_detail(ROOT, spec.key)
            self.assertEqual(result["availability"], "invalid")
            self.assertFalse(result["recorded_match"])

    def test_invalid_program_does_not_invent_instruction_graph(self):
        original = ui._json
        def altered(root, path):
            return {"program_sha256": "wrong"} if path.endswith("execution_ir.json") else original(root, path)
        with patch.object(ui, "_json", side_effect=altered):
            result = ui.evidence_detail(ROOT, "layer-2-scores")
        self.assertTrue(result["recorded_match"])
        self.assertEqual(result["instructions"], [])
        self.assertIsNotNone(result["instruction_notice"])

    def test_scenarios_use_existing_engine_and_preserve_advisory_authority(self):
        before = (ROOT / "policies" / "oxygen_advisory.bpr").read_bytes()
        with patch("bioprocess_runtime.audit.append_decision", side_effect=AssertionError("no audit writes")):
            for item in ui.scenario_catalog()["items"]:
                result = ui.run_scenario(ROOT, item["id"])
                self.assertTrue(result["matches_expected"], item)
                self.assertTrue(result["fresh_execution"])
                self.assertFalse(result["gemma_inference"])
                self.assertFalse(result["equipment_actuation"])
                self.assertFalse(result["audit_file_written"])
                decision = result["decision"]
                self.assertEqual(decision["status"], item["expected_status"])
                self.assertTrue(decision["trace"])
                if decision["recommendation"]:
                    self.assertEqual(decision["recommendation"]["authority"], "advisory_only")
                    self.assertTrue(decision["human_review_required"])
        self.assertEqual(before, (ROOT / "policies" / "oxygen_advisory.bpr").read_bytes())

    def test_verification_log_missing_running_failure_and_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(ui.verification_status(root)["state"], "log_unavailable")
            path = root / "gemma_independent_holdout_verification_v3.log"
            path.write_text("VERIFICATION_POWERSHELL_PID=1\ntest_example ...", encoding="utf-8")
            self.assertEqual(ui.verification_status(root)["state"], "completion_not_recorded")
            path.write_text("Ran 545 tests in 1.5s\nOK\nNo broken requirements found.\nVERIFICATION_EXIT_CODE=0\n", encoding="utf-8")
            self.assertTrue(ui.verification_status(root)["complete"])
            self.assertEqual(ui.verification_status(root)["tests"], 545)
            path.write_text("ERROR: an_example\n" + path.read_text(), encoding="utf-8")
            self.assertFalse(ui.verification_status(root)["complete"])
            self.assertEqual(ui.verification_status(root)["state"], "failure_recorded")

    def test_large_file_and_non_object_json_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "small.json"
            path.write_text("[]", encoding="utf-8")
            with self.assertRaises(ui.DemoDataError):
                ui._json(root, "small.json")
            with patch.object(ui, "MAX_FILE_BYTES", 1), self.assertRaises(ui.DemoDataError):
                ui._json(root, "small.json")

    def test_assets_have_no_inline_script_or_untrusted_html_sinks(self):
        self.assertIn('src="/app.js"', ui.HTML)
        self.assertNotIn("<script>", ui.HTML)
        self.assertNotIn("innerHTML", ui.JAVASCRIPT)
        self.assertNotIn("eval(", ui.JAVASCRIPT)
        self.assertNotIn("https://", ui.HTML + ui.JAVASCRIPT + ui.CSS)
        self.assertIn("SAVED EVIDENCE", ui.HTML)
        self.assertIn("not clinical risk", ui.JAVASCRIPT)


class DemoHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ui.DemoServer(ROOT, 0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_port

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def request(self, path, method="GET", headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            connection.request(method, path, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def test_loopback_and_static_assets(self):
        self.assertEqual(self.server.server_address[0], "127.0.0.1")
        for path, content_type in (("/", "text/html"), ("/app.js", "application/javascript"), ("/app.css", "text/css")):
            status, headers, body = self.request(path)
            self.assertEqual(status, 200)
            self.assertIn(content_type, headers["Content-Type"])
            self.assertEqual(int(headers["Content-Length"]), len(body))
            self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
            self.assertIn("script-src 'self'", headers["Content-Security-Policy"])
            self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(self.request("/", "HEAD")[2], b"")

    def test_read_only_api_and_live_evaluation(self):
        status, _, body = self.request("/api/catalog")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["mode"], "saved_evidence")
        status, _, body = self.request("/api/evidence/layer-2-scores")
        self.assertEqual(status, 200)
        self.assertFalse(json.loads(body)["fresh_execution"])
        status, _, body = self.request("/api/scenarios/low_oxygen")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["decision"]["status"], "RECOMMENDATION")
        for method in ("POST", "PUT", "DELETE", "PATCH"):
            self.assertEqual(self.request("/api/scenarios/low_oxygen", method)[0], 405)

    def test_rebinding_cross_origin_and_traversal_rejected(self):
        self.assertEqual(self.request("/api/catalog", headers={"Host": "attacker.invalid"})[0], 403)
        self.assertEqual(self.request("/api/catalog", headers={"Origin": "https://attacker.invalid"})[0], 403)
        self.assertEqual(self.request("/api/catalog", headers={"Origin": f"http://127.0.0.1:{self.port}"})[0], 200)
        for path in ("/.models/config.json", "/../README.md", "/api/evidence/../../README.md", "/api/evidence/%2e%2e%2f", "/api/catalog?file=README.md", "/missing"):
            self.assertEqual(self.request(path)[0], 404, path)
        self.assertEqual(self.request("/" + "x" * 1100)[0], 414)

    def test_missing_policy_returns_no_path_or_traceback(self):
        with patch.object(ui, "run_scenario", side_effect=FileNotFoundError("sensitive local path")):
            status, _, body = self.request("/api/scenarios/normal")
        self.assertEqual(status, 503)
        self.assertNotIn(b"sensitive", body)
        self.assertNotIn(b"Traceback", body)
