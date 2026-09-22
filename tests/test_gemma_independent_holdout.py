from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from bioprocess_runtime import gemma_independent_holdout as holdout
from bioprocess_runtime.gemma_checkpoint import _load, _publish, CheckpointStore
from bioprocess_runtime.gemma_rotary_slice import _seal, _check_hash


try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
except ModuleNotFoundError:
    raise unittest.SkipTest("requires .[gemma] extras")


class GatingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.cases = [_seal({"case_id": name, "input_token_ids": [[index + 10] * 30]}, "case_sha256") for index, name in enumerate(holdout.CASE_IDS)]
        self.environment = SimpleNamespace(code="test-wrapper", context=object(), global_provider=object(), protocol=_seal({"cases": self.cases}, "protocol_sha256"), guard=Mock(), model=Mock(return_value=object()))
        _publish(self.directory / "protocol.json", self.environment.protocol)
        self.events = []
        self.failed = set()
        self.changed = set()
        self.native_failed = set()
        self.addCleanup(patch.stopall)
        patch.object(holdout.runner, "check_prediction", side_effect=lambda context, value, provider: _check_hash(value, "prediction_sha256")).start()
        self.predict = patch.object(holdout.runner, "predict", side_effect=self.prediction).start()
        self.acquire = patch.object(holdout.runner, "acquire", side_effect=self.observation).start()
        patch.object(holdout.runner, "comparison_report", side_effect=self.report).start()

    def prediction(self, context, model, ids, provider, **options):
        name = next(case["case_id"] for case in self.cases if case["input_token_ids"] == ids)
        self.events.append("predict:" + name)
        failed = name in self.failed
        return _seal({"execution": {"input_token_ids": ids, "target": "selected_token_id", "vocabulary_candidate_enabled": True,
                                     "state_retention": "boundaries", "completed_instruction_count": 4 if failed else 533,
                                     "status": "abstained" if failed else "complete_uncompared", "abstention": {"domain": "failure"} if failed else None,
                                     "checkpoint_resume": None, "stored_boundary_predictions_used": False, "value": 2 if name in self.changed else 1}}, "prediction_sha256")

    def report(self, context, prediction, observations, guards):
        matched = all(value["match"] for value in observations)
        return _seal({"prediction_sha256": prediction["prediction_sha256"], "observations": observations, "acquisition_guards": guards,
                      "match": matched, "original_forward_count": 3, "selected_token_id": 7 if matched else None,
                      "aggregate_mismatch_count": 0 if matched else 1, "qualified": False}, "report_sha256")

    def observation(self, context, model, prediction, provider, path):
        for name in holdout.CASE_IDS:
            self.assertTrue(holdout.prediction_path(self.directory, name).is_file())
        self.assertTrue((self.directory / "plan.json").is_file())
        name = path.name.split(".")[0]
        self.events.append("native:" + name)
        return self.report(context, prediction, [{"match": name not in self.native_failed}] * 3, {"source_unchanged": True})

    def plan(self):
        predictions, _ = holdout.predict_all(self.environment, self.directory)
        plan = holdout.plan_body(self.environment, predictions)
        _publish(self.directory / "plan.json", plan)
        return plan

    def test_both_complete_predictions_are_frozen_before_first_native_call(self):
        plan = self.plan()
        report = holdout.acquire_all(self.environment, self.directory, plan)
        self.assertEqual(self.events, ["predict:distinct_tokens", "predict:repeated_motif", "native:distinct_tokens", "native:repeated_motif"])
        self.assertTrue(report["match"])
        self.assertEqual(report["recorded_forward_count"], 6)
        self.assertFalse(report["qualified"])

    def test_first_abstention_does_not_drop_second_case_and_blocks_all_native(self):
        self.failed.add("distinct_tokens")
        plan = self.plan()
        models = self.environment.model.call_count
        report = holdout.acquire_all(self.environment, self.directory, plan)
        self.assertEqual(self.predict.call_count, 2)
        self.assertFalse(plan["prediction_complete"])
        self.assertEqual(report["recorded_forward_count"], 0)
        self.acquire.assert_not_called()
        self.assertEqual(self.environment.model.call_count, models)

    def test_corrupt_or_missing_second_prediction_blocks_first_native_call(self):
        plan = self.plan()
        path = holdout.prediction_path(self.directory, "repeated_motif")
        path.write_text('{"corrupt":true}', encoding="utf-8")
        with self.assertRaises(ValueError):
            holdout.acquire_all(self.environment, self.directory, plan)
        self.acquire.assert_not_called()

    def test_resealed_plan_cannot_turn_abstention_into_completion(self):
        self.failed.add("repeated_motif")
        plan = self.plan()
        altered = _seal({**{key: value for key, value in plan.items() if key != "plan_sha256"}, "prediction_complete": True}, "plan_sha256")
        (self.directory / "plan.json").write_text(json.dumps(altered))
        with self.assertRaises(ValueError):
            holdout.acquire_all(self.environment, self.directory, altered)
        self.acquire.assert_not_called()

    def test_wrong_case_order_partial_target_and_options_rejected(self):
        predictions, _ = holdout.predict_all(self.environment, self.directory)
        with self.assertRaises(ValueError):
            holdout.plan_body(self.environment, list(reversed(predictions)))
        for key, value in (("target", "hidden.2"), ("completed_instruction_count", 63), ("vocabulary_candidate_enabled", False)):
            changed = copy.deepcopy(predictions[0])
            changed["execution"][key] = value
            changed = _seal({k: v for k, v in changed.items() if k != "prediction_sha256"}, "prediction_sha256")
            with self.subTest(key=key), self.assertRaises(ValueError):
                holdout.plan_body(self.environment, [changed, predictions[1]])

    def test_source_guard_failure_blocks_native(self):
        plan = self.plan()
        self.environment.guard.side_effect = ValueError("changed sources")
        with self.assertRaisesRegex(ValueError, "changed sources"):
            holdout.acquire_all(self.environment, self.directory, plan)
        self.acquire.assert_not_called()

    def test_prediction_resume_reuses_only_valid_completed_case(self):
        original = self.predict.side_effect
        def interrupt(*args, **kwargs):
            if args[2] == self.cases[1]["input_token_ids"]:
                raise RuntimeError("interrupted second case")
            return original(*args, **kwargs)
        self.predict.side_effect = interrupt
        with self.assertRaises(RuntimeError):
            holdout.predict_all(self.environment, self.directory)
        self.predict.side_effect = original
        self.predict.reset_mock()
        predictions, reused = holdout.predict_all(self.environment, self.directory, resume=True)
        self.assertEqual(reused, ["distinct_tokens"])
        self.assertEqual(self.predict.call_count, 1)
        self.assertTrue(holdout.plan_body(self.environment, predictions)["prediction_complete"])

    def test_implicit_prediction_reuse_and_overwrite_are_rejected(self):
        self.plan()
        self.predict.reset_mock()
        with self.assertRaises(ValueError):
            holdout.predict_all(self.environment, self.directory)
        self.predict.assert_not_called()
        path = self.directory / "preserve.json"
        _publish(path, {"preserve": True})
        with self.assertRaises(FileExistsError):
            _publish(path, {"preserve": False})
        self.assertTrue(_load(path)["preserve"])

    def test_native_failure_preserved_and_remaining_case_acquired(self):
        self.native_failed.add("distinct_tokens")
        report = holdout.acquire_all(self.environment, self.directory, self.plan())
        self.assertFalse(report["match"])
        self.assertEqual(self.acquire.call_count, 2)
        self.assertEqual(report["cases"][0]["mismatch_count"], 1)

    def test_native_resume_does_not_repeat_completed_observations(self):
        plan = self.plan()
        original = self.acquire.side_effect
        def interrupt(*args):
            if args[-1].name.startswith("repeated_motif"):
                raise RuntimeError("interrupted native")
            return original(*args)
        self.acquire.side_effect = interrupt
        with self.assertRaises(RuntimeError):
            holdout.acquire_all(self.environment, self.directory, plan)
        self.acquire.side_effect = original
        self.acquire.reset_mock()
        report = holdout.acquire_all(self.environment, self.directory, plan, resume=True)
        self.assertEqual(self.acquire.call_count, 1)
        self.assertEqual(report["fresh_forward_count"], 3)
        self.assertEqual(report["restored_native_cases"], ["distinct_tokens"])

    def test_replay_recomputes_both_cases_before_any_new_native(self):
        plan = self.plan()
        holdout.acquire_all(self.environment, self.directory, plan)
        self.events.clear()
        result = holdout.replay_all(self.environment, self.directory, plan)
        self.assertEqual(self.events, ["predict:distinct_tokens", "predict:repeated_motif", "native:distinct_tokens", "native:repeated_motif"])
        self.assertTrue(result["valid"])
        self.assertTrue(result["match"])

    def test_replay_prediction_mismatch_blocks_all_replay_forwards(self):
        plan = self.plan()
        holdout.acquire_all(self.environment, self.directory, plan)
        self.acquire.reset_mock()
        self.changed.add("repeated_motif")
        result = holdout.replay_all(self.environment, self.directory, plan)
        self.assertFalse(result["valid"])
        self.assertEqual(result["native_forward_count"], 0)
        self.acquire.assert_not_called()

    def test_plan_change_during_native_prevents_report_publication(self):
        plan = self.plan()
        original = self.acquire.side_effect
        def change(*args):
            result = original(*args)
            (self.directory / "plan.json").write_text('{}')
            return result
        self.acquire.side_effect = change
        with self.assertRaises(ValueError):
            holdout.acquire_all(self.environment, self.directory, plan)
        self.assertFalse((self.directory / "distinct_tokens.report.json").exists())

    def test_cli_runs_one_gated_batch_and_rejects_implicit_directory_reuse(self):
        root = self.directory
        self.directory = root / "artifacts" / "batch"
        with patch.object(holdout, "ROOT", root), patch.object(holdout, "Environment", return_value=self.environment), patch("sys.argv", ["holdout", "run", "--directory", str(self.directory)]):
            self.assertEqual(holdout.main(), 0)
            with self.assertRaises(ValueError):
                holdout.main()
        self.assertTrue(_load(self.directory / "report.json")["match"])
        self.assertTrue((self.directory / "binding.json").is_file())

    def test_cli_rejects_unsafe_output_before_source_or_model_loading(self):
        with patch.object(holdout, "Environment", side_effect=AssertionError("no context")), patch("sys.argv", ["holdout", "run", "--directory", str(holdout.ROOT / ".models")]), self.assertRaises(ValueError):
            holdout.main()

    def test_cli_resume_requires_an_existing_run_directory(self):
        root = self.directory
        with patch.object(holdout, "ROOT", root), patch.object(holdout, "Environment", side_effect=AssertionError("no fresh run on resume")), patch("sys.argv", ["holdout", "run", "--directory", str(root / "artifacts" / "missing"), "--resume"]), self.assertRaises(ValueError):
            holdout.main()

    def test_replay_frozen_prediction_mutation_prevents_success(self):
        plan = self.plan()
        holdout.acquire_all(self.environment, self.directory, plan)
        original = self.acquire.side_effect
        def change(*args):
            result = original(*args)
            if args[-1].name.startswith("repeated_motif"):
                holdout.prediction_path(self.directory, "repeated_motif", True).write_text('{}')
            return result
        self.acquire.side_effect = change
        with self.assertRaises(ValueError):
            holdout.replay_all(self.environment, self.directory, plan)
        self.assertFalse((self.directory / "repeated_motif.replay.report.json").exists())

    def test_outer_lease_rejects_changed_orchestrator_binding(self):
        directory = self.directory / "lease"
        with CheckpointStore(directory, {"orchestrator": "first"}) as store:
            store.save({"completed_instruction_count": 0}, {})
            with self.assertRaises(OSError):
                with CheckpointStore(directory, {"orchestrator": "first"}, resume=True):
                    pass
        with self.assertRaises(ValueError):
            with CheckpointStore(directory, {"orchestrator": "changed"}, resume=True):
                pass


class DeclarationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = holdout.ROOT
        if not (root / "artifacts/gemma3_270m_independent_baseline_prediction_v3.json").is_file():
            raise unittest.SkipTest("Ignored baseline evidence unavailable")
        cls.values = [_load(root / holdout.PINNED_FILES[name][0]) for name in ("declaration", "revision2", "revision3")]
        cls.program = _load(root / "results/gemma3_270m_execution_ir.json")
        cls.ids = _load(root / "artifacts/gemma3_270m_independent_baseline_prediction_v3.json")["execution"]["input_token_ids"]
        cls.model_path = root / ".models/gemma-3-270m-it"
        cls.priors = {name: _load(root / path) for name, path in (("first_layer", "results/gemma3_270m_first_layer_holdout_protocol.json"), ("two_layers", "results/gemma3_270m_two_layers_holdout_protocol.json"))}
        cls.provider = _load(root / "results/gemma3_270m_independent_global_rotary_v3.json")

    def validate(self, values=None):
        return holdout.validate_declaration(*(values or self.values), self.program, self.ids, self.model_path, self.priors, self.provider)

    def test_real_tokenizer_exclusions_and_exact_case_regeneration(self):
        with patch.object(holdout.runner, "predict", side_effect=AssertionError("no prediction")), patch.object(holdout.runner, "acquire", side_effect=AssertionError("no native")):
            cases = self.validate()
        self.assertEqual(cases, self.values[0]["cases"])
        self.assertEqual(len(set(cases[0]["input_token_ids"][0] + cases[1]["input_token_ids"][0])), 35)

    def test_changed_declaration_fields_rejected_without_resampling(self):
        for key, value in (("baseline_sequence_sha256", "wrong"), ("cases", []), ("prior_excluded_ids", []), ("final_pool_count", 0), ("tokenizer", {})):
            values = copy.deepcopy(self.values)
            values[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(values)
        values = copy.deepcopy(self.values)
        values[0]["generator"]["seed"] += 1
        with self.assertRaises(ValueError):
            self.validate(values)
        values = copy.deepcopy(self.values)
        values[2]["bindings"]["runner_code_sha256"] = "wrong"
        with self.assertRaises(ValueError):
            self.validate(values)

    def test_real_complete_environment_preflight_has_no_model_forward(self):
        with patch.object(holdout.Environment, "model", side_effect=AssertionError("no model")), patch.object(holdout.runner, "predict", side_effect=AssertionError("no prediction")), patch.object(holdout.runner, "acquire", side_effect=AssertionError("no native")):
            environment = holdout.Environment()
        self.assertEqual(environment.protocol["cases"], self.values[0]["cases"])
        self.assertEqual(environment.protocol["baseline_report_sha256"], holdout.BASELINE_REPORT)
        self.assertFalse(environment.protocol["qualified"])

    def test_missing_receipt_or_declaration_blocks_context_loading(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(holdout.runner, "load_context", side_effect=AssertionError("must fail before context")), self.assertRaises(FileNotFoundError):
            holdout.Environment(Path(directory))


if __name__ == "__main__":
    unittest.main()
