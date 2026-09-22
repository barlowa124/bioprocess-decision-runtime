from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from bioprocess_runtime import gemma_independent as engine
from bioprocess_runtime import gemma_independent_run as runner
from bioprocess_runtime.gemma_rotary_slice import _sha, _seal
from test_gemma_independent import synthetic
from test_gemma_two_layers import synthetic_context
from test_gemma_first_layer import profiles
import test_gemma_third_layer_entry as entry_tests

ROOT = Path(__file__).resolve().parents[1]


try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
except ModuleNotFoundError:
    raise unittest.SkipTest("requires .[gemma] extras")


class VocabularyEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from bioprocess_runtime.gemma_checkpoint import _load
        names = ("gemma_vocab_validation_plan_v1.json", "gemma_vocab_validation_bundle_v1.json", "gemma_vocab_validation_report_v1.json", "gemma_vocab_replay_v1.json", "gemma_vocab_native_probe_v1.json")
        paths = [ROOT / "artifacts" / name for name in names]
        if any(not path.is_file() for path in paths):
            raise unittest.SkipTest("Ignored GEMV component artifacts unavailable")
        cls.values = [_load(path) for path in paths]
        cls.runtime = cls.values[-1]["runtime"]

    def test_pinned_gemv_component_and_runtime_load_without_native_calls(self):
        with patch.object(runner, "model_parameters", side_effect=AssertionError("no native")), patch.object(engine.gemv, "project_bits", side_effect=AssertionError("no new arithmetic")):
            binding = runner.load_vocabulary_evidence(ROOT, self.runtime)
        self.assertEqual(binding["report_sha256"], engine.vocabulary_evidence()["report_sha256"])
        self.assertFalse(binding["full_model_qualified"])
        with self.assertRaisesRegex(ValueError, "runtime"):
            runner.load_vocabulary_evidence(ROOT, {"wrong_runtime": True})

    def test_corrupt_or_resealed_component_is_not_accepted(self):
        for reseal in (False, True):
            values = copy.deepcopy(self.values)
            values[2]["records"][0]["mismatch_count"] = 1
            if reseal:
                values[2] = _seal({key: value for key, value in values[2].items() if key != "report_sha256"}, "report_sha256")
            with self.assertRaises(ValueError):
                runner.check_vocabulary_evidence(*values, self.runtime)

    def test_semantic_guards_beyond_packet_hashes(self):
        for kind in ("case", "duplicate", "count", "control", "descriptor", "match", "source", "replay", "arithmetic"):
            values = copy.deepcopy(self.values)
            plan, bundle, report, replay, probe = values
            if kind == "case":
                bundle["predictions"].pop("dense_wide")
            elif kind == "duplicate":
                report["records"][1] = copy.deepcopy(report["records"][0])
            elif kind == "count":
                report["records"][0]["mismatch_count"] = 1
            elif kind == "control":
                report["records"][0]["profiled_plain_mismatch_count"] = 1
            elif kind == "descriptor":
                report["records"][0]["native_descriptor"]["sha256"] = "wrong"
            elif kind == "match":
                report["match"] = False
            elif kind == "source":
                plan["source_file_sha256"]["arithmetic"] = "wrong"
            elif kind == "replay":
                replay["fresh_integer_prediction_match"] = False
            else:
                plan["code_sha256"] = "wrong"
            with self.subTest(kind=kind), patch.object(runner, "_check_hash"), self.assertRaises(ValueError):
                runner.check_vocabulary_evidence(*values, self.runtime)

    def test_missing_component_artifacts_fail_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, "model_parameters", side_effect=AssertionError("no native")), self.assertRaises(FileNotFoundError):
            runner.load_vocabulary_evidence(Path(directory), self.runtime)


class IndependentRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.program, cls.ids, cls.snapshots, cls.providers = synthetic_context()
        cls.context = runner.Context(cls.program, cls.providers, profiles(), {"synthetic": True}, {"synthetic_sources": True})
        with synthetic(cls.program):
            cls.execution = engine.execute(cls.program, cls.ids, cls.snapshots, cls.providers, profiles(), cls.context.runtime, target="hidden.2", workers=1)
        cls.prediction = _seal({"schema_version": 3, "kind": runner.PREDICTION_KIND, "runner_code_sha256": "synthetic-runner", "sources": cls.context.sources,
                                "parameter_snapshots": runner._snapshot_metadata(cls.snapshots), "global_provider_sha256": None, "execution": cls.execution,
                                "scope": runner.PREDICTION_SCOPE}, "prediction_sha256")

    def test_saved_execution_checker_rejects_changed_boundaries_auxiliary_and_flags(self):
        with synthetic(self.program):
            engine.check_execution(self.program, self.execution)
            for kind in ("state", "root", "flag", "rms", "token", "count"):
                value = copy.deepcopy(self.execution)
                if kind == "state":
                    value["state_bits"]["hidden.2"][0][-1][-1] ^= 1
                elif kind == "root":
                    value["trace_root"] = "0" * 64
                elif kind == "flag":
                    value["qualified"] = True
                elif kind == "rms":
                    value["scalar_stages"][next(iter(value["scalar_stages"]))]["mean_bits"][0] ^= 1
                elif kind == "token":
                    value["selected_token_id"] = 1
                else:
                    value["completed_instruction_count"] -= 1
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    engine.check_execution(self.program, value)

    def test_prediction_binding_and_scope(self):
        with synthetic(self.program), patch.object(runner, "code_sha256", return_value="synthetic-runner"):
            runner.check_prediction(self.context, self.prediction, None)
            changed = copy.deepcopy(self.prediction)
            changed["global_provider_sha256"] = "invented"
            changed = _seal({key: value for key, value in changed.items() if key != "prediction_sha256"}, "prediction_sha256")
            with self.assertRaises(ValueError):
                runner.check_prediction(self.context, changed, None)

    def observed(self):
        states = {name: self.execution["state_bits"][name] for name in ("hidden.1", "hidden.2")}
        return {"state_bits": copy.deepcopy(states), "descriptors": {name: engine.first._descriptor(np.asarray(value, dtype=np.uint16)) for name, value in states.items()},
                "selected_token_id": None, "runtime": self.context.runtime, "runner_code_sha256": "synthetic-runner",
                "native_scope": "original_forward_with_read_only_boundary_hooks_not_all_internal_states", "stopped_at_target": True}

    def test_compare_every_boundary_and_retain_numerical_failure(self):
        observations = [self.observed() for _ in range(3)]
        guards = dict.fromkeys(("source_unchanged", "checkpoint_unchanged", "frozen_files_unchanged"), True)
        report = runner.comparison_report(self.context, self.prediction, observations, guards)
        self.assertTrue(report["match"])
        self.assertFalse(report["qualified"])
        self.assertEqual(report["original_forward_count"], 3)
        observations[1]["state_bits"]["hidden.1"][0][0][0] ^= 1
        report = runner.comparison_report(self.context, self.prediction, observations, guards)
        self.assertFalse(report["match"])
        self.assertEqual(report["first_divergence"]["state"], "hidden.1")
        self.assertGreater(report["aggregate_mismatch_count"], 0)
        with self.assertRaises(ValueError):
            runner.comparison_report(self.context, self.prediction, observations[:2], guards)

    def test_abstention_cannot_trigger_native_evidence(self):
        prediction = copy.deepcopy(self.prediction)
        prediction["execution"]["status"] = "abstained"
        prediction["execution"]["abstention"] = {"instruction_id": "i0003", "message": "provider required"}
        blocked = runner.comparison_report(self.context, prediction, [], {})
        self.assertEqual(blocked["original_forward_count"], 0)
        self.assertFalse(blocked["match"])
        with self.assertRaises(ValueError):
            runner.comparison_report(self.context, prediction, [self.observed()], {})

    def test_model_prediction_freeze_and_acquisition_order(self):
        calls = []
        def snapshot(*args):
            calls.append("snapshot")
            return self.snapshots
        with synthetic(self.program), patch.object(runner, "code_sha256", return_value="synthetic-runner"), patch.object(runner, "snapshot_model", side_effect=snapshot):
            value = runner.predict(self.context, None, self.ids, None, target="hidden.2", workers=1)
        self.assertEqual(calls, ["snapshot", "snapshot"])
        self.assertEqual(value, self.prediction)
        with patch.object(runner, "require_frozen", side_effect=ValueError("not frozen")), patch.object(runner, "capture_boundaries", side_effect=AssertionError("no native")), self.assertRaises(ValueError):
            runner.acquire(self.context, None, self.prediction, None, ROOT)

    def test_cli_operations_and_no_implicit_vocabulary_candidate(self):
        for operation in ("calibrate", "predict", "compare", "run", "verify"):
            args = runner.build_parser().parse_args([operation])
            self.assertFalse(args.allow_vocabulary_candidate)
            self.assertEqual(args.target, "selected_token_id")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prediction.json"
            path.write_text("preserve", encoding="utf-8")
            with patch("sys.argv", ["runner", "predict", "--prediction", str(path)]), patch.object(runner, "load_context", side_effect=AssertionError("no source load")), self.assertRaises(ValueError):
                runner.main()

    def test_real_source_context_preflight_no_model_prediction(self):
        required = [ROOT / "artifacts/gemma3_270m_third_layer_scores_report.json", ROOT / ".models/gemma-3-270m-it/config.json"]
        if any(not path.is_file() for path in required):
            self.skipTest("Ignored source evidence/checkpoint unavailable")
        with patch.object(engine, "execute", side_effect=AssertionError("no new numerical prediction")), patch.object(runner, "model_parameters", side_effect=AssertionError("no model")):
            context = runner.load_context(ROOT, ROOT / ".models/gemma-3-270m-it")
        self.assertEqual(context.sources["score_report_sha256"], runner.SCORE_REPORT)
        self.assertEqual(context.program["configuration"]["layers"], 18)


class NativeBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        entry_tests.NativeCaptureTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        entry_tests.NativeCaptureTests.tearDownClass.__func__(cls)

    def test_global_calibration_uses_primitive_only_and_binds_frequency(self):
        from test_gemma_first_layer import real_program
        program = real_program()
        frequency = self.model.model.rotary_emb.inv_freq.float()
        bits = runner._bits(frequency)
        program["parameter_commitments"][engine.GLOBAL_FREQUENCY]["sha256"] = runner.first._descriptor(bits)["sha256"]
        context = SimpleNamespace(program=program, runtime={"synthetic": True}, sources={"synthetic": True})
        with patch.object(self.model.model.rotary_emb, "inv_freq", frequency), patch.object(runner, "code_sha256", return_value="synthetic"), patch.object(runner, "model_parameters", return_value={engine.GLOBAL_FREQUENCY: frequency}), patch.object(self.model, "forward", side_effect=AssertionError("no model forward during primitive calibration")):
            packet = runner.calibrate_global_rotary(context, self.model)
            runner.check_global_binding(context, packet)
        self.assertFalse(packet["hidden_states_used"])
        self.assertEqual(packet["native_repeat_count"], 3)
        self.assertTrue(packet["native_repeat_exact"])
        self.assertEqual(packet["positions"], list(range(30)))
        engine.check_global_rotary(packet, program, bits, context.runtime)
        with patch.object(runner, "code_sha256", return_value="different"), self.assertRaises(ValueError):
            runner.check_global_binding(context, packet)

    def test_original_prefix_boundary_capture_restores_hooks(self):
        from test_gemma_first_layer import real_program
        before = [(module.forward, dict(module._forward_hooks), dict(module._forward_pre_hooks)) for module in self.model.modules()]
        context = SimpleNamespace(program=real_program())
        with patch.object(runner, "code_sha256", return_value="synthetic"), patch.object(runner.first, "_runtime", return_value={}):
            observed = runner.capture_boundaries(context, self.model, self.ids, "hidden.2")
        self.assertEqual(set(observed["state_bits"]), {"hidden.1", "hidden.2"})
        self.assertTrue(observed["stopped_at_target"])
        self.assertIsNone(observed["selected_token_id"])
        self.assertEqual(before, [(module.forward, dict(module._forward_hooks), dict(module._forward_pre_hooks)) for module in self.model.modules()])
