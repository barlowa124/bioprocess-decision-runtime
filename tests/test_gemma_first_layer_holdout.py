from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from bioprocess_runtime import cli
from bioprocess_runtime import gemma_first_layer as first
from bioprocess_runtime import gemma_first_layer_capture as capture
from bioprocess_runtime import gemma_first_layer_holdout as holdout
from bioprocess_runtime.gemma_rotary_slice import _sha, _seal
from test_gemma_first_layer import ROOT, real_program, profiles, synthetic_context, execute_synthetic, native_pair


try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
except ModuleNotFoundError:
    raise unittest.SkipTest("requires .[gemma] extras")


class FakeTokenizer:
    all_special_ids = [1, 2, 300000]
    init_kwargs = {"secret": "never serialize", "chat_template": "never serialize"}

    def get_vocab(self):
        return {**{str(index): index for index in range(130)}, "outside": 300000, "negative": -1, "alias": 70}


def fake_baseline():
    program = real_program()
    ids = [list(range(30))]
    providers = SimpleNamespace(commitments=lambda: {"synthetic_provider": True})
    source = SimpleNamespace(program=program, runtime={"synthetic": True}, commitments=lambda: {"synthetic_sources": True},
                             validate=lambda: (ids, providers, profiles()))
    kernels = {role: ["synthetic_" + role] for role in first._native_kernel_roles(program)}
    plan = {"plan_sha256": holdout.BASELINE_PLAN_SHA256, "scope": first.SCOPE, "input_token_ids": ids, "model_binding": {"synthetic_checkpoint": True}}
    report = {"observations": [{"traced": {"kernels": kernels}} for _ in range(3)]}
    return holdout.BaselineSources(source, plan, {}, report)


def reseal(value, field):
    return _seal({key: item for key, item in value.items() if key != field}, field)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        for name in holdout.ASSETS:
            (self.path / name).write_text('{"synthetic":true}', encoding="utf-8")
        self.baseline = fake_baseline()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.tokenizer = self.stack.enter_context(patch("transformers.AutoTokenizer.from_pretrained", return_value=FakeTokenizer()))
        self.checked = self.stack.enter_context(patch.object(first, "verify_first_layer", return_value={"valid": True, "first_layer_matches": True, "connected_first_layer_independently_recomputed": True}))

    def protocol(self):
        return holdout.build_holdout_protocol(self.baseline, self.path)

    def test_deterministic_disjoint_raw_id_families_and_public_metadata(self):
        protocol = self.protocol()
        self.assertEqual(protocol, self.protocol())
        holdout._protocol_check(self.baseline, protocol, self.path)
        distinct, repeated = (case["input_token_ids"][0] for case in protocol["cases"])
        self.assertEqual([case["case_id"] for case in protocol["cases"]], list(holdout.CASE_IDS))
        self.assertEqual(len(distinct), 30)
        self.assertEqual(len(set(distinct)), 30)
        self.assertEqual(len(repeated), 30)
        self.assertEqual(len(set(repeated)), 5)
        self.assertEqual(repeated, repeated[:5] * 6)
        self.assertEqual(distinct, [94, 89, 34, 96, 39, 41, 71, 52, 93, 90, 92, 91, 81, 117, 48, 107, 129, 98, 44, 82, 73, 50, 120, 38, 128, 84, 119, 64, 33, 113])
        self.assertEqual(repeated[:5], [105, 67, 124, 99, 35])
        self.assertFalse(set(distinct) & set(repeated))
        self.assertFalse((set(distinct) | set(repeated)) & set(range(30)))
        self.assertFalse((set(distinct) | set(repeated)) & set(FakeTokenizer.all_special_ids))
        self.assertTrue(all(0 <= value < 262144 for value in distinct + repeated))
        self.assertEqual(protocol["tokenizer"]["allowed_pool_count"], 100)
        self.assertEqual(set(protocol["tokenizer"]["assets_sha256"]), set(holdout.ASSETS))
        self.assertNotIn("init_kwargs", json.dumps(protocol))
        self.assertNotIn("never serialize", json.dumps(protocol))
        self.tokenizer.assert_called_with(str(self.path), local_files_only=True, use_fast=True, trust_remote_code=False)
        self.assertEqual(protocol["coverage_per_case"]["instruction_count"], 34)
        self.assertEqual(protocol["coverage_per_case"]["state_count"], 36)
        self.assertEqual(protocol["coverage_per_case"]["rms_scalar_positions"], 810)
        self.assertEqual(protocol["coverage_per_case"]["softmax_fp32_positions"], 3600)
        self.assertEqual(protocol["original_forward_count"], 12)
        self.assertFalse(any(protocol[key] for key in holdout.FALSE_FLAGS))

    def test_protocol_never_predicts_binds_checkpoint_or_captures(self):
        with patch.object(first, "execute_first_layer", side_effect=AssertionError("prediction forbidden")), patch.object(first, "_snapshot_model", side_effect=AssertionError("model forbidden")), patch.object(capture, "capture_first_layer", side_effect=AssertionError("native forbidden")), patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=AssertionError("load forbidden")):
            self.protocol()

    def test_resealed_seed_cases_order_overlap_and_source_changes_rejected(self):
        original = self.protocol()
        mutations = [lambda p: p.update(seed=1), lambda p: p["cases"].reverse(), lambda p: p["cases"].pop(),
                     lambda p: p["cases"][0]["input_token_ids"][0].__setitem__(0, 1),
                     lambda p: p["cases"][0]["input_token_ids"][0].__setitem__(0, p["cases"][1]["input_token_ids"][0][0]),
                     lambda p: p["cases"][0]["input_token_ids"][0].__setitem__(0, 300000),
                     lambda p: p["cases"][1]["input_token_ids"][0].reverse(), lambda p: p.update(code_sha256="different"),
                     lambda p: p["profiles"].update(serial="refitted"), lambda p: p["runtime"].update(changed=True)]
        for mutate in mutations:
            changed = copy.deepcopy(original)
            mutate(changed)
            for case in changed["cases"]:
                case["case_sha256"] = _sha({key: value for key, value in case.items() if key != "case_sha256"})
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                holdout._protocol_check(self.baseline, reseal(changed, "protocol_sha256"), self.path)

    def test_asset_and_actual_vocab_special_set_changes_rejected(self):
        protocol = self.protocol()
        (self.path / "tokenizer.json").write_text("changed", encoding="utf-8")
        with self.assertRaises(ValueError):
            holdout._protocol_check(self.baseline, protocol, self.path)
        (self.path / "tokenizer.json").write_text('{"synthetic":true}', encoding="utf-8")
        with patch.object(FakeTokenizer, "all_special_ids", [1, 2, 33]), self.assertRaises(ValueError):
            holdout._protocol_check(self.baseline, protocol, self.path)
        with patch.object(FakeTokenizer, "get_vocab", return_value={"only": 10}), self.assertRaisesRegex(ValueError, "35"):
            self.protocol()
        with patch.object(FakeTokenizer, "get_vocab", return_value={"bad": True}), self.assertRaisesRegex(ValueError, "integers"):
            self.protocol()

    def test_baseline_failed_disconnected_wrong_pin_scope_and_runtime_profile_guards(self):
        for name in ("valid", "first_layer_matches", "connected_first_layer_independently_recomputed"):
            self.checked.return_value = {"valid": True, "first_layer_matches": True, "connected_first_layer_independently_recomputed": True, name: False}
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "passing full baseline"):
                self.protocol()
        self.checked.return_value = {"valid": True, "first_layer_matches": True, "connected_first_layer_independently_recomputed": True}
        for name, value in (("plan_sha256", "wrong"), ("scope", holdout.SCOPE)):
            with patch.dict(self.baseline.baseline_plan, {name: value}), self.assertRaisesRegex(ValueError, "fixed-case"):
                self.protocol()
        for failure in ("Source primitive runtimes disagree", "Unsupported first-layer frozen arithmetic profiles"):
            with patch.object(self.baseline.first_sources, "validate", side_effect=ValueError(failure)), self.assertRaisesRegex(ValueError, failure):
                self.protocol()

    def test_baseline_validation_cache_is_transparent_and_invocation_local(self):
        source, adapters, contexts = self.baseline.first_sources, [], []

        def verify(adapter, plan, bundle, report):
            self.assertIsNot(adapter, source)
            self.assertIs(adapter.program, source.program)
            self.assertIs(adapter.runtime, source.runtime)
            self.assertEqual(adapter.commitments(), source.commitments())
            self.assertIs(plan, self.baseline.baseline_plan)
            self.assertIs(bundle, self.baseline.baseline_bundle)
            self.assertIs(report, self.baseline.baseline_report)
            contexts.append(adapter.validate())
            adapters.append(adapter)
            return {"valid": True, "first_layer_matches": True, "connected_first_layer_independently_recomputed": True}

        self.checked.side_effect = verify
        with patch.object(source, "validate", wraps=source.validate) as validate:
            first_context = self.baseline.validate()
            validate.assert_called_once_with()
            second_context = self.baseline.validate()
            self.assertEqual(validate.call_count, 2)
        self.assertEqual(self.checked.call_count, 2)
        self.assertIs(first_context, contexts[0])
        self.assertIs(second_context, contexts[1])
        self.assertIsNot(adapters[0], adapters[1])
        self.assertIsNot(first_context, second_context)

    def test_cached_baseline_context_does_not_hide_source_change(self):
        source = self.baseline.first_sources
        commitment = {"synthetic_sources": True}

        def verify(adapter, plan, bundle, report):
            adapter.validate()
            commitment["changed"] = True
            return {"valid": True, "first_layer_matches": True, "connected_first_layer_independently_recomputed": True}

        self.checked.side_effect = verify
        with patch.object(source, "commitments", side_effect=lambda: dict(commitment)), patch.object(source, "validate", wraps=source.validate) as validate:
            with self.assertRaisesRegex(ValueError, "Baseline source changed during validation"):
                self.baseline.validate()
            validate.assert_called_once_with()
        self.checked.assert_called_once()

    def test_code_drift_and_old_public_source_case_guard(self):
        with patch.object(holdout, "SOURCE_SHA256", "changed"), self.assertRaisesRegex(ValueError, "source changed"):
            holdout._code_sha()
        with patch.object(first, "_code_sha", side_effect=ValueError("core source changed")), self.assertRaisesRegex(ValueError, "core source"):
            holdout._code_sha()
        with patch.object(first, "_snapshot_model", side_effect=AssertionError("must not bind")), self.assertRaisesRegex(ValueError, "restricted to the declared source"):
            first.build_first_layer_plan(self.baseline.first_sources, object(), input_token_ids=[list(range(30, 60))])

    def test_exclusive_protocol_file_and_missing_freeze_blocks_prediction(self):
        protocol = self.protocol()
        target = self.path / "protocol.json"
        with patch.object(first, "execute_first_layer", side_effect=AssertionError("not frozen")), self.assertRaises(ValueError):
            holdout.build_holdout_plan(self.baseline, object(), protocol, self.path, target)
        holdout.write_new(target, protocol)
        holdout.require_frozen(target, protocol)
        with self.assertRaises(FileExistsError):
            holdout.write_new(target, protocol)


class SyntheticExecutionTests(unittest.TestCase):
    def test_unchanged_snapshot_model_checks_context_and_full_parameter_binding_twice(self):
        baseline, model = fake_baseline(), object()
        baseline.first_sources.post_feedforward = object()
        ids = holdout._cases(list(range(30, 130)))[0]["input_token_ids"]
        with patch.object(first.ff, "_model_context") as context, patch.object(first, "bind_model_tensors", return_value={"synthetic": True}) as binding, patch.object(first, "snapshot_parameters", return_value={"synthetic": True}) as snapshots:
            actual = first._snapshot_model(baseline.first_sources, model, ids)
        self.assertEqual(actual, {"synthetic": True})
        self.assertEqual(context.call_count, 2)
        self.assertEqual(binding.call_count, 2)
        self.assertTrue(all(call.kwargs == {"verify_hashes": True} for call in binding.call_args_list))
        snapshots.assert_called_once_with(baseline.first_sources.program, ids, {"synthetic": True})

    def test_unchanged_engine_fresh_per_case_and_snapshot_root_guards(self):
        program, ids, snapshots, providers = synthetic_context()
        cases = holdout._cases(list(range(30, 130)))
        roots = []
        for case in cases:
            selected = dict(snapshots)
            selected[first.EMBEDDING] = {**snapshots[first.EMBEDDING], "token_indices": case["input_token_ids"][0]}
            execution = execute_synthetic(program, case["input_token_ids"], selected, providers)
            self.assertEqual(len(execution["records"]), 34)
            self.assertEqual(len(execution["state_bits"]), 36)
            roots.append(execution["records"][-1]["record_hash"])
            with patch.object(first, "PROGRAM_SHA256", program["program_sha256"]), self.assertRaisesRegex(ValueError, "token/commitment"):
                first.check_parameter_snapshots(program, ids, selected)
        self.assertNotEqual(*roots)


class OrderingTests(unittest.TestCase):
    protocol = ProtocolTests.protocol

    def setUp(self):
        ProtocolTests.setUp(self)
        self.frozen = self.path / "protocol.json"
        self.declaration = self.protocol()
        holdout.write_new(self.frozen, self.declaration)
        self.events = []
        self.context = self.baseline.first_sources.validate()
        self.context[1].validate = lambda *args: None
        self.stack.enter_context(patch.object(first, "check_parameter_snapshots", return_value={first.FREQUENCY: None}))
        self.stack.enter_context(patch.object(first, "_snapshot_model", side_effect=lambda source, model, ids: {"synthetic": {"bits": ids, "descriptor": {"sha256": _sha(ids)}}}))
        self.execute = self.stack.enter_context(patch.object(first, "execute_first_layer", side_effect=self.prediction))
        self.stack.enter_context(patch.object(first, "check_execution"))

    def prediction(self, program, ids, snapshots, providers, arithmetic_profiles, runtime, workers):
        self.events.append("predict:" + str(ids[0][0]))
        return {"state_bits": {}, "records": [{"record_hash": _sha(ids)}]}

    def plan(self):
        return holdout.build_holdout_plan(self.baseline, object(), self.declaration, self.path, self.frozen, 1)

    def freeze(self, plan, bundle):
        plan_path, bundle_path = self.path / "plan.json", self.path / "bundle.json"
        holdout.write_new(plan_path, plan)
        holdout.write_new(bundle_path, bundle)
        return plan_path, bundle_path

    def capture_stub(self, model, ids, traced):
        self.events.append("native:" + str(ids[0][0]))
        return {"capture_failure": {"type": "ValueError", "message": "synthetic no hardware", "kind": "capture_abstention_not_numerical_evidence"}}

    def test_all_predictions_before_any_native_and_no_prediction_during_acquire(self):
        plan, bundle = self.plan()
        plan_path, bundle_path = self.freeze(plan, bundle)
        self.assertEqual(self.execute.call_count, 2)
        with patch.object(capture, "capture_first_layer", side_effect=self.capture_stub), patch.object(first, "execute_first_layer", side_effect=AssertionError("no predict during native")):
            report = holdout.acquire_holdout(self.baseline, object(), self.declaration, plan, bundle, self.path, self.frozen, plan_path, bundle_path)
        self.assertTrue(all(item.startswith("predict:") for item in self.events[:2]))
        self.assertEqual(len(self.events), 14)
        self.assertTrue(all(item.startswith("native:") for item in self.events[2:]))
        self.assertFalse(report["holdout_matches"])
        self.assertEqual(report["capture_attempt_count"], 12)
        self.assertEqual(report["original_forward_count"], 0)
        self.assertEqual(len(report["observations"]), 2)

    def test_partial_prediction_failure_retained_without_resampling_and_native_blocked(self):
        self.execute.side_effect = [ValueError("Rsqrt exponent has not passed complete domain validation"), self.prediction(None, [[80] * 30], None, None, None, None, 1)]
        plan, bundle = self.plan()
        self.assertFalse(plan["prediction_complete"])
        self.assertEqual(self.execute.call_count, 2)
        self.assertEqual(plan["cases"][0]["prediction_failure"]["case_id"], "distinct_tokens")
        self.assertIsNone(plan["cases"][1]["prediction_failure"])
        paths = self.freeze(plan, bundle)
        with patch.object(capture, "capture_first_layer", side_effect=AssertionError("blocked")), self.assertRaisesRegex(ValueError, "Incomplete prediction"):
            holdout.acquire_holdout(self.baseline, object(), self.declaration, plan, bundle, self.path, self.frozen, *paths)
        report = holdout.holdout_report(self.declaration, plan, bundle, [])
        self.assertFalse(report["holdout_matches"])
        self.assertFalse(report["native_coverage_complete"])
        self.assertEqual(report["original_forward_count"], 0)

    def test_infrastructure_and_runtime_errors_not_disguised_as_domain_failures(self):
        for error in (RuntimeError("device unavailable"), ValueError("Source changed"), ValueError("Primitive provider runtime/evidence mismatch")):
            self.execute.side_effect = error
            with self.subTest(error=error), self.assertRaises(type(error)):
                self.plan()

    def test_parameter_change_before_after_prediction_or_before_native_blocks(self):
        with patch.object(first, "_snapshot_model", side_effect=[{"synthetic": {"bits": 1}}, {"synthetic": {"bits": 2}}]), self.assertRaisesRegex(ValueError, "snapshots changed"):
            self.plan()
        plan, bundle = self.plan()
        paths = self.freeze(plan, bundle)
        with patch.object(first, "_snapshot_model", return_value={}), patch.object(capture, "capture_first_layer", side_effect=AssertionError("blocked")), self.assertRaisesRegex(ValueError, "membership"):
            holdout.acquire_holdout(self.baseline, object(), self.declaration, plan, bundle, self.path, self.frozen, *paths)

    def test_default_verify_never_reexecutes_and_bundle_case_order_guard(self):
        plan, bundle = self.plan()
        observed = [{"case_id": case["case_id"], "input_token_ids": case["input_token_ids"], "pairs": [{"traced": self.capture_stub(None, case["input_token_ids"], True), "untraced": self.capture_stub(None, case["input_token_ids"], False)} for _ in range(3)]} for case in self.declaration["cases"]]
        report = holdout.holdout_report(self.declaration, plan, bundle, observed)
        with patch.object(first, "execute_first_layer", side_effect=AssertionError("default no numeric recompute")), patch.object(first, "_snapshot_model", side_effect=AssertionError("default no model")), patch.object(capture, "capture_first_layer", side_effect=AssertionError("default no native")):
            checked = holdout.verify_holdout(self.baseline, self.declaration, plan, bundle, report, self.path)
        self.assertTrue(checked["valid"])
        self.assertFalse(checked["connected_numerical_recomputation_performed"])
        self.assertFalse(checked["embedding_membership_revalidated_with_model"])
        for change in (lambda b: b["cases"].reverse(), lambda b: b["cases"].pop()):
            changed = copy.deepcopy(bundle)
            change(changed)
            with self.assertRaisesRegex(ValueError, "every original case"):
                holdout._plan_check(self.baseline, self.declaration, plan, changed, self.context)
        report["original_forward_count"] = 6
        checked = holdout.verify_holdout(self.baseline, self.declaration, plan, bundle, reseal(report, "report_sha256"), self.path)
        self.assertFalse(checked["valid"])

    def test_capture_raises_preserved_abstention_not_numerical_match(self):
        plan, bundle = self.plan()
        paths = self.freeze(plan, bundle)
        with patch.object(capture, "capture_first_layer", side_effect=ValueError("Unsupported native geometry")) as native:
            report = holdout.acquire_holdout(self.baseline, object(), self.declaration, plan, bundle, self.path, self.frozen, *paths)
        self.assertEqual(native.call_count, 12)
        self.assertEqual(report["capture_attempt_count"], 12)
        self.assertEqual(report["original_forward_count"], 0)
        self.assertFalse(report["holdout_matches"])
        self.assertIn("capture_failure", report["observations"][1]["pairs"][2]["untraced"])

    def test_replay_prediction_difference_prevents_every_native_call(self):
        plan, bundle = self.plan()
        paths = self.freeze(plan, bundle)
        with patch.object(capture, "capture_first_layer", side_effect=self.capture_stub):
            report = holdout.acquire_holdout(self.baseline, object(), self.declaration, plan, bundle, self.path, self.frozen, *paths)
        self.execute.side_effect = lambda *args: {"state_bits": {}, "records": [{"record_hash": "changed"}]}
        with patch.object(capture, "capture_first_layer", side_effect=AssertionError("different prediction blocks replay")):
            result = holdout.reexecute_holdout(self.baseline, object(), self.declaration, plan, bundle, report, self.path, self.frozen, *paths, workers=1)
        self.assertFalse(result["valid"])
        self.assertFalse(result["predictions_recomputed_exact"])

    def test_reexecute_rebuilds_both_predictions_before_twelve_calls(self):
        plan, bundle = self.plan()
        paths = self.freeze(plan, bundle)
        with patch.object(capture, "capture_first_layer", side_effect=self.capture_stub):
            report = holdout.acquire_holdout(self.baseline, object(), self.declaration, plan, bundle, self.path, self.frozen, *paths)
            self.events.clear()
            result = holdout.reexecute_holdout(self.baseline, object(), self.declaration, plan, bundle, report, self.path, self.frozen, *paths, workers=1)
        self.assertTrue(result["valid"])
        self.assertEqual(len(self.events), 14)
        self.assertTrue(all(value.startswith("predict:") for value in self.events[:2]))
        self.assertTrue(all(value.startswith("native:") for value in self.events[2:]))
        self.assertFalse(result["holdout_matches"])

    def test_failed_prediction_replay_only_never_native(self):
        self.execute.side_effect = ArithmeticError("synthetic bounded arithmetic failure")
        plan, bundle = self.plan()
        paths = self.freeze(plan, bundle)
        report = holdout.holdout_report(self.declaration, plan, bundle, [])
        with patch.object(capture, "capture_first_layer", side_effect=AssertionError("no native for failed predictions")):
            result = holdout.reexecute_holdout(self.baseline, object(), self.declaration, plan, bundle, report, self.path, self.frozen, *paths, workers=1)
        self.assertTrue(result["valid"])
        self.assertFalse(result["holdout_matches"])
        self.assertFalse(result["prediction_complete"])


class NativeRecountTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        program, ids, snapshots, provider = synthetic_context()
        cls.execution = execute_synthetic(program, ids, snapshots, provider)
        cls.coverage = first._coverage(real_program())
        cls.kernels = {role: ["synthetic_" + role] for role in first._native_kernel_roles(real_program())}

    def fixture(self):
        protocol = _seal({"scope": holdout.SCOPE, "cases": holdout._cases(list(range(30, 130))), "sources": {}, "code_sha256": "synthetic",
                          "runtime": {"synthetic": True}, "program_sha256": "synthetic", "program_payload_sha256": "synthetic", "model_binding": {}, "providers": {}, "profiles": profiles(),
                          "coverage_per_case": self.coverage, "required_case_ids": list(holdout.CASE_IDS), "baseline_kernel_symbols": self.kernels,
                          "case_count": 2, "repetitions_per_case": 3, "original_forward_count": 12, "observation_rule": "synthetic", "epoch_scope": "synthetic"}, "protocol_sha256")
        bundle = {"protocol_sha256": protocol["protocol_sha256"], "cases": [{"case_id": case["case_id"], "parameter_snapshots": {}, "execution": self.execution, "prediction_failure": None} for case in protocol["cases"]]}
        plan = _seal(holdout._plan_body(protocol, bundle), "plan_sha256")
        pair = native_pair(self.execution, self.coverage)
        pair["traced"]["kernels"] = self.kernels
        observations = [{"case_id": case["case_id"], "input_token_ids": case["input_token_ids"], "pairs": [copy.deepcopy(pair) for _ in range(3)]} for case in protocol["cases"]]
        return protocol, plan, bundle, observations

    def test_complete_pass_remains_bounded_and_no_global_gates(self):
        report = holdout.holdout_report(*self.fixture())
        self.assertTrue(report["holdout_matches"])
        self.assertTrue(report["native_coverage_complete"])
        self.assertEqual(report["original_forward_count"], 12)
        self.assertFalse(any(report[key] for key in holdout.FALSE_FLAGS))
        self.assertNotEqual(report["scope"], first.SCOPE)
        for case in report["cases"]:
            self.assertEqual(len(case["comparisons"]), 3)
            self.assertEqual(sum(value["value_count"] for key, value in case["comparisons"][0].items() if ":" in key), 810)
            self.assertEqual(case["comparisons"][0]["softmax_f32_bits"]["value_count"], 3600)

    def test_last_coordinate_fp32_aux_and_plain_failures_never_masked(self):
        for kind in ("last", "softmax", "rms", "plain"):
            protocol, plan, bundle, observations = self.fixture()
            pair = observations[-1]["pairs"][-1]
            if kind == "last":
                pair["traced"]["state_bits"]["hidden.1"][0][-1][-1] ^= 1
            elif kind == "softmax":
                pair["traced"]["softmax_f32_bits"][0][-1][-1][-1] ^= 1
            elif kind == "rms":
                pair["traced"]["scalar_stages"]["layer.0.mlp.post_normalized"]["rsqrt_bits"][-1] ^= 1
            else:
                pair["untraced"]["state_bits"]["hidden.1"][0][-1][-1] ^= 1
            report = holdout.holdout_report(protocol, plan, bundle, observations)
            with self.subTest(kind=kind):
                self.assertFalse(report["holdout_matches"])
                self.assertEqual(report["aggregate_mismatch_count"], 1)
                self.assertEqual(report["first_divergence"]["case_id"], "repeated_motif")
                self.assertEqual(report["first_divergence"]["repetition"], 2)
                if kind in ("softmax", "rms"):
                    self.assertEqual(report["cases"][1]["mismatch_counts"]["hidden.1"], 0)

    def test_symbols_runtime_geometry_capture_checks_and_source_invalid(self):
        mutations = [lambda t: t["kernels"].pop(next(iter(t["kernels"]))), lambda t: t["runtime_after"].update(changed=True),
                     lambda t: t["geometry"]["hidden.1"].update(alignment_mod16=2), lambda t: t["checks"].update(no_later_layer=False),
                     lambda t: t.update(code_before="wrong"), lambda t: t["state_bits"].pop("position_ids")]
        for mutate in mutations:
            protocol, plan, bundle, observations = self.fixture()
            mutate(observations[1]["pairs"][2]["traced"])
            with self.subTest(mutate=mutate):
                self.assertFalse(holdout.holdout_report(protocol, plan, bundle, observations)["holdout_matches"])

    def test_omitted_reordered_cases_and_missing_pairs_cannot_pass(self):
        protocol, plan, bundle, observations = self.fixture()
        for observed in (observations[::-1], observations[:1]):
            with self.assertRaisesRegex(ValueError, "both original cases"):
                holdout.holdout_report(protocol, plan, bundle, observed)
        observations[-1]["pairs"].pop()
        report = holdout.holdout_report(protocol, plan, bundle, observations)
        self.assertFalse(report["holdout_matches"])
        self.assertFalse(report["native_coverage_complete"])


class CliTests(unittest.TestCase):
    def test_parser_inherits_sources_and_own_defaults(self):
        parser = cli.build_parser()
        for operation in ("protocol", "plan", "run", "verify"):
            args = parser.parse_args(["gemma-first-layer-holdout-" + operation, "program.json", "--report" if operation == "verify" else "--output", "new.json"])
            self.assertIs(args.handler, cli.command_first_layer_holdout)
            self.assertEqual(args.baseline_plan, Path("results/gemma3_270m_first_layer_plan.json"))
            self.assertIn("holdout", str(args.bundle))
            self.assertIn("holdout", str(args.plan))
            self.assertIn("holdout", str(args.protocol))
            self.assertTrue(hasattr(args, "prefix_probe_plan"))
            self.assertTrue(hasattr(args, "post_feedforward_plan"))
        args = parser.parse_args(["gemma-first-layer-plan", "program.json", "--output", "new.json"])
        self.assertIs(args.handler, cli.command_first_layer)

    def test_cli_explicit_missing_summary_fails_before_model_or_reexecution(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            paths = {name: folder / (name + ".json") for name in ("protocol", "plan", "bundle", "report", "summary")}
            for name, path in paths.items():
                if name != "summary":
                    holdout.write_new(path, {})
            arguments = ["gemma-first-layer-holdout-verify", "program.json", "--reexecute"]
            for name, path in paths.items():
                arguments.extend(("--" + name, str(path)))
            args = cli.build_parser().parse_args(arguments)
            with patch.object(cli, "_load_holdout_baseline", return_value=object()), patch.object(holdout, "verify_holdout", return_value={"valid": True, "holdout_matches": True}) as verify, patch.object(cli, "_holdout_model", side_effect=AssertionError("missing summary blocks model")) as model, patch.object(holdout, "reexecute_holdout", side_effect=AssertionError("missing summary blocks reexecution")) as replay, patch.object(capture, "capture_first_layer", side_effect=AssertionError("missing summary blocks native")) as native, redirect_stdout(io.StringIO()):
                with self.assertRaises(FileNotFoundError):
                    cli.command_first_layer_holdout(args)
            verify.assert_called_once()
            model.assert_not_called()
            replay.assert_not_called()
            native.assert_not_called()

    def test_cli_existing_summary_still_requires_exact_match(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            paths = {name: folder / (name + ".json") for name in ("protocol", "plan", "bundle", "report", "summary")}
            for name, path in paths.items():
                holdout.write_new(path, {"synthetic_summary": True} if name == "summary" else {})
            arguments = ["gemma-first-layer-holdout-verify", "program.json"]
            for name, path in paths.items():
                arguments.extend(("--" + name, str(path)))
            args = cli.build_parser().parse_args(arguments)
            with patch.object(cli, "_load_holdout_baseline", return_value=object()), patch.object(holdout, "verify_holdout", side_effect=lambda *args: {"valid": True, "holdout_matches": True}), patch.object(holdout, "holdout_summary", return_value={"synthetic_summary": True}) as summary, patch.object(cli, "_holdout_model", side_effect=AssertionError("integrity-only verification")) as model, patch.object(holdout, "reexecute_holdout", side_effect=AssertionError("integrity-only verification")) as replay, patch.object(capture, "capture_first_layer", side_effect=AssertionError("integrity-only verification")) as native:
                with redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(cli.command_first_layer_holdout(args), 0)
                self.assertTrue(json.loads(output.getvalue())["summary_matches_source"])
                summary.return_value = {"synthetic_summary": False}
                args.reexecute = True
                with redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(cli.command_first_layer_holdout(args), 1)
                self.assertFalse(json.loads(output.getvalue())["summary_matches_source"])
            model.assert_not_called()
            replay.assert_not_called()
            native.assert_not_called()

    def test_cli_outputs_cannot_create_files_inside_source_directories(self):
        with tempfile.TemporaryDirectory() as temp:
            args = cli.build_parser().parse_args(["gemma-first-layer-holdout-protocol", "program.json", "--model-path", temp, "--output", str(Path(temp) / "new-tokenizer-asset.json")])
            with patch.object(cli, "_load_holdout_baseline", side_effect=AssertionError("reject output before source loading")), self.assertRaisesRegex(ValueError, "outside all frozen inputs"):
                cli.command_first_layer_holdout(args)

    def test_cli_protocol_writes_only_protocol_never_model(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "protocol.json"
            args = cli.build_parser().parse_args(["gemma-first-layer-holdout-protocol", "program.json", "--output", str(target), "--bundle", str(Path(temp) / "bundle.json")])
            protocol = {"protocol_sha256": "synthetic", "cases": []}
            with patch.object(cli, "_load_holdout_baseline", return_value=object()), patch.object(holdout, "build_holdout_protocol", return_value=protocol), patch.object(cli, "_holdout_model", side_effect=AssertionError("no protocol model")), redirect_stdout(io.StringIO()):
                self.assertEqual(cli.command_first_layer_holdout(args), 0)
            self.assertEqual(json.loads(target.read_text()), protocol)
            self.assertFalse(args.bundle.exists())
            with self.assertRaises(ValueError):
                cli.command_first_layer_holdout(args)

    def test_cli_complete_plan_files_exist_before_any_native_acquisition(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            protocol_path, plan_path, bundle_path, report_path = (folder / name for name in ("protocol.json", "plan.json", "bundle.json", "report.json"))
            holdout.write_new(protocol_path, {})
            plan = {"plan_sha256": "synthetic", "prediction_complete": True, "cases": [{"prediction_failure": None}, {"prediction_failure": None}]}
            bundle, events = {"synthetic": True}, []
            parser = cli.build_parser()
            plan_args = parser.parse_args(["gemma-first-layer-holdout-plan", "program.json", "--protocol", str(protocol_path), "--output", str(plan_path), "--bundle", str(bundle_path)])
            run_args = parser.parse_args(["gemma-first-layer-holdout-run", "program.json", "--protocol", str(protocol_path), "--plan", str(plan_path), "--bundle", str(bundle_path), "--output", str(report_path), "--summary", str(folder / "summary.json")])
            report = {"holdout_matches": True}

            def acquire(*args):
                self.assertEqual(json.loads(plan_path.read_text()), plan)
                self.assertEqual(json.loads(bundle_path.read_text()), bundle)
                self.assertTrue(protocol_path.exists())
                self.assertEqual(events, ["model", "predict1", "predict2", "model"])
                events.extend(["native"] * 12)
                return report

            with patch.object(cli, "_load_holdout_baseline", return_value=object()), patch.object(holdout, "_protocol_check"), patch.object(holdout, "check_holdout_plan"), patch.object(cli, "_holdout_model", side_effect=lambda path: events.append("model")) as model, patch.object(holdout, "build_holdout_plan", side_effect=lambda *args: (events.extend(["predict1", "predict2"]) or (plan, bundle))), patch.object(holdout, "acquire_holdout", side_effect=acquire), patch.object(holdout, "holdout_summary", return_value=report), redirect_stdout(io.StringIO()):
                self.assertEqual(cli.command_first_layer_holdout(plan_args), 0)
                self.assertEqual(cli.command_first_layer_holdout(run_args), 0)
            self.assertEqual(model.call_count, 2)
            self.assertEqual(events.count("native"), 12)
            self.assertEqual(json.loads(report_path.read_text()), report)

    def test_cli_failure_plan_persisted_and_run_blocked_without_model(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            protocol_path, plan_path, bundle_path, report_path = (folder / name for name in ("protocol.json", "plan.json", "bundle.json", "report.json"))
            holdout.write_new(protocol_path, {})
            plan = {"plan_sha256": "synthetic", "prediction_complete": False, "cases": [{"prediction_failure": {"case_id": "distinct_tokens", "type": "ValueError", "message": "bounded failure"}}]}
            bundle = {"cases": plan["cases"]}
            args = cli.build_parser().parse_args(["gemma-first-layer-holdout-plan", "program.json", "--protocol", str(protocol_path), "--output", str(plan_path), "--bundle", str(bundle_path)])
            events = []
            with patch.object(cli, "_load_holdout_baseline", return_value=object()), patch.object(holdout, "_protocol_check"), patch.object(cli, "_holdout_model", side_effect=lambda path: events.append("model")), patch.object(holdout, "build_holdout_plan", side_effect=lambda *args: (events.extend(["predict1", "predict2"]) or (plan, bundle))), redirect_stdout(io.StringIO()):
                self.assertEqual(cli.command_first_layer_holdout(args), 1)
            self.assertEqual(json.loads(plan_path.read_text()), plan)
            self.assertEqual(json.loads(bundle_path.read_text()), bundle)
            self.assertEqual(events, ["model", "predict1", "predict2"])
            args = cli.build_parser().parse_args(["gemma-first-layer-holdout-run", "program.json", "--protocol", str(protocol_path), "--plan", str(plan_path), "--bundle", str(bundle_path), "--output", str(report_path), "--summary", str(folder / "summary.json")])
            report = {"holdout_matches": False}
            with patch.object(cli, "_load_holdout_baseline", return_value=object()), patch.object(holdout, "check_holdout_plan"), patch.object(holdout, "holdout_report", return_value=report), patch.object(holdout, "holdout_summary", return_value=report), patch.object(cli, "_holdout_model", side_effect=AssertionError("incomplete native blocked")), redirect_stdout(io.StringIO()):
                self.assertEqual(cli.command_first_layer_holdout(args), 1)
            self.assertEqual(json.loads(report_path.read_text()), report)


class RecordedEvidenceTests(unittest.TestCase):
    def test_recorded_holdouts_preserve_complete_coverage_and_limits(self):
        load = lambda name: json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))
        protocol = load("gemma3_270m_first_layer_holdout_protocol.json")
        plan = load("gemma3_270m_first_layer_holdout_plan.json")
        summary = load("gemma3_270m_first_layer_holdout_summary.json")
        for payload, field in ((protocol, "protocol_sha256"), (plan, "plan_sha256"), (summary, "summary_sha256")):
            self.assertEqual(payload[field], _sha({key: value for key, value in payload.items() if key != field}))
        self.assertEqual(plan["protocol_sha256"], protocol["protocol_sha256"])
        self.assertEqual(summary["plan_sha256"], plan["plan_sha256"])
        self.assertEqual([case["case_id"] for case in protocol["cases"]], list(holdout.CASE_IDS))
        self.assertEqual([case["case_id"] for case in summary["cases"]], list(holdout.CASE_IDS))
        distinct, repeated = [case["input_token_ids"][0] for case in protocol["cases"]]
        self.assertEqual(len(set(distinct)), 30)
        self.assertEqual(len(set(repeated)), 5)
        self.assertEqual(repeated, repeated[:5] * 6)
        excluded = set(protocol["tokenizer"]["all_special_ids"] + protocol["tokenizer"]["baseline_token_ids"])
        self.assertFalse(set(distinct) & set(repeated))
        self.assertFalse(excluded & set(distinct + repeated))
        self.assertTrue(summary["holdout_matches"])
        self.assertTrue(summary["prediction_complete"])
        self.assertTrue(summary["native_coverage_complete"])
        self.assertEqual(summary["original_forward_count"], 12)
        self.assertEqual(summary["capture_attempt_count"], 12)
        self.assertEqual(summary["aggregate_mismatch_count"], 0)
        for case in summary["cases"]:
            self.assertTrue(case["case_matches"])
            self.assertTrue(all(case["checks"].values()))
            self.assertEqual(set(case["mismatch_counts"].values()), {0})
        for field in holdout.FALSE_FLAGS:
            self.assertFalse(summary[field], field)


class RealPreflightTests(unittest.TestCase):
    def test_local_baseline_and_tokenizer_preflight_no_new_case_observation(self):
        args = cli.build_parser().parse_args(["gemma-first-layer-holdout-protocol", "results/gemma3_270m_execution_ir.json", "--output", "unused-protocol.json"])
        for name, value in vars(args).items():
            if isinstance(value, Path) and not value.is_absolute():
                setattr(args, name, ROOT / value)
        required = [value for name, value in vars(args).items() if isinstance(value, Path) and name not in ("output", "protocol", "plan", "bundle", "summary") and "audit" not in name]
        if any(not path.exists() for path in required) or not (args.model_path / "tokenizer.json").exists():
            self.skipTest("Ignored baseline sources/local tokenizer unavailable")
        with patch.object(first, "execute_first_layer", side_effect=AssertionError("no new prediction")), patch.object(first, "_snapshot_model", side_effect=AssertionError("no checkpoint model")), patch.object(capture, "capture_first_layer", side_effect=AssertionError("no native")), patch.object(holdout, "write_new", side_effect=AssertionError("no protocol artifact")), patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=AssertionError("no model load")):
            baseline = cli._load_holdout_baseline(args)
            ids, providers, arithmetic_profiles = baseline.validate()
            metadata, pool = holdout.tokenizer_context(args.model_path, baseline.first_sources.program, ids)
        self.assertGreaterEqual(len(pool), 35)
        self.assertIn("tokenizer.json", metadata["assets_sha256"])
        self.assertEqual(type(providers), first.Providers)
        self.assertEqual(arithmetic_profiles, profiles())
        self.assertEqual(len(metadata["declared_baseline_sequence_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
