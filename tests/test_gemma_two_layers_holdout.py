from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from bioprocess_runtime import cli
from bioprocess_runtime import gemma_two_layers as two
from bioprocess_runtime import gemma_two_layers_holdout as holdout
from bioprocess_runtime import gemma_first_layer_holdout as prior
from bioprocess_runtime.gemma_rotary_slice import _sha, _seal
from bioprocess_runtime.serialization import canonical_json
from test_gemma_first_layer import real_program, profiles, native_pair as first_pair
from test_gemma_first_layer_holdout import FakeTokenizer, reseal
from test_gemma_second_layer import native_pair as second_pair
from test_gemma_two_layers import synthetic_context, synthetic_patches

ROOT = Path(__file__).resolve().parents[1]


try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
except ModuleNotFoundError:
    raise unittest.SkipTest("requires .[gemma] extras")


def baseline_fixture(stack, program=None, providers=None):
    program = real_program() if program is None else program
    ids = [list(range(30))]
    providers = SimpleNamespace(commitments=lambda: {"synthetic": True}) if providers is None else providers
    cases = [_seal({"case_id": name, "input_token_ids": [values]}, "case_sha256") for name, values in zip(prior.CASE_IDS, (list(range(30, 60)), list(range(60, 65)) * 6))]
    protocol = _seal({"cases": cases}, "protocol_sha256")
    source = SimpleNamespace(program=program, runtime={"synthetic": True}, second_sources=SimpleNamespace(protocol=protocol), commitments=lambda: {"synthetic_source": True})
    source.validate = Mock(side_effect=lambda _: (ids, providers, profiles()))
    kernels = {stage: {role: ["synthetic_kernel"] for role in module._native_kernel_roles(program)} for stage, module in (("first", two.first), ("second", two.second))}
    plan = _seal({"scope": two.SCOPE, "input_token_ids": ids, "expected_kernel_sets": kernels, "model_binding": {"synthetic_checkpoint": True}}, "plan_sha256")
    report = _seal({"kernel_trace": [kernels] * 3, "native_coverage_complete": True, "original_forward_count": 12,
                    "required_forward_count": 12, "checks": {"all": True}, "acquisition_guards": dict.fromkeys(two.GUARDS, True)}, "report_sha256")
    baseline = holdout.BaselineSources(source, plan, {"old_activations": "NEVER_COMPUTATIONAL_INPUT"}, report)
    for name, value in (("BASELINE_PLAN_SHA256", plan["plan_sha256"]), ("BASELINE_REPORT_SHA256", report["report_sha256"]), ("PRIOR_PROTOCOL_SHA256", protocol["protocol_sha256"])):
        stack.enter_context(patch.object(holdout, name, value))
    checked = {"valid": True, "two_layers_match": True, "connected_two_layers_independently_recomputed": True}
    def verify(adapter, *args):
        adapter.validate(args[-1])
        return checked
    stack.enter_context(patch.object(two, "verify_two_layers", side_effect=verify))
    return baseline, checked


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        for name in prior.ASSETS:
            (self.path / name).write_text('{"synthetic":true}', encoding="utf-8")
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.sources, self.checked = baseline_fixture(self.stack)
        self.tokenizer = self.stack.enter_context(patch("transformers.AutoTokenizer.from_pretrained", return_value=FakeTokenizer()))

    def protocol(self):
        return holdout.build_holdout_protocol(self.sources, self.path)

    def test_deterministic_two_families_disjoint_from_all_declared_prior_ids(self):
        protocol = self.protocol()
        self.assertEqual(protocol, self.protocol())
        a, b = [case["input_token_ids"][0] for case in protocol["cases"]]
        self.assertEqual((len(a), len(set(a)), len(b), len(set(b))), (30, 30, 30, 5))
        self.assertEqual(b, b[:5] * 6)
        self.assertFalse(set(a) & set(b))
        self.assertFalse(set(a + b) & (set(range(65)) | set(FakeTokenizer.all_special_ids)))
        self.assertTrue(all(65 <= value < 130 for value in a + b))
        self.assertEqual(protocol["seed"], 0x8E2F6B41)
        self.assertNotEqual(protocol["seed"], prior.SEED)
        self.assertEqual([item["case_id"] for item in protocol["cases"]], list(holdout.CASE_IDS))
        metadata = protocol["tokenizer"]
        self.assertEqual((metadata["base_allowed_pool_count"], metadata["allowed_pool_count"]), (100, 65))
        self.assertEqual(metadata["allowed_pool_sha256"], _sha(list(range(65, 130))))
        exclusions = metadata["prior_first_layer_holdout_exclusions"]
        self.assertEqual(exclusions["prior_token_ids"], list(range(30, 65)))
        self.assertEqual(exclusions["prior_token_ids_sha256"], _sha(list(range(30, 65))))
        self.assertEqual(exclusions["prior_protocol_sha256"], self.sources.two_sources.second_sources.protocol["protocol_sha256"])
        self.assertEqual(len(exclusions["prior_case_sha256"]), 2)
        self.assertEqual(protocol["required_forward_count"], 24)
        coverage = protocol["coverage_per_case"]
        self.assertEqual((coverage["instruction_count"], coverage["state_count"], coverage["parameter_count"], coverage["rms_scalar_positions"], coverage["softmax_fp32_positions"]), (63, 66, 29, 1620, 7200))
        self.assertNotIn("init_kwargs", json.dumps(protocol))
        self.assertNotIn("never serialize", json.dumps(protocol))
        self.tokenizer.assert_called_with(str(self.path), local_files_only=True, use_fast=True, trust_remote_code=False)

    def test_sampler_exact_vector_no_old_generator_or_globals(self):
        pool = list(range(65, 130))
        before = pool.copy()
        with patch.object(prior, "_cases", side_effect=AssertionError("old sampler forbidden")):
            result = holdout._cases(pool)
        self.assertEqual(pool, before)
        state, selected = 0x8E2F6B41, pool.copy()
        for index in range(35):
            state = (state ^ (state << 13)) & 0xFFFFFFFF
            state = (state ^ (state >> 17)) & 0xFFFFFFFF
            state = (state ^ (state << 5)) & 0xFFFFFFFF
            other = index + state % (len(pool) - index)
            selected[index], selected[other] = selected[other], selected[index]
        self.assertEqual(result[0]["input_token_ids"][0], selected[:30])
        self.assertEqual(result[1]["input_token_ids"][0], selected[30:35] * 6)
        for bad in (list(range(34)), [1] * 35, list(reversed(pool)), [True] + pool):
            with self.subTest(bad=bad[:3]), self.assertRaises(ValueError):
                holdout._cases(bad)

    def test_protocol_no_model_prediction_capture_or_write(self):
        with ExitStack() as stack:
            for module, name in ((cli, "_holdout_model"), (two, "execute_two_layers"), (two, "_snapshot_model"), (two.first_capture, "capture_first_layer"), (two.second_capture, "capture_second_layer"), (holdout, "write_new")):
                stack.enter_context(patch.object(module, name, side_effect=AssertionError("protocol read-only")))
            stack.enter_context(patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=AssertionError("no model")))
            self.protocol()

    def test_resealed_protocol_seed_order_cases_sources_flags_and_metadata_tamper(self):
        original = self.protocol()
        changes = [lambda p: p.update(seed=prior.SEED), lambda p: p["cases"].reverse(), lambda p: p["cases"].pop(),
                   lambda p: p["cases"][0]["input_token_ids"][0].__setitem__(0, 30),
                   lambda p: p["cases"][0]["input_token_ids"][0].__setitem__(0, 1),
                   lambda p: p["cases"][0]["input_token_ids"][0].__setitem__(0, 300000),
                   lambda p: p["cases"][1]["input_token_ids"][0].reverse(),
                   lambda p: p["tokenizer"].update(allowed_pool_count=99),
                   lambda p: p["tokenizer"].update(init_kwargs={"secret": "forbidden"}),
                   lambda p: p["tokenizer"]["prior_first_layer_holdout_exclusions"].update(prior_token_ids=[]),
                   lambda p: p["tokenizer"]["assets_sha256"].update(tokenizer="changed"),
                   lambda p: p["sources"].update(extra="changed"), lambda p: p["runtime"].update(changed=True),
                   lambda p: p["profiles"].update(refit=True), lambda p: p.update(code_sha256="changed"),
                   lambda p: p.update(fresh_prompt_holdout=True), lambda p: p.update(scope=two.SCOPE)]
        for change in changes:
            modified = copy.deepcopy(original)
            change(modified)
            modified["cases"] = [reseal(case, "case_sha256") for case in modified["cases"]]
            with self.subTest(change=change), self.assertRaises(ValueError):
                holdout._protocol_check(self.sources, reseal(modified, "protocol_sha256"), self.path)

    def test_asset_special_vocabulary_and_final_pool_fail_closed(self):
        protocol = self.protocol()
        (self.path / "tokenizer.json").write_text("changed", encoding="utf-8")
        with self.assertRaises(ValueError):
            holdout._protocol_check(self.sources, protocol, self.path)
        (self.path / "tokenizer.json").write_text('{"synthetic":true}', encoding="utf-8")
        with patch.object(FakeTokenizer, "all_special_ids", [1, 2, 65]), self.assertRaises(ValueError):
            holdout._protocol_check(self.sources, protocol, self.path)
        with patch.object(FakeTokenizer, "get_vocab", return_value={str(i): i for i in range(90)}), self.assertRaisesRegex(ValueError, "35"):
            self.protocol()

    def test_validation_cache_transparent_local_and_all_baseline_failures_reject(self):
        source = self.sources.two_sources
        self.sources.validate(self.path)
        self.assertEqual(source.validate.call_count, 1)
        self.sources.validate(self.path)
        self.assertEqual(source.validate.call_count, 2)
        for key in self.checked:
            with self.subTest(key=key), patch.dict(self.checked, {key: False}), self.assertRaisesRegex(ValueError, "passing connected"):
                self.protocol()
        for field in ("native_coverage_complete", "original_forward_count"):
            report = dict(self.sources.two_layer_report, **{field: False})
            report = reseal(report, "report_sha256")
            bad = holdout.BaselineSources(source, self.sources.two_layer_plan, self.sources.two_layer_bundle, report)
            with patch.object(holdout, "BASELINE_REPORT_SHA256", report["report_sha256"]), self.assertRaises(ValueError):
                bad.validate(self.path)
        with patch.object(source, "commitments", side_effect=[{}, {"changed": True}]), self.assertRaises(ValueError):
            self.sources.validate(self.path)

    def test_kernel_and_prior_pins_fail_closed(self):
        with patch.dict(self.sources.two_layer_plan, plan_sha256="wrong"), self.assertRaises(ValueError):
            self.protocol()
        for change in (lambda p: p["kernel_trace"][0]["first"].pop(next(iter(p["kernel_trace"][0]["first"]))),
                       lambda p: p["kernel_trace"].pop(), lambda p: p["kernel_trace"][0]["second"].update(extra=["kernel"])):
            report = copy.deepcopy(self.sources.two_layer_report)
            change(report)
            source = holdout.BaselineSources(self.sources.two_sources, self.sources.two_layer_plan, {}, report)
            with self.assertRaises(ValueError):
                holdout._kernel_expectations(source)
        with patch.object(holdout, "PRIOR_PROTOCOL_SHA256", "different"), self.assertRaises(ValueError):
            self.protocol()

    def test_own_scope_flags_code_and_old_public_guard_remain(self):
        self.assertNotEqual(holdout.SCOPE, two.SCOPE)
        self.assertNotEqual(holdout.SCOPE, prior.SCOPE)
        for value in (False, True):
            flags = holdout._flags(value)
            self.assertEqual(flags["connected_two_layers_independently_recomputed"], value)
            for name in two.FALSE_FLAGS:
                self.assertFalse(flags[name], name)
            self.assertTrue(flags["prospective_two_layer_raw_token_cases"])
            self.assertTrue(flags["excluded_from_declared_prior_cases"])
            self.assertFalse(flags["fresh_prompt_holdout"])
        with patch.object(two, "_code_sha", return_value="two"), patch.object(prior, "_code_sha", return_value="prior"):
            self.assertEqual(holdout._code_sha(), _sha({"module": holdout.SOURCE_SHA256, "two_layers": "two", "first_layer_holdout": "prior"}))
        with patch.object(holdout, "SOURCE_SHA256", "changed"), self.assertRaises(ValueError):
            holdout._code_sha()
        with patch.object(two, "_snapshot_model", side_effect=AssertionError("must reject first")), self.assertRaisesRegex(ValueError, "baseline token"):
            two.build_two_layers_plan(self.sources.two_sources, object(), self.path, input_token_ids=[[99] * 30])

    def test_missing_frozen_protocol_prevents_any_prediction(self):
        protocol = self.protocol()
        path = self.path / "protocol.json"
        with patch.object(holdout, "_predict", side_effect=AssertionError("no freeze")), self.assertRaises(ValueError):
            holdout.build_holdout_plan(self.sources, object(), protocol, self.path, path)
        holdout.write_new(path, protocol)
        holdout.require_frozen(path, protocol)
        with self.assertRaises(FileExistsError):
            holdout.write_new(path, protocol)


class ExecutionAndNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.program, _, snapshots, cls.providers = synthetic_context()
        cls.context = ([list(range(30))], cls.providers, profiles())
        with ExitStack() as stack:
            stack.enter_context(synthetic_patches(cls.program))
            cls.sources, _ = baseline_fixture(stack, cls.program, cls.providers)
            stack.enter_context(patch.object(prior, "tokenizer_context", return_value=({"allowed_pool_count": 100, "allowed_pool_sha256": _sha(list(range(30, 130)))}, list(range(30, 130)))))
            cls.protocol = holdout.build_holdout_protocol(cls.sources, ROOT)
            cls.snapshots = {}
            for case in cls.protocol["cases"]:
                fresh = copy.deepcopy(snapshots)
                for snapshot in fresh.values():
                    if isinstance(snapshot["bits"], np.ndarray):
                        snapshot["bits"] = snapshot["bits"].tolist()
                fresh[two.first.EMBEDDING]["token_indices"] = case["input_token_ids"][0].copy()
                cls.snapshots[case["case_id"]] = fresh
            events = []
            engine = two.execute_two_layers
            def execute(*args):
                events.append(("execute", copy.deepcopy(args[1]), len(args[2])))
                return engine(*args)
            def snapshot(source, model, ids):
                events.append(("snapshot", copy.deepcopy(ids)))
                return cls.snapshot(ids)
            stack.enter_context(patch.object(two, "_snapshot_model", side_effect=snapshot))
            stack.enter_context(patch.object(two, "execute_two_layers", side_effect=execute))
            stack.enter_context(patch.object(two, "_predict", side_effect=AssertionError("old fixed predictor")))
            stack.enter_context(patch.object(two, "_source_header", side_effect=AssertionError("old scope")))
            cls.plan, cls.bundle = holdout._predict(cls.sources, object(), cls.protocol, cls.context, 1)
            cls.events = events
            cls.pairs = []
            for outcome in cls.bundle["cases"]:
                execution = outcome["execution"]
                a = first_pair(execution["stage_executions"]["first"], two.first._coverage(cls.program))
                a["traced"]["kernels"] = cls.protocol["expected_kernel_sets"]["first"]
                b = second_pair(execution["stage_executions"]["second"], two.second._coverage(cls.program), cls.protocol["expected_kernel_sets"]["second"])
                cls.pairs.append({"first": a, "second": b})
        cls.observations = [{"case_id": case["case_id"], "input_token_ids": case["input_token_ids"], "pairs": [pair] * 3} for case, pair in zip(cls.protocol["cases"], cls.pairs)]

    @classmethod
    def snapshot(cls, ids):
        name = next(case["case_id"] for case in cls.protocol["cases"] if case["input_token_ids"] == ids)
        return copy.deepcopy(cls.snapshots[name])

    def report(self, observations=None, guards=None):
        return holdout.holdout_report(self.protocol, self.plan, self.bundle, self.observations if observations is None else observations, guards)

    def check(self, plan=None, bundle=None):
        with patch.object(two.first, "PROGRAM_SHA256", self.program["program_sha256"]):
            holdout._plan_check(self.sources, self.protocol, self.plan if plan is None else plan, self.bundle if bundle is None else bundle, self.context)

    def test_new_ids_all_29_snapshots_and_unchanged_connected_engine(self):
        self.check()
        forecasts = [event for event in self.events if event[0] == "execute"]
        self.assertEqual(forecasts, [("execute", case["input_token_ids"], 29) for case in self.protocol["cases"]])
        self.assertEqual([event[0] for event in self.events], ["snapshot", "execute", "snapshot"] * 2)
        for outcome in self.bundle["cases"]:
            execution = outcome["execution"]
            self.assertEqual((len(outcome["parameter_snapshots"]), len(execution["records"]), len(execution["state_bits"]), len(execution["bridge"])), (29, 63, 66, 4))
            self.assertEqual(sum(len(v) for stages in execution["scalar_stages"].values() for v in stages.values()), 1620)
            self.assertEqual(sum(np.asarray(v).size for v in execution["softmax_f32_bits"].values()), 7200)
            self.assertEqual(execution["stage_executions"]["second"]["state_bits"]["hidden.1"], execution["stage_executions"]["first"]["state_bits"]["hidden.1"])

    def test_resealed_bundle_case_order_snapshot_dtype_frame_and_bridge_reject(self):
        changes = [lambda b: b["cases"].reverse(), lambda b: b["cases"].pop(),
                   lambda b: b["cases"][0]["parameter_snapshots"].pop(two.first.EMBEDDING),
                   lambda b: b["cases"][0]["parameter_snapshots"].update({"hidden.1": {}}),
                   lambda b: b["cases"][0]["execution"]["bridge"]["hidden.1"].update(producer_record_hash="cached"),
                   lambda b: b["cases"][0]["execution"].update(root_bits={}),
                   lambda b: b["cases"][0]["execution"]["stage_executions"]["second"]["records"][0]["payload"]["inputs"]["hidden.1"]["descriptor"].update(dtype="torch.float32")]
        for change in changes:
            bundle = copy.deepcopy(self.bundle)
            change(bundle)
            with self.subTest(change=change), self.assertRaises((ValueError, KeyError)):
                plan = _seal(holdout._plan_body(self.protocol, bundle), "plan_sha256")
                self.check(plan, bundle)
        for field in ("prefix_boundary_reused", "stored_intermediate_predictions_used", "same_invocation_two_layer_trace"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check(reseal(dict(self.plan, **{field: True}), "plan_sha256"))

    def test_all_24_forwards_115_comparison_scopes_and_separate_native_flags(self):
        report = self.report()
        self.assertTrue(report["two_layer_holdout_matches"])
        self.assertTrue(report["connected_two_layers_independently_recomputed"])
        self.assertEqual((report["required_forward_count"], report["original_forward_count"], report["capture_attempt_count"]), (24, 24, 24))
        self.assertEqual(report["aggregate_mismatch_count"], 0)
        for case in report["cases"]:
            self.assertEqual(len(case["comparisons"]), 3)
            self.assertEqual(len(case["comparisons"][0]), 115)
            self.assertEqual(sum(name.startswith("cross_invocation_bridge:") for name in case["comparisons"][0]), 4)
        self.assertEqual(report["native_scope"], "separate_prefix_and_second_layer_stopped_forwards")
        self.assertFalse(report["native_same_invocation_fullcoverage"])
        self.assertFalse(report["same_invocation_two_layer_trace"])
        self.assertEqual(report["kernel_role_scope_count_per_case"], 44)

    def test_last_hidden_fp32_and_bridge_mismatch_not_hidden_by_final_bf16(self):
        for case_index, kind in ((0, "last"), (1, "softmax"), (1, "rms"), (0, "bridge")):
            observations = copy.deepcopy(self.observations)
            traced = observations[case_index]["pairs"][0]["second"]["traced"]
            if kind == "last":
                traced["state_bits"]["hidden.2"][0][-1][-1] ^= 1
            elif kind == "softmax":
                traced["softmax_f32_bits"][0][-1][-1][-1] ^= 1
            elif kind == "rms":
                traced["scalar_stages"][next(iter(traced["scalar_stages"]))]["rsqrt_bits"][-1] ^= 1
            else:
                traced["state_bits"]["hidden.1"][0][-1][-1] ^= 1
            report = self.report(observations)
            self.assertFalse(report["two_layer_holdout_matches"])
            self.assertGreater(report["aggregate_mismatch_count"], 0)
            self.assertEqual(report["first_divergence"]["case_id"], holdout.CASE_IDS[case_index])
            if kind == "bridge":
                self.assertGreater(report["cases"][case_index]["mismatch_counts"]["cross_invocation_bridge:hidden.1"], 0)

    def test_incomplete_native_never_passes_zero_count_and_preserves_exception(self):
        observations = copy.deepcopy(self.observations)
        observations[-1]["pairs"][-1]["second"]["untraced"] = {"capture_failure": {"type": "RuntimeError", "message": "capture failed"}}
        report = self.report(observations)
        self.assertFalse(report["two_layer_holdout_matches"])
        self.assertFalse(report["native_coverage_complete"])
        self.assertEqual(report["capture_attempt_count"], 24)
        self.assertLess(report["original_forward_count"], 24)
        self.assertIn("capture failed", json.dumps(report["observations"]))
        self.assertFalse(report["connected_two_layers_independently_recomputed"])
        for key in holdout.GUARDS:
            guarded = self.report(guards=dict.fromkeys(holdout.GUARDS, True) | {key: False})
            self.assertFalse(guarded["two_layer_holdout_matches"], key)

    def test_omitted_reordered_native_cases_runtime_kernel_and_geometry_fail_closed(self):
        for observations in (self.observations[:1], self.observations[::-1]):
            with self.assertRaises(ValueError):
                self.report(observations)
        for field in ("runtime", "kernel", "geometry", "check"):
            observed = copy.deepcopy(self.observations)
            traced = observed[0]["pairs"][0]["first"]["traced"]
            if field == "runtime":
                traced["runtime_after"] = {"different": True}
            elif field == "kernel":
                traced["kernels"] = {"wrong": ["kernel"]}
            elif field == "geometry":
                traced["geometry"]["hidden.1"]["dtype"] = "torch.float32"
            else:
                traced["checks"]["no_later_layer"] = False
            self.assertFalse(self.report(observed)["two_layer_holdout_matches"], field)

    def test_partial_domain_failure_retained_no_replacement_and_blocks_all_native(self):
        error = ValueError(next(iter(prior.DOMAIN_ERRORS)))
        with synthetic_patches(self.program), patch.object(two, "_snapshot_model", side_effect=lambda source, model, ids: self.snapshot(ids)), patch.object(two, "execute_two_layers", side_effect=[error, self.bundle["cases"][1]["execution"]]) as execute:
            plan, bundle = holdout._predict(self.sources, object(), self.protocol, self.context, 1)
        self.assertFalse(plan["prediction_complete"])
        self.assertEqual(execute.call_count, 2)
        self.assertEqual([call.args[1] for call in execute.call_args_list], [case["input_token_ids"] for case in self.protocol["cases"]])
        self.check(plan, bundle)
        report = holdout.holdout_report(self.protocol, plan, bundle, [])
        self.assertFalse(report["two_layer_holdout_matches"])
        self.assertEqual((report["capture_attempt_count"], report["original_forward_count"]), (0, 0))
        self.assertEqual(bundle["cases"][0]["prediction_failure"]["message"], str(error))
        with patch.object(holdout, "_frozen"), patch.object(holdout, "_live_snapshots", side_effect=AssertionError("no model")), self.assertRaisesRegex(ValueError, "ALL native"):
            holdout.acquire_holdout(self.sources, None, self.protocol, plan, bundle, ROOT, ROOT, ROOT, ROOT)
        with self.assertRaises(ValueError):
            holdout.holdout_report(self.protocol, plan, bundle, self.observations)
        with synthetic_patches(self.program), patch.object(two, "_snapshot_model", side_effect=lambda source, model, ids: self.snapshot(ids)), patch.object(two, "execute_two_layers", side_effect=ValueError("infrastructure")), self.assertRaisesRegex(ValueError, "infrastructure"):
            holdout._predict(self.sources, object(), self.protocol, self.context, 1)

    def test_acquire_all_memberships_before_any_native_and_no_recompute(self):
        events = []
        def snapshot(source, model, ids):
            events.append(("snapshot", copy.deepcopy(ids)))
            return self.snapshot(ids)
        def capture(stage, model, ids, traced):
            events.append((stage, copy.deepcopy(ids), traced))
            index = next(i for i, case in enumerate(self.protocol["cases"]) if case["input_token_ids"] == ids)
            return self.pairs[index][stage]["traced" if traced else "untraced"]
        with ExitStack() as stack:
            stack.enter_context(patch.object(holdout, "_frozen", side_effect=lambda *a: events.append(("frozen",))))
            stack.enter_context(patch.object(two, "_file_guard", return_value={}))
            stack.enter_context(patch.object(holdout, "_protocol_check", return_value=self.context))
            stack.enter_context(patch.object(two.first, "PROGRAM_SHA256", self.program["program_sha256"]))
            stack.enter_context(patch.object(two, "_snapshot_model", side_effect=snapshot))
            stack.enter_context(patch.object(two, "execute_two_layers", side_effect=AssertionError("initial acquisition cannot recompute")))
            stack.enter_context(patch.object(two.first_capture, "capture_first_layer", side_effect=lambda *a: capture("first", *a)))
            stack.enter_context(patch.object(two.second_capture, "capture_second_layer", side_effect=lambda *a: capture("second", *a)))
            report = holdout.acquire_holdout(self.sources, object(), self.protocol, self.plan, self.bundle, ROOT, ROOT, ROOT, ROOT)
        self.assertTrue(report["two_layer_holdout_matches"])
        self.assertEqual([event[0] for event in events[:4]], ["frozen", "snapshot", "snapshot", "frozen"])
        calls = [event for event in events if event[0] in two.STAGE_IDS]
        expected = [(stage, case["input_token_ids"], traced) for case in self.protocol["cases"] for _ in range(3) for stage in two.STAGE_IDS for traced in (True, False)]
        self.assertEqual(calls, expected)

    def test_default_verify_no_new_numerical_compute_or_model_failed_report_stays_failed(self):
        observations = copy.deepcopy(self.observations)
        observations[0]["pairs"][0]["second"]["traced"]["state_bits"]["hidden.2"][0][-1][-1] ^= 1
        report = self.report(observations)
        with patch.object(holdout, "_protocol_check", return_value=self.context), patch.object(two.first, "PROGRAM_SHA256", self.program["program_sha256"]), patch.object(two, "_snapshot_model", side_effect=AssertionError("no membership")), patch.object(two, "execute_two_layers", side_effect=AssertionError("no numeric compute")):
            result = holdout.verify_holdout(self.sources, self.protocol, self.plan, self.bundle, report, ROOT)
        self.assertTrue(result["valid"])
        self.assertFalse(result["two_layer_holdout_matches"])
        self.assertFalse(result["connected_numerical_recomputation_performed"])
        self.assertFalse(result["embedding_membership_revalidated_with_model"])

    def test_reexecute_all_predictions_before_any_native_and_mismatch_blocks_native(self):
        report = self.report()
        for same in (True, False):
            events = []
            def predict(*args):
                events.extend(["first_case_all29_all63", "second_case_all29_all63"])
                return (self.plan, self.bundle) if same else ({"different": True}, self.bundle)
            def acquire(*args):
                events.append("all24_native")
                return report
            with patch.object(holdout, "_frozen"), patch.object(two, "_file_guard", return_value={}), patch.object(holdout, "verify_holdout", return_value={"valid": True}), patch.object(holdout, "build_holdout_plan", side_effect=predict), patch.object(holdout, "acquire_holdout", side_effect=acquire):
                result = holdout.reexecute_holdout(self.sources, object(), self.protocol, self.plan, self.bundle, report, ROOT, ROOT, ROOT, ROOT)
            self.assertEqual(result["valid"], same)
            self.assertEqual(events, ["first_case_all29_all63", "second_case_all29_all63"] + (["all24_native"] if same else []))
            self.assertFalse(result["same_invocation_two_layer_trace"])


class CLITests(unittest.TestCase):
    def test_new_handlers_defaults_and_old_protocol_meaning_unchanged(self):
        parser = cli.build_parser()
        for operation in ("protocol", "plan", "run", "verify"):
            args = parser.parse_args(["gemma-two-layers-holdout-" + operation, "program", "--report" if operation == "verify" else "--output", "new.json"])
            self.assertIs(args.handler, cli.command_two_layers_holdout)
            self.assertEqual(args.workers, 4)
            self.assertEqual(args.protocol, Path("results/gemma3_270m_first_layer_holdout_protocol.json"))
            self.assertEqual(args.holdout_protocol, Path("results/gemma3_270m_two_layers_holdout_protocol.json"))
            self.assertEqual(args.plan, Path("results/gemma3_270m_two_layers_holdout_plan.json"))
            self.assertEqual(args.bundle, Path("artifacts/gemma3_270m_two_layers_holdout_predictions.json"))
            self.assertEqual(args.summary, Path("results/gemma3_270m_two_layers_holdout_summary.json"))
            self.assertEqual(args.two_layer_baseline_plan, Path("results/gemma3_270m_two_layers_plan.json"))
            self.assertEqual(args.baseline_plan, Path("results/gemma3_270m_first_layer_plan.json"))
        old = parser.parse_args(["gemma-two-layers-plan", "program"])
        self.assertIs(old.handler, cli.command_two_layers)
        self.assertEqual(old.plan, Path("results/gemma3_270m_two_layers_plan.json"))
        for operation in ("protocol", "plan", "run", "verify"):
            with patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(["gemma-two-layers-holdout-" + operation, "program"])

    def test_missing_summary_and_path_guards_before_model_or_sources(self):
        args = cli.build_parser().parse_args(["gemma-two-layers-holdout-verify", "program", "--report", "report", "--summary", "missing-explicit-summary", "--reexecute"])
        with patch.object(cli, "_load_two_layers_sources", side_effect=AssertionError("no sources")), patch.object(cli, "_holdout_model", side_effect=AssertionError("no model")), self.assertRaisesRegex(ValueError, "summary is missing"):
            cli.command_two_layers_holdout(args)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for output, bundle in ((root / "same", root / "same"), (root / "model" / "new", root / "bundle"), (root / "program", root / "bundle")):
                args = cli.build_parser().parse_args(["gemma-two-layers-holdout-plan", str(root / "program"), "--output", str(output), "--bundle", str(bundle), "--model-path", str(root / "model")])
                with patch.object(cli, "_load_two_layers_sources", side_effect=AssertionError("no source")), self.assertRaises(ValueError):
                    cli.command_two_layers_holdout(args)

    def test_protocol_cli_writes_only_protocol_never_model(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new-protocol"
            args = cli.build_parser().parse_args(["gemma-two-layers-holdout-protocol", "program", "--output", str(output)])
            with patch.object(cli, "_load_two_layers_sources", return_value=object()), patch.object(Path, "read_text", return_value="{}"), patch.object(holdout, "build_holdout_protocol", return_value={"protocol_sha256": "hash", "required_case_ids": list(holdout.CASE_IDS)}), patch.object(holdout, "write_new") as write, patch.object(cli, "_holdout_model", side_effect=AssertionError("no model")), patch("sys.stdout", new=io.StringIO()):
                self.assertEqual(cli.command_two_layers_holdout(args), 0)
            write.assert_called_once()
            self.assertEqual(write.call_args.args[0], output)

    def test_incomplete_run_persists_blocked_report_without_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = cli.build_parser().parse_args(["gemma-two-layers-holdout-run", "program", "--output", str(root / "report"), "--summary", str(root / "summary")])
            with patch.object(cli, "_load_two_layers_sources", return_value=object()), patch.object(Path, "read_text", return_value='{"prediction_complete":false}'), patch.object(holdout, "require_frozen"), patch.object(holdout, "check_holdout_plan"), patch.object(holdout, "holdout_report", return_value={"two_layer_holdout_matches": False}), patch.object(holdout, "holdout_summary", return_value={}), patch.object(holdout, "write_new") as write, patch.object(cli, "_holdout_model", side_effect=AssertionError("no model")), patch.object(holdout, "acquire_holdout", side_effect=AssertionError("no native")), patch("sys.stdout", new=io.StringIO()):
                self.assertEqual(cli.command_two_layers_holdout(args), 1)
            self.assertEqual(write.call_count, 2)

    def test_real_source_and_tokenizer_preflight_read_only_automatic(self):
        args = cli.build_parser().parse_args(["gemma-two-layers-holdout-protocol", "results/gemma3_270m_execution_ir.json", "--output", "unused-never-written"])
        for name, value in vars(args).items():
            if isinstance(value, Path) and not value.is_absolute():
                setattr(args, name, ROOT / value)
        paths = [value for name, value in vars(args).items() if isinstance(value, Path) and name not in ("output", "plan", "bundle", "summary", "holdout_protocol") and "audit" not in name]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            self.skipTest("Required ignored data missing: " + ", ".join(missing))
        with ExitStack() as stack:
            for module, name in ((cli, "_holdout_model"), (two, "_snapshot_model"), (two, "execute_two_layers"), (two.first, "execute_first_layer"), (two.second, "execute_second_layer"), (two.first_capture, "capture_first_layer"), (two.second_capture, "capture_second_layer"), (holdout, "write_new"), (prior, "write_new")):
                stack.enter_context(patch.object(module, name, side_effect=AssertionError("Read-only preflight forbids model/prediction/capture/write")))
            load = lambda path: json.loads(path.read_text(encoding="utf-8"))
            sources = holdout.BaselineSources(cli._load_two_layers_sources(args), load(args.two_layer_baseline_plan), load(args.two_layer_baseline_bundle), load(args.two_layer_baseline_report))
            context = sources.validate(args.model_path)
            body = holdout._protocol_body(sources, args.model_path, context)
        self.assertEqual(sources.two_layer_plan["plan_sha256"], holdout.BASELINE_PLAN_SHA256)
        self.assertEqual(sources.two_layer_report["report_sha256"], holdout.BASELINE_REPORT_SHA256)
        self.assertEqual(body["tokenizer"]["prior_first_layer_holdout_exclusions"]["prior_protocol_sha256"], holdout.PRIOR_PROTOCOL_SHA256)
        self.assertGreaterEqual(body["tokenizer"]["allowed_pool_count"], 35)
        self.assertEqual(body["coverage_per_case"]["instruction_count"], 63)
        self.assertEqual(sum(len(roles) for roles in body["expected_kernel_sets"].values()), 44)
        excluded = set(context[0][0]) | set(body["tokenizer"]["all_special_ids"]) | set(body["tokenizer"]["prior_first_layer_holdout_exclusions"]["prior_token_ids"])
        self.assertFalse({value for case in body["cases"] for value in case["input_token_ids"][0]} & excluded)
        self.assertFalse(body["fresh_prompt_holdout"])
        self.assertTrue(body["prospective_two_layer_raw_token_cases"])


if __name__ == "__main__":
    unittest.main()
