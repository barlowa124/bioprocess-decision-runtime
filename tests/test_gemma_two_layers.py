from __future__ import annotations

import copy
import inspect
import io
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from bioprocess_runtime import gemma_first_layer as first
from bioprocess_runtime import gemma_second_layer as second
from bioprocess_runtime import gemma_two_layers as two
from bioprocess_runtime.gemma_rotary_slice import _sha, _seal
from bioprocess_runtime.operational_semantics import append_chain_record
from test_gemma_first_layer import real_program, profiles, cheap_rms, cheap_rows, cheap_softmax
from test_gemma_first_layer import synthetic_context as first_context, native_pair as first_pair
from test_gemma_second_layer import synthetic_context as second_context, native_pair as second_pair

ROOT = Path(__file__).resolve().parents[1]


def synthetic_context():
    program, ids, snapshots, providers = first_context()
    _, _, right, _ = second_context()
    for name, snapshot in right.items():
        program["parameter_commitments"][name] = snapshot["commitment"]
    program = _seal({key: value for key, value in program.items() if key != "program_sha256"}, "program_sha256")
    return program, ids, {**snapshots, **right}, providers


def synthetic_patches(program):
    stack = ExitStack()
    stack.enter_context(patch.object(first, "PROGRAM_SHA256", program["program_sha256"]))
    for module in (first, second, two):
        stack.enter_context(patch.object(module, "_code_sha", return_value="synthetic-code"))
    stack.enter_context(patch.object(first, "_rms_lookup_row", side_effect=cheap_rms))
    stack.enter_context(patch.object(first, "_project_rows", side_effect=cheap_rows))
    stack.enter_context(patch.object(first, "lookup_softmax_row", side_effect=cheap_softmax))
    return stack


def rechain(execution):
    records = []
    for record in execution["records"]:
        append_chain_record(records, record["payload"])
    execution["records"] = records


class TwoLayerStructureTests(unittest.TestCase):
    def test_recorded_connected_result_and_separate_native_scope(self):
        load = lambda name: json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))
        plan = load("gemma3_270m_two_layers_plan.json")
        summary = load("gemma3_270m_two_layers_summary.json")
        self.assertEqual(plan["plan_sha256"], _sha({key: value for key, value in plan.items() if key != "plan_sha256"}))
        self.assertEqual(summary["summary_sha256"], _sha({key: value for key, value in summary.items() if key != "summary_sha256"}))
        self.assertEqual(summary["plan_sha256"], plan["plan_sha256"])
        self.assertEqual(summary["trace_root"], plan["trace_root"])
        self.assertEqual((summary["coverage"]["instruction_count"], summary["coverage"]["state_count"], summary["coverage"]["parameter_count"]), (63, 66, 29))
        self.assertEqual((summary["coverage"]["rms_scalar_positions"], summary["coverage"]["softmax_fp32_positions"]), (1620, 7200))
        self.assertEqual(summary["native_scope"], two.NATIVE_SCOPE)
        self.assertEqual((summary["original_forward_count"], summary["prefix_stopped_forward_count"], summary["second_layer_stopped_forward_count"]), (12, 6, 6))
        self.assertEqual(summary["aggregate_mismatch_count"], 0)
        self.assertTrue(summary["two_layers_match"])
        self.assertTrue(summary["connected_two_layers_independently_recomputed"])
        self.assertTrue(all(summary["checks"].values()))
        self.assertTrue(all(summary["acquisition_guards"].values()))
        self.assertEqual({name: value["producer_instruction_id"] for name, value in summary["bridge"].items()}, {"hidden.1": "i0035", "rotary.local.cosine": "i0004", "rotary.local.sine": "i0004", "mask.sliding": "i0006"})
        for field in two.FALSE_FLAGS:
            self.assertFalse(summary[field], field)

    def test_exact_actual_63_node_66_state_closure(self):
        program = real_program()
        nodes = two.two_layer_instructions(program)
        self.assertEqual([node["id"] for node in nodes], [f"i{index:04d}" for index in range(65) if index not in (3, 5)])
        self.assertEqual(sum(len(node["outputs"]) for node in nodes), 66)
        self.assertEqual({name for node in nodes for name in node["inputs"]} - {name for node in nodes for name in node["outputs"]}, {"input_ids"})
        coverage = two._coverage(program)
        self.assertEqual((coverage["instruction_count"], coverage["state_count"], coverage["parameter_count"]), (63, 66, 29))
        self.assertEqual((coverage["rms_scalar_positions"], coverage["softmax_fp32_positions"]), (1620, 7200))
        self.assertEqual([node["id"] for node in coverage["excluded_prefix_nodes"]], ["i0003", "i0005"])
        self.assertEqual((coverage["sliding_window"], coverage["first_full_attention_index"]), (512, 5))
        self.assertEqual(coverage["attention_types"], ["sliding_attention"] * 2)
        left, right = two._parameter_names(program)
        self.assertEqual((len(left), len(right), len(left | right), left & right), (16, 13, 29, set()))
        self.assertTrue(all(name.startswith("model.layers.1.") for name in right))

    def test_pure_api_no_source_model_or_boundary_arguments(self):
        self.assertEqual(list(inspect.signature(two.execute_two_layers).parameters), ["program", "token_ids", "parameter_snapshots", "providers", "profiles", "runtime", "workers"])
        self.assertFalse(any("root" in key or "source" in key or "model" in key for key in inspect.signature(two.execute_two_layers).parameters))

    def test_wrong_weight_rope_mask_layer_shape_and_missing_nodes_reject(self):
        changes = [lambda p: p["instructions"][37]["parameter_refs"].__setitem__(0, "model.layers.0.self_attn.q_proj.weight"),
                   lambda p: p["instructions"][45]["inputs"].__setitem__(2, "rotary.global.cosine"),
                   lambda p: p["instructions"][50]["inputs"].__setitem__(1, "mask.full"),
                   lambda p: p["configuration"]["layer_types"].__setitem__(1, "full_attention"),
                   lambda p: p["instructions"].pop(36),
                   lambda p: p["external_inputs"].append("hidden.1"),
                   lambda p: p["tensors"]["hidden.2"].update(shape=[1, 29, 640])]
        for change in changes:
            program = real_program()
            change(program)
            with self.subTest(change=change), self.assertRaises(ValueError):
                two.two_layer_instructions(program)

    def test_flags_do_not_promote_hardware_or_same_invocation_claims(self):
        for connected in (True, False):
            flags = two._flags(connected)
            self.assertTrue(all(flags[name] is False for name in two.FALSE_FLAGS))
            self.assertEqual(flags["connected_two_layers_independently_recomputed"], connected)
            self.assertFalse(flags["prefix_boundary_reused"])
            self.assertFalse(flags["stored_boundary_predictions_used"])
            self.assertFalse(flags["stored_intermediate_predictions_used"])
            self.assertFalse(flags["same_invocation_two_layer_trace"])
            self.assertTrue(flags["empirical_primitive_data_reused"])
        self.assertNotEqual(two.SCOPE, first.SCOPE)
        self.assertNotEqual(two.SCOPE, second.SCOPE)


class TwoLayerExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.program, cls.ids, cls.snapshots, cls.providers = synthetic_context()
        with synthetic_patches(cls.program):
            cls.execution = two.execute_two_layers(cls.program, cls.ids, cls.snapshots, cls.providers, profiles(), {"synthetic": True}, 1)

    def check(self, execution=None, snapshots=None):
        with patch.object(first, "PROGRAM_SHA256", self.program["program_sha256"]):
            two.check_execution(self.program, self.ids, self.snapshots if snapshots is None else snapshots, self.providers, profiles(), {"synthetic": True}, self.execution if execution is None else execution)

    def test_complete_pipeline_and_global_links(self):
        self.check()
        execution = self.execution
        self.assertEqual((len(execution["records"]), len(execution["state_bits"])), (63, 66))
        self.assertEqual(sum(len(values) for stages in execution["scalar_stages"].values() for values in stages.values()), 1620)
        self.assertEqual(sum(np.asarray(value).size for value in execution["softmax_f32_bits"].values()), 7200)
        hashes = {"input_ids": "ROOT:" + first._descriptor(np.asarray(self.ids, dtype=np.int64))["sha256"]}
        for record in execution["records"]:
            payload = record["payload"]
            for name, entry in payload["inputs"].items():
                self.assertEqual(entry["producer_record_hash"], hashes[name])
                self.assertEqual(entry["producer_record_hash"].startswith("ROOT:"), name == "input_ids")
            hashes.update({name: record["record_hash"] for name in payload["outputs"]})
        self.assertEqual(set(execution["bridge"]), set(second.ROOTS))
        for name, bridge in execution["bridge"].items():
            self.assertEqual(bridge["producer_record_hash"], hashes[name])
            self.assertEqual(bridge["producer_instruction_id"], self.program["tensors"][name]["producer"])
            self.assertEqual(bridge["second_stage_root_hash"], "ROOT:" + bridge["descriptor"]["sha256"])
        self.assertEqual({value["producer_instruction_id"] for value in execution["bridge"].values()}, {"i0035", "i0004", "i0006"})

    def test_fresh_stage_order_root_object_identity_and_poisoned_old_helpers(self):
        events, produced = [], {}
        first_engine, second_engine = first.execute_first_layer, second.execute_second_layer
        def a(*args):
            events.append("first")
            produced["first"] = first_engine(*args)
            return produced["first"]
        def b(program, roots, *args):
            events.append("second")
            self.assertEqual(set(roots), set(second.ROOTS))
            for name in second.ROOTS:
                self.assertIs(roots[name], produced["first"]["state_bits"][name])
            return second_engine(program, roots, *args)
        with synthetic_patches(self.program), ExitStack() as stack:
            for module, name in ((first, "build_first_layer_plan"), (second, "build_second_layer_plan"), (first, "bind_model_tensors"), (first.ff, "predict_post_feedforward"), (two.TwoLayerSources, "validate"), (two, "_snapshot_model")):
                stack.enter_context(patch.object(module, name, side_effect=AssertionError("No old helper/bundle/model access in pure API")))
            stack.enter_context(patch.object(first, "execute_first_layer", side_effect=a))
            stack.enter_context(patch.object(second, "execute_second_layer", side_effect=b))
            result = two.execute_two_layers(self.program, self.ids, self.snapshots, self.providers, profiles(), {"synthetic": True}, 1)
        self.assertEqual(events, ["first", "second"])
        self.assertIs(result["stage_executions"]["first"], produced["first"])
        self.assertEqual(result, self.execution)

    def test_changed_token_and_fresh_selected_row_propagates_to_hidden_two(self):
        ids = copy.deepcopy(self.ids)
        ids[0][-1] = 30
        snapshots = dict(self.snapshots)
        snapshots[first.EMBEDDING] = copy.deepcopy(snapshots[first.EMBEDDING])
        snapshot = snapshots[first.EMBEDDING]
        snapshot["token_indices"][-1] = 30
        snapshot["bits"][-1, -1] = 0x3F80
        snapshot["descriptor"] = first._descriptor(snapshot["bits"])
        with synthetic_patches(self.program):
            result = two.execute_two_layers(self.program, ids, snapshots, self.providers, profiles(), {"synthetic": True}, 1)
        self.assertNotEqual(result["state_bits"]["hidden.1"], self.execution["state_bits"]["hidden.1"])
        self.assertNotEqual(result["state_bits"]["hidden.2"], self.execution["state_bits"]["hidden.2"])
        self.assertNotEqual(result["records"][0]["record_hash"], self.execution["records"][0]["record_hash"])

    def test_mutated_first_frame_during_second_execute_rejects(self):
        def corrupt(program, roots, *args):
            roots["hidden.1"][0][-1][-1] ^= 1
            return self.execution["stage_executions"]["second"]
        with synthetic_patches(self.program), patch.object(second, "execute_second_layer", side_effect=corrupt), self.assertRaisesRegex(ValueError, "frame"):
            two.execute_two_layers(self.program, self.ids, self.snapshots, self.providers, profiles(), {"synthetic": True}, 1)

    def test_resealed_wrong_producer_and_cached_bridge_reject(self):
        for kind in ("root_link", "wrong_producer", "provenance", "bridge", "missing_node", "missing_stage", "extra_roots", "boundary_path", "merged_state", "subtrace"):
            changed = copy.deepcopy(self.execution)
            if kind == "root_link":
                changed["records"][34]["payload"]["inputs"]["hidden.1"]["producer_record_hash"] = changed["bridge"]["hidden.1"]["second_stage_root_hash"]
            elif kind == "wrong_producer":
                changed["records"][34]["payload"]["inputs"]["hidden.1"]["producer_record_hash"] = changed["records"][0]["record_hash"]
            elif kind == "provenance":
                changed["records"][34]["payload"]["original_fresh_stage_record_hash"] = "old"
            elif kind == "bridge":
                changed["bridge"]["hidden.1"]["producer_instruction_id"] = "i0064"
            elif kind == "missing_node":
                changed["records"].pop(34)
            elif kind == "missing_stage":
                changed["stage_executions"].pop("first")
            elif kind == "extra_roots":
                changed["root_bits"] = {}
            elif kind == "boundary_path":
                changed["stored_boundary_path"] = "old.json"
            elif kind == "merged_state":
                changed["state_bits"]["hidden.2"][0][-1][-1] ^= 1
            else:
                changed["subtrace_roots"]["second"] = "stale"
            rechain(changed)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.check(changed)

    def test_stale_second_roots_even_with_resealed_stage_reject(self):
        changed = copy.deepcopy(self.execution)
        stage = changed["stage_executions"]["second"]
        stage["state_bits"]["hidden.1"][0][-1][-1] ^= 1
        rechain(stage)
        with self.assertRaisesRegex(ValueError, "root changed"):
            self.check(changed)

    def test_auxiliary_and_last_state_tampering_rejects_integrity(self):
        for stage in two.STAGE_IDS:
            for kind in ("rms", "softmax", "last"):
                changed = copy.deepcopy(self.execution)
                value = changed["stage_executions"][stage]
                if kind == "rms":
                    value["scalar_stages"][next(iter(value["scalar_stages"]))]["rsqrt_bits"][-1] ^= 1
                elif kind == "softmax":
                    value["softmax_f32_bits"][0][-1][-1][-1] ^= 1
                else:
                    value["state_bits"]["hidden.1" if stage == "first" else "hidden.2"][0][-1][-1] ^= 1
                rechain(value)
                with self.subTest(stage=stage, kind=kind), self.assertRaises(ValueError):
                    self.check(changed)

    def test_snapshot_union_invalid_worker_and_wrong_weight_reject(self):
        with synthetic_patches(self.program):
            for workers in (0, 5, True):
                with self.subTest(workers=workers), self.assertRaises(ValueError):
                    two.execute_two_layers(self.program, self.ids, self.snapshots, self.providers, profiles(), {"synthetic": True}, workers)
            for kind in ("missing", "extra_root", "layer_weight"):
                snapshots = dict(self.snapshots)
                if kind == "missing":
                    snapshots.pop(first.EMBEDDING)
                elif kind == "extra_root":
                    snapshots["hidden.1"] = {}
                else:
                    name = "model.layers.1.self_attn.q_proj.weight"
                    snapshots[name] = copy.deepcopy(snapshots[name])
                    snapshots[name]["bits"].reshape(-1)[-1] ^= 1
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    self.check(snapshots=snapshots)


class TwoLayerNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        program, ids, snapshots, providers = synthetic_context()
        with synthetic_patches(program):
            execution = two.execute_two_layers(program, ids, snapshots, providers, profiles(), {"synthetic": True}, 1)
            coverage = two._coverage(program)
            first_coverage, second_coverage = first._coverage(program), second._coverage(program)
            kernels = {"first": {name: ["synthetic_kernel"] for name in first._native_kernel_roles(program)},
                       "second": {name: ["synthetic_kernel"] for name in second._native_kernel_roles(program)}}
        cls.bundle = {"execution": execution, "parameter_snapshots": {}}
        cls.plan = _seal({"scope": two.SCOPE, "native_scope": two.NATIVE_SCOPE, "bundle_sha256": _sha(cls.bundle), "trace_root": execution["records"][-1]["record_hash"],
                          "coverage": coverage, "runtime": {"synthetic": True}, "expected_kernel_sets": kernels,
                          "capture_code_sha256": {"first": two.first_capture._code_sha(), "second": two.second_capture._code_sha()}, **two._flags()}, "plan_sha256")
        a = first_pair(execution["stage_executions"]["first"], first_coverage)
        a["traced"]["kernels"] = kernels["first"]
        b = second_pair(execution["stage_executions"]["second"], second_coverage, kernels["second"])
        cls.pair = {"first": a, "second": b}

    def report(self, pair=None, guards=None):
        return two.two_layers_report(self.plan, self.bundle, [self.pair if pair is None else pair, self.pair, self.pair], guards)

    def test_complete_twelve_calls_all_states_and_separate_invocation_scope(self):
        report = self.report()
        self.assertTrue(report["two_layers_match"])
        self.assertTrue(report["connected_two_layers_independently_recomputed"])
        self.assertEqual((report["original_forward_count"], report["prefix_stopped_forward_count"], report["second_layer_stopped_forward_count"]), (12, 6, 6))
        self.assertEqual(report["aggregate_mismatch_count"], 0)
        self.assertEqual(report["kernel_role_scope_count"], 44)
        self.assertEqual(report["native_scope"], two.NATIVE_SCOPE)
        self.assertFalse(report["same_invocation_two_layer_trace"])
        self.assertTrue(all(report[name] is False for name in two.FALSE_FLAGS))
        self.assertEqual(len([name for name in report["comparisons"][0] if name.startswith("cross_invocation_bridge:")]), 4)

    def test_all_unique_last_native_state_coordinates_compared(self):
        for stage in two.STAGE_IDS:
            for name in self.pair[stage]["traced"]["state_bits"]:
                pair = copy.deepcopy(self.pair)
                dtype = np.int64 if name == "position_ids" else np.uint16
                bits = np.asarray(pair[stage]["traced"]["state_bits"][name], dtype=dtype)
                bits.reshape(-1)[-1] ^= 1
                pair[stage]["traced"]["state_bits"][name] = bits.tolist()
                with self.subTest(stage=stage, name=name):
                    report = self.report(pair)
                    self.assertFalse(report["two_layers_match"])
                    self.assertEqual(report["mismatch_counts"][stage + ":" + name], 1)

    def test_both_stages_rms_and_fp32_mismatch_not_hidden_by_hidden_two(self):
        for stage in two.STAGE_IDS:
            for kind in ("rms", "softmax"):
                pair = copy.deepcopy(self.pair)
                if kind == "rms":
                    pair[stage]["traced"]["scalar_stages"][next(iter(pair[stage]["traced"]["scalar_stages"]))]["rsqrt_bits"][-1] ^= 1
                else:
                    pair[stage]["traced"]["softmax_f32_bits"][0][-1][-1][-1] ^= 1
                report = self.report(pair)
                with self.subTest(stage=stage, kind=kind):
                    self.assertFalse(report["connected_two_layers_independently_recomputed"])
                    self.assertEqual(report["mismatch_counts"]["second:hidden.2"], 0)
                    self.assertEqual(report["aggregate_mismatch_count"], 1)

    def test_plain_boundaries_compared_and_minimal_controls_required(self):
        for stage, name in (("first", "hidden.1"), ("second", "hidden.1"), ("second", "hidden.2")):
            pair = copy.deepcopy(self.pair)
            pair[stage]["untraced"]["state_bits"][name][0][-1][-1] ^= 1
            self.assertFalse(self.report(pair)["two_layers_match"])
        pair = copy.deepcopy(self.pair)
        pair["first"]["untraced"]["kernels"] = {"extra": ["kernel"]}
        self.assertFalse(self.report(pair)["native_coverage_complete"])

    def test_checks_runtime_code_geometry_and_kernel_stability_enforced(self):
        for stage in two.STAGE_IDS:
            changes = [lambda p: p["traced"]["checks"].update(no_later_layer=False),
                       lambda p: p["untraced"]["checks"].pop("plain_minimal_control"),
                       lambda p: p["traced"]["runtime_after"].update(changed=True),
                       lambda p: p["traced"].update(code_after="changed"),
                       lambda p: p["traced"]["geometry"]["hidden.1"].update(device="cpu"),
                       lambda p: p["traced"]["kernels"].update(softmax=["changed_kernel"])]
            for change in changes:
                pair = copy.deepcopy(self.pair)
                change(pair[stage])
                with self.subTest(stage=stage, change=change):
                    self.assertFalse(self.report(pair)["two_layers_match"])
        pair = copy.deepcopy(self.pair)
        pair["second"]["traced"]["checks"].pop("target_layer_one")
        self.assertFalse(self.report(pair)["two_layers_match"])

    def test_omitted_observations_and_failed_capture_preserved(self):
        for observations in ([], [self.pair], [self.pair] * 2):
            report = two.two_layers_report(self.plan, self.bundle, observations)
            self.assertFalse(report["native_coverage_complete"])
            self.assertFalse(report["two_layers_match"])
        pair = copy.deepcopy(self.pair)
        pair["second"]["traced"] = {"capture_failure": {"message": "abstain"}}
        report = self.report(pair)
        self.assertFalse(report["two_layers_match"])
        self.assertEqual(report["original_forward_count"], 11)
        self.assertEqual(report["observations"][0], pair)

    def test_failed_guards_keep_native_evidence_without_conformance(self):
        for name in two.GUARDS:
            guards = dict.fromkeys(two.GUARDS, True)
            guards[name] = False
            report = self.report(guards=guards)
            self.assertFalse(report["two_layers_match"])
            self.assertTrue(report["native_coverage_complete"])
            self.assertEqual(report["aggregate_mismatch_count"], 0)


class TwoLayerSourceAndLifecycleTests(unittest.TestCase):
    def fixture(self, stack):
        program = real_program()
        ids = [list(range(30))]
        baseline = SimpleNamespace(baseline_plan={"input_token_ids": ids}, baseline_report={"observations": [{"traced": {"kernels": {name: ["kernel"] for name in first._native_kernel_roles(program)}}}] * 3})
        old = SimpleNamespace(program=program, runtime={"synthetic": True}, baseline=baseline, commitments=lambda: {"source": True})
        old.validate = Mock(return_value=(ids, "providers", "profiles", "POISON_OLD_ROOTS", "POISON_OLD_BINDINGS"))
        plan = _seal({"test": "plan"}, "plan_sha256")
        report = _seal({"kernel_trace": [{name: ["kernel"] for name in second._native_kernel_roles(program)}] * 3}, "report_sha256")
        source = two.TwoLayerSources(old, plan, {"old": "never_computational"}, report)
        stack.enter_context(patch.object(two, "SECOND_PLAN_SHA256", plan["plan_sha256"]))
        stack.enter_context(patch.object(two, "SECOND_REPORT_SHA256", report["report_sha256"]))
        stack.enter_context(patch.object(two, "_code_sha", return_value="code"))
        checked = {"valid": True, "second_layer_matches": True, "layer_one_independently_recomputed": True}
        def verify(cached, *args):
            cached.validate(ROOT)
            return checked
        stack.enter_context(patch.object(second, "verify_second_layer", side_effect=verify))
        return source, checked

    def test_full_prior_verify_per_call_cache_and_discard_old_boundaries(self):
        with ExitStack() as stack:
            source, _ = self.fixture(stack)
            self.assertEqual(source.validate(ROOT), ([list(range(30))], "providers", "profiles"))
            self.assertEqual(source.second_sources.validate.call_count, 1)
            source.validate(ROOT)
            self.assertEqual(source.second_sources.validate.call_count, 2)
            self.assertEqual(len(two._kernel_expectations(source)["first"]), 22)
            self.assertEqual(len(two._kernel_expectations(source)["second"]), 22)

    def test_any_prerequisite_failure_prevents_snapshot_and_compute(self):
        for key in ("valid", "second_layer_matches", "layer_one_independently_recomputed"):
            with ExitStack() as stack:
                source, checked = self.fixture(stack)
                checked[key] = False
                stack.enter_context(patch.object(two, "_snapshot_model", side_effect=AssertionError("No snapshot")))
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "prerequisite"):
                    two.build_two_layers_plan(source, None, ROOT)

    def test_public_different_tokens_reject_before_snapshot(self):
        source = SimpleNamespace(validate=lambda _: ([list(range(30))], None, None))
        with patch.object(two, "_snapshot_model", side_effect=AssertionError("No snapshot")), self.assertRaisesRegex(ValueError, "baseline token"):
            two.build_two_layers_plan(source, None, ROOT, input_token_ids=[[1] * 30])

    def test_snapshot_uses_both_unchanged_helpers_and_exact_union(self):
        program = real_program()
        source = SimpleNamespace(program=program, second_sources=SimpleNamespace(baseline=SimpleNamespace(first_sources=object())))
        left, right = two._parameter_names(program)
        a, b = dict.fromkeys(left, "fresh_first"), dict.fromkeys(right, "fresh_second")
        model, ids = object(), [list(range(30))]
        with patch.object(second, "_model_context") as bound, patch.object(first, "_snapshot_model", return_value=a) as first_snapshot, patch.object(second, "_snapshot_model", return_value=b) as second_snapshot:
            result = two._snapshot_model(source, model, ids)
        self.assertEqual(result, {**a, **b})
        self.assertEqual(bound.call_count, 2)
        first_snapshot.assert_called_once_with(source.second_sources.baseline.first_sources, model, ids)
        second_snapshot.assert_called_once_with(source.second_sources, model)
        with patch.object(second, "_model_context"), patch.object(first, "_snapshot_model", return_value=a), patch.object(second, "_snapshot_model", return_value=a), self.assertRaises(ValueError):
            two._snapshot_model(source, model, ids)

    def test_default_verify_no_forecast_snapshot_or_model(self):
        source = SimpleNamespace(commitments=lambda: {})
        report = _seal({"observations": [], "acquisition_guards": {}, "two_layers_match": True, "mismatch_counts": {}, "first_divergence": None}, "report_sha256")
        with patch.object(two, "_code_sha", return_value="code"), patch.object(two, "check_two_layers_plan") as check, patch.object(two, "two_layers_report", return_value=report), patch.object(two, "execute_two_layers", side_effect=AssertionError("No forecast")), patch.object(two, "_snapshot_model", side_effect=AssertionError("No model")):
            result = two.verify_two_layers(source, {}, {}, report, ROOT)
        check.assert_called_once()
        self.assertTrue(result["valid"])
        self.assertFalse(result["connected_numerical_recomputation_performed"])
        self.assertFalse(result["embedding_membership_revalidated_with_model"])

    def test_run_and_reexecute_rebuild_both_stages_before_any_native(self):
        for replay in (False, True):
            for same in (False, True):
                events = []
                source = SimpleNamespace(validate=lambda _: ([], None, None))
                plan, bundle, report = {"plan": True}, {"bundle": True}, {"two_layers_match": True}
                def predict(*args):
                    events.extend(("fresh29", "first34", "second29", "global63"))
                    return (plan, bundle) if same else ({"different": True}, bundle)
                def acquire(*args):
                    events.append("six_prefix_plus_six_second")
                    return report
                with patch.object(two.second.holdout, "require_frozen"), patch.object(two, "_file_guard", return_value={}), patch.object(two, "verify_two_layers", return_value={"valid": True}), patch.object(two, "_check_plan"), patch.object(two, "_predict", side_effect=predict), patch.object(two, "_acquire", side_effect=acquire):
                    if replay:
                        result = two.reexecute_two_layers(source, None, plan, bundle, report, ROOT, ROOT, ROOT)
                        self.assertEqual(result["valid"], same)
                    elif same:
                        self.assertEqual(two.acquire_two_layers(source, None, plan, bundle, ROOT, ROOT, ROOT), report)
                    else:
                        with self.assertRaises(ValueError):
                            two.acquire_two_layers(source, None, plan, bundle, ROOT, ROOT, ROOT)
                self.assertEqual(events, ["fresh29", "first34", "second29", "global63"] + (["six_prefix_plus_six_second"] if same else []))

    def test_changed_frozen_file_before_capture_blocks_native(self):
        source = SimpleNamespace(validate=lambda _: ([], None, None))
        plan, bundle = {"plan": True}, {"bundle": True}
        with patch.object(two.second.holdout, "require_frozen"), patch.object(two, "_file_guard", side_effect=[{"file": "old"}, {"file": "new"}]), patch.object(two, "_check_plan"), patch.object(two, "_predict", return_value=(plan, bundle)), patch.object(two, "_acquire", side_effect=AssertionError("No native")), self.assertRaisesRegex(ValueError, "Frozen inputs changed"):
            two.acquire_two_layers(source, None, plan, bundle, ROOT, ROOT, ROOT)

    def test_code_commitment_includes_both_engines_and_capture_helpers(self):
        with patch.object(first, "_code_sha", return_value="first") as a, patch.object(second, "_code_sha", return_value="second") as b, patch.object(two.first_capture, "_code_sha", return_value="first_capture") as c, patch.object(two.second_capture, "_code_sha", return_value="second_capture") as d:
            value = two._code_sha()
            self.assertEqual(value, _sha({"module": two.SOURCE_SHA256, "first": "first", "second": "second", "first_capture": "first_capture", "second_capture": "second_capture"}))
        for call in (a, b, c, d):
            call.assert_called_once()

    def test_capture_protocol_exact_12_calls_and_guards(self):
        events = []
        source = SimpleNamespace(commitments=lambda: {})
        def capture(stage, *args):
            events.append((stage, args[-1]))
            return {"stage": stage}
        with patch.object(two.second.holdout, "require_frozen"), patch.object(two, "_file_guard", return_value={}), patch.object(two, "_code_sha", return_value="code"), patch.object(two, "_snapshot_model", return_value={}), patch.object(two.first_capture, "capture_first_layer", side_effect=lambda *a: capture("first", *a)), patch.object(two.second_capture, "capture_second_layer", side_effect=lambda *a: capture("second", *a)), patch.object(two, "two_layers_report", side_effect=lambda *a: a) as report:
            result = two._acquire(source, object(), {}, {"parameter_snapshots": {}}, ([list(range(30))], None, None), ROOT, ROOT)
        self.assertEqual(events, [("first", True), ("first", False), ("second", True), ("second", False)] * 3)
        self.assertEqual(len(result[2]), 3)
        self.assertEqual(result[3], dict.fromkeys(two.GUARDS, True))
        report.assert_called_once()

    def test_real_source_preflight_read_only_no_forecast_or_cuda_forward(self):
        from bioprocess_runtime import cli
        args = cli.build_parser().parse_args(["gemma-two-layers-plan", "results/gemma3_270m_execution_ir.json"])
        for name, value in vars(args).items():
            if isinstance(value, Path) and not value.is_absolute():
                setattr(args, name, ROOT / value)
        paths = [value for name, value in vars(args).items() if isinstance(value, Path) and name not in ("plan", "bundle", "output", "summary") and "audit" not in name]
        if any(not path.exists() for path in paths):
            self.skipTest("Ignored source evidence unavailable")
        with ExitStack() as stack:
            for module, name in ((cli, "_holdout_model"), (first, "execute_first_layer"), (second, "execute_second_layer"), (two, "execute_two_layers"), (two, "_snapshot_model"), (two.first_capture, "capture_first_layer"), (two.second_capture, "capture_second_layer"), (second.holdout, "write_new")):
                stack.enter_context(patch.object(module, name, side_effect=AssertionError("Read-only preflight forbids forecast/model/capture/write")))
            sources = cli._load_two_layers_sources(args)
            ids, providers, arithmetic = sources.validate(args.model_path)
            coverage = two._coverage(sources.program)
            kernels = two._kernel_expectations(sources)
            code = two._code_sha()
            header = two._source_header(sources, (ids, providers, arithmetic))
        self.assertEqual(header["coverage"], coverage)
        self.assertEqual(header["expected_kernel_sets"], kernels)
        self.assertEqual(header["code_sha256"], code)
        self.assertNotIn("prediction_complete", header)
        self.assertEqual(len(ids[0]), 30)
        self.assertIs(type(providers), first.Providers)
        self.assertEqual(coverage["state_count"], 66)
        self.assertEqual(sum(len(value) for value in kernels.values()), 44)
        self.assertEqual(arithmetic["down"]["chunk_size"], 192)
        self.assertEqual(len(code), 64)


class TwoLayerCLITests(unittest.TestCase):
    def test_defaults_handlers_source_protocol_and_old_behavior(self):
        from bioprocess_runtime import cli
        parser = cli.build_parser()
        for operation in ("plan", "run", "verify"):
            argv = ["gemma-two-layers-" + operation, "program"]
            if operation != "plan":
                argv += ["--report" if operation == "verify" else "--output", "new.json"]
            args = parser.parse_args(argv)
            self.assertIs(args.handler, cli.command_two_layers)
            self.assertEqual(args.workers, 4)
            self.assertEqual(args.plan, Path("results/gemma3_270m_two_layers_plan.json"))
            self.assertEqual(args.bundle, Path("artifacts/gemma3_270m_two_layers_predictions.json"))
            self.assertEqual(args.summary, Path("results/gemma3_270m_two_layers_summary.json"))
            self.assertEqual(args.protocol, Path("results/gemma3_270m_first_layer_holdout_protocol.json"))
            self.assertEqual(args.second_layer_report, Path("artifacts/gemma3_270m_second_layer_report.json"))
            self.assertEqual(args.holdout_plan, Path("results/gemma3_270m_first_layer_holdout_plan.json"))
        old = parser.parse_args(["gemma-second-layer-plan", "program", "--output", "new.json"])
        self.assertIs(old.handler, cli.command_second_layer)
        self.assertEqual(old.plan, Path("results/gemma3_270m_second_layer_plan.json"))
        with patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["gemma-two-layers-run", "program"])

    def test_exclusive_output_input_and_model_directory_guards_before_loading(self):
        from bioprocess_runtime import cli
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for output, bundle in ((root / "same", root / "same"), (root / "model" / "new.json", root / "bundle"), (root / "program", root / "bundle")):
                args = cli.build_parser().parse_args(["gemma-two-layers-plan", str(root / "program"), "--plan", str(output), "--bundle", str(bundle), "--model-path", str(root)])
                with patch.object(cli, "_load_two_layers_sources", side_effect=AssertionError("No sources loaded")), self.assertRaises(ValueError):
                    cli.command_two_layers(args)

    def test_default_verify_no_model_and_reexecute_single_model(self):
        from bioprocess_runtime import cli
        for replay in (False, True):
            args = cli.build_parser().parse_args(["gemma-two-layers-verify", "program", "--report", "report"] + (["--reexecute"] if replay else []))
            args.summary = None
            with patch.object(cli, "_load_two_layers_sources", return_value=object()), patch.object(Path, "read_text", return_value="{}"), patch.object(cli, "_holdout_model", return_value="one-model") as model, patch.object(two, "verify_two_layers", return_value={"valid": True, "two_layers_match": True}), patch.object(two, "reexecute_two_layers", return_value={"valid": True, "two_layers_match": True}) as reexecute, patch("sys.stdout", new=io.StringIO()):
                self.assertEqual(cli.command_two_layers(args), 0)
            self.assertEqual(model.call_count, int(replay))
            self.assertEqual(reexecute.call_count, int(replay))

    def test_plan_validates_before_single_model_and_exclusive_writes(self):
        from bioprocess_runtime import cli
        events = []
        source = SimpleNamespace(validate=lambda _: events.append("validate") or ([], None, None))
        with tempfile.TemporaryDirectory() as directory:
            args = cli.build_parser().parse_args(["gemma-two-layers-plan", "program", "--plan", str(Path(directory) / "plan"), "--bundle", str(Path(directory) / "bundle")])
            with patch.object(cli, "_load_two_layers_sources", return_value=source), patch.object(cli, "_holdout_model", side_effect=lambda _: events.append("model") or object()) as model, patch.object(two, "_predict", side_effect=lambda *a: (events.append("both_stages") or {"plan_sha256": "hash", "coverage": {}}, {})), patch.object(second.holdout, "write_new", side_effect=lambda *a: events.append("exclusive_write")), patch("sys.stdout", new=io.StringIO()):
                self.assertEqual(cli.command_two_layers(args), 0)
            self.assertEqual(events, ["validate", "model", "both_stages", "exclusive_write", "exclusive_write"])
            model.assert_called_once()


if __name__ == "__main__":
    unittest.main()
