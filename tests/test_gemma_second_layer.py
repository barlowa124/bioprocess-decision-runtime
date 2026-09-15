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
from bioprocess_runtime import gemma_first_layer_holdout as holdout
from bioprocess_runtime import gemma_second_layer as second
from bioprocess_runtime import gemma_second_layer_capture as capture
from bioprocess_runtime.gemma_rotary_slice import _sha, _seal
from bioprocess_runtime.operational_semantics import append_chain_record
from test_gemma_first_layer import real_program, profiles, providers, cheap_rms, cheap_rows, cheap_softmax, geometry

ROOT = Path(__file__).resolve().parents[1]


def synthetic_context():
    program = real_program()
    snapshots = {}
    for name in sorted({name for node in second.second_layer_instructions(program) for name in node["parameter_refs"]}):
        commitment = copy.deepcopy(program["parameter_commitments"][name])
        values = np.zeros(commitment["shape"], dtype=np.uint16)
        descriptor = first._descriptor(values)
        commitment["sha256"] = descriptor["sha256"]
        program["parameter_commitments"][name] = commitment
        snapshots[name] = {"format": "full_parameter_bits_v1", "bits": values, "descriptor": descriptor, "commitment": commitment}
    program = _seal({key: value for key, value in program.items() if key != "program_sha256"}, "program_sha256")
    lookup = providers()
    roots = {"hidden.1": np.full((1, 30, 640), 0x3F00, dtype=np.uint16), "rotary.local.cosine": lookup.rotary_cosine.copy(),
             "rotary.local.sine": lookup.rotary_sine.copy(), "mask.sliding": first.causal_mask_bits()}
    return program, roots, snapshots, lookup


def execute_synthetic(program, roots, snapshots, lookup):
    with ExitStack() as stack:
        stack.enter_context(patch.object(first, "PROGRAM_SHA256", program["program_sha256"]))
        stack.enter_context(patch.object(second, "_code_sha", return_value="synthetic-code"))
        stack.enter_context(patch.object(first, "_rms_lookup_row", side_effect=cheap_rms))
        rows = stack.enter_context(patch.object(first, "_project_rows", side_effect=cheap_rows))
        stack.enter_context(patch.object(first, "lookup_softmax_row", side_effect=cheap_softmax))
        stack.enter_context(patch.object(first, "execute_first_layer", side_effect=AssertionError("No relabelled frozen engine")))
        stack.enter_context(patch.object(first, "snapshot_parameters", side_effect=AssertionError("No renamed layer-zero weights")))
        stack.enter_context(patch.object(first, "bind_model_tensors", side_effect=AssertionError("Pure API cannot bind model")))
        execution = second.execute_second_layer(program, roots, snapshots, lookup, profiles(), {"synthetic": True}, 1)
        return execution, rows.call_args_list


def native_pair(execution, coverage, kernels):
    code = capture._code_sha()
    common = {"checks": dict.fromkeys(("token_commitment_unchanged", "stopped_at_decoder_tuple", "decoder_tuple_length_one", "no_later_layer", "no_final_norm", "no_lm_head", "all_needed_states_present", "code_unchanged", "runtime_unchanged", "target_layer_one", "local_rotary_and_sliding_mask"), True),
              "runtime_before": {"synthetic": True}, "runtime_after": {"synthetic": True}, "code_before": code, "code_after": code}
    traced = copy.deepcopy(common)
    traced["checks"].update(dict.fromkeys(("original_operations_once", "all_operand_links_unchanged", "rms_scalar_stages_complete", "softmax_f32_complete"), True))
    traced.update(state_bits=copy.deepcopy(execution["state_bits"]), scalar_stages=copy.deepcopy(execution["scalar_stages"]), softmax_f32_bits=copy.deepcopy(execution["softmax_f32_bits"]),
                  geometry={name: geometry(item["shape"], item["dtype"]) for name, item in coverage["states"].items()}, kernels=copy.deepcopy(kernels))
    for name, stages in traced["scalar_stages"].items():
        shape = coverage["states"][name]["shape"]
        stages["mean_input_metadata"] = {"input_shape": shape, "input_dtype": "torch.float32", "input_strides": geometry(shape, "torch.float32")["strides"], "axes": [-1], "keepdim": True, "alignment_mod16": 0}
    plain = copy.deepcopy(common)
    plain["checks"]["plain_minimal_control"] = True
    plain.update(state_bits={name: copy.deepcopy(execution["state_bits"][name]) for name in ("hidden.1", "hidden.2")}, geometry={name: geometry([1, 30, 640], "torch.bfloat16") for name in ("hidden.1", "hidden.2")})
    return {"traced": traced, "untraced": plain}


class SecondLayerIRTests(unittest.TestCase):
    def test_recorded_layer_one_result_preserves_boundary_scope(self):
        load = lambda name: json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))
        plan = load("gemma3_270m_second_layer_plan.json")
        summary = load("gemma3_270m_second_layer_summary.json")
        self.assertEqual(plan["plan_sha256"], _sha({key: value for key, value in plan.items() if key != "plan_sha256"}))
        self.assertEqual(summary["summary_sha256"], _sha({key: value for key, value in summary.items() if key != "summary_sha256"}))
        self.assertEqual(summary["plan_sha256"], plan["plan_sha256"])
        self.assertEqual(summary["trace_root"], plan["trace_root"])
        coverage = summary["coverage"]
        self.assertEqual((coverage["instruction_count"], coverage["new_state_count"], coverage["root_state_count"], coverage["parameter_count"]), (29, 30, 4, 13))
        self.assertEqual((coverage["layer_index"], coverage["rotary_profile"], coverage["sliding_window"], coverage["first_full_attention_index"]), (1, "local", 512, 5))
        self.assertEqual((coverage["rms_scalar_positions"], coverage["softmax_fp32_positions"]), (810, 3600))
        self.assertEqual(summary["original_forward_count"], 6)
        self.assertEqual(summary["aggregate_mismatch_count"], 0)
        self.assertTrue(summary["second_layer_matches"])
        self.assertTrue(summary["layer_one_independently_recomputed"])
        self.assertTrue(summary["prefix_boundary_reused"])
        self.assertTrue(summary["stored_boundary_predictions_used"])
        self.assertTrue(all(summary["checks"].values()))
        self.assertTrue(all(summary["acquisition_guards"].values()))
        for field in second.FALSE_FLAGS:
            self.assertFalse(summary[field], field)

    def test_actual_slice_regime_root_geometry_and_parameter_refs(self):
        program = real_program()
        nodes = second.second_layer_instructions(program)
        self.assertEqual([node["id"] for node in nodes], [f"i{index:04d}" for index in range(36, 65)])
        self.assertEqual(sum(len(node["outputs"]) for node in nodes), 30)
        self.assertEqual({name for node in nodes for name in node["inputs"]} - {name for node in nodes for name in node["outputs"]}, set(second.ROOTS))
        names = {name for node in nodes for name in node["parameter_refs"]}
        self.assertEqual(len(names), 13)
        self.assertTrue(all(name.startswith("model.layers.1.") for name in names))
        coverage = second._coverage(program)
        self.assertEqual(coverage["first_full_attention_index"], 5)
        self.assertEqual(coverage["sliding_window"], 512)
        self.assertEqual(coverage["rotary_profile"], "local")
        self.assertEqual(set(coverage["states"]), set(capture.STATE_SPECS))
        self.assertEqual(len(second._native_kernel_roles(program)), 22)
        self.assertEqual(sum(int(np.prod(first._shape(program, node["outputs"][0])[:-1])) * 3 for node in nodes if node["opcode"] == "RMS_NORM"), 810)
        for node in nodes:
            self.assertEqual(node["instruction_sha256"], _sha({key: value for key, value in node.items() if key != "instruction_sha256"}))

    def test_altered_instruction_global_root_mask_and_geometry_fail(self):
        mutations = [lambda p: p["instructions"][45]["attributes"].update(rotary_profile="global"),
                     lambda p: p["instructions"][50]["inputs"].__setitem__(1, "mask.full"),
                     lambda p: p["instructions"][45]["inputs"].__setitem__(2, "rotary.global.cosine"),
                     lambda p: p["instructions"][37]["parameter_refs"].__setitem__(0, "model.layers.0.self_attn.q_proj.weight"),
                     lambda p: p["instructions"][49]["attributes"].update(scalar=0.5),
                     lambda p: p["instructions"][36].update(instruction_sha256="0" * 64),
                     lambda p: p["configuration"]["layer_types"].__setitem__(1, "full_attention"),
                     lambda p: p["tensors"]["hidden.2"].update(shape=[1, 29, 640]),
                     lambda p: p["instructions"].pop(36)]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                program = real_program()
                mutation(program)
                with self.assertRaises(ValueError):
                    second.second_layer_instructions(program)
                program = _seal({key: value for key, value in program.items() if key != "program_sha256"}, "program_sha256")
                with self.assertRaises(ValueError):
                    second.second_layer_instructions(program)

    def test_pure_api_and_explicit_linear_profiles(self):
        self.assertEqual(list(inspect.signature(second.execute_second_layer).parameters), ["program", "root_bits", "parameter_snapshots", "providers", "profiles", "runtime", "workers"])
        modes = {node["outputs"][0]: second._linear_mode(node) for node in second.second_layer_instructions(real_program()) if node["opcode"] == "LINEAR"}
        self.assertEqual(modes, second.LINEAR_MODES)
        self.assertEqual(modes["layer.1.attention.projected"], "output")
        self.assertEqual(modes["layer.1.mlp.down"], "down")
        self.assertEqual(modes["layer.1.key.flat"], "key_value")
        with self.assertRaises(ValueError):
            second._linear_mode({"outputs": ["layer.0.query.flat"]})

    def test_roots_and_wrong_constant_provider_frequency_reject(self):
        program, roots, _, lookup = synthetic_context()
        for name in second.ROOTS:
            changed = dict(roots)
            changed.pop(name)
            with self.assertRaises(ValueError):
                second._roots(program, changed, lookup)
        changed = dict(roots, **{"mask.full": roots["mask.sliding"]})
        with self.assertRaises(ValueError):
            second._roots(program, changed, lookup)
        for name in second.ROOTS[1:]:
            changed = {key: value.copy() for key, value in roots.items()}
            changed[name].reshape(-1)[-1] ^= 1
            with self.assertRaisesRegex(ValueError, "checked provider"):
                second._roots(program, changed, lookup)
        with self.assertRaisesRegex(ValueError, "inverse-frequency"):
            lookup.validate({"synthetic": True}, np.zeros(128, dtype=np.uint32))

    def test_qualification_flags_are_explicit_boundary_reuse(self):
        flags = second._flags()
        self.assertTrue(flags["prefix_boundary_reused"])
        self.assertTrue(flags["stored_intermediate_predictions_used"])
        self.assertTrue(flags["stored_boundary_predictions_used"])
        self.assertTrue(flags["empirical_providers_reused"])
        self.assertFalse(flags["internal_layer_one_intermediates_reused"])
        self.assertFalse(flags["shape_transfer_prequalified"])
        self.assertFalse(flags["connected_two_layers_independently_recomputed"])
        self.assertNotEqual(second.SCOPE, first.SCOPE)
        self.assertTrue(all(flags[name] is False for name in second.FALSE_FLAGS))

    def test_small_worker_unchanged_profile_arithmetic(self):
        left = np.asarray([[0x3F80] * 32, [0x3F00] * 32], dtype=np.uint16)
        weights = np.asarray([[0x3F80] * 32, [0] * 32], dtype=np.uint16)
        expected = first._project_rows(left, weights, "serial", profiles()["serial"], 1)
        actual = first._project_rows(left, weights, "serial", profiles()["serial"], 2)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(first._dot([0] * 640, [0] * 640, "key_value", profiles()["key_value"]), 0)
        self.assertEqual(first._dot([0] * 1024, [0] * 1024, "output", profiles()["output"]), 0)
        self.assertEqual(first._dot([0] * 2048, [0] * 2048, "down", profiles()["down"]), 0)

    def test_actual_layer_one_class_methods_and_geometry_without_forward(self):
        import torch
        from transformers.models.gemma3 import modeling_gemma3 as gemma
        from transformers import Gemma3TextConfig
        config = Gemma3TextConfig(hidden_size=640, intermediate_size=2048, num_attention_heads=4, num_key_value_heads=1, head_dim=256, num_hidden_layers=3, layer_types=["sliding_attention"] * 3, sliding_window=512, query_pre_attn_scalar=256, attention_dropout=0.0)
        config._attn_implementation = "eager"
        layer = gemma.Gemma3DecoderLayer(config, 1).to(dtype=torch.bfloat16).eval()
        model = SimpleNamespace(config=config, model=SimpleNamespace(layers=[None, layer], embed_tokens=torch.nn.Embedding(1, 640, dtype=torch.bfloat16)))
        parameters = {"model.layers.1." + name: value for name, value in layer.named_parameters()}
        sources = SimpleNamespace(program=real_program(), runtime={"synthetic": True}, baseline=SimpleNamespace(first_sources=SimpleNamespace(post_feedforward=None)))
        with patch.object(first.ff, "_model_context"), patch.object(first, "bind_model_tensors", return_value=parameters), patch.object(first, "_runtime", return_value=sources.runtime), patch.object(second, "_original", wraps=second._original) as original:
            self.assertEqual(set(second._model_context(sources, model)), set(parameters))
            self.assertEqual(original.call_count, 17)
            for module, attribute, value in ((layer, "layer_idx", 0), (layer.self_attn, "is_sliding", False), (layer.self_attn.q_proj, "in_features", 128), (layer.self_attn.k_norm, "eps", 1e-5)):
                with patch.object(module, attribute, value), self.assertRaises(ValueError):
                    second._model_context(sources, model)
            with patch.object(layer.self_attn, "forward", lambda *args: None), self.assertRaises(ValueError):
                second._model_context(sources, model)
        commitment = second._implementation_commitment()
        self.assertEqual(len(commitment), 11)
        self.assertEqual(commitment, second._implementation_commitment())

    def test_upstream_compact_evidence_hashes_are_pinned_without_prediction(self):
        for filename, field, digest in (("gemma3_270m_first_layer_holdout_protocol.json", "protocol_sha256", second.HOLDOUT_PROTOCOL_SHA256), ("gemma3_270m_first_layer_holdout_plan.json", "plan_sha256", second.HOLDOUT_PLAN_SHA256)):
            payload = json.loads((ROOT / "results" / filename).read_text(encoding="utf-8"))
            self.assertEqual(payload[field], digest)
            self.assertEqual(_sha({key: value for key, value in payload.items() if key != field}), digest)


class SecondLayerExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.program, cls.roots, cls.snapshots, cls.lookup = synthetic_context()
        cls.execution, cls.calls = execute_synthetic(cls.program, cls.roots, cls.snapshots, cls.lookup)

    def check(self, execution=None, roots=None, snapshots=None):
        with patch.object(first, "PROGRAM_SHA256", self.program["program_sha256"]):
            second.check_execution(self.program, self.roots if roots is None else roots, self.snapshots if snapshots is None else snapshots, self.lookup, profiles(), {"synthetic": True}, self.execution if execution is None else execution)

    def test_synthetic_execution_29_nodes_34_states_and_all_auxiliaries(self):
        self.check()
        execution = self.execution
        self.assertEqual(len(execution["records"]), 29)
        self.assertEqual(len(execution["state_bits"]), 34)
        self.assertEqual(sum(len(stage) for stages in execution["scalar_stages"].values() for stage in stages.values()), 810)
        self.assertEqual(np.asarray(execution["softmax_f32_bits"]).size, 3600)
        self.assertEqual(execution["records"][0]["payload"]["inputs"]["hidden.1"]["producer_record_hash"], "ROOT:" + first._descriptor(self.roots["hidden.1"])["sha256"])
        self.assertEqual(len(execution["records"][9]["payload"]["outputs"]), 2)
        qk = [call for call in self.calls if call.args[0].shape == (30, 256) and call.args[1].shape == (30, 256)]
        pv = [call for call in self.calls if call.args[0].shape == (30, 32)]
        self.assertEqual(len(qk), 4)
        self.assertEqual(len(pv), 4)
        for call in pv:
            self.assertTrue(np.all(call.args[0][:, -2:] == 0))
            self.assertTrue(np.all(call.args[1][:, -2:] == 0))
            self.assertEqual(call.args[1].shape, (256, 32))
        self.assertEqual([call.args[2] for call in self.calls if call.args[1].shape == (640, 1024)], ["output"])
        self.assertEqual([call.args[2] for call in self.calls if call.args[1].shape == (640, 2048)], ["down"])

    def test_changed_hidden_one_changes_hidden_two_without_old_predictions(self):
        roots = {name: value.copy() for name, value in self.roots.items()}
        roots["hidden.1"][0, 29, 639] = 0x3F80
        execution, _ = execute_synthetic(self.program, roots, self.snapshots, self.lookup)
        self.assertNotEqual(execution["state_bits"]["hidden.2"], self.execution["state_bits"]["hidden.2"])
        self.assertNotEqual(execution["records"][0]["record_hash"], self.execution["records"][0]["record_hash"])

    def test_parameter_coverage_dtype_hash_and_actual_layer_names(self):
        with patch.object(first, "PROGRAM_SHA256", self.program["program_sha256"]):
            self.assertEqual(len(second.check_parameter_snapshots(self.program, self.snapshots)), 13)
            for kind in ("missing", "old_name", "dtype", "hash", "partial_format"):
                changed = copy.deepcopy(self.snapshots)
                name = next(iter(changed))
                if kind == "missing":
                    changed.pop(name)
                elif kind == "old_name":
                    changed[name.replace("layers.1.", "layers.0.")] = changed.pop(name)
                elif kind == "dtype":
                    changed[name]["bits"] = changed[name]["bits"].astype(np.uint32)
                elif kind == "hash":
                    changed[name]["bits"].reshape(-1)[-1] ^= 1
                else:
                    changed[name]["format"] = "selected_token_rows_v1"
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    second.check_parameter_snapshots(self.program, changed)

    def test_snapshot_reads_actual_thirteen_weights_not_layer_zero(self):
        import torch
        parameters = {name: torch.from_numpy(snapshot["bits"].copy()).view(torch.bfloat16) for name, snapshot in self.snapshots.items()}
        with patch.object(first, "PROGRAM_SHA256", self.program["program_sha256"]), patch.object(first, "snapshot_parameters", side_effect=AssertionError("Layer-zero snapshot forbidden")):
            result = second.snapshot_parameters(self.program, parameters)
        self.assertEqual(set(result), set(self.snapshots))
        self.assertTrue(all(result[name]["descriptor"] == self.snapshots[name]["descriptor"] for name in result))
        parameters[next(iter(parameters))] = next(iter(parameters.values())).float()
        with patch.object(first, "PROGRAM_SHA256", self.program["program_sha256"]), self.assertRaises(ValueError):
            second.snapshot_parameters(self.program, parameters)

    def test_missing_state_root_or_auxiliary_reject(self):
        for field, key in (("state_bits", "hidden.2"), ("state_bits", "hidden.1"), ("scalar_stages", "layer.1.key.normalized")):
            changed = copy.deepcopy(self.execution)
            changed[field].pop(key)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check(changed)
        changed = copy.deepcopy(self.execution)
        changed["softmax_f32_bits"][0][3][29].pop()
        with self.assertRaises(ValueError):
            self.check(changed)

    def test_wrong_producer_and_resealed_ledger_tamper_reject(self):
        for target in ("producer", "output", "root", "auxiliary", "provider"):
            changed = copy.deepcopy(self.execution)
            if target == "producer":
                changed["records"][1]["payload"]["inputs"]["layer.1.attention.normalized"]["producer_record_hash"] = "ROOT:wrong"
            elif target == "output":
                changed["state_bits"]["hidden.2"][0][29][639] ^= 1
            elif target == "root":
                changed["state_bits"]["hidden.1"][0][29][639] ^= 1
            elif target == "auxiliary":
                changed["scalar_stages"]["layer.1.key.normalized"]["rsqrt_bits"][-1] ^= 1
            else:
                changed["records"][1]["payload"]["provider"]["name"] = "key_value"
            records = []
            for record in changed["records"]:
                append_chain_record(records, record["payload"])
            changed["records"] = records
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.check(changed)

    def test_worker_and_profile_invalid_before_math(self):
        with patch.object(first, "PROGRAM_SHA256", self.program["program_sha256"]), patch.object(second, "_code_sha", return_value="synthetic-code"):
            for workers in (0, 5, True):
                with self.subTest(workers=workers), self.assertRaises(ValueError):
                    second.execute_second_layer(self.program, self.roots, self.snapshots, self.lookup, profiles(), {"synthetic": True}, workers)
            changed = profiles()
            changed["down"]["id"] = "old-failed-candidate"
            with self.assertRaises(ValueError):
                second.execute_second_layer(self.program, self.roots, self.snapshots, self.lookup, changed, {"synthetic": True}, 1)


class SecondLayerNativeRecountTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        program, roots, snapshots, lookup = synthetic_context()
        execution, _ = execute_synthetic(program, roots, snapshots, lookup)
        with patch.object(first, "PROGRAM_SHA256", program["program_sha256"]):
            coverage = second._coverage(program)
            roles = second._native_kernel_roles(program)
        cls.bundle = {"execution": execution, "parameter_snapshots": {}}
        kernels = {name: ["synthetic_kernel"] for name in roles}
        cls.plan = _seal({"scope": second.SCOPE, "bundle_sha256": _sha(cls.bundle), "trace_root": execution["records"][-1]["record_hash"], "coverage": coverage,
                          "runtime": {"synthetic": True}, "capture_code_sha256": capture._code_sha(), "kernel_transfer": {"expected_symbols": kernels}, "native_kernel_roles": roles, **second._flags()}, "plan_sha256")
        cls.pair = native_pair(execution, coverage, kernels)

    def report(self, pair=None, guards=None):
        return second.second_layer_report(self.plan, self.bundle, [self.pair if pair is None else pair, self.pair, self.pair], guards)

    def test_full_native_recount_and_root_controls(self):
        report = self.report()
        self.assertTrue(report["second_layer_matches"])
        self.assertTrue(report["layer_one_independently_recomputed"])
        self.assertEqual(report["original_forward_count"], 6)
        self.assertEqual(report["aggregate_mismatch_count"], 0)
        self.assertEqual(set(report["source_root_comparison"][0]), set(second.ROOTS))
        self.assertEqual(len(report["kernel_trace"][0]), 22)
        self.assertFalse(report["shape_transfer_prequalified"])
        self.assertFalse(report["connected_two_layers_independently_recomputed"])

    def test_last_fp32_coordinate_mismatch_not_hidden_by_bf16(self):
        pair = copy.deepcopy(self.pair)
        pair["traced"]["softmax_f32_bits"][0][3][29][29] ^= 1
        report = self.report(pair)
        self.assertFalse(report["second_layer_matches"])
        self.assertEqual(report["mismatch_counts"]["softmax_f32_bits"], 1)
        self.assertEqual(report["mismatch_counts"]["hidden.2"], 0)
        self.assertEqual(report["first_divergence"]["coordinate"], [0, 3, 29, 29])
        self.assertEqual(report["first_divergence"]["state"], "softmax_f32_bits")

    def test_last_rms_coordinate_mismatch_not_hidden_by_output(self):
        pair = copy.deepcopy(self.pair)
        pair["traced"]["scalar_stages"]["layer.1.mlp.post_normalized"]["rsqrt_bits"][-1] ^= 1
        report = self.report(pair)
        self.assertFalse(report["layer_one_independently_recomputed"])
        self.assertEqual(report["first_divergence"]["coordinate"], [29])
        self.assertEqual(report["mismatch_counts"]["hidden.2"], 0)

    def test_root_and_both_plain_boundaries_are_compared(self):
        for kind, name in (("traced", "hidden.1"), ("traced", "mask.sliding"), ("untraced", "hidden.1"), ("untraced", "hidden.2")):
            pair = copy.deepcopy(self.pair)
            value = np.asarray(pair[kind]["state_bits"][name], dtype=np.uint16)
            value.reshape(-1)[-1] ^= 1
            pair[kind]["state_bits"][name] = value.tolist()
            report = self.report(pair)
            with self.subTest(kind=kind, name=name):
                self.assertFalse(report["second_layer_matches"])
                self.assertEqual(report["aggregate_mismatch_count"], 1)
                self.assertEqual(report["first_divergence"]["state"], name if kind == "traced" else "plain_" + name)

    def test_native_guard_kernel_runtime_geometry_and_code_failures(self):
        mutations = [lambda pair: pair["traced"]["checks"].update(target_layer_one=False),
                     lambda pair: pair["untraced"]["checks"].pop("plain_minimal_control"),
                     lambda pair: pair["traced"]["runtime_after"].update(changed=True),
                     lambda pair: pair["traced"].update(code_after="changed"),
                     lambda pair: pair["traced"]["kernels"].update(softmax=["different_kernel"]),
                     lambda pair: pair["traced"]["geometry"]["hidden.2"].update(device="cpu"),
                     lambda pair: pair["traced"]["scalar_stages"]["layer.1.query.normalized"]["mean_input_metadata"].update(input_dtype="torch.bfloat16")]
        for mutation in mutations:
            pair = copy.deepcopy(self.pair)
            mutation(pair)
            with self.subTest(mutation=mutation):
                self.assertFalse(self.report(pair)["second_layer_matches"])

    def test_incomplete_capture_preserved_as_abstention(self):
        for mutation in (lambda pair: pair["traced"]["state_bits"].pop("layer.1.key.rotary"),
                         lambda pair: pair["traced"]["scalar_stages"].pop("layer.1.key.normalized"),
                         lambda pair: pair["untraced"].update(kernels={"extra": ["kernel"]}),
                         lambda pair: pair.update(traced={"capture_failure": {"message": "unsupported"}})):
            pair = copy.deepcopy(self.pair)
            mutation(pair)
            result = self.report(pair)
            self.assertFalse(result["second_layer_matches"])
            self.assertFalse(result["native_coverage_complete"])
            self.assertIn("abstention", result)
            self.assertEqual(result["observations"][0], pair)

    def test_after_acquisition_guard_failure_keeps_all_numerical_evidence(self):
        result = self.report(guards={"checkpoint_unchanged": False, "source_unchanged": True, "frozen_files_unchanged": True})
        self.assertFalse(result["second_layer_matches"])
        self.assertTrue(result["native_coverage_complete"])
        self.assertEqual(result["aggregate_mismatch_count"], 0)
        self.assertEqual(result["first_divergence"], {"check": "acquisition_guards"})


class SecondLayerSourceTests(unittest.TestCase):
    def test_real_source_roots_and_kernel_preflight_without_prediction(self):
        from bioprocess_runtime.cli import build_parser, _load_holdout_baseline
        args = build_parser().parse_args(["gemma-second-layer-plan", "results/gemma3_270m_execution_ir.json", "--output", "unused-plan.json"])
        for name, value in vars(args).items():
            if isinstance(value, Path) and not value.is_absolute():
                setattr(args, name, ROOT / value)
        paths = [value for name, value in vars(args).items() if isinstance(value, Path) and name not in ("output", "plan", "bundle", "summary") and "audit" not in name]
        if any(not path.exists() for path in paths):
            self.skipTest("Ignored source evidence is unavailable")
        load = lambda path: json.loads(path.read_text(encoding="utf-8"))
        source = second.SecondLayerSources(_load_holdout_baseline(args), load(args.protocol), load(args.holdout_plan), load(args.holdout_bundle), load(args.holdout_report))
        with patch.object(second, "execute_second_layer", side_effect=AssertionError("Preflight must not predict")), patch.object(capture, "capture_second_layer", side_effect=AssertionError("Preflight must not acquire")):
            ids, checked_providers, arithmetic, roots, bindings = source.validate(args.model_path)
            transfer = second._kernel_transfer(source.program, source.baseline)
        self.assertEqual(len(ids[0]), 30)
        self.assertIs(type(checked_providers), first.Providers)
        self.assertEqual(set(roots), set(second.ROOTS))
        self.assertEqual(set(bindings), set(second.ROOTS))
        self.assertEqual(arithmetic["down"]["chunk_size"], 192)
        self.assertEqual(len(transfer["expected_symbols"]), 22)
        self.assertFalse(transfer["shape_transfer_prequalified"])

    def fixture(self, stack):
        program = real_program()
        ids = [list(range(30))]
        frequency = np.zeros(128, dtype=np.uint32)
        lookup = providers(first._descriptor(frequency)["sha256"])
        roots = {"hidden.1": np.full((1, 30, 640), 0x3F00, dtype=np.uint16).tolist(), "rotary.local.cosine": lookup.rotary_cosine.tolist(), "rotary.local.sine": lookup.rotary_sine.tolist(), "mask.sliding": first.causal_mask_bits().tolist()}
        descriptors = {name: first._descriptor(first._array(roots[name], shape, "torch.bfloat16")) for name, shape in zip(second.ROOTS, second.ROOT_SHAPES)}
        records = [{"record_hash": "source-producer-" + name, "payload": {"instruction_id": program["tensors"][name]["producer"], "outputs": {name: descriptors[name]}}} for name in second.ROOTS]
        baseline_report = {"observations": [{"traced": {"kernels": {name: ["kernel"] for name in first._native_kernel_roles(program)}}}]}
        baseline_bundle = {"execution": {"state_bits": {**roots, "position_ids": ids}, "records": records}, "parameter_snapshots": {first.FREQUENCY: {"bits": frequency.tolist()}}}
        baseline = SimpleNamespace(first_sources=SimpleNamespace(program=program, runtime={"synthetic": True}), baseline_plan={"input_token_ids": ids, "input_ids": first._descriptor(np.asarray(ids, dtype=np.int64)), "state_descriptors": descriptors}, baseline_bundle=baseline_bundle, baseline_report=baseline_report)
        baseline.validate = Mock(return_value=(ids, lookup, profiles()))
        baseline.commitments = lambda: {"plan": _sha(baseline.baseline_plan), "bundle": _sha(baseline.baseline_bundle), "report": _sha(baseline.baseline_report)}
        protocol = _seal({"test": "protocol"}, "protocol_sha256")
        plan = _seal({"test": "plan"}, "plan_sha256")
        report = _seal({"native_coverage_complete": True, "original_forward_count": 12, "required_case_ids": list(holdout.CASE_IDS), "cases": [{"case_id": name, "case_matches": True, "native_coverage_complete": True} for name in holdout.CASE_IDS]}, "report_sha256")
        sources = second.SecondLayerSources(baseline, protocol, plan, {}, report)
        for name, value in (("HOLDOUT_PROTOCOL_SHA256", protocol["protocol_sha256"]), ("HOLDOUT_PLAN_SHA256", plan["plan_sha256"]), ("HOLDOUT_REPORT_SHA256", report["report_sha256"]), ("_code_sha", None)):
            stack.enter_context(patch.object(second, name, value) if value is not None else patch.object(second, name, return_value="code"))
        checked = {"valid": True, "holdout_matches": True, "prediction_complete": True, "connected_first_layer_independently_recomputed": True}
        def verify(cached, *args):
            cached.validate()
            return checked
        stack.enter_context(patch.object(holdout, "verify_holdout", side_effect=verify))
        return sources, checked, lookup

    def test_transparent_per_call_cache_roots_producers_and_frequency(self):
        with ExitStack() as stack:
            sources, _, _ = self.fixture(stack)
            ids, _, _, roots, bindings = sources.validate(ROOT)
            self.assertEqual(ids, [list(range(30))])
            self.assertEqual(set(roots), set(second.ROOTS))
            self.assertEqual(bindings["hidden.1"]["source_producer_record_hash"], "source-producer-hidden.1")
            self.assertEqual(sources.baseline.validate.call_count, 1)
            sources.validate(ROOT)
            self.assertEqual(sources.baseline.validate.call_count, 2)
            transfer = second._kernel_transfer(sources.program, sources.baseline)
            self.assertEqual(len(transfer["expected_symbols"]), 22)
            self.assertEqual(transfer["role_mapping_for_symbol_sets_only"]["hidden.2"], "hidden.1")
            self.assertFalse(transfer["shape_transfer_prequalified"])

    def test_failed_holdout_any_required_check_rejected(self):
        for key in ("valid", "holdout_matches", "prediction_complete", "connected_first_layer_independently_recomputed"):
            with ExitStack() as stack:
                sources, checked, _ = self.fixture(stack)
                checked[key] = False
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "both-case"):
                    sources.validate(ROOT)

    def test_omitted_holdout_case_cannot_be_hidden_by_valid_boolean(self):
        with ExitStack() as stack:
            sources, _, _ = self.fixture(stack)
            changed = copy.deepcopy(sources.holdout_report)
            changed["cases"].pop()
            changed = _seal({key: value for key, value in changed.items() if key != "report_sha256"}, "report_sha256")
            sources = second.SecondLayerSources(sources.baseline, sources.protocol, sources.holdout_plan, sources.holdout_bundle, changed)
            stack.enter_context(patch.object(second, "HOLDOUT_REPORT_SHA256", changed["report_sha256"]))
            with self.assertRaisesRegex(ValueError, "Omitted"):
                sources.validate(ROOT)

    def test_source_root_position_descriptor_and_frequency_mutations_reject(self):
        for kind in ("root", "position", "producer", "frequency", "provider"):
            with ExitStack() as stack:
                sources, _, lookup = self.fixture(stack)
                if kind == "root":
                    sources.baseline.baseline_bundle["execution"]["state_bits"]["rotary.local.sine"][0][29][255] ^= 1
                elif kind == "position":
                    sources.baseline.baseline_bundle["execution"]["state_bits"]["position_ids"] = [[1] * 30]
                elif kind == "producer":
                    sources.baseline.baseline_bundle["execution"]["records"][0]["payload"]["instruction_id"] = "i0064"
                elif kind == "frequency":
                    sources.baseline.baseline_bundle["parameter_snapshots"][first.FREQUENCY]["bits"][-1] ^= 1
                else:
                    lookup.rotary_cosine[0, 29, 255] ^= 1
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    sources.validate(ROOT)

    def test_default_verify_never_executes_math_or_loads_model(self):
        sources = SimpleNamespace(commitments=lambda: {})
        plan, bundle = {}, {}
        report = _seal({"observations": [], "acquisition_guards": {}, "second_layer_matches": True, "mismatch_counts": {}, "first_divergence": None}, "report_sha256")
        with patch.object(second, "_code_sha", return_value="code"), patch.object(second, "check_second_layer_plan") as check, patch.object(second, "second_layer_report", return_value=report), patch.object(second, "execute_second_layer", side_effect=AssertionError("No replay")), patch.object(second, "_snapshot_model", side_effect=AssertionError("No model")):
            result = second.verify_second_layer(sources, plan, bundle, report, ROOT)
        self.assertTrue(result["valid"])
        self.assertFalse(result["actual_layer1_numerical_recompute"])
        self.assertTrue(result["prefix_boundary_reused"])
        check.assert_called_once()

    def test_reexecute_rebuilds_before_capture_and_blocks_capture_on_difference(self):
        for same in (True, False):
            events = []
            plan, bundle, report = {"plan": True}, {"bundle": True}, {"second_layer_matches": True}
            def build(*args):
                events.append("all29fresh")
                return (plan, bundle) if same else ({"changed": True}, bundle)
            def acquire(*args):
                events.append("sixnative")
                return report
            with patch.object(second, "verify_second_layer", return_value={"valid": True}), patch.object(second, "build_second_layer_plan", side_effect=build), patch.object(second, "acquire_second_layer", side_effect=acquire):
                result = second.reexecute_second_layer(None, None, plan, bundle, report, ROOT, ROOT, ROOT)
            self.assertEqual(events, ["all29fresh", "sixnative"] if same else ["all29fresh"])
            self.assertEqual(result["valid"], same)
            self.assertTrue(result["stored_boundary_predictions_used"])


class SecondLayerCLITests(unittest.TestCase):
    def test_commands_inherit_sources_defaults_without_changing_holdout_headers(self):
        from bioprocess_runtime.cli import build_parser, command_second_layer, command_first_layer_holdout
        parser = build_parser()
        for operation in ("plan", "run", "verify"):
            arguments = ["gemma-second-layer-" + operation, "results/gemma3_270m_execution_ir.json", "--report" if operation == "verify" else "--output", "unused"]
            args = parser.parse_args(arguments)
            self.assertIs(args.handler, command_second_layer)
            self.assertEqual(args.workers, 4)
            self.assertEqual(args.plan, Path("results/gemma3_270m_second_layer_plan.json"))
            self.assertEqual(args.bundle, Path("artifacts/gemma3_270m_second_layer_predictions.json"))
            self.assertEqual(args.protocol, Path("results/gemma3_270m_first_layer_holdout_protocol.json"))
            self.assertEqual(args.holdout_plan, Path("results/gemma3_270m_first_layer_holdout_plan.json"))
            self.assertEqual(args.holdout_report, Path("artifacts/gemma3_270m_first_layer_holdout_report.json"))
            self.assertEqual(args.model_down_plan, Path("results/gemma3_270m_mlp_down_plan_v2.json"))
            self.assertEqual(args.summary, Path("results/gemma3_270m_second_layer_summary.json"))
        for operation in ("protocol", "plan"):
            args = parser.parse_args(["gemma-first-layer-holdout-" + operation, "program", "--output", "unused"])
            self.assertIs(args.handler, command_first_layer_holdout)
            self.assertEqual(args.plan, Path("results/gemma3_270m_first_layer_holdout_plan.json"))

    def test_output_exclusive_distinct_and_model_directory_guards(self):
        from bioprocess_runtime.cli import build_parser, command_second_layer
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for output, bundle in ((root / "same", root / "same"), (ROOT / "results/gemma3_270m_execution_ir.json", root / "bundle"), (root / "inside-model", root / "bundle")):
                args = build_parser().parse_args(["gemma-second-layer-plan", str(ROOT / "results/gemma3_270m_execution_ir.json"), "--output", str(output), "--bundle", str(bundle), "--model-path", str(root)])
                with patch("bioprocess_runtime.cli._load_holdout_baseline", side_effect=AssertionError("Guard before source/model load")), self.assertRaises(ValueError):
                    command_second_layer(args)

    def test_cli_failed_run_persists_report_before_nonzero_exit(self):
        from bioprocess_runtime.cli import build_parser, command_second_layer
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = build_parser().parse_args(["gemma-second-layer-run", "program", "--output", str(root / "report"), "--summary", str(root / "summary")])
            report = {"second_layer_matches": False, "preserved": "numerical-mismatch"}
            with patch("bioprocess_runtime.cli._load_holdout_baseline", return_value=None), patch("bioprocess_runtime.cli.json.loads", return_value={}), patch.object(Path, "read_text", return_value="{}"), patch.object(second, "SecondLayerSources"), patch.object(second, "check_second_layer_plan"), patch("bioprocess_runtime.cli._holdout_model") as loader, patch.object(second, "acquire_second_layer", return_value=report), patch.object(second, "second_layer_summary", return_value={"summary": True}), patch.object(holdout, "write_new") as writer, patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(command_second_layer(args), 1)
            self.assertEqual(writer.call_args_list[0].args, (args.output, report))
            self.assertEqual(writer.call_count, 2)
            loader.assert_called_once()

    def test_cli_default_verify_never_loads_checkpoint(self):
        from bioprocess_runtime.cli import build_parser, command_second_layer
        args = build_parser().parse_args(["gemma-second-layer-verify", "program", "--report", "report"])
        result = {"valid": True, "second_layer_matches": True}
        with patch("bioprocess_runtime.cli._load_holdout_baseline", return_value=None), patch("bioprocess_runtime.cli.json.loads", return_value={}), patch.object(Path, "read_text", return_value="{}"), patch.object(second, "SecondLayerSources"), patch.object(second, "verify_second_layer", return_value=result), patch.object(second, "second_layer_summary", return_value={}), patch("bioprocess_runtime.cli._holdout_model", side_effect=AssertionError("Default verification must not load model")), patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(command_second_layer(args), 0)

    def test_cli_plan_validates_before_single_fresh_model_load(self):
        from bioprocess_runtime.cli import build_parser, command_second_layer
        events = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = build_parser().parse_args(["gemma-second-layer-plan", str(ROOT / "results/gemma3_270m_execution_ir.json"), "--output", str(root / "plan"), "--bundle", str(root / "bundle")])
            source = SimpleNamespace(validate=lambda path: events.append("source") or ("context",))
            plan = {"plan_sha256": "plan", "coverage": {}, "profiles": {}}
            with patch("bioprocess_runtime.cli._load_holdout_baseline", return_value=None), patch("bioprocess_runtime.cli.json.loads", return_value={}), patch.object(Path, "read_text", return_value="{}"), patch.object(second, "SecondLayerSources", return_value=source), patch("bioprocess_runtime.cli._holdout_model", side_effect=lambda path: events.append("load") or "model") as loader, patch.object(second, "_predict", side_effect=lambda *args: (events.append("predict") or plan, {})), patch.object(holdout, "write_new"), patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(command_second_layer(args), 0)
            self.assertEqual(events, ["source", "load", "predict"])
            loader.assert_called_once()


if __name__ == "__main__":
    unittest.main()
