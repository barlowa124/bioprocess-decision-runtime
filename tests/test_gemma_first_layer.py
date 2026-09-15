from __future__ import annotations

import copy
import hashlib
import inspect
import io
import json
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from bioprocess_runtime import gemma_first_layer as first
from bioprocess_runtime import gemma_first_layer_capture as capture
from bioprocess_runtime.gemma_rotary_slice import _sha, _seal
from bioprocess_runtime.operational_semantics import append_chain_record

ROOT = Path(__file__).resolve().parents[1]


def real_program():
    return json.loads((ROOT / "results/gemma3_270m_execution_ir.json").read_text(encoding="utf-8"))


def profiles():
    return {"serial": "operand_alignment_v1", "key_value": list(first.DENSE_SPLIT_PROFILE),
            "output": copy.deepcopy(first._OUTPUT_PROFILE), "down": json.loads(first.SUPPORTED_CANDIDATE_JSON)}


def providers(frequency_hash="frequency"):
    objects = []
    for cls in (first.CheckedRsqrtLookup, first.CheckedExpLookup, first.CheckedGeluLookup):
        value = cls.__new__(cls)
        value._runtime = {"synthetic": True}
        value._table = np.full(65536, 0x3F00, dtype=np.uint16) if cls is first.CheckedGeluLookup else np.zeros(1, dtype=np.uint32)
        value._evidence = {"table_sha256": hashlib.sha256(memoryview(value._table)).hexdigest()}
        objects.append(value)
    cosine = np.full((1, 30, 256), 0x3F80, dtype=np.uint16)
    sine = np.zeros_like(cosine)
    evidence = {"positions": list(range(30)), "table_descriptors": {"cosine": first._descriptor(cosine), "sine": first._descriptor(sine)}}
    return first.Providers(*objects, cosine, sine, frequency_hash, {"synthetic": True}, evidence)


def synthetic_context():
    program = real_program()
    nodes = first.first_layer_instructions(program)
    ids = [list(range(30))]
    snapshots = {}
    for name in sorted({name for node in nodes for name in node["parameter_refs"]}):
        commitment = copy.deepcopy(program["parameter_commitments"][name])
        if name == first.EMBEDDING:
            values = np.full((30, 640), 0x3F00, dtype=np.uint16)
            snapshots[name] = {"format": "selected_token_rows_v1", "token_indices": ids[0].copy(), "bits": values,
                               "descriptor": first._descriptor(values), "full_table_commitment": commitment,
                               "membership_attestation": "selected_by_index_during_fresh_hash_verified_model_binding"}
        else:
            dtype = np.uint32 if name == first.FREQUENCY else np.uint16
            values = np.zeros(commitment["shape"], dtype=dtype)
            if name == "model.embed_tokens.embed_scale":
                values[...] = 0x3F80
            descriptor = first._descriptor(values)
            commitment["sha256"] = descriptor["sha256"]
            program["parameter_commitments"][name] = commitment
            snapshots[name] = {"format": "full_parameter_bits_v1", "bits": values, "descriptor": descriptor, "commitment": commitment}
    program = _seal({key: value for key, value in program.items() if key != "program_sha256"}, "program_sha256")
    lookup = providers(snapshots[first.FREQUENCY]["descriptor"]["sha256"])
    return program, ids, snapshots, lookup


def cheap_rms(row, weight, epsilon, lookup, runtime):
    return {"output_bits": list(row), "mean_bits": int(row[0]) << 16, "denominator_bits": 0x3F800000, "rsqrt_bits": 0x3F800000}


def cheap_rows(left, weights, mode, profile, workers):
    if left.ndim != 2 or weights.ndim != 2 or left.shape[1] != weights.shape[1]:
        raise AssertionError("Wrong per-node operand geometry")
    return np.repeat(left[:, :1], weights.shape[0], axis=1).copy()


def cheap_softmax(row, lookup, runtime):
    return {"output_bf16_bits": [0x3C89] * 30, "output_f32_bits": [0x3D088889] * 30}


def execute_synthetic(program, ids, snapshots, lookup):
    with ExitStack() as stack:
        stack.enter_context(patch.object(first, "PROGRAM_SHA256", program["program_sha256"]))
        stack.enter_context(patch.object(first, "_code_sha", return_value="synthetic-code"))
        stack.enter_context(patch.object(first, "_rms_lookup_row", side_effect=cheap_rms))
        stack.enter_context(patch.object(first, "_project_rows", side_effect=cheap_rows))
        stack.enter_context(patch.object(first, "lookup_softmax_row", side_effect=cheap_softmax))
        stack.enter_context(patch.object(first.ff, "predict_post_feedforward", side_effect=AssertionError("old prediction reuse")))
        stack.enter_context(patch("bioprocess_runtime.gemma_attention_entry.execute_attention_entry", side_effect=AssertionError("old stage helper")))
        stack.enter_context(patch("bioprocess_runtime.gemma_mlp_product.predict_product", side_effect=AssertionError("old product helper")))
        stack.enter_context(patch.object(first, "bind_model_tensors", side_effect=AssertionError("pure engine cannot bind model")))
        return first.execute_first_layer(program, ids, snapshots, lookup, profiles(), {"synthetic": True}, 1)


def geometry(shape, dtype):
    stride, strides = 1, []
    for dimension in reversed(shape):
        strides.insert(0, stride)
        stride *= dimension
    return {"shape": shape, "dtype": dtype, "strides": strides, "device": "cuda:0", "alignment_mod16": 0}


def native_pair(execution, coverage):
    code = capture._code_sha()
    common = {"checks": {name: True for name in ("token_commitment_unchanged", "stopped_at_decoder_tuple", "decoder_tuple_length_one", "no_later_layer", "no_final_norm", "no_lm_head", "all_needed_states_present", "code_unchanged", "runtime_unchanged")},
              "runtime_before": {"synthetic": True}, "runtime_after": {"synthetic": True}, "code_before": code, "code_after": code}
    traced = copy.deepcopy(common)
    traced["checks"].update({name: True for name in ("original_operations_once", "all_operand_links_unchanged", "rms_scalar_stages_complete", "softmax_f32_complete")})
    traced.update(state_bits=copy.deepcopy(execution["state_bits"]), scalar_stages=copy.deepcopy(execution["scalar_stages"]), softmax_f32_bits=copy.deepcopy(execution["softmax_f32_bits"]),
                  geometry={name: geometry(item["shape"], item["dtype"]) for name, item in coverage["states"].items()}, kernels={"synthetic-role": ["synthetic_kernel"]})
    for name, stages in traced["scalar_stages"].items():
        shape = coverage["states"][name]["shape"]
        stages["mean_input_metadata"] = {"input_shape": shape, "input_dtype": "torch.float32", "input_strides": geometry(shape, "torch.float32")["strides"], "axes": [-1], "keepdim": True, "alignment_mod16": 0}
    plain = copy.deepcopy(common)
    plain["checks"]["plain_minimal_control"] = True
    plain.update(state_bits={"hidden.1": copy.deepcopy(execution["state_bits"]["hidden.1"])}, geometry={"hidden.1": geometry([1, 30, 640], "torch.bfloat16")})
    return {"traced": traced, "untraced": plain}


class FirstLayerIRTests(unittest.TestCase):
    def test_recorded_connected_result_and_qualification_limits(self):
        load = lambda name: json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))
        plan = load("gemma3_270m_first_layer_plan.json")
        summary = load("gemma3_270m_first_layer_summary.json")
        self.assertEqual(plan["plan_sha256"], _sha({key: value for key, value in plan.items() if key != "plan_sha256"}))
        self.assertEqual(summary["summary_sha256"], _sha({key: value for key, value in summary.items() if key != "summary_sha256"}))
        self.assertEqual(summary["plan_sha256"], plan["plan_sha256"])
        self.assertEqual(summary["trace_root"], plan["trace_root"])
        self.assertEqual(summary["coverage"]["instruction_count"], 34)
        self.assertEqual(summary["coverage"]["state_count"], 36)
        self.assertEqual(summary["coverage"]["rms_scalar_positions"], 810)
        self.assertEqual(summary["coverage"]["softmax_fp32_positions"], 3600)
        self.assertEqual(summary["coverage"]["excluded_prefix_nodes"], first.EXCLUDED)
        self.assertTrue(summary["first_layer_matches"])
        self.assertTrue(summary["connected_first_layer_independently_recomputed"])
        self.assertTrue(summary["empirical_primitive_data_reused"])
        self.assertTrue(all(summary["checks"].values()))
        self.assertEqual(set(summary["mismatch_counts"].values()), {0})
        self.assertIsNone(summary["first_divergence"])
        for field in first.FALSE_FLAGS:
            self.assertFalse(summary[field], field)

    def test_real_serialized_source_and_kernel_preflight_without_prediction(self):
        from bioprocess_runtime.cli import build_parser, _load_post_feedforward_sources
        args = build_parser().parse_args(["gemma-first-layer-plan", "results/gemma3_270m_execution_ir.json", "--bundle", "unused", "--output", "unused-plan"])
        for name, value in vars(args).items():
            if isinstance(value, Path) and not value.is_absolute():
                setattr(args, name, ROOT / value)
        required = [value for name, value in vars(args).items() if isinstance(value, Path) and name not in ("bundle", "plan", "output") and "audit" not in name]
        if any(not path.exists() for path in required):
            self.skipTest("Ignored first-layer source evidence is unavailable")
        load = lambda path: json.loads(path.read_text(encoding="utf-8"))
        sources = first.FirstLayerSources(_load_post_feedforward_sources(args), load(args.post_feedforward_plan), load(args.post_feedforward_bundle), load(args.post_feedforward_report))
        with patch.object(first, "execute_first_layer", side_effect=AssertionError("Preflight must not predict")):
            ids, primitive_providers, arithmetic_profiles = sources.validate()
            symbols = first._source_kernel_expectations(sources)
        self.assertEqual(len(ids[0]), 30)
        self.assertEqual(type(primitive_providers), first.Providers)
        self.assertEqual(arithmetic_profiles["output"]["id"], "k128:bfloat16_rne:sequential_float32_rne")
        self.assertEqual(arithmetic_profiles["down"]["id"], "k192:bfloat16_rne:sequential_float32_rne")
        self.assertEqual(len(symbols), 10)
        self.assertTrue(all(names for names in symbols.values()))

    def test_exact_dependency_cone_and_exclusions(self):
        program = real_program()
        nodes = first.first_layer_instructions(program)
        self.assertEqual(len(nodes), 34)
        self.assertEqual(sum(len(node["outputs"]) for node in nodes), 36)
        self.assertEqual([node["id"] for node in nodes], [f"i{index:04d}" for index in range(36) if index not in (3, 5)])
        coverage = first._coverage(program)
        self.assertEqual(coverage["excluded_prefix_nodes"], first.EXCLUDED)
        self.assertFalse(coverage["all_36_prefix_instructions_claimed"])
        self.assertEqual(set(coverage["states"]), set(capture.STATE_SPECS))
        self.assertEqual(len(first._native_kernel_roles(program)), 22)
        self.assertEqual(sum(np.prod(first._shape(program, node["outputs"][0])[:-1]) * 3 for node in nodes if node["opcode"] == "RMS_NORM"), 810)

    def test_unknown_attributes_opcode_shape_layer_parameters_and_order_fail(self):
        mutations = [lambda p: p["instructions"][20]["attributes"].update(scalar=0.5),
                     lambda p: p["instructions"][8].update(opcode="MAGIC"),
                     lambda p: p["instructions"][8].update(layer=1),
                     lambda p: p["instructions"][8].update(parameter_refs=["model.layers.0.self_attn.k_proj.weight"]),
                     lambda p: p["instructions"][8]["attributes"].update(bias=True),
                     lambda p: p["tensors"]["hidden.1"].update(shape=[1, 29, 640]),
                     lambda p: p["instructions"].__setitem__(slice(8, 10), p["instructions"][8:10][::-1]),
                     lambda p: p["instructions"].pop(8),
                     lambda p: p["instructions"].insert(9, copy.deepcopy(p["instructions"][8])),
                     lambda p: p["external_inputs"].append("old_hidden")]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                program = real_program()
                mutation(program)
                program["instructions"] = [_seal({key: value for key, value in node.items() if key != "instruction_sha256"}, "instruction_sha256") for node in program["instructions"]]
                program = _seal({key: value for key, value in program.items() if key != "program_sha256"}, "program_sha256")
                with self.assertRaises(ValueError):
                    first.first_layer_instructions(program)

    def test_root_ids_are_strict_int64(self):
        program = real_program()
        for ids in ([[True] * 30], [[-1] * 30], [[262144] * 30], [[0] * 29], np.zeros((1, 30), dtype=np.uint16)):
            with self.subTest(ids=type(ids)), self.assertRaises(ValueError):
                first._tokens(program, ids)

    def test_descriptor_distinguishes_int_bf16_fp32(self):
        for dtype in (np.int64, np.uint16, np.uint32):
            value = np.asarray([[1, 2]], dtype=dtype)
            result = first._descriptor(value)
            self.assertEqual(result["dtype"], {np.int64: "torch.int64", np.uint16: "torch.bfloat16", np.uint32: "torch.float32"}[dtype])
            with self.assertRaises(ValueError):
                first._array(value, [1, 2], "torch.float32" if dtype != np.uint32 else "torch.bfloat16")
        self.assertNotEqual(first._descriptor(np.zeros(1, dtype=np.uint16))["sha256"], first._descriptor(np.zeros(1, dtype=np.uint32))["sha256"])

    def test_descriptors_match_existing_torch_hashes(self):
        import torch
        from bioprocess_runtime.operational_semantics import tensor_descriptor
        for bits, dtype in ((np.asarray(0x3F80, dtype=np.uint16), torch.bfloat16), (np.arange(30, dtype=np.int64)[None], torch.int64), (np.zeros(128, dtype=np.uint32), torch.float32)):
            self.assertEqual(first._descriptor(bits), tensor_descriptor(torch.from_numpy(bits).view(dtype)))

    def test_profile_fields_remain_exact(self):
        first._profiles(profiles())
        for key in ("output", "down"):
            changed = profiles()
            changed[key]["partitions"][-1][1] -= 1
            with self.assertRaises(ValueError):
                first._profiles(changed)
        self.assertEqual(first.DENSE_SPLIT_PROFILE, (64, "bfloat16_rne", "sequential_float32_rne"))
        self.assertEqual(first._dot([0] * 1024, [0] * 1024, "output", profiles()["output"]), 0)
        self.assertEqual(first._dot([0] * 2048, [0] * 2048, "down", profiles()["down"]), 0)
        self.assertEqual(first._dot([0] * 640, [0] * 640, "key_value", profiles()["key_value"]), 0)

    def test_provider_objects_bound_methods_and_table_hashes(self):
        provider = providers()
        provider.validate({"synthetic": True})
        changed = first.Providers(lambda value: value, provider.exp, provider.gelu, provider.rotary_cosine, provider.rotary_sine, "frequency", provider.rotary_runtime, provider.rotary_evidence)
        with self.assertRaisesRegex(ValueError, "registered"):
            changed.validate({"synthetic": True})
        with patch.object(provider.exp, "predict_bits", lambda *args: 0):
            with self.assertRaisesRegex(ValueError, "registered"):
                provider.validate({"synthetic": True})
        provider.gelu._table[0] ^= 1
        with self.assertRaisesRegex(ValueError, "table changed"):
            provider.validate({"synthetic": True})
        with self.assertRaisesRegex(ValueError, "runtime"):
            providers().validate({"wrong": True})

    def test_worker_small_exact_arithmetic_and_order(self):
        left = np.full((2, 16), 0x3F80, dtype=np.uint16)
        weights = np.full((3, 16), 0x3F80, dtype=np.uint16)
        actual = first._project_rows(left, weights, "serial", "operand_alignment_v1", 1)
        self.assertTrue(np.all(actual == 0x4180))
        with patch.object(first, "_code_sha", return_value="pinned"):
            first._worker_init(weights.tolist(), "serial", "operand_alignment_v1", "pinned", _sha(weights.tolist()))
            self.assertEqual(first._worker_row((7, left[0].tolist())), (7, [0x4180] * 3))
            with self.assertRaises(ValueError):
                first._worker_init(weights.tolist(), "serial", None, "stale", _sha(weights.tolist()))
        for workers in (0, 5, True):
            with self.assertRaises(ValueError):
                first._project_rows(left, weights, "serial", None, workers)

    def test_pure_boundary_has_no_source_model_or_bundle_arguments(self):
        self.assertEqual(list(inspect.signature(first.execute_first_layer).parameters), ["program", "token_ids", "parameter_snapshots", "providers", "profiles", "runtime", "workers"])
        text = inspect.getsource(first.execute_first_layer)
        for forbidden in ("entry_bundle", "post_bundle", "predict_product", "predict_output_projection", "model.forward", "sources."):
            self.assertNotIn(forbidden, text)

    def test_full_source_guard_detects_unsealed_nested_payload_change(self):
        from dataclasses import dataclass

        @dataclass
        class Evidence:
            report: dict

        source = Evidence({"report_sha256": "unchanged", "observations": [{"bits": [1]}]})
        before = first._source_tree_commitments(source)
        source.report["observations"][0]["bits"][0] = 2
        self.assertNotEqual(before, first._source_tree_commitments(source))

    def test_full_previous_lineage_gate_is_invoked_once_and_failure_stops(self):
        source = first.FirstLayerSources(SimpleNamespace(program=real_program()), {"plan_sha256": first.POST_FEEDFORWARD_PLAN_SHA256}, {}, {})
        with patch.object(first.ff, "verify_post", return_value={"valid": False}) as verify:
            with self.assertRaisesRegex(ValueError, "full previous lineage"):
                source.validate()
            verify.assert_called_once()

    def test_source_scope_rejected_before_old_lineage_recompute(self):
        source = first.FirstLayerSources(SimpleNamespace(program=real_program()), {"plan_sha256": "old"}, {}, {})
        with patch.object(first.ff, "verify_post", side_effect=AssertionError("must reject old scope first")):
            with self.assertRaisesRegex(ValueError, "latest"):
                source.validate()


class ConnectedSyntheticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.program, cls.ids, cls.snapshots, cls.lookup = synthetic_context()
        cls.execution = execute_synthetic(cls.program, cls.ids, cls.snapshots, cls.lookup)
        with patch.object(first, "PROGRAM_SHA256", cls.program["program_sha256"]):
            cls.coverage = first._coverage(cls.program)

    def validate(self, execution=None, snapshots=None, ids=None):
        with patch.object(first, "PROGRAM_SHA256", self.program["program_sha256"]):
            first.check_execution(self.program, ids or self.ids, snapshots or self.snapshots, self.lookup, profiles(), {"synthetic": True}, execution or self.execution)

    def test_all_nodes_fresh_and_ledger_covers_all_outputs(self):
        self.validate()
        self.assertEqual(len(self.execution["records"]), 34)
        self.assertEqual(len(self.execution["state_bits"]), 36)
        self.assertEqual(self.execution["state_bits"]["position_ids"], [list(range(30))])
        self.assertEqual(sum(len(value[key]) for value in self.execution["scalar_stages"].values() for key in first.STAGES), 810)
        self.assertEqual(np.asarray(self.execution["softmax_f32_bits"]).size, 3600)
        for record in self.execution["records"]:
            self.assertTrue(record["payload"]["inputs"])
            for value in record["payload"]["inputs"].values():
                self.assertIn("producer_record_hash", value)

    def test_altered_root_token_rows_propagate_to_hidden1(self):
        ids = copy.deepcopy(self.ids)
        ids[0][0] = 99
        snapshots = dict(self.snapshots)
        embedding = copy.deepcopy(snapshots[first.EMBEDDING])
        embedding["token_indices"] = ids[0]
        embedding["bits"][0, :] = 0x3F80
        embedding["descriptor"] = first._descriptor(embedding["bits"])
        snapshots[first.EMBEDDING] = embedding
        result = execute_synthetic(self.program, ids, snapshots, self.lookup)
        self.assertNotEqual(result["state_bits"]["hidden.1"][0][0], self.execution["state_bits"]["hidden.1"][0][0])
        self.assertNotEqual(result["records"][-1]["record_hash"], self.execution["records"][-1]["record_hash"])
        self.assertEqual(result["state_bits"]["hidden.1"][0][1:], self.execution["state_bits"]["hidden.1"][0][1:])

    def test_embedding_wrong_indices_shape_dtype_hash_and_repeated_membership(self):
        for key, value in (("token_indices", list(reversed(self.ids[0]))), ("bits", np.zeros((29, 640), dtype=np.uint16)), ("bits", np.zeros((30, 640), dtype=np.uint32)), ("descriptor", {"sha256": "wrong"})):
            changed = dict(self.snapshots)
            changed[first.EMBEDDING] = {**changed[first.EMBEDDING], key: value}
            with self.subTest(key=key), patch.object(first, "PROGRAM_SHA256", self.program["program_sha256"]), self.assertRaises(ValueError):
                first.check_parameter_snapshots(self.program, self.ids, changed)
        changed = dict(self.snapshots)
        changed[first.EMBEDDING] = copy.deepcopy(changed[first.EMBEDDING])
        ids = copy.deepcopy(self.ids)
        ids[0][1] = ids[0][0]
        changed[first.EMBEDDING]["token_indices"] = ids[0]
        changed[first.EMBEDDING]["bits"][1, 0] ^= 1
        changed[first.EMBEDDING]["descriptor"] = first._descriptor(changed[first.EMBEDDING]["bits"])
        with patch.object(first, "PROGRAM_SHA256", self.program["program_sha256"]), self.assertRaisesRegex(ValueError, "Repeated token"):
            first.check_parameter_snapshots(self.program, ids, changed)

    def test_missing_state_missing_record_and_last_coordinate_tamper(self):
        for kind in ("state", "record", "last-coordinate", "aux"):
            changed = copy.deepcopy(self.execution)
            if kind == "state":
                del changed["state_bits"]["layer.0.key.repeated"]
            elif kind == "record":
                changed["records"].pop(5)
            elif kind == "aux":
                changed["softmax_f32_bits"][0][-1][-1][-1] ^= 1
            else:
                changed["state_bits"]["hidden.1"][0][-1][-1] ^= 1
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.validate(changed)

    def test_stale_input_producer_rejected_even_after_valid_rechaining(self):
        changed = copy.deepcopy(self.execution)
        changed["records"][5]["payload"]["inputs"]["hidden.0"]["producer_record_hash"] = "stale"
        records = []
        for record in changed["records"]:
            append_chain_record(records, record["payload"])
        changed["records"] = records
        with self.assertRaisesRegex(ValueError, "ledger"):
            self.validate(changed)

    def report_fixture(self):
        bundle = {"execution": self.execution}
        plan = _seal({"scope": first.SCOPE, "coverage": self.coverage, "runtime": {"synthetic": True}, "bundle_sha256": _sha(bundle),
                      "trace_root": self.execution["records"][-1]["record_hash"], "native_kernel_roles": ["synthetic-role"], "prior_role_kernel_symbols": {"synthetic-role": ["synthetic_kernel"]},
                      **{key: False for key in first.FALSE_FLAGS}}, "plan_sha256")
        pair = native_pair(self.execution, self.coverage)
        return plan, bundle, [copy.deepcopy(pair) for _ in range(3)]

    def test_native_full_coverage_pass_is_scoped_not_qualification(self):
        plan, bundle, observations = self.report_fixture()
        report = first.first_layer_report(plan, bundle, observations)
        self.assertTrue(report["first_layer_matches"])
        self.assertTrue(report["connected_first_layer_independently_recomputed"])
        self.assertFalse(any(report[key] for key in first.FALSE_FLAGS))
        self.assertEqual(sum(report["mismatch_counts"].values()), 0)

    def test_last_native_coordinate_and_aux_fp32_mismatch_not_hidden_by_bf16(self):
        for kind in ("last", "softmax", "rms", "plain"):
            plan, bundle, observations = self.report_fixture()
            traced = observations[2]["traced"]
            if kind == "last":
                traced["state_bits"]["hidden.1"][0][-1][-1] ^= 1
            elif kind == "softmax":
                traced["softmax_f32_bits"][0][-1][-1][-1] ^= 1
            elif kind == "rms":
                traced["scalar_stages"]["layer.0.mlp.post_normalized"]["rsqrt_bits"][-1] ^= 1
            else:
                observations[2]["untraced"]["state_bits"]["hidden.1"][0][-1][-1] ^= 1
            with self.subTest(kind=kind):
                report = first.first_layer_report(plan, bundle, observations)
                self.assertFalse(report["first_layer_matches"])
                self.assertFalse(report["connected_first_layer_independently_recomputed"])
                self.assertEqual(sum(report["mismatch_counts"].values()), 1)
                self.assertEqual(report["first_divergence"]["repetition"], 2)
                if kind in ("softmax", "rms"):
                    self.assertEqual(report["mismatch_counts"]["hidden.1"], 0)

    def test_minimal_untraced_control_and_native_geometry(self):
        plan, bundle, observations = self.report_fixture()
        observations[0]["untraced"]["scalar_stages"] = {"forged": True}
        with self.assertRaisesRegex(ValueError, "Plain control"):
            first.first_layer_report(plan, bundle, observations)
        plan, bundle, observations = self.report_fixture()
        observations[0]["traced"]["geometry"]["layer.0.query.heads"]["strides"] = [30720, 256, 1024, 1]
        self.assertTrue(first.first_layer_report(plan, bundle, observations)["first_layer_matches"])
        observations[0]["traced"]["geometry"]["position_ids"]["dtype"] = "torch.bfloat16"
        self.assertFalse(first.first_layer_report(plan, bundle, observations)["first_layer_matches"])

    def test_missing_kernel_role_unstable_symbols_false_checks_and_runtime_fail(self):
        for kind in ("missing-role", "symbols", "checks", "runtime", "code"):
            plan, bundle, observations = self.report_fixture()
            traced = observations[1]["traced"]
            if kind == "missing-role":
                traced["kernels"] = {}
            elif kind == "symbols":
                traced["kernels"]["synthetic-role"] = ["another_kernel"]
            elif kind == "checks":
                traced["checks"]["no_later_layer"] = False
            elif kind == "runtime":
                traced["runtime_after"] = {"synthetic": False}
            else:
                traced["code_after"] = "stale"
            with self.subTest(kind=kind):
                report = first.first_layer_report(plan, bundle, observations)
                self.assertFalse(report["first_layer_matches"])
                self.assertFalse(report["connected_first_layer_independently_recomputed"])

    def test_default_verification_labels_integrity_and_recounts_tampered_report(self):
        plan, bundle, observations = self.report_fixture()
        report = first.first_layer_report(plan, bundle, observations)
        sources = SimpleNamespace(commitments=lambda: {}, validate=lambda: ())
        with patch.object(first, "_check_plan"), patch.object(first, "_code_sha", return_value="code"), patch.object(first, "execute_first_layer", side_effect=AssertionError("default verify cannot numerically reexecute")), patch.object(first, "_snapshot_model", side_effect=AssertionError("no default model binding")):
            result = first.verify_first_layer(sources, plan, bundle, report)
            self.assertTrue(result["valid"])
            self.assertFalse(result["connected_numerical_recomputation_performed"])
            self.assertFalse(result["embedding_membership_revalidated_with_model"])
            changed = {**report, "mismatch_counts": {"hidden.1": 999}}
            changed = _seal({key: value for key, value in changed.items() if key != "report_sha256"}, "report_sha256")
            self.assertFalse(first.verify_first_layer(sources, plan, bundle, changed)["valid"])

    def test_plan_scope_and_bundle_hash_tamper_fail(self):
        plan, bundle, observations = self.report_fixture()
        plan["full_first_layer_qualified"] = True
        plan = _seal({key: value for key, value in plan.items() if key != "plan_sha256"}, "plan_sha256")
        with self.assertRaises(ValueError):
            first.first_layer_report(plan, bundle, observations)
        plan, bundle, observations = self.report_fixture()
        with self.assertRaises(ValueError):
            first.first_layer_report(plan, {**bundle, "old_predictions": {}}, observations)


class FirstLayerCLITests(unittest.TestCase):
    def test_shared_parser_inherits_all_source_flags_without_collisions(self):
        from bioprocess_runtime.cli import build_parser, command_first_layer, command_post_feedforward
        parser = build_parser()
        for operation in ("plan", "run", "verify"):
            tail = ["--report", "report.json"] if operation == "verify" else ["--output", "out.json"]
            args = parser.parse_args(["gemma-first-layer-" + operation, "program.json", *tail])
            self.assertIs(args.handler, command_first_layer)
            self.assertEqual(args.workers, 4)
            self.assertEqual(args.plan.name, "gemma3_270m_first_layer_plan.json")
            self.assertEqual(args.bundle.name, "gemma3_270m_first_layer_predictions.json")
            self.assertEqual(args.dense_plan.name, "gemma3_270m_k128_dense_plan.json")
            self.assertEqual(args.k2048_dense_plan.name, "gemma3_270m_k2048_dense_plan.json")
            self.assertEqual(args.model_down_plan.name, "gemma3_270m_mlp_down_plan_v2.json")
            self.assertEqual(args.prefix_probe_plan.name, "gemma3_270m_k1024_probes_plan.json")
            self.assertEqual(args.probe_plan.name, "gemma3_270m_k2048_probes_plan.json")
            self.assertEqual(args.post_feedforward_plan.name, "gemma3_270m_post_feedforward_plan.json")
        args = parser.parse_args(["gemma-post-feedforward-verify", "program.json", "--report", "report.json"])
        self.assertIs(args.handler, command_post_feedforward)
        self.assertEqual(args.bundle.name, "gemma3_270m_post_feedforward_predictions.json")

    def test_default_cli_verify_does_not_load_model(self):
        from bioprocess_runtime.cli import build_parser, command_first_layer
        from transformers import AutoModelForCausalLM
        args = build_parser().parse_args(["gemma-first-layer-verify", "program.json", "--report", "report.json"])
        with patch("bioprocess_runtime.cli._load_post_feedforward_sources", return_value=object()), patch.object(Path, "read_text", return_value="{}"), patch.object(first, "verify_first_layer", return_value={"valid": True}), patch.object(AutoModelForCausalLM, "from_pretrained", side_effect=AssertionError("integrity verify must not load a checkpoint")), patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(command_first_layer(args), 0)

    def test_source_path_collision_rejected_before_loading(self):
        from bioprocess_runtime.cli import build_parser, command_first_layer
        args = build_parser().parse_args(["gemma-first-layer-plan", "program.json", "--output", "same.json", "--bundle", "same.json"])
        with patch("bioprocess_runtime.cli._load_post_feedforward_sources", side_effect=AssertionError("must guard writes first")):
            with self.assertRaisesRegex(ValueError, "new, distinct"):
                command_first_layer(args)

    def test_reexecute_runs_all_fresh_predictions_before_native_and_never_skips(self):
        calls = []
        plan, bundle, report = {"plan": 1}, {"bundle": 1}, {"first_layer_matches": True}
        def build(*args):
            calls.append("fresh_snapshots_and_34_nodes")
            return plan, bundle
        def acquire(*args):
            calls.append("six_native_forwards")
            return report
        with patch.object(first, "build_first_layer_plan", side_effect=build), patch.object(first, "acquire_first_layer", side_effect=acquire):
            result = first.reexecute_first_layer(object(), object(), plan, bundle, report, 1)
            self.assertTrue(result["valid"])
            self.assertEqual(calls, ["fresh_snapshots_and_34_nodes", "six_native_forwards"])
            calls.clear()
            result = first.reexecute_first_layer(object(), object(), plan, {"bundle": "tampered"}, report, 1)
            self.assertFalse(result["valid"])
            self.assertEqual(calls, ["fresh_snapshots_and_34_nodes"])

    def test_public_source_token_case_cannot_be_changed(self):
        sources = SimpleNamespace(commitments=lambda: {}, validate=lambda: ([[0] * 30], None, None))
        with patch.object(first, "_code_sha", return_value="code"), patch.object(first, "_snapshot_model", side_effect=AssertionError("must reject input scope first")):
            with self.assertRaisesRegex(ValueError, "source token"):
                first.build_first_layer_plan(sources, object(), 1, [[1] * 30])


if __name__ == "__main__":
    unittest.main()
