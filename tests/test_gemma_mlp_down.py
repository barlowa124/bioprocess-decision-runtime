from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from bioprocess_runtime import gemma_mlp_down as down
from bioprocess_runtime.gemma_float_semantics import decode_finite_bfloat16, encode_bfloat16_rne
from bioprocess_runtime.gemma_rotary_slice import _seal, _sha
from bioprocess_runtime.gemma_wmma_candidate import _operand_aligned_accumulator, _merge_split_partials

ROOT = Path(__file__).resolve().parent.parent
HAS_GEMMA = importlib.util.find_spec("torch") and importlib.util.find_spec("transformers")


try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
except ModuleNotFoundError:
    raise unittest.SkipTest("requires .[gemma] extras")


def archived_v1():
    path = ROOT / "artifacts/gemma_mlp_down_v1_source.py"
    if not path.exists():
        raise unittest.SkipTest("Ignored v1 source snapshot is unavailable")
    name = "bioprocess_runtime._gemma_mlp_down_v1_source"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def candidate():
    return json.loads(down.SUPPORTED_CANDIDATE_JSON)


def program():
    return json.loads((ROOT / "results/gemma3_270m_execution_ir.json").read_text(encoding="utf-8"))


def geometry(shape, strides):
    return {"shape": shape, "strides": strides, "dtype": "torch.bfloat16", "alignment_mod16": 0, "device": "cuda:0"}


def observation(inputs, weights, actual):
    return {"order": list(down.ORDER), "postnorm_executed": False, "input_bits": inputs.tolist(), "output_bits": actual.tolist(),
            "weight": down._descriptor(weights), "input_geometry": geometry([1, 30, 2048], [61440, 2048, 1]),
            "weight_geometry": geometry([640, 2048], [2048, 1]), "output_geometry": geometry([1, 30, 640], [19200, 640, 1]),
            "linear_geometry": {"input": geometry([1, 30, 2048], [61440, 2048, 1]), "weight": geometry([640, 2048], [2048, 1]), "output": geometry([1, 30, 640], [19200, 640, 1])},
            "weight_pointer_matches": True, "linear_input_matches_hook": True, "linear_output_matches_hook": True,
            "linear_input": down._descriptor(inputs), "linear_weight": down._descriptor(weights), "linear_output": down._descriptor(actual), "kernel_names": ["synthetic"]}


class MlpDownTests(unittest.TestCase):
    def test_v1_snapshot_source_commitment(self):
        old = archived_v1()
        plan = json.loads((ROOT / "results/gemma3_270m_mlp_down_plan.json").read_text(encoding="utf-8"))
        down._check_hash(plan, "plan_sha256")
        self.assertEqual(plan["plan_sha256"], "50186e55487e6ad7b436b68a6b90a148866068e83e10552f83409ffee868cf81")
        snapshot = Path(old.__file__).read_bytes()
        self.assertEqual(old.SOURCE_SHA256, "2793e7dbc05b2a88f4297674bdffec4d4213cd1fd19e138d32ad66720ba45d26")
        self.assertEqual(old.SOURCE_SHA256, hashlib.sha256(snapshot).hexdigest())
        self.assertEqual(old._code_sha(), plan["code_sha256"])
        self.assertEqual(_sha({"module": hashlib.sha256(snapshot).hexdigest(), "product": down.product_code_sha(), "controlled_probe": down.probe_code_sha()}), plan["code_sha256"])
        if down._code_sha() == plan["code_sha256"]:
            self.assertEqual(snapshot, Path(down.__file__).read_bytes())
        self.assertEqual(inspect.getsource(old.down_dot), inspect.getsource(down.down_dot))
        print(json.dumps({"v1_snapshot_source_sha256": old.SOURCE_SHA256, "current_module_source_sha256": down.SOURCE_SHA256}, sort_keys=True))

    def test_candidate_arithmetic_unchanged_contiguous_partitions(self):
        selected = candidate()
        self.assertEqual(selected["partitions"], [[start, min(start + 192, 2048)] for start in range(0, 2048, 192)])
        left, right = [0x3F81] * 2048, [0] * 2048
        for position, bits in ((0, 0x4001), (191, 0x4F80), (192, 0xCF80), (2047, 0xBF81)):
            right[position] = bits
        partials = [_operand_aligned_accumulator(left[start:stop], right[start:stop]) for start, stop in selected["partitions"]]
        rounded = [decode_finite_bfloat16(encode_bfloat16_rne(value))[0] for value in partials]
        self.assertEqual(down.down_dot(left, right, selected), _merge_split_partials(rounded, "sequential_float32_rne"))
        with patch.object(down, "_operand_aligned_accumulator", wraps=_operand_aligned_accumulator) as accumulator:
            down.down_dot(left, right, selected)
            self.assertEqual([len(call.args[0]) for call in accumulator.call_args_list], [192] * 10 + [128])
        for key, value in (("merge", "exact"), ("partial_format", "float32"), ("partitions", [[0, 2048]])):
            with self.subTest(key=key), self.assertRaises(ValueError):
                down.down_dot(left, right, {**selected, key: value})
        with self.assertRaises(ValueError):
            down.down_dot(left[:-1], right, selected)

    def test_workers_consistent_on_small_row_and_column_counts(self):
        inputs = np.full((1, 2, 2048), 0x3F80, dtype=np.uint16)
        weights = np.zeros((3, 2048), dtype=np.uint16)
        weights[0, 191], weights[1, 192], weights[2, 2047] = 0x3F80, 0x4000, 0xBF80
        with patch.multiple(down, INPUT_SHAPE=(1, 2, 2048), WEIGHT_SHAPE=(3, 2048), OUTPUT_SHAPE=(1, 2, 3)):
            serial = down.project_down_bits(inputs, weights, candidate(), 1)
            self.assertEqual(serial.tolist(), [[[0x3F80, 0x4000, 0xBF80]] * 2])
            for workers in (2, 3, 4):
                self.assertTrue(np.array_equal(serial, down.project_down_bits(inputs, weights, candidate(), workers)))
            for workers in (0, 5, True):
                with self.assertRaises(ValueError):
                    down.project_down_bits(inputs, weights, candidate(), workers)
        with self.assertRaises(ValueError):
            down.project_down_bits(inputs, weights, candidate())
        with self.assertRaises(ValueError):
            down._worker_init(weights.tolist(), candidate(), "changed")

    def test_ir_shape_layer_bias_and_source_guards(self):
        source = program()
        self.assertEqual(down.down_instructions(source)[0]["id"], "i0033")
        for field, value in (("layer", 1), ("inputs", ["layer.0.mlp.up"]), ("attributes", {"bias": "unexpected"}), ("parameter_refs", ["other"]), ("opcode", "MUL")):
            changed = copy.deepcopy(source)
            node = next(item for item in changed["instructions"] if item["outputs"] == ["layer.0.mlp.down"])
            node[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                down.down_instructions(changed)
        for name in ("layer.0.mlp.product", "layer.0.mlp.down"):
            changed = copy.deepcopy(source)
            changed["tensors"][name]["shape"][-1] -= 1
            with self.assertRaises(ValueError):
                down.down_instructions(changed)
        changed = copy.deepcopy(source)
        changed["parameter_commitments"][down.WEIGHT]["shape"] = [2048, 640]
        with self.assertRaises(ValueError):
            down.down_instructions(changed)
        with patch.object(down.Path, "read_bytes", return_value=b"changed"):
            with self.assertRaises(ValueError):
                down._code_sha()

    def test_source_requires_full_product_and_matching_probe_summary(self):
        values = np.zeros(down.INPUT_SHAPE, dtype=np.uint16)
        prefix = SimpleNamespace(entry=SimpleNamespace(post=SimpleNamespace(program=program())))
        probe = {"input_shape": [1, 30, 2048], "weight_shape": [640, 2048], "matrix_value_count": 19200, "candidates": [candidate()]}
        sources = down.DownSources(prefix, {}, {"product": values.tolist()}, {}, {}, probe, {}, {})
        success = {"valid": True, "survivors_supported_in_declared_scope": True, "unique_survivor_in_frozen_grid": True, "surviving_candidates": [down.SUPPORTED_ID]}
        with patch.object(down, "verify_product", return_value={"valid": True, "activation_and_product_match": True}) as product_check, patch.object(down, "product_summary", return_value={"computed": "summary"}), patch.object(down, "verify_probes", return_value=success) as probe_check:
            actual, selected = sources.material()
            self.assertEqual(selected, candidate())
            self.assertTrue(np.array_equal(values, actual))
            self.assertEqual(probe_check.call_args.args[1], {"computed": "summary"})
            product_check.assert_called_once_with(prefix, {}, sources.product_bundle, {})
            for field, value in (("valid", False), ("unique_survivor_in_frozen_grid", False), ("survivors_supported_in_declared_scope", False), ("surviving_candidates", ["other"])):
                probe_check.return_value = {**success, field: value}
                with self.assertRaises(ValueError):
                    sources.material()
            probe_check.return_value = success
            probe["input_shape"] = [1, 30, 1024]
            with self.assertRaises(ValueError):
                sources.material()
            product_check.return_value = {"valid": False}
            probe_check.reset_mock()
            with self.assertRaises(ValueError):
                sources.material()
            probe_check.assert_not_called()

    def test_reversed_kernel_symbol_order_is_not_a_launch_order_failure(self):
        inputs = np.zeros(down.INPUT_SHAPE, dtype=np.uint16)
        weights = np.zeros(down.WEIGHT_SHAPE, dtype=np.uint16)
        predicted = np.zeros(down.OUTPUT_SHAPE, dtype=np.uint16)
        names = ["ampere_bf16_s16816", "_ZN_splitKreduce_kernel"]
        self.assertEqual(sorted(names), list(reversed(names)))
        plan = {"plan_sha256": "a" * 64, "runtime": {}, "kernel_provenance": "distinct-symbol-set-only; launch order not established", "expected_kernel_names": names}
        record = observation(inputs, weights, predicted)
        record["kernel_names"] = sorted(names)
        pairs = [{"traced": record, "untraced": record}] * 3
        old_report = archived_v1().down_report(plan, inputs, weights, predicted, pairs, {})
        self.assertFalse(old_report["down_matches"])
        self.assertEqual(old_report["mismatch_counts"], [0, 0, 0])
        self.assertEqual(old_report["first_divergence"], "kernel_names_match")
        self.assertTrue(all(all(value for key, value in item.items() if key != "kernel_names_match") for item in old_report["checks"]))
        report = down.down_report(plan, inputs, weights, predicted, pairs, {})
        self.assertTrue(report["down_matches"])
        self.assertTrue(all(all(item.values()) for item in report["checks"]))
        self.assertEqual(report["kernel_provenance"], "distinct-symbol-set-only; launch order not established")
        self.assertTrue(report["metadata_correction_development_regression"])
        self.assertTrue(report["down_stage_previously_acquired_by_prefix_workflow"])
        self.assertFalse(report["fresh_down_holdout"])
        for changed_names in ([names[0]], [names[0], "replacement"]):
            bad = {**record, "kernel_names": changed_names}
            failed = down.down_report(plan, inputs, weights, predicted, [{"traced": bad, "untraced": record}] * 3, {})
            self.assertFalse(failed["down_matches"])
            self.assertEqual(failed["first_divergence"], "kernel_names_match")
        for malformed in ([], [names[0], names[0]], [names[0], ""], [names[0], None], [names[0], 1], names[0], tuple(names), None):
            with self.subTest(malformed=malformed):
                bad = {**record, "kernel_names": malformed}
                with self.assertRaises(ValueError):
                    down.down_report(plan, inputs, weights, predicted, [{"traced": bad, "untraced": record}] * 3, {})
                with self.assertRaises(ValueError):
                    down.down_report({**plan, "expected_kernel_names": malformed}, inputs, weights, predicted, pairs, {})
        with self.assertRaises(KeyError):
            down.down_report(plan, inputs, weights, predicted, [{"traced": {key: value for key, value in record.items() if key != "kernel_names"}, "untraced": record}] * 3, {})
        with self.assertRaises(ValueError):
            down.down_report({**plan, "kernel_provenance": "launch order established"}, inputs, weights, predicted, pairs, {})

    def test_last_coordinate_failures_preserved_and_geometry_lineage_checked(self):
        inputs = np.zeros(down.INPUT_SHAPE, dtype=np.uint16)
        weights = np.zeros(down.WEIGHT_SHAPE, dtype=np.uint16)
        predicted = np.zeros(down.OUTPUT_SHAPE, dtype=np.uint16)
        plan = {"plan_sha256": "a" * 64, "runtime": {}, "kernel_provenance": "distinct-symbol-set-only; launch order not established", "expected_kernel_names": ["synthetic"]}
        record = observation(inputs, weights, predicted)
        report = down.down_report(plan, inputs, weights, predicted, [{"traced": record, "untraced": record}] * 3, {})
        self.assertTrue(report["down_matches"])
        for flag in down.FALSE_FLAGS:
            self.assertIs(report[flag], False)
        actual = predicted.copy()
        actual[0, 29, 639] = 1
        record = observation(inputs, weights, actual)
        report = down.down_report(plan, inputs, weights, predicted, [{"traced": record, "untraced": record}] * 3, {})
        self.assertEqual(report["mismatch_counts"], [1, 1, 1])
        self.assertEqual(report["first_divergence"]["coordinate"], [0, 29, 639])
        self.assertFalse(report["down_matches"])
        for field, value in (("weight_pointer_matches", False), ("linear_output_matches_hook", False), ("postnorm_executed", True), ("kernel_names", ["wrong"]), ("order", down.ORDER + ["down_output"])):
            bad = {**record, field: value}
            report = down.down_report(plan, inputs, weights, actual, [{"traced": bad, "untraced": record}] * 3, {})
            self.assertFalse(report["down_matches"])
        bad = copy.deepcopy(record)
        bad["input_bits"][0][29][2047] = 1
        bad["input_geometry"]["strides"] = [1, 2, 3]
        report = down.down_report(plan, inputs, weights, actual, [{"traced": bad, "untraced": record}] * 3, {})
        self.assertFalse(report["checks"][0]["source_input_matches"])
        self.assertFalse(report["checks"][0]["boundary_geometry_matches"])
        report = down.down_report(plan, inputs, weights, actual, [{"traced": record, "untraced": record}] * 3, {"changed": True})
        self.assertFalse(report["runtime_matches_plan"])

    @unittest.skipUnless(HAS_GEMMA, "Gemma dependencies unavailable")
    def test_integrity_mode_scope_and_parameter_tamper(self):
        inputs = np.zeros(down.INPUT_SHAPE, dtype=np.uint16)
        weights = np.zeros(down.WEIGHT_SHAPE, dtype=np.uint16)
        predicted = np.zeros(down.OUTPUT_SHAPE, dtype=np.uint16)
        ir = program()
        ir["parameter_commitments"][down.WEIGHT]["sha256"] = down._descriptor(weights)["sha256"]
        sources = SimpleNamespace(program=ir, material=lambda: (inputs, candidate()), commitments=lambda: {}, product_plan={"runtime": {}, "model_binding": {}}, probe_plan={"expected_kernel_names": ["synthetic"]})
        bundle = {"weight_bits": weights.tolist(), "down_bits": predicted.tolist()}
        plan = _seal(down._plan_body(sources, inputs, candidate(), weights, predicted, bundle), "plan_sha256")
        record = observation(inputs, weights, predicted)
        report = down.down_report(plan, inputs, weights, predicted, [{"traced": record, "untraced": record}] * 3, {})
        with patch.object(down, "project_down_bits", side_effect=AssertionError("No prediction in integrity mode")):
            verified = down.verify_down(sources, plan, bundle, report)
            self.assertTrue(verified["valid"])
            self.assertFalse(verified["down_predictions_recomputed"])
            for flag in down.FALSE_FLAGS:
                promoted = _seal({**{key: value for key, value in plan.items() if key != "plan_sha256"}, flag: True}, "plan_sha256")
                self.assertFalse(down.verify_down(sources, promoted, bundle, report)["valid"])
            for key, value in (("kernel_provenance", "launch order established"), ("down_stage_previously_acquired_by_prefix_workflow", False), ("metadata_correction_development_regression", False)):
                promoted = _seal({**{name: item for name, item in plan.items() if name != "plan_sha256"}, key: value}, "plan_sha256")
                self.assertFalse(down.verify_down(sources, promoted, bundle, report)["valid"])
            changed = copy.deepcopy(report)
            changed["observations"][0]["traced"]["output_bits"][0][29][639] ^= 1
            self.assertFalse(down.verify_down(sources, plan, bundle, _seal({key: value for key, value in changed.items() if key != "report_sha256"}, "report_sha256"))["valid"])
            bundle["weight_bits"][639][2047] = 1
            changed_weights = down._state_array(bundle["weight_bits"], list(down.WEIGHT_SHAPE))
            resealed = _seal(down._plan_body(sources, inputs, candidate(), changed_weights, predicted, bundle), "plan_sha256")
            self.assertFalse(down.verify_down(sources, resealed, bundle, report)["valid"])

    @unittest.skipUnless(HAS_GEMMA, "Gemma dependencies unavailable")
    def test_capture_profiles_only_down_stops_before_postnorm_and_removes_hooks(self):
        import torch

        class Mlp(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = torch.nn.Linear(4, 8, bias=False, dtype=torch.bfloat16)
                self.up_proj = torch.nn.Linear(4, 8, bias=False, dtype=torch.bfloat16)
                self.act_fn = torch.nn.GELU()
                self.down_proj = torch.nn.Linear(8, 4, bias=False, dtype=torch.bfloat16)

            def forward(self, x):
                return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Module()
                self.layer.mlp = Mlp()
                self.layer.post_feedforward_layernorm = torch.nn.Identity()
                self.model = SimpleNamespace(layers=[self.layer])
                self.completed = False

            def forward(self, **kwargs):
                value = self.layer.mlp(torch.ones((1, 2, 4), dtype=torch.bfloat16))
                self.layer.post_feedforward_layernorm(value)
                self.completed = True

        model = Model()
        profiler = SimpleNamespace(events=lambda: [SimpleNamespace(name="synthetic", device_type=torch.autograd.DeviceType.CUDA)])
        with patch("torch.profiler.profile") as profile, patch("torch.cuda.synchronize"):
            profile.return_value.__enter__.return_value = profiler
            traced = down._capture_down(model, [[1, 2]], True)
            plain = down._capture_down(model, [[1, 2]], False)
            self.assertEqual(profile.call_count, 1)
            self.assertEqual(traced["output_bits"], plain["output_bits"])
            self.assertEqual(traced["order"], down.ORDER)
            self.assertTrue(traced["weight_pointer_matches"])
            self.assertTrue(traced["linear_input_matches_hook"])
            self.assertTrue(traced["linear_output_matches_hook"])
            self.assertFalse(model.completed)
        for module in model.modules():
            self.assertFalse(module._forward_hooks)
            self.assertFalse(module._forward_pre_hooks)
        with patch.object(model.layer.mlp.down_proj, "forward", side_effect=RuntimeError("synthetic failure")):
            with self.assertRaises(RuntimeError):
                down._capture_down(model, [[1, 2]], False)
        for module in model.modules():
            self.assertFalse(module._forward_hooks)
            self.assertFalse(module._forward_pre_hooks)

    def test_cli_inheritance_output_guards_and_old_defaults(self):
        from bioprocess_runtime.cli import build_parser, command_mlp_down, command_mlp_product
        parser = build_parser()
        for operation in ("plan", "run", "verify"):
            argv = ["gemma-mlp-down-" + operation, "program", "--bundle", "bundle"]
            argv += ["--report", "report", "--reexecute"] if operation == "verify" else ["--output", "output"]
            args = parser.parse_args(argv)
            self.assertIs(args.handler, command_mlp_down)
            self.assertEqual(args.workers, 4)
            self.assertEqual(args.plan.name, "gemma3_270m_mlp_down_plan.json")
            self.assertEqual(args.probe_plan.name, "gemma3_270m_k2048_probes_plan.json")
            self.assertEqual(args.prefix_probe_plan.name, "gemma3_270m_k1024_probes_plan.json")
            self.assertEqual(args.product_plan.name, "gemma3_270m_mlp_product_plan.json")
            self.assertEqual(args.entry_plan.name, "gemma3_270m_mlp_entry_plan.json")
        old = parser.parse_args(["gemma-mlp-product-plan", "program", "--bundle", "bundle", "--output", "output"])
        self.assertIs(old.handler, command_mlp_product)
        self.assertEqual(old.probe_plan.name, "gemma3_270m_k1024_probes_plan.json")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.json"
            path.write_text("preserve", encoding="utf-8")
            with self.assertRaises(ValueError):
                command_mlp_down(Namespace(operation="plan", output=path, bundle=path.with_name("new.json")))
            self.assertEqual(path.read_text(encoding="utf-8"), "preserve")
            new = path.with_name("new.json")
            with self.assertRaises(ValueError):
                command_mlp_down(Namespace(operation="plan", output=new, bundle=new))
            with self.assertRaises(ValueError):
                command_mlp_down(Namespace(operation="run", output=new, summary=new))

    @unittest.skipUnless(HAS_GEMMA, "Gemma dependencies unavailable")
    def test_cli_reexecute_rebuilds_before_acquisition_and_stops_on_prediction_change(self):
        from bioprocess_runtime.cli import command_mlp_down
        from transformers import AutoModelForCausalLM
        sources, original = object(), object()
        plan, bundle, report = {"plan": 1}, {"bundle": 1}, {"report": 1}
        args = Namespace(operation="verify", plan=Path("plan"), bundle=Path("bundle"), report=Path("report"), summary=None,
                         reexecute=True, workers=4, model_path=Path("unused-model"))
        payloads = {"plan": plan, "bundle": bundle, "report": report}
        events = []

        def predict(received_sources, model, workers):
            self.assertIs(received_sources, sources)
            self.assertIs(model, original)
            self.assertEqual(workers, 4)
            events.append("predict")
            return plan, bundle

        def acquire(received_sources, model, received_plan, received_bundle):
            self.assertIs(model, original)
            self.assertEqual(events, ["load", "predict"])
            events.append("acquire")
            return report

        with patch("bioprocess_runtime.cli._load_down_sources", return_value=sources), patch.object(Path, "read_text", autospec=True, side_effect=lambda path, **kwargs: json.dumps(payloads[str(path)])), patch("torch.cuda.is_available", return_value=True), patch.object(AutoModelForCausalLM, "from_pretrained") as load_model, patch.object(down, "verify_down", return_value={"valid": True}), patch.object(down, "build_down_plan", side_effect=predict) as build, patch.object(down, "acquire_down", side_effect=acquire) as capture, patch("builtins.print"):
            load_model.return_value.to.return_value.eval.side_effect = lambda: events.append("load") or original
            self.assertEqual(command_mlp_down(args), 0)
            self.assertEqual(events, ["load", "predict", "acquire"])
            events.clear()
            capture.reset_mock()
            build.side_effect = lambda *args: ({"changed": True}, bundle)
            self.assertEqual(command_mlp_down(args), 1)
            capture.assert_not_called()
        with patch("torch.nn.functional.linear"):
            with self.assertRaisesRegex(ValueError, "F.linear"):
                down._model_context(sources, original)

    def test_recorded_v2_scope_and_unchanged_predictions(self):
        load = lambda name: json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))
        original = load("gemma3_270m_mlp_down_plan.json")
        plan = load("gemma3_270m_mlp_down_plan_v2.json")
        summary = load("gemma3_270m_mlp_down_summary_v2.json")
        self.assertEqual(plan["bundle_sha256"], original["bundle_sha256"])
        self.assertEqual(plan["candidate"], original["candidate"])
        self.assertEqual(summary["summary_sha256"], _sha({key: value for key, value in summary.items() if key != "summary_sha256"}))
        self.assertEqual(summary["plan_sha256"], plan["plan_sha256"])
        self.assertEqual(summary["kernel_provenance"], down.KERNEL_PROVENANCE)
        self.assertEqual(summary["mismatch_counts"], [0, 0, 0])
        self.assertEqual(summary["value_count"], 19200)
        self.assertTrue(summary["down_matches"])
        self.assertTrue(summary["metadata_correction_development_regression"])
        self.assertTrue(all(all(check.values()) for check in summary["checks"]))
        for field in ("global_exactness_activation_allowed", "full_first_layer_qualified", "hardware_semantics_established", "post_feedforward_layernorm_executed", "prefix_independently_recomputed", "fresh_down_holdout"):
            self.assertFalse(summary[field], field)

    @unittest.skipUnless(HAS_GEMMA, "Gemma dependencies unavailable")
    def test_real_source_and_archived_v1_report_integrity_without_prediction(self):
        from bioprocess_runtime.cli import build_parser, _load_down_sources
        args = build_parser().parse_args(["gemma-mlp-down-plan", "results/gemma3_270m_execution_ir.json", "--bundle", "unused", "--output", "unused-plan"])
        for name, value in vars(args).items():
            if isinstance(value, Path) and not value.is_absolute():
                setattr(args, name, ROOT / value)
        required = [value for name, value in vars(args).items() if isinstance(value, Path) and name not in ("bundle", "output", "plan") and not name.endswith("audit_dir")]
        if any(not path.exists() for path in required):
            self.skipTest("Ignored source evidence is unavailable")
        sources = _load_down_sources(args)
        summary = down.product_summary(sources.product_plan, sources.product_report)
        checked = down.verify_probes(sources.probe_binding, summary, sources.probe_plan, sources.probe_bundle, sources.probe_report)
        self.assertTrue(checked["valid"])
        self.assertEqual(checked["surviving_candidates"], [down.SUPPORTED_ID])
        self.assertEqual(sources.probe_plan["plan_sha256"], "90c5793b9f40405a8beda303fee9b8fe92fdc08a5d7a0489f2eaa58f6e247866")
        self.assertEqual(sources.probe_plan["product_summary_sha256"], summary["summary_sha256"])
        self.assertEqual(sources.probe_plan["bundle_sha256"], _sha(sources.probe_bundle))
        self.assertEqual(down.down_instructions(sources.program)[0]["layer"], 0)
        old = archived_v1()
        paths = [ROOT / name for name in ("results/gemma3_270m_mlp_down_plan.json", "artifacts/gemma3_270m_mlp_down_predictions.json", "artifacts/gemma3_270m_mlp_down_report.json", "results/gemma3_270m_mlp_down_summary.json")]
        if any(not path.exists() for path in paths):
            self.skipTest("Ignored v1 down evidence is unavailable")
        before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
        plan, bundle, report, summary = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        self.assertEqual(plan["plan_sha256"], "50186e55487e6ad7b436b68a6b90a148866068e83e10552f83409ffee868cf81")
        self.assertEqual(old._code_sha(), plan["code_sha256"])
        with patch.object(old, "project_down_bits", side_effect=AssertionError("No down prediction in archive verification")), patch.object(down, "project_down_bits", side_effect=AssertionError("No down prediction in archive verification")):
            checked = old.verify_down(sources, plan, bundle, report)
        self.assertTrue(checked["valid"], checked)
        self.assertFalse(checked["down_matches"])
        self.assertFalse(checked["down_predictions_recomputed"])
        self.assertEqual(checked["mismatch_counts"], [0, 0, 0])
        self.assertEqual(report["first_divergence"], "kernel_names_match")
        self.assertTrue(all(all(value for key, value in item.items() if key != "kernel_names_match") for item in report["checks"]))
        self.assertEqual(old.down_summary(plan, report), summary)
        for pair in report["observations"]:
            self.assertNotEqual(pair["traced"]["kernel_names"], plan["expected_kernel_names"])
            self.assertEqual(sorted(pair["traced"]["kernel_names"]), sorted(plan["expected_kernel_names"]))
        inputs = down._state_array(sources.product_bundle["product"], list(down.INPUT_SHAPE))
        weights = down._state_array(bundle["weight_bits"], list(down.WEIGHT_SHAPE))
        predicted = down._state_array(bundle["down_bits"], list(down.OUTPUT_SHAPE))
        comparison_plan = {"plan_sha256": "in-memory-regression-only", "kernel_provenance": down.KERNEL_PROVENANCE,
                           "runtime": plan["runtime"], "expected_kernel_names": plan["expected_kernel_names"]}
        corrected = down.down_report(comparison_plan, inputs, weights, predicted, report["observations"], report["runtime"])
        self.assertTrue(corrected["down_matches"])
        self.assertEqual(corrected["mismatch_counts"], [0, 0, 0])
        self.assertEqual(before, [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths])


if __name__ == "__main__":
    unittest.main()
