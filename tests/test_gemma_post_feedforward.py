from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from bioprocess_runtime import gemma_post_feedforward as ff
from bioprocess_runtime.gemma_rotary_slice import _sha, _seal

ROOT = Path(__file__).resolve().parent.parent


def program():
    return json.loads((ROOT / "results/gemma3_270m_execution_ir.json").read_text(encoding="utf-8"))


def reseal(value, key):
    return _seal({name: item for name, item in value.items() if name != key}, key)


class SyntheticRoot:
    def predict_bits(self, bits, runtime):
        return 0x3F800000


def fixture():
    values = np.zeros(ff.SHAPE, dtype=np.uint16)
    weight = np.zeros(640, dtype=np.uint16)
    stages = {name: [0x3F800000] * 30 for name in ff.STAGES}
    prediction = {"normalized_bits": values.tolist(), "residual_bits": values.tolist(), "stages": stages}

    def geometry(array):
        return {"tensor": ff._descriptor(array), "shape": list(array.shape), "strides": [19200, 640, 1] if array.ndim == 3 else [1],
                "dtype": "torch.bfloat16", "alignment_mod16": 0, "device": "cuda:0", "contiguous": True}

    boundary = {"bits": values.tolist(), "geometry": geometry(values)}
    plain = {"order": ff.ORDER.copy(), "boundaries": {name: copy.deepcopy(boundary) for name in ff.ORDER},
             "weight_before": geometry(weight), "weight_after": geometry(weight), "weight_identity_unchanged": True,
             "code_before": "code", "code_after": "code", "runtime_before": {}, "runtime_after": {}, "stopped_at_decoder_tuple": True}
    traced = copy.deepcopy(plain)
    traced.update({"stages": {**copy.deepcopy(stages), "mean_input_metadata": copy.deepcopy(ff.MEAN_GEOMETRY)}, "cuda_events": ["rms_b", "rms_a"],
                   "add": {"operator": "aten.add.Tensor", "alpha": 1, "operand_roles": ["residual_base", "normalized"], "before": [copy.deepcopy(boundary)] * 2,
                           "after": [copy.deepcopy(boundary)] * 2, "output": copy.deepcopy(boundary), "kernel_names": ["add"],
                           "ordered_operand_identity_matches": True, "operands_unchanged": True, "reaches_layer_output": True}})
    plan = {"plan_sha256": "a" * 64, "runtime": {}, "code_sha256": "code", "weight": ff._descriptor(weight), "expected_rms_symbols": ["rms_a", "rms_b"]}
    return plan, values, prediction, [{"traced": copy.deepcopy(traced), "untraced": copy.deepcopy(plain)} for _ in range(3)]


class ArithmeticAndIRTests(unittest.TestCase):
    def test_recorded_output_stage_and_scope(self):
        load = lambda name: json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))
        plan = load("gemma3_270m_post_feedforward_plan.json")
        summary = load("gemma3_270m_post_feedforward_summary.json")
        self.assertEqual(plan["plan_sha256"], _sha({key: value for key, value in plan.items() if key != "plan_sha256"}))
        self.assertEqual(summary["summary_sha256"], _sha({key: value for key, value in summary.items() if key != "summary_sha256"}))
        self.assertEqual(summary["plan_sha256"], plan["plan_sha256"])
        self.assertEqual([node["outputs"] for node in plan["instructions"]], [["layer.0.mlp.post_normalized"], ["hidden.1"]])
        self.assertEqual(summary["normalized_value_count"], 19200)
        self.assertEqual(summary["residual_value_count"], 19200)
        self.assertEqual(summary["scalar_stage_positions"], 90)
        self.assertEqual(summary["mismatch_counts"], {name: [0, 0, 0] for name in (*ff.STAGES, "normalized", "residual")})
        self.assertEqual(summary["untraced_mismatch_counts"], {name: [0, 0, 0] for name in ("normalized", "residual")})
        self.assertTrue(summary["post_feedforward_matches"])
        self.assertTrue(summary["prefix_boundary_reused"])
        self.assertTrue(all(all(item.values()) for item in summary["checks"]))
        for field in ff.FALSE_FLAGS:
            self.assertFalse(summary[field], field)

    def test_actual_independent_rms_lookup_and_add(self):
        values = np.full(ff.SHAPE, 0x3F80, dtype=np.uint16)
        residual = np.full_like(values, 0x3F00)
        lookup = SyntheticRoot()
        with patch.object(lookup, "predict_bits", wraps=lookup.predict_bits) as calls:
            result = ff.predict_post_feedforward(values, residual, [0] * 640, 1e-6, lookup, {})
        self.assertEqual(calls.call_count, 30)
        self.assertTrue(np.all(np.asarray(result["normalized_bits"]) == 0x3F80))
        self.assertTrue(np.all(np.asarray(result["residual_bits"]) == 0x3FC0))
        self.assertEqual(result["stages"]["mean_bits"], [0x3F800000] * 30)
        self.assertEqual(result["stages"]["denominator_bits"], [0x3F800008] * 30)
        self.assertEqual(result["stages"]["rsqrt_bits"], [0x3F800000] * 30)

    def test_signed_zero_and_invalid_domain(self):
        values = np.full(ff.SHAPE, 0x8000, dtype=np.uint16)
        result = ff.predict_post_feedforward(values, values, [0] * 640, 1e-6, SyntheticRoot(), {})
        self.assertTrue(np.all(np.asarray(result["normalized_bits"]) == 0x8000))
        self.assertTrue(np.all(np.asarray(result["residual_bits"]) == 0x8000))
        for bad, residual, weights, epsilon in ((values[..., :639], values, [0] * 640, 1e-6), (values, values.astype(np.int32), [0] * 640, 1e-6),
                                               (values, values, [0] * 639, 1e-6), (values, values, [0] * 640, 1e-5)):
            with self.subTest(shape=bad.shape, epsilon=epsilon), self.assertRaises(ValueError):
                ff.predict_post_feedforward(bad, residual, weights, epsilon, SyntheticRoot(), {})
        bad = values.copy()
        bad[0, 0, 0] = 0x7F80
        with self.assertRaises(ValueError):
            ff.predict_post_feedforward(bad, values, [0] * 640, 1e-6, SyntheticRoot(), {})
        lookup = Mock()
        lookup.predict_bits.side_effect = ValueError("uncovered denominator domain")
        with self.assertRaisesRegex(ValueError, "uncovered"):
            ff.predict_post_feedforward(values, values, [0] * 640, 1e-6, lookup, {})

    def test_exact_ir_and_resealed_wrong_bindings(self):
        original = program()
        nodes = ff.post_instructions(original)
        self.assertEqual([node["id"] for node in nodes], ["i0034", "i0035"])
        changes = [("i0035", "inputs", ["hidden.0", "layer.0.mlp.post_normalized"]), ("i0034", "attributes", {"epsilon": 1e-5}),
                   ("i0034", "layer", 1), ("i0035", "attributes", {"output_dtype": "torch.float32"}), ("i0034", "parameter_refs", [])]
        for identifier, key, value in changes:
            changed = copy.deepcopy(original)
            index = next(i for i, node in enumerate(changed["instructions"]) if node["id"] == identifier)
            changed["instructions"][index][key] = value
            changed["instructions"][index] = reseal(changed["instructions"][index], "instruction_sha256")
            with self.subTest(key=key), self.assertRaises(ValueError):
                ff.post_instructions(reseal(changed, "program_sha256"))
        for key, value in (("shape", ["B", "S", 639]), ("dtype", "torch.float32"), ("producer", "i0033")):
            changed = copy.deepcopy(original)
            changed["tensors"]["hidden.1"][key] = value
            with self.assertRaises(ValueError):
                ff.post_instructions(reseal(changed, "program_sha256"))
        changed = copy.deepcopy(original)
        changed["configuration"]["rms_norm_epsilon"] = 1e-5
        with self.assertRaises(ValueError):
            ff.post_instructions(reseal(changed, "program_sha256"))
        changed = copy.deepcopy(original)
        changed["instructions"][34]["instruction_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            ff.post_instructions(reseal(changed, "program_sha256"))

    def test_registry_and_source_gate(self):
        entry = SimpleNamespace(post=SimpleNamespace(lookup=SyntheticRoot()))
        source = SimpleNamespace(down=SimpleNamespace(product=SimpleNamespace(entry=entry)))
        sources = ff.PostFeedforwardSources(source, {}, {}, {})
        with patch.object(ff, "_code_sha", return_value="code"), patch.object(ff.dense, "verify_dense") as verify:
            with self.assertRaisesRegex(ValueError, "registered"):
                sources.boundaries()
            verify.assert_not_called()
            entry.post.lookup = object.__new__(ff.CheckedRsqrtLookup)
            for result in ({"valid": False}, {"valid": True, "candidate_passes_dense_holdout": False}):
                verify.return_value = result
                with self.assertRaisesRegex(ValueError, "full lineage"):
                    sources.boundaries()
            verify.assert_called_with(source, {}, {}, {})
            entry.post.lookup.predict_bits = lambda bits, runtime: 0
            with self.assertRaisesRegex(ValueError, "registered"):
                sources.boundaries()

    def test_source_branches_are_selected_and_cross_bound(self):
        ir = program()
        values = np.full(ff.SHAPE, 0x3F80, dtype=np.uint16)
        residual = np.full(ff.SHAPE, 0x3F00, dtype=np.uint16)
        entry = SimpleNamespace(post=SimpleNamespace(lookup=object.__new__(ff.CheckedRsqrtLookup), program=ir, runtime={}),
                                post_plan={"runtime": {}, "model_binding": {}}, post_bundle={"predictions": {"residual_bits": residual.tolist()}})
        source = SimpleNamespace(down=SimpleNamespace(program=ir, product=SimpleNamespace(entry=entry)),
                                 model_plan={"runtime": {}, "candidate": {"id": "frozen"}, "model_binding": {}}, model_bundle={"down_bits": values.tolist()})
        sources = ff.PostFeedforwardSources(source, {"runtime": {}, "candidate": {"id": "frozen"}}, {}, {})
        with patch.object(ff, "_code_sha", return_value="code"), patch.object(ff.dense, "verify_dense", return_value={"valid": True, "candidate_passes_dense_holdout": True}):
            actual_down, actual_residual = sources.boundaries()
            np.testing.assert_array_equal(actual_down, values)
            np.testing.assert_array_equal(actual_residual, residual)
            entry.post_plan["runtime"] = {"different": True}
            with self.assertRaisesRegex(ValueError, "runtimes"):
                sources.boundaries()
            entry.post_plan["runtime"] = {}
            source.model_plan["candidate"] = {"id": "refitted"}
            with self.assertRaisesRegex(ValueError, "candidate"):
                sources.boundaries()
            source.model_plan["candidate"] = {"id": "frozen"}
            entry.post_plan["model_binding"] = {"different": True}
            with self.assertRaisesRegex(ValueError, "bindings"):
                sources.boundaries()

    def test_source_change_guard(self):
        with patch.object(ff.Path, "read_bytes", return_value=b"changed"):
            with self.assertRaisesRegex(ValueError, "changed after import"):
                ff._code_sha()


class ReportTests(unittest.TestCase):
    def test_passing_report_scoped_and_symbol_sets_unordered(self):
        plan, values, prediction, observations = fixture()
        observations[1]["traced"]["cuda_events"].reverse()
        report = ff.post_report(plan, values, values, prediction, observations, {})
        self.assertTrue(report["post_feedforward_matches"])
        self.assertTrue(report["prefix_boundary_reused"])
        self.assertEqual(report["value_count"], 38400)
        self.assertEqual(report["scalar_stage_positions"], 90)
        for name in ff.FALSE_FLAGS:
            self.assertIs(report[name], False)

    def test_last_coordinate_mismatches_preserved_in_both_modes(self):
        for stage in ("normalized", "residual"):
            plan, values, prediction, observations = fixture()
            for pair in observations:
                for mode in ("traced", "untraced"):
                    boundary = pair[mode]["boundaries"][stage]
                    boundary["bits"][0][29][639] = 1
                    boundary["geometry"]["tensor"] = ff._descriptor(np.asarray(boundary["bits"], dtype=np.uint16))
                if stage == "normalized":
                    for key in ("before", "after"):
                        pair["traced"]["add"][key][1] = copy.deepcopy(pair["traced"]["boundaries"][stage])
                else:
                    pair["traced"]["add"]["output"] = copy.deepcopy(pair["traced"]["boundaries"][stage])
            report = ff.post_report(plan, values, values, prediction, observations, {})
            self.assertFalse(report["post_feedforward_matches"])
            self.assertEqual(report["mismatch_counts"][stage], [1, 1, 1])
            self.assertEqual(report["untraced_mismatch_counts"][stage], [1, 1, 1])
            self.assertEqual(report["first_divergence"]["coordinate"], [0, 29, 639])

    def test_scalar_mismatch_cannot_hide_behind_equal_bf16(self):
        for stage in ff.STAGES:
            plan, values, prediction, observations = fixture()
            for pair in observations:
                pair["traced"]["stages"][stage][29] ^= 1
            report = ff.post_report(plan, values, values, prediction, observations, {})
            self.assertFalse(report["post_feedforward_matches"])
            self.assertEqual(report["mismatch_counts"][stage], [1, 1, 1])
            self.assertEqual(report["mismatch_counts"]["normalized"], [0, 0, 0])
            self.assertEqual(report["first_divergence"]["stage"], stage)

    def test_geometry_lineage_weight_runtime_and_symbols_fail_closed(self):
        changes = [(lambda x: x["stages"]["mean_input_metadata"].update(input_dtype="torch.bfloat16")),
                   (lambda x: x["add"].update(operand_roles=["normalized", "residual_base"])),
                   (lambda x: x["add"].update(ordered_operand_identity_matches=False)),
                   (lambda x: x["add"].update(reaches_layer_output=False)),
                   (lambda x: x["add"].update(alpha=2)),
                   (lambda x: x["weight_after"].update(strides=[2])),
                   (lambda x: x.update(code_after="changed")),
                   (lambda x: x.update(runtime_after={"changed": True})),
                   (lambda x: x["add"].update(kernel_names=["new_symbol"])),
                   (lambda x: x["boundaries"]["down"]["bits"][0][29].__setitem__(639, 1))]
        for change in changes:
            plan, values, prediction, observations = fixture()
            change(observations[1]["traced"])
            self.assertFalse(ff.post_report(plan, values, values, prediction, observations, {})["post_feedforward_matches"])
        for symbols in ([], ["add", "add"], [""]):
            plan, values, prediction, observations = fixture()
            observations[0]["traced"]["add"]["kernel_names"] = symbols
            with self.assertRaises(ValueError):
                ff.post_report(plan, values, values, prediction, observations, {})

    def test_forged_passing_report_not_accepted(self):
        plan, values, prediction, observations = fixture()
        report = ff.post_report(plan, values, values, prediction, observations, {})
        with patch.object(ff, "check_post_plan", return_value=(values, values, prediction)), patch.object(ff, "_code_sha", return_value="code"):
            self.assertTrue(ff.verify_post(None, plan, {}, report)["valid"])
            report["observations"][0]["traced"]["stages"]["mean_bits"][29] ^= 1
            changed = reseal(report, "report_sha256")
            self.assertFalse(ff.verify_post(None, plan, {}, changed)["valid"])


class CaptureTests(unittest.TestCase):
    def fake_model(self, variant="normal"):
        import torch

        class Norm(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(640, dtype=torch.bfloat16))

            def forward(self, value):
                value = value.float()
                mean = value.pow(2).mean(-1, keepdim=True)
                return (value * torch.rsqrt(mean + 1e-6) * (1 + self.weight.float())).to(torch.bfloat16)

        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pre_feedforward_layernorm = torch.nn.Identity()
                self.mlp = torch.nn.Module()
                self.mlp.down_proj = torch.nn.Identity()
                self.post_feedforward_layernorm = Norm()

            def forward(self, value):
                residual = value + value
                value = self.pre_feedforward_layernorm(residual)
                value = self.mlp.down_proj(value)
                value = self.post_feedforward_layernorm(value)
                if variant == "swap":
                    output = value + residual
                elif variant == "clone":
                    output = residual.clone() + value
                elif variant == "alpha":
                    output = torch.add(residual, value, alpha=2)
                else:
                    output = residual + value
                if variant == "repeat":
                    output = residual + value
                if variant == "disconnect":
                    output = output.clone()
                return output if variant == "tensor" else (output, output) if variant == "tuple2" else (output,)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.layers = torch.nn.ModuleList([Layer(), torch.nn.Identity()])
                self.model.norm = torch.nn.Identity()

            def forward(self, **kwargs):
                value = torch.ones(ff.SHAPE, dtype=torch.bfloat16)
                value = self.model.layers[0](value)[0]
                value = self.model.layers[1](value)
                return self.model.norm(value)

        return Model()

    def capture(self, model, traced):
        import torch

        profiler = Mock()
        profiler.__enter__ = Mock(return_value=profiler)
        profiler.__exit__ = Mock(return_value=False)
        profiler.events.return_value = [SimpleNamespace(name="synthetic", device_type=torch.autograd.DeviceType.CUDA)]
        with patch.object(ff, "_code_sha", return_value="code"), patch.object(ff, "_runtime", return_value={}), patch("torch.cuda.synchronize"), patch("torch.profiler.profile", return_value=profiler) as profiles:
            result = ff._capture_post(model, [[0] * 30], traced)
            self.assertEqual(profiles.call_count, 2 if traced else 0)
        return result

    def assert_clean(self, model):
        for module in model.modules():
            self.assertFalse(module._forward_hooks)
            self.assertFalse(module._forward_pre_hooks)
        for module in (model.model.layers[0], model.model.layers[0].post_feedforward_layernorm):
            self.assertIs(module.forward.__func__, type(module).forward)

    def test_original_tuple_capture_and_plain_stop_cleanup(self):
        model = self.fake_model()
        with patch.object(model.model.layers[1], "forward", side_effect=AssertionError("layer1 executed")), patch.object(model.model.norm, "forward", side_effect=AssertionError("final norm executed")):
            traced = self.capture(model, True)
            plain = self.capture(model, False)
        self.assertEqual(traced["boundaries"], plain["boundaries"])
        self.assertEqual(traced["order"], ff.ORDER)
        self.assertTrue(traced["add"]["reaches_layer_output"])
        self.assertNotIn("add", plain)
        self.assertNotIn("stages", plain)
        self.assertEqual(traced["stages"]["mean_input_metadata"], ff.MEAN_GEOMETRY)
        self.assert_clean(model)

    def test_bad_add_lineage_and_tuple_cleanup(self):
        for variant in ("swap", "clone", "alpha", "repeat", "disconnect", "tensor", "tuple2"):
            model = self.fake_model(variant)
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                self.capture(model, True)
            self.assert_clean(model)
        for variant in ("tensor", "tuple2"):
            model = self.fake_model(variant)
            with self.assertRaises(ValueError):
                self.capture(model, False)
            self.assert_clean(model)


class PlanAndCLITests(unittest.TestCase):
    def test_plan_recomputes_independently_and_rejects_resealed_tamper(self):
        values = np.zeros(ff.SHAPE, dtype=np.uint16)
        weights = np.zeros(640, dtype=np.uint16)
        source_program = program()
        source_program["parameter_commitments"][ff.WEIGHT]["sha256"] = ff._descriptor(weights)["sha256"]
        source_program = reseal(source_program, "program_sha256")
        sources = SimpleNamespace(program=source_program, runtime={}, lookup=SyntheticRoot(), boundaries=lambda: (values, values), commitments=lambda: {"synthetic": True},
                                  dense_sources=SimpleNamespace(model_plan={"model_binding": {}}), entry=SimpleNamespace(post_report={"observations": [{"traced": {"cuda_events": ["synthetic"]}}] * 3}))
        prediction = ff.predict_post_feedforward(values, values, weights.tolist(), 1e-6, sources.lookup, {})
        bundle = {"weight_bits": weights.tolist(), "predictions": prediction}
        with patch.object(ff, "_code_sha", return_value="code"):
            plan = _seal(ff._plan_body(sources, values, values, weights, bundle), "plan_sha256")
            self.assertEqual(ff.check_post_plan(sources, plan, bundle)[2], prediction)
            changed = copy.deepcopy(bundle)
            changed["predictions"]["stages"]["rsqrt_bits"][29] ^= 1
            changed_plan = _seal(ff._plan_body(sources, values, values, weights, changed), "plan_sha256")
            with self.assertRaisesRegex(ValueError, "independently"):
                ff.check_post_plan(sources, changed_plan, changed)

    def test_cli_defaults_preserve_dense_and_prefix_flags(self):
        from bioprocess_runtime.cli import build_parser, command_post_feedforward, command_dense_k2048

        parser = build_parser()
        for operation in ("plan", "run", "verify"):
            argv = ["gemma-post-feedforward-" + operation, "program"] + (["--report", "report"] if operation == "verify" else ["--output", "output"])
            args = parser.parse_args(argv)
            self.assertIs(args.handler, command_post_feedforward)
            self.assertEqual(args.plan.name, "gemma3_270m_post_feedforward_plan.json")
            self.assertEqual(args.bundle.name, "gemma3_270m_post_feedforward_predictions.json")
            self.assertEqual(args.k2048_dense_plan.name, "gemma3_270m_k2048_dense_plan.json")
            self.assertEqual(args.dense_plan.name, "gemma3_270m_k128_dense_plan.json")
            self.assertEqual(args.model_down_plan.name, "gemma3_270m_mlp_down_plan_v2.json")
        dense_args = parser.parse_args(["gemma-k2048-dense-plan", "program", "--bundle", "bundle", "--output", "output"])
        self.assertIs(dense_args.handler, command_dense_k2048)
        self.assertFalse(hasattr(dense_args, "k2048_dense_plan"))

    def test_cli_output_guards_before_loading(self):
        from bioprocess_runtime.cli import command_post_feedforward

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            existing = base / "existing"
            existing.write_text("preserve", encoding="utf-8")
            for output, bundle, source in ((existing, base / "bundle", base / "source"), (base / "same", base / "same", base / "source"), (base / "source", base / "bundle", base / "source")):
                args = SimpleNamespace(operation="plan", output=output, bundle=bundle, k2048_dense_plan=source)
                with patch("bioprocess_runtime.cli._load_post_feedforward_sources") as load, self.assertRaises(ValueError):
                    command_post_feedforward(args)
                load.assert_not_called()
            self.assertEqual(existing.read_text(encoding="utf-8"), "preserve")

    def test_failed_run_preserves_report_with_exclusive_writes(self):
        from bioprocess_runtime.cli import command_post_feedforward
        from transformers import AutoModelForCausalLM

        args = SimpleNamespace(operation="run", plan=Path("plan"), bundle=Path("bundle"), output=Path("new-report"), summary=Path("new-summary"), model_path=Path("model"))
        failed = {"post_feedforward_matches": False, "mismatch_counts": {"rsqrt_bits": [0, 0, 1]}}
        original = Mock()
        with ExitStack() as stack:
            stack.enter_context(patch("bioprocess_runtime.cli._load_post_feedforward_sources", return_value="sources"))
            stack.enter_context(patch.object(Path, "exists", return_value=False))
            stack.enter_context(patch.object(Path, "read_text", return_value="{}"))
            stack.enter_context(patch.object(Path, "mkdir"))
            opened = stack.enter_context(patch.object(Path, "open"))
            stack.enter_context(patch("torch.cuda.is_available", return_value=True))
            stack.enter_context(patch.object(AutoModelForCausalLM, "from_pretrained", return_value=original))
            stack.enter_context(patch.object(ff, "acquire_post", return_value=failed))
            stack.enter_context(patch.object(ff, "post_summary", return_value=failed))
            stack.enter_context(redirect_stdout(io.StringIO()))
            self.assertEqual(command_post_feedforward(args), 1)
        self.assertEqual(opened.call_count, 2)
        self.assertTrue(all(call.args == ("x",) for call in opened.call_args_list))
        written = "".join(call.args[0] for call in opened.return_value.__enter__.return_value.write.call_args_list)
        self.assertIn('"post_feedforward_matches": false', written)
        self.assertIn('"rsqrt_bits"', written)

    def test_cli_reexecute_rebuilds_before_replay_single_model_no_refit(self):
        from bioprocess_runtime.cli import command_post_feedforward
        from transformers import AutoModelForCausalLM

        for reexecute, same, valid in ((False, True, True), (True, True, True), (True, False, True), (True, True, False)):
            calls = []
            args = SimpleNamespace(operation="verify", plan=Path("plan"), bundle=Path("bundle"), report=Path("report"), model_path=Path("model"), summary=None, reexecute=reexecute)
            plan, bundle, report = {"plan": 1}, {"bundle": 1}, {"report": 1}
            original = Mock()
            original.to.return_value.eval.return_value = original
            with ExitStack() as stack:
                stack.enter_context(patch("bioprocess_runtime.cli._load_post_feedforward_sources", return_value="sources"))
                payloads = iter((plan, bundle, report))
                stack.enter_context(patch.object(Path, "read_text", side_effect=lambda **kwargs: json.dumps(next(payloads))))
                stack.enter_context(patch.object(ff, "verify_post", side_effect=lambda *a: calls.append("verify") or {"valid": valid}))
                stack.enter_context(patch("torch.cuda.is_available", return_value=True))
                loader = stack.enter_context(patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=lambda *a, **k: calls.append("load") or original))
                stack.enter_context(patch.object(ff, "build_post_plan", side_effect=lambda *a: calls.append("rebuild") or (plan if same else {}, bundle)))
                replay = stack.enter_context(patch.object(ff, "acquire_post", side_effect=lambda *a: calls.append("replay") or report))
                stack.enter_context(redirect_stdout(io.StringIO()))
                result = command_post_feedforward(args)
            expected = ["verify"] + (["load", "rebuild"] + (["replay"] if same else []) if reexecute and valid else [])
            self.assertEqual(calls, expected)
            self.assertEqual(loader.call_count, int(reexecute and valid))
            self.assertEqual(replay.call_count, int(reexecute and valid and same))
            self.assertEqual(result, 0 if valid and (not reexecute or same) else 1)
