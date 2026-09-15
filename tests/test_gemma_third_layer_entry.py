from __future__ import annotations

import copy
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from bioprocess_runtime import cli
from bioprocess_runtime import gemma_third_layer_entry as entry
from bioprocess_runtime.gemma_rotary_slice import _sha, _seal
from test_gemma_first_layer import real_program, providers, profiles, cheap_rms, cheap_rows, geometry

ROOT = Path(__file__).resolve().parents[1]


class EntryTests(unittest.TestCase):
    def test_recorded_entry_preserves_scope_and_complete_comparisons(self):
        load = lambda name: json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))
        plan = load("gemma3_270m_third_layer_entry_plan.json")
        summary = load("gemma3_270m_third_layer_entry_summary.json")
        for payload, key in ((plan, "plan_sha256"), (summary, "summary_sha256")):
            self.assertEqual(payload[key], _sha({name: value for name, value in payload.items() if name != key}))
        self.assertEqual(summary["plan_sha256"], plan["plan_sha256"])
        self.assertEqual(summary["trace_root"], plan["trace_root"])
        self.assertEqual(summary["coverage"]["instruction_ids"], [f"i{i:04d}" for i in range(65, 74)])
        self.assertEqual(summary["coverage"]["parameter_count"], 6)
        self.assertEqual(summary["coverage"]["rms_scalar_positions"], 540)
        self.assertEqual(summary["coverage"]["value_heads_mapping"], entry.V_MAPPING)
        self.assertEqual(summary["aggregate_mismatch_count"], 0)
        self.assertEqual(summary["original_forward_count"], 6)
        self.assertTrue(summary["entry_matches"])
        self.assertTrue(summary["prefix_boundary_reused"])
        self.assertTrue(all(all(check.values()) for check in summary["checks"]))
        self.assertTrue(all(summary["acquisition_guards"].values()))
        for field in entry.FALSE_FLAGS:
            self.assertFalse(summary[field], field)

    @classmethod
    def setUpClass(cls):
        cls.program = real_program()
        cls.snapshots = {}
        for node in entry.instructions(cls.program):
            for name in node["parameter_refs"]:
                commitment = copy.deepcopy(cls.program["parameter_commitments"][name])
                value = np.zeros(commitment["shape"], dtype=np.uint16)
                descriptor = entry.first._descriptor(value)
                commitment["sha256"] = descriptor["sha256"]
                cls.program["parameter_commitments"][name] = commitment
                cls.snapshots[name] = {"bits": value.tolist(), "descriptor": descriptor, "commitment": commitment}
        cls.program = _seal({key: value for key, value in cls.program.items() if key != "program_sha256"}, "program_sha256")
        cls.lookup, cls.profiles = providers(), profiles()
        cls.hidden = np.full((1, 30, 640), 0x3F80, dtype=np.uint16).tolist()
        with cls.synthetic():
            cls.execution = entry.execute_entry(cls.program, cls.hidden, cls.snapshots, cls.lookup, cls.profiles, {"synthetic": True}, 1)

    @classmethod
    def synthetic(cls):
        stack = ExitStack()
        stack.enter_context(patch.object(entry.first, "PROGRAM_SHA256", cls.program["program_sha256"]))
        stack.enter_context(patch.object(entry, "_code_sha", return_value="synthetic-code"))
        stack.enter_context(patch.object(entry.first, "_rms_lookup_row", side_effect=cheap_rms))
        stack.enter_context(patch.object(entry.first, "_project_rows", side_effect=cheap_rows))
        return stack

    def test_actual_ir_scope_and_regime(self):
        program = real_program()
        nodes = entry.instructions(program)
        self.assertEqual([node["id"] for node in nodes], [f"i{i:04d}" for i in range(65, 74)])
        self.assertEqual(len({name for node in nodes for name in node["parameter_refs"]}), 6)
        self.assertTrue(all(node["layer"] == 2 for node in nodes))
        self.assertEqual(entry._coverage(program)["rms_scalar_positions"], 540)
        self.assertEqual(entry._coverage(program)["target_value_count"], 46080)
        changed = copy.deepcopy(program)
        changed["instructions"][65]["inputs"] = ["hidden.1"]
        with self.assertRaises(ValueError):
            entry.instructions(changed)
        self.assertTrue(entry._flags()["prefix_boundary_reused"])
        self.assertFalse(entry._flags()["connected_three_layers_independently_recomputed"])

    def test_fresh_internal_execution_and_root_propagation(self):
        with self.synthetic():
            changed = np.full((1, 30, 640), 0x4000, dtype=np.uint16).tolist()
            output = entry.execute_entry(self.program, changed, self.snapshots, self.lookup, self.profiles, {"synthetic": True}, 1)
        self.assertEqual(len(output["records"]), 9)
        self.assertEqual(len(output["state_bits"]), 10)
        self.assertEqual(sum(len(values) for stages in output["scalar_stages"].values() for values in stages.values()), 540)
        self.assertNotEqual(output["state_bits"]["layer.2.query.normalized"], self.execution["state_bits"]["layer.2.query.normalized"])
        for node in entry.instructions(real_program()):
            if node["opcode"] == "LINEAR":
                self.assertEqual(entry._mode(node), "serial" if "query" in node["outputs"][0] else "key_value")

    def test_snapshot_root_and_ledger_tampering(self):
        with self.synthetic():
            entry.check_execution(self.program, self.hidden, self.snapshots, self.lookup, self.profiles, {"synthetic": True}, self.execution)
            for part in ("root", "value", "producer", "missing", "scalar"):
                changed = copy.deepcopy(self.execution)
                if part == "root":
                    changed["state_bits"][entry.ROOT][0][-1][-1] ^= 1
                elif part == "value":
                    changed["state_bits"]["layer.2.key.normalized"][0][0][-1][-1] ^= 1
                elif part == "producer":
                    changed["records"][0]["payload"]["inputs"][entry.ROOT]["producer_record_hash"] = "wrong"
                elif part == "missing":
                    changed["records"].pop()
                else:
                    changed["scalar_stages"]["layer.2.key.normalized"]["mean_bits"].pop()
                with self.subTest(part=part), self.assertRaises(ValueError):
                    entry.check_execution(self.program, self.hidden, self.snapshots, self.lookup, self.profiles, {"synthetic": True}, changed)
            snapshots = copy.deepcopy(self.snapshots)
            snapshots[next(iter(snapshots))]["bits"][0] ^= 1
            with self.assertRaises(ValueError):
                entry.check_snapshots(self.program, snapshots)
            snapshots = dict(self.snapshots)
            snapshots["model.layers.0.input_layernorm.weight"] = snapshots.pop("model.layers.2.input_layernorm.weight")
            with self.assertRaises(ValueError):
                entry.check_snapshots(self.program, snapshots)

    def test_provider_geometry_and_source_guards(self):
        with self.synthetic():
            with self.assertRaises(ValueError):
                entry.execute_entry(self.program, self.hidden, self.snapshots, object(), self.profiles, {}, 1)
            with self.assertRaises(ValueError):
                entry.execute_entry(self.program, self.hidden, self.snapshots, self.lookup, self.profiles, {}, 0)
        with patch.object(entry.Path, "read_bytes", return_value=b"changed"):
            with self.assertRaises(ValueError):
                entry._code_sha()

    def native_fixture(self):
        coverage = entry._coverage(real_program())
        kernels = {node["outputs"][0]: ["synthetic"] for node in entry.instructions(real_program()) if node["opcode"] in ("RMS_NORM", "LINEAR")}
        bundle = {"parameter_snapshots": self.snapshots, "execution": self.execution}
        plan = _seal({"scope": entry.SCOPE, "code_sha256": "synthetic-code", "runtime": {}, "coverage": coverage, "expected_kernel_sets": kernels, "bundle_sha256": _sha(bundle), **entry._flags()}, "plan_sha256")
        plain = {"state_bits": copy.deepcopy(self.execution["state_bits"]), "geometry": {name: geometry(value["shape"], value["dtype"]) for name, value in coverage["states"].items()},
                 "scalar_stages": {}, "kernels": {}, "code_before": "synthetic-code", "code_after": "synthetic-code", "runtime_before": {}, "runtime_after": {},
                 "stopped_before_target_rotary": True, "token_commitment_unchanged": True, "value_heads_mapping": entry.V_MAPPING}
        traced = copy.deepcopy(plain)
        traced["kernels"] = kernels
        traced["scalar_stages"] = copy.deepcopy(self.execution["scalar_stages"])
        for name, stages in traced["scalar_stages"].items():
            shape = coverage["states"][name]["shape"]
            stages["mean_input_metadata"] = {"input_shape": shape, "input_dtype": "torch.float32", "axes": [-1], "keepdim": True, "alignment_mod16": 0, "input_strides": geometry(shape, "torch.float32")["strides"]}
        return plan, bundle, [{"traced": copy.deepcopy(traced), "untraced": copy.deepcopy(plain)} for _ in range(3)]

    def test_native_report_all_coordinates_and_scalar_precision(self):
        plan, bundle, observed = self.native_fixture()
        guards = dict.fromkeys(("checkpoint_unchanged", "source_unchanged", "frozen_files_unchanged"), True)
        report = entry.entry_report(plan, bundle, observed, guards)
        self.assertTrue(report["entry_matches"])
        for mode, kind in (("traced", "value"), ("untraced", "value"), ("traced", "scalar")):
            changed = copy.deepcopy(observed)
            if kind == "value":
                changed[0][mode]["state_bits"]["layer.2.key.normalized"][0][0][-1][-1] ^= 1
            else:
                changed[0][mode]["scalar_stages"]["layer.2.key.normalized"]["rsqrt_bits"][-1] ^= 1
            failed = entry.entry_report(plan, bundle, changed, guards)
            self.assertFalse(failed["entry_matches"])
            self.assertGreater(failed["aggregate_mismatch_count"], 0)
        for field in guards:
            self.assertFalse(entry.entry_report(plan, bundle, observed, {**guards, field: False})["entry_matches"])

    def test_runtime_kernels_mapping_and_missing_observations(self):
        plan, bundle, observed = self.native_fixture()
        guards = dict.fromkeys(("checkpoint_unchanged", "source_unchanged", "frozen_files_unchanged"), True)
        for kind in ("runtime", "kernels", "mapping", "source"):
            changed = copy.deepcopy(observed)
            traced = changed[0]["traced"]
            if kind == "runtime":
                traced["runtime_after"] = {"changed": True}
            elif kind == "kernels":
                traced["kernels"] = {}
            elif kind == "mapping":
                traced["value_heads_mapping"] = "original operation observed"
            else:
                traced["code_after"] = "changed"
            self.assertFalse(entry.entry_report(plan, bundle, changed, guards)["entry_matches"])
        with self.assertRaises(ValueError):
            entry.entry_report(plan, bundle, observed[:2], guards)

    def test_cli_defaults_and_exclusive_guards(self):
        parser = cli.build_parser()
        for mode in ("plan", "run", "verify"):
            argv = ["gemma-third-layer-entry-" + mode, "program"]
            argv += ["--report", "report"] if mode == "verify" else ["--output", "output"]
            args = parser.parse_args(argv)
            self.assertEqual(args.plan.name, "gemma3_270m_third_layer_entry_plan.json")
            self.assertEqual(args.two_holdout_plan.name, "gemma3_270m_two_layers_holdout_plan.json")
            self.assertEqual(args.protocol.name, "gemma3_270m_first_layer_holdout_protocol.json")
            self.assertEqual(args.holdout_protocol.name, "gemma3_270m_two_layers_holdout_protocol.json")
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "existing.json"
            output.write_text("preserve", encoding="utf-8")
            args = parser.parse_args(["gemma-third-layer-entry-plan", "program", "--output", str(output)])
            with patch.object(cli, "_load_third_entry_sources", side_effect=AssertionError("reject before loading")), self.assertRaises(ValueError):
                cli.command_third_layer_entry(args)
            self.assertEqual(output.read_text(), "preserve")

    def test_integrity_verification_does_not_predict_and_rejects_resealed_report(self):
        plan, bundle, observed = self.native_fixture()
        report = entry.entry_report(plan, bundle, observed, dict.fromkeys(("checkpoint_unchanged", "source_unchanged", "frozen_files_unchanged"), True))
        source = SimpleNamespace(commitments=lambda: {})
        with patch.object(entry, "check_entry_plan"), patch.object(entry, "_code_sha", return_value="synthetic-code"), patch.object(entry, "execute_entry", side_effect=AssertionError("no prediction in integrity mode")):
            checked = entry.verify_entry(source, plan, bundle, report, ROOT)
            self.assertTrue(checked["valid"])
            self.assertFalse(checked["numerical_recomputation_performed"])
            changed = copy.deepcopy(report)
            changed["entry_matches"] = False
            changed = _seal({key: value for key, value in changed.items() if key != "report_sha256"}, "report_sha256")
            self.assertFalse(entry.verify_entry(source, plan, bundle, changed, ROOT)["valid"])

    def test_replay_rebuilds_before_native_and_blocks_changed_predictions(self):
        plan, bundle, report = {"plan": 1}, {"bundle": 1}, {"entry_matches": True}
        for changed in (False, True):
            events = []
            def build(*args):
                events.append("predict")
                return ({"plan": 2} if changed else plan), bundle
            def acquire(*args):
                self.assertEqual(events, ["predict"])
                events.append("native")
                return report
            with patch.object(entry, "verify_entry", return_value={"valid": True}), patch.object(entry, "build_entry_plan", side_effect=build), patch.object(entry, "acquire_entry", side_effect=acquire):
                checked = entry.replay_entry(None, None, plan, bundle, report, ROOT, ROOT, ROOT)
            self.assertEqual(events, ["predict"] if changed else ["predict", "native"])
            self.assertEqual(checked["valid"], not changed)

    def test_source_report_pin_and_missing_summary_fail_before_execution(self):
        args = cli.build_parser().parse_args(["gemma-third-layer-entry-verify", "program", "--report", "report", "--summary", str(ROOT / "artifacts" / "absent-entry-summary.json")])
        with patch.object(cli, "_holdout_model", side_effect=AssertionError("no model")), patch.object(cli, "_load_third_entry_sources", side_effect=AssertionError("no sources")), self.assertRaises(ValueError):
            cli.command_third_layer_entry(args)
        protocol, plan, report = [{"x": index} for index in range(3)]
        for payload, field in ((protocol, "protocol_sha256"), (plan, "plan_sha256"), (report, "report_sha256")):
            payload[field] = _sha(payload)
        source = entry.EntrySources(None, protocol, plan, {}, report)
        with patch.object(entry, "_code_sha", return_value="synthetic"), patch.object(entry.EntrySources, "commitments", return_value={}), self.assertRaisesRegex(ValueError, "declared two-layer"):
            source.validate(ROOT)

    def test_real_source_preflight_no_prediction(self):
        args = cli.build_parser().parse_args(["gemma-third-layer-entry-plan", "results/gemma3_270m_execution_ir.json", "--output", "unused"])
        for name, value in vars(args).items():
            if isinstance(value, Path) and not value.is_absolute():
                setattr(args, name, ROOT / value)
        paths = [value for name, value in vars(args).items() if isinstance(value, Path) and name not in ("output", "plan", "bundle", "summary") and "audit" not in name]
        if any(not path.exists() for path in paths):
            self.skipTest("Ignored source evidence unavailable")
        with patch.object(entry, "execute_entry", side_effect=AssertionError("no prediction")), patch.object(entry, "capture_entry", side_effect=AssertionError("no native acquisition")):
            source = cli._load_third_entry_sources(args)
            ids, lookup, arithmetic, hidden, binding, kernels = source.validate(args.model_path)
        self.assertEqual(len(ids[0]), 30)
        self.assertEqual(binding["instruction_id"], "i0064")
        self.assertEqual(len(kernels), 6)
        self.assertIs(type(lookup), entry.first.Providers)


class NativeCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import Gemma3ForCausalLM, Gemma3TextConfig
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1202)
            config = Gemma3TextConfig(vocab_size=32, hidden_size=640, intermediate_size=2048, head_dim=256, num_attention_heads=4,
                                     num_key_value_heads=1, num_hidden_layers=4, layer_types=["sliding_attention"] * 4,
                                     sliding_window=512, max_position_embeddings=512, query_pre_attn_scalar=256,
                                     hidden_activation="gelu_pytorch_tanh", attention_dropout=0.0, attn_implementation="eager")
            cls.model = Gemma3ForCausalLM(config).to(dtype=torch.bfloat16).eval()
        cls.ids = [[1 + index % 31 for index in range(30)]]

    @classmethod
    def tearDownClass(cls):
        import torch
        torch.set_num_threads(cls.threads)

    def capture(self, traced):
        import torch
        owner = self
        class Profile:
            def __init__(self, **kwargs):
                owner.assertEqual(kwargs["activities"], [torch.profiler.ProfilerActivity.CPU])
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def events(self):
                return [SimpleNamespace(name="synthetic", device_type=torch.autograd.DeviceType.CUDA)]
        with patch.object(entry, "_code_sha", return_value="synthetic-code"), patch.object(entry.first, "_runtime", return_value={}), patch.object(torch.profiler, "profile", new=Profile), patch.object(torch.cuda, "synchronize", side_effect=AssertionError("no CUDA")):
            return entry.capture_entry(self.model, self.ids, traced)

    def test_snapshot_uses_actual_layer_two_weights_without_forward(self):
        program = real_program()
        names = {name for node in entry.instructions(program) for name in node["parameter_refs"]}
        bound = dict(self.model.named_parameters(remove_duplicate=False))
        bound.update(dict(self.model.named_buffers(remove_duplicate=False)))
        for name in names:
            program["parameter_commitments"][name]["sha256"] = entry.first._descriptor(entry.first._bits(bound[name]))["sha256"]
        program = _seal({key: value for key, value in program.items() if key != "program_sha256"}, "program_sha256")
        source = SimpleNamespace(program=program, baseline=SimpleNamespace(two_sources=SimpleNamespace(second_sources=None)))
        with patch.object(entry.first, "PROGRAM_SHA256", program["program_sha256"]), patch.object(entry.second, "_model_context", return_value=bound), patch.object(entry.first, "bind_model_tensors", return_value=bound), patch.object(self.model, "forward", side_effect=AssertionError("snapshot must not execute model")):
            snapshots = entry._snapshot_model(source, self.model)
            self.assertEqual(set(snapshots), names)
            for name in names:
                self.assertEqual(snapshots[name]["descriptor"]["sha256"], program["parameter_commitments"][name]["sha256"])
            with patch.object(self.model.model.layers[2], "layer_idx", 1), self.assertRaises(ValueError):
                entry._snapshot_model(source, self.model)

    def test_synthetic_original_capture_and_minimal_observer_difference(self):
        before = [(module, module.forward, dict(module._forward_hooks), dict(module._forward_pre_hooks)) for module in self.model.modules()]
        traced, plain = self.capture(True), self.capture(False)
        self.assertEqual(traced["state_bits"], plain["state_bits"])
        self.assertEqual(len(traced["state_bits"]), 10)
        self.assertEqual(len(traced["kernels"]), 6)
        self.assertEqual(sum(len(stages[key]) for stages in traced["scalar_stages"].values() for key in entry.first.STAGES), 540)
        self.assertEqual(plain["kernels"], {})
        self.assertEqual(plain["scalar_stages"], {})
        self.assertEqual(traced["value_heads_mapping"], entry.V_MAPPING)
        self.assertEqual(before, [(module, module.forward, dict(module._forward_hooks), dict(module._forward_pre_hooks)) for module in self.model.modules()])

    def test_stop_before_rotary_and_later_operations(self):
        from transformers.models.gemma3 import modeling_gemma3 as gemma
        forbidden = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("out of scope"))
        with patch.object(gemma, "apply_rotary_pos_emb", wraps=gemma.apply_rotary_pos_emb) as rotary, patch.object(self.model.model.layers[2].self_attn.o_proj, "forward", side_effect=forbidden), patch.object(self.model.model.layers[2].mlp, "forward", side_effect=forbidden), patch.object(self.model.model.layers[3], "forward", side_effect=forbidden):
            self.assertTrue(self.capture(True)["stopped_before_target_rotary"])
            self.assertEqual(rotary.call_count, 2)

    def test_capture_failure_restores_every_hook_and_method(self):
        modules = list(self.model.modules())
        before = [(module.forward, dict(module._forward_hooks), dict(module._forward_pre_hooks)) for module in modules]
        with patch.object(entry, "_observe_rms_module", side_effect=ValueError("synthetic failure")), self.assertRaisesRegex(ValueError, "synthetic failure"):
            self.capture(True)
        self.assertEqual(before, [(module.forward, dict(module._forward_hooks), dict(module._forward_pre_hooks)) for module in modules])
