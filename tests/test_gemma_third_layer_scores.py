from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from bioprocess_runtime import cli
from bioprocess_runtime import gemma_third_layer_scores as scores
from bioprocess_runtime.gemma_rotary_slice import _sha, _seal
from test_gemma_first_layer import real_program, profiles, geometry
import test_gemma_third_layer_entry as entry_tests

ROOT = Path(__file__).resolve().parents[1]


def roots():
    result = {name: np.full(scores.SHAPES[name], 0x3F80, dtype=np.uint16) for name in scores.ROOTS}
    result[scores.SIN].fill(0)
    result[scores.MASK] = scores.first.causal_mask_bits()
    return {name: value.tolist() for name, value in result.items()}


def uniform_dot(left, right, mode, profile, workers):
    if not np.all(left == left[0]) or not np.all(right == right[0]):
        raise AssertionError("Uniform test shortcut used on nonuniform rows")
    value = scores.first._dot(left[0].tolist(), right[0].tolist(), mode, profile)
    return np.full((left.shape[0], right.shape[0]), value, dtype=np.uint16)


class ScoreTests(unittest.TestCase):
    def test_recorded_rotary_score_scope_and_complete_comparisons(self):
        load = lambda name: json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))
        plan = load("gemma3_270m_third_layer_scores_plan.json")
        summary = load("gemma3_270m_third_layer_scores_summary.json")
        for payload, key in ((plan, "plan_sha256"), (summary, "summary_sha256")):
            self.assertEqual(payload[key], _sha({name: value for name, value in payload.items() if name != key}))
        self.assertEqual(summary["plan_sha256"], plan["plan_sha256"])
        self.assertEqual(summary["trace_root"], plan["trace_root"])
        self.assertEqual(summary["coverage"]["instruction_ids"], [f"i{i:04d}" for i in range(74, 80)])
        self.assertEqual(summary["coverage"]["new_state_count"], 7)
        self.assertEqual(summary["coverage"]["rotated_value_count"], 38400)
        self.assertNotIn(scores.SCALE, summary["coverage"]["plain_observed_states"])
        self.assertEqual(summary["aggregate_mismatch_count"], 0)
        self.assertEqual(summary["original_forward_count"], 6)
        self.assertTrue(summary["rotary_scores_match"])
        self.assertTrue(all(all(value.values()) for value in summary["checks"]))
        self.assertTrue(all(summary["acquisition_guards"].values()))
        for field in scores.FALSE_FLAGS:
            self.assertFalse(summary[field], field)

    @classmethod
    def setUpClass(cls):
        cls.program, cls.roots, cls.profiles = real_program(), roots(), profiles()
        with patch.object(scores.first, "_project_rows", side_effect=uniform_dot):
            cls.execution = scores.execute_scores(cls.program, cls.roots, cls.profiles, 1)

    def test_actual_ir_complete_companion_and_stop_scope(self):
        nodes = scores.instructions(self.program)
        self.assertEqual([node["id"] for node in nodes], [f"i{i:04d}" for i in range(74, 80)])
        self.assertEqual([name for node in nodes for name in node["outputs"]], list(scores.OUTPUTS))
        self.assertTrue(scores._coverage()["value_repeat_is_companion_not_masked_score_dependency"])
        self.assertEqual(scores._coverage()["rotated_value_count"], 38400)
        changed = copy.deepcopy(self.program)
        changed["instructions"][74]["attributes"]["rotary_profile"] = "global"
        with self.assertRaises(ValueError):
            scores.instructions(changed)

    def test_independent_rotation_dot_scale_and_mask(self):
        states = self.execution["state_bits"]
        self.assertEqual(states[scores.QR], self.roots[scores.Q])
        self.assertEqual(states[scores.KR], self.roots[scores.K])
        self.assertTrue(np.all(np.asarray(states[scores.RAW]) == 0x4380))
        self.assertTrue(np.all(np.asarray(states[scores.SCALE]) == 0x4180))
        self.assertEqual(states[scores.MASKED][0][3][29][29], 0x4180)
        self.assertEqual(states[scores.MASKED][0][0][0][29], 0xFF7F)
        changed = copy.deepcopy(self.roots)
        changed[scores.Q] = np.full(scores.SHAPES[scores.Q], 0x4000, dtype=np.uint16).tolist()
        with patch.object(scores.first, "_project_rows", side_effect=uniform_dot):
            output = scores.execute_scores(self.program, changed, self.profiles, 1)
        self.assertEqual(output["state_bits"][scores.RAW][0][0][0][0], 0x4400)

    def test_root_profile_and_ledger_tampering(self):
        scores.check_execution(self.program, self.roots, self.execution)
        for case in ("root", "output", "record", "missing"):
            changed = copy.deepcopy(self.execution)
            if case == "root":
                changed["state_bits"][scores.Q][0][0][0][0] ^= 1
            elif case == "output":
                changed["state_bits"][scores.MASKED][0][3][29][29] ^= 1
            elif case == "record":
                changed["records"][0]["payload"]["inputs"][scores.Q]["producer_record_hash"] = "wrong"
            else:
                changed["records"].pop()
            with self.subTest(case=case), self.assertRaises(ValueError):
                scores.check_execution(self.program, self.roots, changed)
        with self.assertRaises(ValueError):
            scores.execute_scores(self.program, self.roots, {}, 1)
        with self.assertRaises(ValueError):
            scores.execute_scores(self.program, {}, self.profiles, 1)
        with patch.object(scores.Path, "read_bytes", return_value=b"changed"), self.assertRaises(ValueError):
            scores._code_sha()

    def native_fixture(self):
        plan = _seal({"scope": scores.SCOPE, "coverage": scores._coverage(), "code_sha256": "synthetic", "runtime": {}, "bundle_sha256": _sha(self.execution),
                      "expected_score_kernel_sets": {name: ["synthetic"] for name in (scores.RAW, scores.SCALE, scores.MASKED)}, **scores._flags()}, "plan_sha256")
        record = {"state_bits": copy.deepcopy(self.execution["state_bits"]), "geometry": {name: geometry(shape, "torch.bfloat16") for name, shape in scores.SHAPES.items()},
                  "kernels": {name: ["synthetic"] for name in ("rotary_apply", scores.RAW, scores.SCALE, scores.MASKED)},
                  "code_before": "synthetic", "code_after": "synthetic", "runtime_before": {}, "runtime_after": {},
                  "stopped_before_softmax": True, "token_commitment_unchanged": True, "all_original_operands_unchanged": True}
        plain = copy.deepcopy(record)
        plain["state_bits"].pop(scores.SCALE)
        plain["geometry"].pop(scores.SCALE)
        plain["kernels"] = {}
        return plan, [{"traced": copy.deepcopy(record), "untraced": copy.deepcopy(plain)} for _ in range(3)]

    def test_all_stages_compared_and_plain_missing_scale_explicit(self):
        plan, observations = self.native_fixture()
        guards = dict.fromkeys(scores.GUARDS, True)
        self.assertTrue(scores.score_report(plan, self.execution, observations, guards)["rotary_scores_match"])
        for mode, name in (("traced", scores.QR), ("traced", scores.SCALE), ("traced", scores.MASKED), ("untraced", scores.RAW), ("untraced", scores.MASKED)):
            changed = copy.deepcopy(observations)
            value = changed[0][mode]["state_bits"][name]
            value[0][-1][-1][-1] ^= 1
            report = scores.score_report(plan, self.execution, changed, guards)
            self.assertFalse(report["rotary_scores_match"])
            self.assertGreater(report["aggregate_mismatch_count"], 0)
        for key in scores.GUARDS:
            self.assertFalse(scores.score_report(plan, self.execution, observations, {**guards, key: False})["rotary_scores_match"])

    def test_provenance_runtime_and_coverage_fail_closed(self):
        plan, observations = self.native_fixture()
        for key in ("runtime", "device", "kernel", "lineage"):
            changed = copy.deepcopy(observations)
            traced = changed[1]["traced"]
            if key == "runtime":
                traced["runtime_after"] = {"changed": True}
            elif key == "device":
                traced["geometry"][scores.Q]["device"] = "cuda:1"
            elif key == "kernel":
                traced["kernels"][scores.RAW] = ["different"]
            else:
                traced["all_original_operands_unchanged"] = False
            self.assertFalse(scores.score_report(plan, self.execution, changed, dict.fromkeys(scores.GUARDS, True))["rotary_scores_match"])
        with self.assertRaises(ValueError):
            scores.score_report(plan, self.execution, observations[:2], dict.fromkeys(scores.GUARDS, True))

    def test_default_verify_and_replay_order(self):
        plan, observations = self.native_fixture()
        report = scores.score_report(plan, self.execution, observations, dict.fromkeys(scores.GUARDS, True))
        source = SimpleNamespace(commitments=lambda: {})
        with patch.object(scores, "check_score_plan"), patch.object(scores, "_code_sha", return_value="synthetic"), patch.object(scores, "execute_scores", side_effect=AssertionError("no numerical rerun")):
            self.assertTrue(scores.verify_scores(source, plan, self.execution, report, ROOT)["valid"])
        for changed in (False, True):
            calls = []
            def build(*args):
                calls.append("predict")
                return ({"changed": True} if changed else plan), self.execution
            def capture(*args):
                self.assertEqual(calls, ["predict"])
                calls.append("native")
                return report
            with patch.object(scores, "verify_scores", return_value={"valid": True}), patch.object(scores, "build_score_plan", side_effect=build), patch.object(scores, "acquire_scores", side_effect=capture):
                result = scores.replay_scores(source, None, plan, self.execution, report, ROOT, ROOT, ROOT)
            self.assertEqual(result["valid"], not changed)
            self.assertEqual(calls, ["predict"] if changed else ["predict", "native"])

    def test_cli_defaults_and_existing_output_guard(self):
        parser = cli.build_parser()
        for operation in ("plan", "run", "verify"):
            args = ["gemma-third-layer-scores-" + operation, "program"]
            args += ["--report", "report"] if operation == "verify" else ["--output", "output"]
            parsed = parser.parse_args(args)
            self.assertEqual(parsed.plan.name, "gemma3_270m_third_layer_scores_plan.json")
            self.assertEqual(parsed.third_entry_plan.name, "gemma3_270m_third_layer_entry_plan.json")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "existing.json"
            path.write_text("preserve", encoding="utf-8")
            args = parser.parse_args(["gemma-third-layer-scores-plan", "program", "--output", str(path)])
            with patch.object(cli, "_load_third_score_sources", side_effect=AssertionError("reject before loading")), self.assertRaises(ValueError):
                cli.command_third_layer_scores(args)
            self.assertEqual(path.read_text(), "preserve")

    def test_real_source_preflight_without_new_prediction(self):
        args = cli.build_parser().parse_args(["gemma-third-layer-scores-plan", "results/gemma3_270m_execution_ir.json", "--output", "unused"])
        for name, value in vars(args).items():
            if isinstance(value, Path) and not value.is_absolute():
                setattr(args, name, ROOT / value)
        paths = [value for name, value in vars(args).items() if isinstance(value, Path) and name not in ("output", "plan", "bundle", "summary") and "audit" not in name]
        if any(not path.exists() for path in paths):
            self.skipTest("Required ignored source evidence is unavailable")
        with patch.object(scores, "execute_scores", side_effect=AssertionError("preflight cannot predict")), patch.object(scores, "capture_scores", side_effect=AssertionError("preflight cannot acquire")):
            source = cli._load_third_score_sources(args)
            ids, providers, arithmetic, boundaries, bindings, kernels = source.validate(args.model_path)
        self.assertEqual(set(boundaries), set(scores.ROOTS))
        self.assertEqual(len(bindings), 6)
        self.assertEqual(set(kernels), {scores.RAW, scores.SCALE, scores.MASKED})
        self.assertEqual(len(ids[0]), 30)


class NativeScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        entry_tests.NativeCaptureTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        entry_tests.NativeCaptureTests.tearDownClass.__func__(cls)

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
        with patch.object(scores, "_code_sha", return_value="synthetic"), patch.object(scores.first, "_runtime", return_value={}), patch.object(torch.profiler, "profile", new=Profile), patch.object(torch.cuda, "synchronize", side_effect=AssertionError("no CUDA")):
            return scores.capture_scores(self.model, self.ids, traced)

    def test_original_native_states_controls_prefix_and_softmax_stop(self):
        import torch
        original = torch.nn.functional.softmax
        with patch.object(torch.nn.functional, "softmax", wraps=original) as softmax:
            traced = self.capture(True)
            self.assertEqual(softmax.call_count, 2)
        plain = self.capture(False)
        self.assertEqual(set(traced["state_bits"]), set(scores.SHAPES))
        self.assertEqual(set(plain["state_bits"]), set(scores.SHAPES) - {scores.SCALE})
        self.assertTrue(all(plain["state_bits"][name] == traced["state_bits"][name] for name in plain["state_bits"]))
        self.assertEqual(set(traced["kernels"]), {"rotary_apply", scores.RAW, scores.SCALE, scores.MASKED})
        self.assertEqual(plain["kernels"], {})
        self.assertTrue(traced["stopped_before_softmax"])

    def test_cleanup_after_native_failure(self):
        from transformers.models.gemma3 import modeling_gemma3 as gemma
        before = [(module.forward, dict(module._forward_hooks), dict(module._forward_pre_hooks)) for module in self.model.modules()]
        original = gemma.apply_rotary_pos_emb
        def broken(q, k, cosine, sine, *args, **kwargs):
            if calls[0] == 2:
                raise ValueError("synthetic rotary failure")
            calls[0] += 1
            return original(q, k, cosine, sine, *args, **kwargs)
        calls = [0]
        with patch.object(gemma, "apply_rotary_pos_emb", new=broken), self.assertRaisesRegex(ValueError, "synthetic rotary failure"):
            self.capture(True)
        self.assertIs(gemma.apply_rotary_pos_emb, original)
        self.assertEqual(before, [(module.forward, dict(module._forward_hooks), dict(module._forward_pre_hooks)) for module in self.model.modules()])
