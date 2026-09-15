from __future__ import annotations

import copy
import inspect
import json
import unittest
from argparse import Namespace
from contextlib import nullcontext
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from bioprocess_runtime import gemma_k2048_dense as dense
from bioprocess_runtime import gemma_mlp_down as down
from bioprocess_runtime.gemma_float_semantics import decode_finite_bfloat16
from bioprocess_runtime.gemma_rotary_slice import _seal, _sha

ROOT = Path(__file__).resolve().parent.parent


def candidate():
    return json.loads(down.SUPPORTED_CANDIDATE_JSON)


def fake_sources(excluded=None):
    return SimpleNamespace(check=lambda: (candidate(), excluded or []), commitments=lambda: {"synthetic_source": "unit-test-only"},
                           model_plan={"runtime": {"unit_test": True}}, down=SimpleNamespace(probe_plan={"expected_kernel_names": ["z_ampere", "a_splitKreduce"]}))


def snapshot(descriptor, role):
    shape, strides = dense.GEOMETRY[role]
    return {"tensor": descriptor, "shape": list(shape), "strides": strides, "dtype": "torch.bfloat16", "alignment_mod16": 0, "device": "cuda:0", "contiguous": True}


def observations(plan, predicted):
    pairs = []
    for repetition in range(3):
        pair = {}
        for index, mode in enumerate(("untraced", "traced")):
            operands = {role: snapshot(plan[role], role) for role in ("input", "weight")}
            pair[mode] = {"call": 2 * repetition + index, "mode": mode, "before": operands, "after": copy.deepcopy(operands),
                          "output": snapshot(dense._descriptor(predicted), "output"), "output_bits": predicted.tolist(),
                          "kernel_names": list(reversed(plan["expected_kernel_names"])) if mode == "traced" else [],
                          "operand_identity_unchanged": True, "linear_identity_matches": True, "code_before": plan["code_sha256"], "code_after": plan["code_sha256"],
                          "runtime_before": plan["runtime"], "runtime_after": plan["runtime"]}
        pairs.append(pair)
    return pairs


def reseal(plan, **changes):
    return _seal({**{key: value for key, value in plan.items() if key != "plan_sha256"}, **changes}, "plan_sha256")


class DenseK2048Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs, cls.weights, cls.columns = dense.dense_vectors()
        cls.predicted = np.zeros(dense.OUTPUT_SHAPE, dtype=np.uint16)
        cls.sources = fake_sources()
        cls.bundle = {"input_bits": cls.inputs.tolist(), "weight_bits": cls.weights.tolist(), "prediction_bits": cls.predicted.tolist()}
        cls.plan = _seal(dense._plan_body(cls.sources, candidate(), [], cls.inputs, cls.weights, cls.columns, cls.predicted, cls.bundle), "plan_sha256")

    def test_full_generator_deterministic_populated_unique_and_no_selection(self):
        with patch.object(down, "project_down_bits", side_effect=AssertionError("Generator must not predict")):
            inputs, weights, columns = dense.dense_vectors()
        self.assertTrue(np.array_equal(inputs, self.inputs))
        self.assertTrue(np.array_equal(weights, self.weights))
        self.assertEqual(columns, self.columns)
        hashes = dense._validate_vectors(inputs, weights, [])
        self.assertEqual([len(set(items)) for items in hashes], [30, 640])
        self.assertEqual(len(set(hashes[0] + hashes[1])), 670)
        for values in (inputs, weights):
            self.assertTrue(np.all((values & 127) != 0))
            self.assertTrue(np.all((((values >> 7) & 255) > 0) & (((values >> 7) & 255) < 255)))
        self.assertEqual(sum(item["family"] == "dense" for item in columns), 320)
        self.assertEqual(sum(item["pairing"] == "adjacent" for item in columns), 160)
        self.assertEqual(sum(item["pairing"] == "across_halves" for item in columns), 160)
        expected_coordinates = {start + offset % min(192, 2048 - start) for start in range(0, 2048, 192) for offset in (0, 7, 8, 63, 64, 127, 128, 191)}
        for pairing in ("adjacent", "across_halves"):
            self.assertEqual({item["anchor_row"] for item in columns if item["pairing"] == pairing}, set(range(30)))
            self.assertEqual({item["perturbed_coordinate"] for item in columns if item["pairing"] == pairing}, expected_coordinates)
        self.assertEqual({item["perturbed_coordinate"] // 192 for item in columns[320:]}, set(range(11)))
        self.assertEqual(candidate()["partitions"][-1], [1920, 2048])
        self.assertNotEqual(dense.SEED, 0xB632D9E1)
        self.assertNotIn("project_down_bits", inspect.getsource(dense.dense_vectors))
        self.assertNotIn("torch", inspect.getsource(dense.dense_vectors))
        self.assertFalse(self.plan["prediction_based_case_selection"])

    def test_recorded_dense_summary_preserves_scope(self):
        load = lambda name: json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))
        plan = load("gemma3_270m_k2048_dense_plan.json")
        summary = load("gemma3_270m_k2048_dense_summary.json")
        self.assertEqual(summary["summary_sha256"], _sha({key: value for key, value in summary.items() if key != "summary_sha256"}))
        self.assertEqual(plan["plan_sha256"], _sha({key: value for key, value in plan.items() if key != "plan_sha256"}))
        self.assertEqual(summary["plan_sha256"], plan["plan_sha256"])
        self.assertEqual(summary["candidate"], candidate())
        self.assertEqual(summary["value_count"], 19200)
        self.assertEqual((summary["input_row_count"], summary["weight_vector_count"]), (30, 640))
        self.assertEqual(summary["linear_call_count"], 6)
        self.assertEqual(summary["mismatch_counts"], [0, 0, 0])
        self.assertEqual(summary["untraced_mismatch_counts"], [0, 0, 0])
        self.assertTrue(summary["candidate_passes_dense_holdout"])
        self.assertTrue(summary["vectors_disjoint_from_declared_sources"])
        self.assertTrue(all(all(item.values()) for item in summary["checks"]))
        self.assertEqual(len(set(plan["input_row_hashes"] + plan["weight_vector_hashes"])), 670)
        self.assertFalse(set(plan["excluded_vector_hashes"]) & set(plan["input_row_hashes"] + plan["weight_vector_hashes"]))
        for field in dense.FALSE_FLAGS:
            self.assertFalse(summary[field], field)

    def test_exact_cancellation_before_single_perturbation_both_pairings_and_tail(self):
        decode = lambda bits: decode_finite_bfloat16(int(bits))[0]
        for index in (0, 1, 10, 21, 87, 175, 319):
            metadata = self.columns[320 + index]
            row = self.inputs[0, metadata["anchor_row"]]
            values = self.weights[320 + index].copy()
            coordinate = metadata["perturbed_coordinate"]
            start = ((index // 2) % 11) * 192
            self.assertEqual(coordinate, start + (0, 7, 8, 63, 64, 127, 128, 191)[(index // 22) % 8] % min(192, 2048 - start))
            partner = coordinate ^ 1 if metadata["pairing"] == "adjacent" else (coordinate + 1024) % 2048
            power = ((int(values[partner]) >> 7) & 255) - ((int(row[coordinate]) >> 7) & 255)
            self.assertIn(power, range(-3, 4))
            original = (int(row[partner]) & 0x807F) | (((int(row[partner]) >> 7 & 255) + power) << 7)
            if (metadata["pairing"] == "adjacent" and coordinate % 2) or (metadata["pairing"] == "across_halves" and coordinate >= 1024):
                original ^= 0x8000
            self.assertEqual(int(values[coordinate]), original + (-1 if original & 127 == 127 else 1))
            residual = decode(row[coordinate]) * (decode(values[coordinate]) - decode(original))
            self.assertNotEqual(residual, 0)
            self.assertEqual(sum((decode(a) * decode(b) for a, b in zip(row, values)), Fraction()), residual)
            values[coordinate] = original
            for pair in range(1024):
                a, b = (pair * 2, pair * 2 + 1) if metadata["pairing"] == "adjacent" else (pair, pair + 1024)
                self.assertEqual(decode(row[a]) * decode(values[a]) + decode(row[b]) * decode(values[b]), 0)
            self.assertEqual(sum((decode(a) * decode(b) for a, b in zip(row, values)), Fraction()), 0)

    def test_vector_domain_overlap_and_exclusion_rejection(self):
        for excluded in ([_sha(self.inputs[0, 29].tolist())], [_sha(self.weights[639].tolist())]):
            with self.assertRaises(ValueError):
                dense._validate_vectors(self.inputs, self.weights, excluded)
        for bits in (0, 0x3F80, 0x7F81, 1):
            changed = self.weights.copy()
            changed[639, 2047] = bits
            with self.assertRaises(ValueError):
                dense._validate_vectors(self.inputs, changed, [])
        changed = self.weights.copy()
        changed[-1] = self.inputs[0, -1]
        with self.assertRaises(ValueError):
            dense._validate_vectors(self.inputs, changed, [])
        changed[-1] = changed[0]
        with self.assertRaises(ValueError):
            dense._validate_vectors(self.inputs, changed, [])
        inputs = self.inputs.copy()
        inputs[0, -1] = inputs[0, 0]
        with self.assertRaises(ValueError):
            dense._validate_vectors(inputs, self.weights, [])

    def test_source_full_pool_model_and_backend_exclusions_material_only_once(self):
        model_inputs = np.arange(30 * 2048, dtype=np.uint16).reshape(dense.INPUT_SHAPE)
        model_weights = np.zeros(dense.WEIGHT_SHAPE, dtype=np.uint16)
        model_weights[:, 0] = np.arange(640, dtype=np.uint16)
        source = SimpleNamespace(material=Mock(return_value=(model_inputs, candidate())), commitments=lambda: {"full_prefix": True},
                                 probe_plan={"runtime": {}, "excluded_backend_vector_hashes": ["b" * 64]})
        plan = {"kernel_provenance": dense.KERNEL_PROVENANCE, "metadata_correction_development_regression": True, "value_count": 19200, "candidate": candidate(), "runtime": {}, "plan_sha256": "p"}
        sources = dense.DenseSources(source, plan, {"weights": "committed"}, {"report_sha256": "r"})

        def check(cached, *args):
            inputs, _ = cached.material()
            return inputs, model_weights, self.predicted

        def verify(cached, *args):
            check(cached)
            return {"valid": True, "down_matches": True, "mismatch_counts": [0, 0, 0]}

        with patch.object(down, "verify_down", side_effect=verify) as verified, patch.object(down, "check_down_plan", side_effect=check), patch.object(down, "down_summary", return_value={"summary_sha256": "computed"}):
            selected, excluded = sources.check()
            source.material.assert_called_once()
            self.assertEqual(selected, candidate())
            left, pool = dense.probe_pool()
            expected = {_sha(left), *(_sha(item["right_bits"]) for item in pool), "b" * 64, *(_sha(row) for row in model_inputs[0].tolist()), *(_sha(row) for row in model_weights.tolist())}
            self.assertEqual(excluded, sorted(expected))
            self.assertEqual(len(pool), 256)
            self.assertEqual(sources.commitments()["model_summary_sha256"], "computed")
            self.assertEqual(sources.commitments()["model_bundle_sha256"], _sha(sources.model_bundle))
            verified.side_effect = None
            for failure in ({"valid": False}, {"valid": True, "down_matches": False}, {"valid": True, "down_matches": True, "mismatch_counts": [0, 0, 1]}):
                verified.return_value = failure
                with self.assertRaises(ValueError):
                    sources.check()
            verified.side_effect = verify
            for key, value in (("kernel_provenance", "launch order"), ("metadata_correction_development_regression", False), ("runtime", {"changed": True}), ("candidate", {**candidate(), "id": "other"})):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    dense.DenseSources(source, {**plan, key: value}, {}, {}).check()

    def test_integrity_no_prediction_scope_source_runtime_and_bundle_tamper(self):
        pairs = observations(self.plan, self.predicted)
        report = dense.dense_report(self.plan, self.predicted, pairs, self.plan["runtime"])
        with patch.object(down, "project_down_bits", side_effect=AssertionError("No numerical prediction in integrity verification")), patch.object(dense, "_runtime", side_effect=AssertionError("No CUDA runtime lookup in integrity verification")):
            checked = dense.verify_dense(self.sources, self.plan, self.bundle, report)
            self.assertTrue(checked["valid"], checked)
            self.assertEqual(checked["mode"], dense.VERIFY_MODE)
            self.assertFalse(checked["predictions_recomputed"])
            for changes in ({"excluded_vector_hashes": ["extra"]}, {"sources": {}}, {"runtime": {}}, {"kernel_provenance": "launch order"}, {"columns": self.columns[:-1]}, *({flag: True} for flag in ("global_exactness_activation_allowed", "generic_domain_qualified", "original_model_revalidated", "native_registers_observed"))):
                bad = dense.verify_dense(self.sources, reseal(self.plan, **changes), self.bundle, report)
                self.assertFalse(bad["valid"], changes)
            bundle = {**self.bundle, "weight_bits": copy.deepcopy(self.bundle["weight_bits"])}
            bundle["weight_bits"][639][2047] ^= 1
            self.assertFalse(dense.verify_dense(self.sources, reseal(self.plan, bundle_sha256=_sha(bundle)), bundle, report)["valid"])
        for flag in dense.FALSE_FLAGS:
            self.assertIs(report[flag], False)
            self.assertIs(self.plan[flag], False)
        with patch.object(dense.Path, "read_bytes", return_value=b"source changed"):
            with self.assertRaises(ValueError):
                dense._code_sha()

    def test_report_symbols_order_missing_duplicates_runtime_and_geometry(self):
        pairs = observations(self.plan, self.predicted)
        report = dense.dense_report(self.plan, self.predicted, pairs, self.plan["runtime"])
        self.assertTrue(report["candidate_passes_dense_holdout"])
        for names in (["z_ampere"], ["replacement", "a_splitKreduce"]):
            bad = copy.deepcopy(pairs)
            bad[0]["traced"]["kernel_names"] = names
            failed = dense.dense_report(self.plan, self.predicted, bad, self.plan["runtime"])
            self.assertFalse(failed["candidate_passes_dense_holdout"])
            self.assertTrue(failed["scope_abstained"])
        for names in ([], ["x", "x"], [None], [""], "x"):
            bad = copy.deepcopy(pairs)
            bad[0]["traced"]["kernel_names"] = names
            with self.assertRaises(ValueError):
                dense.dense_report(self.plan, self.predicted, bad, self.plan["runtime"])
        bad = copy.deepcopy(pairs)
        del bad[0]["traced"]["kernel_names"]
        with self.assertRaises(KeyError):
            dense.dense_report(self.plan, self.predicted, bad, self.plan["runtime"])
        for field, value in (("operand_identity_unchanged", False), ("linear_identity_matches", False), ("code_after", "changed"), ("runtime_after", {}), ("call", 5)):
            bad = copy.deepcopy(pairs)
            bad[0]["untraced"][field] = value
            self.assertFalse(dense.dense_report(self.plan, self.predicted, bad, self.plan["runtime"])["candidate_passes_dense_holdout"])
        for role in ("input", "weight", "output"):
            bad = copy.deepcopy(pairs)
            record = bad[0]["traced"]["output"] if role == "output" else bad[0]["traced"]["after"][role]
            record["strides"] = [1]
            record["tensor"]["sha256"] = "changed"
            self.assertFalse(dense.dense_report(self.plan, self.predicted, bad, self.plan["runtime"])["candidate_passes_dense_holdout"])
        self.assertTrue(dense.dense_report(self.plan, self.predicted, pairs, {})["scope_abstained"])

    def test_last_row_last_column_family_pairing_and_untraced_failure(self):
        pairs = observations(self.plan, self.predicted)
        for mode, column in (("traced", 639), ("untraced", 319), ("traced", 320)):
            actual = self.predicted.copy()
            actual[0, 29, column] = 1
            record = pairs[2][mode]
            record["output_bits"] = actual.tolist()
            record["output"] = snapshot(dense._descriptor(actual), "output")
            report = dense.dense_report(self.plan, self.predicted, pairs, self.plan["runtime"])
            self.assertFalse(report["candidate_passes_dense_holdout"])
            family = self.columns[column]["family"]
            self.assertEqual(report["family_mismatch_counts"][family][mode], [0, 0, 1])
            self.assertEqual(next(item for item in report["mismatches"] if item["mode"] == mode)["column"], column)
            self.assertFalse(report["checks"][2]["untraced_output_matches"])
            if column == 639:
                self.assertEqual(report["pairing_mismatch_counts"]["across_halves"][mode], [0, 0, 1])
            pairs = observations(self.plan, self.predicted)

    def test_build_reuses_unchanged_predictor_and_validates_before_prediction(self):
        with patch.object(down, "project_down_bits", return_value=self.predicted) as predict:
            plan, bundle = dense.build_dense_plan(self.sources, 4)
            self.assertEqual(plan, self.plan)
            self.assertEqual(bundle, self.bundle)
            self.assertEqual(predict.call_args.args[2:], (candidate(), 4))
            self.assertTrue(np.array_equal(predict.call_args.args[0], self.inputs))
            predict.reset_mock()
            with self.assertRaises(ValueError):
                dense.build_dense_plan(fake_sources([_sha(self.weights[-1].tolist())]), 4)
            predict.assert_not_called()
            for workers in (0, 5, True):
                with self.assertRaises(ValueError):
                    dense.build_dense_plan(self.sources, workers)
            predict.assert_not_called()
        inputs = np.full((1, 1, 2048), 0x3F81, dtype=np.uint16)
        weights = np.zeros((1, 2048), dtype=np.uint16)
        weights[0, -1] = 0x4000
        with patch.multiple(down, INPUT_SHAPE=(1, 1, 2048), WEIGHT_SHAPE=(1, 2048), OUTPUT_SHAPE=(1, 1, 1)):
            serial = down.project_down_bits(inputs, weights, candidate(), 1)
            for workers in (2, 3, 4):
                self.assertTrue(np.array_equal(serial, down.project_down_bits(inputs, weights, candidate(), workers)))

    def test_capture_has_one_original_call_no_hidden_warmup(self):
        events = []
        tensor = SimpleNamespace(data_ptr=lambda: 16, _version=0)
        linear = Mock(side_effect=lambda *args: events.append("linear") or tensor)
        profiler = SimpleNamespace(events=lambda: [SimpleNamespace(name="kernel", device_type="CUDA")])
        fake_torch = SimpleNamespace(nn=SimpleNamespace(functional=SimpleNamespace(linear=linear)), _C=SimpleNamespace(_nn=SimpleNamespace(linear=linear)),
                                     cuda=SimpleNamespace(synchronize=lambda: None), autograd=SimpleNamespace(DeviceType=SimpleNamespace(CUDA="CUDA")))
        fake_profiler = SimpleNamespace(profile=lambda **kwargs: nullcontext(profiler), ProfilerActivity=SimpleNamespace(CPU="CPU", CUDA="CUDA"))
        with patch.dict("sys.modules", {"torch": fake_torch, "torch.profiler": fake_profiler}), patch.object(dense, "_snapshot", return_value={}), patch.object(dense, "_bits", return_value=np.zeros((1,), dtype=np.uint16)), patch.object(dense, "_code_sha", return_value="fixed"), patch.object(dense, "_runtime", return_value={}):
            for call, mode in enumerate(("untraced", "traced") * 3):
                record = dense._capture_linear(tensor, tensor, call, mode)
                self.assertTrue(record["operand_identity_unchanged"])
                self.assertTrue(record["linear_identity_matches"])
                self.assertEqual(record["kernel_names"], ["kernel"] if mode == "traced" else [])
            self.assertEqual(len(events), 6)
            fake_torch.nn.functional.linear = Mock()
            with self.assertRaises(ValueError):
                dense._capture_linear(tensor, tensor, 6, "untraced")
        self.assertEqual([item["mode"] for item in dense.SCHEDULE], ["untraced", "traced"] * 3)
        self.assertNotIn("_profile_call", inspect.getsource(dense))

    def test_cli_defaults_inheritance_and_write_guards(self):
        from bioprocess_runtime import cli
        parser = cli.build_parser()
        for operation in ("plan", "run", "verify"):
            argv = ["gemma-k2048-dense-" + operation, "program", "--bundle", "bundle"]
            argv += ["--report", "report"] if operation == "verify" else ["--output", "output"]
            args = parser.parse_args(argv)
            self.assertIs(args.handler, cli.command_dense_k2048)
            self.assertEqual(args.workers, 4)
            self.assertEqual(args.plan.name, "gemma3_270m_k2048_dense_plan.json")
            self.assertEqual(args.model_down_plan.name, "gemma3_270m_mlp_down_plan_v2.json")
            self.assertEqual(args.model_down_bundle.name, "gemma3_270m_mlp_down_predictions_v2.json")
            self.assertEqual(args.model_down_report.name, "gemma3_270m_mlp_down_report_v2.json")
            self.assertEqual(args.probe_plan.name, "gemma3_270m_k2048_probes_plan.json")
            self.assertEqual(args.prefix_probe_plan.name, "gemma3_270m_k1024_probes_plan.json")
            self.assertEqual(args.entry_plan.name, "gemma3_270m_mlp_entry_plan.json")
            self.assertEqual(args.gelu_table.name, "gemma3_270m_gelu_table.bin")
        old = parser.parse_args(["gemma-mlp-down-plan", "program", "--bundle", "bundle", "--output", "output"])
        self.assertIs(old.handler, cli.command_mlp_down)
        self.assertEqual(old.plan.name, "gemma3_270m_mlp_down_plan.json")
        product = parser.parse_args(["gemma-mlp-product-plan", "program", "--bundle", "bundle", "--output", "output"])
        self.assertEqual(product.probe_plan.name, "gemma3_270m_k1024_probes_plan.json")
        with patch.object(cli, "_load_dense_k2048_sources", side_effect=AssertionError("Guard must precede loading")):
            for guarded in (Namespace(operation="plan", output=ROOT / "AGENTS.md", bundle=Path("new")), Namespace(operation="plan", output=Path("same"), bundle=Path("same")), Namespace(operation="run", output=Path("same"), summary=Path("same")), Namespace(operation="plan", output=Path("source"), bundle=Path("new"), model_down_plan=Path("source"))):
                with self.assertRaises(ValueError):
                    cli.command_dense_k2048(guarded)
        self.assertNotIn("from_pretrained", inspect.getsource(cli.command_dense_k2048))
        prefix = Mock()
        with patch.object(cli, "_load_product_sources", return_value=prefix) as load_product, patch.object(Path, "read_text", return_value="{}"):
            cli._load_down_sources(old)
            inherited = load_product.call_args.args[0]
            self.assertEqual(inherited.probe_plan, old.prefix_probe_plan)
            self.assertEqual(old.probe_plan.name, "gemma3_270m_k2048_probes_plan.json")

    def test_cli_reexecute_regenerates_before_acquisition_no_refit(self):
        from bioprocess_runtime import cli
        plan, bundle, report = {"plan": 1}, {"bundle": 1}, {"report": 1}
        payloads = {"plan": plan, "bundle": bundle, "report": report}
        args = Namespace(operation="verify", plan=Path("plan"), bundle=Path("bundle"), report=Path("report"), summary=None, reexecute=True, workers=4)
        events = []
        with patch.object(cli, "_load_dense_k2048_sources", return_value=self.sources), patch.object(Path, "read_text", autospec=True, side_effect=lambda path, **kwargs: json.dumps(payloads[str(path)])), patch.object(dense, "verify_dense", return_value={"valid": True}), patch.object(dense, "build_dense_plan", side_effect=lambda *args: events.append("predict") or (plan, bundle)) as predict, patch.object(dense, "acquire_dense", side_effect=lambda *args: events.append("acquire") or report) as acquire, patch("builtins.print"):
            self.assertEqual(cli.command_dense_k2048(args), 0)
            self.assertEqual(events, ["predict", "acquire"])
            predict.assert_called_once_with(self.sources, 4)
            acquire.assert_called_once_with(self.sources, plan, bundle)
            acquire.reset_mock()
            predict.side_effect = lambda *args: ({"changed": True}, bundle)
            self.assertEqual(cli.command_dense_k2048(args), 1)
            acquire.assert_not_called()

    def test_cli_failed_run_writes_report_exclusively_before_exit_one(self):
        from bioprocess_runtime import cli
        args = Namespace(operation="run", plan=Path("plan"), bundle=Path("bundle"), output=Path("new-report"), summary=Path("new-summary"))
        failed = {"candidate_passes_dense_holdout": False, "untraced_mismatch_counts": [0, 0, 1]}
        with patch.object(cli, "_load_dense_k2048_sources", return_value=self.sources), patch.object(Path, "exists", return_value=False), patch.object(Path, "read_text", return_value="{}"), patch.object(Path, "mkdir"), patch.object(Path, "open") as opened, patch.object(dense, "acquire_dense", return_value=failed), patch.object(dense, "dense_summary", return_value=failed), patch("builtins.print"):
            self.assertEqual(cli.command_dense_k2048(args), 1)
            self.assertEqual(opened.call_count, 2)
            self.assertTrue(all(call.args == ("x",) for call in opened.call_args_list))
            written = "".join(call.args[0] for call in opened.return_value.__enter__.return_value.write.call_args_list)
            self.assertIn('"candidate_passes_dense_holdout": false', written)
            self.assertIn('"untraced_mismatch_counts"', written)

    def test_optional_real_v2_source_integrity_only(self):
        from bioprocess_runtime.cli import build_parser, _load_dense_k2048_sources
        args = build_parser().parse_args(["gemma-k2048-dense-plan", "results/gemma3_270m_execution_ir.json", "--bundle", "unused", "--output", "unused-plan"])
        for name, value in vars(args).items():
            if isinstance(value, Path) and not value.is_absolute():
                setattr(args, name, ROOT / value)
        required = [value for name, value in vars(args).items() if isinstance(value, Path) and name not in ("bundle", "output", "plan", "model_path") and not name.endswith("audit_dir")]
        if any(not path.exists() for path in required):
            self.skipTest("Ignored serialized source tensors unavailable")
        with patch.object(down, "project_down_bits", side_effect=AssertionError("No real prediction")), patch.object(dense, "acquire_dense", side_effect=AssertionError("No acquisition")):
            sources = _load_dense_k2048_sources(args)
            selected, excluded = sources.check()
            self.assertEqual(selected["id"], dense.SUPPORTED_ID)
            dense._validate_vectors(self.inputs, self.weights, excluded)
            self.assertEqual(sources.model_report["mismatch_counts"], [0, 0, 0])
            self.assertEqual(sources.commitments()["model_verification_mode"], dense.SOURCE_MODE)


if __name__ == "__main__":
    unittest.main()
