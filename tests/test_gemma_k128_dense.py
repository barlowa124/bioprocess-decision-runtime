from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bioprocess_runtime.gemma_k128_dense import dense_vectors, _validate_vectors, _code_sha, dense_report
from bioprocess_runtime.gemma_float_semantics import decode_finite_bfloat16


class K128DenseTests(unittest.TestCase):
    def test_dense_distinct_vectors_and_overlap_rejection(self) -> None:
        inputs, weights, columns = dense_vectors()
        input_hashes, weight_hashes = _validate_vectors(inputs, weights, [])
        self.assertEqual(len(set(input_hashes)), 30)
        self.assertEqual(len(set(weight_hashes)), 640)
        self.assertEqual(sum(item["family"] == "dense" for item in columns), 320)
        self.assertEqual(sum(item["family"] == "row_anchored_cancellation" for item in columns), 320)
        repeated = inputs.copy()
        repeated[0, 1] = repeated[0, 0]
        with self.assertRaises(ValueError):
            _validate_vectors(repeated, weights, [])
        with self.assertRaises(ValueError):
            _validate_vectors(inputs, weights, [input_hashes[0]])
        with self.assertRaises(ValueError):
            _validate_vectors(inputs, weights, [weight_hashes[0]])

    def test_cancellation_anchors_have_one_perturbed_pair(self) -> None:
        inputs, weights, columns = dense_vectors()
        for column in (320, 351, 399, 639):
            anchor = columns[column]["anchor_row"]
            nonzero = []
            for position in range(0, 1024, 2):
                total = sum(decode_finite_bfloat16(int(inputs[0, anchor, index]))[0] * decode_finite_bfloat16(int(weights[column, index]))[0] for index in (position, position + 1))
                if total:
                    nonzero.append(position // 2)
            self.assertEqual(nonzero, [columns[column]["perturbed_coordinate"] // 2])
        again = dense_vectors()
        self.assertTrue(np.array_equal(inputs, again[0]))
        self.assertTrue(np.array_equal(weights, again[1]))
        self.assertEqual(columns, again[2])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is unavailable")
    def test_every_matrix_coordinate_is_compared(self) -> None:
        from bioprocess_runtime.gemma_attention_entry import _descriptor

        inputs, weights, columns = dense_vectors()
        predictions = np.zeros((1, 30, 640), dtype=np.uint16)
        actual = predictions.copy()
        actual[0, 29, 639] = 1
        descriptors = {"input": _descriptor(inputs), "weight": _descriptor(weights)}
        observation = {name: {"tensor": {**descriptor, "device": "cuda:0"}, "strides": [30720, 1024, 1] if name == "input" else [1024, 1]} for name, descriptor in descriptors.items()}
        observation.update({"output_bits": actual.tolist(), "kernel_names": ["synthetic"]})
        plan = {**descriptors, "plan_sha256": "a" * 64, "columns": columns, "runtime": {}, "expected_kernel_names": ["synthetic"]}
        report = dense_report(plan, predictions, [observation] * 3, {})
        self.assertEqual(report["mismatch_counts"], [1, 1, 1])
        self.assertEqual(report["first_divergence"]["row"], 29)
        self.assertEqual(report["first_divergence"]["column"], 639)
        self.assertFalse(report["candidate_passes_dense_holdout"])
        self.assertTrue(report["complete_matrix_compared"])

    def test_recorded_dense_scope_and_complete_agreement(self) -> None:
        from bioprocess_runtime.gemma_rotary_slice import _sha

        root = Path(__file__).resolve().parent.parent
        plan = json.loads((root / "results/gemma3_270m_k128_dense_plan.json").read_text(encoding="utf-8"))
        summary = json.loads((root / "results/gemma3_270m_k128_dense_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["summary_sha256"], _sha({key: value for key, value in summary.items() if key != "summary_sha256"}))
        self.assertEqual(summary["plan_sha256"], plan["plan_sha256"])
        self.assertTrue(summary["candidate_passes_dense_holdout"])
        self.assertTrue(summary["complete_matrix_compared"])
        self.assertEqual(summary["mismatch_counts"], [0, 0, 0])
        self.assertEqual(len(set(plan["input_row_hashes"])), 30)
        self.assertEqual(len(set(plan["weight_vector_hashes"])), 640)
        self.assertFalse(set(plan["excluded_vector_hashes"]) & set(plan["input_row_hashes"] + plan["weight_vector_hashes"]))
        self.assertFalse(plan["prediction_based_case_selection"])
        self.assertFalse(summary["original_model_revalidated"])
        self.assertFalse(summary["hardware_partitioning_established"])
        self.assertFalse(summary["global_exactness_activation_allowed"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is unavailable")
    def test_real_dense_evidence_rejects_overlap_claim_and_output_tampering(self) -> None:
        from bioprocess_runtime.cli import build_parser, _load_dense_k128_sources
        from bioprocess_runtime.gemma_k128_dense import verify_dense
        from bioprocess_runtime.gemma_rotary_slice import _seal

        root = Path(__file__).resolve().parent.parent
        if any(not (root / "artifacts" / ("gemma3_270m_" + name)).exists() for name in ("k128_dense_inputs.json", "k128_dense_report.json", "attention_output_predictions.json", "k1024_probes_inputs.json", "k1024_probes_report.json")):
            self.skipTest("Dense/source tensors intentionally remain outside Git")
        args = build_parser().parse_args(["gemma-k128-dense-verify", "--bundle", "artifacts/gemma3_270m_k128_dense_inputs.json", "--report", "artifacts/gemma3_270m_k128_dense_report.json"])
        for name, value in vars(args).items():
            if isinstance(value, Path) and not value.is_absolute():
                setattr(args, name, root / value)
        load = lambda path: json.loads(path.read_text(encoding="utf-8"))
        sources = _load_dense_k128_sources(args)
        plan, bundle, report = load(args.plan), load(args.bundle), load(args.report)
        self.assertTrue(verify_dense(sources, plan, bundle, report)["valid"])
        altered = copy.deepcopy(plan)
        altered["excluded_vector_hashes"] = []
        altered = _seal({key: value for key, value in altered.items() if key != "plan_sha256"}, "plan_sha256")
        self.assertFalse(verify_dense(sources, altered, bundle, report)["valid"])
        changed = copy.deepcopy(report)
        changed["observations"][0]["output_bits"][0][29][639] ^= 1
        changed = _seal({key: value for key, value in changed.items() if key != "report_sha256"}, "report_sha256")
        self.assertFalse(verify_dense(sources, plan, bundle, changed)["valid"])

    def test_cli_output_and_source_guards(self) -> None:
        from bioprocess_runtime.cli import build_parser, command_dense_k128

        parser = build_parser()
        for operation in ("plan", "run", "verify"):
            argv = ["gemma-k128-dense-" + operation, "--bundle", "bundle"]
            argv += ["--report", "report", "--reexecute"] if operation == "verify" else ["--output", "output"]
            self.assertEqual(parser.parse_args(argv).operation, operation)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "source.json"
            output.write_text("preserve", encoding="utf-8")
            with self.assertRaises(ValueError):
                command_dense_k128(Namespace(operation="plan", output=output, bundle=Path(directory) / "bundle"))
            self.assertEqual(output.read_text(encoding="utf-8"), "preserve")
        with patch("bioprocess_runtime.gemma_k128_dense.Path.read_bytes", return_value=b"changed"):
            with self.assertRaises(ValueError):
                _code_sha()
