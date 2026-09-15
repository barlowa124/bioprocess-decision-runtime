from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path
from unittest.mock import patch

from bioprocess_runtime.gemma_k2048_probes import candidates, candidate_predictions, probe_pool, select_probes, FAMILIES, _code_sha, _worker_init, _worker_predictions, _sources
from bioprocess_runtime.gemma_float_semantics import decode_finite_bfloat16

ROOT = Path(__file__).resolve().parent.parent


class K2048ProbeTests(unittest.TestCase):
    def test_grid_and_merge_discrimination(self) -> None:
        grid = candidates()
        self.assertEqual(len(grid), 256)
        self.assertEqual(len({item["id"] for item in grid}), 256)
        self.assertTrue(all(sum(stop - start for start, stop in item["partitions"]) == 2048 for item in grid))
        right = [0] * 2048
        right[0], right[64], right[128] = 0x4E80, 0x3F80, 0xCE80
        prediction = dict(zip((item["id"] for item in grid), candidate_predictions([0x3F80] * 2048, right)))
        self.assertEqual(prediction["k64:bfloat16_rne:exact"], 0x3F80)
        self.assertEqual(prediction["k64:bfloat16_rne:sequential_float32_rne"], 0)
        with self.assertRaises(ValueError):
            candidate_predictions([0] * 1024, [0] * 1024)

    def test_pool_nonunit_inputs_and_exact_cancellation(self) -> None:
        left, pool = probe_pool()
        self.assertEqual((left, pool), probe_pool())
        self.assertEqual(len(left), 2048)
        self.assertTrue(all(bits & 127 for bits in left))
        self.assertEqual(len(pool), 256)
        for item in pool[:64]:
            terms = [decode_finite_bfloat16(a)[0] * decode_finite_bfloat16(b)[0] for a, b in zip(left, item["right_bits"]) if b]
            self.assertEqual(len(terms), 3)
            self.assertTrue(any(a + b == 0 for a, b in combinations(terms, 2)))
        predictions = [[(index + candidate) % 8 for candidate in range(256)] for index in range(256)]
        selected, separated, groups = select_probes(pool, predictions)
        self.assertEqual(len(set(selected)), 128)
        self.assertEqual(len(groups), 8)
        self.assertGreater(separated, 0)
        for family in FAMILIES:
            self.assertEqual(sum(pool[index]["family"] == family for index in selected), 32)

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"), "Gemma dependencies are unavailable")
    def test_worker_and_source_kernel_binding(self) -> None:
        pair = ([0x3F80] * 2048, [0] * 2048)
        with ProcessPoolExecutor(max_workers=1, initializer=_worker_init, initargs=(_code_sha(),)) as pool:
            result = pool.submit(_worker_predictions, pair).result(timeout=120)
        self.assertEqual(result, [0] * 256)
        load = lambda path: json.loads((ROOT / path).read_text(encoding="utf-8"))
        names, exclusions = _sources(load("results/gemma3_270m_reduction_backend_binding.json"), load("results/gemma3_270m_mlp_product_summary.json"))
        self.assertTrue(any("ampere_bf16_s16816" in name for name in names))
        self.assertTrue(any("splitKreduce" in name for name in names))
        self.assertTrue(exclusions)

    def test_last_coordinate_and_geometry_scope(self) -> None:
        import numpy as np
        from bioprocess_runtime.gemma_attention_entry import _descriptor
        from bioprocess_runtime.gemma_k2048_probes import _matrices, report_from_observations

        bundle = {"left_bits": [0x3F80] * 2048, "cases": [{"right_bits": [0] * 2048} for _ in range(128)]}
        plan = {"plan_sha256": "a" * 64, "candidates": candidates(), "cases": [{"prediction_bits": [0] * 256} for _ in range(128)], "runtime": {}, "expected_kernel_names": ["synthetic"]}
        left, right = _matrices(bundle)
        record = {"input": {"tensor": {**_descriptor(left), "device": "cuda:0"}, "strides": [61440, 2048, 1]},
                  "weight": {"tensor": {**_descriptor(right), "device": "cuda:0"}, "strides": [2048, 1]},
                  "output_bits": np.zeros((1, 30, 640), dtype=np.uint16).tolist(), "kernel_names": ["synthetic"]}
        matching = report_from_observations(plan, bundle, [record] * 3, {})
        self.assertTrue(matching["survivors_supported_in_declared_scope"])
        self.assertFalse(matching["unique_survivor_in_frozen_grid"])
        self.assertFalse(matching["global_exactness_activation_allowed"])
        record["kernel_names"] = ["other"]
        changed_kernel = report_from_observations(plan, bundle, [record] * 3, {})
        self.assertFalse(changed_kernel["survivors_supported_in_declared_scope"])
        record["kernel_names"] = ["synthetic"]
        record["output_bits"][0][29][639] = 1
        failed = report_from_observations(plan, bundle, [record] * 3, {})
        self.assertEqual(failed["surviving_candidates"], [])
        self.assertFalse(failed["duplicate_coordinates_match"])
        self.assertTrue(all(item["mismatch_counts"] == [1, 1, 1] for item in failed["candidate_results"]))
        self.assertEqual(failed["candidate_results"][0]["first_mismatches"][0]["coordinate"], [0, 29, 639])
        record["weight"]["strides"] = [1, 640]
        with self.assertRaises(ValueError):
            report_from_observations(plan, bundle, [record] * 3, {})

    def test_real_evidence_tamper_and_scope_guards(self) -> None:
        from bioprocess_runtime.gemma_k2048_probes import verify_probes
        from bioprocess_runtime.gemma_rotary_slice import _seal

        if not (ROOT / "artifacts/gemma3_270m_k2048_probes_report.json").exists():
            self.skipTest("Tensor-rich K2048 observations intentionally remain outside Git")
        load = lambda name: json.loads((ROOT / name).read_text(encoding="utf-8"))
        binding = load("results/gemma3_270m_reduction_backend_binding.json")
        product = load("results/gemma3_270m_mlp_product_summary.json")
        plan = load("results/gemma3_270m_k2048_probes_plan.json")
        bundle = load("artifacts/gemma3_270m_k2048_probes_inputs.json")
        report = load("artifacts/gemma3_270m_k2048_probes_report.json")
        checked = verify_probes(binding, product, plan, bundle, report)
        self.assertTrue(checked["valid"])
        self.assertEqual(checked["surviving_candidates"], ["k192:bfloat16_rne:sequential_float32_rne"])
        changed = copy.deepcopy(report)
        changed["observations"][0]["output_bits"][0][29][639] ^= 1
        changed = _seal({key: value for key, value in changed.items() if key != "report_sha256"}, "report_sha256")
        self.assertFalse(verify_probes(binding, product, plan, bundle, changed)["valid"])
        for field, value in (("partial_arithmetic_transfer_prequalified", True), ("runtime", {}), ("expected_kernel_names", ["other"]), ("candidate_pairs_separated", 0), ("global_exactness_activation_allowed", True)):
            promoted = _seal({**{key: item for key, item in plan.items() if key != "plan_sha256"}, field: value}, "plan_sha256")
            self.assertFalse(verify_probes(binding, product, promoted, bundle, report)["valid"], field)
        bad_binding = copy.deepcopy(binding)
        bad_binding["records"][0]["cuda_kernel_names"] = ["other"]
        self.assertFalse(verify_probes(bad_binding, product, plan, bundle, report)["valid"])
        bundle["cases"][0]["right_bits"][-1] ^= 1
        self.assertFalse(verify_probes(binding, product, plan, bundle, report)["valid"])

    def test_cli_output_and_source_guards(self) -> None:
        from bioprocess_runtime.cli import build_parser, command_k2048_probes
        parser = build_parser()
        for operation in ("plan", "run", "verify"):
            argv = ["gemma-k2048-probes-" + operation, "--bundle", "bundle"]
            argv += ["--report", "report", "--reexecute"] if operation == "verify" else ["--output", "output"]
            args = parser.parse_args(argv)
            self.assertEqual(args.operation, operation)
            self.assertEqual(args.workers, 4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.json"
            path.write_text("preserve", encoding="utf-8")
            with self.assertRaises(ValueError):
                command_k2048_probes(Namespace(operation="plan", output=path, bundle=Path(directory) / "bundle"))
            self.assertEqual(path.read_text(encoding="utf-8"), "preserve")
        with patch("bioprocess_runtime.gemma_k2048_probes.Path.read_bytes", return_value=b"changed"):
            with self.assertRaises(ValueError):
                _code_sha()
