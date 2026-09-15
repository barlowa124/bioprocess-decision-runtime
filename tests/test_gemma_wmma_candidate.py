from __future__ import annotations

import copy
from fractions import Fraction
import hashlib
import json
import unittest
from unittest.mock import patch
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from bioprocess_runtime.serialization import canonical_json


ROOT = Path(__file__).resolve().parent.parent


def _load(name: str) -> dict:
    return json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class GemmaWmmaCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.program = _load("gemma3_270m_execution_ir.json")
        cls.reduction = _load("gemma3_270m_reduction_characterization.json")
        cls.suite = _load("gemma3_270m_nsight_kernel_suite.json")
        cls.backend = _load("gemma3_270m_reduction_backend_binding.json")
        cls.position = _load("gemma3_270m_wmma_probe.json")
        cls.accumulator = _load("gemma3_270m_wmma_accumulator_probe.json")
        cls.magnitude = _load("gemma3_270m_wmma_magnitude_probe.json")
        cls.search = _load("gemma3_270m_wmma_candidate_search.json")

    def test_unique_probe_consistent_candidate(self) -> None:
        from bioprocess_runtime.gemma_wmma_candidate import (
            verify_wmma_candidate_search,
        )

        verification = verify_wmma_candidate_search(
            self.program,
            self.reduction,
            self.suite,
            self.backend,
            self.position,
            self.accumulator,
            self.magnitude,
            self.search,
        )
        self.assertTrue(verification["valid"], verification)
        self.assertEqual(verification["profile_count"], 64)
        self.assertEqual(verification["total_records"], 5384)
        self.assertEqual(verification["best_total_matches"], 5384)
        self.assertEqual(verification["complete_matching_profile_count"], 1)
        self.assertTrue(verification["unique_all_matching_candidate_in_search_space"])
        self.assertFalse(verification["complete_numeric_transition_established"])
        self.assertEqual(self.search["complete_matching_profiles"], [
            {
                "precision_bits": 25,
                "rounding_mode": "toward_zero",
                "half_order": "lower_then_upper",
            }
        ])
        self.assertEqual(self.search["mismatch_frontier"], [])

    def test_holdout_plan_freezes_dense_predictions(self) -> None:
        from bioprocess_runtime.gemma_wmma_candidate import build_k16_holdout_plan, k16_half_transition_bits

        for malformed in (None, [], {}, {"search_sha256": "bad"}):
            with self.assertRaises(ValueError):
                build_k16_holdout_plan(malformed)
        plan = build_k16_holdout_plan(self.search)
        self.assertEqual(plan, build_k16_holdout_plan(self.search))
        self.assertEqual(len(plan["cases"]), 1024)
        self.assertEqual(len({tuple(case["value_bits"]) for case in plan["cases"]}), 1024)
        self.assertFalse(plan["candidate_refitting_allowed"])
        for case in plan["cases"]:
            self.assertEqual(len(case["value_bits"]), 16)
            self.assertTrue(all(bits & 0x7F80 != 0 and bits & 0x7F80 != 0x7F80 for bits in case["value_bits"]))
            self.assertEqual(case["predicted_bits"], k16_half_transition_bits(case["value_bits"], 25, "toward_zero", "lower_then_upper"))

    def test_holdout_failure_is_valid_evidence_not_conformance(self) -> None:
        from bioprocess_runtime.gemma_wmma_candidate import build_k16_holdout_plan, _holdout_report, verify_k16_holdout

        plan = build_k16_holdout_plan(self.search)
        predicted = [case["predicted_bits"] for case in plan["cases"]]
        observed = [predicted.copy() for _ in range(3)]
        observed[0][0] ^= 1
        report = _holdout_report(plan, observed, [["synthetic_kernel"]] * 3, {})
        verified = verify_k16_holdout(plan, self.search, report)
        self.assertTrue(verified["valid"])
        self.assertFalse(verified["candidate_passes_holdout"])
        self.assertEqual(verified["mismatch_count"], 1)
        self.assertIsNone(verified["reexecution_exact"])
        damaged = copy.deepcopy(report)
        damaged["candidate_passes_holdout"] = True
        damaged["report_sha256"] = _sha256({k: v for k, v in damaged.items() if k != "report_sha256"})
        self.assertFalse(verify_k16_holdout(plan, self.search, damaged)["valid"])
        plan["cases"][0]["predicted_bits"] ^= 1
        self.assertFalse(verify_k16_holdout(plan, self.search, report)["valid"])

    def test_holdout_cli_preserves_failed_acquisition(self) -> None:
        from bioprocess_runtime.cli import command_k16_holdout_run, command_k16_holdout_verify
        from bioprocess_runtime.gemma_wmma_candidate import build_k16_holdout_plan, _holdout_report

        plan = build_k16_holdout_plan(self.search)
        observed = [[case["predicted_bits"] for case in plan["cases"]] for _ in range(3)]
        observed[0][0] ^= 1
        report = _holdout_report(plan, observed, [["synthetic_kernel"]] * 3, {})
        args = Namespace(search=Path("search"), plan=Path("plan"), report=Path("report"), output=Path("output"), reexecute=False)
        with patch.object(Path, "read_text", side_effect=[json.dumps(self.search), json.dumps(plan)]), patch("bioprocess_runtime.gemma_wmma_candidate.acquire_k16_holdout", return_value=report), patch("bioprocess_runtime.cli._write_json") as writer:
            self.assertEqual(command_k16_holdout_run(args), 1)
            writer.assert_called_once_with(args.output, report)
        with patch.object(Path, "read_text", side_effect=[json.dumps(self.search), json.dumps(plan), json.dumps(report)]), redirect_stdout(StringIO()):
            self.assertEqual(command_k16_holdout_verify(args), 0)

    def test_serial_composition_carries_across_fragments(self) -> None:
        from bioprocess_runtime.gemma_wmma_candidate import serial_k8_composition_bits, k16_half_transition_bits

        values = [0] * 32
        values[0], values[16] = 0x3F80, 0x3F80
        self.assertEqual(serial_k8_composition_bits(values), 0x4000)
        values[0], values[16], values[24] = 0x4E80, 0xCE80, 0x3F80
        self.assertEqual(serial_k8_composition_bits(values), 0x3F80)
        self.assertEqual(serial_k8_composition_bits(values[:16]), k16_half_transition_bits(values[:16], 25, "toward_zero", "lower_then_upper"))
        for invalid in ([], [0] * 17, [0x7F80] * 16):
            with self.assertRaises(ValueError):
                serial_k8_composition_bits(invalid)

    def test_composition_plan_binds_vectors_and_predictions(self) -> None:
        from bioprocess_runtime.gemma_wmma_candidate import build_composition_holdout_plan, _composition_vectors, serial_k8_composition_bits

        plan = build_composition_holdout_plan(self.search)
        self.assertEqual(plan, build_composition_holdout_plan(self.search))
        self.assertEqual(len(plan["cases"]), 1024)
        self.assertFalse(plan["candidate_refitting_allowed"])
        self.assertTrue(plan["cross_fragment_composition_tested"])
        vectors = _composition_vectors()
        self.assertEqual(len({_sha256(values) for _, values in vectors}), 1024)
        for case, (family, values) in zip(plan["cases"], vectors):
            self.assertEqual(case["family"], family)
            self.assertEqual(case["input_bits_sha256"], _sha256(values))
            self.assertGreaterEqual(len(case["active_k16_blocks"]), 2)
        for index in (0, 256, 512, 768):
            self.assertEqual(plan["cases"][index]["predicted_bits"], serial_k8_composition_bits(vectors[index][1]))

    def test_composition_rejects_refitting_but_accepts_failed_evidence(self) -> None:
        from bioprocess_runtime.gemma_wmma_candidate import build_composition_holdout_plan, _holdout_report, verify_composition_holdout, COMPOSITION_FAMILIES, COMPOSITION_SCOPE

        plan = build_composition_holdout_plan(self.search)
        observed = [[case["predicted_bits"] for case in plan["cases"]] for _ in range(3)]
        observed[0][0] ^= 1
        report = _holdout_report(plan, observed, [["synthetic_kernel"]] * 3, {}, COMPOSITION_FAMILIES, COMPOSITION_SCOPE)
        result = verify_composition_holdout(plan, self.search, report)
        self.assertTrue(result["valid"])
        self.assertFalse(result["candidate_passes_holdout"])
        self.assertEqual(result["mismatch_count"], 1)
        report["candidate_passes_holdout"] = True
        report["report_sha256"] = _sha256({k: v for k, v in report.items() if k != "report_sha256"})
        self.assertFalse(verify_composition_holdout(plan, self.search, report)["valid"])
        from bioprocess_runtime.gemma_wmma_candidate import acquire_composition_holdout

        with patch("bioprocess_runtime.gemma_wmma_candidate.build_composition_holdout_plan", return_value=plan), patch("bioprocess_runtime.gemma_wmma_candidate._composition_vectors", return_value=[("adjacent_carry", [0] * 640)] * 1024), patch("bioprocess_runtime.gemma_wmma_candidate._acquire_holdout_vectors") as acquire:
            with self.assertRaisesRegex(ValueError, "committed hashes"):
                acquire_composition_holdout(plan, self.search)
            acquire.assert_not_called()

    def test_recorded_composition_holdout_preserves_failure(self) -> None:
        from bioprocess_runtime.gemma_wmma_candidate import verify_composition_holdout

        plan = _load("gemma3_270m_composition_holdout_plan.json")
        report = _load("gemma3_270m_composition_holdout.json")
        verified = verify_composition_holdout(plan, self.search, report)
        self.assertTrue(verified["valid"], verified)
        self.assertFalse(verified["candidate_passes_holdout"])
        self.assertEqual(verified["mismatch_count"], 177)
        self.assertEqual(len({item["index"] for item in report["mismatches"]}), 59)
        self.assertTrue(verified["repeated_outputs_identical"])
        self.assertEqual(verified["matching_cases_by_family"], {
            "adjacent_carry": [256] * 3, "dense_k640": [256] * 3,
            "distant_cancellation": [212] * 3, "k128_boundary": [241] * 3,
        })
        damaged_plan = copy.deepcopy(plan)
        damaged_plan["composition"]["inter_half_state"] = "float32 RNE"
        damaged_plan["plan_sha256"] = _sha256({k: v for k, v in damaged_plan.items() if k != "plan_sha256"})
        self.assertFalse(verify_composition_holdout(damaged_plan, self.search, report)["valid"])

    def test_composition_cli_preserves_failed_acquisition(self) -> None:
        from bioprocess_runtime.cli import command_k16_holdout_run, command_k16_holdout_verify

        plan = _load("gemma3_270m_composition_holdout_plan.json")
        report = _load("gemma3_270m_composition_holdout.json")
        args = Namespace(search=Path("search"), plan=Path("plan"), report=Path("report"), output=Path("output"), reexecute=False, composition=True)
        with patch.object(Path, "read_text", side_effect=[json.dumps(self.search), json.dumps(plan)]), patch("bioprocess_runtime.gemma_wmma_candidate.acquire_composition_holdout", return_value=report), patch("bioprocess_runtime.cli._write_json") as writer:
            self.assertEqual(command_k16_holdout_run(args), 1)
            writer.assert_called_once_with(args.output, report)
        with patch.object(Path, "read_text", side_effect=[json.dumps(self.search), json.dumps(plan), json.dumps(report)]), redirect_stdout(StringIO()):
            self.assertEqual(command_k16_holdout_verify(args), 0)

    def test_float32_carry_rounding_variants(self) -> None:
        from bioprocess_runtime.gemma_wmma_candidate import _float32_carry, serial_k8_float32_carry_bits

        value = Fraction((1 << 30) + 96)
        self.assertEqual(_float32_carry(value, "nearest_even"), (1 << 30) + 128)
        self.assertEqual(_float32_carry(value, "toward_zero"), 1 << 30)
        self.assertEqual(_float32_carry(-value, "nearest_even"), -((1 << 30) + 128))
        self.assertEqual(_float32_carry(-value, "toward_zero"), -(1 << 30))
        self.assertEqual(_float32_carry(Fraction((1 << 30) + 64), "nearest_even"), 1 << 30)
        self.assertEqual(_float32_carry(Fraction(1, 1 << 149), "toward_zero"), Fraction(1, 1 << 149))
        self.assertEqual(_float32_carry(Fraction(1, 1 << 150), "nearest_even"), 0)
        self.assertEqual(_float32_carry(-Fraction(1, 1 << 149), "toward_zero"), -Fraction(1, 1 << 149))
        self.assertEqual(_float32_carry(-Fraction(3, 1 << 150), "toward_zero"), -Fraction(1, 1 << 149))
        self.assertEqual(_float32_carry(Fraction((1 << 30) - 32), "toward_zero"), (1 << 30) - 64)
        self.assertEqual(_float32_carry(-Fraction((1 << 30) - 32), "toward_zero"), -((1 << 30) - 64))
        for mode in ("nearest_even", "toward_zero"):
            for interval in (8, 16):
                self.assertEqual(serial_k8_float32_carry_bits([0x3F80] * 32, mode, interval), 0x4200)
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            _float32_carry(Fraction(1 << 129), "nearest_even")
        with self.assertRaises(ValueError):
            serial_k8_float32_carry_bits([0] * 16, "toward_zero", 7)

    def test_carry_diagnosis_is_development_only(self) -> None:
        from bioprocess_runtime.gemma_wmma_candidate import verify_carry_diagnosis

        plan = _load("gemma3_270m_composition_holdout_plan.json")
        report = _load("gemma3_270m_composition_holdout.json")
        diagnosis = _load("gemma3_270m_carry_diagnosis.json")
        result = verify_carry_diagnosis(self.search, plan, report, diagnosis)
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["all_matching_development_profiles"], [{"mode": "toward_zero", "interval": 8}])
        self.assertFalse(result["held_out_validation_established"])
        self.assertEqual([item["mismatch_count"] for item in result["profile_mismatch_counts"]], [96, 96, 0, 30])

    def test_carry_revision_freezes_disjoint_inputs(self) -> None:
        from bioprocess_runtime.gemma_wmma_candidate import build_carry_revision_plan, _composition_vectors, CARRY_REVISION_SEED

        diagnosis = _load("gemma3_270m_carry_diagnosis.json")
        plan = build_carry_revision_plan(self.search, diagnosis)
        self.assertEqual(plan["carry_revision"], {"mode": "toward_zero", "interval": 8})
        self.assertEqual(len(plan["cases"]), 1024)
        self.assertTrue(plan["development_vectors_disjoint"])
        self.assertFalse(plan["candidate_refitting_allowed"])
        previous = {_sha256(values) for _, values in _composition_vectors()}
        fresh = {_sha256(values) for _, values in _composition_vectors(CARRY_REVISION_SEED)}
        self.assertEqual(len(fresh), 1024)
        self.assertFalse(previous & fresh)
        self.assertEqual(fresh, {case["input_bits_sha256"] for case in plan["cases"]})

    def test_carry_revision_report_and_refit_rejection(self) -> None:
        from bioprocess_runtime.gemma_wmma_candidate import verify_carry_revision_holdout

        diagnosis = _load("gemma3_270m_carry_diagnosis.json")
        plan = _load("gemma3_270m_carry_revision_plan.json")
        report = _load("gemma3_270m_carry_revision_holdout.json")
        verified = verify_carry_revision_holdout(plan, self.search, diagnosis, report)
        self.assertTrue(verified["valid"], verified)
        self.assertTrue(verified["candidate_passes_holdout"])
        self.assertEqual(verified["mismatch_count"], 0)
        self.assertFalse(verified["global_exactness_activation_allowed"])
        self.assertIsNone(verified["reexecution_exact"])
        mutated = copy.deepcopy(plan)
        mutated["carry_revision"]["interval"] = 16
        mutated["plan_sha256"] = _sha256({key: value for key, value in mutated.items() if key != "plan_sha256"})
        self.assertFalse(verify_carry_revision_holdout(mutated, self.search, diagnosis, report)["valid"])
        mutated_report = copy.deepcopy(report)
        mutated_report["observed_bits"][0][0] ^= 1
        mutated_report["report_sha256"] = _sha256({key: value for key, value in mutated_report.items() if key != "report_sha256"})
        self.assertFalse(verify_carry_revision_holdout(plan, self.search, diagnosis, mutated_report)["valid"])

    def test_carry_revision_cli_writes_failure_and_verifies_evidence(self) -> None:
        from bioprocess_runtime.cli import command_carry_revision
        from bioprocess_runtime.gemma_wmma_candidate import _holdout_report, COMPOSITION_FAMILIES, CARRY_REVISION_SCOPE

        diagnosis = _load("gemma3_270m_carry_diagnosis.json")
        plan = _load("gemma3_270m_carry_revision_plan.json")
        observed = [[case["predicted_bits"] for case in plan["cases"]] for _ in range(3)]
        observed[0][0] ^= 1
        report = _holdout_report(plan, observed, [["synthetic_kernel"]] * 3, {}, COMPOSITION_FAMILIES, CARRY_REVISION_SCOPE)
        args = Namespace(search=Path("search"), diagnosis=Path("diagnosis"), plan=Path("plan"), report=Path("report"), output=Path("output"), operation="run", reexecute=False)
        with patch.object(Path, "read_text", side_effect=[json.dumps(self.search), json.dumps(diagnosis), json.dumps(plan)]), patch("bioprocess_runtime.gemma_wmma_candidate.acquire_carry_revision_holdout", return_value=report), patch("bioprocess_runtime.cli._write_json") as writer:
            self.assertEqual(command_carry_revision(args), 1)
            writer.assert_called_once_with(args.output, report)
        args.operation = "verify"
        with patch.object(Path, "read_text", side_effect=[json.dumps(self.search), json.dumps(diagnosis), json.dumps(plan), json.dumps(report)]), redirect_stdout(StringIO()):
            self.assertEqual(command_carry_revision(args), 0)

    def test_nonunit_product_oracle(self) -> None:
        from bioprocess_runtime.gemma_wmma_candidate import serial_k8_product_bits, serial_k8_float32_carry_bits

        left = [0x3FC0] * 16
        right = [0x4000] * 16
        self.assertEqual(serial_k8_product_bits(left, right), 0x4240)
        self.assertEqual(serial_k8_product_bits(left, [0x3F80] * 16), serial_k8_float32_carry_bits(left, "toward_zero", 8))
        self.assertEqual(serial_k8_product_bits(left, right), serial_k8_product_bits(right, left))
        for a, b in (([], []), ([0] * 16, [0] * 32), ([0x7F80] * 16, right)):
            with self.assertRaises(ValueError):
                serial_k8_product_bits(a, b)

    def test_product_plan_binds_nonunit_vectors_and_shapes(self) -> None:
        from bioprocess_runtime.gemma_wmma_candidate import build_product_holdout_plan, _product_vectors, PRODUCT_SHAPES, serial_k8_product_bits

        plan = build_product_holdout_plan(self.search, _load("gemma3_270m_carry_diagnosis.json"))
        self.assertFalse(plan["candidate_refitting_allowed"])
        self.assertEqual(sum(len(item["cases"]) for item in plan["shapes"]), 384)
        for shape, (left, cases), (role, rows, outputs, inner) in zip(plan["shapes"], _product_vectors(), PRODUCT_SHAPES):
            self.assertEqual(shape["input_shape"], [rows, inner])
            self.assertEqual(shape["weight_shape"], [outputs, inner])
            self.assertEqual(shape["role"], role)
            self.assertTrue(all(bits & 127 for bits in left))
            self.assertEqual(shape["left_bits_sha256"], _sha256(left))
            for item, (family, right) in zip(shape["cases"], cases):
                self.assertEqual(item["family"], family)
                self.assertEqual(item["right_bits_sha256"], _sha256(right))
            self.assertEqual(shape["cases"][0]["predicted_bits"], serial_k8_product_bits(left, cases[0][1]))

    def test_product_holdout_preserves_replayed_failures(self) -> None:
        from bioprocess_runtime.gemma_wmma_candidate import verify_product_holdout

        diagnosis = _load("gemma3_270m_carry_diagnosis.json")
        plan = _load("gemma3_270m_product_holdout_plan.json")
        report = _load("gemma3_270m_product_holdout.json")
        result = verify_product_holdout(plan, self.search, diagnosis, report)
        self.assertTrue(result["valid"], result)
        self.assertFalse(result["candidate_passes_holdout"])
        self.assertEqual(result["mismatch_count"], 303)
        self.assertEqual([item["mismatch_count"] for item in result["shape_results"]], [15, 171, 117])
        self.assertEqual(result["tested_case_count"], 384)
        for item, acquired, expected_matches in zip(result["shape_results"], report["acquisitions"], (123, 71, 89)):
            self.assertTrue(item["repeated_outputs_identical"])
            self.assertEqual(acquired["observed_bits"][0], acquired["observed_bits"][1])
            self.assertEqual(acquired["observed_bits"][0], acquired["observed_bits"][2])
            for repetition in range(3):
                self.assertEqual(sum(counts[repetition] for counts in item["matching_cases_by_family"].values()), expected_matches)
        mutated = copy.deepcopy(report)
        mutated["candidate_passes_holdout"] = True
        mutated["report_sha256"] = _sha256({key: value for key, value in mutated.items() if key != "report_sha256"})
        self.assertFalse(verify_product_holdout(plan, self.search, diagnosis, mutated)["valid"])
        mutated_plan = copy.deepcopy(plan)
        mutated_plan["shapes"][0]["weight_shape"] = [128, 640]
        mutated_plan["plan_sha256"] = _sha256({key: value for key, value in mutated_plan.items() if key != "plan_sha256"})
        self.assertFalse(verify_product_holdout(mutated_plan, self.search, diagnosis, report)["valid"])

    def test_product_holdout_cli_distinguishes_integrity_from_conformance(self) -> None:
        from bioprocess_runtime.cli import command_carry_revision

        diagnosis = _load("gemma3_270m_carry_diagnosis.json")
        plan = _load("gemma3_270m_product_holdout_plan.json")
        report = _load("gemma3_270m_product_holdout.json")
        args = Namespace(search=Path("search"), diagnosis=Path("diagnosis"), plan=Path("plan"), report=Path("report"), output=Path("output"), operation="run", product=True, reexecute=False)
        with patch.object(Path, "read_text", side_effect=[json.dumps(self.search), json.dumps(diagnosis), json.dumps(plan)]), patch("bioprocess_runtime.gemma_wmma_candidate.acquire_product_holdout", return_value=report), patch("bioprocess_runtime.cli._write_json") as writer:
            self.assertEqual(command_carry_revision(args), 1)
            writer.assert_called_once_with(args.output, report)
        args.operation = "verify"
        with patch.object(Path, "read_text", side_effect=[json.dumps(self.search), json.dumps(diagnosis), json.dumps(plan), json.dumps(report)]), redirect_stdout(StringIO()):
            self.assertEqual(command_carry_revision(args), 0)

    def test_matched_product_exact_rescaling(self) -> None:
        from bioprocess_runtime.gemma_wmma_candidate import _scale_normal_bfloat_bits
        from bioprocess_runtime.gemma_float_semantics import decode_finite_bfloat16

        for bits in (0x3F81, 0xBF81, 0x4083, 0, 0x8000):
            doubled = _scale_normal_bfloat_bits(bits, 1)
            self.assertEqual(decode_finite_bfloat16(doubled)[0], 2 * decode_finite_bfloat16(bits)[0])
            self.assertEqual(_scale_normal_bfloat_bits(doubled, -1), bits)
        for bits, shift in ((0x0001, 1), (0x7F7F, 1), (0x0080, -1), (0x7F80, 0)):
            with self.assertRaises(ValueError):
                _scale_normal_bfloat_bits(bits, shift)

    def test_matched_product_plan_preserves_products_and_predictions(self) -> None:
        from bioprocess_runtime.gemma_wmma_candidate import build_matched_product_plan

        plan = build_matched_product_plan(self.search, _load("gemma3_270m_carry_diagnosis.json"))
        self.assertEqual(len(plan["shapes"]), 6)
        self.assertTrue(plan["source_query_vectors_reused"])
        self.assertTrue(plan["exact_products_identical_ignoring_trailing_zeros"])
        self.assertTrue(plan["all_candidate_predictions_identical"])
        self.assertFalse(plan["fresh_holdout_validation_established"])
        expected = [case["predicted_bits"] for case in plan["shapes"][0]["cases"]]
        for shape in plan["shapes"]:
            self.assertEqual([case["predicted_bits"] for case in shape["cases"]], expected)

    def test_matched_product_report_separates_rescaling_from_shape(self) -> None:
        from bioprocess_runtime.gemma_wmma_candidate import verify_matched_products

        diagnosis = _load("gemma3_270m_carry_diagnosis.json")
        plan = _load("gemma3_270m_matched_products_plan.json")
        report = _load("gemma3_270m_matched_products.json")
        verified = verify_matched_products(plan, self.search, diagnosis, report)
        self.assertTrue(verified["valid"], verified)
        self.assertFalse(verified["candidate_passes_controlled_comparison"])
        self.assertFalse(verified["fresh_holdout_validation_established"])
        self.assertEqual([item["difference_count"] for item in verified["comparisons"]], [0, 0, 0, 204, 204, 126, 126])
        self.assertTrue(all(item["repeated_outputs_identical"] for item in report["shape_results"]))
        damaged = copy.deepcopy(report)
        damaged["comparisons"][3]["difference_count"] = 0
        damaged["report_sha256"] = _sha256({k: v for k, v in damaged.items() if k != "report_sha256"})
        self.assertFalse(verify_matched_products(plan, self.search, diagnosis, damaged)["valid"])
        damaged_plan = copy.deepcopy(plan)
        damaged_plan["shapes"][1]["rescaling_exponent"] = 2
        damaged_plan["plan_sha256"] = _sha256({k: v for k, v in damaged_plan.items() if k != "plan_sha256"})
        self.assertFalse(verify_matched_products(damaged_plan, self.search, diagnosis, report)["valid"])

    def test_matched_product_cli_preserves_controlled_failure(self) -> None:
        from bioprocess_runtime.cli import command_carry_revision

        diagnosis = _load("gemma3_270m_carry_diagnosis.json")
        plan = _load("gemma3_270m_matched_products_plan.json")
        report = _load("gemma3_270m_matched_products.json")
        args = Namespace(search=Path("search"), diagnosis=Path("diagnosis"), plan=Path("plan"), report=Path("report"), output=Path("output"), operation="run", matched=True, reexecute=False)
        with patch.object(Path, "read_text", side_effect=[json.dumps(self.search), json.dumps(diagnosis), json.dumps(plan)]), patch("bioprocess_runtime.gemma_wmma_candidate.acquire_matched_products", return_value=report), patch("bioprocess_runtime.cli._write_json") as writer:
            self.assertEqual(command_carry_revision(args), 1)
            writer.assert_called_once_with(args.output, report)
        args.operation = "verify"
        with patch.object(Path, "read_text", side_effect=[json.dumps(self.search), json.dumps(diagnosis), json.dumps(plan), json.dumps(report)]), redirect_stdout(StringIO()):
            self.assertEqual(command_carry_revision(args), 0)

    def test_query_reduction_protocol_binds_five_source_cases(self) -> None:
        from bioprocess_runtime.gemma_wmma_candidate import build_query_reduction_plan

        plan = build_query_reduction_plan(self.search, _load("gemma3_270m_carry_diagnosis.json"), _load("gemma3_270m_product_holdout_plan.json"), _load("gemma3_270m_product_holdout.json"))
        self.assertEqual(plan["case_indices"], [69, 75, 98, 103, 119])
        self.assertFalse(plan["candidate_refitting_allowed"])
        self.assertFalse(plan["fresh_holdout_validation_established"])
        self.assertEqual(plan["weight_shape"], [1024, 640])

    def test_query_reduction_journals_predictions_and_reaches_singleton_fixed_point(self) -> None:
        from bioprocess_runtime.gemma_wmma_candidate import _query_reduction_engine, serial_k8_product_bits, QUERY_CHUNKS

        left, right = [0x3F80] * 640, [0x3F80] * 4 + [0] * 636
        plan = {"case_indices": [0], "chunk_sizes": list(QUERY_CHUNKS), "max_trials_per_case": 256, "plan_sha256": "synthetic"}
        baseline = {"observed_bits": [[0] * 128 for _ in range(3)], "kernel_names": [["synthetic"] for _ in range(3)], "environment": {}}
        journal = []

        def observe(index: int, values: list[int]) -> dict:
            predicted = serial_k8_product_bits(left, values)
            self.assertEqual(journal[-1]["pending"]["predicted_bits"], predicted)
            self.assertEqual(journal[-1]["pending"]["right_bits_sha256"], _sha256(values))
            result = predicted ^ 1 if sum(bool(bits) for bits in values) >= 2 else predicted
            return {"bits": [result] * 3, "kernel_names": baseline["kernel_names"], "environment": {}, "background_differences": []}

        with patch("bioprocess_runtime.gemma_wmma_candidate._product_vectors", return_value=[(left, [("synthetic", right)] * 128)]):
            report = _query_reduction_engine(plan, baseline, observe, lambda pending: journal.append(copy.deepcopy(pending)))
        final = report["final_cases"][0]
        self.assertEqual(len(final["terms"]), 2)
        self.assertTrue(final["singleton_deletion_irreducible_on_observed_trials"])
        self.assertFalse(final["cardinality_minimal_proven"])
        self.assertIsNone(journal[-1]["pending"])
        self.assertEqual(len(journal), 2 * report["trial_count"])

    def test_query_reduction_rejects_changed_kernel_and_incomplete_budget(self) -> None:
        from bioprocess_runtime.gemma_wmma_candidate import _query_reduction_engine

        left, right = [0x3F80] * 640, [0x3F80] * 2 + [0] * 638
        plan = {"case_indices": [0], "chunk_sizes": [1], "max_trials_per_case": 1, "plan_sha256": "synthetic"}
        baseline = {"observed_bits": [[0] * 128 for _ in range(3)], "kernel_names": [["original"]] * 3, "environment": {}}
        with patch("bioprocess_runtime.gemma_wmma_candidate._product_vectors", return_value=[(left, [("synthetic", right)] * 128)]):
            report = _query_reduction_engine(plan, baseline, lambda index, values: {"bits": [0] * 3, "kernel_names": [["changed"]] * 3, "environment": {}, "background_differences": []})
        self.assertFalse(report["trials"][0]["accepted"])
        self.assertEqual(len(report["final_cases"][0]["terms"]), 2)
        self.assertTrue(report["final_cases"][0]["budget_exhausted"])
        self.assertFalse(report["final_cases"][0]["singleton_deletion_irreducible_on_observed_trials"])

    def test_query_reduction_record_recomputation_and_tamper_rejection(self) -> None:
        from bioprocess_runtime.gemma_wmma_candidate import verify_query_reduction

        diagnosis = _load("gemma3_270m_carry_diagnosis.json")
        source_plan, source_report = _load("gemma3_270m_product_holdout_plan.json"), _load("gemma3_270m_product_holdout.json")
        plan, report = _load("gemma3_270m_query_reduction_plan.json"), _load("gemma3_270m_query_reduction.json")
        result = verify_query_reduction(plan, self.search, diagnosis, source_plan, source_report, report)
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["trial_count"], 224)
        self.assertEqual([len(item["terms"]) for item in result["final_cases"]], [3, 3, 11, 6, 5])
        for item in result["final_cases"]:
            self.assertTrue(item["singleton_deletion_irreducible_on_observed_trials"])
            self.assertFalse(item["cardinality_minimal_proven"])
            self.assertNotEqual(item["predicted_bits"], item["observed_bits"][0])
        damaged = copy.deepcopy(report)
        damaged["trials"][0]["accepted"] = not damaged["trials"][0]["accepted"]
        damaged["report_sha256"] = _sha256({k: v for k, v in damaged.items() if k != "report_sha256"})
        self.assertFalse(verify_query_reduction(plan, self.search, diagnosis, source_plan, source_report, damaged)["valid"])
        damaged = copy.deepcopy(report)
        damaged["trials"] = []
        self.assertFalse(verify_query_reduction(plan, self.search, diagnosis, source_plan, source_report, damaged)["valid"])
        damaged = copy.deepcopy(report)
        damaged["candidate_refitted"] = 0
        self.assertFalse(verify_query_reduction(plan, self.search, diagnosis, source_plan, source_report, damaged)["valid"])

    def test_query_reduction_cli_writes_diagnostic_and_protects_sources(self) -> None:
        from bioprocess_runtime.cli import command_query_reduction

        contents = [self.search, _load("gemma3_270m_carry_diagnosis.json"), _load("gemma3_270m_product_holdout_plan.json"), _load("gemma3_270m_product_holdout.json"), _load("gemma3_270m_query_reduction_plan.json")]
        report = _load("gemma3_270m_query_reduction.json")
        args = Namespace(search=Path("search"), diagnosis=Path("diagnosis"), source_plan=Path("source_plan"), source_report=Path("source_report"), plan=Path("plan"), output=Path("output"), journal=Path("journal.json"), operation="run")
        with patch.object(Path, "read_text", side_effect=[json.dumps(item) for item in contents]), patch.object(Path, "mkdir"), patch("bioprocess_runtime.gemma_wmma_candidate.acquire_query_reduction", return_value=report), patch("bioprocess_runtime.cli._write_json") as writer:
            self.assertEqual(command_query_reduction(args), 0)
            writer.assert_called_once_with(args.output, report)
        args.journal = args.source_report
        with patch.object(Path, "read_text", side_effect=[json.dumps(item) for item in contents]), self.assertRaises(ValueError):
            command_query_reduction(args)
        args.journal = Path("journal.json")
        args.output = args.source_report
        with self.assertRaises(ValueError):
            command_query_reduction(args)

    def test_operand_alignment_preserves_original_oracle_and_explains_triplets(self) -> None:
        from bioprocess_runtime.gemma_wmma_candidate import operand_aligned_product_bits, serial_k8_product_bits, serial_k8_float32_carry_bits

        reduced = _load("gemma3_270m_query_reduction.json")
        for item in reduced["final_cases"][:2]:
            left, right = [0] * 640, [0] * 640
            for term in item["terms"]:
                left[term["position"]], right[term["position"]] = term["left_bits"], term["right_bits"]
            self.assertEqual(operand_aligned_product_bits(left, right), item["observed_bits"][0])
            self.assertEqual(serial_k8_product_bits(left, right), item["predicted_bits"])
        values = [0x3F81, 0xC380, 0x4380, 0x3FC2] * 4
        self.assertEqual(operand_aligned_product_bits([0x3F80] * 16, values), serial_k8_float32_carry_bits(values, "toward_zero", 8))
        for left, right in (([], []), ([1] * 16, [0x3F80] * 16), ([0x7F80] * 16, [0] * 16)):
            with self.assertRaises(ValueError):
                operand_aligned_product_bits(left, right)

    def test_operand_alignment_plan_freezes_disjoint_inputs(self) -> None:
        from bioprocess_runtime.gemma_wmma_candidate import build_operand_alignment_holdout_plan, _operand_alignment_vectors, OPERAND_ALIGNMENT_PROFILE

        plan = build_operand_alignment_holdout_plan(_load("gemma3_270m_operand_alignment_diagnosis.json"), _load("gemma3_270m_product_holdout.json"))
        self.assertIsNot(plan["profile"], OPERAND_ALIGNMENT_PROFILE)
        self.assertEqual(len(plan["shapes"]), 8)
        self.assertEqual(sum(len(shape["cases"]) for shape in plan["shapes"]), 1024)
        excluded = set(plan["development_left_hashes"])
        for shape, (left, cases) in zip(plan["shapes"], _operand_alignment_vectors()):
            self.assertEqual(shape["left_bits_sha256"], _sha256(left))
            self.assertNotIn(_sha256(left), excluded)
            self.assertEqual(shape["weight_shape"], [1024, 640])
            for case, (_, right) in zip(shape["cases"], cases):
                self.assertEqual(case["right_bits_sha256"], _sha256(right))
        self.assertFalse(plan["candidate_refitting_allowed"])
        self.assertFalse(plan["global_exactness_activation_allowed"])

    def test_operand_alignment_diagnosis_recomputes_development_fit(self) -> None:
        from bioprocess_runtime.gemma_wmma_candidate import diagnose_operand_alignment

        expected = diagnose_operand_alignment(self.search, _load("gemma3_270m_carry_diagnosis.json"), _load("gemma3_270m_product_holdout_plan.json"), _load("gemma3_270m_product_holdout.json"), _load("gemma3_270m_query_reduction_plan.json"), _load("gemma3_270m_query_reduction.json"))
        self.assertEqual(expected, _load("gemma3_270m_operand_alignment_diagnosis.json"))
        self.assertTrue(expected["all_development_predictions_match"])
        self.assertFalse(expected["fresh_holdout_validation_established"])
        self.assertEqual([expected["phase_summaries"][key]["case_count"] for key in ("original_query", "deletion_trial", "final_reduction")], [128, 224, 5])

    def test_operand_alignment_holdout_rejects_refits_and_tampering(self) -> None:
        from bioprocess_runtime.gemma_wmma_candidate import verify_operand_alignment_holdout, _operand_alignment_holdout_report

        diagnosis, source = _load("gemma3_270m_operand_alignment_diagnosis.json"), _load("gemma3_270m_product_holdout.json")
        plan, report = _load("gemma3_270m_operand_alignment_plan.json"), _load("gemma3_270m_operand_alignment_holdout.json")
        result = verify_operand_alignment_holdout(plan, diagnosis, source, report)
        self.assertTrue(result["valid"], result)
        self.assertTrue(result["candidate_passes_within_declared_scope"])
        self.assertEqual(result["tested_case_count"], 1024)
        self.assertEqual(result["mismatch_count"], 0)
        damaged_plan = copy.deepcopy(plan)
        damaged_plan["profile"]["precision_bits"] = 26
        damaged_plan["plan_sha256"] = _sha256({k: v for k, v in damaged_plan.items() if k != "plan_sha256"})
        self.assertFalse(verify_operand_alignment_holdout(damaged_plan, diagnosis, source, report)["valid"])
        damaged = copy.deepcopy(report)
        damaged["acquisitions"][0]["observed_bits"][0][0] ^= 1
        damaged["report_sha256"] = _sha256({k: v for k, v in damaged.items() if k != "report_sha256"})
        self.assertFalse(verify_operand_alignment_holdout(plan, diagnosis, source, damaged)["valid"])
        acquisitions = copy.deepcopy(report["acquisitions"])
        acquisitions[0]["environment"]["torch"] = "different"
        outside_scope = _operand_alignment_holdout_report(plan, acquisitions)
        self.assertTrue(outside_scope["candidate_passes_holdout"])
        self.assertFalse(outside_scope["candidate_passes_within_declared_scope"])

    def test_operand_alignment_cli_keeps_failed_evidence(self) -> None:
        from bioprocess_runtime.cli import command_operand_alignment_holdout
        from bioprocess_runtime.gemma_wmma_candidate import _operand_alignment_holdout_report

        diagnosis, source = _load("gemma3_270m_operand_alignment_diagnosis.json"), _load("gemma3_270m_product_holdout.json")
        plan, report = _load("gemma3_270m_operand_alignment_plan.json"), _load("gemma3_270m_operand_alignment_holdout.json")
        acquisitions = copy.deepcopy(report["acquisitions"])
        acquisitions[0]["observed_bits"][0][0] ^= 1
        failed = _operand_alignment_holdout_report(plan, acquisitions)
        args = Namespace(diagnosis=Path("diagnosis"), source_report=Path("source"), plan=Path("plan"), report=Path("report"), output=Path("output"), operation="run", reexecute=False)
        with patch.object(Path, "read_text", side_effect=[json.dumps(item) for item in (diagnosis, source, plan)]), patch("bioprocess_runtime.gemma_wmma_candidate.acquire_operand_alignment_holdout", return_value=failed), patch("bioprocess_runtime.cli._write_json") as writer:
            self.assertEqual(command_operand_alignment_holdout(args), 1)
            writer.assert_called_once_with(args.output, failed)
        args.operation = "verify"
        with patch.object(Path, "read_text", side_effect=[json.dumps(item) for item in (diagnosis, source, plan, failed)]), redirect_stdout(StringIO()):
            self.assertEqual(command_operand_alignment_holdout(args), 0)

    def test_wide_query_generator_has_distinct_rows_and_anchored_cancellation(self) -> None:
        from bioprocess_runtime.gemma_wmma_candidate import _wide_query_vectors
        from bioprocess_runtime.gemma_float_semantics import decode_finite_bfloat16 as decode

        inputs, weights = _wide_query_vectors()
        self.assertEqual(len({_sha256(row) for row in inputs}), 30)
        self.assertEqual(len(weights), 1024)
        self.assertTrue(all(len(row) == 640 for row in inputs))
        self.assertTrue(all(len(right) == 640 and any(right) for _, right in weights))
        self.assertTrue(all(any(row[k] != row[k + 1] for k in range(0, 640, 2)) for row in inputs))
        paired = weights[512][1]
        self.assertEqual(sum(decode(inputs[0][k])[0] * decode(paired[k])[0] for k in (2, 3)), 0)
        self.assertTrue(any(sum(decode(inputs[7][k])[0] * decode(paired[k])[0] for k in (start, start + 1)) for start in range(2, 640, 2)))
        boundary = weights[768][1]
        self.assertEqual(sum(decode(inputs[0][k])[0] * decode(boundary[k])[0] for k in (126, 128)), 0)

    def test_wide_query_plan_binds_four_complete_output_rows(self) -> None:
        from bioprocess_runtime.gemma_wmma_candidate import build_wide_query_plan, _wide_query_vectors, _operand_alignment_holdout_report

        plan = build_wide_query_plan(_load("gemma3_270m_operand_alignment_plan.json"), _load("gemma3_270m_operand_alignment_holdout.json"))
        self.assertEqual(plan, _load("gemma3_270m_wide_query_plan.json"))
        inputs, weights = _wide_query_vectors()
        self.assertEqual(plan["selected_output_rows"], [0, 7, 15, 29])
        self.assertEqual(plan["tested_case_count"], 4096)
        self.assertEqual(plan["input_bits_sha256"], _sha256(inputs))
        self.assertEqual(plan["weight_bits_sha256"], _sha256([right for _, right in weights]))
        self.assertTrue(all(len(row) == 1024 for row in plan["predicted_bits"]))
        self.assertFalse(plan["complete_output_compared"])
        self.assertFalse(plan["candidate_refitting_allowed"])
        empty_source = _load("gemma3_270m_operand_alignment_plan.json")
        empty_source["shapes"] = []
        empty_source["plan_sha256"] = _sha256({k: v for k, v in empty_source.items() if k != "plan_sha256"})
        with self.assertRaises(ValueError):
            build_wide_query_plan(empty_source, _operand_alignment_holdout_report(empty_source, []))

    def test_wide_query_report_maps_coordinates_and_rejects_truncated_rows(self) -> None:
        from bioprocess_runtime.gemma_wmma_candidate import _wide_query_report, WIDE_QUERY_FAMILIES

        plan = {"predicted_bits": [[0] * 1024 for _ in range(4)], "column_families": [family for family in WIDE_QUERY_FAMILIES for _ in range(256)],
                "expected_environment": {}, "expected_kernel_names": ["synthetic"], "plan_sha256": "synthetic", "predictions_sha256": "synthetic"}
        observed = [[[0] * 1024 for _ in range(4)] for _ in range(3)]
        observed[1][2][1023] = 1
        report = _wide_query_report(plan, observed, [["synthetic"]] * 3, {})
        self.assertEqual(report["mismatch_count"], 1)
        self.assertEqual(report["mismatches"][0]["row"], 15)
        self.assertEqual(report["mismatches"][0]["column"], 1023)
        self.assertFalse(report["candidate_passes_holdout"])
        observed[0][0].pop()
        with self.assertRaises(ValueError):
            _wide_query_report(plan, observed, [["synthetic"]] * 3, {})

    def test_wide_query_fixture_and_tamper_rejection(self) -> None:
        from bioprocess_runtime.gemma_wmma_candidate import verify_wide_query_holdout

        source_plan, source_report = _load("gemma3_270m_operand_alignment_plan.json"), _load("gemma3_270m_operand_alignment_holdout.json")
        plan, report = _load("gemma3_270m_wide_query_plan.json"), _load("gemma3_270m_wide_query_holdout.json")
        result = verify_wide_query_holdout(plan, source_plan, source_report, report)
        self.assertTrue(result["valid"], result)
        self.assertTrue(result["candidate_passes_within_declared_scope"])
        self.assertEqual([item["matching_cases"] for item in result["row_summaries"]], [[1024] * 3] * 4)
        self.assertEqual(result["mismatch_count"], 0)
        self.assertFalse(result["complete_output_compared"])
        with patch("bioprocess_runtime.gemma_wmma_candidate.build_wide_query_plan", return_value=plan):
            damaged = copy.deepcopy(report)
            damaged["observed_bits"][0][3][1023] ^= 1
            damaged["report_sha256"] = _sha256({k: v for k, v in damaged.items() if k != "report_sha256"})
            self.assertFalse(verify_wide_query_holdout(plan, source_plan, source_report, damaged)["valid"])
            altered_plan = copy.deepcopy(plan)
            altered_plan["selected_output_rows"][-1] = 28
            altered_plan["plan_sha256"] = _sha256({k: v for k, v in altered_plan.items() if k != "plan_sha256"})
            self.assertFalse(verify_wide_query_holdout(altered_plan, source_plan, source_report, report)["valid"])

    def test_wide_query_cli_preserves_failed_predictions(self) -> None:
        from bioprocess_runtime.cli import command_wide_query
        from bioprocess_runtime.gemma_wmma_candidate import _wide_query_report

        source_plan, source_report = _load("gemma3_270m_operand_alignment_plan.json"), _load("gemma3_270m_operand_alignment_holdout.json")
        plan, report = _load("gemma3_270m_wide_query_plan.json"), _load("gemma3_270m_wide_query_holdout.json")
        observations = copy.deepcopy(report["observed_bits"])
        observations[0][0][0] ^= 1
        failed = _wide_query_report(plan, observations, report["cuda_kernel_names"], report["environment"])
        args = Namespace(source_plan=Path("source_plan"), source_report=Path("source_report"), plan=Path("plan"), report=Path("report"), output=Path("output"), operation="run", reexecute=False)
        with patch.object(Path, "read_text", side_effect=[json.dumps(item) for item in (source_plan, source_report, plan)]), patch("bioprocess_runtime.gemma_wmma_candidate.acquire_wide_query_holdout", return_value=failed), patch("bioprocess_runtime.cli._write_json") as writer:
            self.assertEqual(command_wide_query(args), 1)
            writer.assert_called_once_with(args.output, failed)
        args.operation = "verify"
        with patch.object(Path, "read_text", side_effect=[json.dumps(item) for item in (source_plan, source_report, plan, failed)]), patch("bioprocess_runtime.gemma_wmma_candidate.build_wide_query_plan", return_value=plan), redirect_stdout(StringIO()):
            self.assertEqual(command_wide_query(args), 0)

    def test_split_k_partial_accumulator_retains_float32_precision(self) -> None:
        from bioprocess_runtime.gemma_wmma_candidate import _operand_aligned_accumulator, operand_aligned_product_bits
        from bioprocess_runtime.gemma_float_semantics import decode_finite_bfloat16, encode_bfloat16_rne

        left, right = [0x3F81] + [0] * 15, [0x3F81] + [0] * 15
        accumulator = _operand_aligned_accumulator(left, right)
        self.assertEqual(accumulator, Fraction(16641, 16384))
        result = operand_aligned_product_bits(left, right)
        self.assertEqual(result, encode_bfloat16_rne(accumulator))
        self.assertNotEqual(accumulator, decode_finite_bfloat16(result)[0])

    def test_split_k_reduction_orders_and_partition_validation(self) -> None:
        from bioprocess_runtime.gemma_wmma_candidate import _merge_split_partials, split_k_candidate_bits, operand_aligned_product_bits

        values = [Fraction(1 << 30), Fraction(1), Fraction(-(1 << 30)), Fraction(1)]
        self.assertEqual(_merge_split_partials(values, "exact"), 0x4000)
        self.assertEqual(_merge_split_partials(values, "sequential_float32_rne"), 0x3F80)
        self.assertEqual(_merge_split_partials(values, "pairwise_float32_rne"), 0)
        self.assertEqual(_merge_split_partials([Fraction(256), Fraction(1), Fraction(1)], "sequential_bfloat16_rne"), 0x4380)
        self.assertEqual(_merge_split_partials([Fraction(256), Fraction(1), Fraction(1)], "sequential_float32_rne"), 0x4381)
        left, right = [0x3F80] * 640, [0] * 640
        for index, bits in zip((0, 64, 128, 192), (0x4E80, 0x3F80, 0xCE80, 0x3F80)):
            right[index] = bits
        self.assertEqual(split_k_candidate_bits(left, right, 64, "bfloat16_rne", "exact"), 0x4000)
        self.assertEqual(split_k_candidate_bits(left, right, 640, "float32", "exact"), operand_aligned_product_bits(left, right))
        with self.assertRaises(ValueError):
            split_k_candidate_bits(left, right, 48, "float32", "exact")
        with self.assertRaises(ValueError):
            split_k_candidate_bits(left, right, 64, "unknown", "exact")

    def test_split_merge_plan_discriminates_all_finalists_before_cuda(self) -> None:
        from bioprocess_runtime.gemma_wmma_candidate import build_split_merge_plan, _split_merge_vectors

        plan = build_split_merge_plan(_load("gemma3_270m_split_k_diagnosis.json"))
        self.assertEqual(len(plan["cases"]), 256)
        self.assertEqual(plan["input_value_bits"], 0x4000)
        self.assertFalse(plan["candidate_refitting_allowed"])
        self.assertFalse(plan["hypothetical_partials_observed"])
        self.assertEqual(len({case["right_bits_sha256"] for case in plan["cases"]}), 256)
        for case, (partials, right) in zip(plan["cases"], _split_merge_vectors()):
            self.assertEqual(len(set(case["predicted_bits"].values())), 3)
            self.assertEqual(case["right_bits_sha256"], _sha256(right))
            self.assertEqual(case["hypothetical_partial_bits"], partials)
            self.assertEqual(sum(bool(bits) for bits in right), 4)

    def test_split_k_diagnosis_recomputes_three_development_finalists(self) -> None:
        from bioprocess_runtime.gemma_wmma_candidate import diagnose_split_k

        expected = diagnose_split_k(self.search, _load("gemma3_270m_carry_diagnosis.json"), _load("gemma3_270m_product_holdout_plan.json"), _load("gemma3_270m_product_holdout.json"), _load("gemma3_270m_matched_products_plan.json"), _load("gemma3_270m_matched_products.json"))
        self.assertEqual(expected, _load("gemma3_270m_split_k_diagnosis.json"))
        self.assertEqual(expected["candidate_count"], 80)
        self.assertEqual(len(expected["records"]), 256)
        self.assertEqual(len(expected["all_matching_candidates"]), 3)
        self.assertTrue(all(item["chunk_size"] == 64 and item["partial_format"] == "bfloat16_rne" for item in expected["all_matching_candidates"]))
        self.assertFalse(expected["unique_all_matching_candidate_in_search_space"])
        self.assertFalse(expected["fresh_holdout_validation_established"])

    def test_split_merge_holdout_rejects_forged_survivors_and_refits(self) -> None:
        from bioprocess_runtime.gemma_wmma_candidate import verify_split_merge_holdout

        diagnosis = _load("gemma3_270m_split_k_diagnosis.json")
        plan, report = _load("gemma3_270m_split_merge_plan.json"), _load("gemma3_270m_split_merge_holdout.json")
        result = verify_split_merge_holdout(plan, diagnosis, report)
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["surviving_candidates"], ["sequential_float32_rne"])
        self.assertEqual(result["candidate_mismatch_counts"], {"exact": 768, "sequential_float32_rne": 0, "pairwise_float32_rne": 768})
        damaged = copy.deepcopy(report)
        damaged["surviving_candidates"] = ["exact"]
        damaged["report_sha256"] = _sha256({k: v for k, v in damaged.items() if k != "report_sha256"})
        self.assertFalse(verify_split_merge_holdout(plan, diagnosis, damaged)["valid"])
        damaged_plan = copy.deepcopy(plan)
        damaged_plan["finalists"][0]["chunk_size"] = 128
        damaged_plan["plan_sha256"] = _sha256({k: v for k, v in damaged_plan.items() if k != "plan_sha256"})
        self.assertFalse(verify_split_merge_holdout(damaged_plan, diagnosis, report)["valid"])

    def test_split_merge_cli_preserves_rejected_candidate_set(self) -> None:
        from bioprocess_runtime.cli import command_split_merge
        from bioprocess_runtime.gemma_wmma_candidate import _split_merge_report

        diagnosis = _load("gemma3_270m_split_k_diagnosis.json")
        plan, report = _load("gemma3_270m_split_merge_plan.json"), _load("gemma3_270m_split_merge_holdout.json")
        observed = copy.deepcopy(report["observed_bits"])
        observed[0][0] = next(bits for bits in range(65536) if bits not in plan["cases"][0]["predicted_bits"].values())
        failed = _split_merge_report(plan, observed, report["cuda_kernel_names"], report["environment"])
        args = Namespace(diagnosis=Path("diagnosis"), plan=Path("plan"), report=Path("report"), output=Path("output"), operation="run", reexecute=False)
        with patch.object(Path, "read_text", side_effect=[json.dumps(item) for item in (diagnosis, plan)]), patch("bioprocess_runtime.gemma_wmma_candidate.acquire_split_merge_holdout", return_value=failed), patch("bioprocess_runtime.cli._write_json") as writer:
            self.assertEqual(command_split_merge(args), 1)
            writer.assert_called_once_with(args.output, failed)
        args.operation = "verify"
        with patch.object(Path, "read_text", side_effect=[json.dumps(item) for item in (diagnosis, plan, failed)]), redirect_stdout(StringIO()):
            self.assertEqual(command_split_merge(args), 0)

    def test_dense_split_plan_freezes_stratified_fresh_inputs(self) -> None:
        from bioprocess_runtime.gemma_wmma_candidate import build_dense_split_plan, _dense_split_vectors, DENSE_SPLIT_COLUMNS

        plan = build_dense_split_plan(_load("gemma3_270m_split_k_diagnosis.json"), _load("gemma3_270m_split_merge_plan.json"), _load("gemma3_270m_split_merge_holdout.json"))
        self.assertEqual(plan, _load("gemma3_270m_dense_split_plan.json"))
        inputs, weights = _dense_split_vectors()
        self.assertEqual(plan["tested_case_count"], 1024)
        self.assertEqual(plan["weight_shape"], [256, 640])
        self.assertEqual(plan["selected_output_rows"], [0, 7, 15, 29])
        self.assertEqual(plan["input_bits_sha256"], _sha256(inputs))
        self.assertEqual(plan["weight_bits_sha256"], _sha256([right for _, right in weights]))
        self.assertFalse(set(plan["input_row_hashes"]) & set(plan["excluded_input_row_hashes"]))
        self.assertEqual(len(set(plan["input_row_hashes"])), 30)
        self.assertEqual([sum(column % 4 == offset for column in DENSE_SPLIT_COLUMNS[-64:]) for offset in range(4)], [16] * 4)
        self.assertTrue(all(len(row) == 256 for row in plan["predicted_bits"]))
        self.assertFalse(plan["candidate_refitting_allowed"])
        self.assertFalse(plan["complete_output_compared"])

    def test_dense_split_fixture_and_tamper_rejection(self) -> None:
        from bioprocess_runtime.gemma_wmma_candidate import verify_dense_split_holdout

        sources = [_load(name) for name in ("gemma3_270m_split_k_diagnosis.json", "gemma3_270m_split_merge_plan.json", "gemma3_270m_split_merge_holdout.json")]
        plan, report = _load("gemma3_270m_dense_split_plan.json"), _load("gemma3_270m_dense_split_holdout.json")
        result = verify_dense_split_holdout(plan, *sources, report)
        self.assertTrue(result["valid"], result)
        self.assertTrue(result["candidate_passes_within_declared_scope"])
        self.assertEqual(result["mismatch_count"], 0)
        self.assertEqual(result["tested_case_count"], 1024)
        self.assertFalse(result["complete_output_compared"])
        with patch("bioprocess_runtime.gemma_wmma_candidate.build_dense_split_plan", return_value=plan):
            damaged = copy.deepcopy(report)
            damaged["observed_bits"][0][3][255] ^= 1
            damaged["report_sha256"] = _sha256({k: v for k, v in damaged.items() if k != "report_sha256"})
            self.assertFalse(verify_dense_split_holdout(plan, *sources, damaged)["valid"])
            altered = copy.deepcopy(plan)
            altered["split_profile"]["chunk_size"] = 128
            altered["plan_sha256"] = _sha256({k: v for k, v in altered.items() if k != "plan_sha256"})
            self.assertFalse(verify_dense_split_holdout(altered, *sources, report)["valid"])

    def test_dense_split_cli_preserves_failed_predictions(self) -> None:
        from bioprocess_runtime.cli import command_dense_split
        from bioprocess_runtime.gemma_wmma_candidate import _dense_split_report

        sources = [_load(name) for name in ("gemma3_270m_split_k_diagnosis.json", "gemma3_270m_split_merge_plan.json", "gemma3_270m_split_merge_holdout.json")]
        plan, report = _load("gemma3_270m_dense_split_plan.json"), _load("gemma3_270m_dense_split_holdout.json")
        observed = copy.deepcopy(report["observed_bits"])
        observed[0][0][0] ^= 1
        failed = _dense_split_report(plan, observed, report["cuda_kernel_names"], report["environment"])
        args = Namespace(diagnosis=Path("diagnosis"), merge_plan=Path("merge_plan"), merge_report=Path("merge_report"), plan=Path("plan"), report=Path("report"), output=Path("output"), operation="run", reexecute=False)
        with patch.object(Path, "read_text", side_effect=[json.dumps(item) for item in (*sources, plan)]), patch("bioprocess_runtime.gemma_wmma_candidate.acquire_dense_split_holdout", return_value=failed), patch("bioprocess_runtime.cli._write_json") as writer:
            self.assertEqual(command_dense_split(args), 1)
            writer.assert_called_once_with(args.output, failed)
        args.operation = "verify"
        with patch.object(Path, "read_text", side_effect=[json.dumps(item) for item in (*sources, plan, failed)]), patch("bioprocess_runtime.gemma_wmma_candidate.build_dense_split_plan", return_value=plan), redirect_stdout(StringIO()):
            self.assertEqual(command_dense_split(args), 0)

    def test_shared_exponent_precision_and_rounding(self) -> None:
        from bioprocess_runtime.gemma_wmma_candidate import _shared_exponent_sum

        values = [Fraction(1 << 30), Fraction(103), Fraction(-(1 << 30))]
        self.assertEqual(
            _shared_exponent_sum(values, 25, "toward_zero"),
            Fraction(64),
        )
        self.assertEqual(
            _shared_exponent_sum(values, 25, "nearest_even"),
            Fraction(128),
        )
        self.assertEqual(_shared_exponent_sum([], 25, "toward_zero"), Fraction(0))

    def test_candidate_reproduces_position_sensitive_examples(self) -> None:
        from bioprocess_runtime.gemma_wmma_candidate import k16_half_transition_bits

        values = [0] * 16
        values[0:3] = [0x4E80, 0xCE80, 0x3F80]
        self.assertEqual(
            k16_half_transition_bits(values, 25, "toward_zero", "lower_then_upper"),
            0x0000,
        )
        values = [0] * 16
        values[0] = 0x4E80
        values[1] = 0xCE80
        values[8] = 0x3F80
        self.assertEqual(
            k16_half_transition_bits(values, 25, "toward_zero", "lower_then_upper"),
            0x3F80,
        )
        self.assertEqual(
            k16_half_transition_bits(values, 25, "toward_zero", "upper_then_lower"),
            0x0000,
        )

    def test_record_vector_rejects_modulo_lane_collision(self) -> None:
        from bioprocess_runtime.gemma_wmma_candidate import _record_vector

        with self.assertRaisesRegex(ValueError, "multiple values"):
            _record_vector({
                "positions": [0, 16],
                "value_bits": ["0x3f80", "0x4000"],
            })

    def test_candidate_search_rejects_tampering(self) -> None:
        from bioprocess_runtime.gemma_wmma_candidate import (
            verify_wmma_candidate_search,
        )

        damaged = copy.deepcopy(self.search)
        damaged["complete_matching_profiles"][0]["precision_bits"] = 24
        body = {
            key: value for key, value in damaged.items() if key != "search_sha256"
        }
        damaged["search_sha256"] = _sha256(body)
        verification = verify_wmma_candidate_search(
            self.program,
            self.reduction,
            self.suite,
            self.backend,
            self.position,
            self.accumulator,
            self.magnitude,
            damaged,
        )
        self.assertFalse(verification["valid"])
        self.assertFalse(verification["derived_search_exact_match"])
        activated = copy.deepcopy(self.search)
        activated["complete_numeric_transition_established"] = True
        body = {
            key: value for key, value in activated.items() if key != "search_sha256"
        }
        activated["search_sha256"] = _sha256(body)
        self.assertFalse(
            verify_wmma_candidate_search(
                self.program,
                self.reduction,
                self.suite,
                self.backend,
                self.position,
                self.accumulator,
                self.magnitude,
                activated,
            )["valid"]
        )


if __name__ == "__main__":
    unittest.main()
