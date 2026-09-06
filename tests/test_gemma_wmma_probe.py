from __future__ import annotations

import copy
import hashlib
import json
import unittest
from pathlib import Path

from bioprocess_runtime.serialization import canonical_json


ROOT = Path(__file__).resolve().parent.parent


def _load(name: str) -> dict:
    return json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class GemmaWmmaProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.program = _load("gemma3_270m_execution_ir.json")
        cls.reduction = _load("gemma3_270m_reduction_characterization.json")
        cls.suite = _load("gemma3_270m_nsight_kernel_suite.json")
        cls.backend = _load("gemma3_270m_reduction_backend_binding.json")
        cls.certificate = _load("gemma3_270m_wmma_probe.json")

    def test_probe_certificate_maps_position_sensitive_outputs(self) -> None:
        from bioprocess_runtime.gemma_wmma_probe import verify_wmma_probe_certificate

        verification = verify_wmma_probe_certificate(
            self.program,
            self.reduction,
            self.suite,
            self.backend,
            self.certificate,
        )
        self.assertTrue(verification["valid"], verification)
        self.assertEqual(verification["probe_count"], 1024)
        self.assertEqual(verification["distinct_result_count"], 2)
        self.assertFalse(verification["all_results_match_exact_sum"])
        self.assertFalse(verification["operand_position_invariance_established"])
        self.assertFalse(verification["wmma_accumulator_mapping_identified"])

    def test_all_probes_have_identical_exact_sum_but_two_observed_results(self) -> None:
        self.assertTrue(self.certificate["all_exact_sums_identical"])
        self.assertEqual(
            {record["exact_result_bits"] for record in self.certificate["records"]},
            {"0x4000"},
        )
        self.assertEqual(self.certificate["result_histogram"], {
            "0x0000": 965,
            "0x3f80": 59,
        })
        permutation_records = [
            record
            for record in self.certificate["records"]
            if record["family"] == "within_fragment_permutation"
        ]
        self.assertEqual(len(permutation_records), 480)
        self.assertEqual(
            {record["actual_result_bits"] for record in permutation_records},
            {"0x0000"},
        )

    def test_lane_five_is_the_only_shift_retaining_one(self) -> None:
        retained = [
            record
            for record in self.certificate["records"]
            if record["family"] == "within_fragment_lane_shift"
            and record["actual_result_bits"] == "0x3f80"
        ]
        self.assertEqual(len(retained), 40)
        self.assertEqual({record["lane_start"] for record in retained}, {5})
        self.assertEqual({record["tile"] for record in retained}, set(range(40)))

    def test_cuda_reexecution_rejects_self_consistent_result_forgery(self) -> None:
        import torch

        from bioprocess_runtime.gemma_wmma_probe import verify_wmma_probe_certificate

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for WMMA probe re-execution")
        forged = copy.deepcopy(self.certificate)
        forged["records"][0]["actual_result_bits"] = "0x3f80"
        record_body = {
            key: value
            for key, value in forged["records"][0].items()
            if key != "record_sha256"
        }
        forged["records"][0]["record_sha256"] = _sha256(record_body)
        forged["result_histogram"] = {"0x0000": 964, "0x3f80": 60}
        forged["family_histograms"]["within_fragment_permutation"] = {
            "0x0000": 479,
            "0x3f80": 1,
        }
        body = {
            key: value
            for key, value in forged.items()
            if key != "certificate_sha256"
        }
        forged["certificate_sha256"] = _sha256(body)
        verification = verify_wmma_probe_certificate(
            self.program,
            self.reduction,
            self.suite,
            self.backend,
            forged,
            reexecute=True,
        )
        self.assertFalse(verification["valid"])
        self.assertFalse(verification["reexecution_exact"])

    def test_probe_verifier_rejects_position_and_claim_tampering(self) -> None:
        from bioprocess_runtime.gemma_wmma_probe import verify_wmma_probe_certificate

        damaged = copy.deepcopy(self.certificate)
        damaged["records"][0]["positions"][0] = 15
        record_body = {
            key: value
            for key, value in damaged["records"][0].items()
            if key != "record_sha256"
        }
        damaged["records"][0]["record_sha256"] = _sha256(record_body)
        body = {
            key: value
            for key, value in damaged.items()
            if key != "certificate_sha256"
        }
        damaged["certificate_sha256"] = _sha256(body)
        verification = verify_wmma_probe_certificate(
            self.program, self.reduction, self.suite, self.backend, damaged
        )
        self.assertFalse(verification["valid"])
        self.assertFalse(verification["records_valid"])
        activated = copy.deepcopy(self.certificate)
        activated["wmma_accumulator_mapping_identified"] = True
        body = {
            key: value
            for key, value in activated.items()
            if key != "certificate_sha256"
        }
        activated["certificate_sha256"] = _sha256(body)
        self.assertFalse(
            verify_wmma_probe_certificate(
                self.program,
                self.reduction,
                self.suite,
                self.backend,
                activated,
            )["valid"]
        )


if __name__ == "__main__":
    unittest.main()
