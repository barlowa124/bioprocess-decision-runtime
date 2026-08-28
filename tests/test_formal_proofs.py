from __future__ import annotations

import copy
import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from bioprocess_runtime.serialization import canonical_json


Z3_AVAILABLE = importlib.util.find_spec("z3") is not None


@unittest.skipUnless(Z3_AVAILABLE, "Proof optional dependencies are not installed")
class FormalProofTests(unittest.TestCase):
    def test_all_declared_properties_are_proved_by_unsatisfiable_counterexample_queries(self) -> None:
        from bioprocess_runtime.formal_proofs import build_formal_proof_certificate

        certificate = build_formal_proof_certificate()
        self.assertEqual(certificate["proved"], certificate["total"])
        self.assertTrue(all(proof["solver_result"] == "unsat" for proof in certificate["proofs"]))
        self.assertIn("selected IEEE-754 bfloat16", certificate["scope"])
        self.assertIn("CUDA SASS instruction-level semantic equivalence", certificate["unresolved"])

    def test_proof_certificate_is_reexecuted_and_tampering_is_detected(self) -> None:
        from bioprocess_runtime.formal_proofs import build_formal_proof_certificate, verify_formal_proof_certificate

        certificate = build_formal_proof_certificate()
        self.assertTrue(verify_formal_proof_certificate(certificate)["valid"])
        damaged = copy.deepcopy(certificate)
        damaged["proofs"][0]["proved"] = False
        self.assertFalse(verify_formal_proof_certificate(damaged)["valid"])
        forged_scope = copy.deepcopy(certificate)
        forged_scope["scope"] = "unrestricted full-transformer proof"
        body = {key: value for key, value in forged_scope.items() if key != "certificate_sha256"}
        forged_scope["certificate_sha256"] = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
        verification = verify_formal_proof_certificate(forged_scope)
        self.assertTrue(verification["integrity_valid"])
        self.assertFalse(verification["reexecution_metadata_match"])
        self.assertFalse(verification["valid"])
        damaged_query = copy.deepcopy(certificate)
        damaged_query["proofs"][0]["smt2_sha256"] = "0" * 64
        self.assertFalse(verify_formal_proof_certificate(damaged_query)["valid"])


class CudaProvenanceTests(unittest.TestCase):
    def test_embedded_image_and_architecture_parsing(self) -> None:
        from bioprocess_runtime.cuda_provenance import _compatible_architecture, _locate_embedded_image, _parse_embedded_images

        listing = "\n".join(
            [
                "ELF file 1: torch_cuda.1.sm_86.cubin",
                "ELF file 2: torch_cuda.2.sm_89.cubin",
                "ELF file 3: torch_cuda.3.sm_90a.cubin",
            ]
        )
        parsed = _parse_embedded_images(listing)
        self.assertEqual(parsed["total_images"], 3)
        self.assertEqual(parsed["architectures"]["sm_90a"], 1)
        self.assertEqual(_compatible_architecture(["sm_80", "sm_86", "sm_90a"], (8, 9)), "sm_86")
        output = "Fatbin elf code:\nno match\nFatbin elf code:\nFunction : target\n"
        self.assertIsNone(_locate_embedded_image(output, listing, "sm_86", "target"))
        same_arch_listing = "\n".join(["ELF file 1: first.sm_86.cubin", "ELF file 2: second.sm_86.cubin"])
        self.assertEqual(_locate_embedded_image(output, same_arch_listing, "sm_86", "target"), "second.sm_86.cubin")

    def test_binary_inventory_supports_windows_and_linux_names(self) -> None:
        from bioprocess_runtime.cuda_provenance import _binary_inventory

        for names in (
            ("torch_cuda.dll", "cublas64_12.dll", "cublasLt64_12.dll", "cudart64_12.dll"),
            ("libtorch_cuda.so", "libcublas.so.12", "libcublasLt.so.12", "libcudart.so.12"),
        ):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for name in names:
                    (root / name).write_bytes(name.encode("utf-8"))
                binaries, torch_cuda = _binary_inventory(root)
                self.assertEqual(len(binaries), 4)
                self.assertIn("torch_cuda", torch_cuda.name)
                self.assertEqual({binary["role"] for binary in binaries}, {"torch_cuda", "cublas", "cublas_lt", "cuda_runtime"})

    def test_invalid_kernel_filter_is_rejected_before_tool_execution(self) -> None:
        from bioprocess_runtime.cuda_provenance import _disassemble_profiled_function

        with self.assertRaisesRegex(ValueError, "Invalid --kernel"):
            _disassemble_profiled_function("cuobjdump", "nvdisasm", Path("torch_cuda"), {}, "sm_86", "", "(")

    def test_cuda_command_fails_when_no_profiled_symbol_is_bound(self) -> None:
        from bioprocess_runtime.cli import command_cuda_provenance

        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                prompt="test",
                prompt_file=None,
                model_path=Path("model"),
                kernel=None,
                redact=False,
                output=Path(directory) / "manifest.json",
            )
            manifest = {"profiled_symbol_disassembly": {"profiled_kernel_name": None}}
            with (
                patch("bioprocess_runtime.interpretability.load_local_gemma", return_value=(object(), object())),
                patch("bioprocess_runtime.interpretability._model_device", return_value="cpu"),
                patch("bioprocess_runtime.interpretability._tokenize", return_value={"input_ids": object()}),
                patch("bioprocess_runtime.cuda_provenance.build_cuda_provenance_manifest", return_value=manifest),
                patch("bioprocess_runtime.cli._write_json"),
            ):
                self.assertEqual(command_cuda_provenance(args), 1)

    def test_manifest_verifier_checks_the_whole_manifest_hash(self) -> None:
        from bioprocess_runtime.cuda_provenance import verify_cuda_provenance_manifest

        manifest = {
            "scope": "Observed CUDA launches; not instruction-level semantic verification.",
            "runtime_distribution_binaries": [],
        }
        manifest["manifest_sha256"] = hashlib.sha256(canonical_json(manifest).encode("utf-8")).hexdigest()
        self.assertTrue(verify_cuda_provenance_manifest(manifest)["valid"])
        damaged = copy.deepcopy(manifest)
        damaged["scope"] = "changed"
        self.assertFalse(verify_cuda_provenance_manifest(damaged)["valid"])

    def test_cuda_summary_preserves_evidence_boundaries(self) -> None:
        from bioprocess_runtime.cuda_provenance import summarize_cuda_provenance

        manifest = {
            "scope": "Observed launches; not instruction-level semantic verification.",
            "privacy": {"redacted": False},
            "runtime": {"device_capability": [8, 9]},
            "execution_binding": {"model_state_sha256": "model", "input_ids": [[1, 2]]},
            "manifest_sha256": "abc",
            "profile": {
                "cuda_event_count": 3,
                "unique_kernel_count": 2,
                "kernel_launch_counts": {"Memcpy DtoH (Device -> Pinned)": 1, "_ZN2at6native_kernel": 2},
            },
            "runtime_distribution_binaries": [],
            "torch_cuda_embedded_images": {"total_images": 1},
            "profiled_symbol_disassembly": {
                "profiled_kernel_name": "_ZN2at6native_kernel",
                "embedded_architecture": "sm_86",
                "embedded_image": "image.cubin",
                "embedded_image_sha256": "def",
                "profiled_symbol_present_in_image": True,
                "image_instruction_count": 4,
                "image_opcode_histogram": {"MOV": 4},
                "nvdisasm_output_sha256": "ghi",
                "binding": "compatible image only",
            },
            "unresolved": ["driver selection"],
        }
        summary = summarize_cuda_provenance(manifest)
        self.assertEqual(summary["profile"]["families"]["pytorch_native"]["launches"], 2)
        self.assertIn("not instruction-level", summary["scope"])
        self.assertEqual(summary["profiled_symbol_disassembly"]["unique_opcodes"], 1)
