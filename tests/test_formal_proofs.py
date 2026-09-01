from __future__ import annotations

import copy
import ctypes
import hashlib
import struct
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from bioprocess_runtime.serialization import canonical_json


Z3_AVAILABLE = importlib.util.find_spec("z3") is not None
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(Z3_AVAILABLE, "Proof optional dependencies are not installed")
class FormalProofTests(unittest.TestCase):
    def test_all_declared_properties_are_proved_by_unsatisfiable_counterexample_queries(self) -> None:
        from bioprocess_runtime.formal_proofs import build_formal_proof_certificate

        certificate = build_formal_proof_certificate()
        self.assertEqual(certificate["proved"], certificate["total"])
        self.assertTrue(all(proof["solver_result"] == "unsat" for proof in certificate["proofs"]))
        self.assertIn("selected IEEE-754 bfloat16", certificate["scope"])
        self.assertIn("conditional architecture-composition", certificate["scope"])
        self.assertEqual(certificate["counterexamples_found"], 2)
        self.assertTrue(all(item["solver_result"] == "sat" for item in certificate["counterexamples"]))
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


@unittest.skipUnless(Z3_AVAILABLE, "Proof optional dependencies are not installed")
class SassSemanticsTests(unittest.TestCase):
    def test_declared_sass_subset_reexecutes(self) -> None:
        from bioprocess_runtime.sass_semantics import build_sass_semantics_certificate, verify_sass_semantics_certificate

        certificate = build_sass_semantics_certificate()
        self.assertEqual(certificate["proved"], certificate["total"])
        self.assertGreaterEqual(certificate["total"], 54)
        self.assertEqual(certificate["proof_strength_summary"]["independent_reduced_width_references"], 4)
        self.assertEqual(certificate["proof_strength_summary"]["full_width_definitional_or_compositional_instances"], 9)
        self.assertTrue(
            {
                "uldc_matches_little_endian_constant_memory_read",
                "uldc_u8_matches_little_endian_constant_memory_read",
                "register_transfer_shares_proposed_bit_copy_equation",
                "register_transfer_full_32bit_definition_instance",
            }.issubset({proof["name"] for proof in certificate["proofs"]})
        )
        self.assertTrue(
            {
                "FFMA",
                "FMUL",
                "FADD",
                "SHF.R.U32.HI",
                "ISETP.NE.AND",
                "ULDC.64",
                "LDG.E.64",
                "LDG.E.LTC128B.128",
                "STG.E.64",
                "STG.E.128",
                "LDGSTS.E.BYPASS.LTC128B.128",
                "IMAD.WIDE",
                "IMAD.WIDE.U32",
                "UIADD3",
                "UIMAD",
                "UMOV",
                "ULDC",
                "ULDC.U8",
                "S2R",
                "S2UR",
                "R2UR",
                "ULEA",
                "ULEA.HI.X",
                "ULEA.HI.X.SX32",
                "LEA.HI.SX32",
                "LEA.HI.X.SX32",
            }.issubset(
                certificate["covered_base_opcodes"]
            )
        )
        self.assertEqual(
            certificate["proposed_memory_operand_roles"]["LDGSTS"],
            [["candidate_write", "shared"], ["candidate_read", "global"]],
        )
        self.assertTrue(verify_sass_semantics_certificate(certificate)["valid"])
        self.assertIn("not NVIDIA-certified", certificate["scope"])

    def test_sass_certificate_tampering_is_detected(self) -> None:
        from bioprocess_runtime.sass_semantics import build_sass_semantics_certificate, verify_sass_semantics_certificate

        certificate = build_sass_semantics_certificate()
        damaged = copy.deepcopy(certificate)
        damaged["proofs"][0]["proved"] = False
        self.assertFalse(verify_sass_semantics_certificate(damaged)["valid"])
        forged_roles = copy.deepcopy(certificate)
        forged_roles["proposed_memory_operand_roles"]["LDG"][0][0] = "candidate_write"
        body = {key: value for key, value in forged_roles.items() if key != "certificate_sha256"}
        forged_roles["certificate_sha256"] = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
        verification = verify_sass_semantics_certificate(forged_roles)
        self.assertTrue(verification["integrity_valid"])
        self.assertFalse(verification["reexecution_claims_match"])
        self.assertFalse(verification["valid"])

    def test_sass_image_coverage_counts_only_exact_base_opcodes(self) -> None:
        from bioprocess_runtime.sass_semantics import sass_image_coverage

        coverage = sass_image_coverage({"MOV": 3, "IADD3": 2, "IADD3.X": 7, "BRA": 5})
        self.assertEqual(coverage["covered_instruction_lines"], 10)
        self.assertEqual(coverage["total_instruction_lines"], 17)
        self.assertIn("Syntactic", coverage["scope"])


class CuptiAttestationTests(unittest.TestCase):
    def test_module_capture_integrity_and_static_image_binding(self) -> None:
        from bioprocess_runtime.cupti_attestation import summarize_cupti_module_capture, verify_cupti_module_capture

        cubin_hash = hashlib.sha256(b"abc").hexdigest()
        report = {
            "scope": "module capture",
            "cupti_library": "cupti",
            "module_load_events": 1,
            "unique_cubins": 1,
            "modules": [{"module_id": 7, "cubin_size": 3, "cubin_sha256": cubin_hash, "artifact": "module.cubin"}],
            "callback_errors": [],
            "execution_binding": {"selected_token_id": 1},
            "limitations": ["not per-launch"],
        }
        report["record_sha256"] = hashlib.sha256(canonical_json(report).encode("utf-8")).hexdigest()
        self.assertTrue(verify_cupti_module_capture(report)["valid"])
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "module.cubin").write_bytes(b"abc")
            verification = verify_cupti_module_capture(report, Path(directory))
            self.assertTrue(verification["local_artifacts_match"])
            self.assertTrue(verification["valid"])
        cuda_summary = {
            "profiled_symbol_disassembly": {
                "profiled_kernel_name": "kernel",
                "embedded_image_sha256": cubin_hash,
            }
        }
        summary = summarize_cupti_module_capture(report, cuda_summary)
        self.assertTrue(summary["profiled_static_image_binding"]["matched"])
        self.assertEqual(summary["profiled_static_image_binding"]["matching_module_loads"][0]["module_id"], 7)
        damaged = copy.deepcopy(report)
        damaged["module_load_events"] = 2
        self.assertFalse(verify_cupti_module_capture(damaged)["valid"])


class LaunchArgumentTests(unittest.TestCase):
    def test_packed_parameter_scanning_uses_driver_reported_size(self) -> None:
        from bioprocess_runtime.launch_arguments import CuptiLaunchArgumentCapture, _handle_hash

        class Driver:
            @staticmethod
            def cuFuncGetParamInfo(function: int, index: int, offset: object, size: object) -> int:
                if index:
                    return 1
                ctypes.cast(offset, ctypes.POINTER(ctypes.c_size_t))[0] = 0
                ctypes.cast(size, ctypes.POINTER(ctypes.c_size_t))[0] = 360
                return 0

        value = ctypes.create_string_buffer(360)
        pointer = 0x1234567890
        struct.pack_into("<Q", value, 280, pointer)
        values = (ctypes.c_void_p * 1)(ctypes.addressof(value))
        capture = object.__new__(CuptiLaunchArgumentCapture)
        capture.driver = Driver()
        parameters = capture._parameters(1, values)
        self.assertEqual(parameters[0]["size_bytes"], 360)
        candidate = next(item for item in parameters[0]["aligned_pointer_candidates"] if item["byte_offset"] == 280)
        self.assertEqual(candidate["pointer_value_sha256"], _handle_hash(pointer))

    def test_pointer_candidate_summary_preserves_type_and_access_boundaries(self) -> None:
        from bioprocess_runtime.launch_arguments import (
            build_launch_argument_summary,
            redact_launch_argument_artifact,
            verify_launch_argument_artifact,
            verify_launch_argument_summary,
        )

        invocation = {
            "module": "module",
            "inputs": [{"data_pointer_sha256": "input_pointer", "sha256": "input"}],
            "parameters": [{"data_pointer_sha256": "weight_pointer", "sha256": "weight"}],
            "outputs": [{"data_pointer_sha256": "output_pointer", "sha256": "output"}],
        }
        invocation["invocation_sha256"] = hashlib.sha256(canonical_json(invocation).encode("utf-8")).hexdigest()
        module_report = {"patterns": ["module"], "invocations": [invocation], "lifecycle_errors": []}
        module_report["report_sha256"] = hashlib.sha256(canonical_json(module_report).encode("utf-8")).hexdigest()
        launch = {
            "qualified_module_stack": ["module"],
            "callback": "cuLaunchKernel",
            "correlation_id": 1,
            "grid": [1, 1, 1],
            "block": [32, 1, 1],
            "shared_memory_bytes": 0,
            "parameters": [
                {
                    "size_bytes": 24,
                    "value_sha256": "value",
                    "aligned_pointer_candidates": [
                        {"byte_offset": 0, "pointer_value_sha256": "input_pointer"},
                        {"byte_offset": 8, "pointer_value_sha256": "output_pointer"},
                        {"byte_offset": 16, "pointer_value_sha256": "weight_pointer"},
                    ],
                }
            ],
            "parameter_pointer_matches": [
                {"parameter_index": 0, "parameter_byte_offset": 0, "matches": [{"module": "module", "role": "input", "tensor_sha256": "input"}]},
                {"parameter_index": 0, "parameter_byte_offset": 8, "matches": [{"module": "module", "role": "output", "tensor_sha256": "output"}]},
                {"parameter_index": 0, "parameter_byte_offset": 16, "matches": [{"module": "module", "role": "parameter", "tensor_sha256": "weight"}]},
            ],
            "parameter_storage_range_matches": [
                {"parameter_index": 0, "parameter_byte_offset": 0, "module": "module", "role": "input", "tensor_sha256": "input", "storage_offset_bytes": 0, "equals_tensor_data_pointer": True},
                {"parameter_index": 0, "parameter_byte_offset": 8, "module": "module", "role": "output", "tensor_sha256": "output", "storage_offset_bytes": 0, "equals_tensor_data_pointer": True},
                {"parameter_index": 0, "parameter_byte_offset": 16, "module": "module", "role": "parameter", "tensor_sha256": "weight", "storage_offset_bytes": 0, "equals_tensor_data_pointer": True},
            ],
        }
        launch["launch_sha256"] = hashlib.sha256(canonical_json(launch).encode("utf-8")).hexdigest()
        launch_report = {"kernel_name": "kernel", "launches": [launch], "callback_errors": [], "module_report_sha256": module_report["report_sha256"]}
        launch_report["report_sha256"] = hashlib.sha256(canonical_json(launch_report).encode("utf-8")).hexdigest()
        artifact = {
            "privacy": {"redacted": False},
            "model_state_sha256": "model",
            "selected_token_id": 1,
            "input_ids_tensor": {"sha256": "input_ids"},
            "output_logits_tensor": {"sha256": "logits"},
            "module_invocation_report": module_report,
            "launch_argument_report": launch_report,
        }
        artifact["artifact_sha256"] = hashlib.sha256(canonical_json(artifact).encode("utf-8")).hexdigest()
        self.assertTrue(verify_launch_argument_artifact(artifact)["valid"])
        redacted = redact_launch_argument_artifact(artifact)
        self.assertTrue(verify_launch_argument_artifact(redacted)["valid"])
        self.assertEqual(redacted["model_state_sha256"], "redacted")
        self.assertEqual(
            redacted["launch_argument_report"]["launches"][0]["parameters"][0]["value_sha256"],
            "redacted",
        )
        summary = build_launch_argument_summary([(artifact, "module")])
        self.assertEqual(summary["entries_with_input_boundary_match"], 1)
        self.assertEqual(summary["entries_with_retrospective_output_address_match"], 1)
        self.assertEqual(summary["entries_with_parameter_boundary_match"], 1)
        self.assertEqual(summary["storage_range_match_count"], 3)
        self.assertFalse(summary["typed_kernel_signatures_established"])
        self.assertFalse(summary["complete_argument_binding_established"])
        self.assertTrue(verify_launch_argument_summary(summary)["valid"])


@unittest.skipUnless(bool(os.environ.get("CUDA_PATH")), "CUDA toolkit is not installed")
class CudaMetadataTests(unittest.TestCase):
    def test_local_headers_and_ctypes_layouts_conform(self) -> None:
        from bioprocess_runtime.cuda_metadata import build_cuda_metadata_conformance, verify_cuda_metadata_conformance

        certificate = build_cuda_metadata_conformance()
        self.assertTrue(certificate["all_checks_pass"])
        self.assertEqual(certificate["callback_ids"]["cuLaunchKernel"], 307)
        self.assertEqual(certificate["ctypes_layouts"]["launch_kernel"]["size"], 64)
        self.assertTrue(verify_cuda_metadata_conformance(certificate)["valid"])


class SassExpressionTests(unittest.TestCase):
    def test_expression_dag_and_compact_summary_integrity(self) -> None:
        from bioprocess_runtime.sass_expressions import (
            _build_expression_semantics_snapshot,
            _build_launch_coordinate_domains,
            _call_string_expression_snapshots,
            _reachable_nodes,
            _semantic_requirement,
            _special_registers,
            _transfer_definitions,
            _unsupported_expression_nodes,
            build_sass_expression_summary,
            _verify_launch_coordinate_domains,
            verify_sass_expression_certificate,
            verify_sass_expression_summary,
        )

        semantics = json.loads(
            (Path(__file__).parents[1] / "results" / "sass_semantics_proofs.json").read_text(
                encoding="utf-8"
            )
        )
        semantics_snapshot = _build_expression_semantics_snapshot(semantics)
        nsight = {
            "certificate_sha256": "a" * 64,
            "details": {
                "block_size": "(32, 4, 1)",
                "grid_size": "(1, 4, 1)",
                "metrics": {
                    "Block Size": {"value": "128"},
                    "Grid Size": {"value": "4"},
                },
            },
        }
        launch_domains = _build_launch_coordinate_domains(nsight)
        self.assertTrue(_verify_launch_coordinate_domains(launch_domains))
        damaged_launch_domains = copy.deepcopy(launch_domains)
        damaged_launch_domains["block_size"][0] = 31
        self.assertFalse(_verify_launch_coordinate_domains(damaged_launch_domains))
        missing_obligation = copy.deepcopy(semantics)
        missing_obligation["proofs"] = [
            proof
            for proof in missing_obligation["proofs"]
            if proof["name"] != "mov_is_identity"
        ]
        missing_body = {
            key: value
            for key, value in missing_obligation.items()
            if key != "certificate_sha256"
        }
        missing_obligation["certificate_sha256"] = hashlib.sha256(
            canonical_json(missing_body).encode("utf-8")
        ).hexdigest()
        with self.assertRaises(ValueError):
            _build_expression_semantics_snapshot(missing_obligation)
        self.assertIsNone(
            _semantic_requirement("LOP3.LUT", "R4,R5,0x96,RZ,0xc0,!PT")
        )
        self.assertIsNone(_semantic_requirement("LOP3.LUT", "R4,0x96,!PT"))
        self.assertIsNone(_semantic_requirement("IMAD.U32", "R4,R5"))
        self.assertIsNone(
            _semantic_requirement("UIADD3", "UR8,UP1,UR11,UR8,URZ")
        )
        self.assertEqual(
            _semantic_requirement("UIADD3", "UR4,UR35,UR4,URZ"),
            [
                "uniform_iadd3_matches_ripple_carry_sum",
                "uniform_iadd3_full_32bit_definition_instance",
            ],
        )
        self.assertIsNone(_semantic_requirement("UIMAD", "R27,R26,R9,R31"))
        self.assertEqual(
            _semantic_requirement("UIMAD", "UR27,UR26,UR9,UR31"),
            [
                "uniform_imad_matches_shift_add_multiply_accumulate",
                "uniform_imad_full_32bit_definition_instance",
            ],
        )
        self.assertEqual(
            _semantic_requirement("UMOV", "UR41,URZ"),
            ["uniform_move_shares_proposed_identity_equation"],
        )
        self.assertIsNone(_semantic_requirement("ULDC", "R4,c[0x0][0x1e8]"))
        self.assertEqual(
            _semantic_requirement("ULDC", "UR4,c[0x0][0x1e8]"),
            ["uldc_matches_little_endian_constant_memory_read"],
        )
        self.assertEqual(
            _semantic_requirement("ULDC.U8", "UR17,c[0x0][0x1d4]"),
            ["uldc_u8_matches_little_endian_constant_memory_read"],
        )
        transfer_obligations = [
            "register_transfer_shares_proposed_bit_copy_equation",
            "register_transfer_full_32bit_definition_instance",
        ]
        self.assertEqual(_semantic_requirement("S2R", "R5,SR_TID.Y"), transfer_obligations)
        self.assertEqual(
            _semantic_requirement("S2UR", "UR19,SR_CTAID.X"),
            transfer_obligations,
        )
        self.assertEqual(_semantic_requirement("R2UR", "UR28,R6"), transfer_obligations)
        self.assertIsNone(_semantic_requirement("S2R", "UR5,SR_TID.Y"))
        self.assertEqual(_special_registers("R5,SR_TID.Y"), ["SR_TID.Y"])
        self.assertEqual(
            _special_registers("R5,SR_MACHINE_ID_0"), ["SR_MACHINE_ID_0"]
        )
        self.assertEqual(
            _semantic_requirement("S2R", "R5,SR_MACHINE_ID_0"),
            transfer_obligations,
        )
        self.assertEqual(
            _semantic_requirement("LOP3.LUT", "R4,R5,R6,RZ,0x96,!PT"),
            ["lop3_lut_0x96_is_three_input_xor"],
        )
        instructions = [
            {"offset": 0, "predicate": None, "opcode": "ULDC.64", "operands": "UR2,c[0x0][0x160]"},
            {"offset": 16, "predicate": None, "opcode": "MOV", "operands": "R4,UR2"},
            {"offset": 32, "predicate": None, "opcode": "LDG.E", "operands": "R6,[R4.64]"},
        ]
        snapshots, registry, graph = _call_string_expression_snapshots(
            instructions,
            0x160,
            semantics_snapshot=semantics_snapshot,
            launch_coordinate_domains=launch_domains,
        )
        roots = [
            snapshots[32][0]["definitions"]["R4"][0],
            snapshots[32][0]["definitions"]["R5"][0],
        ]
        self.assertEqual(graph["call_context_count"], 1)
        atomic = _transfer_definitions(
            {"offset": 48, "predicate": None, "opcode": "ATOM.E.ADD", "operands": "R4,[R8.64],R10"},
            3,
            (0, ()),
            {"R4": frozenset({"old"})},
        )
        self.assertNotEqual(atomic["R4"], frozenset({"old"}))
        predicated = _transfer_definitions(
            {"offset": 64, "predicate": "@P0", "opcode": "MOV", "operands": "R4,R8"},
            4,
            (0, ()),
            {"R4": frozenset({"old"})},
        )
        self.assertIn("old", predicated["R4"])
        self.assertEqual(len(predicated["R4"]), 2)
        nodes = _reachable_nodes(roots, registry)
        self.assertIn("query_ptr", {field for node in nodes for field in node.get("parameter_fields", [])})
        special_instructions = [
            {"offset": 0, "predicate": None, "opcode": "S2R", "operands": "R4,SR_TID.X"},
            {"offset": 16, "predicate": None, "opcode": "MOV", "operands": "R6,R4"},
            {"offset": 32, "predicate": None, "opcode": "LDG.E", "operands": "R8,[R6.64]"},
        ]
        special_snapshots, special_registry, _ = _call_string_expression_snapshots(
            special_instructions,
            0x160,
            semantics_snapshot=semantics_snapshot,
            launch_coordinate_domains=launch_domains,
        )
        special_roots = [
            definition
            for definitions in special_snapshots[32][0]["definitions"].values()
            for definition in definitions
        ]
        special_nodes = _reachable_nodes(special_roots, special_registry)
        special_leaf = next(node for node in special_nodes if node["kind"] == "special_register")
        self.assertEqual(special_leaf["launch_domain_assumption"]["maximum_exclusive"], 32)
        self.assertFalse(
            special_leaf["launch_domain_assumption"]["coordinate_correspondence_established"]
        )
        unknown_special = [
            {"offset": 0, "predicate": None, "opcode": "S2R", "operands": "R4,SR_MACHINE_ID_0"},
            {"offset": 16, "predicate": None, "opcode": "LDG.E", "operands": "R6,[R4.64]"},
        ]
        unknown_snapshots, unknown_registry, _ = _call_string_expression_snapshots(
            unknown_special,
            0x160,
            semantics_snapshot=semantics_snapshot,
            launch_coordinate_domains=launch_domains,
        )
        unknown_roots = [
            definition
            for definitions in unknown_snapshots[16][0]["definitions"].values()
            for definition in definitions
        ]
        unknown_nodes = _reachable_nodes(unknown_roots, unknown_registry)
        unknown_leaf = next(node for node in unknown_nodes if node["kind"] == "special_register")
        self.assertIsNone(unknown_leaf["launch_domain_assumption"])
        certificate = {
            "scope": "synthetic expression DAG",
            "kernel_name": "kernel",
            "cubin_sha256": "cubin",
            "sass_canonical_sha256": "sass",
            "sass_memory_certificate_sha256": "memory",
            "nsight_certificate_sha256": nsight["certificate_sha256"],
            "launch_coordinate_domains": launch_domains,
            "sass_semantics_certificate_sha256": semantics["certificate_sha256"],
            "expression_opcode_semantics": semantics_snapshot,
            "logical_bounds_certificate_sha256": "bounds",
            "expression_reaching_definition_summary": graph,
            "selections": [
                {
                    "field": "query_ptr",
                    "available": True,
                    "closed_supported_formula": False,
                    "expression_nodes": nodes,
                    "root_nodes": roots,
                    "node_count": len(nodes),
                    "represented_parameter_fields": ["query_ptr"],
                    "target_field_represented": True,
                    "ambiguous_reaching_definition_node_count": 0,
                    "cyclic_reaching_definition_node_count": 0,
                    "special_register_leaf_count": 0,
                    "coordinate_special_register_leaf_count": 0,
                    "launch_domain_bound_special_register_leaf_count": 0,
                    "instruction_definition_node_count": 2,
                    "proof_backed_instruction_node_count": 2,
                    "unmodeled_instruction_node_count": 0,
                    "unmodeled_opcode_histogram": {},
                    "referenced_semantics_obligations": [
                        "mov_is_identity",
                        "uldc64_matches_little_endian_constant_memory_read",
                    ],
                    "unsupported_or_entry_node_count": len(_unsupported_expression_nodes(nodes)),
                }
            ],
            "checks": {"synthetic": True},
            "all_checks_pass": True,
            "selected_sass_address_expression_dags_established": True,
            "bounded_call_string_expression_reaching_definitions_established": True,
            "proposed_semantics_proof_bindings_established": True,
            "proof_premises_established_for_bound_instructions": False,
            "special_register_launch_domain_assumptions_bound": True,
            "special_register_coordinate_correspondence_established": False,
            "special_register_concrete_values_established": False,
            "special_register_hardware_acquisition_established": False,
            "all_expression_instruction_semantics_bound": True,
            "expression_call_string_depth_overflow_free": True,
            "unbounded_context_sensitive_expression_reaching_definitions_established": False,
            "hardware_instruction_semantics_established": False,
            "closed_supported_sass_formulas_established": False,
            "sass_effective_address_formula_bound": False,
            "sass_to_logical_stride_correspondence_established": False,
            "sass_effective_address_bounds_established": False,
            "kernel_memory_safety_established": False,
        }
        certificate["certificate_sha256"] = hashlib.sha256(canonical_json(certificate).encode("utf-8")).hexdigest()
        verification = verify_sass_expression_certificate(certificate)
        self.assertTrue(verification["valid"], verification)
        summary = build_sass_expression_summary(certificate)
        self.assertTrue(verify_sass_expression_summary(summary)["valid"])
        forged_summary = copy.deepcopy(summary)
        forged_summary["special_register_coordinate_correspondence_established"] = True
        summary_body = {
            key: value for key, value in forged_summary.items() if key != "summary_sha256"
        }
        forged_summary["summary_sha256"] = hashlib.sha256(
            canonical_json(summary_body).encode("utf-8")
        ).hexdigest()
        self.assertFalse(verify_sass_expression_summary(forged_summary)["valid"])
        damaged = copy.deepcopy(certificate)
        damaged["selections"][0]["expression_nodes"][0]["kind"] = "forged"
        body = {key: value for key, value in damaged.items() if key != "certificate_sha256"}
        damaged["certificate_sha256"] = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
        self.assertFalse(verify_sass_expression_certificate(damaged)["valid"])
        with self.assertRaises(ValueError):
            build_sass_expression_summary(damaged)
        forged_semantics = copy.deepcopy(certificate)
        proof = forged_semantics["expression_opcode_semantics"]["obligations"][0]
        proof["proved"] = False
        proof_body = {
            key: value for key, value in proof.items() if key != "proof_record_sha256"
        }
        proof["proof_record_sha256"] = hashlib.sha256(
            canonical_json(proof_body).encode("utf-8")
        ).hexdigest()
        snapshot = forged_semantics["expression_opcode_semantics"]
        snapshot_body = {
            key: value for key, value in snapshot.items() if key != "snapshot_sha256"
        }
        snapshot["snapshot_sha256"] = hashlib.sha256(
            canonical_json(snapshot_body).encode("utf-8")
        ).hexdigest()
        forged_body = {
            key: value for key, value in forged_semantics.items() if key != "certificate_sha256"
        }
        forged_semantics["certificate_sha256"] = hashlib.sha256(
            canonical_json(forged_body).encode("utf-8")
        ).hexdigest()
        self.assertFalse(verify_sass_expression_certificate(forged_semantics)["valid"])
        damaged_domain = copy.deepcopy(certificate)
        damaged_domain["launch_coordinate_domains"]["block_size"][0] = 31
        damaged_body = {
            key: value for key, value in damaged_domain.items() if key != "certificate_sha256"
        }
        damaged_domain["certificate_sha256"] = hashlib.sha256(
            canonical_json(damaged_body).encode("utf-8")
        ).hexdigest()
        self.assertFalse(verify_sass_expression_certificate(damaged_domain)["valid"])

    def test_expression_reaching_definitions_preserve_branch_ambiguity(self) -> None:
        from bioprocess_runtime.sass_expressions import (
            _call_string_expression_snapshots,
            _reachable_nodes,
        )

        instructions = [
            {"offset": 0, "predicate": None, "opcode": "ULDC.64", "operands": "UR2,c[0x0][0x160]"},
            {"offset": 16, "predicate": "@P0", "opcode": "BRA.U", "operands": "0x40"},
            {"offset": 32, "predicate": None, "opcode": "ULDC.64", "operands": "UR2,c[0x0][0x168]"},
            {"offset": 48, "predicate": None, "opcode": "BRA", "operands": "0x50"},
            {"offset": 64, "predicate": None, "opcode": "MOV", "operands": "UR2,UR2"},
            {"offset": 80, "predicate": None, "opcode": "MOV", "operands": "R4,UR2"},
            {"offset": 96, "predicate": None, "opcode": "LDG.E", "operands": "R6,[R4.64]"},
        ]
        snapshots, registry, _ = _call_string_expression_snapshots(instructions, 0x160)
        roots = [
            definition
            for definitions in snapshots[96][0]["definitions"].values()
            for definition in definitions
        ]
        nodes = _reachable_nodes(roots, registry)
        self.assertIn("reaching_definition_join", {node["kind"] for node in nodes})
        self.assertEqual(
            {field for node in nodes for field in node.get("parameter_fields", [])},
            {"query_ptr", "key_ptr"},
        )

    def test_expression_reaching_definitions_return_to_call_context(self) -> None:
        from bioprocess_runtime.sass_expressions import (
            _call_string_expression_snapshots,
            _reachable_nodes,
        )

        instructions = [
            {"offset": 0, "predicate": None, "opcode": "ULDC.64", "operands": "UR2,c[0x0][0x160]"},
            {"offset": 16, "predicate": None, "opcode": "CALL.REL.NOINC", "operands": "0x50"},
            {"offset": 32, "predicate": None, "opcode": "MOV", "operands": "R4,UR2"},
            {"offset": 48, "predicate": None, "opcode": "LDG.E", "operands": "R6,[R4.64]"},
            {"offset": 64, "predicate": None, "opcode": "EXIT", "operands": ""},
            {"offset": 80, "predicate": None, "opcode": "ULDC.64", "operands": "UR2,c[0x0][0x168]"},
            {"offset": 96, "predicate": None, "opcode": "RET.REL.NODEC", "operands": "R2,0x0"},
        ]
        snapshots, registry, graph = _call_string_expression_snapshots(
            instructions, 0x160, maximum_call_depth=2
        )
        roots = [
            definition
            for definitions in snapshots[48][0]["definitions"].values()
            for definition in definitions
        ]
        nodes = _reachable_nodes(roots, registry)
        self.assertEqual(
            {field for node in nodes for field in node.get("parameter_fields", [])},
            {"key_ptr"},
        )
        self.assertEqual(graph["maximum_observed_call_depth"], 1)
        self.assertEqual(graph["unresolved_return_context_count"], 0)


class SassMemoryTests(unittest.TestCase):
    def test_parameter_base_and_linear_address_taint(self) -> None:
        from bioprocess_runtime.sass_memory import (
            MEMORY_BASES,
            _address_operand_specs,
            _address_taint_slices,
            _base_opcode,
            _cfg_address_taint_slices,
            _derive_parameter_base,
            _destination_register_count,
            _memory_slice,
            _memory_width_bytes,
            _registers,
            _source_registers_for_opcode,
            _transfer_taint,
        )

        instructions = [
            {"offset": 0, "opcode": "ULDC.64", "operands": "UR2,c[0x0][0x160]"},
            {"offset": 16, "opcode": "ULDC.64", "operands": "UR4,c[0x0][0x168]"},
            {"offset": 32, "opcode": "ULDC.64", "operands": "UR6,c[0x0][0x170]"},
            {"offset": 48, "opcode": "ULDC.64", "operands": "UR8,c[0x0][0x1a0]"},
            {"offset": 64, "opcode": "ULDC.64", "operands": "UR10,c[0x0][0x1a8]"},
            {"offset": 80, "opcode": "MOV", "operands": "R4,UR2"},
            {"offset": 96, "opcode": "LDG.E", "operands": "R6,[R4.64]"},
            {"offset": 112, "opcode": "IMAD.WIDE", "operands": "R8,R6,0x4,RZ"},
            {"offset": 128, "opcode": "STG.E", "operands": "[R8.64],R10"},
            {"offset": 144, "opcode": "LDGDEPBAR", "operands": ""},
        ]
        base = _derive_parameter_base(instructions)
        self.assertEqual(base["base_constant_offset"], 0x160)
        slices = _address_taint_slices(instructions, base["base_constant_offset"])
        self.assertEqual(slices[0]["source_parameter_fields"], ["query_ptr"])
        self.assertEqual(slices[1]["source_parameter_fields"], [])
        self.assertEqual(len(slices), 2)
        self.assertNotIn(_base_opcode("LDGDEPBAR"), MEMORY_BASES)
        self.assertIn(_base_opcode("LDGSTS.E.BYPASS.LTC128B.128"), MEMORY_BASES)
        self.assertIn(_base_opcode("ST.E"), MEMORY_BASES)
        self.assertEqual(_memory_width_bytes("LDG.E"), 4)
        self.assertEqual(_memory_width_bytes("LDG.E.64"), 8)
        self.assertEqual(_memory_width_bytes("LDG.E.LTC128B.128"), 16)
        cfg_slices, graph = _cfg_address_taint_slices(instructions, base["base_constant_offset"])
        self.assertEqual(cfg_slices, slices)
        self.assertEqual(graph["unresolved_direct_targets"], 0)
        stale = {f"R{index}": {"query_ptr"} for index in range(4, 8)}
        cleared = _transfer_taint(
            {"offset": 0, "predicate": None, "opcode": "LDG.E.LTC128B.128", "operands": "R4,[R20.64]"},
            stale,
            0x160,
        )
        self.assertTrue(all(not cleared[f"R{index}"] for index in range(4, 8)))
        self.assertEqual(_destination_register_count("LDSM.16.M88.4"), 4)
        self.assertEqual(_destination_register_count("IMAD.WIDE.U32"), 2)
        self.assertEqual(_registers("[R4.64]"), ["R4", "R5"])
        self.assertEqual(_registers("[UR14.128]"), ["UR14", "UR15", "UR16", "UR17"])
        self.assertEqual(
            _source_registers_for_opcode("IMAD.WIDE", ",R6,0x4,RZ"), ["R6"]
        )
        self.assertEqual(
            _source_registers_for_opcode("IMAD.WIDE", ",R6,0x4,R20"),
            ["R6", "R20", "R21"],
        )
        cleared = _transfer_taint(
            {"offset": 0, "predicate": None, "opcode": "LDSM.16.M88.4", "operands": "R4,[R20]"},
            stale,
            0x160,
        )
        self.assertTrue(all(not cleared[f"R{index}"] for index in range(4, 8)))
        self.assertEqual(
            _address_operand_specs("LDGSTS.E.BYPASS.LTC128B.128", ["R2", "R4.64"]),
            [("candidate_write", "shared", "R2"), ("candidate_read", "global", "R4.64")],
        )
        memory_slice = _memory_slice(
            {
                "offset": 0,
                "opcode": "LDGSTS.E.BYPASS.LTC128B.128",
                "operands": "[R2],[R4.64],P0",
            },
            {"R2": {"output_ptr"}, "R4": {"query_ptr"}},
        )
        self.assertEqual(memory_slice["address_operands"][0]["source_parameter_fields"], ["output_ptr"])
        self.assertEqual(memory_slice["address_operands"][1]["source_parameter_fields"], ["query_ptr"])

    def test_cfg_joins_direct_branch_reaching_fields(self) -> None:
        from bioprocess_runtime.sass_memory import _cfg_address_taint_slices

        instructions = [
            {"offset": 0, "predicate": None, "opcode": "ULDC.64", "operands": "UR2,c[0x0][0x160]"},
            {"offset": 16, "predicate": "@P0", "opcode": "BRA.U", "operands": "0x40"},
            {"offset": 32, "predicate": None, "opcode": "ULDC.64", "operands": "UR2,c[0x0][0x168]"},
            {"offset": 48, "predicate": None, "opcode": "BRA", "operands": "0x50"},
            {"offset": 64, "predicate": None, "opcode": "MOV", "operands": "UR2,UR2"},
            {"offset": 80, "predicate": None, "opcode": "MOV", "operands": "R4,UR2"},
            {"offset": 96, "predicate": None, "opcode": "LDG.E", "operands": "R6,[R4.64]"},
        ]
        slices, graph = _cfg_address_taint_slices(instructions, 0x160)
        self.assertEqual(slices[0]["source_parameter_fields"], ["key_ptr", "query_ptr"])
        self.assertEqual(graph["direct_branches"], 2)
        self.assertEqual(graph["unresolved_direct_targets"], 0)
        self.assertGreater(graph["basic_block_count"], 1)

    def test_call_return_and_barrier_edges_are_conservative(self) -> None:
        from bioprocess_runtime.sass_memory import _cfg_successors

        instructions = [
            {"offset": 0, "predicate": None, "opcode": "CALL.REL.NOINC", "operands": "0x40"},
            {"offset": 16, "predicate": None, "opcode": "EXIT", "operands": ""},
            {"offset": 32, "predicate": None, "opcode": "NOP", "operands": ""},
            {"offset": 48, "predicate": None, "opcode": "NOP", "operands": ""},
            {"offset": 64, "predicate": None, "opcode": "BSSY", "operands": "B0,0x80"},
            {"offset": 80, "predicate": "@P0", "opcode": "BREAK", "operands": "B0"},
            {"offset": 96, "predicate": None, "opcode": "BSYNC", "operands": "B0"},
            {"offset": 112, "predicate": None, "opcode": "RET.REL.NODEC", "operands": "R2,0x0"},
            {"offset": 128, "predicate": None, "opcode": "EXIT", "operands": ""},
        ]
        successors, graph = _cfg_successors(instructions)
        self.assertEqual(successors[0], [1, 4])
        self.assertEqual(successors[5], [6, 8])
        self.assertEqual(successors[6], [8])
        self.assertEqual(successors[7], [1])
        self.assertEqual(graph["context_insensitive_return_edges"], 1)
        self.assertEqual(graph["lexically_matched_bssy_bsync"], 1)
        self.assertEqual(graph["unmatched_barrier_controls"], 0)

    def test_bounded_call_string_returns_to_callsite(self) -> None:
        from bioprocess_runtime.sass_memory import _call_string_address_taint_slices

        instructions = [
            {"offset": 0, "predicate": None, "opcode": "ULDC.64", "operands": "UR2,c[0x0][0x160]"},
            {"offset": 16, "predicate": None, "opcode": "CALL.REL.NOINC", "operands": "0x50"},
            {"offset": 32, "predicate": None, "opcode": "MOV", "operands": "R4,UR2"},
            {"offset": 48, "predicate": None, "opcode": "LDG.E", "operands": "R6,[R4.64]"},
            {"offset": 64, "predicate": None, "opcode": "EXIT", "operands": ""},
            {"offset": 80, "predicate": None, "opcode": "ULDC.64", "operands": "UR2,c[0x0][0x168]"},
            {"offset": 96, "predicate": None, "opcode": "RET.REL.NODEC", "operands": "R2,0x0"},
        ]
        slices, graph = _call_string_address_taint_slices(instructions, 0x160, maximum_call_depth=2)
        self.assertEqual(slices[0]["source_parameter_fields"], ["key_ptr"])
        self.assertEqual(graph["maximum_observed_call_depth"], 1)
        self.assertEqual(graph["abstracted_call_overflow_count"], 0)
        self.assertEqual(graph["unresolved_return_context_count"], 0)
        recursive = [
            {"offset": 0, "predicate": None, "opcode": "CALL.REL.NOINC", "operands": "0x0"},
            {"offset": 16, "predicate": None, "opcode": "EXIT", "operands": ""},
        ]
        slices, graph = _call_string_address_taint_slices(recursive, 0x160, maximum_call_depth=1)
        self.assertGreater(graph["abstracted_call_overflow_count"], 0)
        self.assertEqual(graph["maximum_observed_call_depth"], 1)

    def test_predicated_terminators_and_indirect_branches(self) -> None:
        from bioprocess_runtime.sass_memory import _cfg_successors

        instructions = [
            {"offset": 0, "predicate": "@P0", "opcode": "RET.REL.NODEC", "operands": "R2,0x0"},
            {"offset": 16, "predicate": None, "opcode": "NOP", "operands": ""},
            {"offset": 32, "predicate": "@!P0", "opcode": "EXIT", "operands": ""},
            {"offset": 48, "predicate": None, "opcode": "NOP", "operands": ""},
            {"offset": 64, "predicate": None, "opcode": "BRX", "operands": "R2,0x0"},
            {"offset": 80, "predicate": None, "opcode": "NOP", "operands": ""},
        ]
        successors, graph = _cfg_successors(instructions)
        self.assertEqual(successors[0], [1])
        self.assertEqual(successors[2], [3])
        self.assertEqual(successors[4], [])
        self.assertEqual(graph["unresolved_indirect_transfers"], 1)
        final_call = [
            {"offset": 0, "predicate": None, "opcode": "RET.REL.NODEC", "operands": "R2,0x0"},
            {"offset": 16, "predicate": None, "opcode": "CALL.REL.NOINC", "operands": "0x0"},
        ]
        successors, graph = _cfg_successors(final_call)
        self.assertEqual(graph["direct_calls"], 1)
        self.assertEqual(graph["direct_call_fallthroughs"], 0)
        self.assertEqual(graph["context_insensitive_return_edges"], 0)


@unittest.skipUnless(Z3_AVAILABLE, "Proof optional dependencies are not installed")
class AttentionLogicalBoundsTests(unittest.TestCase):
    def test_logical_index_bounds_detect_storage_overrun(self) -> None:
        from bioprocess_runtime.attention_bounds import _prove_tensor_storage_bound

        tensor = {
            "shape": [2, 3],
            "stride": [3, 1],
            "element_size_bytes": 2,
            "storage_nbytes": 12,
            "data_pointer_offset_bytes": 0,
            "sha256": "logical",
            "storage_base_pointer_sha256": "base",
            "data_pointer_sha256": "data",
        }
        self.assertTrue(_prove_tensor_storage_bound("bounded", tensor)["proved"])
        tensor["storage_nbytes"] = 10
        proof = _prove_tensor_storage_bound("overrun", tensor)
        self.assertFalse(proof["proved"])
        self.assertIsNotNone(proof["counterexample"])

    def test_full_bounds_certificate_reexecutes_redacts_and_detects_tampering(self) -> None:
        from bioprocess_runtime.attention_bounds import (
            build_attention_logical_bounds_certificate,
            redact_attention_logical_bounds_certificate,
            verify_attention_logical_bounds_certificate,
        )

        def tensor(name: str, stride: list[int]) -> dict[str, Any]:
            return {
                "shape": [1, 4, 30, 256],
                "stride": stride,
                "element_size_bytes": 2,
                "storage_nbytes": 61440,
                "data_pointer_offset_bytes": 0,
                "sha256": name,
                "storage_base_pointer_sha256": f"{name}_base",
                "data_pointer_sha256": f"{name}_data",
            }

        qkv_stride = [30720, 7680, 256, 1]
        output_stride = [30720, 256, 1024, 1]
        artifact = {
            "artifact_sha256": "artifact",
            "attention_dispatch_report": {
                "operations": [
                    {
                        "input_names": ["query", "key", "value"],
                        "inputs": [tensor("query", qkv_stride), tensor("key", qkv_stride), tensor("value", qkv_stride)],
                        "outputs": [tensor("output", output_stride)],
                    }
                ]
            },
        }
        attention = {
            "certificate_sha256": "attention",
            "qkv_pointer_and_logical_commitments_bound": True,
            "source_named_dispatch_output_pointer_bound": True,
            "decoded_scalars": {
                "q_strideB": 30720,
                "q_strideH": 7680,
                "q_strideM": 256,
                "k_strideB": 30720,
                "k_strideH": 7680,
                "k_strideM": 256,
                "v_strideB": 30720,
                "v_strideH": 7680,
                "v_strideM": 256,
                "num_queries": 30,
                "o_strideM": 1024,
                "head_dim_value": 256,
            },
        }
        with (
            patch("bioprocess_runtime.attention_bounds.verify_launch_argument_artifact", return_value={"valid": True}),
            patch("bioprocess_runtime.attention_bounds.verify_attention_parameter_certificate", return_value={"valid": True}),
        ):
            certificate = build_attention_logical_bounds_certificate(artifact, attention)
        self.assertTrue(certificate["checks"]["decoded_strides_match_retained_tensors"])
        self.assertTrue(verify_attention_logical_bounds_certificate(certificate)["valid"])
        wrong_output = copy.deepcopy(artifact)
        wrong_output["attention_dispatch_report"]["operations"][0]["outputs"][0]["stride"] = qkv_stride
        with (
            patch("bioprocess_runtime.attention_bounds.verify_launch_argument_artifact", return_value={"valid": True}),
            patch("bioprocess_runtime.attention_bounds.verify_attention_parameter_certificate", return_value={"valid": True}),
        ):
            mismatch = build_attention_logical_bounds_certificate(wrong_output, attention)
        self.assertFalse(mismatch["checks"]["decoded_strides_match_retained_tensors"])
        self.assertFalse(mismatch["all_checks_pass"])
        redacted = redact_attention_logical_bounds_certificate(certificate)
        self.assertEqual(redacted["proofs"][0]["data_pointer_sha256"], "redacted")
        self.assertTrue(verify_attention_logical_bounds_certificate(redacted)["valid"])
        damaged = copy.deepcopy(certificate)
        damaged["proofs"][0]["storage_nbytes"] = 1
        body = {key: value for key, value in damaged.items() if key != "certificate_sha256"}
        damaged["certificate_sha256"] = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
        self.assertFalse(verify_attention_logical_bounds_certificate(damaged)["valid"])
        damaged = copy.deepcopy(certificate)
        damaged["logical_index_storage_bounds_established"] = False
        body = {key: value for key, value in damaged.items() if key != "certificate_sha256"}
        damaged["certificate_sha256"] = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
        self.assertFalse(verify_attention_logical_bounds_certificate(damaged)["valid"])


class AttentionParameterTests(unittest.TestCase):
    def test_attention_decoder_and_certificate_boundaries(self) -> None:
        from bioprocess_runtime.attention_parameters import verify_attention_parameter_certificate
        from bioprocess_runtime.kernel_signatures import _AttentionParams, decode_attention_params

        parameters = _AttentionParams()
        parameters.query_ptr = 1
        parameters.key_ptr = 2
        parameters.value_ptr = 3
        parameters.output_ptr = 4
        parameters.head_dim = 256
        parameters.num_queries = 30
        parameters.scale = 0.0625
        decoded = decode_attention_params(bytes(parameters))
        self.assertEqual(decoded["size_bytes"], 264)
        self.assertEqual(decoded["scalars"]["head_dim"], 256)
        self.assertEqual(decoded["scalars"]["num_queries"], 30)
        self.assertEqual(decoded["scalars"]["scale"], 0.0625)
        certificate = {
            "checks": {"decoded": True},
            "all_checks_pass": True,
            "source_layout_reconstructed": True,
            "compiled_layout_verified": False,
            "typed_signature_established": False,
            "qkv_pointer_and_logical_commitments_bound": True,
            "source_named_dispatch_output_pointer_bound": True,
            "kernel_read_write_semantics_established": False,
            "memory_access_semantics_established": False,
            "query_pointer_temporal_observation": {
                "typed_output_binding": False,
                "resolved_as_dispatch_query_input": True,
            },
        }
        certificate["certificate_sha256"] = hashlib.sha256(canonical_json(certificate).encode("utf-8")).hexdigest()
        self.assertTrue(verify_attention_parameter_certificate(certificate)["valid"])
        damaged = copy.deepcopy(certificate)
        damaged["kernel_read_write_semantics_established"] = True
        body = {key: value for key, value in damaged.items() if key != "certificate_sha256"}
        damaged["certificate_sha256"] = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
        self.assertFalse(verify_attention_parameter_certificate(damaged)["valid"])


@unittest.skipUnless(TORCH_AVAILABLE, "Torch is not installed")
class KernelSignatureTests(unittest.TestCase):
    def test_gelu_signature_is_partially_typed_from_installed_header(self) -> None:
        from bioprocess_runtime.kernel_signatures import (
            build_kernel_signature_certificate,
            verify_kernel_signature_certificate,
        )

        summary = {
            "summary_sha256": "summary",
            "entries": [
                {
                    "kernel_name": "vectorized_elementwise_kernel_GeluCUDAKernelImpl_St5arrayIPcLy2E",
                    "expected_module": "model.layers.0.mlp.act_fn",
                    "parameter_sizes": [4, 1, 16],
                    "boundary_pointer_matches": [
                        {"parameter_index": 2, "parameter_byte_offset": 0, "role": "output"},
                        {"parameter_index": 2, "parameter_byte_offset": 8, "role": "input"},
                    ],
                },
                {
                    "kernel_name": "fmha_cutlass_AttentionKernel_Params",
                    "expected_module": "model.layers.0.self_attn",
                    "parameter_sizes": [264],
                    "boundary_pointer_matches": [
                        {"parameter_index": 0, "parameter_byte_offset": 0, "role": "output", "temporal_status": "post_launch_retrospective"},
                        {"parameter_index": 0, "parameter_byte_offset": 0, "role": "input", "temporal_status": "pre_launch_retained"},
                        {"parameter_index": 0, "parameter_byte_offset": 8, "role": "parameter", "temporal_status": "pre_launch_retained"},
                    ],
                },
            ],
        }
        certificate = build_kernel_signature_certificate(summary)
        self.assertEqual(certificate["typed_entries"], 0)
        self.assertEqual(certificate["partial_parameter_schema_entries"], 1)
        self.assertEqual(certificate["source_layout_reconstructed_entries"], 1)
        self.assertFalse(certificate["all_signatures_typed"])
        self.assertTrue(certificate["source"]["wheel_revision_matches"])
        self.assertEqual(certificate["entries"][0]["typed_fields"][2]["observed_boundary_role"], "output")
        self.assertEqual(certificate["entries"][1]["reconstructed_layout_size_bytes"], 264)
        self.assertEqual(len(certificate["entries"][1]["observed_role_conflicts"]), 1)
        self.assertEqual(certificate["entries"][1]["observed_role_conflicts"][0]["source_field"], "query_ptr")
        self.assertFalse(certificate["complete_field_semantics_established"])
        self.assertTrue(verify_kernel_signature_certificate(certificate)["valid"])
        damaged = copy.deepcopy(certificate)
        damaged["source"]["cutlass_gitlink_commit"] = "0" * 40
        body = {key: value for key, value in damaged.items() if key != "certificate_sha256"}
        damaged["certificate_sha256"] = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
        verification = verify_kernel_signature_certificate(damaged)
        self.assertFalse(verification["source_identities_valid"])
        self.assertFalse(verification["valid"])
        forged_fields = copy.deepcopy(certificate)
        forged_fields["source"]["attention_source_field_order"][0:2] = reversed(
            forged_fields["source"]["attention_source_field_order"][0:2]
        )
        body = {key: value for key, value in forged_fields.items() if key != "certificate_sha256"}
        forged_fields["certificate_sha256"] = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
        verification = verify_kernel_signature_certificate(forged_fields)
        self.assertFalse(verification["source_identities_valid"])
        self.assertFalse(verification["valid"])


class ModuleInvocationTests(unittest.TestCase):
    @unittest.skipUnless(TORCH_AVAILABLE, "Torch is not installed")
    def test_module_nvtx_capture_balances_normal_and_exception_paths(self) -> None:
        import torch

        from bioprocess_runtime.module_invocation import ModuleNvtxCapture

        class Child(torch.nn.Module):
            def __init__(self, fail: bool = False) -> None:
                super().__init__()
                self.fail = fail

            def forward(self, value: torch.Tensor) -> torch.Tensor:
                if self.fail:
                    raise RuntimeError("expected")
                return value + 1

        class Parent(torch.nn.Module):
            def __init__(self, fail: bool = False) -> None:
                super().__init__()
                self.child = Child(fail)

            def forward(self, value: torch.Tensor) -> torch.Tensor:
                return self.child(value)

        for fail in (False, True):
            model = Parent(fail)
            with (
                patch("bioprocess_runtime.module_invocation.torch.cuda.is_available", return_value=True),
                patch("bioprocess_runtime.module_invocation.torch.cuda.nvtx.range_push") as push,
                patch("bioprocess_runtime.module_invocation.torch.cuda.nvtx.range_pop") as pop,
            ):
                capture = ModuleNvtxCapture(model, ("child",))
                if fail:
                    with self.assertRaisesRegex(RuntimeError, "expected"):
                        with capture:
                            model(torch.tensor([1.0]))
                else:
                    with capture:
                        model(torch.tensor([1.0]))
                    self.assertTrue(capture.report()["invocations"])
                self.assertEqual(push.call_count, pop.call_count)
                self.assertFalse(capture.open_stack)
                self.assertFalse(capture.lifecycle_errors)

    @unittest.skipUnless(TORCH_AVAILABLE, "Torch is not installed")
    def test_attention_dispatch_capture_retains_qkv_and_outputs(self) -> None:
        import torch

        from bioprocess_runtime.module_invocation import AttentionDispatchCapture, verify_attention_dispatch_report

        class Function:
            def __str__(self) -> str:
                return "aten._scaled_dot_product_efficient_attention.default"

            def __call__(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor]:
                return (query + key + value,)

        module_capture = SimpleNamespace(open_stack=[{"module": "model.layers.0.self_attn"}])
        capture = AttentionDispatchCapture(module_capture)
        tensors = tuple(torch.ones((1, 1, 2, 2)) * value for value in (1, 2, 3))
        output = capture.__torch_dispatch__(Function(), (), tensors, {})
        self.assertEqual(len(capture.operations), 1)
        self.assertEqual([id(tensor) for tensor in capture.operations[0]["input_tensors"]], [id(tensor) for tensor in tensors])
        self.assertEqual(capture.operations[0]["input_names"], ["argument_0", "argument_1", "argument_2"])
        self.assertEqual(id(capture.operations[0]["output_tensors"][0]), id(output[0]))
        self.assertTrue(verify_attention_dispatch_report(capture.report())["valid"])

    def test_raw_nvtx_identity_extracts_qualified_module_stack(self) -> None:
        from bioprocess_runtime.module_invocation import _raw_nvtx_identity

        column = "thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg"
        output = (
            f'"Process ID","Kernel Name","{column}"\n'
            '"42","kernel","1  ""<default domain>:gemma_bound_forward:none:none:none:none:none:none""  ""<default domain>:gemma_module:model.layers.0.self_attn:none:none:none:none:none:none"" "\n'
        )
        identity = _raw_nvtx_identity(output)
        self.assertEqual(identity["process_id"], 42)
        self.assertEqual(identity["qualified_modules"], ["model.layers.0.self_attn"])

    def test_module_invocation_integrity_and_semantic_boundary(self) -> None:
        from bioprocess_runtime.module_invocation import (
            build_module_invocation_summary,
            verify_module_invocation_certificate,
            verify_module_invocation_report,
            verify_module_invocation_summary,
        )

        invocation = {"module": "module", "inputs": [{"sha256": "in"}], "outputs": [{"sha256": "out"}]}
        invocation["invocation_sha256"] = hashlib.sha256(canonical_json(invocation).encode("utf-8")).hexdigest()
        report = {"patterns": ["module"], "invocations": [invocation]}
        report["report_sha256"] = hashlib.sha256(canonical_json(report).encode("utf-8")).hexdigest()
        self.assertTrue(verify_module_invocation_report(report)["valid"])
        certificate = {
            "checks": {"bound": True},
            "all_checks_pass": True,
            "raw_identity": {"kernel_name": "kernel", "qualified_modules": ["module"]},
            "expected_innermost_module": "module",
            "matched_module_invocations": [invocation],
        }
        certificate["certificate_sha256"] = hashlib.sha256(canonical_json(certificate).encode("utf-8")).hexdigest()
        self.assertTrue(verify_module_invocation_certificate(certificate)["valid"])
        summary = build_module_invocation_summary([certificate])
        self.assertTrue(summary["complete"])
        self.assertFalse(summary["full_kernel_argument_binding_established"])
        self.assertTrue(verify_module_invocation_summary(summary)["valid"])
        damaged_summary = copy.deepcopy(summary)
        damaged_summary["full_kernel_argument_binding_established"] = True
        self.assertFalse(verify_module_invocation_summary(damaged_summary)["valid"])


class NsightAttestationTests(unittest.TestCase):
    def test_launch_details_and_sass_normalization(self) -> None:
        from bioprocess_runtime.nsight_attestation import (
            _parse_cuobjdump_sass,
            _parse_details,
            _parse_elf_function_symbols,
            _parse_nsight_sass,
        )

        details = (
            '"Process ID","Kernel Name","Context","Stream","Block Size","Grid Size","CC","Metric Name","Metric Unit","Metric Value"\n'
            '"42","kernel","1","7","(128, 1, 1)","(1, 1, 1)","8.9","Threads","thread","128"\n'
        )
        parsed = _parse_details(details)
        self.assertEqual(parsed["process_id"], 42)
        self.assertEqual(parsed["kernel_name"], "kernel")
        symbols = _parse_elf_function_symbols("STT_FUNC STB_GLOBAL STO_ENTRY exact\nSTT_OBJECT STB_GLOBAL STO_ENTRY ignored")
        self.assertEqual(symbols, {"exact"})
        nsight = (
            "0x100 IADD3 R0, R1, R2, R3\n"
            "0x110 @P0 BRA 0x100\n"
            "0x120 @PT NOP\n"
            "0x130 @!PT EXIT\n"
            "0x140 @UP0 MOV R0, R1\n"
            "0x150 @UPT NOP\n"
        )
        static = (
            "/*0000*/ IADD3 R0, R1, R2, R3 ;\n"
            "/*0010*/ @P0 BRA 0x0;\n"
            "/*0020*/ @PT NOP;\n"
            "/*0030*/ @!PT EXIT;\n"
            "/*0040*/ @UP0 MOV R0, R1;\n"
            "/*0050*/ @UPT NOP;\n"
        )
        self.assertEqual(_parse_nsight_sass(nsight), _parse_cuobjdump_sass(static))

    def test_tool_specific_sass_renderings_normalize_equally(self) -> None:
        from bioprocess_runtime.nsight_attestation import _parse_cuobjdump_sass, _parse_nsight_sass

        nsight = (
            "0x100 F2FP.F16.F32.PACK_AB R1, R2, R3\n"
            "0x110 BRA.U 0x130\n"
            "0x120 LDGSTS.E.128 P4, [R1][R2.64]\n"
            "0x130 LDG.E.64 R5, P3, [R6.64]\n"
            "0x140 BRX R7, -0x20\n"
            "0x150 RET.REL.NODEC R2, 0x100\n"
        )
        static = (
            "/*0000*/ F2FP.PACK_AB R1, R2.reuse, R3;\n"
            "/*0010*/ BRA.U 0x30;\n"
            "/*0020*/ LDGSTS.E.128 [R1], [R2.64], P4;\n"
            "/*0030*/ LDG.E.64 R5, [R6.64], P3;\n"
            "/*0040*/ BRX R7 -0x20;\n"
            "/*0050*/ RET.REL.NODEC R2 0x0;\n"
        )
        self.assertEqual(_parse_nsight_sass(nsight), _parse_cuobjdump_sass(static))

    def test_sass_normalization_preserves_semantic_differences(self) -> None:
        from bioprocess_runtime.nsight_attestation import _parse_nsight_sass

        base = _parse_nsight_sass("0x100 LDG.E.64 R5, P3, [R6.64]\n0x110 BRA.U 0x130\n")
        different_destination = _parse_nsight_sass("0x100 LDG.E.64 R7, P3, [R6.64]\n0x110 BRA.U 0x130\n")
        different_predicate = _parse_nsight_sass("0x100 LDG.E.64 R5, P4, [R6.64]\n0x110 BRA.U 0x130\n")
        different_target = _parse_nsight_sass("0x100 LDG.E.64 R5, P3, [R6.64]\n0x110 BRA.U 0x140\n")
        self.assertNotEqual(base, different_destination)
        self.assertNotEqual(base, different_predicate)
        self.assertNotEqual(base, different_target)

    def test_kernel_suite_verifier_recomputes_coverage_claims(self) -> None:
        from bioprocess_runtime.nsight_attestation import verify_nsight_kernel_suite

        certificate = {
            "checks": {"launch_sass_matches_loaded_cubin_function": True},
            "all_checks_pass": True,
            "details": {"kernel_name": "gelu_kernel"},
            "cupti_module": {"cubin_sha256": "cubin"},
            "launch_sass": {"canonical_sha256": "sass", "instruction_count": 10},
            "loaded_cubin_function_sass": {"canonical_sha256": "sass"},
        }
        certificate["certificate_sha256"] = hashlib.sha256(canonical_json(certificate).encode("utf-8")).hexdigest()
        entry = {
            "index": 0,
            "kernel_name": "gelu_kernel",
            "family": "gelu",
            "observed_forward_launch_count": 18,
            "instruction_count": 10,
            "syntactic_proposed_semantics_opcode_lines": 2,
            "certificate_sha256": certificate["certificate_sha256"],
            "cubin_sha256": "cubin",
            "sass_canonical_sha256": "sass",
            "session_exact_kernel_filter_value_visible": True,
            "capture_request_sha256": None,
        }
        suite = {
            "expected_distinct_kernels": 1,
            "attested_distinct_kernels": 1,
            "session_exact_filter_value_kernels": 1,
            "capture_request_backed_kernels": 0,
            "distinct_coverage_fraction": 1.0,
            "expected_kernel_launches": 18,
            "attested_kernel_launches": 18,
            "launch_weighted_coverage_fraction": 1.0,
            "distinct_function_instruction_lines": 10,
            "syntactic_proposed_semantics_opcode_lines": 2,
            "syntactic_proposed_semantics_opcode_fraction": 0.2,
            "families": {
                "gelu": {
                    "expected_distinct": 1,
                    "attested_distinct": 1,
                    "expected_launches": 18,
                    "attested_launches": 18,
                }
            },
            "entries": [entry],
            "failures": [],
            "complete": True,
        }
        suite["suite_sha256"] = hashlib.sha256(canonical_json(suite).encode("utf-8")).hexdigest()
        self.assertTrue(verify_nsight_kernel_suite(suite)["valid"])
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "k00_certificate.json").write_text(json.dumps(certificate), encoding="utf-8")
            verification = verify_nsight_kernel_suite(
                suite,
                Path(directory),
                {"modules": [{"cubin_sha256": "cubin"}]},
            )
            self.assertTrue(verification["per_certificate_claims_valid"])
            self.assertTrue(verification["cupti_links_valid"])
            self.assertTrue(verification["valid"])
        damaged = copy.deepcopy(suite)
        damaged["launch_weighted_coverage_fraction"] = 0.5
        self.assertFalse(verify_nsight_kernel_suite(damaged)["valid"])

    def test_capture_request_binds_exact_kernel_before_execution(self) -> None:
        from bioprocess_runtime.nsight_attestation import capture_nsight_launch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_base = root / "report"
            binding = root / "binding.json"
            request_path = root / "request.json"

            def run(command: list[str], **kwargs: object) -> SimpleNamespace:
                self.assertIn("--kernel-name", command)
                self.assertEqual(command[command.index("--kernel-name") + 1], "exact_kernel")
                report_base.with_suffix(".ncu-rep").write_bytes(b"report")
                binding.write_text("{}", encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with (
                patch("bioprocess_runtime.nsight_attestation.shutil.which", return_value="ncu"),
                patch("bioprocess_runtime.nsight_attestation.subprocess.run", side_effect=run),
            ):
                request = capture_nsight_launch(
                    "exact_kernel", Path("model"), "synthetic prompt", report_base, binding, request_path
                )
            stored = json.loads(request_path.read_text(encoding="utf-8"))
            self.assertEqual(stored, request)
            body = {key: value for key, value in request.items() if key != "request_sha256"}
            self.assertEqual(request["request_sha256"], hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest())

    def test_kernel_suite_builder_enumerates_manifest_symbols(self) -> None:
        from bioprocess_runtime.nsight_attestation import build_nsight_kernel_suite

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            cupti = root / "cupti.json"
            manifest.write_text(json.dumps({"profile": {"kernel_launch_counts": {"kernel_a": 2, "kernel_b": 3}}}), encoding="utf-8")
            cupti.write_text(
                json.dumps(
                    {
                        "modules": [
                            {"artifact": "a.cubin", "cubin_sha256": "hash_a"},
                            {"artifact": "b.cubin", "cubin_sha256": "hash_b"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            for label in ("k00", "k01"):
                (root / f"{label}.ncu-rep").write_bytes(b"report")
                (root / f"{label}_binding.json").write_text("{}", encoding="utf-8")
            completed = [
                SimpleNamespace(stdout="STT_FUNC STB_GLOBAL STO_ENTRY kernel_a"),
                SimpleNamespace(stdout="STT_FUNC STB_GLOBAL STO_ENTRY kernel_b"),
            ]

            def certificate(report: Path, *args: object) -> dict[str, object]:
                index = 0 if report.name.startswith("k00") else 1
                return {
                    "all_checks_pass": True,
                    "checks": {"bound": True},
                    "cupti_module": {"module_id": index, "cubin_sha256": f"hash_{'a' if index == 0 else 'b'}"},
                    "launch_sass": {"instruction_count": 1, "canonical_sha256": f"sass_{index}", "opcode_histogram": {"MOV": 1}},
                    "report_sha256": f"report_{index}",
                    "certificate_sha256": f"certificate_{index}",
                    "session_exact_kernel_filter_value_visible": True,
                    "capture_request_sha256": None,
                }

            with (
                patch("bioprocess_runtime.nsight_attestation.shutil.which", return_value="cuobjdump"),
                patch("bioprocess_runtime.nsight_attestation.subprocess.run", side_effect=completed),
                patch("bioprocess_runtime.nsight_attestation.build_nsight_launch_certificate", side_effect=certificate),
            ):
                suite = build_nsight_kernel_suite(root, manifest, cupti, root)
            self.assertTrue(suite["complete"])
            self.assertEqual(suite["attested_distinct_kernels"], 2)
            self.assertEqual(suite["attested_kernel_launches"], 5)
            self.assertEqual(suite["syntactic_proposed_semantics_opcode_fraction"], 1.0)

    def test_launch_certificate_verifier_detects_claim_changes(self) -> None:
        from bioprocess_runtime.nsight_attestation import verify_nsight_launch_certificate

        certificate = {
            "checks": {"launch_sass_matches_loaded_cubin_function": True, "other": True},
            "all_checks_pass": True,
            "launch_sass": {"canonical_sha256": "same"},
            "loaded_cubin_function_sass": {"canonical_sha256": "same"},
        }
        certificate["certificate_sha256"] = hashlib.sha256(canonical_json(certificate).encode("utf-8")).hexdigest()
        self.assertTrue(verify_nsight_launch_certificate(certificate)["valid"])
        damaged = copy.deepcopy(certificate)
        damaged["launch_sass"]["canonical_sha256"] = "different"
        self.assertFalse(verify_nsight_launch_certificate(damaged)["valid"])


class OperatorCorrespondenceTests(unittest.TestCase):
    def test_stage_pattern_presence_does_not_become_semantic_equivalence(self) -> None:
        from bioprocess_runtime.operator_correspondence import (
            STAGE_PATTERNS,
            build_operator_correspondence,
            verify_operator_correspondence,
        )

        entries = []
        for index, pattern in enumerate(dict.fromkeys(pattern for patterns in STAGE_PATTERNS.values() for pattern in patterns)):
            entries.append(
                {
                    "index": index,
                    "kernel_name": f"prefix_{pattern}_suffix",
                    "module_id": index,
                    "sass_canonical_sha256": f"sass_{index}",
                }
            )
        correspondence = build_operator_correspondence({"entries": entries, "suite_sha256": "suite", "complete": True})
        self.assertTrue(correspondence["all_stage_patterns_observed_and_attested"])
        self.assertFalse(correspondence["full_operator_semantic_equivalence_established"])
        self.assertTrue(all(not stage["semantic_equivalence_established"] for stage in correspondence["stages"]))
        self.assertTrue(verify_operator_correspondence(correspondence)["valid"])
        damaged = copy.deepcopy(correspondence)
        damaged["full_operator_semantic_equivalence_established"] = True
        self.assertFalse(verify_operator_correspondence(damaged)["valid"])


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

    def test_nsight_permission_probe_records_counter_permission_failure(self) -> None:
        from bioprocess_runtime.cuda_provenance import probe_nsight_compute_permission

        completed = SimpleNamespace(returncode=1, stdout="ERR_NVGPUCTRPERM", stderr="")
        with (
            patch("bioprocess_runtime.cuda_provenance._require_cuda"),
            patch("bioprocess_runtime.cuda_provenance.shutil.which", return_value="ncu"),
            patch("bioprocess_runtime.cuda_provenance.subprocess.run", return_value=completed),
            patch("bioprocess_runtime.cuda_provenance._tool_version", return_value="ncu test"),
        ):
            record = probe_nsight_compute_permission()
        self.assertFalse(record["permission_granted"])
        self.assertEqual(record["error_codes"], ["ERR_NVGPUCTRPERM"])
        self.assertIn("blocked", record["impact"])

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
        self.assertEqual(summary["syntactic_proposed_semantics_opcode_coverage"]["coverage_fraction"], 1.0)
