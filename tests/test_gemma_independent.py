from __future__ import annotations

import copy
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np

from bioprocess_runtime import gemma_independent as independent
from bioprocess_runtime import gemma_first_layer as first
from bioprocess_runtime import gemma_two_layers as two
from bioprocess_runtime.gemma_rotary_slice import _seal
from test_gemma_first_layer import real_program, profiles, cheap_rows, cheap_rms, cheap_softmax
from test_gemma_two_layers import synthetic_context, synthetic_patches


def extend_context(target):
    program, ids, snapshots, providers = synthetic_context()
    with patch.object(first, "PROGRAM_SHA256", program["program_sha256"]):
        nodes = independent.dependency_cone(program, target)
    required = {name for node in nodes for name in node["parameter_refs"]}
    snapshots = {name: snapshot for name, snapshot in snapshots.items() if name in required}
    for name in required - set(snapshots):
        commitment = copy.deepcopy(program["parameter_commitments"][name])
        value = np.zeros(commitment["shape"], dtype=np.uint32 if commitment["dtype"] == "torch.float32" else np.uint16)
        descriptor = first._descriptor(value)
        commitment["sha256"] = descriptor["sha256"]
        program["parameter_commitments"][name] = commitment
        snapshots[name] = {"format": "full_parameter_bits_v1", "bits": value, "descriptor": descriptor, "commitment": commitment}
    program = _seal({key: value for key, value in program.items() if key != "program_sha256"}, "program_sha256")
    return program, ids, snapshots, providers


def global_packet(program, snapshots):
    cosine = np.full((1, 30, 256), 0x3F80, dtype=np.uint16)
    sine = np.zeros_like(cosine)
    return _seal({"kind": "fixed_position_empirical_global_rotary_v1", "program_sha256": program["program_sha256"], "positions": list(range(30)),
                  "runtime": {"synthetic": True}, "frequency": snapshots[independent.GLOBAL_FREQUENCY]["descriptor"],
                  "cosine_bits": cosine.tolist(), "sine_bits": sine.tolist(), "descriptors": {"cosine": first._descriptor(cosine), "sine": first._descriptor(sine)},
                  "empirical_primitive_data": True, "hidden_states_used": False, "native_repeat_count": 3, "native_repeat_exact": True}, "provider_sha256")


def synthetic(program):
    stack = synthetic_patches(program)
    stack.enter_context(patch.object(independent, "code_sha256", return_value="synthetic-independent"))
    stack.enter_context(patch.object(independent.ColumnProjector, "project", side_effect=lambda left, weights, mode, profile: cheap_rows(left, weights, mode, profile, 1)))
    return stack


class IndependentTests(unittest.TestCase):
    def test_full_target_and_pruned_prefix_cones(self):
        program = real_program()
        full = independent.dependency_cone(program, "selected_token_id")
        self.assertEqual(len(full), 533)
        self.assertEqual([node["opcode"] for node in full[-4:]], ["RMS_NORM", "SLICE_LAST_TOKEN", "LINEAR", "ARGMAX"])
        prefix = independent.dependency_cone(program, "hidden.2")
        self.assertEqual(len(prefix), 63)
        self.assertNotIn("i0003", [node["id"] for node in prefix])
        self.assertNotIn("i0005", [node["id"] for node in prefix])
        self.assertEqual(len(independent.dependency_cone(program, "hidden.18")), 529)

    def test_every_layer_role_is_selected_by_actual_weight_reference(self):
        for node in real_program()["instructions"]:
            if node["opcode"] == "LINEAR":
                mode = independent.linear_mode(node)
                self.assertIn(mode, {*profiles(), "vocabulary_candidate"})
                if node["layer"] is not None:
                    changed = copy.deepcopy(node)
                    changed["layer"] = (node["layer"] + 1) % 18
                    with self.assertRaises(ValueError):
                        independent.linear_mode(changed)
        self.assertEqual(independent.linear_mode(real_program()["instructions"][-2]), "vocabulary_candidate")

    def test_generic_two_layer_execution_matches_unchanged_engine(self):
        program, ids, snapshots, providers = synthetic_context()
        with synthetic(program):
            old = two.execute_two_layers(program, ids, snapshots, providers, profiles(), {"synthetic": True}, 1)
            new = independent.execute(program, ids, snapshots, providers, profiles(), {"synthetic": True}, target="hidden.2", workers=1, retain_states=True)
        self.assertEqual(new["status"], "complete_uncompared")
        self.assertEqual(new["state_bits"], old["state_bits"])
        self.assertEqual(new["scalar_stages"], old["scalar_stages"])
        self.assertEqual(new["softmax_f32_bits"]["layer.0.attention.probability"], old["stage_executions"]["first"]["softmax_f32_bits"])
        self.assertEqual(new["softmax_f32_bits"]["layer.1.attention.probability"], old["stage_executions"]["second"]["softmax_f32_bits"])
        self.assertEqual(new["completed_instruction_count"], 63)
        for flag in independent.FALSE_FLAGS:
            self.assertFalse(new[flag], flag)

    def test_all_decoder_layers_share_engine_with_local_and_global_roots(self):
        program, ids, snapshots, providers = extend_context("hidden.18")
        packet = global_packet(program, snapshots)
        with synthetic(program):
            result = independent.execute(program, ids, snapshots, providers, profiles(), {"synthetic": True}, target="hidden.18", global_rotary=packet, workers=1)
        self.assertEqual(result["status"], "complete_uncompared")
        self.assertEqual(result["completed_instruction_count"], 529)
        self.assertEqual(len(result["softmax_f32_bits"]), 18)
        self.assertEqual(len(result["scalar_stages"]), 108)
        self.assertTrue(all("hidden." + str(index) in result["state_bits"] for index in range(19)))
        self.assertNotIn("layer.2.mlp.up", result["state_bits"])
        self.assertIn("layer.2.mlp.up", result["state_descriptors"])
        for index in (5, 11, 17):
            rotary = next(record["payload"] for record in result["records"] if record["payload"]["layer"] == index and record["payload"]["opcode"] == "ROTARY_APPLY_PAIR")
            self.assertIn("rotary.global.cosine", rotary["inputs"])
            self.assertNotIn("rotary.local.cosine", rotary["inputs"])

    def test_missing_global_provider_abstains_with_partial_trace(self):
        program, ids, snapshots, providers = extend_context("hidden.6")
        with synthetic(program):
            result = independent.execute(program, ids, snapshots, providers, profiles(), {"synthetic": True}, target="hidden.6", workers=1)
        self.assertEqual(result["status"], "abstained")
        self.assertEqual(result["abstention"]["instruction_id"], "i0003")
        self.assertEqual(result["completed_instruction_count"], 3)
        self.assertIsNone(result["selected_token_id"])
        self.assertFalse(result["framework_arithmetic_fallback_used"])

    def test_changed_global_frequency_or_table_rejects(self):
        program, ids, snapshots, providers = extend_context("hidden.6")
        packet = global_packet(program, snapshots)
        for kind in ("frequency", "table", "runtime", "positions"):
            changed = copy.deepcopy(packet)
            if kind == "frequency":
                changed["frequency"]["sha256"] = "wrong"
            elif kind == "table":
                changed["cosine_bits"][0][0][0] ^= 1
            elif kind == "runtime":
                changed["runtime"] = {}
            else:
                changed["positions"].reverse()
            changed = _seal({key: value for key, value in changed.items() if key != "provider_sha256"}, "provider_sha256")
            with self.assertRaises(ValueError):
                independent.check_global_rotary(changed, program, snapshots[independent.GLOBAL_FREQUENCY]["bits"], {"synthetic": True})

    def test_vocabulary_projection_requires_explicit_candidate_and_keeps_every_column(self):
        from bioprocess_runtime import gemma_gemv
        program = real_program()
        node = program["instructions"][-2]
        left = np.full((1, 1, 640), 0x3F80, dtype=np.uint16)
        weights = np.broadcast_to(left.reshape(1, 640), (262144, 640))
        output = np.full((1, 262144), 0x4420, dtype=np.uint16)
        with independent.ColumnProjector(1) as projector, patch.object(projector, "project", side_effect=AssertionError("no legacy fallback")), patch.object(gemma_gemv, "project_bits", return_value=output) as project:
            with self.assertRaises(independent.UnsupportedArithmetic):
                independent._operation(node, [left], {"lm_head.weight": weights}, {}, None, profiles(), {}, None, program, projector, False)
            project.assert_not_called()
            values, auxiliary, evidence = independent._operation(node, [left], {"lm_head.weight": weights}, {}, None, profiles(), {}, None, program, projector, True)
            self.assertEqual(project.call_count, 1)
            np.testing.assert_array_equal(project.call_args.args[0], left.reshape(1, 640))
        self.assertEqual(values[0].shape, (1, 1, 262144))
        self.assertTrue(np.all(values[0] == 0x4420))
        self.assertEqual(evidence["mode"], "gemv")
        self.assertEqual(evidence["profile"], gemma_gemv.PROFILE)
        self.assertEqual(evidence["gemv_evidence"], independent.vocabulary_evidence())
        self.assertFalse(evidence["transfer_prequalified"])
        self.assertEqual(auxiliary, {})

    def test_vocabulary_shape_and_domain_errors_never_fall_back(self):
        from bioprocess_runtime import gemma_gemv
        program = real_program()
        node = program["instructions"][-2]
        left = np.zeros((1, 1, 640), dtype=np.uint16)
        weights = np.broadcast_to(left.reshape(1, 640), (262144, 640))
        with independent.ColumnProjector(1) as projector, patch.object(projector, "project", side_effect=AssertionError("no fallback")), patch.object(gemma_gemv, "project_bits", side_effect=gemma_gemv.UnsupportedGemvArithmetic("out of domain")) as project:
            with self.assertRaises(independent.UnsupportedArithmetic):
                independent._operation(node, [left], {"lm_head.weight": weights[:3]}, {}, None, profiles(), {}, None, program, projector, True)
            project.assert_not_called()
            with self.assertRaisesRegex(independent.UnsupportedArithmetic, "out of domain"):
                independent._operation(node, [left], {"lm_head.weight": weights}, {}, None, profiles(), {}, None, program, projector, True)

    def test_vocabulary_ledger_profile_source_and_scope_are_checked(self):
        program = real_program()
        node = program["instructions"][-2]
        provider = {"opcode": "LINEAR", "transfer_prequalified": False, **independent.linear_evidence(node, profiles(), True)}
        independent.check_linear_evidence(program, node, provider, profiles(), True)
        for key in ("mode", "profile", "vocabulary_shape_status", "gemv_evidence"):
            changed = copy.deepcopy(provider)
            changed[key] = "wrong"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "evidence"):
                independent.check_linear_evidence(program, node, changed, profiles(), True)
        with patch.object(independent.gemv, "code_sha256", return_value="changed"), self.assertRaisesRegex(ValueError, "evidence"):
            independent.check_linear_evidence(program, node, provider, profiles(), True)

    def test_legacy_execution_schema_is_not_accepted_as_v3(self):
        program, ids, snapshots, providers = synthetic_context()
        with synthetic(program):
            result = independent.execute(program, ids, snapshots, providers, profiles(), {"synthetic": True}, target="hidden.2", workers=1)
            self.assertEqual(result["schema_version"], 3)
            result["schema_version"] = 2
            with self.assertRaisesRegex(ValueError, "schema"):
                independent.check_execution(program, result)

    def test_masks_and_argmax_exact_ties_signed_zero_and_nonfinite(self):
        self.assertTrue(np.array_equal(independent.causal_mask(30, None), first.causal_mask_bits()))
        self.assertTrue(np.array_equal(independent.causal_mask(30, 512), first.causal_mask_bits()))
        for values, expected in (([0x8000, 0, 0xBF80], 0), ([0xBF80, 0xBF00, 0xBF00], 1), ([0x3F80, 0x4000, 0x4000], 1), ([0x8001, 0], 1)):
            self.assertEqual(independent.argmax_bfloat16(np.asarray([[values]], dtype=np.uint16)).tolist(), [expected])
        for value in (0x7F80, 0xFF80, 0x7FC1):
            with self.assertRaises(independent.UnsupportedArithmetic):
                independent.argmax_bfloat16(np.asarray([[[0, value]]], dtype=np.uint16))

    def test_column_tiling_preserves_serial_dot_and_covers_tail(self):
        left = np.full((2, 16), 0x3F80, dtype=np.uint16)
        weights = np.full((513, 16), 0x3F80, dtype=np.uint16)
        with independent.ColumnProjector(1) as projector:
            result = projector.project(left, weights, "serial", profiles()["serial"])
        self.assertEqual(result.shape, (2, 513))
        self.assertTrue(np.all(result == 0x4180))

    def test_domain_failure_retained_and_infrastructure_failure_raises(self):
        program, ids, snapshots, providers = synthetic_context()
        with synthetic(program), patch.object(first, "_rms_lookup_row", side_effect=ArithmeticError("declared numerical domain")):
            result = independent.execute(program, ids, snapshots, providers, profiles(), {"synthetic": True}, target="hidden.2", workers=1)
        self.assertEqual(result["status"], "abstained")
        self.assertEqual(result["abstention"]["instruction_id"], "i0007")
        self.assertIsNone(result["selected_token_id"])
        with synthetic(program), patch.object(first, "_rms_lookup_row", side_effect=ValueError("infrastructure failure")), self.assertRaisesRegex(ValueError, "infrastructure"):
            independent.execute(program, ids, snapshots, providers, profiles(), {"synthetic": True}, target="hidden.2", workers=1)

    def test_source_and_parameter_mutation_fail_closed(self):
        program, ids, snapshots, providers = synthetic_context()
        def mutate(event):
            if event["index"] == 1:
                snapshots["model.layers.0.input_layernorm.weight"]["bits"][0] ^= 1
        with synthetic(program), self.assertRaises(ValueError):
            independent.execute(program, ids, snapshots, providers, profiles(), {"synthetic": True}, target="hidden.2", workers=1, progress=mutate)
        with patch.object(independent.Path, "read_bytes", return_value=b"changed"), self.assertRaises(ValueError):
            independent.code_sha256()
