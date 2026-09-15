from __future__ import annotations

import copy
import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path


DEPENDENCIES_AVAILABLE = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None


@unittest.skipUnless(DEPENDENCIES_AVAILABLE, "Gemma optional dependencies are not installed")
class ReferenceGemmaTests(unittest.TestCase):
    def setUp(self) -> None:
        import torch
        from transformers import Gemma3ForCausalLM, Gemma3TextConfig

        config = Gemma3TextConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=64,
            sliding_window=8,
            layer_types=["sliding_attention", "full_attention"],
            query_pre_attn_scalar=8,
            rope_theta=10000.0,
            rope_local_base_freq=1000.0,
        )
        self.torch = torch
        self.model = Gemma3ForCausalLM(config).eval()
        self.model.config._attn_implementation = "eager"
        self.input_ids = torch.tensor([[2, 4, 5, 6]], dtype=torch.long)

    def test_projection_prefix_stops_before_linear_and_binds_weights(self) -> None:
        from unittest.mock import patch
        from bioprocess_runtime.gemma_ir import compile_gemma_ir
        from bioprocess_runtime.gemma_ir_interpreter import bind_model_tensors, _projection_prefix
        from bioprocess_runtime.operational_semantics import build_architecture_manifest, verify_trace_chain

        tokenizer = type("Tokenizer", (), {"vocab_size": 32, "bos_token_id": 2, "eos_token_id": 1, "pad_token_id": 0})()
        with tempfile.TemporaryDirectory() as directory:
            program = compile_gemma_ir(build_architecture_manifest(self.model, tokenizer, Path(directory)))
        parameters = bind_model_tensors(program, self.model)
        with self.torch.no_grad(), patch("torch.nn.functional.linear", side_effect=AssertionError("Projection executed before prediction")):
            normalized, records = _projection_prefix(program, parameters, self.input_ids)
            expected = self.model.model.layers[0].input_layernorm(self.model.model.embed_tokens(self.input_ids))
        self.assertTrue(self.torch.equal(normalized, expected))
        self.assertEqual([record["payload"]["opcode"] for record in records], ["EMBEDDING", "SCALE", "RMS_NORM"])
        self.assertTrue(verify_trace_chain(records)["valid"])
        with self.assertRaises(ValueError):
            _projection_prefix(program, parameters, self.torch.tensor([[32]], dtype=self.torch.int64))
        with self.torch.no_grad():
            self.model.model.layers[0].self_attn.q_proj.weight[0, 0] += 1
        with self.assertRaises(ValueError):
            bind_model_tensors(program, self.model)

    def test_projection_slice_reports_exact_coordinates_and_prefix_failures(self) -> None:
        from bioprocess_runtime.gemma_ir_interpreter import _projection_slice_report, _projection_bits_tensor, PROJECTION_SLICE_ROLES

        predictions = {role: [[[0] * width for _ in range(30)]] for role, _, width in PROJECTION_SLICE_ROLES}
        bundle = {"prediction_bits": predictions}
        plan = {"plan_sha256": "synthetic", "bundle_sha256": "synthetic", "normalized_input": {"sha256": "synthetic"}, "runtime": {},
                "projection_records": {role: {"expected_kernel_names": ["synthetic"], "expected_environment": {}} for role, _, _ in PROJECTION_SLICE_ROLES}}
        observed = {role: {"bits": [copy.deepcopy(predictions[role]) for _ in range(3)], "kernel_names": [["synthetic"]] * 3} for role, _, _ in PROJECTION_SLICE_ROLES}
        observed["value"]["bits"][1][0][29][255] = 1
        report = _projection_slice_report(plan, bundle, observed, plan["normalized_input"], {})
        self.assertEqual(report["mismatch_count"], 1)
        self.assertEqual(report["first_divergence"], {"role": "value", "repetition": 1, "row": 29, "column": 255, "predicted_bits": 0, "observed_bits": 1})
        self.assertFalse(report["slice_passes_declared_comparison"])
        self.assertFalse(report["full_first_layer_qualified"])
        report = _projection_slice_report(plan, bundle, observed, {"sha256": "different"}, {})
        self.assertEqual(report["first_divergence"], "shared_prefix")
        for bits in ([[[True, 0]]], [[[1.5, 0]]], [[[0]]]):
            with self.assertRaises(ValueError):
                _projection_bits_tensor(bits, [1, 1, 2], "cpu")

    def test_actual_projection_summary_preserves_scope_and_output_hashes(self) -> None:
        import json
        from bioprocess_runtime.serialization import canonical_json

        root = Path(__file__).resolve().parent.parent
        plan = json.loads((root / "results/gemma3_270m_projection_slice_plan.json").read_text(encoding="utf-8"))
        summary = json.loads((root / "results/gemma3_270m_projection_slice_summary.json").read_text(encoding="utf-8"))
        body = {key: value for key, value in summary.items() if key != "summary_sha256"}
        self.assertEqual(summary["summary_sha256"], hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest())
        self.assertEqual(summary["plan_sha256"], plan["plan_sha256"])
        self.assertEqual(summary["compared_values_per_repetition"], 46080)
        self.assertEqual(summary["mismatch_count"], 0)
        self.assertTrue(summary["original_prefix_matches_shared_ir"])
        for role in ("query", "key", "value"):
            self.assertEqual(summary["observed_output_hashes"][role], [plan["projection_records"][role]["prediction"]["sha256"]] * 3)
        self.assertFalse(summary["full_first_layer_qualified"])
        self.assertFalse(summary["full_model_independently_qualified"])
        self.assertFalse(summary["global_exactness_activation_allowed"])

    def test_actual_projection_artifact_integrity_rejects_tampering(self) -> None:
        import json
        from bioprocess_runtime.gemma_ir_interpreter import verify_projection_slice, projection_slice_summary
        from bioprocess_runtime.serialization import canonical_json

        root = Path(__file__).resolve().parent.parent
        bundle_path = root / "artifacts/gemma3_270m_projection_slice_predictions.json"
        report_path = root / "artifacts/gemma3_270m_projection_slice_report.json"
        if not bundle_path.exists() or not report_path.exists():
            self.skipTest("Tensor-rich projection artifacts are intentionally not committed")
        program = json.loads((root / "results/gemma3_270m_execution_ir.json").read_text(encoding="utf-8"))
        plan = json.loads((root / "results/gemma3_270m_projection_slice_plan.json").read_text(encoding="utf-8"))
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        verified = verify_projection_slice(program, plan, bundle, report)
        self.assertTrue(verified["valid"], verified)
        self.assertEqual(projection_slice_summary(plan, report), json.loads((root / "results/gemma3_270m_projection_slice_summary.json").read_text(encoding="utf-8")))
        damaged = copy.deepcopy(report)
        damaged["observations"]["key"]["bits"][0][0][29][255] ^= 1
        damaged["report_sha256"] = hashlib.sha256(canonical_json({k: v for k, v in damaged.items() if k != "report_sha256"}).encode("utf-8")).hexdigest()
        self.assertFalse(verify_projection_slice(program, plan, bundle, damaged)["valid"])
        damaged_plan = copy.deepcopy(plan)
        damaged_plan["projection_records"]["query"]["numerical_provider"] = "unbound_override"
        damaged_plan["plan_sha256"] = hashlib.sha256(canonical_json({k: v for k, v in damaged_plan.items() if k != "plan_sha256"}).encode("utf-8")).hexdigest()
        self.assertFalse(verify_projection_slice(program, damaged_plan, bundle, report)["valid"])
        damaged_plan = copy.deepcopy(plan)
        damaged_plan["shared_prefix_independently_qualified"] = True
        damaged_plan["plan_sha256"] = hashlib.sha256(canonical_json({k: v for k, v in damaged_plan.items() if k != "plan_sha256"}).encode("utf-8")).hexdigest()
        self.assertFalse(verify_projection_slice(program, damaged_plan, bundle, report)["valid"])

        from argparse import Namespace
        from contextlib import redirect_stdout
        from io import StringIO
        from unittest.mock import patch
        from bioprocess_runtime.cli import command_projection_slice

        args = Namespace(operation="run", program=root / "results/gemma3_270m_execution_ir.json", fixture=root / "results/gemma3_270m_ir_execution_summary.json",
                         query_evidence=root / "results/gemma3_270m_wide_query_holdout.json", split_evidence=root / "results/gemma3_270m_dense_split_holdout.json",
                         plan=root / "results/gemma3_270m_projection_slice_plan.json", bundle=bundle_path, output=Path("synthetic-report"), summary=None, model_path=Path("synthetic-model"))
        stdout = StringIO()
        with patch("torch.cuda.is_available", return_value=True), patch("transformers.AutoModelForCausalLM.from_pretrained"), patch("bioprocess_runtime.gemma_ir_interpreter.acquire_projection_slice", return_value=report), patch.object(Path, "mkdir"), patch.object(Path, "write_text") as writer, redirect_stdout(stdout):
            self.assertEqual(command_projection_slice(args), 0)
        writer.assert_called_once()
        self.assertEqual(json.loads(writer.call_args.args[0]), report)
        printed = json.loads(stdout.getvalue())
        self.assertNotIn("observations", printed)
        self.assertEqual(printed["source_report_sha256"], report["report_sha256"])

    def test_projection_cli_protects_sources_and_prediction_bundle(self) -> None:
        from argparse import Namespace
        from bioprocess_runtime.cli import command_projection_slice

        args = Namespace(operation="plan", program=Path("program"), fixture=Path("fixture"), query_evidence=Path("query"),
                         split_evidence=Path("split"), output=Path("program"), bundle=Path("bundle"), summary=None)
        with self.assertRaises(ValueError):
            command_projection_slice(args)
        args.output = args.bundle
        with self.assertRaises(ValueError):
            command_projection_slice(args)
        args.operation, args.plan = "run", Path("plan")
        with self.assertRaises(ValueError):
            command_projection_slice(args)
        args.operation, args.report, args.summary = "verify", Path("report"), args.program
        with self.assertRaises(ValueError):
            command_projection_slice(args)

    def test_rms_float32_roots_and_source_reduction_schedule(self) -> None:
        from fractions import Fraction
        from bioprocess_runtime.reference_gemma import rms_root_bits, rms_sum_bits, _rms_f32_value, _rms_f32_round

        self.assertEqual(rms_root_bits(0x40800000, "rsqrt_rne"), 0x3F000000)
        self.assertEqual(rms_root_bits(0x40800000, "sqrt_rne_then_reciprocal_rne"), 0x3F000000)
        for bits in (1, 0x3DCCCCCD, 0x40400000, 0x7F7FFFFF):
            result = rms_root_bits(bits, "rsqrt_rne")
            value = _rms_f32_value(bits)
            lower_midpoint = (_rms_f32_value(result - 1) + _rms_f32_value(result)) / 2
            upper_midpoint = (_rms_f32_value(result) + _rms_f32_value(result + 1)) / 2
            self.assertLessEqual(lower_midpoint * lower_midpoint * value, 1)
            self.assertGreaterEqual(upper_midpoint * upper_midpoint * value, 1)
        values = [0x3F800000] + [_rms_f32_round(Fraction(1, 1 << 24))] * 255
        self.assertEqual(rms_sum_bits(values, "sequential_float32"), 0x3F800000)
        self.assertEqual(rms_sum_bits(values, "source_vec4_warp32"), 0x3F80007F)
        self.assertEqual(rms_sum_bits(values, "exact_sum_float32"), 0x3F800080)
        for bits in (0, 0x80000000, 0xBF800000, 0x7F800000):
            with self.assertRaises(ValueError):
                rms_root_bits(bits, "rsqrt_rne")

    def test_rms_row_rounding_and_signed_zero(self) -> None:
        from bioprocess_runtime.reference_gemma import rms_row_candidate

        result = rms_row_candidate([0x3F80] * 256, [0] * 256, 1e-6, "source_vec4_warp32", "rsqrt_rne")
        self.assertEqual(result["mean_bits"], 0x3F800000)
        self.assertEqual(result["output_bits"], [0x3F80] * 256)
        zeros = rms_row_candidate([0x8000] + [0] * 255, [0] * 256, 1e-6, "source_vec4_warp32", "rsqrt_rne")
        self.assertEqual(zeros["output_bits"], [0x8000] + [0] * 255)
        with self.assertRaises(ValueError):
            rms_row_candidate([0] * 16, [0] * 16, 1e-6, "source_vec4_warp32", "rsqrt_rne")

    def test_rms_stage_capture_observes_original_module_without_changing_output(self) -> None:
        from bioprocess_runtime.reference_gemma import _observe_rms_module, _rms_tensor_f32_bits

        with self.torch.no_grad():
            values = self.model.model.embed_tokens(self.input_ids)
            module = self.model.model.layers[0].input_layernorm
            expected = module(values)
            output, stages = _observe_rms_module(module, values)
            mean = values.float().pow(2).mean(-1, keepdim=True)
        self.assertTrue(self.torch.equal(output, expected))
        self.assertEqual(stages["mean_bits"], _rms_tensor_f32_bits(mean))
        self.assertEqual(stages["mean_input_metadata"]["axes"], [-1])
        self.assertEqual(len(stages["rsqrt_bits"]), 4)

    def test_rms_summary_separates_output_agreement_from_stage_conformance(self) -> None:
        import json
        from bioprocess_runtime.serialization import canonical_json

        root = Path(__file__).resolve().parent.parent
        summary = json.loads((root / "results/gemma3_270m_rms_slice_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["summary_sha256"], hashlib.sha256(canonical_json({k: v for k, v in summary.items() if k != "summary_sha256"}).encode("utf-8")).hexdigest())
        self.assertFalse(summary["all_stages_candidate_passes"])
        self.assertEqual(summary["fully_matching_profiles"], [])
        self.assertTrue(summary["actual_inputs_match_plan"])
        self.assertTrue(summary["source_template_geometry_matches"])
        for role, root_failures in (("input_norm", 8), ("query_norm", 30), ("key_norm", 5)):
            candidate = summary["comparisons"][role]["source_vec4_warp32:rsqrt_rne"]
            self.assertEqual(candidate["output_mismatch_count"], 0)
            self.assertEqual(candidate["stage_mismatch_counts"]["mean_bits"], [0] * 3)
            self.assertEqual(candidate["stage_mismatch_counts"]["denominator_bits"], [0] * 3)
            self.assertEqual(candidate["stage_mismatch_counts"]["rsqrt_bits"], [root_failures] * 3)
        self.assertFalse(summary["global_exactness_activation_allowed"])

    def test_rms_artifact_integrity_and_tamper_rejection(self) -> None:
        import json
        from bioprocess_runtime.reference_gemma import verify_rms_slice, rms_slice_summary, check_rms_projection_source
        from bioprocess_runtime.serialization import canonical_json

        root = Path(__file__).resolve().parent.parent
        bundle_path, report_path = root / "artifacts/gemma3_270m_rms_slice_predictions.json", root / "artifacts/gemma3_270m_rms_slice_report.json"
        if not bundle_path.exists() or not report_path.exists():
            self.skipTest("Tensor-rich RMS artifacts are intentionally not committed")
        program = json.loads((root / "results/gemma3_270m_execution_ir.json").read_text(encoding="utf-8"))
        plan = json.loads((root / "results/gemma3_270m_rms_slice_plan.json").read_text(encoding="utf-8"))
        bundle, report = json.loads(bundle_path.read_text(encoding="utf-8")), json.loads(report_path.read_text(encoding="utf-8"))
        result = verify_rms_slice(program, plan, bundle, report)
        self.assertTrue(result["valid"], result)
        self.assertFalse(result["all_stages_candidate_passes"])
        projection_sources = [json.loads((root / path).read_text(encoding="utf-8")) for path in ("results/gemma3_270m_projection_slice_plan.json", "artifacts/gemma3_270m_projection_slice_predictions.json", "results/gemma3_270m_projection_slice_summary.json")]
        check_rms_projection_source(program, *projection_sources, plan)
        wrong_input = copy.deepcopy(plan)
        wrong_input["roles"]["query_norm"]["input"]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            check_rms_projection_source(program, *projection_sources, wrong_input)
        self.assertEqual(rms_slice_summary(plan, report), json.loads((root / "results/gemma3_270m_rms_slice_summary.json").read_text(encoding="utf-8")))
        for role in report["observations"]:
            repetitions = report["observations"][role]["repetitions"]
            self.assertEqual(repetitions[0], repetitions[1])
            self.assertEqual(repetitions[1], repetitions[2])
        damaged = copy.deepcopy(report)
        damaged["all_stages_candidate_passes"] = True
        damaged["report_sha256"] = hashlib.sha256(canonical_json({k: v for k, v in damaged.items() if k != "report_sha256"}).encode("utf-8")).hexdigest()
        self.assertFalse(verify_rms_slice(program, plan, bundle, damaged)["valid"])
        damaged = copy.deepcopy(report)
        predicted = bundle["roles"]["input_norm"]["predictions"]["source_vec4_warp32:rsqrt_rne"]["rsqrt_bits"][0]
        observed = damaged["observations"]["input_norm"]["repetitions"][0]["rsqrt_bits"][0]
        damaged["observations"]["input_norm"]["repetitions"][0]["rsqrt_bits"][0] = predicted if observed != predicted else predicted ^ 1
        damaged["report_sha256"] = hashlib.sha256(canonical_json({k: v for k, v in damaged.items() if k != "report_sha256"}).encode("utf-8")).hexdigest()
        self.assertFalse(verify_rms_slice(program, plan, bundle, damaged)["valid"])

        from argparse import Namespace
        from contextlib import redirect_stdout
        from io import StringIO
        from unittest.mock import patch
        from bioprocess_runtime.cli import command_rms_slice

        args = Namespace(operation="run", program=root / "results/gemma3_270m_execution_ir.json", projection_plan=root / "results/gemma3_270m_projection_slice_plan.json",
                         projection_bundle=root / "artifacts/gemma3_270m_projection_slice_predictions.json", projection_summary=root / "results/gemma3_270m_projection_slice_summary.json",
                         plan=root / "results/gemma3_270m_rms_slice_plan.json", bundle=bundle_path, report=report_path, output=Path("synthetic-rms-report"), summary=None, model_path=Path("synthetic-model"))
        stdout = StringIO()
        with patch("torch.cuda.is_available", return_value=True), patch("transformers.AutoModelForCausalLM.from_pretrained"), patch("bioprocess_runtime.reference_gemma.acquire_rms_slice", return_value=report), patch.object(Path, "mkdir"), patch.object(Path, "write_text") as writer, redirect_stdout(stdout):
            self.assertEqual(command_rms_slice(args), 1)
        writer.assert_called_once()
        self.assertEqual(json.loads(writer.call_args.args[0]), report)
        self.assertNotIn("observations", json.loads(stdout.getvalue()))
        args.operation, args.reexecute = "verify", False
        with redirect_stdout(StringIO()):
            self.assertEqual(command_rms_slice(args), 0)

    def test_independent_orchestration_exactly_matches_eager_boundaries(self) -> None:
        from bioprocess_runtime.reference_gemma import fixed_input_equivalence_certificate

        certificate = fixed_input_equivalence_certificate(self.model, self.input_ids, absolute_tolerance=0.0)
        self.assertTrue(certificate["all_boundaries_within_tolerance"])
        self.assertTrue(certificate["selected_token_exact_match"])
        self.assertTrue(certificate["logits"]["exact_equal"])
        self.assertTrue(all(boundary["exact_equal"] for boundary in certificate["boundaries"]))
        self.assertEqual(len(certificate["attention_kernel_comparisons"]), 2)

    def test_typed_ir_interpreter_predicts_eager_output_exactly(self) -> None:
        from bioprocess_runtime.gemma_ir import compile_gemma_ir, verify_gemma_ir
        from bioprocess_runtime.gemma_ir_interpreter import (
            bind_model_tensors,
            build_ir_execution_certificate,
            compare_ir_execution,
            execute_gemma_ir,
            recompute_ir_execution_certificate,
            summarize_ir_execution,
            verify_ir_execution_certificate,
            verify_ir_execution_summary,
        )
        from bioprocess_runtime.operational_semantics import (
            append_chain_record,
            build_architecture_manifest,
        )
        from bioprocess_runtime.reference_gemma import model_state_sha256
        from bioprocess_runtime.serialization import canonical_json

        tokenizer = type(
            "Tokenizer",
            (),
            {
                "vocab_size": 32,
                "bos_token_id": 2,
                "eos_token_id": 1,
                "pad_token_id": 0,
            },
        )()
        with tempfile.TemporaryDirectory() as directory:
            manifest = build_architecture_manifest(
                self.model, tokenizer, Path(directory)
            )
        program = compile_gemma_ir(manifest)
        self.assertTrue(verify_gemma_ir(program, manifest)["valid"])
        parameters = bind_model_tensors(program, self.model)
        ir_input_ids = (self.torch.arange(12, dtype=self.torch.long) % 30).unsqueeze(0)
        prediction = execute_gemma_ir(program, parameters, ir_input_ids)
        with self.torch.no_grad():
            observed = self.model(
                input_ids=ir_input_ids,
                attention_mask=self.torch.ones_like(ir_input_ids),
                use_cache=False,
                logits_to_keep=1,
            )
        self.assertTrue(self.torch.equal(prediction.logits, observed.logits))
        self.assertTrue(
            self.torch.equal(
                prediction.selected_token_id,
                self.torch.argmax(observed.logits[:, -1, :], dim=-1),
            )
        )
        certificate = build_ir_execution_certificate(
            program,
            ir_input_ids,
            prediction,
            observed.logits,
            "huggingface_eager",
            model_state_sha256(self.model),
        )
        certificate_verification = verify_ir_execution_certificate(
            program, certificate
        )
        self.assertTrue(certificate_verification["valid"], certificate_verification)
        replay = recompute_ir_execution_certificate(program, certificate, self.model)
        self.assertTrue(replay["valid"], replay)
        self.assertTrue(replay["reexecution_performed"])
        self.assertTrue(certificate["fixed_input_canonical_eager_prediction_established"])
        self.assertFalse(certificate["deployed_sdpa_correspondence_established"])
        self.assertEqual(len(certificate["selection_proofs"]), 1)
        self.assertEqual(
            certificate["selection_proofs"][0]["selected_token_id"],
            certificate["predicted_token_ids"][0],
        )
        self.assertTrue(certificate["selection_proofs"][0]["argmax_recomputed"])
        self.assertTrue(
            certificate["selection_proofs"][0]["all_competitors_not_greater"]
        )
        summary = summarize_ir_execution(program, certificate)
        self.assertTrue(
            verify_ir_execution_summary(program, summary, certificate)["valid"]
        )
        self.assertFalse(summary["full_execution_records_committed"])
        wrong_source = copy.deepcopy(certificate)
        wrong_source["certificate_sha256"] = "0" * 64
        wrong_source_verification = verify_ir_execution_summary(
            program, summary, wrong_source
        )
        self.assertFalse(wrong_source_verification["valid"])
        self.assertFalse(
            wrong_source_verification["source_certificate_binding_valid"]
        )
        damaged_summary = copy.deepcopy(summary)
        damaged_summary["predicted_token_ids"] = [
            (summary["observed_token_ids"][0] + 1) % 32
        ]
        self.assertFalse(
            verify_ir_execution_summary(program, damaged_summary, certificate)["valid"]
        )
        damaged_certificate = copy.deepcopy(certificate)
        damaged_certificate["observed_logits"]["sha256"] = "0" * 64
        self.assertFalse(
            verify_ir_execution_certificate(program, damaged_certificate)["valid"]
        )
        ragged_certificate = copy.deepcopy(certificate)
        ragged_certificate["input_token_ids"] = [[1, 2], [3]]
        ragged_body = {
            key: value
            for key, value in ragged_certificate.items()
            if key != "certificate_sha256"
        }
        ragged_certificate["certificate_sha256"] = hashlib.sha256(
            canonical_json(ragged_body).encode("utf-8")
        ).hexdigest()
        ragged_verification = verify_ir_execution_certificate(
            program, ragged_certificate
        )
        self.assertFalse(ragged_verification["valid"])
        self.assertFalse(ragged_verification["input_value_binding_valid"])
        forged_selection = copy.deepcopy(certificate)
        forged_selection["selection_proofs"][0]["runner_up_token_id"] = (
            forged_selection["predicted_token_ids"][0]
        )
        selection_body = {
            key: value
            for key, value in forged_selection.items()
            if key != "certificate_sha256"
        }
        forged_selection["certificate_sha256"] = hashlib.sha256(
            canonical_json(selection_body).encode("utf-8")
        ).hexdigest()
        selection_verification = verify_ir_execution_certificate(
            program, forged_selection
        )
        self.assertFalse(selection_verification["valid"])
        self.assertFalse(selection_verification["selection_proofs_valid"])
        forged_certificate = copy.deepcopy(certificate)
        logits_name = program["declared_outputs"][0]
        forged_payloads = [
            copy.deepcopy(record["payload"])
            for record in forged_certificate["execution_records"]
        ]
        logits_payload = next(
            payload for payload in forged_payloads if logits_name in payload["outputs"]
        )
        logits_payload["outputs"][logits_name]["sha256"] = "f" * 64
        forged_records = []
        for payload in forged_payloads:
            append_chain_record(forged_records, payload)
        forged_certificate["execution_records"] = forged_records
        forged_certificate["execution_root_sha256"] = forged_records[-1]["record_hash"]
        certificate_body = {
            key: value
            for key, value in forged_certificate.items()
            if key != "certificate_sha256"
        }
        forged_certificate["certificate_sha256"] = hashlib.sha256(
            canonical_json(certificate_body).encode("utf-8")
        ).hexdigest()
        forged_verification = verify_ir_execution_certificate(
            program, forged_certificate
        )
        self.assertFalse(forged_verification["valid"])
        self.assertFalse(forged_verification["record_output_binding_valid"])
        self.assertEqual(len(prediction.records), len(program["instructions"]))
        self.assertEqual(prediction.records[-1]["payload"]["opcode"], "ARGMAX")
        exact_comparison = compare_ir_execution(
            program, prediction, list(prediction.records)
        )
        self.assertTrue(exact_comparison["exact_match"])
        damaged_records = copy.deepcopy(list(prediction.records))
        first_output = next(iter(damaged_records[10]["payload"]["outputs"].values()))
        first_output["sha256"] = "0" * 64
        divergence = compare_ir_execution(program, prediction, damaged_records)
        self.assertFalse(divergence["exact_match"])
        self.assertEqual(divergence["first_divergence"]["index"], 10)
        batched_input = self.torch.cat((ir_input_ids, ir_input_ids.flip(1)), dim=0)
        batched_prediction = execute_gemma_ir(program, parameters, batched_input)
        with self.torch.no_grad():
            batched_observed = self.model(
                input_ids=batched_input,
                attention_mask=self.torch.ones_like(batched_input),
                use_cache=False,
                logits_to_keep=1,
            )
        self.assertTrue(
            self.torch.equal(batched_prediction.logits, batched_observed.logits)
        )
        invalid_input = ir_input_ids.clone()
        invalid_input[0, 0] = 32
        with self.assertRaisesRegex(ValueError, "declared vocabulary"):
            execute_gemma_ir(program, parameters, invalid_input)

    def test_reference_matches_eager_beyond_sliding_window(self) -> None:
        from bioprocess_runtime.reference_gemma import fixed_input_equivalence_certificate

        long_input = (self.torch.arange(12, dtype=self.torch.long) % 30).unsqueeze(0)
        certificate = fixed_input_equivalence_certificate(self.model, long_input, absolute_tolerance=0.0)
        self.assertTrue(certificate["all_boundaries_within_tolerance"])

    def test_sdpa_comparison_rejects_attention_softcapping(self) -> None:
        from bioprocess_runtime.reference_gemma import reference_gemma_forward

        for layer in self.model.model.layers:
            layer.self_attn.attn_logit_softcapping = 50.0
        with self.assertRaisesRegex(ValueError, "softcapped"):
            reference_gemma_forward(self.model, self.input_ids, compare_attention_kernels=True)
        output = reference_gemma_forward(self.model, self.input_ids, compare_attention_kernels=False)
        self.assertEqual(output.logits.shape[-1], 32)

    def test_reference_trace_names_intermediate_computations(self) -> None:
        from bioprocess_runtime.reference_gemma import reference_gemma_forward

        output = reference_gemma_forward(self.model, self.input_ids)
        stages = {record["payload"]["stage"] for record in output.records}
        self.assertTrue({"query", "key", "value", "attention_probability", "mlp_gate", "mlp_intermediate_product", "vocabulary_logits"}.issubset(stages))

    def test_certificate_verifier_detects_changes(self) -> None:
        from bioprocess_runtime.reference_gemma import (
            fixed_input_equivalence_certificate,
            recompute_fixed_input_certificate,
            verify_fixed_input_certificate,
        )

        certificate = fixed_input_equivalence_certificate(self.model, self.input_ids, absolute_tolerance=0.0)
        self.assertTrue(verify_fixed_input_certificate(certificate)["valid"])
        self.assertTrue(recompute_fixed_input_certificate(self.model, certificate)["valid"])
        damaged = copy.deepcopy(certificate)
        damaged["reference_selected_token_id"] += 1
        self.assertFalse(verify_fixed_input_certificate(damaged)["valid"])
        forged_claim = copy.deepcopy(certificate)
        forged_claim["all_boundaries_within_tolerance"] = False
        body = {key: value for key, value in forged_claim.items() if key != "certificate_sha256"}
        from bioprocess_runtime.serialization import canonical_json
        forged_claim["certificate_sha256"] = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
        verification = verify_fixed_input_certificate(forged_claim)
        self.assertFalse(verification["valid"])
        self.assertFalse(verification["claims_consistent"])

    def test_operator_equations_match_fixed_vector_implementations(self) -> None:
        from bioprocess_runtime.reference_gemma import operator_equation_conformance

        report = operator_equation_conformance()
        self.assertLess(report["maximum_absolute_error"], 1e-6)
        self.assertEqual({case["operator"] for case in report["cases"]}, {"linear", "rms_norm", "gelu_tanh", "stable_softmax"})
        if self.torch.cuda.is_available():
            self.assertTrue(all("cuda" in case for case in report["cases"]))

    def test_coordinate_registry_separates_structural_and_domain_meaning(self) -> None:
        from bioprocess_runtime.reference_gemma import semantic_coordinate_registry

        tokenizer = type("Tokenizer", (), {})()
        registry = semantic_coordinate_registry(self.model, tokenizer)
        self.assertEqual(registry["dimensions"]["hidden_coordinates"], 16)
        self.assertIn("tensor role", registry["semantic_boundary"]["exact_by_construction"])
        self.assertIn(
            "biological concept assigned to a hidden coordinate",
            registry["semantic_boundary"]["requires_empirical_or_formal_domain_evidence"],
        )

    def test_summary_reports_eager_equivalence_and_deployed_path(self) -> None:
        from bioprocess_runtime.reference_gemma import (
            fixed_input_equivalence_certificate,
            operator_equation_conformance,
            semantic_coordinate_registry,
            summarize_reference_evidence,
        )

        certificate = fixed_input_equivalence_certificate(self.model, self.input_ids, absolute_tolerance=0.0)
        registry = semantic_coordinate_registry(self.model, type("Tokenizer", (), {})())
        summary = summarize_reference_evidence(certificate, operator_equation_conformance(), registry)
        self.assertTrue(summary["reference_vs_huggingface_eager"]["all_boundaries_exact"])
        self.assertTrue(summary["reference_vs_deployed_path"]["selected_token_matches_reference"])

    def test_bounded_domain_certificate_exhausts_declared_grid(self) -> None:
        from bioprocess_runtime.reference_gemma import (
            bounded_domain_equivalence_certificate,
            recompute_bounded_domain_certificate,
            summarize_bounded_domain,
            verify_bounded_domain_certificate,
        )

        torch = self.torch

        class GridTokenizer:
            chat_template = None

            def __call__(self, prompt, return_tensors="pt"):
                checksum = sum(prompt.encode("utf-8")) % 29 + 3
                ids = torch.tensor([[2, checksum]], dtype=torch.long)
                return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

        tokenizer = GridTokenizer()
        certificate = bounded_domain_equivalence_certificate(
            self.model,
            tokenizer,
            oxygen_values=(25.0, 45.0),
            slope_values=(-1.0, 0.0),
            sensor_agreement_values=(False, True),
        )
        verification = verify_bounded_domain_certificate(certificate)
        self.assertTrue(verification["valid"])
        self.assertTrue(recompute_bounded_domain_certificate(self.model, tokenizer, certificate)["valid"])
        self.assertEqual(verification["verified_state_count"], 8)
        self.assertEqual(certificate["summary"]["reference_eager_exact_states"], 8)
        summary = summarize_bounded_domain(certificate)
        self.assertEqual(sum(summary["reference_output_token_counts"].values()), 8)
        incomplete = copy.deepcopy(certificate)
        incomplete["states"].pop()
        self.assertFalse(verify_bounded_domain_certificate(incomplete)["valid"])
        with self.assertRaises(ValueError):
            bounded_domain_equivalence_certificate(
                self.model,
                GridTokenizer(),
                oxygen_values=(25.0, 25.0),
                slope_values=(0.0,),
            )

    def test_sliding_mask_excludes_distant_and_future_positions(self) -> None:
        from bioprocess_runtime.reference_gemma import reference_causal_mask

        mask = reference_causal_mask(5, self.torch.float32, self.torch.device("cpu"), sliding_window=2)[0, 0]
        self.assertEqual(float(mask[4, 4]), 0.0)
        self.assertEqual(float(mask[4, 3]), 0.0)
        self.assertLess(float(mask[4, 2]), -1e30)
        self.assertLess(float(mask[1, 2]), -1e30)


if __name__ == "__main__":
    unittest.main()
