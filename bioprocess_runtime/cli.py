from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from .audit import AuditIntegrityError, append_decision, find_decision, verify_file
from .domain import Observation
from .policy import PolicySyntaxError, load_policy
from .runtime import evaluate
from .simulator import Scenario, scenarios
from .training import train_transparent_policy, write_training_report


DEFAULT_POLICY = Path(__file__).resolve().parent.parent / "policies" / "oxygen_advisory.bpr"


def _render(decision: dict[str, Any]) -> str:
    lines = [
        f"Decision: {decision['status']}",
        f"ID: {decision['decision_id']}",
        f"Reason: {decision['reason']}",
        f"Human review required: {str(decision['human_review_required']).lower()}",
    ]
    model_output = decision.get("model_output")
    if model_output:
        lines.append(f"Model score: {model_output['score']:.6f}")
        lines.append("Exact model contributions:")
        lines.extend(f"  {name}: {value:+.6f}" for name, value in model_output["contributions"].items())
    if decision.get("recommendation"):
        recommendation = decision["recommendation"]
        lines.append(
            f"Advisory recommendation: {recommendation['field']} "
            f"{recommendation['current']} -> {recommendation['proposed']} {recommendation['unit']}"
        )
    lines.append("Execution trace:")
    lines.extend(f"  [{step['result']}] {step['stage']}: {step['detail']}" for step in decision["trace"])
    return "\n".join(lines)


def _print_decision(decision: dict[str, Any], as_json: bool) -> None:
    print(json.dumps(decision, indent=2, sort_keys=True) if as_json else _render(decision))


def _run_scenario(policy_path: Path, scenario: Scenario, audit_path: Path | None) -> dict[str, Any]:
    policy = load_policy(policy_path)
    decision = evaluate(policy, scenario.observations, scenario.evaluated_at)
    if audit_path is not None:
        append_decision(audit_path, decision)
    return decision.to_dict()


def command_scenario(args: argparse.Namespace) -> int:
    available = scenarios()
    if args.name not in available:
        print(f"Unknown scenario {args.name!r}. Available: {', '.join(sorted(available))}", file=sys.stderr)
        return 2
    decision = _run_scenario(args.policy, available[args.name], args.audit)
    _print_decision(decision, args.json)
    return 0


def command_suite(args: argparse.Namespace) -> int:
    results = []
    passed = 0
    for scenario in scenarios().values():
        decision = _run_scenario(args.policy, scenario, args.audit)
        matched = decision["status"] == scenario.expected_status
        passed += int(matched)
        results.append(
            {
                "scenario": scenario.name,
                "description": scenario.description,
                "expected": scenario.expected_status,
                "actual": decision["status"],
                "passed": matched,
                "decision_id": decision["decision_id"],
            }
        )
    report = {
        "scope": "Synthetic demonstration only; not validated for GMP, clinical, or production use.",
        "policy": str(args.policy),
        "passed": passed,
        "total": len(results),
        "results": results,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if passed == len(results) else 1


def command_train(args: argparse.Namespace) -> int:
    result = train_transparent_policy(
        template_path=args.template,
        output_policy_path=args.output_policy,
        samples=args.samples,
        seed=args.seed,
        version=args.version,
    )
    if args.report:
        write_training_report(args.report, result.report)
    print(json.dumps(result.report, indent=2, sort_keys=True))
    print(f"Executable policy written to {result.policy_path}")
    return 0


def command_verify(args: argparse.Namespace) -> int:
    try:
        count = verify_file(args.audit)
    except AuditIntegrityError as exc:
        print(f"Audit verification failed: {exc}", file=sys.stderr)
        return 1
    print(f"Audit chain verified: {count} record(s)")
    return 0


def _decode_observation_value(value: Any) -> Any:
    return {
        "NaN": float("nan"),
        "Infinity": float("inf"),
        "-Infinity": float("-inf"),
    }.get(value, value) if isinstance(value, str) else value


def _observations_from_payload(payload: dict[str, Any]) -> dict[str, Observation]:
    return {
        name: Observation(
            value=_decode_observation_value(item["value"]),
            unit=item["unit"],
            observed_at=datetime.fromisoformat(item["observed_at"]),
            source=item["source"],
            quality=item["quality"],
        )
        for name, item in payload["inputs"].items()
    }


def command_replay(args: argparse.Namespace) -> int:
    try:
        verify_file(args.audit)
        original = find_decision(args.audit, args.decision_id)
    except AuditIntegrityError as exc:
        print(f"Replay refused: audit verification failed: {exc}", file=sys.stderr)
        return 1
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    policy = load_policy(args.policy)
    if original["policy"]["sha256"] != policy.source_sha256:
        print("Replay refused: the supplied policy does not match the recorded policy hash", file=sys.stderr)
        return 1
    replayed = evaluate(policy, _observations_from_payload(original), datetime.fromisoformat(original["evaluated_at"])).to_dict()
    if replayed != original:
        print("Replay failed: recomputed decision differs from the audit payload", file=sys.stderr)
        return 1
    print(f"Replay verified for decision {args.decision_id}")
    return 0


def _write_json(path: Path | None, payload: dict[str, Any]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


def command_gemma_interpret(args: argparse.Namespace) -> int:
    from .interpretability import load_local_gemma, run_concept_experiment

    model, tokenizer = load_local_gemma(args.model_path)
    report = run_concept_experiment(model, tokenizer, args.model_path)
    _write_json(args.output, report)
    return 0


def command_program_evaluate(args: argparse.Namespace) -> int:
    from .decision_program import execute_decision_program, parse_decision_program

    available = scenarios()
    if args.scenario not in available:
        print(f"Unknown scenario {args.scenario!r}", file=sys.stderr)
        return 2
    scenario = available[args.scenario]
    policy = load_policy(args.policy)
    program = parse_decision_program(args.program.read_text(encoding="utf-8"))
    result = execute_decision_program(program, policy, scenario.observations, scenario.evaluated_at).to_dict()
    _write_json(args.output, result)
    return 0 if result["accepted"] else 1


def command_gemma_program(args: argparse.Namespace) -> int:
    from .gemma_api import GemmaAPIConfig, generate_and_execute_program

    available = scenarios()
    if args.scenario not in available:
        print(f"Unknown scenario {args.scenario!r}", file=sys.stderr)
        return 2
    scenario = available[args.scenario]
    policy = load_policy(args.policy)
    config = GemmaAPIConfig(
        base_url=args.base_url,
        model=args.model,
        seed=args.seed,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout,
    )
    result = generate_and_execute_program(config, policy, scenario.observations, scenario.evaluated_at)
    _write_json(args.output, result)
    return 0 if result["execution"]["accepted"] else 1


def command_gemma_program_suite(args: argparse.Namespace) -> int:
    from .gemma_api import GemmaAPIConfig, run_program_suite

    policy = load_policy(args.policy)
    config = GemmaAPIConfig(
        base_url=args.base_url,
        model=args.model,
        seed=args.seed,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout,
    )
    result = run_program_suite(config, policy, scenarios())
    _write_json(args.output, result)
    return 0


def command_gemma_deconstruct(args: argparse.Namespace) -> int:
    from .interpretability import load_local_gemma
    from .operational_semantics import build_architecture_manifest

    model, tokenizer = load_local_gemma(args.model_path)
    manifest = build_architecture_manifest(model, tokenizer, args.model_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "model_class": manifest["model"]["class"],
        "unique_parameter_count": manifest["model"]["unique_parameter_count"],
        "module_count": len(manifest["modules"]),
        "manifest_sha256": manifest["manifest_sha256"],
        "full_manifest": str(args.output),
    }, indent=2, sort_keys=True))
    return 0


def command_gemma_trace(args: argparse.Namespace) -> int:
    from .interpretability import load_local_gemma
    from .operational_semantics import predict_with_provenance

    prompt = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else args.prompt
    model, tokenizer = load_local_gemma(args.model_path)
    report = predict_with_provenance(
        model,
        tokenizer,
        args.model_path,
        prompt,
        max_new_tokens=args.max_new_tokens,
        trace_level=args.trace_level,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "generated_text": report["prediction"]["generated_text"],
        "generated_token_ids": report["prediction"]["generated_token_ids"],
        "output_commitment_sha256": report["prediction"]["output_commitment_sha256"],
        "trace_level": report["trace"]["level"],
        "trace_record_count": report["trace"]["record_count"],
        "trace_root_sha256": report["trace"]["root_sha256"],
        "trace_chain_valid": report["trace"]["chain_verification"]["valid"],
        "reference_generate_exact_match": report["reference_generate_comparison"]["exact_match"],
        "full_report": str(args.output) if args.output else None,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["trace_chain_valid"] and summary["reference_generate_exact_match"] else 1


def command_trace_verify(args: argparse.Namespace) -> int:
    from .operational_semantics import verify_trace_chain

    report = json.loads(args.report.read_text(encoding="utf-8"))
    trace = report.get("trace", {})
    verification = verify_trace_chain(trace.get("records", []), trace.get("root_sha256"))
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if verification["valid"] else 1


def command_semantics_summary(args: argparse.Namespace) -> int:
    from .operational_semantics import summarize_operational_evidence

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    module_report = json.loads(args.module_trace.read_text(encoding="utf-8"))
    aten_report = json.loads(args.aten_trace.read_text(encoding="utf-8"))
    summary = summarize_operational_evidence(manifest, module_report, aten_report, args.model_label)
    _write_json(args.output, summary)
    return 0


def command_reference_compare(args: argparse.Namespace) -> int:
    from .interpretability import _model_device, _tokenize, load_local_gemma
    from .reference_gemma import fixed_input_equivalence_certificate, verify_fixed_input_certificate

    prompt = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else args.prompt
    model, tokenizer = load_local_gemma(args.model_path)
    input_ids = _tokenize(tokenizer, prompt, _model_device(model))["input_ids"]
    certificate = fixed_input_equivalence_certificate(model, input_ids, args.absolute_tolerance)
    verification = verify_fixed_input_certificate(certificate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    deployed = certificate["deployed_path"]
    summary = {
        "certificate": str(args.output),
        "certificate_sha256": certificate["certificate_sha256"],
        "certificate_valid": verification["valid"],
        "reference_vs_eager_logits_exact": certificate["logits"]["exact_equal"],
        "reference_vs_eager_max_error": certificate["logits"]["max_absolute_error"],
        "deployed_attention_implementation": deployed["attention_implementation"],
        "reference_vs_deployed_logits_exact": deployed["logits"]["exact_equal"],
        "reference_vs_deployed_max_logit_error": deployed["logits"]["max_absolute_error"],
        "deployed_selected_token_matches_reference": deployed["selected_token_matches_reference"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if verification["valid"] else 1


def command_reference_verify(args: argparse.Namespace) -> int:
    from .reference_gemma import recompute_fixed_input_certificate, verify_fixed_input_certificate

    certificate = json.loads(args.certificate.read_text(encoding="utf-8"))
    if args.model_path:
        from .interpretability import load_local_gemma

        model, _ = load_local_gemma(args.model_path)
        verification = recompute_fixed_input_certificate(model, certificate)
    else:
        integrity = verify_fixed_input_certificate(certificate)
        verification = {
            "valid": integrity["valid"],
            "mode": "integrity_only",
            "reexecution_performed": False,
            "integrity": integrity,
            "reason": "Supply --model-path to re-execute the computation",
        }
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if verification["valid"] else 1


def command_operator_conformance(args: argparse.Namespace) -> int:
    from .reference_gemma import operator_equation_conformance

    report = operator_equation_conformance()
    _write_json(args.output, report)
    return 0


def command_coordinate_registry(args: argparse.Namespace) -> int:
    from .interpretability import load_local_gemma
    from .reference_gemma import semantic_coordinate_registry

    model, tokenizer = load_local_gemma(args.model_path)
    registry = semantic_coordinate_registry(model, tokenizer)
    _write_json(args.output, registry)
    return 0


def command_reference_summary(args: argparse.Namespace) -> int:
    from .reference_gemma import summarize_reference_evidence

    certificate = json.loads(args.certificate.read_text(encoding="utf-8"))
    conformance = json.loads(args.conformance.read_text(encoding="utf-8"))
    coordinates = json.loads(args.coordinates.read_text(encoding="utf-8"))
    summary = summarize_reference_evidence(certificate, conformance, coordinates)
    _write_json(args.output, summary)
    return 0


def _float_axis(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _boolean_axis(value: str) -> tuple[bool, ...]:
    mapping = {"true": True, "false": False}
    try:
        return tuple(mapping[item.strip().lower()] for item in value.split(",") if item.strip())
    except KeyError as exc:
        raise ValueError("Boolean axes accept only true and false") from exc


def command_bounded_domain(args: argparse.Namespace) -> int:
    from .interpretability import load_local_gemma
    from .reference_gemma import bounded_domain_equivalence_certificate, verify_bounded_domain_certificate

    model, tokenizer = load_local_gemma(args.model_path)
    certificate = bounded_domain_equivalence_certificate(
        model,
        tokenizer,
        _float_axis(args.oxygen_values),
        _float_axis(args.slope_values),
        _boolean_axis(args.sensor_agreement),
    )
    verification = verify_bounded_domain_certificate(certificate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"certificate": str(args.output), **verification, "summary": certificate["summary"]}, indent=2, sort_keys=True))
    return 0 if verification["valid"] else 1


def command_bounded_domain_verify(args: argparse.Namespace) -> int:
    from .reference_gemma import recompute_bounded_domain_certificate, verify_bounded_domain_certificate

    certificate = json.loads(args.certificate.read_text(encoding="utf-8"))
    if args.model_path:
        from .interpretability import load_local_gemma

        model, tokenizer = load_local_gemma(args.model_path)
        verification = recompute_bounded_domain_certificate(model, tokenizer, certificate)
    else:
        integrity = verify_bounded_domain_certificate(certificate)
        verification = {
            "valid": integrity["valid"],
            "mode": "integrity_only",
            "reexecution_performed": False,
            "integrity": integrity,
            "reason": "Supply --model-path to re-execute the domain",
        }
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if verification["valid"] else 1


def command_bounded_domain_summary(args: argparse.Namespace) -> int:
    from .reference_gemma import summarize_bounded_domain, verify_bounded_domain_certificate

    certificate = json.loads(args.certificate.read_text(encoding="utf-8"))
    verification = verify_bounded_domain_certificate(certificate)
    if not verification["valid"]:
        raise ValueError("Cannot summarize an invalid bounded-domain certificate")
    _write_json(args.output, summarize_bounded_domain(certificate))
    return 0


def command_formal_proofs(args: argparse.Namespace) -> int:
    from .formal_proofs import build_formal_proof_certificate

    certificate = build_formal_proof_certificate()
    _write_json(args.output, certificate)
    return 0 if certificate["proved"] == certificate["total"] else 1


def command_formal_proofs_verify(args: argparse.Namespace) -> int:
    from .formal_proofs import verify_formal_proof_certificate

    certificate = json.loads(args.certificate.read_text(encoding="utf-8"))
    verification = verify_formal_proof_certificate(certificate)
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if verification["valid"] else 1


def command_sass_semantics(args: argparse.Namespace) -> int:
    from .sass_semantics import build_sass_semantics_certificate

    certificate = build_sass_semantics_certificate()
    _write_json(args.output, certificate)
    return 0 if certificate["proved"] == certificate["total"] else 1


def command_sass_semantics_verify(args: argparse.Namespace) -> int:
    from .sass_semantics import verify_sass_semantics_certificate

    certificate = json.loads(args.certificate.read_text(encoding="utf-8"))
    verification = verify_sass_semantics_certificate(certificate)
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if verification["valid"] else 1


def command_cupti_module_capture(args: argparse.Namespace) -> int:
    from .cupti_attestation import CuptiModuleCapture

    capture = CuptiModuleCapture(args.artifact_directory)
    capture.start()
    try:
        import torch

        from .interpretability import _model_device, _tokenize, load_local_gemma
        from .operational_semantics import tensor_descriptor
        from .reference_gemma import model_state_sha256

        prompt = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else args.prompt
        model, tokenizer = load_local_gemma(args.model_path)
        inputs = _tokenize(tokenizer, prompt, _model_device(model))
        with torch.no_grad():
            output = model(**inputs, use_cache=False, logits_to_keep=1)
        torch.cuda.synchronize()
        binding = {
            "model_state_sha256": model_state_sha256(model),
            "input_ids_tensor": tensor_descriptor(inputs["input_ids"]),
            "output_logits_tensor": tensor_descriptor(output.logits),
            "selected_token_id": int(torch.argmax(output.logits[0, -1]).item()),
        }
    finally:
        capture.stop()
    report = capture.report(binding)
    _write_json(args.output, report)
    return 0 if report["module_load_events"] > 0 and not report["callback_errors"] else 1


def command_cupti_module_verify(args: argparse.Namespace) -> int:
    from .cupti_attestation import verify_cupti_module_capture

    report = json.loads(args.report.read_text(encoding="utf-8"))
    verification = verify_cupti_module_capture(report, args.artifact_directory)
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if verification["valid"] else 1


def command_cupti_module_summary(args: argparse.Namespace) -> int:
    from .cupti_attestation import summarize_cupti_module_capture

    report = json.loads(args.report.read_text(encoding="utf-8"))
    cuda_summary = json.loads(args.cuda_summary.read_text(encoding="utf-8"))
    summary = summarize_cupti_module_capture(report, cuda_summary)
    _write_json(args.output, summary)
    return 0 if summary["profiled_static_image_binding"]["matched"] else 1


def command_nsight_target(args: argparse.Namespace) -> int:
    import torch

    from .interpretability import _model_device, _tokenize, load_local_gemma
    from .operational_semantics import tensor_descriptor
    from .reference_gemma import model_state_sha256
    from .serialization import canonical_json

    prompt = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else args.prompt
    model, tokenizer = load_local_gemma(args.model_path)
    inputs = _tokenize(tokenizer, prompt, _model_device(model))
    torch.cuda.nvtx.range_push("gemma_bound_forward")
    try:
        with torch.no_grad():
            output = model(**inputs, use_cache=False, logits_to_keep=1)
        torch.cuda.synchronize()
    finally:
        torch.cuda.nvtx.range_pop()
    binding = {
        "scope": "Execution binding emitted by the Nsight target process; the report must independently match its process ID and NVTX-filtered kernel.",
        "process_id": os.getpid(),
        "model_state_sha256": model_state_sha256(model),
        "input_ids_tensor": tensor_descriptor(inputs["input_ids"]),
        "output_logits_tensor": tensor_descriptor(output.logits),
        "selected_token_id": int(torch.argmax(output.logits[0, -1]).item()),
        "nvtx_range": "gemma_bound_forward",
    }
    binding["binding_sha256"] = hashlib.sha256(canonical_json(binding).encode("utf-8")).hexdigest()
    _write_json(args.output, binding)
    return 0


def command_nsight_launch_certificate(args: argparse.Namespace) -> int:
    from .nsight_attestation import build_nsight_launch_certificate

    certificate = build_nsight_launch_certificate(
        args.report,
        args.binding,
        args.cuda_summary,
        args.cupti_report,
        args.cupti_artifact_directory,
    )
    _write_json(args.output, certificate)
    return 0 if certificate["all_checks_pass"] else 1


def command_nsight_launch_verify(args: argparse.Namespace) -> int:
    from .nsight_attestation import verify_nsight_launch_certificate

    certificate = json.loads(args.certificate.read_text(encoding="utf-8"))
    verification = verify_nsight_launch_certificate(certificate)
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if verification["valid"] else 1


def command_nsight_permission(args: argparse.Namespace) -> int:
    from .cuda_provenance import probe_nsight_compute_permission

    record = probe_nsight_compute_permission()
    _write_json(args.output, record)
    return 0 if record["permission_granted"] else 1


def command_cuda_provenance(args: argparse.Namespace) -> int:
    from .cuda_provenance import build_cuda_provenance_manifest
    from .interpretability import _model_device, _tokenize, load_local_gemma

    prompt = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else args.prompt
    model, tokenizer = load_local_gemma(args.model_path)
    inputs = _tokenize(tokenizer, prompt, _model_device(model))
    manifest = build_cuda_provenance_manifest(model, inputs, args.kernel, args.redact)
    _write_json(args.output, manifest)
    if manifest["profiled_symbol_disassembly"]["profiled_kernel_name"] is None:
        print("Warning: no profiled kernel was bound to an extracted compatible image", file=sys.stderr)
        return 1
    return 0


def command_cuda_provenance_verify(args: argparse.Namespace) -> int:
    from .cuda_provenance import verify_cuda_provenance_manifest

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    verification = verify_cuda_provenance_manifest(manifest, args.verify_local_binaries)
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if verification["valid"] else 1


def command_cuda_provenance_summary(args: argparse.Namespace) -> int:
    from .cuda_provenance import summarize_cuda_provenance, verify_cuda_provenance_manifest

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not verify_cuda_provenance_manifest(manifest)["valid"]:
        raise ValueError("Cannot summarize an invalid CUDA provenance manifest")
    _write_json(args.output, summarize_cuda_provenance(manifest))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bioprocess-runtime",
        description="Interpretable advisory decisions over synthetic bioprocess observations.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scenario_parser = subparsers.add_parser("scenario", help="Evaluate one built-in synthetic scenario")
    scenario_parser.add_argument("name")
    scenario_parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    scenario_parser.add_argument("--audit", type=Path)
    scenario_parser.add_argument("--json", action="store_true")
    scenario_parser.set_defaults(handler=command_scenario)

    suite_parser = subparsers.add_parser("suite", help="Evaluate all built-in synthetic scenarios")
    suite_parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    suite_parser.add_argument("--audit", type=Path)
    suite_parser.add_argument("--output", type=Path)
    suite_parser.set_defaults(handler=command_suite)

    train_parser = subparsers.add_parser("train", help="Learn a transparent model from illustrative synthetic data")
    train_parser.add_argument("--template", type=Path, default=DEFAULT_POLICY)
    train_parser.add_argument("--output-policy", type=Path, required=True)
    train_parser.add_argument("--report", type=Path)
    train_parser.add_argument("--samples", type=int, default=2000)
    train_parser.add_argument("--seed", type=int, default=17)
    train_parser.add_argument("--version", default="synthetic-1.0.0")
    train_parser.set_defaults(handler=command_train)

    verify_parser = subparsers.add_parser("audit-verify", help="Verify a hash-chained NDJSON audit file")
    verify_parser.add_argument("audit", type=Path)
    verify_parser.set_defaults(handler=command_verify)

    replay_parser = subparsers.add_parser("replay", help="Recompute and compare a recorded decision")
    replay_parser.add_argument("audit", type=Path)
    replay_parser.add_argument("decision_id")
    replay_parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    replay_parser.set_defaults(handler=command_replay)

    interpret_parser = subparsers.add_parser("gemma-interpret", help="Run activation and causal-intervention experiments on local Gemma")
    interpret_parser.add_argument("--model-path", type=Path, default=Path(".models/gemma-3-270m-it"))
    interpret_parser.add_argument("--output", type=Path)
    interpret_parser.set_defaults(handler=command_gemma_interpret)

    program_parser = subparsers.add_parser("program-evaluate", help="Evaluate a saved evidence-bound decision program")
    program_parser.add_argument("program", type=Path)
    program_parser.add_argument("--scenario", default="low_oxygen")
    program_parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    program_parser.add_argument("--output", type=Path)
    program_parser.set_defaults(handler=command_program_evaluate)

    gemma_program_parser = subparsers.add_parser("gemma-program", help="Ask local Gemma 4 for a decision program and independently interpret it")
    gemma_program_parser.add_argument("--scenario", default="low_oxygen")
    gemma_program_parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    gemma_program_parser.add_argument("--base-url", default="http://127.0.0.1:5000/v1")
    gemma_program_parser.add_argument("--model", default="gemma-4-31B-it-IQ4_XS.gguf")
    gemma_program_parser.add_argument("--seed", type=int, default=17)
    gemma_program_parser.add_argument("--max-tokens", type=int, default=512)
    gemma_program_parser.add_argument("--timeout", type=int, default=300)
    gemma_program_parser.add_argument("--output", type=Path)
    gemma_program_parser.set_defaults(handler=command_gemma_program)

    gemma_suite_parser = subparsers.add_parser("gemma-program-suite", help="Evaluate local Gemma 4 program generation across all scenarios")
    gemma_suite_parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    gemma_suite_parser.add_argument("--base-url", default="http://127.0.0.1:5000/v1")
    gemma_suite_parser.add_argument("--model", default="gemma-4-31B-it-IQ4_XS.gguf")
    gemma_suite_parser.add_argument("--seed", type=int, default=17)
    gemma_suite_parser.add_argument("--max-tokens", type=int, default=512)
    gemma_suite_parser.add_argument("--timeout", type=int, default=300)
    gemma_suite_parser.add_argument("--output", type=Path)
    gemma_suite_parser.set_defaults(handler=command_gemma_program_suite)

    deconstruct_parser = subparsers.add_parser("gemma-deconstruct", help="Write a static architecture and parameter-byte manifest")
    deconstruct_parser.add_argument("--model-path", type=Path, default=Path(".models/gemma-3-270m-it"))
    deconstruct_parser.add_argument("--output", type=Path, required=True)
    deconstruct_parser.set_defaults(handler=command_gemma_deconstruct)

    trace_parser = subparsers.add_parser("gemma-trace", help="Predict greedy tokens with module or ATen operation provenance")
    trace_parser.add_argument("--model-path", type=Path, default=Path(".models/gemma-3-270m-it"))
    prompt_group = trace_parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt")
    prompt_group.add_argument("--prompt-file", type=Path)
    trace_parser.add_argument("--max-new-tokens", type=int, default=1)
    trace_parser.add_argument("--trace-level", choices=("module", "aten"), default="module")
    trace_parser.add_argument("--output", type=Path, required=True)
    trace_parser.set_defaults(handler=command_gemma_trace)

    trace_verify_parser = subparsers.add_parser("trace-verify", help="Verify an operation trace hash chain")
    trace_verify_parser.add_argument("report", type=Path)
    trace_verify_parser.set_defaults(handler=command_trace_verify)

    summary_parser = subparsers.add_parser("gemma-semantics-summary", help="Build a compact reproducible summary from full provenance artifacts")
    summary_parser.add_argument("--manifest", type=Path, required=True)
    summary_parser.add_argument("--module-trace", type=Path, required=True)
    summary_parser.add_argument("--aten-trace", type=Path, required=True)
    summary_parser.add_argument("--model-label", default="unsloth/gemma-3-270m-it")
    summary_parser.add_argument("--output", type=Path, required=True)
    summary_parser.set_defaults(handler=command_semantics_summary)

    reference_parser = subparsers.add_parser("gemma-reference-compare", help="Compare independent Python orchestration with Hugging Face eager and deployed attention paths")
    reference_parser.add_argument("--model-path", type=Path, default=Path(".models/gemma-3-270m-it"))
    reference_prompt = reference_parser.add_mutually_exclusive_group(required=True)
    reference_prompt.add_argument("--prompt")
    reference_prompt.add_argument("--prompt-file", type=Path)
    reference_parser.add_argument("--absolute-tolerance", type=float, default=0.0)
    reference_parser.add_argument("--output", type=Path, required=True)
    reference_parser.set_defaults(handler=command_reference_compare)

    reference_verify_parser = subparsers.add_parser("reference-verify", help="Re-execute and verify a fixed-input reference equivalence certificate")
    reference_verify_parser.add_argument("certificate", type=Path)
    reference_verify_parser.add_argument("--model-path", type=Path)
    reference_verify_parser.set_defaults(handler=command_reference_verify)

    conformance_parser = subparsers.add_parser("operator-conformance", help="Compare operator implementations with separate scalar equations")
    conformance_parser.add_argument("--output", type=Path)
    conformance_parser.set_defaults(handler=command_operator_conformance)

    coordinate_parser = subparsers.add_parser("gemma-coordinate-registry", help="Describe architecture-defined tensor coordinate roles")
    coordinate_parser.add_argument("--model-path", type=Path, default=Path(".models/gemma-3-270m-it"))
    coordinate_parser.add_argument("--output", type=Path)
    coordinate_parser.set_defaults(handler=command_coordinate_registry)

    reference_summary_parser = subparsers.add_parser("gemma-reference-summary", help="Build a compact summary from reference proof artifacts")
    reference_summary_parser.add_argument("--certificate", type=Path, required=True)
    reference_summary_parser.add_argument("--conformance", type=Path, required=True)
    reference_summary_parser.add_argument("--coordinates", type=Path, required=True)
    reference_summary_parser.add_argument("--output", type=Path, required=True)
    reference_summary_parser.set_defaults(handler=command_reference_summary)

    domain_parser = subparsers.add_parser("gemma-bounded-domain", help="Exhaustively verify a declared canonical oxygen-state grid")
    domain_parser.add_argument("--model-path", type=Path, default=Path(".models/gemma-3-270m-it"))
    domain_parser.add_argument("--oxygen-values", default="25,35,45")
    domain_parser.add_argument("--slope-values", default="-1,0,1")
    domain_parser.add_argument("--sensor-agreement", default="false,true")
    domain_parser.add_argument("--output", type=Path, required=True)
    domain_parser.set_defaults(handler=command_bounded_domain)

    domain_verify_parser = subparsers.add_parser("bounded-domain-verify", help="Re-execute and verify a bounded-domain equivalence certificate")
    domain_verify_parser.add_argument("certificate", type=Path)
    domain_verify_parser.add_argument("--model-path", type=Path)
    domain_verify_parser.set_defaults(handler=command_bounded_domain_verify)

    domain_summary_parser = subparsers.add_parser("bounded-domain-summary", help="Build a compact summary from a verified bounded-domain certificate")
    domain_summary_parser.add_argument("certificate", type=Path)
    domain_summary_parser.add_argument("--output", type=Path, required=True)
    domain_summary_parser.set_defaults(handler=command_bounded_domain_summary)

    formal_parser = subparsers.add_parser("formal-proofs", help="Machine-check universal finite-bitvector and integer operator properties")
    formal_parser.add_argument("--output", type=Path)
    formal_parser.set_defaults(handler=command_formal_proofs)

    formal_verify_parser = subparsers.add_parser("formal-proofs-verify", help="Re-execute an SMT proof certificate")
    formal_verify_parser.add_argument("certificate", type=Path)
    formal_verify_parser.set_defaults(handler=command_formal_proofs_verify)

    sass_parser = subparsers.add_parser("sass-semantics", help="Check a proposed bitvector semantics for a small SASS subset; not NVIDIA-certified")
    sass_parser.add_argument("--output", type=Path)
    sass_parser.set_defaults(handler=command_sass_semantics)

    sass_verify_parser = subparsers.add_parser("sass-semantics-verify", help="Re-execute a proposed SASS-semantics certificate")
    sass_verify_parser.add_argument("certificate", type=Path)
    sass_verify_parser.set_defaults(handler=command_sass_semantics_verify)

    cupti_parser = subparsers.add_parser("cupti-module-capture", help="Capture cubins presented during CUDA module-load callbacks")
    cupti_parser.add_argument("--model-path", type=Path, default=Path(".models/gemma-3-270m-it"))
    cupti_prompt = cupti_parser.add_mutually_exclusive_group(required=True)
    cupti_prompt.add_argument("--prompt")
    cupti_prompt.add_argument("--prompt-file", type=Path)
    cupti_parser.add_argument("--artifact-directory", type=Path, default=Path("artifacts/cupti_modules"))
    cupti_parser.add_argument("--output", type=Path, required=True)
    cupti_parser.set_defaults(handler=command_cupti_module_capture)

    cupti_verify_parser = subparsers.add_parser("cupti-module-verify", help="Verify CUPTI module-capture integrity and optional local cubins")
    cupti_verify_parser.add_argument("report", type=Path)
    cupti_verify_parser.add_argument("--artifact-directory", type=Path)
    cupti_verify_parser.set_defaults(handler=command_cupti_module_verify)

    cupti_summary_parser = subparsers.add_parser("cupti-module-summary", help="Bind a CUPTI module capture to the statically disassembled image")
    cupti_summary_parser.add_argument("report", type=Path)
    cupti_summary_parser.add_argument("--cuda-summary", type=Path, required=True)
    cupti_summary_parser.add_argument("--output", type=Path, required=True)
    cupti_summary_parser.set_defaults(handler=command_cupti_module_summary)

    nsight_target_parser = subparsers.add_parser("gemma-nsight-target", help="Run one NVTX-bounded Gemma forward and emit an execution binding")
    nsight_target_parser.add_argument("--model-path", type=Path, default=Path(".models/gemma-3-270m-it"))
    nsight_target_prompt = nsight_target_parser.add_mutually_exclusive_group(required=True)
    nsight_target_prompt.add_argument("--prompt")
    nsight_target_prompt.add_argument("--prompt-file", type=Path)
    nsight_target_parser.add_argument("--output", type=Path, required=True)
    nsight_target_parser.set_defaults(handler=command_nsight_target)

    nsight_certificate_parser = subparsers.add_parser("nsight-launch-certificate", help="Bind an Nsight launch SASS view to Gemma execution and CUPTI cubin evidence")
    nsight_certificate_parser.add_argument("--report", type=Path, required=True)
    nsight_certificate_parser.add_argument("--binding", type=Path, required=True)
    nsight_certificate_parser.add_argument("--cuda-summary", type=Path, required=True)
    nsight_certificate_parser.add_argument("--cupti-report", type=Path, required=True)
    nsight_certificate_parser.add_argument("--cupti-artifact-directory", type=Path, required=True)
    nsight_certificate_parser.add_argument("--output", type=Path, required=True)
    nsight_certificate_parser.set_defaults(handler=command_nsight_launch_certificate)

    nsight_verify_parser = subparsers.add_parser("nsight-launch-verify", help="Verify an Nsight launch certificate's integrity and internal claims")
    nsight_verify_parser.add_argument("certificate", type=Path)
    nsight_verify_parser.set_defaults(handler=command_nsight_launch_verify)

    nsight_parser = subparsers.add_parser("cuda-nsight-permission", help="Probe permission for launch-specific Nsight Compute evidence")
    nsight_parser.add_argument("--output", type=Path)
    nsight_parser.set_defaults(handler=command_nsight_permission)

    cuda_parser = subparsers.add_parser("cuda-provenance", help="Profile CUDA launches and fingerprint compatible embedded device code")
    cuda_parser.add_argument("--model-path", type=Path, default=Path(".models/gemma-3-270m-it"))
    cuda_prompt = cuda_parser.add_mutually_exclusive_group(required=True)
    cuda_prompt.add_argument("--prompt")
    cuda_prompt.add_argument("--prompt-file", type=Path)
    cuda_parser.add_argument("--kernel", help="Regular expression selecting an observed PyTorch-native kernel symbol")
    cuda_parser.add_argument("--redact", action="store_true", help="Redact device name, input IDs, and binary hashes")
    cuda_parser.add_argument("--output", type=Path, required=True)
    cuda_parser.set_defaults(handler=command_cuda_provenance)

    cuda_verify_parser = subparsers.add_parser("cuda-provenance-verify", help="Verify a CUDA provenance manifest and optional local binary hashes")
    cuda_verify_parser.add_argument("manifest", type=Path)
    cuda_verify_parser.add_argument("--verify-local-binaries", action="store_true")
    cuda_verify_parser.set_defaults(handler=command_cuda_provenance_verify)

    cuda_summary_parser = subparsers.add_parser("cuda-provenance-summary", help="Build a compact summary from a verified CUDA provenance manifest")
    cuda_summary_parser.add_argument("manifest", type=Path)
    cuda_summary_parser.add_argument("--output", type=Path, required=True)
    cuda_summary_parser.set_defaults(handler=command_cuda_provenance_summary)
    return parser


def main() -> int:
    try:
        args = build_parser().parse_args()
        return args.handler(args)
    except (PolicySyntaxError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
