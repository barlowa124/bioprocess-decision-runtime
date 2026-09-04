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


def command_gemma_ir_compile(args: argparse.Namespace) -> int:
    from .gemma_ir import compile_gemma_ir, verify_gemma_ir

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    program = compile_gemma_ir(manifest)
    verification = verify_gemma_ir(program, manifest)
    _write_json(args.output, program)
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if verification["valid"] else 1


def command_gemma_ir_verify(args: argparse.Namespace) -> int:
    from .gemma_ir import verify_gemma_ir

    program = json.loads(args.program.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    verification = verify_gemma_ir(program, manifest)
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if verification["valid"] else 1


def command_gemma_ir_rationale(args: argparse.Namespace) -> int:
    from .gemma_ir import build_rationale_slice, verify_rationale_slice

    program = json.loads(args.program.read_text(encoding="utf-8"))
    rationale = build_rationale_slice(program, args.output_tensor)
    verification = verify_rationale_slice(program, rationale)
    _write_json(args.output, rationale)
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if verification["valid"] else 1


def command_gemma_ir_rationale_verify(args: argparse.Namespace) -> int:
    from .gemma_ir import verify_rationale_slice

    program = json.loads(args.program.read_text(encoding="utf-8"))
    rationale = json.loads(args.rationale.read_text(encoding="utf-8"))
    verification = verify_rationale_slice(program, rationale)
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if verification["valid"] else 1


def command_gemma_ir_predict(args: argparse.Namespace) -> int:
    import torch

    from .gemma_ir_interpreter import (
        bind_model_tensors,
        build_ir_execution_certificate,
        execute_gemma_ir,
        verify_ir_execution_certificate,
    )
    from .interpretability import _model_device, _tokenize, load_local_gemma
    from .reference_gemma import model_state_sha256

    program = json.loads(args.program.read_text(encoding="utf-8"))
    prompt = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else args.prompt
    model, tokenizer = load_local_gemma(args.model_path)
    input_ids = _tokenize(tokenizer, prompt, _model_device(model))["input_ids"]
    parameters = bind_model_tensors(program, model)
    original_attention = model.config._attn_implementation
    model.config._attn_implementation = "eager"
    try:
        with torch.no_grad():
            prediction = execute_gemma_ir(program, parameters, input_ids)
            observed = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                logits_to_keep=1,
            )
    finally:
        model.config._attn_implementation = original_attention
    certificate = build_ir_execution_certificate(
        program,
        input_ids,
        prediction,
        observed.logits,
        "huggingface_eager",
        model_state_sha256(model),
    )
    verification = verify_ir_execution_certificate(program, certificate)
    _write_json(args.output, certificate)
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if verification["valid"] else 1


def command_gemma_ir_execution_verify(args: argparse.Namespace) -> int:
    from .gemma_ir_interpreter import (
        recompute_ir_execution_certificate,
        verify_ir_execution_certificate,
    )

    program = json.loads(args.program.read_text(encoding="utf-8"))
    certificate = json.loads(args.certificate.read_text(encoding="utf-8"))
    if args.model_path:
        from .interpretability import load_local_gemma

        model, _ = load_local_gemma(args.model_path)
        verification = recompute_ir_execution_certificate(
            program, certificate, model
        )
    else:
        integrity = verify_ir_execution_certificate(program, certificate)
        verification = {
            "valid": integrity["valid"],
            "mode": "integrity_only",
            "reexecution_performed": False,
            "integrity": integrity,
            "reason": "Supply --model-path to re-execute the prediction",
        }
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if verification["valid"] else 1


def command_gemma_ir_execution_summary(args: argparse.Namespace) -> int:
    from .gemma_ir_interpreter import summarize_ir_execution

    program = json.loads(args.program.read_text(encoding="utf-8"))
    certificate = json.loads(args.certificate.read_text(encoding="utf-8"))
    summary = summarize_ir_execution(program, certificate)
    _write_json(args.output, summary)
    return 0


def command_gemma_ir_execution_summary_verify(args: argparse.Namespace) -> int:
    from .gemma_ir_interpreter import verify_ir_execution_summary

    program = json.loads(args.program.read_text(encoding="utf-8"))
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    certificate = json.loads(args.certificate.read_text(encoding="utf-8"))
    verification = verify_ir_execution_summary(program, summary, certificate)
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if verification["valid"] else 1


def command_gemma_ir_primitive_qualification(args: argparse.Namespace) -> int:
    from .gemma_ir_primitives import (
        build_primitive_qualification_certificate,
        verify_primitive_qualification_certificate,
    )

    certificate = build_primitive_qualification_certificate()
    verification = verify_primitive_qualification_certificate(certificate)
    _write_json(args.output, certificate)
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if verification["valid"] else 1


def command_gemma_ir_primitive_qualification_verify(args: argparse.Namespace) -> int:
    from .gemma_ir_primitives import verify_primitive_qualification_certificate

    certificate = json.loads(args.certificate.read_text(encoding="utf-8"))
    verification = verify_primitive_qualification_certificate(certificate)
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if verification["valid"] else 1


def command_gemma_bfloat16_semantics(args: argparse.Namespace) -> int:
    from .gemma_float_semantics import (
        build_bfloat16_semantics_certificate,
        verify_bfloat16_semantics_certificate,
    )

    certificate = build_bfloat16_semantics_certificate()
    verification = verify_bfloat16_semantics_certificate(certificate)
    _write_json(args.output, certificate)
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if verification["valid"] else 1


def command_gemma_bfloat16_semantics_verify(args: argparse.Namespace) -> int:
    from .gemma_float_semantics import verify_bfloat16_semantics_certificate

    certificate = json.loads(args.certificate.read_text(encoding="utf-8"))
    verification = verify_bfloat16_semantics_certificate(certificate)
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if verification["valid"] else 1


def command_gemma_reduction_characterization(args: argparse.Namespace) -> int:
    from .gemma_reduction_semantics import (
        build_reduction_characterization_certificate,
        verify_reduction_characterization_certificate,
    )

    certificate = build_reduction_characterization_certificate()
    verification = verify_reduction_characterization_certificate(certificate)
    _write_json(args.output, certificate)
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if verification["valid"] else 1


def command_gemma_reduction_characterization_verify(args: argparse.Namespace) -> int:
    from .gemma_reduction_semantics import (
        verify_reduction_characterization_certificate,
    )

    certificate = json.loads(args.certificate.read_text(encoding="utf-8"))
    verification = verify_reduction_characterization_certificate(certificate)
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if verification["valid"] else 1


def command_gemma_ir_primitive_gate(args: argparse.Namespace) -> int:
    from .gemma_ir_primitives import (
        build_primitive_qualification_gate,
        verify_primitive_qualification_gate,
    )

    program = json.loads(args.program.read_text(encoding="utf-8"))
    certificate = json.loads(args.certificate.read_text(encoding="utf-8"))
    bfloat16_certificate = json.loads(
        args.bfloat16_certificate.read_text(encoding="utf-8")
    )
    reduction_certificate = json.loads(
        args.reduction_certificate.read_text(encoding="utf-8")
    )
    gate = build_primitive_qualification_gate(
        program, certificate, bfloat16_certificate, reduction_certificate
    )
    verification = verify_primitive_qualification_gate(
        program,
        certificate,
        bfloat16_certificate,
        reduction_certificate,
        gate,
    )
    _write_json(args.output, gate)
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if verification["valid"] else 1


def command_gemma_ir_primitive_gate_verify(args: argparse.Namespace) -> int:
    from .gemma_ir_primitives import verify_primitive_qualification_gate

    program = json.loads(args.program.read_text(encoding="utf-8"))
    certificate = json.loads(args.certificate.read_text(encoding="utf-8"))
    bfloat16_certificate = json.loads(
        args.bfloat16_certificate.read_text(encoding="utf-8")
    )
    reduction_certificate = json.loads(
        args.reduction_certificate.read_text(encoding="utf-8")
    )
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    verification = verify_primitive_qualification_gate(
        program,
        certificate,
        bfloat16_certificate,
        reduction_certificate,
        gate,
    )
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if verification["valid"] else 1


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


def command_operator_correspondence(args: argparse.Namespace) -> int:
    from .operator_correspondence import build_operator_correspondence

    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    correspondence = build_operator_correspondence(suite)
    _write_json(args.output, correspondence)
    return 0 if correspondence["all_stage_patterns_observed_and_attested"] else 1


def command_operator_correspondence_verify(args: argparse.Namespace) -> int:
    from .operator_correspondence import verify_operator_correspondence

    correspondence = json.loads(args.correspondence.read_text(encoding="utf-8"))
    verification = verify_operator_correspondence(correspondence)
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


def command_cuda_metadata(args: argparse.Namespace) -> int:
    from .cuda_metadata import build_cuda_metadata_conformance

    certificate = build_cuda_metadata_conformance()
    _write_json(args.output, certificate)
    return 0 if certificate["all_checks_pass"] else 1


def command_cuda_metadata_verify(args: argparse.Namespace) -> int:
    from .cuda_metadata import verify_cuda_metadata_conformance

    certificate = json.loads(args.certificate.read_text(encoding="utf-8"))
    verification = verify_cuda_metadata_conformance(certificate)
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if verification["valid"] else 1


def command_launch_arguments(args: argparse.Namespace) -> int:
    from .launch_arguments import CuptiLaunchArgumentCapture, redact_launch_argument_artifact

    capture = CuptiLaunchArgumentCapture(args.kernel)
    capture.start()
    try:
        import torch

        from .interpretability import _model_device, _tokenize, load_local_gemma
        from .module_invocation import AttentionDispatchCapture, ModuleNvtxCapture
        from .operational_semantics import tensor_descriptor
        from .reference_gemma import model_state_sha256
        from .serialization import canonical_json

        prompt = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else args.prompt
        model, tokenizer = load_local_gemma(args.model_path)
        inputs = _tokenize(tokenizer, prompt, _model_device(model))
        module_capture = ModuleNvtxCapture(model, tuple(args.module_nvtx_pattern))
        module_capture.__enter__()
        attention_dispatch = AttentionDispatchCapture(module_capture) if "fmha_cutlass" in args.kernel else None
        if attention_dispatch is not None:
            attention_dispatch.__enter__()
        capture.range_provider = lambda: [invocation["module"] for invocation in module_capture.open_stack]
        torch.cuda.nvtx.range_push("gemma_bound_forward")
        try:
            with torch.no_grad():
                output = model(**inputs, use_cache=False, logits_to_keep=1)
            torch.cuda.synchronize()
        finally:
            if attention_dispatch is not None:
                attention_dispatch.__exit__(*sys.exc_info())
            module_capture.__exit__(*sys.exc_info())
            torch.cuda.nvtx.range_pop()
        tensor_storage_ranges = module_capture.tensor_storage_ranges()
        if attention_dispatch is not None:
            tensor_storage_ranges.extend(attention_dispatch.tensor_storage_ranges())
        module_report = module_capture.report()
        attention_dispatch_report = attention_dispatch.report() if attention_dispatch is not None else None
        launch_report = capture.report(module_report, tensor_storage_ranges)
        body = {
            "scope": "CUPTI launch-parameter and qualified-module evidence for one exact kernel symbol; not a typed kernel-signature proof.",
            "privacy": {"redacted": False},
            "model_state_sha256": model_state_sha256(model),
            "input_ids_tensor": tensor_descriptor(inputs["input_ids"]),
            "output_logits_tensor": tensor_descriptor(output.logits),
            "selected_token_id": int(torch.argmax(output.logits[0, -1]).item()),
            "attention_expectations": {
                "batch_size": int(inputs["input_ids"].shape[0]),
                "sequence_length": int(inputs["input_ids"].shape[1]),
                "head_dim": int(model.config.head_dim),
                "num_attention_heads": int(model.config.num_attention_heads),
                "num_key_value_heads": int(model.config.num_key_value_heads),
                "scaling": float(model.model.layers[0].self_attn.scaling),
                "is_sliding": bool(model.model.layers[0].self_attn.is_sliding),
                "sliding_window": int(model.config.sliding_window),
            },
            "module_invocation_report": module_report,
            "attention_dispatch_report": attention_dispatch_report,
            "launch_argument_report": launch_report,
        }
        body["artifact_sha256"] = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    finally:
        capture.stop()
    if args.redact:
        body = redact_launch_argument_artifact(body)
    _write_json(args.output, body)
    return 0 if launch_report["launches"] and not launch_report["callback_errors"] else 1


def command_launch_argument_summary(args: argparse.Namespace) -> int:
    from .launch_arguments import build_launch_argument_summary

    artifacts = []
    for value in args.artifact:
        path, separator, module = value.partition("=")
        if not separator or not path or not module:
            raise ValueError("--artifact values must use PATH=QUALIFIED_MODULE")
        artifacts.append((json.loads(Path(path).read_text(encoding="utf-8")), module))
    summary = build_launch_argument_summary(artifacts)
    _write_json(args.output, summary)
    return 0 if summary["valid_entries"] == summary["total_entries"] else 1


def command_launch_argument_summary_verify(args: argparse.Namespace) -> int:
    from .launch_arguments import verify_launch_argument_summary

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    verification = verify_launch_argument_summary(summary)
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if verification["valid"] else 1


def command_sass_expressions(args: argparse.Namespace) -> int:
    from .sass_expressions import build_sass_expression_certificate

    memory = json.loads(args.sass_memory.read_text(encoding="utf-8"))
    bounds = json.loads(args.logical_bounds.read_text(encoding="utf-8"))
    semantics = json.loads(args.sass_semantics.read_text(encoding="utf-8"))
    nsight = json.loads(args.nsight.read_text(encoding="utf-8"))
    certificate = build_sass_expression_certificate(
        args.cuobjdump, args.cubin, args.kernel, memory, bounds, semantics, nsight
    )
    _write_json(args.output, certificate)
    return 0 if certificate["all_checks_pass"] else 1


def command_sass_expression_summary(args: argparse.Namespace) -> int:
    from .sass_expressions import build_sass_expression_summary

    certificate = json.loads(args.certificate.read_text(encoding="utf-8"))
    summary = build_sass_expression_summary(certificate)
    _write_json(args.output, summary)
    return 0


def command_sass_expression_summary_verify(args: argparse.Namespace) -> int:
    from .sass_expressions import verify_sass_expression_summary

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    verification = verify_sass_expression_summary(summary)
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if verification["valid"] else 1


def command_sass_expressions_verify(args: argparse.Namespace) -> int:
    from .sass_expressions import verify_sass_expression_certificate

    certificate = json.loads(args.certificate.read_text(encoding="utf-8"))
    verification = verify_sass_expression_certificate(certificate)
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if verification["valid"] else 1


def command_sass_carry_evidence_template(args: argparse.Namespace) -> int:
    from .sass_expressions import build_carry_evidence_template

    certificate = json.loads(args.certificate.read_text(encoding="utf-8"))
    template = build_carry_evidence_template(certificate)
    _write_json(args.output, template)
    return 0


def command_sass_carry_evidence_verify(args: argparse.Namespace) -> int:
    from .sass_expressions import verify_carry_evidence_bundle

    certificate = json.loads(args.certificate.read_text(encoding="utf-8"))
    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    verification = verify_carry_evidence_bundle(bundle, certificate)
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if verification["valid"] else 1


def command_sass_carry_reproduction_verify(args: argparse.Namespace) -> int:
    from .sass_expressions import verify_carry_evidence_reproduction

    certificate = json.loads(args.certificate.read_text(encoding="utf-8"))
    primary = json.loads(args.primary.read_text(encoding="utf-8"))
    replicate = json.loads(args.replicate.read_text(encoding="utf-8"))
    verification = verify_carry_evidence_reproduction(
        primary, replicate, certificate
    )
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if verification["valid"] else 1


def command_sass_dynamic_summary(args: argparse.Namespace) -> int:
    from .sass_dynamic import build_sass_dynamic_summary

    reports = {
        "encoding": json.loads(args.encoding.read_text(encoding="utf-8")),
        "low": json.loads(args.low.read_text(encoding="utf-8")),
        "high": json.loads(args.high.read_text(encoding="utf-8")),
        "query_pair": json.loads(args.query_pair.read_text(encoding="utf-8")),
        "key_pair": json.loads(args.key_pair.read_text(encoding="utf-8")),
        "value_triple": json.loads(args.value_triple.read_text(encoding="utf-8")),
        "p2r_encoding": json.loads(
            args.p2r_encoding.read_text(encoding="utf-8")
        ),
        "combined": json.loads(args.combined.read_text(encoding="utf-8")),
    }
    summary = build_sass_dynamic_summary(
        reports, args.tool_directory, args.acquisition_tool_sha256
    )
    _write_json(args.output, summary)
    return 0


def command_sass_dynamic_summary_verify(args: argparse.Namespace) -> int:
    from .sass_dynamic import verify_sass_dynamic_summary

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    verification = verify_sass_dynamic_summary(
        summary, args.tool_directory, args.acquisition_tool_sha256
    )
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if verification["valid"] else 1


def command_sass_output_reachability_summary(args: argparse.Namespace) -> int:
    from .sass_output_reachability import build_sass_output_reachability_summary

    report = json.loads(args.report.read_text(encoding="utf-8"))
    summary = build_sass_output_reachability_summary(
        report, args.acquisition_tool_sha256
    )
    _write_json(args.output, summary)
    return 0


def command_sass_output_reachability_verify(args: argparse.Namespace) -> int:
    from .sass_output_reachability import verify_sass_output_reachability_summary

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    verification = verify_sass_output_reachability_summary(
        summary, args.acquisition_tool_sha256
    )
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if verification["valid"] else 1


def command_attention_bounds(args: argparse.Namespace) -> int:
    from .attention_bounds import (
        build_attention_logical_bounds_certificate,
        redact_attention_logical_bounds_certificate,
    )

    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    attention = json.loads(args.attention.read_text(encoding="utf-8"))
    certificate = build_attention_logical_bounds_certificate(artifact, attention)
    if args.redact:
        certificate = redact_attention_logical_bounds_certificate(certificate)
    _write_json(args.output, certificate)
    return 0 if certificate["all_checks_pass"] else 1


def command_attention_bounds_verify(args: argparse.Namespace) -> int:
    from .attention_bounds import verify_attention_logical_bounds_certificate

    certificate = json.loads(args.certificate.read_text(encoding="utf-8"))
    verification = verify_attention_logical_bounds_certificate(certificate)
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if verification["valid"] else 1


def command_sass_memory(args: argparse.Namespace) -> int:
    from .sass_memory import build_sass_memory_certificate

    nsight = json.loads(args.nsight.read_text(encoding="utf-8"))
    attention = json.loads(args.attention.read_text(encoding="utf-8"))
    semantics = json.loads(args.sass_semantics.read_text(encoding="utf-8"))
    certificate = build_sass_memory_certificate(
        args.cuobjdump, args.cubin, args.kernel, nsight, attention, semantics
    )
    _write_json(args.output, certificate)
    return 0 if certificate["all_checks_pass"] else 1


def command_sass_memory_verify(args: argparse.Namespace) -> int:
    from .sass_memory import verify_sass_memory_certificate

    certificate = json.loads(args.certificate.read_text(encoding="utf-8"))
    nsight = json.loads(args.nsight.read_text(encoding="utf-8"))
    attention = json.loads(args.attention.read_text(encoding="utf-8"))
    semantics = json.loads(args.sass_semantics.read_text(encoding="utf-8"))
    verification = verify_sass_memory_certificate(
        certificate, args.cuobjdump, args.cubin, nsight, attention, semantics
    )
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if verification["valid"] else 1


def command_attention_parameters(args: argparse.Namespace) -> int:
    from .attention_parameters import build_attention_parameter_certificate

    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    signatures = json.loads(args.signatures.read_text(encoding="utf-8"))
    certificate = build_attention_parameter_certificate(artifact, signatures)
    _write_json(args.output, certificate)
    return 0 if certificate["all_checks_pass"] else 1


def command_attention_parameters_verify(args: argparse.Namespace) -> int:
    from .attention_parameters import verify_attention_parameter_certificate

    certificate = json.loads(args.certificate.read_text(encoding="utf-8"))
    verification = verify_attention_parameter_certificate(certificate)
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if verification["valid"] else 1


def command_kernel_signatures(args: argparse.Namespace) -> int:
    from .kernel_signatures import build_kernel_signature_certificate

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    certificate = build_kernel_signature_certificate(summary)
    _write_json(args.output, certificate)
    return 0 if certificate["partial_parameter_schema_entries"] or certificate["source_layout_reconstructed_entries"] else 1


def command_kernel_signatures_verify(args: argparse.Namespace) -> int:
    from .kernel_signatures import verify_kernel_signature_certificate

    certificate = json.loads(args.certificate.read_text(encoding="utf-8"))
    verification = verify_kernel_signature_certificate(certificate)
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if verification["valid"] else 1


def command_nsight_target(args: argparse.Namespace) -> int:
    import torch

    from .interpretability import _model_device, _tokenize, load_local_gemma
    from .operational_semantics import tensor_descriptor
    from .reference_gemma import model_state_sha256
    from .serialization import canonical_json

    prompt = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else args.prompt
    model, tokenizer = load_local_gemma(args.model_path)
    inputs = _tokenize(tokenizer, prompt, _model_device(model))
    module_capture = None
    if args.module_nvtx_pattern:
        from .module_invocation import ModuleNvtxCapture

        module_capture = ModuleNvtxCapture(model, tuple(args.module_nvtx_pattern))
        module_capture.__enter__()
    torch.cuda.nvtx.range_push("gemma_bound_forward")
    try:
        with torch.no_grad():
            output = model(**inputs, use_cache=False, logits_to_keep=1)
        torch.cuda.synchronize()
    finally:
        if module_capture is not None:
            module_capture.__exit__(*sys.exc_info())
        torch.cuda.nvtx.range_pop()
    binding = {
        "scope": "Execution binding emitted by the Nsight target process; the report must independently match its process ID and NVTX-filtered kernel.",
        "process_id": os.getpid(),
        "model_state_sha256": model_state_sha256(model),
        "input_ids_tensor": tensor_descriptor(inputs["input_ids"]),
        "output_logits_tensor": tensor_descriptor(output.logits),
        "selected_token_id": int(torch.argmax(output.logits[0, -1]).item()),
        "nvtx_range": "gemma_bound_forward",
        "module_invocation_report": module_capture.report() if module_capture is not None else None,
    }
    binding["binding_sha256"] = hashlib.sha256(canonical_json(binding).encode("utf-8")).hexdigest()
    _write_json(args.output, binding)
    return 0


def command_nsight_capture(args: argparse.Namespace) -> int:
    from .nsight_attestation import capture_nsight_launch

    prompt = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else args.prompt
    request = capture_nsight_launch(
        args.kernel,
        args.model_path,
        prompt,
        args.report_base,
        args.binding,
        args.request,
        tuple(args.module_nvtx_pattern),
    )
    print(json.dumps(request, indent=2, sort_keys=True))
    return 0


def command_nsight_launch_certificate(args: argparse.Namespace) -> int:
    from .nsight_attestation import build_nsight_launch_certificate

    certificate = build_nsight_launch_certificate(
        args.report,
        args.binding,
        args.cuda_summary,
        args.cupti_report,
        args.cupti_artifact_directory,
        capture_request_path=args.capture_request,
    )
    _write_json(args.output, certificate)
    return 0 if certificate["all_checks_pass"] else 1


def command_nsight_launch_verify(args: argparse.Namespace) -> int:
    from .nsight_attestation import verify_nsight_launch_certificate

    certificate = json.loads(args.certificate.read_text(encoding="utf-8"))
    verification = verify_nsight_launch_certificate(certificate)
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if verification["valid"] else 1


def command_nsight_kernel_suite(args: argparse.Namespace) -> int:
    from .nsight_attestation import build_nsight_kernel_suite

    suite = build_nsight_kernel_suite(
        args.report_directory,
        args.cuda_manifest,
        args.cupti_report,
        args.cupti_artifact_directory,
        args.redact,
    )
    _write_json(args.output, suite)
    return 0 if suite["complete"] else 1


def command_nsight_kernel_suite_verify(args: argparse.Namespace) -> int:
    from .nsight_attestation import verify_nsight_kernel_suite

    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    cupti_report = json.loads(args.cupti_report.read_text(encoding="utf-8")) if args.cupti_report else None
    verification = verify_nsight_kernel_suite(suite, args.certificate_directory, cupti_report)
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if verification["valid"] else 1


def command_module_invocation_certificate(args: argparse.Namespace) -> int:
    from .module_invocation import build_module_invocation_certificate

    certificate = build_module_invocation_certificate(
        args.report,
        args.binding,
        args.launch_certificate,
        args.expected_innermost_module,
    )
    _write_json(args.output, certificate)
    return 0 if certificate["all_checks_pass"] else 1


def command_module_invocation_verify(args: argparse.Namespace) -> int:
    from .module_invocation import verify_module_invocation_certificate

    certificate = json.loads(args.certificate.read_text(encoding="utf-8"))
    verification = verify_module_invocation_certificate(certificate)
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if verification["valid"] else 1


def command_module_invocation_summary(args: argparse.Namespace) -> int:
    from .module_invocation import build_module_invocation_summary

    certificates = [json.loads(path.read_text(encoding="utf-8")) for path in args.certificates]
    summary = build_module_invocation_summary(certificates)
    _write_json(args.output, summary)
    return 0 if summary["complete"] else 1


def command_module_invocation_summary_verify(args: argparse.Namespace) -> int:
    from .module_invocation import verify_module_invocation_summary

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    verification = verify_module_invocation_summary(summary)
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

    gemma_ir_parser = subparsers.add_parser("gemma-ir-compile", help="Compile a frozen Gemma architecture manifest into a typed structural execution program")
    gemma_ir_parser.add_argument("--manifest", type=Path, required=True)
    gemma_ir_parser.add_argument("--output", type=Path, required=True)
    gemma_ir_parser.set_defaults(handler=command_gemma_ir_compile)

    gemma_ir_verify_parser = subparsers.add_parser("gemma-ir-verify", help="Verify Gemma IR integrity, dependencies, primitive coverage, and manifest binding")
    gemma_ir_verify_parser.add_argument("program", type=Path)
    gemma_ir_verify_parser.add_argument("--manifest", type=Path, required=True)
    gemma_ir_verify_parser.set_defaults(handler=command_gemma_ir_verify)

    gemma_rationale_parser = subparsers.add_parser("gemma-ir-rationale", help="Derive a complete structural backward slice for one declared Gemma IR output")
    gemma_rationale_parser.add_argument("program", type=Path)
    gemma_rationale_parser.add_argument("--output-tensor", default="selected_token_id")
    gemma_rationale_parser.add_argument("--output", type=Path, required=True)
    gemma_rationale_parser.set_defaults(handler=command_gemma_ir_rationale)

    gemma_rationale_verify_parser = subparsers.add_parser("gemma-ir-rationale-verify", help="Recompute and verify a Gemma IR structural rationale slice")
    gemma_rationale_verify_parser.add_argument("program", type=Path)
    gemma_rationale_verify_parser.add_argument("rationale", type=Path)
    gemma_rationale_verify_parser.set_defaults(handler=command_gemma_ir_rationale_verify)

    gemma_ir_predict_parser = subparsers.add_parser("gemma-ir-predict", help="Predict a prompt with the typed IR before comparing it bit-for-bit with pinned Hugging Face eager execution")
    gemma_ir_predict_parser.add_argument("program", type=Path)
    gemma_ir_predict_parser.add_argument("--model-path", type=Path, default=Path(".models/gemma-3-270m-it"))
    gemma_ir_prompt = gemma_ir_predict_parser.add_mutually_exclusive_group(required=True)
    gemma_ir_prompt.add_argument("--prompt")
    gemma_ir_prompt.add_argument("--prompt-file", type=Path)
    gemma_ir_predict_parser.add_argument("--output", type=Path, required=True)
    gemma_ir_predict_parser.set_defaults(handler=command_gemma_ir_predict)

    gemma_ir_execution_verify_parser = subparsers.add_parser("gemma-ir-execution-verify", help="Verify a fixed-input typed-IR prediction certificate and its complete instruction witness")
    gemma_ir_execution_verify_parser.add_argument("program", type=Path)
    gemma_ir_execution_verify_parser.add_argument("certificate", type=Path)
    gemma_ir_execution_verify_parser.add_argument("--model-path", type=Path)
    gemma_ir_execution_verify_parser.set_defaults(handler=command_gemma_ir_execution_verify)

    gemma_ir_summary_parser = subparsers.add_parser("gemma-ir-execution-summary", help="Build a compact result from a verified full typed-IR prediction certificate")
    gemma_ir_summary_parser.add_argument("program", type=Path)
    gemma_ir_summary_parser.add_argument("certificate", type=Path)
    gemma_ir_summary_parser.add_argument("--output", type=Path, required=True)
    gemma_ir_summary_parser.set_defaults(handler=command_gemma_ir_execution_summary)

    gemma_ir_summary_verify_parser = subparsers.add_parser("gemma-ir-execution-summary-verify", help="Verify a compact typed-IR prediction result")
    gemma_ir_summary_verify_parser.add_argument("program", type=Path)
    gemma_ir_summary_verify_parser.add_argument("summary", type=Path)
    gemma_ir_summary_verify_parser.add_argument("--certificate", type=Path, required=True)
    gemma_ir_summary_verify_parser.set_defaults(handler=command_gemma_ir_execution_summary_verify)

    primitive_qualification_parser = subparsers.add_parser("gemma-ir-primitive-qualification", help="Build independently replayable bounded conformance evidence for non-arithmetic IR primitives")
    primitive_qualification_parser.add_argument("--output", type=Path, required=True)
    primitive_qualification_parser.set_defaults(handler=command_gemma_ir_primitive_qualification)

    primitive_qualification_verify_parser = subparsers.add_parser("gemma-ir-primitive-qualification-verify", help="Re-execute and verify non-arithmetic IR primitive conformance evidence")
    primitive_qualification_verify_parser.add_argument("certificate", type=Path)
    primitive_qualification_verify_parser.set_defaults(handler=command_gemma_ir_primitive_qualification_verify)

    bfloat16_semantics_parser = subparsers.add_parser("gemma-bfloat16-semantics", help="Build independent exact-rational bfloat16 semantics and bounded CPU/CUDA conformance evidence")
    bfloat16_semantics_parser.add_argument("--output", type=Path, required=True)
    bfloat16_semantics_parser.set_defaults(handler=command_gemma_bfloat16_semantics)

    bfloat16_semantics_verify_parser = subparsers.add_parser("gemma-bfloat16-semantics-verify", help="Re-execute and verify bfloat16 software-semantics conformance evidence")
    bfloat16_semantics_verify_parser.add_argument("certificate", type=Path)
    bfloat16_semantics_verify_parser.set_defaults(handler=command_gemma_bfloat16_semantics_verify)

    reduction_characterization_parser = subparsers.add_parser("gemma-reduction-characterization", help="Compare bfloat16 dot-product kernels with explicit reduction-order candidates")
    reduction_characterization_parser.add_argument("--output", type=Path, required=True)
    reduction_characterization_parser.set_defaults(handler=command_gemma_reduction_characterization)

    reduction_characterization_verify_parser = subparsers.add_parser("gemma-reduction-characterization-verify", help="Re-execute and verify bounded reduction-order characterization")
    reduction_characterization_verify_parser.add_argument("certificate", type=Path)
    reduction_characterization_verify_parser.set_defaults(handler=command_gemma_reduction_characterization_verify)

    primitive_gate_parser = subparsers.add_parser("gemma-ir-primitive-gate", help="Build a gate separating independently tested indexing primitives from unresolved floating-point semantics")
    primitive_gate_parser.add_argument("program", type=Path)
    primitive_gate_parser.add_argument("certificate", type=Path)
    primitive_gate_parser.add_argument("bfloat16_certificate", type=Path)
    primitive_gate_parser.add_argument("reduction_certificate", type=Path)
    primitive_gate_parser.add_argument("--output", type=Path, required=True)
    primitive_gate_parser.set_defaults(handler=command_gemma_ir_primitive_gate)

    primitive_gate_verify_parser = subparsers.add_parser("gemma-ir-primitive-gate-verify", help="Verify an IR primitive-qualification gate and its negative activation boundaries")
    primitive_gate_verify_parser.add_argument("program", type=Path)
    primitive_gate_verify_parser.add_argument("certificate", type=Path)
    primitive_gate_verify_parser.add_argument("bfloat16_certificate", type=Path)
    primitive_gate_verify_parser.add_argument("reduction_certificate", type=Path)
    primitive_gate_verify_parser.add_argument("gate", type=Path)
    primitive_gate_verify_parser.set_defaults(handler=command_gemma_ir_primitive_gate_verify)

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

    correspondence_parser = subparsers.add_parser("operator-correspondence", help="Map attested kernel symbols to expected Gemma operator-stage patterns")
    correspondence_parser.add_argument("--suite", type=Path, required=True)
    correspondence_parser.add_argument("--output", type=Path, required=True)
    correspondence_parser.set_defaults(handler=command_operator_correspondence)

    correspondence_verify_parser = subparsers.add_parser("operator-correspondence-verify", help="Verify operator-correspondence integrity and semantic boundaries")
    correspondence_verify_parser.add_argument("correspondence", type=Path)
    correspondence_verify_parser.set_defaults(handler=command_operator_correspondence_verify)

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

    cuda_metadata_parser = subparsers.add_parser("cuda-metadata-conformance", help="Validate CUPTI IDs and ctypes layouts against local CUDA headers")
    cuda_metadata_parser.add_argument("--output", type=Path)
    cuda_metadata_parser.set_defaults(handler=command_cuda_metadata)

    cuda_metadata_verify_parser = subparsers.add_parser("cuda-metadata-verify", help="Verify a CUDA metadata conformance certificate")
    cuda_metadata_verify_parser.add_argument("certificate", type=Path)
    cuda_metadata_verify_parser.set_defaults(handler=command_cuda_metadata_verify)

    launch_arguments_parser = subparsers.add_parser("gemma-launch-arguments", help="Capture CUPTI launch parameter commitments for one exact kernel")
    launch_arguments_parser.add_argument("--kernel", required=True)
    launch_arguments_parser.add_argument("--model-path", type=Path, default=Path(".models/gemma-3-270m-it"))
    launch_arguments_prompt = launch_arguments_parser.add_mutually_exclusive_group(required=True)
    launch_arguments_prompt.add_argument("--prompt")
    launch_arguments_prompt.add_argument("--prompt-file", type=Path)
    launch_arguments_parser.add_argument("--module-nvtx-pattern", action="append", required=True)
    launch_arguments_parser.add_argument("--redact", action="store_true")
    launch_arguments_parser.add_argument("--output", type=Path, required=True)
    launch_arguments_parser.set_defaults(handler=command_launch_arguments)

    launch_argument_summary_parser = subparsers.add_parser("launch-argument-summary", help="Summarize CUPTI argument-pointer evidence for qualified modules")
    launch_argument_summary_parser.add_argument("--artifact", action="append", required=True, help="PATH=QUALIFIED_MODULE")
    launch_argument_summary_parser.add_argument("--output", type=Path, required=True)
    launch_argument_summary_parser.set_defaults(handler=command_launch_argument_summary)

    launch_argument_verify_parser = subparsers.add_parser("launch-argument-summary-verify", help="Verify launch-argument summary integrity and boundaries")
    launch_argument_verify_parser.add_argument("summary", type=Path)
    launch_argument_verify_parser.set_defaults(handler=command_launch_argument_summary_verify)

    sass_expressions_parser = subparsers.add_parser("attention-sass-expressions", help="Extract selected symbolic SASS address-expression DAGs")
    sass_expressions_parser.add_argument("--cuobjdump", required=True)
    sass_expressions_parser.add_argument("--cubin", type=Path, required=True)
    sass_expressions_parser.add_argument("--kernel", required=True)
    sass_expressions_parser.add_argument("--sass-memory", type=Path, required=True)
    sass_expressions_parser.add_argument("--sass-semantics", type=Path, required=True)
    sass_expressions_parser.add_argument("--nsight", type=Path, required=True)
    sass_expressions_parser.add_argument("--logical-bounds", type=Path, required=True)
    sass_expressions_parser.add_argument("--output", type=Path, required=True)
    sass_expressions_parser.set_defaults(handler=command_sass_expressions)

    sass_expressions_verify_parser = subparsers.add_parser("attention-sass-expressions-verify", help="Verify selected SASS address-expression DAG integrity")
    sass_expressions_verify_parser.add_argument("certificate", type=Path)
    sass_expressions_verify_parser.set_defaults(handler=command_sass_expressions_verify)

    sass_expression_summary_parser = subparsers.add_parser("attention-sass-expression-summary", help="Build a compact summary from a full SASS expression certificate")
    sass_expression_summary_parser.add_argument("certificate", type=Path)
    sass_expression_summary_parser.add_argument("--output", type=Path, required=True)
    sass_expression_summary_parser.set_defaults(handler=command_sass_expression_summary)

    sass_expression_summary_verify_parser = subparsers.add_parser("attention-sass-expression-summary-verify", help="Verify a compact SASS expression summary")
    sass_expression_summary_verify_parser.add_argument("summary", type=Path)
    sass_expression_summary_verify_parser.set_defaults(handler=command_sass_expression_summary_verify)

    sass_carry_template_parser = subparsers.add_parser("attention-sass-carry-evidence-template", help="Build an incomplete dynamic carry-evidence template")
    sass_carry_template_parser.add_argument("certificate", type=Path)
    sass_carry_template_parser.add_argument("--output", type=Path, required=True)
    sass_carry_template_parser.set_defaults(handler=command_sass_carry_evidence_template)

    sass_carry_verify_parser = subparsers.add_parser("attention-sass-carry-evidence-verify", help="Verify a dynamic carry-evidence bundle without activating semantics")
    sass_carry_verify_parser.add_argument("certificate", type=Path)
    sass_carry_verify_parser.add_argument("bundle", type=Path)
    sass_carry_verify_parser.set_defaults(handler=command_sass_carry_evidence_verify)

    sass_carry_reproduction_parser = subparsers.add_parser("attention-sass-carry-reproduction-verify", help="Compare two distinct-tool carry captures without qualifying semantics")
    sass_carry_reproduction_parser.add_argument("certificate", type=Path)
    sass_carry_reproduction_parser.add_argument("primary", type=Path)
    sass_carry_reproduction_parser.add_argument("replicate", type=Path)
    sass_carry_reproduction_parser.set_defaults(handler=command_sass_carry_reproduction_verify)

    sass_dynamic_summary_parser = subparsers.add_parser("attention-sass-dynamic-summary", help="Build compact commitments from isolated NVBit carry reports")
    sass_dynamic_summary_parser.add_argument("--encoding", type=Path, required=True)
    sass_dynamic_summary_parser.add_argument("--low", type=Path, required=True)
    sass_dynamic_summary_parser.add_argument("--high", type=Path, required=True)
    sass_dynamic_summary_parser.add_argument("--query-pair", type=Path, required=True)
    sass_dynamic_summary_parser.add_argument("--key-pair", type=Path, required=True)
    sass_dynamic_summary_parser.add_argument("--value-triple", type=Path, required=True)
    sass_dynamic_summary_parser.add_argument("--p2r-encoding", type=Path, required=True)
    sass_dynamic_summary_parser.add_argument("--combined", type=Path, required=True)
    sass_dynamic_summary_parser.add_argument("--tool-directory", type=Path, default=Path("tools/nvbit_carry_trace"))
    sass_dynamic_summary_parser.add_argument("--acquisition-tool-sha256", required=True)
    sass_dynamic_summary_parser.add_argument("--output", type=Path, required=True)
    sass_dynamic_summary_parser.set_defaults(handler=command_sass_dynamic_summary)

    sass_dynamic_verify_parser = subparsers.add_parser("attention-sass-dynamic-summary-verify", help="Verify compact isolated NVBit carry commitments and boundaries")
    sass_dynamic_verify_parser.add_argument("summary", type=Path)
    sass_dynamic_verify_parser.add_argument("--tool-directory", type=Path, default=Path("tools/nvbit_carry_trace"))
    sass_dynamic_verify_parser.add_argument("--acquisition-tool-sha256", required=True)
    sass_dynamic_verify_parser.set_defaults(handler=command_sass_dynamic_summary_verify)

    sass_output_reachability_parser = subparsers.add_parser("attention-sass-output-reachability-summary", help="Build compact non-writing output-pair reachability commitments")
    sass_output_reachability_parser.add_argument("--report", type=Path, required=True)
    sass_output_reachability_parser.add_argument("--acquisition-tool-sha256", required=True)
    sass_output_reachability_parser.add_argument("--output", type=Path, required=True)
    sass_output_reachability_parser.set_defaults(handler=command_sass_output_reachability_summary)

    sass_output_reachability_verify_parser = subparsers.add_parser("attention-sass-output-reachability-verify", help="Verify compact output-pair reachability boundaries")
    sass_output_reachability_verify_parser.add_argument("summary", type=Path)
    sass_output_reachability_verify_parser.add_argument("--acquisition-tool-sha256", required=True)
    sass_output_reachability_verify_parser.set_defaults(handler=command_sass_output_reachability_verify)

    attention_bounds_parser = subparsers.add_parser("attention-logical-bounds", help="Prove logical Q/K/V/output indices remain within retained storage")
    attention_bounds_parser.add_argument("--artifact", type=Path, required=True)
    attention_bounds_parser.add_argument("--attention", type=Path, required=True)
    attention_bounds_parser.add_argument("--redact", action="store_true")
    attention_bounds_parser.add_argument("--output", type=Path, required=True)
    attention_bounds_parser.set_defaults(handler=command_attention_bounds)

    attention_bounds_verify_parser = subparsers.add_parser("attention-logical-bounds-verify", help="Re-execute a logical attention storage-bounds certificate")
    attention_bounds_verify_parser.add_argument("certificate", type=Path)
    attention_bounds_verify_parser.set_defaults(handler=command_attention_bounds_verify)

    sass_memory_parser = subparsers.add_parser("attention-sass-memory", help="Trace syntactic SASS memory-address provenance to attention parameters")
    sass_memory_parser.add_argument("--cuobjdump", required=True)
    sass_memory_parser.add_argument("--cubin", type=Path, required=True)
    sass_memory_parser.add_argument("--kernel", required=True)
    sass_memory_parser.add_argument("--nsight", type=Path, required=True)
    sass_memory_parser.add_argument("--attention", type=Path, required=True)
    sass_memory_parser.add_argument("--sass-semantics", type=Path, required=True)
    sass_memory_parser.add_argument("--output", type=Path, required=True)
    sass_memory_parser.set_defaults(handler=command_sass_memory)

    sass_memory_verify_parser = subparsers.add_parser("attention-sass-memory-verify", help="Replay a SASS memory-address provenance certificate")
    sass_memory_verify_parser.add_argument("certificate", type=Path)
    sass_memory_verify_parser.add_argument("--cuobjdump", required=True)
    sass_memory_verify_parser.add_argument("--cubin", type=Path, required=True)
    sass_memory_verify_parser.add_argument("--nsight", type=Path, required=True)
    sass_memory_verify_parser.add_argument("--attention", type=Path, required=True)
    sass_memory_verify_parser.add_argument("--sass-semantics", type=Path, required=True)
    sass_memory_verify_parser.set_defaults(handler=command_sass_memory_verify)

    attention_parameters_parser = subparsers.add_parser("attention-parameters", help="Validate decoded fused-attention parameters against Gemma expectations")
    attention_parameters_parser.add_argument("--artifact", type=Path, required=True)
    attention_parameters_parser.add_argument("--signatures", type=Path, required=True)
    attention_parameters_parser.add_argument("--output", type=Path, required=True)
    attention_parameters_parser.set_defaults(handler=command_attention_parameters)

    attention_parameters_verify_parser = subparsers.add_parser("attention-parameters-verify", help="Verify a decoded attention-parameter certificate")
    attention_parameters_verify_parser.add_argument("certificate", type=Path)
    attention_parameters_verify_parser.set_defaults(handler=command_attention_parameters_verify)

    kernel_signature_parser = subparsers.add_parser("kernel-signatures", help="Derive partial typed signatures from installed headers and launch evidence")
    kernel_signature_parser.add_argument("--summary", type=Path, required=True)
    kernel_signature_parser.add_argument("--output", type=Path, required=True)
    kernel_signature_parser.set_defaults(handler=command_kernel_signatures)

    kernel_signature_verify_parser = subparsers.add_parser("kernel-signatures-verify", help="Verify a partial kernel-signature certificate")
    kernel_signature_verify_parser.add_argument("certificate", type=Path)
    kernel_signature_verify_parser.set_defaults(handler=command_kernel_signatures_verify)

    nsight_target_parser = subparsers.add_parser("gemma-nsight-target", help="Run one NVTX-bounded Gemma forward and emit an execution binding")
    nsight_target_parser.add_argument("--model-path", type=Path, default=Path(".models/gemma-3-270m-it"))
    nsight_target_prompt = nsight_target_parser.add_mutually_exclusive_group(required=True)
    nsight_target_prompt.add_argument("--prompt")
    nsight_target_prompt.add_argument("--prompt-file", type=Path)
    nsight_target_parser.add_argument("--module-nvtx-pattern", action="append", default=[])
    nsight_target_parser.add_argument("--output", type=Path, required=True)
    nsight_target_parser.set_defaults(handler=command_nsight_target)

    nsight_capture_parser = subparsers.add_parser("nsight-capture", help="Record and execute one exact NVTX-bounded Nsight kernel request")
    nsight_capture_parser.add_argument("--kernel", required=True)
    nsight_capture_parser.add_argument("--model-path", type=Path, default=Path(".models/gemma-3-270m-it"))
    nsight_capture_prompt = nsight_capture_parser.add_mutually_exclusive_group(required=True)
    nsight_capture_prompt.add_argument("--prompt")
    nsight_capture_prompt.add_argument("--prompt-file", type=Path)
    nsight_capture_parser.add_argument("--report-base", type=Path, required=True)
    nsight_capture_parser.add_argument("--binding", type=Path, required=True)
    nsight_capture_parser.add_argument("--request", type=Path, required=True)
    nsight_capture_parser.add_argument("--module-nvtx-pattern", action="append", default=[])
    nsight_capture_parser.set_defaults(handler=command_nsight_capture)

    nsight_certificate_parser = subparsers.add_parser("nsight-launch-certificate", help="Bind an Nsight launch SASS view to Gemma execution and CUPTI cubin evidence")
    nsight_certificate_parser.add_argument("--report", type=Path, required=True)
    nsight_certificate_parser.add_argument("--binding", type=Path, required=True)
    nsight_certificate_parser.add_argument("--cuda-summary", type=Path, required=True)
    nsight_certificate_parser.add_argument("--cupti-report", type=Path, required=True)
    nsight_certificate_parser.add_argument("--cupti-artifact-directory", type=Path, required=True)
    nsight_certificate_parser.add_argument("--capture-request", type=Path)
    nsight_certificate_parser.add_argument("--output", type=Path, required=True)
    nsight_certificate_parser.set_defaults(handler=command_nsight_launch_certificate)

    nsight_verify_parser = subparsers.add_parser("nsight-launch-verify", help="Verify an Nsight launch certificate's integrity and internal claims")
    nsight_verify_parser.add_argument("certificate", type=Path)
    nsight_verify_parser.set_defaults(handler=command_nsight_launch_verify)

    nsight_suite_parser = subparsers.add_parser("nsight-kernel-suite", help="Build launch certificates for every recorded distinct forward kernel")
    nsight_suite_parser.add_argument("--report-directory", type=Path, required=True)
    nsight_suite_parser.add_argument("--cuda-manifest", type=Path, required=True)
    nsight_suite_parser.add_argument("--cupti-report", type=Path, required=True)
    nsight_suite_parser.add_argument("--cupti-artifact-directory", type=Path, required=True)
    nsight_suite_parser.add_argument("--redact", action="store_true", help="Redact report, cubin, SASS, and certificate hashes")
    nsight_suite_parser.add_argument("--output", type=Path, required=True)
    nsight_suite_parser.set_defaults(handler=command_nsight_kernel_suite)

    nsight_suite_verify_parser = subparsers.add_parser("nsight-kernel-suite-verify", help="Verify aggregate and optional per-certificate Nsight suite claims")
    nsight_suite_verify_parser.add_argument("suite", type=Path)
    nsight_suite_verify_parser.add_argument("--certificate-directory", type=Path)
    nsight_suite_verify_parser.add_argument("--cupti-report", type=Path)
    nsight_suite_verify_parser.set_defaults(handler=command_nsight_kernel_suite_verify)

    module_certificate_parser = subparsers.add_parser("module-invocation-certificate", help="Bind an Nsight launch to a qualified module NVTX stack and tensor commitments")
    module_certificate_parser.add_argument("--report", type=Path, required=True)
    module_certificate_parser.add_argument("--binding", type=Path, required=True)
    module_certificate_parser.add_argument("--launch-certificate", type=Path, required=True)
    module_certificate_parser.add_argument("--expected-innermost-module", required=True)
    module_certificate_parser.add_argument("--output", type=Path, required=True)
    module_certificate_parser.set_defaults(handler=command_module_invocation_certificate)

    module_verify_parser = subparsers.add_parser("module-invocation-verify", help="Verify a module invocation certificate")
    module_verify_parser.add_argument("certificate", type=Path)
    module_verify_parser.set_defaults(handler=command_module_invocation_verify)

    module_summary_parser = subparsers.add_parser("module-invocation-summary", help="Build a compact summary from module invocation certificates")
    module_summary_parser.add_argument("certificates", type=Path, nargs="+")
    module_summary_parser.add_argument("--output", type=Path, required=True)
    module_summary_parser.set_defaults(handler=command_module_invocation_summary)

    module_summary_verify_parser = subparsers.add_parser("module-invocation-summary-verify", help="Verify a module invocation summary")
    module_summary_verify_parser.add_argument("summary", type=Path)
    module_summary_verify_parser.set_defaults(handler=command_module_invocation_summary_verify)

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
