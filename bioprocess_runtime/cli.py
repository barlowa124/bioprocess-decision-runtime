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


def command_k2048_probes(args: argparse.Namespace) -> int:
    from .gemma_k2048_probes import build_plan, acquire_probes, verify_probes, probe_summary
    from .serialization import canonical_json

    outputs = [args.output, args.bundle] if args.operation == "plan" else [args.output] + ([args.summary] if args.summary else []) if args.operation == "run" else []
    if len({path.resolve() for path in outputs}) != len(outputs) or any(path.exists() for path in outputs):
        raise ValueError("K2048 probe output paths must be new and distinct")
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    binding, product = load(args.binding), load(args.product_summary)

    def write_new(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")

    if args.operation == "plan":
        plan, bundle = build_plan(binding, product, args.workers)
        write_new(args.bundle, bundle)
        write_new(args.output, plan)
        print(json.dumps({"plan_sha256": plan["plan_sha256"], "case_count": plan["case_count"], "candidate_count": plan["candidate_count"], "candidate_pairs_separated": plan["candidate_pairs_separated"], "equivalence_class_count": len(plan["prediction_equivalence_classes"])}, indent=2))
        return 0
    plan, bundle = load(args.plan), load(args.bundle)
    if args.operation == "run":
        report = acquire_probes(binding, product, plan, bundle)
        write_new(args.output, report)
        summary = probe_summary(plan, report)
        if args.summary:
            write_new(args.summary, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if report["survivors_supported_in_declared_scope"] else 1
    report = load(args.report)
    result = verify_probes(binding, product, plan, bundle, report)
    if args.summary and result["valid"]:
        result["summary_matches_source"] = canonical_json(load(args.summary)) == canonical_json(probe_summary(plan, report))
        result["valid"] = result["summary_matches_source"]
    if args.reexecute and result["valid"]:
        replay_plan, replay_bundle = build_plan(binding, product, args.workers)
        same = canonical_json(replay_plan) == canonical_json(plan) and canonical_json(replay_bundle) == canonical_json(bundle)
        replay = acquire_probes(binding, product, plan, bundle) if same else None
        exact = canonical_json(replay) == canonical_json(report)
        result.update({"valid": same and exact, "mode": "full_k2048_prediction_selection_and_cuda_replay", "predictions_recomputed_exact": same, "reexecution_exact": exact})
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


def command_gelu_table(args: argparse.Namespace) -> int:
    from .gemma_gelu_lookup import build_gelu_plan, acquire_gelu_table, load_gelu_table, acquire_gelu_layouts, check_gelu_layouts
    from .serialization import canonical_json

    outputs = [args.output, args.table] if args.operation == "run" else [args.output] if args.operation != "verify" else [args.replay_output] if args.reexecute else []
    if any(path is None for path in outputs) or len({path.resolve() for path in outputs}) != len(outputs) or any(path.exists() for path in outputs):
        raise ValueError("GELU outputs must be new and distinct; replay requires --replay-output")
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))

    def write_new(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")

    if args.operation == "plan":
        plan = build_gelu_plan(load(args.source_summary))
        write_new(args.output, plan)
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    plan = load(args.plan)
    if args.operation == "run":
        args.table.parent.mkdir(parents=True, exist_ok=True)
        manifest = acquire_gelu_table(plan, args.table, args.audit_dir)
        write_new(args.output, manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0 if manifest["base_table_usable"] else 1
    manifest = load(args.manifest)
    if args.operation == "layouts":
        report = acquire_gelu_layouts(plan, manifest, args.table, args.audit_dir)
        write_new(args.output, report)
        print(json.dumps({key: value for key, value in report.items() if key not in ("windows", "vector_replay")}, indent=2))
        return 0 if report["declared_layouts_match"] else 1
    report = load(args.layout_report)
    table = load_gelu_table(plan, manifest, args.table, args.audit_dir)
    check_gelu_layouts(plan, manifest, table, report, args.audit_dir)
    result = {"valid": True, "mode": "integrity_only", "base_table_usable": manifest["base_table_usable"], "declared_layouts_match": report["declared_layouts_match"]}
    if args.reexecute:
        replay = acquire_gelu_layouts(plan, manifest, args.table, args.audit_dir)
        write_new(args.replay_output, replay)
        exact = canonical_json(replay) == canonical_json(report)
        result.update({"valid": exact, "mode": "native_finite_domain_and_layout_replay", "reexecution_exact": exact})
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


def _load_dense_k128_sources(args: argparse.Namespace) -> Any:
    from .gemma_k128_dense import DenseSources
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    return DenseSources(load(args.source_summary), load(args.probe_plan), load(args.probe_bundle), load(args.probe_report), load(args.model_plan), load(args.model_bundle))


def command_dense_k128(args: argparse.Namespace) -> int:
    from .gemma_k128_dense import build_dense_plan, acquire_dense, verify_dense, dense_summary
    from .serialization import canonical_json

    outputs = [args.output, args.bundle] if args.operation == "plan" else [args.output] + ([args.summary] if args.summary else []) if args.operation == "run" else []
    if len({path.resolve() for path in outputs}) != len(outputs) or any(path.exists() for path in outputs):
        raise ValueError("Dense K128 output paths must be new and distinct")
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    sources = _load_dense_k128_sources(args)

    def write_new(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")

    if args.operation == "plan":
        plan, bundle = build_dense_plan(sources)
        write_new(args.bundle, bundle)
        write_new(args.output, plan)
        print(json.dumps({"plan_sha256": plan["plan_sha256"], "value_count": plan["value_count"], "complete_matrix_compared": True, "vectors_disjoint_from_declared_sources": True}, indent=2))
        return 0
    plan, bundle = load(args.plan), load(args.bundle)
    if args.operation == "run":
        report = acquire_dense(sources, plan, bundle)
        write_new(args.output, report)
        summary = dense_summary(plan, report)
        if args.summary:
            write_new(args.summary, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if report["candidate_passes_dense_holdout"] else 1
    report = load(args.report)
    result = verify_dense(sources, plan, bundle, report)
    if args.summary and result["valid"]:
        result["summary_matches_source"] = canonical_json(load(args.summary)) == canonical_json(dense_summary(plan, report))
        result["valid"] = result["summary_matches_source"]
    if args.reexecute and result["valid"]:
        replay_plan, replay_bundle = build_dense_plan(sources)
        same = canonical_json(replay_plan) == canonical_json(plan) and canonical_json(replay_bundle) == canonical_json(bundle)
        replay = acquire_dense(sources, plan, bundle) if same else None
        exact = canonical_json(replay) == canonical_json(report)
        result.update({"valid": same and exact, "mode": "dense_prediction_regeneration_and_cuda_replay", "predictions_recomputed_exact": same, "reexecution_exact": exact})
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


def command_k1024_probes(args: argparse.Namespace) -> int:
    from .gemma_k1024_probes import build_probe_plan, acquire_probes, verify_probes, probe_summary
    from .serialization import canonical_json

    outputs = [args.output, args.bundle] if args.operation == "plan" else [args.output] + ([args.summary] if args.summary else []) if args.operation == "run" else []
    if len({path.resolve() for path in outputs}) != len(outputs) or any(path.exists() for path in outputs):
        raise ValueError("Controlled K1024 output paths must be new and distinct")
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    source = load(args.source_summary)

    def write_new(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")

    if args.operation == "plan":
        plan, bundle = build_probe_plan(source)
        write_new(args.bundle, bundle)
        write_new(args.output, plan)
        print(json.dumps({"plan_sha256": plan["plan_sha256"], "case_count": plan["case_count"], "candidate_count": plan["candidate_count"], "candidate_pairs_separated": plan["candidate_pairs_separated"], "equivalence_class_count": len(plan["prediction_equivalence_classes"])}, indent=2))
        return 0
    plan, bundle = load(args.plan), load(args.bundle)
    if args.operation == "run":
        report = acquire_probes(source, plan, bundle)
        write_new(args.output, report)
        summary = probe_summary(plan, report)
        if args.summary:
            write_new(args.summary, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if report["survivors_supported_in_declared_scope"] else 1
    report = load(args.report)
    result = verify_probes(source, plan, bundle, report)
    if args.summary and result["valid"]:
        result["summary_matches_source"] = canonical_json(load(args.summary)) == canonical_json(probe_summary(plan, report))
        result["valid"] = result["summary_matches_source"]
    if args.reexecute and result["valid"]:
        replay_plan, replay_bundle = build_probe_plan(source)
        same = canonical_json(replay_plan) == canonical_json(plan) and canonical_json(replay_bundle) == canonical_json(bundle)
        replay = acquire_probes(source, plan, bundle) if same else None
        exact = canonical_json(replay) == canonical_json(report)
        result.update({"valid": same and exact, "mode": "full_candidate_selection_prediction_and_cuda_replay", "predictions_recomputed_exact": same, "reexecution_exact": exact})
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


def command_exp_lookup(args: argparse.Namespace) -> int:
    from .gemma_exp_lookup import build_exp_plan, acquire_exp_table, load_exp_table, replay_exp_table
    import os
    import tempfile

    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    outputs = [args.output] if args.operation == "plan" else [args.output, args.table, args.journal] if args.operation == "run" else [args.replay_output] if args.reexecute else []
    if any(path is None for path in outputs) or len({path.resolve() for path in outputs}) != len(outputs) or any(path.exists() for path in outputs):
        raise ValueError("Exponential outputs must be new and distinct; replay requires --replay-output")

    def write_new(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")

    if args.operation == "plan":
        plan = build_exp_plan(load(args.source_summary))
        write_new(args.output, plan)
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    plan = load(args.plan)
    if args.operation == "run":
        write_new(args.journal, {"plan_sha256": plan["plan_sha256"], "complete": False, "completed_chunks": 0})
        args.table.parent.mkdir(parents=True, exist_ok=True)

        def checkpoint(payload: dict[str, Any]) -> None:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.journal.parent, prefix=args.journal.name + ".", suffix=".tmp", delete=False) as stream:
                json.dump(payload, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
                temporary = stream.name
            os.replace(temporary, args.journal)

        manifest = acquire_exp_table(plan, args.table, args.audit_dir, checkpoint)
        write_new(args.output, manifest)
        print(json.dumps({key: value for key, value in manifest.items() if key not in ("records", "special_observations")}, indent=2, sort_keys=True))
        return 0 if manifest["table_usable"] else 1
    manifest = load(args.manifest)
    load_exp_table(plan, manifest, args.table, args.audit_dir)
    result = {"valid": True, "mode": "integrity_only", "table_usable": manifest["table_usable"], "covered_input_encoding_count": manifest["covered_input_encoding_count"]}
    if args.reexecute:
        replay = replay_exp_table(plan, manifest, args.table, args.audit_dir)
        write_new(args.replay_output, replay)
        result.update({"valid": replay["reexecution_exact"], "mode": "exhaustive_native_exp_replay", "reexecution_exact": replay["reexecution_exact"], "replay_sha256": replay["replay_sha256"]})
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


def _load_softmax_sources(args: argparse.Namespace) -> Any:
    from .gemma_attention_scores import ScoreSources
    from .gemma_softmax_slice import SoftmaxSources
    from .gemma_rsqrt_lookup import CheckedRsqrtLookup

    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    lookup = CheckedRsqrtLookup(load(args.rsqrt_table_plan), load(args.rsqrt_table_manifest), args.rsqrt_table,
                                load(args.rsqrt_domain_plan), load(args.rsqrt_domain_report), args.rsqrt_audit_dir)
    scores = ScoreSources(load(args.program), load(args.rotary_plan), load(args.rotary_bundle), load(args.rotary_report), lookup,
                          load(args.table_plan), load(args.table_manifest), load(args.table_bundle))
    return SoftmaxSources(scores, load(args.score_plan), load(args.score_bundle), load(args.score_report))


def _load_output_sources(args: argparse.Namespace) -> Any:
    from .gemma_exp_lookup import CheckedExpLookup
    from .gemma_attention_output import OutputSources

    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    lookup = CheckedExpLookup(load(args.exp_plan), load(args.exp_manifest), args.exp_table, args.exp_audit_dir)
    return OutputSources(_load_softmax_sources(args), load(args.original_plan), load(args.original_bundle), load(args.original_report), lookup,
                         load(args.lookup_plan), load(args.lookup_bundle), load(args.lookup_report))


def _load_survivor_context(args: argparse.Namespace) -> tuple[Any, ...]:
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    return (_load_output_sources(args), load(args.base_plan), load(args.base_bundle), load(args.base_report),
            load(args.probe_source), load(args.probe_plan), load(args.probe_bundle), load(args.probe_report))


def _load_post_attention_sources(args: argparse.Namespace) -> Any:
    from .gemma_post_attention import PostAttentionSources
    from .gemma_k128_dense import DenseSources
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    context = _load_survivor_context(args)
    dense = DenseSources(context[4], context[5], context[6], context[7], context[1], context[2])
    return PostAttentionSources(context, load(args.survivor_plan), load(args.survivor_bundle), load(args.survivor_report), dense,
                                load(args.dense_plan), load(args.dense_bundle), load(args.dense_report))


def _load_mlp_entry_sources(args: argparse.Namespace) -> Any:
    from .gemma_mlp_entry import MlpEntrySources
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    return MlpEntrySources(_load_post_attention_sources(args), load(args.post_plan), load(args.post_bundle), load(args.post_report))


def _load_product_sources(args: argparse.Namespace) -> Any:
    from .gemma_mlp_product import ProductSources
    from .gemma_gelu_lookup import CheckedGeluLookup
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    gelu = CheckedGeluLookup(load(args.gelu_plan), load(args.gelu_manifest), args.gelu_table, load(args.gelu_layouts), args.gelu_audit_dir)
    return ProductSources(_load_mlp_entry_sources(args), load(args.entry_plan), load(args.entry_bundle), load(args.entry_report), gelu)


def _load_down_sources(args: argparse.Namespace) -> Any:
    from .gemma_mlp_down import DownSources
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    prefix_args = argparse.Namespace(**vars(args))
    for name in ("plan", "bundle", "report"):
        setattr(prefix_args, "probe_" + name, getattr(args, "prefix_probe_" + name))
    return DownSources(_load_product_sources(prefix_args), load(args.product_plan), load(args.product_bundle), load(args.product_report),
                       load(args.probe_binding), load(args.probe_plan), load(args.probe_bundle), load(args.probe_report))


def _load_dense_k2048_sources(args: argparse.Namespace) -> Any:
    from .gemma_k2048_dense import DenseSources
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    return DenseSources(_load_down_sources(args), load(args.model_down_plan), load(args.model_down_bundle), load(args.model_down_report))


def _load_post_feedforward_sources(args: argparse.Namespace) -> Any:
    from .gemma_post_feedforward import PostFeedforwardSources
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    return PostFeedforwardSources(_load_dense_k2048_sources(args), load(args.k2048_dense_plan), load(args.k2048_dense_bundle), load(args.k2048_dense_report))


def _load_holdout_baseline(args: argparse.Namespace) -> Any:
    from .gemma_first_layer import FirstLayerSources
    from .gemma_first_layer_holdout import BaselineSources
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    source = FirstLayerSources(_load_post_feedforward_sources(args), load(args.post_feedforward_plan), load(args.post_feedforward_bundle), load(args.post_feedforward_report))
    return BaselineSources(source, load(args.baseline_plan), load(args.baseline_bundle), load(args.baseline_report))


def _holdout_model(model_path: Path) -> Any:
    import torch
    from transformers import AutoModelForCausalLM
    if not torch.cuda.is_available():
        raise RuntimeError("Holdout checkpoint binding requires the declared CUDA runtime")
    return AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False,
                                              torch_dtype=torch.bfloat16, attn_implementation="eager").to("cuda").eval()


def command_first_layer_holdout(args: argparse.Namespace) -> int:
    from . import gemma_first_layer_holdout as holdout
    from .serialization import canonical_json
    operation = args.operation
    outputs = [args.output] if operation == "protocol" else [args.output, args.bundle] if operation == "plan" else [args.output] + ([args.summary] if args.summary else []) if operation == "run" else []
    excluded = {"output"}
    if operation in ("protocol", "plan"):
        excluded.update(("bundle", "plan", "summary"))
    if operation == "protocol":
        excluded.add("protocol")
    if operation == "run":
        excluded.add("summary")
    inputs = [value.resolve() for key, value in vars(args).items() if isinstance(value, Path) and key not in excluded]
    if len({path.resolve() for path in outputs}) != len(outputs) or any(path.exists() or path.resolve() in inputs or any(root.is_dir() and root in path.resolve().parents for root in inputs) for path in outputs):
        raise ValueError("Holdout artifacts require new distinct paths outside all frozen inputs")
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    baseline = _load_holdout_baseline(args)
    if operation == "protocol":
        protocol = holdout.build_holdout_protocol(baseline, args.model_path)
        holdout.write_new(args.output, protocol)
        print(json.dumps({"protocol_sha256": protocol["protocol_sha256"], "cases": protocol["cases"]}, indent=2))
        return 0
    protocol = load(args.protocol)
    holdout.require_frozen(args.protocol, protocol)
    if operation == "plan":
        holdout._protocol_check(baseline, protocol, args.model_path)
        plan, bundle = holdout.build_holdout_plan(baseline, _holdout_model(args.model_path), protocol, args.model_path, args.protocol, args.workers)
        holdout.write_new(args.bundle, bundle)
        holdout.write_new(args.output, plan)
        print(json.dumps({"plan_sha256": plan["plan_sha256"], "prediction_complete": plan["prediction_complete"],
                          "case_failures": [case["prediction_failure"] for case in plan["cases"]]}, indent=2))
        return 0 if plan["prediction_complete"] else 1
    plan, bundle = load(args.plan), load(args.bundle)
    if operation == "run":
        holdout.check_holdout_plan(baseline, protocol, plan, bundle, args.model_path)
        if not plan["prediction_complete"]:
            report = holdout.holdout_report(protocol, plan, bundle, [])
        else:
            report = holdout.acquire_holdout(baseline, _holdout_model(args.model_path), protocol, plan, bundle, args.model_path, args.protocol, args.plan, args.bundle)
        holdout.write_new(args.output, report)
        summary = holdout.holdout_summary(plan, report)
        if args.summary:
            holdout.write_new(args.summary, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if report["holdout_matches"] else 1
    report = load(args.report)
    result = holdout.verify_holdout(baseline, protocol, plan, bundle, report, args.model_path)
    if args.summary and result["valid"]:
        result["summary_matches_source"] = canonical_json(load(args.summary)) == canonical_json(holdout.holdout_summary(plan, report))
        result["valid"] = result["summary_matches_source"]
    if args.reexecute and result["valid"]:
        result = holdout.reexecute_holdout(baseline, _holdout_model(args.model_path), protocol, plan, bundle, report, args.model_path, args.protocol, args.plan, args.bundle, args.workers)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] and result.get("holdout_matches") else 1


def _load_two_layers_sources(args: argparse.Namespace) -> Any:
    from .gemma_second_layer import SecondLayerSources
    from .gemma_two_layers import TwoLayerSources
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    second_sources = SecondLayerSources(_load_holdout_baseline(args), load(args.protocol), load(args.holdout_plan), load(args.holdout_bundle), load(args.holdout_report))
    return TwoLayerSources(second_sources, load(args.second_layer_plan), load(args.second_layer_bundle), load(args.second_layer_report))


def _load_third_entry_sources(args: argparse.Namespace) -> Any:
    from . import gemma_third_layer_entry as entry
    from .gemma_two_layers_holdout import BaselineSources
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    baseline = BaselineSources(_load_two_layers_sources(args), load(args.two_layer_baseline_plan), load(args.two_layer_baseline_bundle), load(args.two_layer_baseline_report))
    return entry.EntrySources(baseline, load(args.holdout_protocol), load(args.two_holdout_plan), load(args.two_holdout_bundle), load(args.two_holdout_report))


def _load_third_score_sources(args: argparse.Namespace) -> Any:
    from .gemma_third_layer_scores import ScoreSources
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    return ScoreSources(_load_third_entry_sources(args), load(args.third_entry_plan), load(args.third_entry_bundle), load(args.third_entry_report))


def command_third_layer_scores(args: argparse.Namespace) -> int:
    from . import gemma_third_layer_scores as scores
    from .gemma_first_layer_holdout import write_new
    from .serialization import canonical_json
    outputs = [args.output, args.bundle] if args.operation == "plan" else [args.output, args.summary] if args.operation == "run" else []
    outputs = [path for path in outputs if path is not None]
    excluded = {"output", "summary"} | ({"plan", "bundle"} if args.operation == "plan" else set())
    inputs = [value.resolve() for key, value in vars(args).items() if isinstance(value, Path) and key not in excluded]
    if len({path.resolve() for path in outputs}) != len(outputs) or any(path.exists() or path.resolve() in inputs or any(root.is_dir() and root in path.resolve().parents for root in inputs) for path in outputs):
        raise ValueError("Layer-2 score outputs must be new and outside source evidence")
    repository = Path(__file__).resolve().parents[1]
    raw = [args.bundle] if args.operation == "plan" else [args.output] if args.operation == "run" else []
    if any(repository in path.resolve().parents and repository / "artifacts" not in path.resolve().parents for path in raw):
        raise ValueError("Tensor-rich score artifacts must remain under ignored artifacts/")
    if args.operation == "verify" and args.summary is not None and not args.summary.is_file():
        raise ValueError("Requested score summary is missing")
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    sources = _load_third_score_sources(args)
    if args.operation == "plan":
        plan, bundle = scores.build_score_plan(sources, _holdout_model(args.model_path), args.model_path, args.workers)
        write_new(args.bundle, bundle)
        write_new(args.output, plan)
        print(json.dumps({"plan_sha256": plan["plan_sha256"], "coverage": plan["coverage"]}, indent=2))
        return 0
    plan, bundle = load(args.plan), load(args.bundle)
    if args.operation == "run":
        report = scores.acquire_scores(sources, _holdout_model(args.model_path), plan, bundle, args.model_path, args.plan, args.bundle)
        write_new(args.output, report)
        if args.summary:
            write_new(args.summary, scores.score_summary(plan, report))
        print(json.dumps(scores.score_summary(plan, report), indent=2, sort_keys=True))
        return 0 if report["rotary_scores_match"] else 1
    report = load(args.report)
    result = scores.verify_scores(sources, plan, bundle, report, args.model_path)
    if args.summary and result["valid"]:
        result["valid"] = canonical_json(load(args.summary)) == canonical_json(scores.score_summary(plan, report))
    if args.reexecute and result["valid"]:
        result = scores.replay_scores(sources, _holdout_model(args.model_path), plan, bundle, report, args.model_path, args.plan, args.bundle, args.workers)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] and result.get("rotary_scores_match") else 1


def command_third_layer_entry(args: argparse.Namespace) -> int:
    from . import gemma_third_layer_entry as entry
    from .gemma_first_layer_holdout import write_new
    from .serialization import canonical_json
    outputs = [args.output, args.bundle] if args.operation == "plan" else [args.output, args.summary] if args.operation == "run" else []
    outputs = [path for path in outputs if path is not None]
    excluded = {"output", "summary"} | ({"plan", "bundle"} if args.operation == "plan" else set())
    inputs = [value.resolve() for key, value in vars(args).items() if isinstance(value, Path) and key not in excluded]
    if len({path.resolve() for path in outputs}) != len(outputs) or any(path.exists() or path.resolve() in inputs or any(root.is_dir() and root in path.resolve().parents for root in inputs) for path in outputs):
        raise ValueError("Layer-2 entry outputs must be new, distinct, and outside source evidence")
    repository = Path(__file__).resolve().parents[1]
    raw = [args.bundle] if args.operation == "plan" else [args.output] if args.operation == "run" else []
    if any(repository in path.resolve().parents and repository / "artifacts" not in path.resolve().parents for path in raw):
        raise ValueError("Tensor-rich entry artifacts must remain under ignored artifacts/")
    if args.operation == "verify" and args.summary is not None and not args.summary.is_file():
        raise ValueError("Requested entry summary is missing")
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    sources = _load_third_entry_sources(args)
    if args.operation == "plan":
        plan, bundle = entry.build_entry_plan(sources, _holdout_model(args.model_path), args.model_path, args.workers)
        write_new(args.bundle, bundle)
        write_new(args.output, plan)
        print(json.dumps({"plan_sha256": plan["plan_sha256"], "coverage": plan["coverage"]}, indent=2))
        return 0
    plan, bundle = load(args.plan), load(args.bundle)
    if args.operation == "run":
        report = entry.acquire_entry(sources, _holdout_model(args.model_path), plan, bundle, args.model_path, args.plan, args.bundle)
        write_new(args.output, report)
        if args.summary:
            write_new(args.summary, entry.entry_summary(plan, report))
        print(json.dumps(entry.entry_summary(plan, report), indent=2, sort_keys=True))
        return 0 if report["entry_matches"] else 1
    report = load(args.report)
    result = entry.verify_entry(sources, plan, bundle, report, args.model_path)
    if args.summary and result["valid"]:
        result["valid"] = canonical_json(load(args.summary)) == canonical_json(entry.entry_summary(plan, report))
    if args.reexecute and result["valid"]:
        result = entry.replay_entry(sources, _holdout_model(args.model_path), plan, bundle, report, args.model_path, args.plan, args.bundle, args.workers)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] and result.get("entry_matches") else 1


def command_two_layers_holdout(args: argparse.Namespace) -> int:
    from . import gemma_two_layers_holdout as holdout
    from . import gemma_two_layers as two
    from .serialization import canonical_json

    operation = args.operation
    if operation == "verify" and args.summary is not None and not args.summary.is_file():
        raise ValueError("Requested two-layer holdout summary is missing")
    outputs = [args.output] if operation == "protocol" else [args.output, args.bundle] if operation == "plan" else [args.output] + ([args.summary] if args.summary else []) if operation == "run" else []
    excluded = {"output"}
    if operation == "protocol":
        excluded.update(("holdout_protocol", "plan", "bundle", "summary"))
    elif operation == "plan":
        excluded.update(("plan", "bundle", "summary"))
    elif operation == "run":
        excluded.add("summary")
    inputs = [value.resolve() for key, value in vars(args).items() if isinstance(value, Path) and key not in excluded]
    if len({path.resolve() for path in outputs}) != len(outputs) or any(path.exists() or path.resolve() in inputs or args.model_path.resolve() in path.resolve().parents or any(root.is_dir() and root in path.resolve().parents for root in inputs) for path in outputs):
        raise ValueError("Two-layer holdout artifacts require new distinct paths outside all inputs and model directory")
    repository = Path(__file__).resolve().parents[1]
    raw_outputs = [args.bundle] if operation == "plan" else [args.output] if operation == "run" else []
    if any(repository in path.resolve().parents and repository / "artifacts" not in path.resolve().parents for path in raw_outputs):
        raise ValueError("Tensor-rich holdout bundles/reports inside the repository must remain under Git-ignored artifacts/")
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    frozen_inputs = tuple(path for path in inputs if path.is_file())
    files = two._file_guard(frozen_inputs)

    def load_model() -> Any:
        if files != two._file_guard(frozen_inputs):
            raise ValueError("Frozen inputs changed before model loading")
        model = _holdout_model(args.model_path)
        if files != two._file_guard(frozen_inputs):
            raise ValueError("Frozen inputs changed during model loading; no prediction or native calls allowed")
        return model

    sources = holdout.BaselineSources(_load_two_layers_sources(args), load(args.two_layer_baseline_plan), load(args.two_layer_baseline_bundle), load(args.two_layer_baseline_report))
    if operation == "protocol":
        protocol = holdout.build_holdout_protocol(sources, args.model_path)
        if files != two._file_guard(frozen_inputs):
            raise ValueError("Frozen inputs changed during protocol declaration")
        holdout.write_new(args.output, protocol)
        print(json.dumps({"protocol_sha256": protocol["protocol_sha256"], "required_case_ids": protocol["required_case_ids"]}, indent=2))
        return 0
    protocol = load(args.holdout_protocol)
    holdout.require_frozen(args.holdout_protocol, protocol)
    if operation == "plan":
        holdout._protocol_check(sources, protocol, args.model_path)
        plan, bundle = holdout.build_holdout_plan(sources, load_model(), protocol, args.model_path, args.holdout_protocol, args.workers, frozen_inputs)
        if files != two._file_guard(frozen_inputs):
            raise ValueError("Frozen inputs changed during prediction")
        holdout.write_new(args.bundle, bundle)
        holdout.write_new(args.output, plan)
        print(json.dumps({"plan_sha256": plan["plan_sha256"], "prediction_complete": plan["prediction_complete"]}, indent=2))
        return 0 if plan["prediction_complete"] else 1
    plan, bundle = load(args.plan), load(args.bundle)
    holdout._frozen(protocol, plan, bundle, args.holdout_protocol, args.plan, args.bundle)
    if operation == "run":
        holdout.check_holdout_plan(sources, protocol, plan, bundle, args.model_path)
        if not plan["prediction_complete"]:
            report = holdout.holdout_report(protocol, plan, bundle, [])
        else:
            report = holdout.acquire_holdout(sources, load_model(), protocol, plan, bundle, args.model_path, args.holdout_protocol, args.plan, args.bundle, frozen_inputs)
        if files != two._file_guard(frozen_inputs):
            report = holdout.holdout_report(protocol, plan, bundle, report["observations"], dict(report["acquisition_guards"], frozen_files_unchanged=False))
        holdout.write_new(args.output, report)
        summary = holdout.holdout_summary(plan, report)
        if args.summary:
            holdout.write_new(args.summary, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if report["two_layer_holdout_matches"] else 1
    report = load(args.report)
    result = holdout.verify_holdout(sources, protocol, plan, bundle, report, args.model_path)
    if args.summary and result["valid"]:
        result["summary_matches_source"] = canonical_json(load(args.summary)) == canonical_json(holdout.holdout_summary(plan, report))
        result["valid"] = result["summary_matches_source"]
    if args.reexecute and result["valid"]:
        result = holdout.reexecute_holdout(sources, _holdout_model(args.model_path), protocol, plan, bundle, report, args.model_path, args.holdout_protocol, args.plan, args.bundle, args.workers, frozen_inputs)
    if files != two._file_guard(frozen_inputs):
        result.update(valid=False, two_layer_holdout_matches=False, connected_two_layers_independently_recomputed=False, reason="Frozen input files changed")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] and result.get("two_layer_holdout_matches") else 1


def command_two_layers(args: argparse.Namespace) -> int:
    from . import gemma_two_layers as two
    from . import gemma_first_layer_holdout as holdout
    from .serialization import canonical_json

    operation = args.operation
    plan_output = args.output if operation == "plan" and args.output is not None else args.plan
    outputs = [plan_output, args.bundle] if operation == "plan" else [args.output] + ([args.summary] if args.summary else []) if operation == "run" else []
    excluded = {"output"}
    if operation == "plan":
        excluded.update(("plan", "bundle", "summary"))
    if operation == "run":
        excluded.add("summary")
    inputs = [value.resolve() for key, value in vars(args).items() if isinstance(value, Path) and key not in excluded]
    if len({path.resolve() for path in outputs}) != len(outputs) or any(path.exists() or path.resolve() in inputs or any(root.is_dir() and root in path.resolve().parents for root in inputs) for path in outputs):
        raise ValueError("Two-layer artifacts require new distinct paths outside all frozen inputs and model directory")
    repository = Path(__file__).resolve().parents[1]
    private_outputs = [args.bundle] if operation == "plan" else [args.output] if operation == "run" else []
    if any(repository in path.resolve().parents and repository / "artifacts" not in path.resolve().parents for path in private_outputs):
        raise ValueError("Tensor-rich two-layer bundles and native reports inside the repository must remain under Git-ignored artifacts/")
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    frozen_inputs = tuple(path for path in inputs if path.is_file())
    files = two._file_guard(frozen_inputs)
    sources = _load_two_layers_sources(args)
    if operation == "plan":
        context = sources.validate(args.model_path)
        plan, bundle = two._predict(sources, _holdout_model(args.model_path), context, args.workers)
        if files != two._file_guard(frozen_inputs):
            raise ValueError("Frozen input files changed during fresh two-layer prediction")
        holdout.write_new(args.bundle, bundle)
        holdout.write_new(plan_output, plan)
        print(json.dumps({"plan_sha256": plan["plan_sha256"], "coverage": plan["coverage"], "prefix_boundary_reused": False}, indent=2))
        return 0
    plan, bundle = load(args.plan), load(args.bundle)
    if operation == "run":
        two.check_two_layers_plan(sources, plan, bundle, args.model_path)
        report = two.acquire_two_layers(sources, _holdout_model(args.model_path), plan, bundle, args.model_path, args.plan, args.bundle, args.workers, frozen_inputs)
        if files != two._file_guard(frozen_inputs):
            guards = dict(report["acquisition_guards"], frozen_files_unchanged=False)
            report = two.two_layers_report(plan, bundle, report["observations"], guards)
        holdout.write_new(args.output, report)
        summary = two.two_layers_summary(plan, report)
        if args.summary:
            holdout.write_new(args.summary, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if report["two_layers_match"] else 1
    report = load(args.report)
    result = two.verify_two_layers(sources, plan, bundle, report, args.model_path)
    if args.summary and result["valid"]:
        result["summary_matches_source"] = canonical_json(load(args.summary)) == canonical_json(two.two_layers_summary(plan, report))
        result["valid"] = result["summary_matches_source"]
    if args.reexecute and result["valid"]:
        result = two.reexecute_two_layers(sources, _holdout_model(args.model_path), plan, bundle, report, args.model_path, args.plan, args.bundle, args.workers, frozen_inputs)
    if files != two._file_guard(frozen_inputs):
        result.update(valid=False, two_layers_match=False, connected_two_layers_independently_recomputed=False, reason="Frozen input files changed")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] and result.get("two_layers_match") else 1


def command_second_layer(args: argparse.Namespace) -> int:
    from . import gemma_second_layer as second
    from . import gemma_first_layer_holdout as holdout
    from .serialization import canonical_json

    operation = args.operation
    outputs = [args.output, args.bundle] if operation == "plan" else [args.output] + ([args.summary] if args.summary else []) if operation == "run" else []
    excluded = {"output"}
    if operation == "plan":
        excluded.update(("plan", "bundle", "summary"))
    if operation == "run":
        excluded.add("summary")
    inputs = [value.resolve() for key, value in vars(args).items() if isinstance(value, Path) and key not in excluded]
    if len({path.resolve() for path in outputs}) != len(outputs) or any(path.exists() or path.resolve() in inputs or any(root.is_dir() and root in path.resolve().parents for root in inputs) for path in outputs):
        raise ValueError("Second-layer artifacts require new distinct paths outside all frozen inputs and model directory")
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    sources = second.SecondLayerSources(_load_holdout_baseline(args), load(args.protocol), load(args.holdout_plan), load(args.holdout_bundle), load(args.holdout_report))
    if operation == "plan":
        context = sources.validate(args.model_path)
        plan, bundle = second._predict(sources, _holdout_model(args.model_path), context, args.workers)
        holdout.write_new(args.bundle, bundle)
        holdout.write_new(args.output, plan)
        print(json.dumps({"plan_sha256": plan["plan_sha256"], "coverage": plan["coverage"], "profiles": plan["profiles"], "prefix_boundary_reused": True}, indent=2))
        return 0
    plan, bundle = load(args.plan), load(args.bundle)
    if operation == "run":
        second.check_second_layer_plan(sources, plan, bundle, args.model_path)
        report = second.acquire_second_layer(sources, _holdout_model(args.model_path), plan, bundle, args.model_path, args.plan, args.bundle)
        holdout.write_new(args.output, report)
        summary = second.second_layer_summary(plan, report)
        if args.summary:
            holdout.write_new(args.summary, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if report["second_layer_matches"] else 1
    report = load(args.report)
    result = second.verify_second_layer(sources, plan, bundle, report, args.model_path)
    if args.summary and result["valid"]:
        result["summary_matches_source"] = canonical_json(load(args.summary)) == canonical_json(second.second_layer_summary(plan, report))
        result["valid"] = result["summary_matches_source"]
    if args.reexecute and result["valid"]:
        result = second.reexecute_second_layer(sources, _holdout_model(args.model_path), plan, bundle, report, args.model_path, args.plan, args.bundle, args.workers)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] and result.get("second_layer_matches") else 1


def command_first_layer(args: argparse.Namespace) -> int:
    from .gemma_first_layer import FirstLayerSources, build_first_layer_plan, acquire_first_layer, verify_first_layer, reexecute_first_layer, first_layer_summary
    from .serialization import canonical_json

    outputs = [args.output, args.bundle] if args.operation == "plan" else [args.output] + ([args.summary] if args.summary else []) if args.operation == "run" else []
    inputs = [value.resolve() for key, value in vars(args).items() if isinstance(value, Path) and key not in ("output", "summary") and not (args.operation == "plan" and key in ("bundle", "plan"))]
    if len({path.resolve() for path in outputs}) != len(outputs) or any(path.exists() or path.resolve() in inputs for path in outputs):
        raise ValueError("First-layer artifacts must use new, distinct paths outside source evidence")
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    sources = FirstLayerSources(_load_post_feedforward_sources(args), load(args.post_feedforward_plan), load(args.post_feedforward_bundle), load(args.post_feedforward_report))

    def write_new(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")

    def model() -> Any:
        import torch
        from transformers import AutoModelForCausalLM
        if not torch.cuda.is_available():
            raise RuntimeError("First-layer checkpoint binding requires the declared CUDA runtime")
        return AutoModelForCausalLM.from_pretrained(str(args.model_path), local_files_only=True, torch_dtype=torch.bfloat16,
                                                  attn_implementation="eager").to("cuda").eval()

    if args.operation == "plan":
        plan, bundle = build_first_layer_plan(sources, model(), args.workers)
        write_new(args.bundle, bundle)
        write_new(args.output, plan)
        print(json.dumps({"plan_sha256": plan["plan_sha256"], "coverage": plan["coverage"]}, indent=2))
        return 0
    plan, bundle = load(args.plan), load(args.bundle)
    if args.operation == "run":
        report = acquire_first_layer(sources, model(), plan, bundle)
        write_new(args.output, report)
        summary = first_layer_summary(plan, report)
        if args.summary:
            write_new(args.summary, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if report["first_layer_matches"] else 1
    report = load(args.report)
    result = verify_first_layer(sources, plan, bundle, report)
    if args.summary and result["valid"]:
        result["summary_matches_source"] = canonical_json(load(args.summary)) == canonical_json(first_layer_summary(plan, report))
        result["valid"] = result["summary_matches_source"]
    if args.reexecute and result["valid"]:
        result = reexecute_first_layer(sources, model(), plan, bundle, report, args.workers)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


def command_post_feedforward(args: argparse.Namespace) -> int:
    from .gemma_post_feedforward import build_post_plan, acquire_post, verify_post, post_summary
    from .serialization import canonical_json

    outputs = [args.output, args.bundle] if args.operation == "plan" else [args.output] + ([args.summary] if args.summary else []) if args.operation == "run" else []
    inputs = [value.resolve() for key, value in vars(args).items() if isinstance(value, Path) and key not in ("output", "summary") and not (args.operation == "plan" and key in ("bundle", "plan"))]
    if len({path.resolve() for path in outputs}) != len(outputs) or any(path.exists() or path.resolve() in inputs for path in outputs):
        raise ValueError("Post-feedforward artifacts must use new, distinct paths outside source evidence")
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    sources = _load_post_feedforward_sources(args)

    def write_new(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")

    def model() -> Any:
        import torch
        from transformers import AutoModelForCausalLM
        if not torch.cuda.is_available():
            raise RuntimeError("Original post-feedforward comparison requires CUDA")
        return AutoModelForCausalLM.from_pretrained(str(args.model_path), local_files_only=True, torch_dtype=torch.bfloat16,
                                                  attn_implementation="eager").to("cuda").eval()

    if args.operation == "plan":
        plan, bundle = build_post_plan(sources, model())
        write_new(args.bundle, bundle)
        write_new(args.output, plan)
        print(json.dumps({"plan_sha256": plan["plan_sha256"], "value_count": plan["value_count"]}, indent=2))
        return 0
    plan, bundle = load(args.plan), load(args.bundle)
    if args.operation == "run":
        report = acquire_post(sources, model(), plan, bundle)
        write_new(args.output, report)
        summary = post_summary(plan, report)
        if args.summary:
            write_new(args.summary, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if report["post_feedforward_matches"] else 1
    report = load(args.report)
    result = verify_post(sources, plan, bundle, report)
    if args.summary and result["valid"]:
        result["summary_matches_source"] = canonical_json(load(args.summary)) == canonical_json(post_summary(plan, report))
        result["valid"] = result["summary_matches_source"]
    if args.reexecute and result["valid"]:
        original = model()
        replay_plan, replay_bundle = build_post_plan(sources, original)
        same = canonical_json(replay_plan) == canonical_json(plan) and canonical_json(replay_bundle) == canonical_json(bundle)
        replay = acquire_post(sources, original, replay_plan, replay_bundle) if same else None
        exact = same and canonical_json(replay) == canonical_json(report)
        result.update({"valid": same and exact, "mode": "post_feedforward_plan_bundle_regeneration_and_six_original_forward_replay",
                       "predictions_recomputed_exact": same, "reexecution_exact": exact, "prefix_boundary_reused": True,
                       "connected_first_layer_independently_recomputed": False, "completeFirstLayerQualified": False,
                       "global_exactness_activation_allowed": False})
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


def command_dense_k2048(args: argparse.Namespace) -> int:
    from .gemma_k2048_dense import build_dense_plan, acquire_dense, verify_dense, dense_summary
    from .serialization import canonical_json

    outputs = [args.output, args.bundle] if args.operation == "plan" else [args.output] + ([args.summary] if args.summary else []) if args.operation == "run" else []
    inputs = [value.resolve() for key, value in vars(args).items() if isinstance(value, Path) and key not in ("output", "summary") and not (args.operation == "plan" and key in ("bundle", "plan"))]
    if len({path.resolve() for path in outputs}) != len(outputs) or any(path.exists() or path.resolve() in inputs for path in outputs):
        raise ValueError("Dense K2048 artifacts must use new, distinct paths outside source evidence")
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    sources = _load_dense_k2048_sources(args)

    def write_new(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")

    if args.operation == "plan":
        plan, bundle = build_dense_plan(sources, args.workers)
        write_new(args.bundle, bundle)
        write_new(args.output, plan)
        print(json.dumps({"plan_sha256": plan["plan_sha256"], "value_count": plan["value_count"], "candidate": plan["candidate"]["id"]}, indent=2))
        return 0
    plan, bundle = load(args.plan), load(args.bundle)
    if args.operation == "run":
        report = acquire_dense(sources, plan, bundle)
        write_new(args.output, report)
        summary = dense_summary(plan, report)
        if args.summary:
            write_new(args.summary, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if report["candidate_passes_dense_holdout"] else 1
    report = load(args.report)
    result = verify_dense(sources, plan, bundle, report)
    if args.summary and result["valid"]:
        result["summary_matches_source"] = canonical_json(load(args.summary)) == canonical_json(dense_summary(plan, report))
        result["valid"] = result["summary_matches_source"]
    if args.reexecute and result["valid"]:
        replay_plan, replay_bundle = build_dense_plan(sources, args.workers)
        same = canonical_json(replay_plan) == canonical_json(plan) and canonical_json(replay_bundle) == canonical_json(bundle)
        replay = acquire_dense(sources, replay_plan, replay_bundle) if same else None
        exact = canonical_json(replay) == canonical_json(report)
        result.update({"valid": same and exact, "mode": "entire_dense_plan_bundle_regeneration_prediction_and_six_call_cuda_replay", "predictions_recomputed": True,
                       "predictions_recomputed_exact": same, "reexecution_exact": exact})
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


def command_mlp_down(args: argparse.Namespace) -> int:
    from .gemma_mlp_down import build_down_plan, acquire_down, verify_down, down_summary
    from .serialization import canonical_json

    outputs = [args.output, args.bundle] if args.operation == "plan" else [args.output] + ([args.summary] if args.summary else []) if args.operation == "run" else []
    if len({path.resolve() for path in outputs}) != len(outputs) or any(path.exists() for path in outputs):
        raise ValueError("MLP-down artifacts must use new, distinct paths")
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    sources = _load_down_sources(args)

    def write_new(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")

    def model() -> Any:
        import torch
        from transformers import AutoModelForCausalLM
        if not torch.cuda.is_available():
            raise RuntimeError("Original MLP-down model binding requires CUDA")
        return AutoModelForCausalLM.from_pretrained(str(args.model_path), local_files_only=True, torch_dtype=torch.bfloat16,
                                                  attn_implementation="eager").to("cuda").eval()

    if args.operation == "plan":
        plan, bundle = build_down_plan(sources, model(), args.workers)
        write_new(args.bundle, bundle)
        write_new(args.output, plan)
        print(json.dumps({"plan_sha256": plan["plan_sha256"], "value_count": plan["value_count"], "candidate": plan["candidate"]["id"]}, indent=2))
        return 0
    plan, bundle = load(args.plan), load(args.bundle)
    if args.operation == "run":
        report = acquire_down(sources, model(), plan, bundle)
        write_new(args.output, report)
        summary = down_summary(plan, report)
        if args.summary:
            write_new(args.summary, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if report["down_matches"] else 1
    report = load(args.report)
    result = verify_down(sources, plan, bundle, report)
    if args.summary and result["valid"]:
        result["summary_matches_source"] = canonical_json(load(args.summary)) == canonical_json(down_summary(plan, report))
        result["valid"] = result["summary_matches_source"]
    if args.reexecute and result["valid"]:
        original = model()
        replay_plan, replay_bundle = build_down_plan(sources, original, args.workers)
        same = canonical_json(replay_plan) == canonical_json(plan) and canonical_json(replay_bundle) == canonical_json(bundle)
        replay = acquire_down(sources, original, plan, bundle) if same else None
        exact = canonical_json(replay) == canonical_json(report)
        result.update({"valid": same and exact, "mode": "down_dot_recomputation_and_original_forward_replay", "down_predictions_recomputed": True,
                       "predictions_recomputed_exact": same, "reexecution_exact": exact, "prefix_independently_recomputed": False})
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


def command_mlp_product(args: argparse.Namespace) -> int:
    from .gemma_mlp_product import build_product_plan, acquire_product, verify_product, product_summary
    from .serialization import canonical_json

    outputs = [args.output, args.bundle] if args.operation == "plan" else [args.output] + ([args.summary] if args.summary else []) if args.operation == "run" else []
    if len({path.resolve() for path in outputs}) != len(outputs) or any(path.exists() for path in outputs):
        raise ValueError("MLP-product artifacts must use new, distinct paths")
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    sources = _load_product_sources(args)

    def write_new(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")

    def model() -> Any:
        import torch
        from transformers import AutoModelForCausalLM
        if not torch.cuda.is_available():
            raise RuntimeError("Original activation/product comparison requires CUDA")
        return AutoModelForCausalLM.from_pretrained(str(args.model_path), local_files_only=True, torch_dtype=torch.bfloat16,
                                                  attn_implementation="eager").to("cuda").eval()

    if args.operation == "plan":
        plan, bundle = build_product_plan(sources)
        write_new(args.bundle, bundle)
        write_new(args.output, plan)
        print(json.dumps({"plan_sha256": plan["plan_sha256"], "activation_value_count": plan["activation_value_count"], "product_value_count": plan["product_value_count"]}, indent=2))
        return 0
    plan, bundle = load(args.plan), load(args.bundle)
    if args.operation == "run":
        report = acquire_product(sources, model(), plan, bundle)
        write_new(args.output, report)
        summary = product_summary(plan, report)
        if args.summary:
            write_new(args.summary, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if report["activation_and_product_match"] else 1
    report = load(args.report)
    result = verify_product(sources, plan, bundle, report)
    if args.summary and result["valid"]:
        result["summary_matches_source"] = canonical_json(load(args.summary)) == canonical_json(product_summary(plan, report))
        result["valid"] = result["summary_matches_source"]
    if args.reexecute and result["valid"]:
        replay_plan, replay_bundle = build_product_plan(sources)
        same = canonical_json(replay_plan) == canonical_json(plan) and canonical_json(replay_bundle) == canonical_json(bundle)
        replay = acquire_product(sources, model(), plan, bundle) if same else None
        exact = canonical_json(replay) == canonical_json(report)
        result.update({"valid": same and exact, "mode": "activation_product_recomputation_and_original_forward_replay", "predictions_recomputed_exact": same,
                       "reexecution_exact": exact, "prefix_independently_recomputed": False})
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


def command_mlp_entry(args: argparse.Namespace) -> int:
    from .gemma_mlp_entry import build_mlp_plan, acquire_mlp, verify_mlp, mlp_summary
    from .serialization import canonical_json

    outputs = [args.output, args.bundle] if args.operation == "plan" else [args.output] + ([args.summary] if args.summary else []) if args.operation == "run" else []
    if len({path.resolve() for path in outputs}) != len(outputs) or any(path.exists() for path in outputs):
        raise ValueError("MLP-entry evidence must use new, distinct paths")
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    sources = _load_mlp_entry_sources(args)

    def write_new(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")

    def model() -> Any:
        import torch
        from transformers import AutoModelForCausalLM
        if not torch.cuda.is_available():
            raise RuntimeError("Original MLP-entry comparison requires CUDA")
        return AutoModelForCausalLM.from_pretrained(str(args.model_path), local_files_only=True, torch_dtype=torch.bfloat16,
                                                  attn_implementation="eager").to("cuda").eval()

    if args.operation == "plan":
        plan, bundle = build_mlp_plan(sources, model(), args.workers)
        write_new(args.bundle, bundle)
        write_new(args.output, plan)
        print(json.dumps({"plan_sha256": plan["plan_sha256"], "projection_value_count": plan["projection_value_count"], "normalization_value_count": plan["normalization_value_count"]}, indent=2))
        return 0
    plan, bundle = load(args.plan), load(args.bundle)
    if args.operation == "run":
        report = acquire_mlp(sources, model(), plan, bundle)
        write_new(args.output, report)
        summary = mlp_summary(plan, report)
        if args.summary:
            write_new(args.summary, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if report["mlp_entry_matches"] else 1
    report = load(args.report)
    result = verify_mlp(sources, plan, bundle, report)
    if args.summary and result["valid"]:
        result["summary_matches_source"] = canonical_json(load(args.summary)) == canonical_json(mlp_summary(plan, report))
        result["valid"] = result["summary_matches_source"]
    if args.reexecute and result["valid"]:
        original = model()
        replay_plan, replay_bundle = build_mlp_plan(sources, original, args.workers)
        same = canonical_json(replay_plan) == canonical_json(plan) and canonical_json(replay_bundle) == canonical_json(bundle)
        replay = acquire_mlp(sources, original, plan, bundle) if same else None
        exact = canonical_json(replay) == canonical_json(report)
        result.update({"valid": same and exact, "mode": "mlp_entry_recomputation_and_original_forward_replay", "predictions_recomputed_exact": same,
                       "reexecution_exact": exact, "prefix_independently_recomputed": False})
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


def command_post_attention(args: argparse.Namespace) -> int:
    from .gemma_post_attention import build_post_plan, acquire_post, verify_post, post_summary
    from .serialization import canonical_json

    outputs = [args.output, args.bundle] if args.operation == "plan" else [args.output] + ([args.summary] if args.summary else []) if args.operation == "run" else []
    if len({path.resolve() for path in outputs}) != len(outputs) or any(path.exists() for path in outputs):
        raise ValueError("Post-attention evidence must use new, distinct paths")
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    sources = _load_post_attention_sources(args)

    def write_new(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")

    def model() -> Any:
        import torch
        from transformers import AutoModelForCausalLM
        if not torch.cuda.is_available():
            raise RuntimeError("Original post-attention comparison requires CUDA")
        return AutoModelForCausalLM.from_pretrained(str(args.model_path), local_files_only=True, torch_dtype=torch.bfloat16,
                                                  attn_implementation="eager").to("cuda").eval()

    if args.operation == "plan":
        plan, bundle = build_post_plan(sources, model())
        write_new(args.bundle, bundle)
        write_new(args.output, plan)
        print(json.dumps({"plan_sha256": plan["plan_sha256"], "value_count": plan["value_count"], "scalar_stage_positions": plan["scalar_stage_positions"]}, indent=2))
        return 0
    plan, bundle = load(args.plan), load(args.bundle)
    if args.operation == "run":
        report = acquire_post(sources, model(), plan, bundle)
        write_new(args.output, report)
        summary = post_summary(plan, report)
        if args.summary:
            write_new(args.summary, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if report["post_attention_matches"] else 1
    report = load(args.report)
    result = verify_post(sources, plan, bundle, report)
    if args.summary and result["valid"]:
        result["summary_matches_source"] = canonical_json(load(args.summary)) == canonical_json(post_summary(plan, report))
        result["valid"] = result["summary_matches_source"]
    if args.reexecute and result["valid"]:
        original = model()
        replay_plan, replay_bundle = build_post_plan(sources, original)
        same = canonical_json(replay_plan) == canonical_json(plan) and canonical_json(replay_bundle) == canonical_json(bundle)
        replay = acquire_post(sources, original, plan, bundle) if same else None
        exact = canonical_json(replay) == canonical_json(report)
        result.update({"valid": same and exact, "mode": "post_attention_recomputation_and_original_forward_replay", "predictions_recomputed_exact": same,
                       "reexecution_exact": exact, "prefix_independently_recomputed": False})
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


def command_output_survivor(args: argparse.Namespace) -> int:
    from .gemma_output_survivor import build_survivor_plan, acquire_survivor, verify_survivor, survivor_summary
    from .serialization import canonical_json

    outputs = [args.output, args.bundle] if args.operation == "plan" else [args.output] + ([args.summary] if args.summary else []) if args.operation == "run" else []
    if len({path.resolve() for path in outputs}) != len(outputs) or any(path.exists() for path in outputs):
        raise ValueError("Output-survivor evidence must use new, distinct paths")
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    context = _load_survivor_context(args)

    def write_new(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")

    def model() -> Any:
        import torch
        from transformers import AutoModelForCausalLM
        if not torch.cuda.is_available():
            raise RuntimeError("Output-survivor comparison requires CUDA")
        return AutoModelForCausalLM.from_pretrained(str(args.model_path), local_files_only=True, torch_dtype=torch.bfloat16,
                                                  attn_implementation="eager").to("cuda").eval()

    if args.operation == "plan":
        plan, bundle = build_survivor_plan(*context)
        write_new(args.bundle, bundle)
        write_new(args.output, plan)
        print(json.dumps({"plan_sha256": plan["plan_sha256"], "candidate": plan["candidate"], "fresh_model_holdout": False}, indent=2))
        return 0
    plan, bundle = load(args.plan), load(args.bundle)
    if args.operation == "run":
        report = acquire_survivor(context, plan, bundle, model())
        write_new(args.output, report)
        summary = survivor_summary(plan, report)
        if args.summary:
            write_new(args.summary, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if report["survivor_reproduces_model_case"] else 1
    report = load(args.report)
    result = verify_survivor(context, plan, bundle, report)
    if args.summary and result["valid"]:
        result["summary_matches_source"] = canonical_json(load(args.summary)) == canonical_json(survivor_summary(plan, report))
        result["valid"] = result["summary_matches_source"]
    if args.reexecute and result["valid"]:
        replay_plan, replay_bundle = build_survivor_plan(*context)
        same = canonical_json(replay_plan) == canonical_json(plan) and canonical_json(replay_bundle) == canonical_json(bundle)
        replay = acquire_survivor(context, plan, bundle, model()) if same else None
        exact = canonical_json(replay) == canonical_json(report)
        result.update({"valid": same and exact, "mode": "survivor_projection_recomputation_and_model_case_replay", "predictions_recomputed_exact": same,
                       "reexecution_exact": exact, "fresh_model_holdout": False})
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


def command_output_split(args: argparse.Namespace) -> int:
    from .gemma_output_split import build_split_output_plan, acquire_split_output, verify_split_output, split_output_summary
    from .serialization import canonical_json

    outputs = [args.output, args.bundle] if args.operation == "plan" else [args.output] + ([args.summary] if args.summary else []) if args.operation == "run" else []
    if len({path.resolve() for path in outputs}) != len(outputs) or any(path.exists() for path in outputs):
        raise ValueError("Split-output artifacts must be new and distinct")
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    sources = _load_output_sources(args)
    context = (sources, load(args.base_plan), load(args.base_bundle), load(args.base_report))

    def write_new(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")

    def model() -> Any:
        import torch
        from transformers import AutoModelForCausalLM
        if not torch.cuda.is_available():
            raise RuntimeError("Split output comparison requires CUDA")
        return AutoModelForCausalLM.from_pretrained(str(args.model_path), local_files_only=True, torch_dtype=torch.bfloat16,
                                                  attn_implementation="eager").to("cuda").eval()

    if args.operation == "plan":
        plan, bundle = build_split_output_plan(*context)
        write_new(args.bundle, bundle)
        write_new(args.output, plan)
        print(json.dumps({"plan_sha256": plan["plan_sha256"], "value_count": plan["value_count"], "fresh_holdout_validation": False}, indent=2))
        return 0
    plan, bundle = load(args.plan), load(args.bundle)
    if args.operation == "run":
        report = acquire_split_output(*context, plan, bundle, model())
        write_new(args.output, report)
        summary = split_output_summary(plan, report)
        if args.summary:
            write_new(args.summary, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if report["split_projection_matches_case"] else 1
    report = load(args.report)
    result = verify_split_output(*context, plan, bundle, report)
    if args.summary and result["valid"]:
        result["summary_matches_source"] = canonical_json(load(args.summary)) == canonical_json(split_output_summary(plan, report))
        result["valid"] = result["summary_matches_source"]
    if args.reexecute and result["valid"]:
        replay_plan, replay_bundle = build_split_output_plan(*context)
        same = canonical_json(replay_plan) == canonical_json(plan) and canonical_json(replay_bundle) == canonical_json(bundle)
        replay = acquire_split_output(*context, plan, bundle, model()) if same else None
        exact = canonical_json(replay) == canonical_json(report)
        result.update({"valid": same and exact, "mode": "split_projection_recomputation_and_original_forward_replay",
                       "predictions_recomputed_exact": same, "reexecution_exact": exact, "fresh_holdout_validation": False})
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


def command_attention_output(args: argparse.Namespace) -> int:
    from .gemma_attention_output import build_output_plan, acquire_output, verify_output, output_summary
    from .serialization import canonical_json

    outputs = [args.output, args.bundle] if args.operation == "plan" else [args.output] + ([args.summary] if args.summary else []) if args.operation == "run" else []
    if len({path.resolve() for path in outputs}) != len(outputs) or any(path.exists() for path in outputs):
        raise ValueError("Attention-output artifacts must be new and distinct")
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    sources = _load_output_sources(args)

    def write_new(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")

    def model() -> Any:
        import torch
        from transformers import AutoModelForCausalLM
        if not torch.cuda.is_available():
            raise RuntimeError("Original attention-output comparison requires CUDA")
        return AutoModelForCausalLM.from_pretrained(str(args.model_path), local_files_only=True, torch_dtype=torch.bfloat16,
                                                  attn_implementation="eager").to("cuda").eval()

    if args.operation == "plan":
        plan, bundle = build_output_plan(sources, model())
        write_new(args.bundle, bundle)
        write_new(args.output, plan)
        print(json.dumps({"plan_sha256": plan["plan_sha256"], "aggregation_value_count": plan["aggregation_value_count"], "projection_value_count": plan["projection_value_count"]}, indent=2))
        return 0
    plan, bundle = load(args.plan), load(args.bundle)
    if args.operation == "run":
        report = acquire_output(sources, model(), plan, bundle)
        write_new(args.output, report)
        summary = output_summary(plan, report)
        if args.summary:
            write_new(args.summary, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if report["candidate_passes"] else 1
    report = load(args.report)
    result = verify_output(sources, plan, bundle, report)
    if args.summary and result["valid"]:
        result["summary_matches_source"] = canonical_json(load(args.summary)) == canonical_json(output_summary(plan, report))
        result["valid"] = result["summary_matches_source"]
    if args.reexecute and result["valid"]:
        original = model()
        replay_plan, replay_bundle = build_output_plan(sources, original)
        same = canonical_json(replay_plan) == canonical_json(plan) and canonical_json(replay_bundle) == canonical_json(bundle)
        replay = acquire_output(sources, original, plan, bundle) if same else None
        exact = canonical_json(replay) == canonical_json(report)
        result.update({"valid": same and exact, "mode": "independent_attention_output_and_original_forward_replay",
                       "predictions_recomputed_exact": same, "reexecution_exact": exact, "prefix_independently_recomputed": False})
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


def command_lookup_softmax(args: argparse.Namespace) -> int:
    from .gemma_exp_lookup import CheckedExpLookup
    from .gemma_softmax_lookup import build_lookup_softmax_plan, acquire_lookup_softmax, verify_lookup_softmax, lookup_softmax_summary
    from .serialization import canonical_json

    outputs = [args.output, args.bundle] if args.operation == "plan" else [args.output] + ([args.summary] if args.summary else []) if args.operation == "run" else []
    if len({path.resolve() for path in outputs}) != len(outputs) or any(path.exists() for path in outputs):
        raise ValueError("Lookup-softmax outputs must be new and distinct")
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    sources = _load_softmax_sources(args)
    lookup = CheckedExpLookup(load(args.exp_plan), load(args.exp_manifest), args.exp_table, args.exp_audit_dir)
    context = (sources, load(args.original_plan), load(args.original_bundle), load(args.original_report), lookup)

    def write_new(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")

    def model() -> Any:
        import torch
        from transformers import AutoModelForCausalLM
        if not torch.cuda.is_available():
            raise RuntimeError("Lookup-softmax comparison requires CUDA")
        return AutoModelForCausalLM.from_pretrained(str(args.model_path), local_files_only=True, torch_dtype=torch.bfloat16,
                                                  attn_implementation="eager").to("cuda").eval()

    if args.operation == "plan":
        plan, bundle = build_lookup_softmax_plan(*context)
        write_new(args.bundle, bundle)
        write_new(args.output, plan)
        print(json.dumps({"plan_sha256": plan["plan_sha256"], "value_count": plan["value_count"], "lookup_evidence": plan["lookup_evidence"]}, indent=2))
        return 0
    plan, bundle = load(args.plan), load(args.bundle)
    if args.operation == "run":
        report = acquire_lookup_softmax(*context, plan, bundle, model())
        write_new(args.output, report)
        summary = lookup_softmax_summary(plan, report)
        if args.summary:
            write_new(args.summary, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if report["lookup_softmax_passes"] else 1
    report = load(args.report)
    result = verify_lookup_softmax(*context, plan, bundle, report)
    if args.summary and result["valid"]:
        result["summary_matches_source"] = canonical_json(load(args.summary)) == canonical_json(lookup_softmax_summary(plan, report))
        result["valid"] = result["summary_matches_source"]
    if args.reexecute and result["valid"]:
        regenerated_plan, regenerated_bundle = build_lookup_softmax_plan(*context)
        same = canonical_json(regenerated_plan) == canonical_json(plan) and canonical_json(regenerated_bundle) == canonical_json(bundle)
        replay = acquire_lookup_softmax(*context, plan, bundle, model()) if same else None
        exact = canonical_json(replay) == canonical_json(report)
        result.update({"valid": same and exact, "mode": "lookup_softmax_and_original_forward_replay", "predictions_recomputed_exact": same,
                       "reexecution_exact": exact, "prefix_independently_recomputed": False})
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


def command_softmax_slice(args: argparse.Namespace) -> int:
    from .gemma_softmax_slice import build_softmax_plan, acquire_softmax, verify_softmax, softmax_summary
    from .serialization import canonical_json

    outputs = [args.output, args.bundle] if args.operation == "plan" else [args.output] + ([args.summary] if args.summary else []) if args.operation == "run" else []
    if len({path.resolve() for path in outputs}) != len(outputs) or any(path.exists() for path in outputs):
        raise ValueError("Softmax outputs must be new, distinct files")
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    sources = _load_softmax_sources(args)

    def write_new(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")

    def model() -> Any:
        import torch
        from transformers import AutoModelForCausalLM

        if not torch.cuda.is_available():
            raise RuntimeError("Native softmax acquisition requires CUDA")
        return AutoModelForCausalLM.from_pretrained(str(args.model_path), local_files_only=True, torch_dtype=torch.bfloat16,
                                                  attn_implementation="eager").to("cuda").eval()

    if args.operation == "plan":
        plan, bundle = build_softmax_plan(sources)
        write_new(args.bundle, bundle)
        write_new(args.output, plan)
        print(json.dumps({"plan_sha256": plan["plan_sha256"], "row_count": plan["row_count"], "value_count": plan["value_count"]}, indent=2))
        return 0
    plan, bundle = load(args.plan), load(args.bundle)
    if args.operation == "run":
        report = acquire_softmax(sources, model(), plan, bundle)
        write_new(args.output, report)
        summary = softmax_summary(plan, report)
        if args.summary:
            write_new(args.summary, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if report["candidate_passes"] else 1
    report = load(args.report)
    result = verify_softmax(sources, plan, bundle, report)
    if args.summary and result["valid"]:
        result["summary_matches_source"] = canonical_json(load(args.summary)) == canonical_json(softmax_summary(plan, report))
        result["valid"] = result["summary_matches_source"]
    if args.reexecute and result["valid"]:
        replay_plan, replay_bundle = build_softmax_plan(sources)
        predictions_match = canonical_json(replay_plan) == canonical_json(plan) and canonical_json(replay_bundle) == canonical_json(bundle)
        replay = acquire_softmax(sources, model(), plan, bundle) if predictions_match else None
        exact = canonical_json(replay) == canonical_json(report)
        result.update({"valid": predictions_match and exact, "mode": "independent_softmax_and_original_forward_replay",
                       "predictions_recomputed_exact": predictions_match, "reexecution_exact": exact, "prefix_independently_recomputed": False})
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


def command_attention_scores(args: argparse.Namespace) -> int:
    from .gemma_attention_scores import ScoreSources, build_score_plan, acquire_scores, verify_scores, score_summary
    from .gemma_rsqrt_lookup import CheckedRsqrtLookup
    from .serialization import canonical_json

    outputs = [args.output, args.bundle] if args.operation == "plan" else [args.output] + ([args.summary] if args.summary else []) if args.operation == "run" else []
    if len({path.resolve() for path in outputs}) != len(outputs) or any(path.exists() for path in outputs):
        raise ValueError("Attention-score outputs must be new, distinct files")
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    lookup = CheckedRsqrtLookup(load(args.rsqrt_table_plan), load(args.rsqrt_table_manifest), args.rsqrt_table,
                                load(args.rsqrt_domain_plan), load(args.rsqrt_domain_report), args.rsqrt_audit_dir)
    sources = ScoreSources(load(args.program), load(args.rotary_plan), load(args.rotary_bundle), load(args.rotary_report), lookup,
                           load(args.table_plan), load(args.table_manifest), load(args.table_bundle))

    def write_new(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")

    def model() -> Any:
        import torch
        from transformers import AutoModelForCausalLM

        if not torch.cuda.is_available():
            raise RuntimeError("Attention-score acquisition requires CUDA")
        return AutoModelForCausalLM.from_pretrained(str(args.model_path), local_files_only=True, torch_dtype=torch.bfloat16,
                                                  attn_implementation="eager").to("cuda").eval()

    if args.operation == "plan":
        plan, bundle = build_score_plan(sources)
        write_new(args.bundle, bundle)
        write_new(args.output, plan)
        print(json.dumps({"plan_sha256": plan["plan_sha256"], "score_count": plan["score_count"], "prefix_boundary_reused": True}, indent=2))
        return 0
    plan, bundle = load(args.plan), load(args.bundle)
    if args.operation == "run":
        report = acquire_scores(sources, model(), plan, bundle)
        write_new(args.output, report)
        summary = score_summary(plan, report)
        if args.summary:
            write_new(args.summary, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if report["candidate_passes"] else 1
    report = load(args.report)
    result = verify_scores(sources, plan, bundle, report)
    if args.summary and result["valid"]:
        result["summary_matches_source"] = canonical_json(load(args.summary)) == canonical_json(score_summary(plan, report))
        result["valid"] = result["summary_matches_source"]
    if args.reexecute and result["valid"]:
        replay_plan, replay_bundle = build_score_plan(sources)
        predictions_match = canonical_json(replay_plan) == canonical_json(plan) and canonical_json(replay_bundle) == canonical_json(bundle)
        replay = acquire_scores(sources, model(), plan, bundle) if predictions_match else None
        exact = canonical_json(replay) == canonical_json(report)
        result.update({"valid": predictions_match and exact, "mode": "score_prediction_and_original_forward_replay",
                       "predictions_recomputed_exact": predictions_match, "reexecution_exact": exact, "prefix_independently_recomputed": False})
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


def command_rotary_slice(args: argparse.Namespace) -> int:
    from .gemma_rotary_slice import build_rotary_table_plan, acquire_rotary_table, check_rotary_table, build_rotary_slice, acquire_rotary_slice, verify_rotary_slice, rotary_slice_summary
    from .gemma_rsqrt_lookup import CheckedRsqrtLookup
    from .serialization import canonical_json

    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    outputs = [] if args.operation.endswith("verify") else [args.output]
    if args.operation in ("table-run", "plan"):
        outputs.append(args.bundle)
    if args.operation == "run" and args.summary:
        outputs.append(args.summary)
    if any(path is None for path in outputs) or len({path.resolve() for path in outputs}) != len(outputs) or any(path.exists() for path in outputs):
        raise ValueError("Rotary outputs must be new, distinct files; a bundle is required for table-run and plan")

    def write_new(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")

    def model() -> Any:
        import torch
        from transformers import AutoModelForCausalLM

        if not torch.cuda.is_available():
            raise RuntimeError("Original rotary acquisition/replay requires CUDA")
        return AutoModelForCausalLM.from_pretrained(str(args.model_path), local_files_only=True, torch_dtype=torch.bfloat16,
                                                  attn_implementation="eager").to("cuda").eval()

    program = load(args.program)
    if args.operation == "table-plan":
        plan = build_rotary_table_plan(program, model())
        write_new(args.output, plan)
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    table_plan = load(args.table_plan)
    if args.operation == "table-run":
        manifest, bundle = acquire_rotary_table(program, model(), table_plan)
        write_new(args.bundle, bundle)
        write_new(args.output, manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0 if manifest["table_usable"] else 1
    table_manifest, table_bundle = load(args.table_manifest), load(args.table_bundle)
    check_rotary_table(program, table_plan, table_manifest, table_bundle)
    if args.operation == "table-verify":
        result = {"valid": True, "mode": "integrity_only", "table_usable": table_manifest["table_usable"]}
        if args.reexecute:
            replay_manifest, replay_bundle = acquire_rotary_table(program, model(), table_plan)
            exact = canonical_json(replay_manifest) == canonical_json(table_manifest) and canonical_json(replay_bundle) == canonical_json(table_bundle)
            result.update({"valid": exact, "mode": "native_rotary_table_replay", "reexecution_exact": exact})
        print(json.dumps(result, indent=2))
        return 0 if result["valid"] else 1
    fixture = load(args.fixture)
    lookup = CheckedRsqrtLookup(load(args.rsqrt_table_plan), load(args.rsqrt_table_manifest), args.rsqrt_table,
                                load(args.rsqrt_domain_plan), load(args.rsqrt_domain_report), args.rsqrt_audit_dir)
    evidence = (lookup, table_plan, table_manifest, table_bundle)
    if args.operation == "plan":
        plan, bundle = build_rotary_slice(program, model(), fixture, *evidence)
        write_new(args.bundle, bundle)
        write_new(args.output, plan)
        print(json.dumps({"plan_sha256": plan["plan_sha256"], "instruction_count": plan["instruction_count"], "target_value_count": plan["target_value_count"]}, indent=2))
        return 0
    plan, bundle = load(args.plan), load(args.bundle)
    if bundle["entry_plan"]["fixture_sha256"] != hashlib.sha256(canonical_json(fixture).encode("utf-8")).hexdigest():
        raise ValueError("Rotary fixture differs from the frozen plan")
    if args.operation == "run":
        report = acquire_rotary_slice(program, model(), plan, bundle, *evidence)
        write_new(args.output, report)
        summary = rotary_slice_summary(plan, report)
        if args.summary:
            write_new(args.summary, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if report["rotated_attention_inputs_bit_exact"] else 1
    report = load(args.report)
    result = verify_rotary_slice(program, plan, bundle, report, *evidence)
    if args.summary and result["valid"]:
        result["summary_matches_source"] = canonical_json(load(args.summary)) == canonical_json(rotary_slice_summary(plan, report))
        result["valid"] = result["summary_matches_source"]
    if args.reexecute and result["valid"]:
        original = model()
        replay_plan, replay_bundle = build_rotary_slice(program, original, fixture, *evidence)
        exact_prediction = canonical_json(replay_plan) == canonical_json(plan) and canonical_json(replay_bundle) == canonical_json(bundle)
        replay_report = acquire_rotary_slice(program, original, plan, bundle, *evidence) if exact_prediction else None
        exact = canonical_json(replay_report) == canonical_json(report)
        result.update({"valid": exact_prediction and exact, "predictions_recomputed_exact": exact_prediction, "reexecution_exact": exact,
                       "mode": "connected_independent_prediction_and_original_attention_input_replay"})
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


def command_attention_entry(args: argparse.Namespace) -> int:
    from .gemma_attention_entry import build_attention_entry_plan, acquire_attention_entry, verify_attention_entry, attention_entry_summary
    from .gemma_rsqrt_lookup import CheckedRsqrtLookup
    from .serialization import canonical_json

    def read_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    outputs = [args.output, args.bundle] if args.operation == "plan" else [args.output] + ([args.summary] if args.summary else []) if args.operation == "run" else []
    if len({path.resolve() for path in outputs}) != len(outputs) or any(path.exists() for path in outputs):
        raise ValueError("Entry outputs must be new, distinct files")
    program, fixture = read_json(args.program), read_json(args.fixture)
    lookup = CheckedRsqrtLookup(read_json(args.table_plan), read_json(args.table_manifest), args.table, read_json(args.domain_plan), read_json(args.domain_report), args.audit_dir)

    def load_model() -> Any:
        import torch
        from transformers import AutoModelForCausalLM

        if not torch.cuda.is_available():
            raise RuntimeError("Original attention entry comparison requires CUDA")
        return AutoModelForCausalLM.from_pretrained(str(args.model_path), local_files_only=True, torch_dtype=torch.bfloat16,
                                                  attn_implementation="eager").to("cuda").eval()

    def write_new(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")

    if args.operation == "plan":
        plan, bundle = build_attention_entry_plan(program, load_model(), fixture, lookup)
        write_new(args.bundle, bundle)
        write_new(args.output, plan)
        print(json.dumps({"plan_sha256": plan["plan_sha256"], "independent_instruction_count": len(plan["records"]), "target_value_count": plan["target_value_count"]}, indent=2))
        return 0
    plan, bundle = read_json(args.plan), read_json(args.bundle)
    if plan["fixture_sha256"] != hashlib.sha256(canonical_json(fixture).encode("utf-8")).hexdigest() or canonical_json(plan["input_token_ids"]) != canonical_json(fixture["input_token_ids"]):
        raise ValueError("Entry input fixture differs from the prediction plan")
    if args.operation == "run":
        report = acquire_attention_entry(program, load_model(), plan, bundle, lookup)
        write_new(args.output, report)
        summary = attention_entry_summary(plan, report)
        if args.summary:
            write_new(args.summary, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if report["connected_slice_bit_exact"] else 1
    report = read_json(args.report)
    result = verify_attention_entry(program, plan, bundle, report, lookup)
    if args.summary:
        result["summary_matches_source"] = canonical_json(read_json(args.summary)) == canonical_json(attention_entry_summary(plan, report))
        result["valid"] = result["valid"] and result["summary_matches_source"]
    if args.reexecute and result["valid"]:
        model = load_model()
        recomputed_plan, recomputed_bundle = build_attention_entry_plan(program, model, fixture, lookup)
        recomputed = canonical_json(recomputed_plan) == canonical_json(plan) and canonical_json(recomputed_bundle) == canonical_json(bundle)
        replay = acquire_attention_entry(program, model, plan, bundle, lookup) if recomputed else None
        exact = canonical_json(replay) == canonical_json(report)
        result.update({"mode": "independent_entry_and_original_forward_replay", "predictions_recomputed_exact": recomputed,
                       "reexecution_exact": exact, "valid": recomputed and exact})
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


def command_rsqrt_lookup(args: argparse.Namespace) -> int:
    from .gemma_rsqrt_lookup import build_table_plan, acquire_table, load_table, build_domain_plan, acquire_domain, verify_domain
    from .gemma_rsqrt_lookup import CheckedRsqrtLookup, build_rms_lookup_plan, acquire_rms_lookup, verify_rms_lookup

    def read_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def write_new(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        status = {key: payload[key] for key in ("plan_sha256", "manifest_sha256", "report_sha256", "table_sha256", "table_bytes", "base_domain_usable", "tested_value_count", "mismatch_count", "all_positive_normal_values_match", "lookup_rms_check_passes", "all_stored_rms_values_and_stages_match", "all_native_root_layout_replays_match") if key in payload}
        print(json.dumps(status, indent=2, sort_keys=True))

    if args.operation == "table-plan":
        write_new(args.output, build_table_plan(read_json(args.rms_summary)))
        return 0
    table_plan = read_json(args.table_plan)
    if args.operation == "table-build":
        if args.output.exists() or args.table.exists() or args.output.resolve() == args.table.resolve():
            raise ValueError("Lookup table and manifest outputs must be new, distinct files")
        args.table.parent.mkdir(parents=True, exist_ok=True)
        manifest = acquire_table(table_plan, args.table, args.audit_dir)
        write_new(args.output, manifest)
        return 0 if manifest["base_domain_usable"] else 1
    manifest = read_json(args.manifest)
    if args.operation.startswith("rms-"):
        domain_plan, domain_report = read_json(args.domain_plan), read_json(args.domain_report)
        lookup = CheckedRsqrtLookup(table_plan, manifest, args.table, domain_plan, domain_report, args.audit_dir)
        sources = [read_json(path) for path in (args.program, args.rms_plan, args.rms_bundle, args.rms_report)]
        if args.operation == "rms-plan":
            if args.bundle.exists() or args.output.exists() or args.bundle.resolve() == args.output.resolve():
                raise ValueError("Lookup RMS outputs must be new and distinct")
            plan, bundle = build_rms_lookup_plan(*sources, lookup)
            args.bundle.parent.mkdir(parents=True, exist_ok=True)
            with args.bundle.open("x", encoding="utf-8") as stream:
                json.dump(bundle, stream, sort_keys=True)
            write_new(args.output, plan)
            return 0
        plan, bundle = read_json(args.plan), read_json(args.bundle)
        if args.operation == "rms-run":
            if args.output.exists():
                raise ValueError("Lookup RMS output must be a new file")
            report = acquire_rms_lookup(*sources, lookup, plan, bundle, args.audit_dir)
            write_new(args.output, report)
            return 0 if report["lookup_rms_check_passes"] else 1
        result = verify_rms_lookup(*sources, lookup, plan, bundle, read_json(args.report), args.audit_dir, args.reexecute)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["valid"] else 1
    if args.operation == "table-verify":
        load_table(table_plan, manifest, args.table)
        print(json.dumps({"valid": True, "mode": "artifact_integrity", "outside_base_domain_enabled": False}, indent=2))
        return 0
    if args.operation == "domain-plan":
        write_new(args.output, build_domain_plan(table_plan, manifest, args.table))
        return 0
    plan = read_json(args.plan)
    if args.operation == "domain-run":
        temporary = args.journal.with_suffix(".tmp")
        if any(path.exists() for path in (args.output, args.journal, temporary)) or len({path.resolve() for path in (args.output, args.journal, temporary)}) != 3:
            raise ValueError("Domain report and journal paths must be new and distinct")
        args.journal.parent.mkdir(parents=True, exist_ok=True)

        def persist(payload: dict[str, Any]) -> None:
            with temporary.open("x", encoding="utf-8") as stream:
                json.dump(payload, stream, sort_keys=True)
            temporary.replace(args.journal)

        report = acquire_domain(table_plan, manifest, args.table, plan, args.audit_dir, persist)
        write_new(args.output, report)
        return 0 if report["all_positive_normal_values_match"] else 1
    report = read_json(args.report)
    result = verify_domain(table_plan, manifest, args.table, plan, report, args.audit_dir, args.reexecute)
    result["covered_exponent_count"] = len(result.pop("covered_exponents", []))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


def command_rms_slice(args: argparse.Namespace) -> int:
    from .reference_gemma import build_rms_slice, acquire_rms_slice, verify_rms_slice, rms_slice_summary, check_rms_projection_source, rms_observer_controls
    from .serialization import canonical_json

    sources = [args.program, args.projection_plan, args.projection_bundle, args.projection_summary]
    inputs = sources + ([args.plan, args.bundle] if args.operation != "plan" else []) + ([args.report] if args.operation == "verify" else [])
    outputs = [args.output, args.bundle] if args.operation == "plan" else [args.output] + ([args.summary] if args.summary else []) if args.operation == "run" else []
    if len({path.resolve() for path in outputs}) != len(outputs) or {path.resolve() for path in outputs}.intersection(path.resolve() for path in inputs):
        raise ValueError("RMS outputs must not overwrite source evidence or each other")
    if args.operation == "verify" and args.summary and args.summary.resolve() in {path.resolve() for path in inputs}:
        raise ValueError("RMS summary must be distinct from its sources")
    program, projection_plan, projection_bundle, projection_summary = [json.loads(path.read_text(encoding="utf-8")) for path in sources]

    def load_model() -> Any:
        import torch
        from transformers import AutoModelForCausalLM

        if not torch.cuda.is_available():
            raise RuntimeError("RMS slice requires CUDA")
        return AutoModelForCausalLM.from_pretrained(str(args.model_path), local_files_only=True, torch_dtype=torch.bfloat16,
                                                  attn_implementation="eager").to("cuda").eval()

    if args.operation == "plan":
        plan, bundle = build_rms_slice(program, load_model(), projection_plan, projection_bundle, projection_summary)
        args.bundle.parent.mkdir(parents=True, exist_ok=True)
        args.bundle.write_text(json.dumps(bundle, sort_keys=True) + "\n", encoding="utf-8")
        _write_json(args.output, plan)
        return 0
    plan, bundle = [json.loads(path.read_text(encoding="utf-8")) for path in (args.plan, args.bundle)]
    if plan["source_projection_plan_sha256"] != projection_plan["plan_sha256"] or plan["source_projection_summary_sha256"] != projection_summary["summary_sha256"]:
        raise ValueError("RMS source references differ from the prediction plan")
    if args.operation == "run":
        report = acquire_rms_slice(program, load_model(), projection_plan, projection_bundle, projection_summary, plan, bundle)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_json(args.summary, rms_slice_summary(plan, report))
        return 0 if report["all_stages_candidate_passes"] else 1
    report = json.loads(args.report.read_text(encoding="utf-8"))
    result = verify_rms_slice(program, plan, bundle, report)
    if result["valid"]:
        try:
            check_rms_projection_source(program, projection_plan, projection_bundle, projection_summary, plan)
            result["source_tensor_links_valid"] = True
        except (KeyError, TypeError, ValueError, RuntimeError, AttributeError, StopIteration) as error:
            result.update({"valid": False, "source_tensor_links_valid": False, "reason": str(error)})
    if args.summary:
        result["summary_matches_source"] = canonical_json(json.loads(args.summary.read_text(encoding="utf-8"))) == canonical_json(rms_slice_summary(plan, report))
        result["valid"] = result["valid"] and result["summary_matches_source"]
    if args.reexecute and result["valid"]:
        model = load_model()
        recomputed_plan, recomputed_bundle = build_rms_slice(program, model, projection_plan, projection_bundle, projection_summary)
        recomputed = canonical_json(recomputed_plan) == canonical_json(plan) and canonical_json(recomputed_bundle) == canonical_json(bundle)
        replay = acquire_rms_slice(program, model, projection_plan, projection_bundle, projection_summary, plan, bundle) if recomputed else None
        exact = canonical_json(replay) == canonical_json(report)
        result.update({"mode": "arithmetic_and_cuda_replay", "predictions_recomputed_exact": recomputed, "reexecution_exact": exact, "valid": recomputed and exact})
        if result["valid"]:
            controls = rms_observer_controls(program, model, projection_plan, projection_bundle, projection_summary, report)
            result["observer_controls"] = controls
            result["valid"] = all(all(check.values()) for check in controls.values())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


def command_projection_slice(args: argparse.Namespace) -> int:
    from .gemma_ir_interpreter import build_projection_slice, acquire_projection_slice, verify_projection_slice, projection_slice_summary
    from .serialization import canonical_json

    sources = [args.program, args.fixture, args.query_evidence, args.split_evidence]
    inputs = sources + ([args.plan, args.bundle] if args.operation != "plan" else []) + ([args.report] if args.operation == "verify" else [])
    outputs = [args.output, args.bundle] if args.operation == "plan" else [args.output] + ([args.summary] if args.summary else []) if args.operation == "run" else []
    if len({path.resolve() for path in outputs}) != len(outputs) or {path.resolve() for path in outputs}.intersection(path.resolve() for path in inputs):
        raise ValueError("Projection outputs must be distinct from source evidence and each other")
    if args.operation == "verify" and args.summary and args.summary.resolve() in {path.resolve() for path in inputs}:
        raise ValueError("Expected projection summary must be distinct from its source files")
    program, fixture, query_evidence, split_evidence = [json.loads(path.read_text(encoding="utf-8")) for path in sources]

    def load_model() -> Any:
        import torch
        from transformers import AutoModelForCausalLM

        if not torch.cuda.is_available():
            raise RuntimeError("Actual projection comparison requires CUDA")
        return AutoModelForCausalLM.from_pretrained(str(args.model_path), local_files_only=True, torch_dtype=torch.bfloat16,
                                                  attn_implementation="eager").to("cuda").eval()

    if args.operation == "plan":
        plan, bundle = build_projection_slice(program, load_model(), fixture, query_evidence, split_evidence)
        args.bundle.parent.mkdir(parents=True, exist_ok=True)
        args.bundle.write_text(json.dumps(bundle, sort_keys=True) + "\n", encoding="utf-8")
        _write_json(args.output, plan)
        return 0
    plan, bundle = [json.loads(path.read_text(encoding="utf-8")) for path in (args.plan, args.bundle)]
    if canonical_json(plan["input_token_ids"]) != canonical_json(fixture["input_token_ids"]) or plan["input_fixture_sha256"] != hashlib.sha256(canonical_json(fixture).encode("utf-8")).hexdigest() or any(
        plan["projection_records"][role]["source_evidence_sha256"] != evidence["report_sha256"] for role, evidence in (("query", query_evidence), ("key", split_evidence), ("value", split_evidence))
    ):
        raise ValueError("Projection source references differ from the prediction plan")
    if args.operation == "run":
        report = acquire_projection_slice(program, load_model(), plan, bundle)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_json(args.summary, projection_slice_summary(plan, report))
        return 0 if report["slice_passes_declared_comparison"] else 1
    report = json.loads(args.report.read_text(encoding="utf-8"))
    result = verify_projection_slice(program, plan, bundle, report)
    if args.summary:
        result["summary_matches_source"] = canonical_json(json.loads(args.summary.read_text(encoding="utf-8"))) == canonical_json(projection_slice_summary(plan, report))
        result["valid"] = result["valid"] and result["summary_matches_source"]
    if args.reexecute and result["valid"]:
        model = load_model()
        predicted_plan, predicted_bundle = build_projection_slice(program, model, fixture, query_evidence, split_evidence)
        recomputed = canonical_json(predicted_plan) == canonical_json(plan) and canonical_json(predicted_bundle) == canonical_json(bundle)
        replay = acquire_projection_slice(program, model, predicted_plan, predicted_bundle) if recomputed else None
        result.update({"mode": "independent_arithmetic_and_cuda_replay", "predictions_recomputed_exact": recomputed,
                       "reexecution_exact": canonical_json(replay) == canonical_json(report),
                       "valid": recomputed and canonical_json(replay) == canonical_json(report)})
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


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


def command_gemma_reduction_backend(args: argparse.Namespace) -> int:
    from .gemma_reduction_backend import (
        build_gemma_reduction_backend_binding,
        verify_gemma_reduction_backend_binding,
    )

    program = json.loads(args.program.read_text(encoding="utf-8"))
    reduction = json.loads(args.reduction.read_text(encoding="utf-8"))
    suite = json.loads(args.nsight_suite.read_text(encoding="utf-8"))
    binding = build_gemma_reduction_backend_binding(
        program,
        reduction,
        suite,
        args.sequence_length,
        args.max_tensor_elements,
    )
    verification = verify_gemma_reduction_backend_binding(
        program, reduction, suite, binding
    )
    _write_json(args.output, binding)
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if verification["valid"] else 1


def command_gemma_reduction_backend_verify(args: argparse.Namespace) -> int:
    from .gemma_reduction_backend import verify_gemma_reduction_backend_binding

    program = json.loads(args.program.read_text(encoding="utf-8"))
    reduction = json.loads(args.reduction.read_text(encoding="utf-8"))
    suite = json.loads(args.nsight_suite.read_text(encoding="utf-8"))
    binding = json.loads(args.binding.read_text(encoding="utf-8"))
    verification = verify_gemma_reduction_backend_binding(
        program, reduction, suite, binding
    )
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if verification["valid"] else 1


def command_gemma_wmma_probe(args: argparse.Namespace) -> int:
    from .gemma_wmma_probe import (
        build_wmma_probe_certificate,
        verify_wmma_probe_certificate,
    )

    program = json.loads(args.program.read_text(encoding="utf-8"))
    reduction = json.loads(args.reduction.read_text(encoding="utf-8"))
    suite = json.loads(args.nsight_suite.read_text(encoding="utf-8"))
    backend = json.loads(args.backend.read_text(encoding="utf-8"))
    certificate = build_wmma_probe_certificate(
        program, reduction, suite, backend
    )
    verification = verify_wmma_probe_certificate(
        program, reduction, suite, backend, certificate
    )
    if not verification["valid"]:
        print(json.dumps(verification, indent=2, sort_keys=True))
        return 1
    _write_json(args.output, certificate)
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0


def command_gemma_wmma_probe_verify(args: argparse.Namespace) -> int:
    from .gemma_wmma_probe import verify_wmma_probe_certificate

    program = json.loads(args.program.read_text(encoding="utf-8"))
    reduction = json.loads(args.reduction.read_text(encoding="utf-8"))
    suite = json.loads(args.nsight_suite.read_text(encoding="utf-8"))
    backend = json.loads(args.backend.read_text(encoding="utf-8"))
    certificate = json.loads(args.certificate.read_text(encoding="utf-8"))
    verification = verify_wmma_probe_certificate(
        program,
        reduction,
        suite,
        backend,
        certificate,
        reexecute=args.reexecute,
    )
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if verification["valid"] else 1


def command_gemma_wmma_accumulator_probe(args: argparse.Namespace) -> int:
    from .gemma_wmma_accumulator_probe import (
        build_wmma_accumulator_probe_certificate,
        verify_wmma_accumulator_probe_certificate,
    )

    program = json.loads(args.program.read_text(encoding="utf-8"))
    reduction = json.loads(args.reduction.read_text(encoding="utf-8"))
    suite = json.loads(args.nsight_suite.read_text(encoding="utf-8"))
    backend = json.loads(args.backend.read_text(encoding="utf-8"))
    certificate = build_wmma_accumulator_probe_certificate(
        program, reduction, suite, backend
    )
    verification = verify_wmma_accumulator_probe_certificate(
        program, reduction, suite, backend, certificate
    )
    if not verification["valid"]:
        print(json.dumps(verification, indent=2, sort_keys=True))
        return 1
    _write_json(args.output, certificate)
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0


def command_gemma_wmma_accumulator_probe_verify(args: argparse.Namespace) -> int:
    from .gemma_wmma_accumulator_probe import (
        verify_wmma_accumulator_probe_certificate,
    )

    program = json.loads(args.program.read_text(encoding="utf-8"))
    reduction = json.loads(args.reduction.read_text(encoding="utf-8"))
    suite = json.loads(args.nsight_suite.read_text(encoding="utf-8"))
    backend = json.loads(args.backend.read_text(encoding="utf-8"))
    certificate = json.loads(args.certificate.read_text(encoding="utf-8"))
    verification = verify_wmma_accumulator_probe_certificate(
        program,
        reduction,
        suite,
        backend,
        certificate,
        reexecute=args.reexecute,
    )
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if verification["valid"] else 1


def command_gemma_wmma_magnitude_probe(args: argparse.Namespace) -> int:
    from .gemma_wmma_magnitude_probe import (
        build_wmma_magnitude_probe_certificate,
        verify_wmma_magnitude_probe_certificate,
    )

    program = json.loads(args.program.read_text(encoding="utf-8"))
    reduction = json.loads(args.reduction.read_text(encoding="utf-8"))
    suite = json.loads(args.nsight_suite.read_text(encoding="utf-8"))
    backend = json.loads(args.backend.read_text(encoding="utf-8"))
    certificate = build_wmma_magnitude_probe_certificate(
        program, reduction, suite, backend
    )
    verification = verify_wmma_magnitude_probe_certificate(
        program, reduction, suite, backend, certificate
    )
    if not verification["valid"]:
        print(json.dumps(verification, indent=2, sort_keys=True))
        return 1
    _write_json(args.output, certificate)
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0


def command_gemma_wmma_magnitude_probe_verify(args: argparse.Namespace) -> int:
    from .gemma_wmma_magnitude_probe import verify_wmma_magnitude_probe_certificate

    program = json.loads(args.program.read_text(encoding="utf-8"))
    reduction = json.loads(args.reduction.read_text(encoding="utf-8"))
    suite = json.loads(args.nsight_suite.read_text(encoding="utf-8"))
    backend = json.loads(args.backend.read_text(encoding="utf-8"))
    certificate = json.loads(args.certificate.read_text(encoding="utf-8"))
    verification = verify_wmma_magnitude_probe_certificate(
        program,
        reduction,
        suite,
        backend,
        certificate,
        reexecute=args.reexecute,
    )
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if verification["valid"] else 1


def command_gemma_wmma_candidate_search(args: argparse.Namespace) -> int:
    from .gemma_wmma_candidate import (
        build_wmma_candidate_search,
        verify_wmma_candidate_search,
    )

    program = json.loads(args.program.read_text(encoding="utf-8"))
    reduction = json.loads(args.reduction.read_text(encoding="utf-8"))
    suite = json.loads(args.nsight_suite.read_text(encoding="utf-8"))
    backend = json.loads(args.backend.read_text(encoding="utf-8"))
    position = json.loads(args.position_probe.read_text(encoding="utf-8"))
    accumulator = json.loads(args.accumulator_probe.read_text(encoding="utf-8"))
    magnitude = json.loads(args.magnitude_probe.read_text(encoding="utf-8"))
    search = build_wmma_candidate_search(
        program,
        reduction,
        suite,
        backend,
        position,
        accumulator,
        magnitude,
    )
    verification = verify_wmma_candidate_search(
        program,
        reduction,
        suite,
        backend,
        position,
        accumulator,
        magnitude,
        search,
    )
    if not verification["valid"]:
        print(json.dumps(verification, indent=2, sort_keys=True))
        return 1
    _write_json(args.output, search)
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0


def command_gemma_wmma_candidate_search_verify(args: argparse.Namespace) -> int:
    from .gemma_wmma_candidate import verify_wmma_candidate_search

    program = json.loads(args.program.read_text(encoding="utf-8"))
    reduction = json.loads(args.reduction.read_text(encoding="utf-8"))
    suite = json.loads(args.nsight_suite.read_text(encoding="utf-8"))
    backend = json.loads(args.backend.read_text(encoding="utf-8"))
    position = json.loads(args.position_probe.read_text(encoding="utf-8"))
    accumulator = json.loads(args.accumulator_probe.read_text(encoding="utf-8"))
    magnitude = json.loads(args.magnitude_probe.read_text(encoding="utf-8"))
    search = json.loads(args.search.read_text(encoding="utf-8"))
    verification = verify_wmma_candidate_search(
        program,
        reduction,
        suite,
        backend,
        position,
        accumulator,
        magnitude,
        search,
    )
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if verification["valid"] else 1


def command_k16_holdout_plan(args: argparse.Namespace) -> int:
    from .gemma_wmma_candidate import build_k16_holdout_plan, build_composition_holdout_plan

    search = json.loads(args.search.read_text(encoding="utf-8"))
    builder = build_composition_holdout_plan if getattr(args, "composition", False) else build_k16_holdout_plan
    _write_json(args.output, builder(search))
    return 0


def command_k16_holdout_run(args: argparse.Namespace) -> int:
    from .gemma_wmma_candidate import acquire_k16_holdout, acquire_composition_holdout

    search = json.loads(args.search.read_text(encoding="utf-8"))
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    acquire = acquire_composition_holdout if getattr(args, "composition", False) else acquire_k16_holdout
    report = acquire(plan, search)
    _write_json(args.output, report)
    return 0 if report["candidate_passes_holdout"] else 1


def command_k16_holdout_verify(args: argparse.Namespace) -> int:
    from .gemma_wmma_candidate import verify_k16_holdout, verify_composition_holdout

    search = json.loads(args.search.read_text(encoding="utf-8"))
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    report = json.loads(args.report.read_text(encoding="utf-8"))
    verify = verify_composition_holdout if getattr(args, "composition", False) else verify_k16_holdout
    result = verify(plan, search, report, args.reexecute)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


def command_carry_revision(args: argparse.Namespace) -> int:
    from .gemma_wmma_candidate import build_carry_revision_plan, acquire_carry_revision_holdout, verify_carry_revision_holdout
    from .gemma_wmma_candidate import build_product_holdout_plan, acquire_product_holdout, verify_product_holdout

    from .gemma_wmma_candidate import build_matched_product_plan, acquire_matched_products, verify_matched_products

    product = getattr(args, "product", False)
    matched = getattr(args, "matched", False)
    builder = build_product_holdout_plan if product else build_carry_revision_plan
    acquire = acquire_product_holdout if product else acquire_carry_revision_holdout
    verify = verify_product_holdout if product else verify_carry_revision_holdout
    if matched:
        builder, acquire, verify = build_matched_product_plan, acquire_matched_products, verify_matched_products
    search = json.loads(args.search.read_text(encoding="utf-8"))
    diagnosis = json.loads(args.diagnosis.read_text(encoding="utf-8"))
    if args.operation == "plan":
        _write_json(args.output, builder(search, diagnosis))
        return 0
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if args.operation == "run":
        report = acquire(plan, search, diagnosis)
        _write_json(args.output, report)
        field = "candidate_passes_controlled_comparison" if matched else "candidate_passes_holdout"
        return 0 if report[field] else 1
    report = json.loads(args.report.read_text(encoding="utf-8"))
    result = verify(plan, search, diagnosis, report, args.reexecute)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


def command_dense_split(args: argparse.Namespace) -> int:
    from .gemma_wmma_candidate import build_dense_split_plan, acquire_dense_split_holdout, verify_dense_split_holdout

    inputs = [args.diagnosis, args.merge_plan, args.merge_report] + ([args.plan] if args.operation != "plan" else [])
    if args.operation != "verify" and args.output.resolve() in {path.resolve() for path in inputs}:
        raise ValueError("Dense split-K output must not overwrite source evidence")
    sources = [json.loads(path.read_text(encoding="utf-8")) for path in inputs[:3]]
    if args.operation == "plan":
        _write_json(args.output, build_dense_split_plan(*sources))
        return 0
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if args.operation == "run":
        report = acquire_dense_split_holdout(plan, *sources)
        _write_json(args.output, report)
        return 0 if report["candidate_passes_within_declared_scope"] else 1
    report = json.loads(args.report.read_text(encoding="utf-8"))
    result = verify_dense_split_holdout(plan, *sources, report, args.reexecute)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


def command_split_merge(args: argparse.Namespace) -> int:
    from .gemma_wmma_candidate import build_split_merge_plan, acquire_split_merge_holdout, verify_split_merge_holdout

    inputs = [args.diagnosis] + ([args.plan] if args.operation != "plan" else [])
    if args.operation != "verify" and args.output.resolve() in {path.resolve() for path in inputs}:
        raise ValueError("Split-K output must not overwrite source evidence")
    diagnosis = json.loads(args.diagnosis.read_text(encoding="utf-8"))
    if args.operation == "plan":
        _write_json(args.output, build_split_merge_plan(diagnosis))
        return 0
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if args.operation == "run":
        report = acquire_split_merge_holdout(plan, diagnosis)
        _write_json(args.output, report)
        return 0 if report["any_candidate_passes_within_declared_scope"] else 1
    report = json.loads(args.report.read_text(encoding="utf-8"))
    result = verify_split_merge_holdout(plan, diagnosis, report, args.reexecute)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


def command_split_k_diagnosis(args: argparse.Namespace) -> int:
    from .gemma_wmma_candidate import diagnose_split_k

    paths = [args.search, args.carry_diagnosis, args.product_plan, args.product_report, args.matched_plan, args.matched_report]
    if args.output and args.output.resolve() in {path.resolve() for path in paths}:
        raise ValueError("Split-K diagnosis output must not overwrite source evidence")
    expected = diagnose_split_k(*(json.loads(path.read_text(encoding="utf-8")) for path in paths))
    if args.diagnosis:
        actual = json.loads(args.diagnosis.read_text(encoding="utf-8"))
        valid = json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)
        print(json.dumps({"valid": valid, "mode": "software_recomputation", "candidate_count": expected["candidate_count"],
                          "minimum_mismatch_count": expected["minimum_mismatch_count"], "best_candidates": expected["best_candidates"],
                          "all_matching_candidates": expected["all_matching_candidates"], "fresh_holdout_validation_established": False}, indent=2))
        return 0 if valid else 1
    _write_json(args.output, expected)
    return 0


def command_wide_query(args: argparse.Namespace) -> int:
    from .gemma_wmma_candidate import build_wide_query_plan, acquire_wide_query_holdout, verify_wide_query_holdout

    inputs = [args.source_plan, args.source_report]
    if args.operation != "plan":
        inputs.append(args.plan)
    if args.operation != "verify" and args.output.resolve() in {path.resolve() for path in inputs}:
        raise ValueError("Wide query output must not overwrite source evidence")
    source_plan, source_report = [json.loads(path.read_text(encoding="utf-8")) for path in inputs[:2]]
    if args.operation == "plan":
        _write_json(args.output, build_wide_query_plan(source_plan, source_report))
        return 0
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if args.operation == "run":
        report = acquire_wide_query_holdout(plan, source_plan, source_report)
        _write_json(args.output, report)
        return 0 if report["candidate_passes_within_declared_scope"] else 1
    report = json.loads(args.report.read_text(encoding="utf-8"))
    result = verify_wide_query_holdout(plan, source_plan, source_report, report, args.reexecute)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


def command_operand_alignment_holdout(args: argparse.Namespace) -> int:
    from .gemma_wmma_candidate import build_operand_alignment_holdout_plan, acquire_operand_alignment_holdout, verify_operand_alignment_holdout

    inputs = [args.diagnosis, args.source_report]
    if args.operation != "plan":
        inputs.append(args.plan)
    if args.operation != "verify" and args.output.resolve() in {path.resolve() for path in inputs}:
        raise ValueError("Alignment output must not overwrite source evidence")
    diagnosis, source_report = [json.loads(path.read_text(encoding="utf-8")) for path in inputs[:2]]
    if args.operation == "plan":
        _write_json(args.output, build_operand_alignment_holdout_plan(diagnosis, source_report))
        return 0
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if args.operation == "run":
        report = acquire_operand_alignment_holdout(plan, diagnosis, source_report)
        _write_json(args.output, report)
        return 0 if report["candidate_passes_within_declared_scope"] else 1
    report = json.loads(args.report.read_text(encoding="utf-8"))
    result = verify_operand_alignment_holdout(plan, diagnosis, source_report, report, args.reexecute)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


def command_operand_alignment_diagnosis(args: argparse.Namespace) -> int:
    from .gemma_wmma_candidate import diagnose_operand_alignment

    paths = [args.search, args.carry_diagnosis, args.source_plan, args.source_report, args.reduction_plan, args.reduction_report]
    if args.output and args.output.resolve() in {path.resolve() for path in paths}:
        raise ValueError("Diagnosis output must not overwrite source evidence")
    expected = diagnose_operand_alignment(*(json.loads(path.read_text(encoding="utf-8")) for path in paths))
    if args.diagnosis:
        actual = json.loads(args.diagnosis.read_text(encoding="utf-8"))
        valid = json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)
        print(json.dumps({"valid": valid, "mode": "software_recomputation", "phase_summaries": expected["phase_summaries"], "fresh_holdout_validation_established": False}, indent=2))
        return 0 if valid else 1
    _write_json(args.output, expected)
    return 0


def command_query_reduction(args: argparse.Namespace) -> int:
    from .gemma_wmma_candidate import build_query_reduction_plan, acquire_query_reduction, verify_query_reduction

    inputs = [args.search, args.diagnosis, args.source_plan, args.source_report]
    if args.operation != "plan":
        inputs.append(args.plan)
    if args.operation != "verify" and args.output.resolve() in {path.resolve() for path in inputs}:
        raise ValueError("Reduction output must not overwrite its source evidence")
    sources = [json.loads(path.read_text(encoding="utf-8")) for path in inputs[:4]]
    if args.operation == "plan":
        _write_json(args.output, build_query_reduction_plan(*sources))
        return 0
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if args.operation == "run":
        protected = {path.resolve() for path in (args.search, args.diagnosis, args.source_plan, args.source_report, args.plan, args.output)}
        temporary = args.journal.with_suffix(".tmp")
        if args.journal.resolve() in protected or temporary.resolve() in protected or temporary == args.journal:
            raise ValueError("Journal and temporary paths must be distinct from experiment inputs and output")
        args.journal.parent.mkdir(parents=True, exist_ok=True)

        def persist(pending: dict[str, Any]) -> None:
            temporary.write_text(json.dumps(pending, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            temporary.replace(args.journal)

        report = acquire_query_reduction(plan, *sources, persist)
        _write_json(args.output, report)
        return 0
    report = json.loads(args.report.read_text(encoding="utf-8"))
    result = verify_query_reduction(plan, *sources, report, args.reexecute)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


def command_carry_diagnosis(args: argparse.Namespace) -> int:
    from .gemma_wmma_candidate import diagnose_float32_carry, verify_carry_diagnosis

    search = json.loads(args.search.read_text(encoding="utf-8"))
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if args.diagnosis:
        diagnosis = json.loads(args.diagnosis.read_text(encoding="utf-8"))
        result = verify_carry_diagnosis(search, plan, report, diagnosis)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["valid"] else 1
    _write_json(args.output, diagnose_float32_carry(search, plan, report))
    return 0


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
    reduction_backend = json.loads(
        args.reduction_backend.read_text(encoding="utf-8")
    )
    nsight_suite = json.loads(args.nsight_suite.read_text(encoding="utf-8"))
    wmma_probe = json.loads(args.wmma_probe.read_text(encoding="utf-8"))
    wmma_accumulator_probe = json.loads(
        args.wmma_accumulator_probe.read_text(encoding="utf-8")
    )
    wmma_magnitude_probe = json.loads(
        args.wmma_magnitude_probe.read_text(encoding="utf-8")
    )
    wmma_candidate_search = json.loads(
        args.wmma_candidate_search.read_text(encoding="utf-8")
    )
    gate = build_primitive_qualification_gate(
        program,
        certificate,
        bfloat16_certificate,
        reduction_certificate,
        reduction_backend,
        nsight_suite,
        wmma_probe,
        wmma_accumulator_probe,
        wmma_magnitude_probe,
        wmma_candidate_search,
    )
    verification = verify_primitive_qualification_gate(
        program,
        certificate,
        bfloat16_certificate,
        reduction_certificate,
        reduction_backend,
        nsight_suite,
        wmma_probe,
        wmma_accumulator_probe,
        wmma_magnitude_probe,
        wmma_candidate_search,
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
    reduction_backend = json.loads(
        args.reduction_backend.read_text(encoding="utf-8")
    )
    nsight_suite = json.loads(args.nsight_suite.read_text(encoding="utf-8"))
    wmma_probe = json.loads(args.wmma_probe.read_text(encoding="utf-8"))
    wmma_accumulator_probe = json.loads(
        args.wmma_accumulator_probe.read_text(encoding="utf-8")
    )
    wmma_magnitude_probe = json.loads(
        args.wmma_magnitude_probe.read_text(encoding="utf-8")
    )
    wmma_candidate_search = json.loads(
        args.wmma_candidate_search.read_text(encoding="utf-8")
    )
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    verification = verify_primitive_qualification_gate(
        program,
        certificate,
        bfloat16_certificate,
        reduction_certificate,
        reduction_backend,
        nsight_suite,
        wmma_probe,
        wmma_accumulator_probe,
        wmma_magnitude_probe,
        wmma_candidate_search,
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

    for operation in ("plan", "run", "verify"):
        projection_slice = subparsers.add_parser(f"gemma-projection-slice-{operation}", help=f"{operation.title()} actual layer-0 Q/K/V arithmetic with an explicitly shared prefix")
        for name in ("program", "fixture", "query_evidence", "split_evidence"):
            projection_slice.add_argument(name, type=Path)
        if operation != "plan":
            projection_slice.add_argument("plan", type=Path)
        projection_slice.add_argument("--bundle", type=Path, required=True, help="Tensor-rich prediction JSON; keep outside version control")
        projection_slice.add_argument("--model-path", type=Path, default=Path(".models/gemma-3-270m-it"))
        if operation == "verify":
            projection_slice.add_argument("report", type=Path)
            projection_slice.add_argument("--reexecute", action="store_true")
        else:
            projection_slice.add_argument("--output", type=Path, required=True, help="Tensor-rich observation report; keep outside version control" if operation == "run" else "Compact prediction plan")
        projection_slice.add_argument("--summary", type=Path, help="Compact output for run; expected summary for verify")
        projection_slice.set_defaults(handler=command_projection_slice, operation=operation)

    for operation in ("plan", "run", "verify"):
        rms_slice = subparsers.add_parser(f"gemma-rms-slice-{operation}", help=f"{operation.title()} actual layer-0 RMS arithmetic candidates and observed ATen stages")
        for name in ("program", "projection_plan", "projection_bundle", "projection_summary"):
            rms_slice.add_argument(name, type=Path)
        if operation != "plan":
            rms_slice.add_argument("plan", type=Path)
        rms_slice.add_argument("--bundle", type=Path, required=True, help="Tensor-rich RMS predictions; keep outside version control")
        rms_slice.add_argument("--model-path", type=Path, default=Path(".models/gemma-3-270m-it"))
        if operation == "verify":
            rms_slice.add_argument("report", type=Path)
            rms_slice.add_argument("--reexecute", action="store_true")
        else:
            rms_slice.add_argument("--output", type=Path, required=True, help="Tensor-rich report for run; compact plan for plan")
        rms_slice.add_argument("--summary", type=Path)
        rms_slice.set_defaults(handler=command_rms_slice, operation=operation)

    for operation in ("table-plan", "table-build", "table-verify", "domain-plan", "domain-run", "domain-verify", "rms-plan", "rms-run", "rms-verify"):
        rsqrt_lookup = subparsers.add_parser(f"gemma-rsqrt-{operation}", help="Build or verify an explicit runtime-bound rsqrt lookup specification")
        if operation == "table-plan":
            rsqrt_lookup.add_argument("rms_summary", type=Path)
        else:
            rsqrt_lookup.add_argument("table_plan", type=Path)
            rsqrt_lookup.add_argument("--table", type=Path, required=True, help="64 MiB binary table; keep under ignored artifacts/")
            if operation != "table-build":
                rsqrt_lookup.add_argument("manifest", type=Path)
            if operation.startswith("rms-"):
                rsqrt_lookup.add_argument("domain_plan", type=Path)
                rsqrt_lookup.add_argument("domain_report", type=Path)
                rsqrt_lookup.add_argument("--program", type=Path, default=Path("results/gemma3_270m_execution_ir.json"))
                rsqrt_lookup.add_argument("--rms-plan", type=Path, default=Path("results/gemma3_270m_rms_slice_plan.json"))
                rsqrt_lookup.add_argument("--rms-bundle", type=Path, default=Path("artifacts/gemma3_270m_rms_slice_predictions.json"))
                rsqrt_lookup.add_argument("--rms-report", type=Path, default=Path("artifacts/gemma3_270m_rms_slice_report.json"))
                rsqrt_lookup.add_argument("--bundle", type=Path, required=True, help="Ignored tensor-rich lookup RMS prediction bundle")
            if operation in ("domain-run", "domain-verify", "rms-run", "rms-verify"):
                rsqrt_lookup.add_argument("plan", type=Path)
            if operation in ("domain-verify", "rms-verify"):
                rsqrt_lookup.add_argument("report", type=Path)
                rsqrt_lookup.add_argument("--reexecute", action="store_true")
            if operation in ("table-build", "domain-run", "domain-verify", "rms-plan", "rms-run", "rms-verify"):
                rsqrt_lookup.add_argument("--audit-dir", type=Path, required=True, help="Ignored mismatch-payload directory; also needed when validating source domain evidence")
            if operation == "domain-run":
                rsqrt_lookup.add_argument("--journal", type=Path, required=True)
        if operation not in ("table-verify", "domain-verify", "rms-verify"):
            rsqrt_lookup.add_argument("--output", type=Path, required=True)
        rsqrt_lookup.set_defaults(handler=command_rsqrt_lookup, operation=operation)

    for operation in ("plan", "run", "verify"):
        attention_entry = subparsers.add_parser(f"gemma-attention-entry-{operation}", help=f"{operation.title()} connected independent layer-0 pre-RoPE execution")
        attention_entry.add_argument("program", type=Path)
        attention_entry.add_argument("fixture", type=Path)
        if operation != "plan":
            attention_entry.add_argument("plan", type=Path)
        attention_entry.add_argument("--bundle", type=Path, required=True, help="Tensor-rich execution/prediction bundle; keep in Git-ignored artifacts/")
        attention_entry.add_argument("--table-plan", type=Path, default=Path("results/gemma3_270m_rsqrt_table_plan.json"))
        attention_entry.add_argument("--table-manifest", type=Path, default=Path("results/gemma3_270m_rsqrt_table_manifest.json"))
        attention_entry.add_argument("--domain-plan", type=Path, default=Path("results/gemma3_270m_rsqrt_domain_plan.json"))
        attention_entry.add_argument("--domain-report", type=Path, default=Path("results/gemma3_270m_rsqrt_domain_report.json"))
        attention_entry.add_argument("--table", type=Path, default=Path("artifacts/gemma3_270m_rsqrt_table.bin"))
        attention_entry.add_argument("--audit-dir", type=Path, default=Path("artifacts/rsqrt_mismatches"))
        attention_entry.add_argument("--model-path", type=Path, default=Path(".models/gemma-3-270m-it"))
        attention_entry.add_argument("--summary", type=Path)
        if operation == "verify":
            attention_entry.add_argument("report", type=Path)
            attention_entry.add_argument("--reexecute", action="store_true")
        else:
            attention_entry.add_argument("--output", type=Path, required=True, help="Compact plan for plan; tensor-rich report for run (keep in Git-ignored artifacts/)")
        attention_entry.set_defaults(handler=command_attention_entry, operation=operation)

    for operation in ("table-plan", "table-run", "table-verify", "plan", "run", "verify"):
        rotary = subparsers.add_parser(f"gemma-rotary-{operation}", help=f"{operation} fixed-position rotary specification or connected RoPE slice")
        rotary.add_argument("program", type=Path)
        rotary.add_argument("--fixture", type=Path, default=Path("results/gemma3_270m_ir_execution_summary.json"))
        rotary.add_argument("--model-path", type=Path, default=Path(".models/gemma-3-270m-it"))
        rotary.add_argument("--table-plan", type=Path, default=Path("results/gemma3_270m_rotary_table_plan.json"))
        rotary.add_argument("--table-manifest", type=Path, default=Path("results/gemma3_270m_rotary_table_manifest.json"))
        rotary.add_argument("--table-bundle", type=Path, default=Path("artifacts/gemma3_270m_rotary_table.json"))
        rotary.add_argument("--plan", type=Path, default=Path("results/gemma3_270m_rotary_slice_plan_v2.json"))
        rotary.add_argument("--bundle", type=Path, required=operation in ("table-run", "plan", "run", "verify"), help="Tensor-rich bundle; keep under Git-ignored artifacts/")
        rotary.add_argument("--report", type=Path, required=operation == "verify")
        rotary.add_argument("--summary", type=Path)
        rotary.add_argument("--rsqrt-table", type=Path, default=Path("artifacts/gemma3_270m_rsqrt_table.bin"))
        rotary.add_argument("--rsqrt-table-plan", type=Path, default=Path("results/gemma3_270m_rsqrt_table_plan.json"))
        rotary.add_argument("--rsqrt-table-manifest", type=Path, default=Path("results/gemma3_270m_rsqrt_table_manifest.json"))
        rotary.add_argument("--rsqrt-domain-plan", type=Path, default=Path("results/gemma3_270m_rsqrt_domain_plan.json"))
        rotary.add_argument("--rsqrt-domain-report", type=Path, default=Path("results/gemma3_270m_rsqrt_domain_report.json"))
        rotary.add_argument("--rsqrt-audit-dir", type=Path, default=Path("artifacts/rsqrt_mismatches"))
        if operation.endswith("verify"):
            rotary.add_argument("--reexecute", action="store_true")
        else:
            rotary.add_argument("--output", type=Path, required=True, help="New compact plan/manifest or tensor-rich run report (under artifacts/)")
        rotary.set_defaults(handler=command_rotary_slice, operation=operation)

    for operation in ("plan", "run", "verify"):
        scores = subparsers.add_parser(f"gemma-attention-scores-{operation}", help=f"{operation.title()} frozen QK score/scaling/mask characterization before softmax")
        scores.add_argument("program", type=Path)
        scores.add_argument("--model-path", type=Path, default=Path(".models/gemma-3-270m-it"))
        scores.add_argument("--rotary-plan", type=Path, default=Path("results/gemma3_270m_rotary_slice_plan_v2.json"))
        scores.add_argument("--rotary-bundle", type=Path, default=Path("artifacts/gemma3_270m_rotary_slice_predictions_v2.json"))
        scores.add_argument("--rotary-report", type=Path, default=Path("artifacts/gemma3_270m_rotary_slice_report.json"))
        scores.add_argument("--table-plan", type=Path, default=Path("results/gemma3_270m_rotary_table_plan.json"))
        scores.add_argument("--table-manifest", type=Path, default=Path("results/gemma3_270m_rotary_table_manifest.json"))
        scores.add_argument("--table-bundle", type=Path, default=Path("artifacts/gemma3_270m_rotary_table.json"))
        scores.add_argument("--rsqrt-table", type=Path, default=Path("artifacts/gemma3_270m_rsqrt_table.bin"))
        scores.add_argument("--rsqrt-table-plan", type=Path, default=Path("results/gemma3_270m_rsqrt_table_plan.json"))
        scores.add_argument("--rsqrt-table-manifest", type=Path, default=Path("results/gemma3_270m_rsqrt_table_manifest.json"))
        scores.add_argument("--rsqrt-domain-plan", type=Path, default=Path("results/gemma3_270m_rsqrt_domain_plan.json"))
        scores.add_argument("--rsqrt-domain-report", type=Path, default=Path("results/gemma3_270m_rsqrt_domain_report.json"))
        scores.add_argument("--rsqrt-audit-dir", type=Path, default=Path("artifacts/rsqrt_mismatches"))
        scores.add_argument("--plan", type=Path, default=Path("results/gemma3_270m_attention_scores_plan.json"))
        scores.add_argument("--bundle", type=Path, required=True, help="Private prediction bundle under Git-ignored artifacts/")
        scores.add_argument("--summary", type=Path)
        if operation == "verify":
            scores.add_argument("--report", type=Path, required=True)
            scores.add_argument("--reexecute", action="store_true")
        else:
            scores.add_argument("--output", type=Path, required=True, help="Compact plan or tensor-rich run report (keep reports under artifacts/)")
        scores.set_defaults(handler=command_attention_scores, operation=operation)

    for command in ("plan", "run", "verify", "lookup-plan", "lookup-run", "lookup-verify", "output-plan", "output-run", "output-verify", "output-split-plan", "output-split-run", "output-split-verify", "output-survivor-plan", "output-survivor-run", "output-survivor-verify", "post-plan", "post-run", "post-verify", "mlp-plan", "mlp-run", "mlp-verify", "product-plan", "product-run", "product-verify", "down-plan", "down-run", "down-verify", "k2048-dense-plan", "k2048-dense-run", "k2048-dense-verify", "post-ff-plan", "post-ff-run", "post-ff-verify", "first-layer-plan", "first-layer-run", "first-layer-verify", "first-layer-holdout-protocol", "first-layer-holdout-plan", "first-layer-holdout-run", "first-layer-holdout-verify", "second-layer-plan", "second-layer-run", "second-layer-verify", "two-layers-plan", "two-layers-run", "two-layers-verify", "two-layers-holdout-protocol", "two-layers-holdout-plan", "two-layers-holdout-run", "two-layers-holdout-verify", "third-entry-plan", "third-entry-run", "third-entry-verify", "third-scores-plan", "third-scores-run", "third-scores-verify"):
        third_scores_mode = command.startswith("third-scores-")
        third_entry_mode = command.startswith("third-entry-") or third_scores_mode
        two_layers_holdout_mode = command.startswith("two-layers-holdout-") or third_entry_mode
        two_layers_mode = command.startswith("two-layers-") or third_entry_mode
        second_layer_mode = command.startswith("second-layer-") or two_layers_mode
        holdout_mode = command.startswith("first-layer-holdout-")
        first_layer_mode = command.startswith("first-layer-") or holdout_mode or second_layer_mode
        post_ff_mode = command.startswith("post-ff-") or first_layer_mode
        dense_k2048_mode = command.startswith("k2048-dense-") or post_ff_mode
        down_mode = command.startswith("down-") or dense_k2048_mode
        product_mode = command.startswith("product-") or down_mode
        mlp_mode = command.startswith("mlp-") or product_mode
        post_mode = command.startswith("post-") or mlp_mode
        survivor_mode = command.startswith("output-survivor-") or post_mode
        split_mode = command.startswith("output-split-")
        output_mode = command.startswith("output-") or post_mode
        lookup_mode = command.startswith("lookup-") or output_mode
        operation = command.split("-")[-1]
        command_name = f"gemma-third-layer-scores-{operation}" if third_scores_mode else f"gemma-third-layer-entry-{operation}" if third_entry_mode else f"gemma-two-layers-holdout-{operation}" if two_layers_holdout_mode else f"gemma-two-layers-{operation}" if two_layers_mode else f"gemma-second-layer-{operation}" if second_layer_mode else f"gemma-first-layer-holdout-{operation}" if holdout_mode else f"gemma-first-layer-{operation}" if first_layer_mode else f"gemma-post-feedforward-{operation}" if post_ff_mode else f"gemma-k2048-dense-{operation}" if dense_k2048_mode else f"gemma-mlp-down-{operation}" if down_mode else f"gemma-mlp-product-{operation}" if product_mode else f"gemma-mlp-entry-{operation}" if mlp_mode else f"gemma-post-attention-{operation}" if post_mode else f"gemma-attention-{command}" if output_mode else f"gemma-softmax-{command}"
        softmax = subparsers.add_parser(command_name, help=f"{command.title()} frozen numerical specification and original-forward comparison")
        softmax.add_argument("program", type=Path)
        softmax.add_argument("--model-path", type=Path, default=Path(".models/gemma-3-270m-it"))
        for flag, default in (
            ("rotary-plan", "results/gemma3_270m_rotary_slice_plan_v2.json"),
            ("rotary-bundle", "artifacts/gemma3_270m_rotary_slice_predictions_v2.json"),
            ("rotary-report", "artifacts/gemma3_270m_rotary_slice_report.json"),
            ("table-plan", "results/gemma3_270m_rotary_table_plan.json"),
            ("table-manifest", "results/gemma3_270m_rotary_table_manifest.json"),
            ("table-bundle", "artifacts/gemma3_270m_rotary_table.json"),
            ("rsqrt-table", "artifacts/gemma3_270m_rsqrt_table.bin"),
            ("rsqrt-table-plan", "results/gemma3_270m_rsqrt_table_plan.json"),
            ("rsqrt-table-manifest", "results/gemma3_270m_rsqrt_table_manifest.json"),
            ("rsqrt-domain-plan", "results/gemma3_270m_rsqrt_domain_plan.json"),
            ("rsqrt-domain-report", "results/gemma3_270m_rsqrt_domain_report.json"),
            ("rsqrt-audit-dir", "artifacts/rsqrt_mismatches"),
            ("score-plan", "results/gemma3_270m_attention_scores_plan.json"),
            ("score-bundle", "artifacts/gemma3_270m_attention_scores_predictions.json"),
            ("score-report", "artifacts/gemma3_270m_attention_scores_report.json"),
        ):
            softmax.add_argument("--" + flag, type=Path, default=Path(default))
        if lookup_mode:
            for flag, default in (("exp-plan", "results/gemma3_270m_exp_plan.json"), ("exp-manifest", "results/gemma3_270m_exp_manifest.json"),
                                  ("exp-table", "artifacts/gemma3_270m_exp_table.bin"), ("exp-audit-dir", "artifacts/exp_mismatches"),
                                  ("original-plan", "results/gemma3_270m_softmax_plan.json"), ("original-bundle", "artifacts/gemma3_270m_softmax_predictions.json"),
                                  ("original-report", "artifacts/gemma3_270m_softmax_report.json")):
                softmax.add_argument("--" + flag, type=Path, default=Path(default))
        if output_mode:
            for flag, default in (("lookup-plan", "results/gemma3_270m_softmax_lookup_plan.json"), ("lookup-bundle", "artifacts/gemma3_270m_softmax_lookup_predictions.json"), ("lookup-report", "artifacts/gemma3_270m_softmax_lookup_report.json")):
                softmax.add_argument("--" + flag, type=Path, default=Path(default))
        if split_mode or survivor_mode:
            for flag, default in (("base-plan", "results/gemma3_270m_attention_output_plan.json"), ("base-bundle", "artifacts/gemma3_270m_attention_output_predictions.json"), ("base-report", "artifacts/gemma3_270m_attention_output_report.json")):
                softmax.add_argument("--" + flag, type=Path, default=Path(default))
        if survivor_mode:
            for flag, default in (("probe-source", "results/gemma3_270m_attention_output_summary.json"), ("probe-plan", "results/gemma3_270m_k1024_probes_plan.json"), ("probe-bundle", "artifacts/gemma3_270m_k1024_probes_inputs.json"), ("probe-report", "artifacts/gemma3_270m_k1024_probes_report.json")):
                inherited_flag = "prefix-" + flag if down_mode and flag != "probe-source" else flag
                softmax.add_argument("--" + inherited_flag, type=Path, default=Path(default))
        if post_mode:
            for flag, default in (("survivor-plan", "results/gemma3_270m_output_survivor_plan.json"), ("survivor-bundle", "artifacts/gemma3_270m_output_survivor_predictions.json"), ("survivor-report", "artifacts/gemma3_270m_output_survivor_report.json"),
                                  ("dense-plan", "results/gemma3_270m_k128_dense_plan.json"), ("dense-bundle", "artifacts/gemma3_270m_k128_dense_inputs.json"), ("dense-report", "artifacts/gemma3_270m_k128_dense_report.json")):
                softmax.add_argument("--" + flag, type=Path, default=Path(default))
        if mlp_mode:
            for flag, default in (("post-plan", "results/gemma3_270m_post_attention_plan.json"), ("post-bundle", "artifacts/gemma3_270m_post_attention_predictions.json"), ("post-report", "artifacts/gemma3_270m_post_attention_report.json")):
                softmax.add_argument("--" + flag, type=Path, default=Path(default))
            softmax.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=4, help="Deterministic CPU projection workers")
        if product_mode:
            for flag, default in (("entry-plan", "results/gemma3_270m_mlp_entry_plan.json"), ("entry-bundle", "artifacts/gemma3_270m_mlp_entry_predictions.json"), ("entry-report", "artifacts/gemma3_270m_mlp_entry_report.json"),
                                  ("gelu-plan", "results/gemma3_270m_gelu_table_plan.json"), ("gelu-manifest", "results/gemma3_270m_gelu_table_manifest.json"), ("gelu-table", "artifacts/gemma3_270m_gelu_table.bin"),
                                  ("gelu-layouts", "results/gemma3_270m_gelu_layouts.json"), ("gelu-audit-dir", "artifacts/gelu_mismatches")):
                softmax.add_argument("--" + flag, type=Path, default=Path(default))
        if down_mode:
            for flag, default in (("product-plan", "results/gemma3_270m_mlp_product_plan.json"), ("product-bundle", "artifacts/gemma3_270m_mlp_product_predictions.json"), ("product-report", "artifacts/gemma3_270m_mlp_product_report.json"),
                                  ("probe-binding", "results/gemma3_270m_reduction_backend_binding.json"), ("probe-plan", "results/gemma3_270m_k2048_probes_plan.json"),
                                  ("probe-bundle", "artifacts/gemma3_270m_k2048_probes_inputs.json"), ("probe-report", "artifacts/gemma3_270m_k2048_probes_report.json")):
                softmax.add_argument("--" + flag, type=Path, default=Path(default))
        if dense_k2048_mode:
            for flag, default in (("model-down-plan", "results/gemma3_270m_mlp_down_plan_v2.json"), ("model-down-bundle", "artifacts/gemma3_270m_mlp_down_predictions_v2.json"), ("model-down-report", "artifacts/gemma3_270m_mlp_down_report_v2.json")):
                softmax.add_argument("--" + flag, type=Path, default=Path(default))
        if post_ff_mode:
            for flag, default in (("k2048-dense-plan", "results/gemma3_270m_k2048_dense_plan.json"), ("k2048-dense-bundle", "artifacts/gemma3_270m_k2048_dense_inputs.json"), ("k2048-dense-report", "artifacts/gemma3_270m_k2048_dense_report.json")):
                softmax.add_argument("--" + flag, type=Path, default=Path(default))
        if first_layer_mode:
            for flag, default in (("post-feedforward-plan", "results/gemma3_270m_post_feedforward_plan.json"), ("post-feedforward-bundle", "artifacts/gemma3_270m_post_feedforward_predictions.json"), ("post-feedforward-report", "artifacts/gemma3_270m_post_feedforward_report.json")):
                softmax.add_argument("--" + flag, type=Path, default=Path(default))
        if holdout_mode or second_layer_mode:
            for flag, default in (("baseline-plan", "results/gemma3_270m_first_layer_plan.json"), ("baseline-bundle", "artifacts/gemma3_270m_first_layer_predictions.json"), ("baseline-report", "artifacts/gemma3_270m_first_layer_report.json"), ("protocol", "results/gemma3_270m_first_layer_holdout_protocol.json")):
                softmax.add_argument("--" + flag, type=Path, default=Path(default))
        if second_layer_mode:
            for flag, default in (("holdout-plan", "results/gemma3_270m_first_layer_holdout_plan.json"), ("holdout-bundle", "artifacts/gemma3_270m_first_layer_holdout_predictions.json"), ("holdout-report", "artifacts/gemma3_270m_first_layer_holdout_report.json")):
                softmax.add_argument("--" + flag, type=Path, default=Path(default))
        if two_layers_mode:
            for flag, default in (("second-layer-plan", "results/gemma3_270m_second_layer_plan.json"), ("second-layer-bundle", "artifacts/gemma3_270m_second_layer_predictions.json"), ("second-layer-report", "artifacts/gemma3_270m_second_layer_report.json")):
                softmax.add_argument("--" + flag, type=Path, default=Path(default))
        if two_layers_holdout_mode:
            for flag, default in (("two-layer-baseline-plan", "results/gemma3_270m_two_layers_plan.json"), ("two-layer-baseline-bundle", "artifacts/gemma3_270m_two_layers_predictions.json"), ("two-layer-baseline-report", "artifacts/gemma3_270m_two_layers_report.json"), ("holdout-protocol", "results/gemma3_270m_two_layers_holdout_protocol.json")):
                softmax.add_argument("--" + flag, type=Path, default=Path(default))
        if third_entry_mode:
            for flag, default in (("two-holdout-plan", "results/gemma3_270m_two_layers_holdout_plan.json"), ("two-holdout-bundle", "artifacts/gemma3_270m_two_layers_holdout_predictions.json"), ("two-holdout-report", "artifacts/gemma3_270m_two_layers_holdout_report.json")):
                softmax.add_argument("--" + flag, type=Path, default=Path(default))
        if third_scores_mode:
            for flag, default in (("third-entry-plan", "results/gemma3_270m_third_layer_entry_plan.json"), ("third-entry-bundle", "artifacts/gemma3_270m_third_layer_entry_predictions.json"), ("third-entry-report", "artifacts/gemma3_270m_third_layer_entry_report.json")):
                softmax.add_argument("--" + flag, type=Path, default=Path(default))
        default_plan = "results/gemma3_270m_third_layer_scores_plan.json" if third_scores_mode else "results/gemma3_270m_third_layer_entry_plan.json" if third_entry_mode else "results/gemma3_270m_two_layers_holdout_plan.json" if two_layers_holdout_mode else "results/gemma3_270m_two_layers_plan.json" if two_layers_mode else "results/gemma3_270m_second_layer_plan.json" if second_layer_mode else "results/gemma3_270m_first_layer_holdout_plan.json" if holdout_mode else "results/gemma3_270m_first_layer_plan.json" if first_layer_mode else "results/gemma3_270m_post_feedforward_plan.json" if post_ff_mode else "results/gemma3_270m_k2048_dense_plan.json" if dense_k2048_mode else "results/gemma3_270m_mlp_down_plan.json" if down_mode else "results/gemma3_270m_mlp_product_plan.json" if product_mode else "results/gemma3_270m_mlp_entry_plan.json" if mlp_mode else "results/gemma3_270m_post_attention_plan.json" if post_mode else "results/gemma3_270m_output_survivor_plan.json" if survivor_mode else "results/gemma3_270m_output_split_plan.json" if split_mode else "results/gemma3_270m_attention_output_plan.json" if output_mode else "results/gemma3_270m_softmax_lookup_plan.json" if lookup_mode else "results/gemma3_270m_softmax_plan.json"
        softmax.add_argument("--plan", type=Path, default=Path(default_plan))
        softmax.add_argument("--bundle", type=Path, required=not post_ff_mode, default=Path("artifacts/gemma3_270m_third_layer_scores_predictions.json") if third_scores_mode else Path("artifacts/gemma3_270m_third_layer_entry_predictions.json") if third_entry_mode else Path("artifacts/gemma3_270m_two_layers_holdout_predictions.json") if two_layers_holdout_mode else Path("artifacts/gemma3_270m_two_layers_predictions.json") if two_layers_mode else Path("artifacts/gemma3_270m_second_layer_predictions.json") if second_layer_mode else Path("artifacts/gemma3_270m_first_layer_holdout_predictions.json") if holdout_mode else Path("artifacts/gemma3_270m_first_layer_predictions.json") if first_layer_mode else Path("artifacts/gemma3_270m_post_feedforward_predictions.json") if post_ff_mode else None, help="Tensor-rich prediction bundle under Git-ignored artifacts/")
        softmax.add_argument("--summary", type=Path, default=Path("results/gemma3_270m_third_layer_scores_summary.json") if third_scores_mode else Path("results/gemma3_270m_third_layer_entry_summary.json") if third_entry_mode else Path("results/gemma3_270m_two_layers_holdout_summary.json") if two_layers_holdout_mode else Path("results/gemma3_270m_two_layers_summary.json") if two_layers_mode else Path("results/gemma3_270m_second_layer_summary.json") if second_layer_mode else Path("results/gemma3_270m_first_layer_holdout_summary.json") if holdout_mode else None)
        if operation == "verify":
            softmax.add_argument("--report", type=Path, required=True)
            softmax.add_argument("--reexecute", action="store_true")
        else:
            softmax.add_argument("--output", type=Path, required=two_layers_holdout_mode or not (two_layers_mode and operation == "plan"), help="Compact plan or tensor-rich run report (under artifacts/)")
        handler = command_third_layer_scores if third_scores_mode else command_third_layer_entry if third_entry_mode else command_two_layers_holdout if two_layers_holdout_mode else command_two_layers if two_layers_mode else command_second_layer if second_layer_mode else command_first_layer_holdout if holdout_mode else command_first_layer if first_layer_mode else command_post_feedforward if post_ff_mode else command_dense_k2048 if dense_k2048_mode else command_mlp_down if down_mode else command_mlp_product if product_mode else command_mlp_entry if mlp_mode else command_post_attention if post_mode else command_output_survivor if survivor_mode else command_output_split if split_mode else command_attention_output if output_mode else command_lookup_softmax if lookup_mode else command_softmax_slice
        softmax.set_defaults(handler=handler, operation=operation)

    for operation in ("plan", "run", "verify"):
        exponential = subparsers.add_parser(f"gemma-exp-{operation}", help=f"{operation.title()} explicit exponential table and exhaustive nonpositive-domain evidence")
        exponential.add_argument("--source-summary", type=Path, default=Path("results/gemma3_270m_softmax_summary.json"))
        exponential.add_argument("--plan", type=Path, default=Path("results/gemma3_270m_exp_plan.json"))
        exponential.add_argument("--manifest", type=Path, default=Path("results/gemma3_270m_exp_manifest.json"))
        exponential.add_argument("--table", type=Path, default=Path("artifacts/gemma3_270m_exp_table.bin"))
        exponential.add_argument("--audit-dir", type=Path, default=Path("artifacts/exp_mismatches"))
        exponential.add_argument("--journal", type=Path, default=Path("artifacts/gemma3_270m_exp_journal.json"))
        if operation == "verify":
            exponential.add_argument("--reexecute", action="store_true")
            exponential.add_argument("--replay-output", type=Path)
        else:
            exponential.add_argument("--output", type=Path, required=True)
        exponential.set_defaults(handler=command_exp_lookup, operation=operation)

    for operation in ("plan", "run", "layouts", "verify"):
        gelu = subparsers.add_parser(f"gemma-gelu-table-{operation}", help=f"{operation.title()} explicit finite-BF16 GELU-tanh specification")
        for flag, default in (("source-summary", "results/gemma3_270m_mlp_entry_summary.json"), ("plan", "results/gemma3_270m_gelu_table_plan.json"),
                              ("manifest", "results/gemma3_270m_gelu_table_manifest.json"), ("table", "artifacts/gemma3_270m_gelu_table.bin"),
                              ("layout-report", "results/gemma3_270m_gelu_layouts.json"), ("audit-dir", "artifacts/gelu_mismatches")):
            gelu.add_argument("--" + flag, type=Path, default=Path(default))
        if operation == "verify":
            gelu.add_argument("--reexecute", action="store_true")
            gelu.add_argument("--replay-output", type=Path)
        else:
            gelu.add_argument("--output", type=Path, required=True)
        gelu.set_defaults(handler=command_gelu_table, operation=operation)

    for operation in ("plan", "run", "verify"):
        probes = subparsers.add_parser(f"gemma-k1024-probes-{operation}", help=f"{operation.title()} controlled K1024 split/merge discrimination")
        probes.add_argument("--source-summary", type=Path, default=Path("results/gemma3_270m_attention_output_summary.json"))
        probes.add_argument("--plan", type=Path, default=Path("results/gemma3_270m_k1024_probes_plan.json"))
        probes.add_argument("--bundle", type=Path, required=True, help="Controlled input vectors; keep under Git-ignored artifacts/")
        probes.add_argument("--summary", type=Path)
        if operation == "verify":
            probes.add_argument("--report", type=Path, required=True)
            probes.add_argument("--reexecute", action="store_true")
        else:
            probes.add_argument("--output", type=Path, required=True, help="Compact frozen plan or tensor-rich report (under artifacts/)")
        probes.set_defaults(handler=command_k1024_probes, operation=operation)

    for operation in ("plan", "run", "verify"):
        down = subparsers.add_parser(f"gemma-k2048-probes-{operation}", help=f"{operation.title()} controlled K2048 down-projection hypotheses")
        down.add_argument("--binding", type=Path, default=Path("results/gemma3_270m_reduction_backend_binding.json"))
        down.add_argument("--product-summary", type=Path, default=Path("results/gemma3_270m_mlp_product_summary.json"))
        down.add_argument("--plan", type=Path, default=Path("results/gemma3_270m_k2048_probes_plan.json"))
        down.add_argument("--bundle", type=Path, required=True, help="Controlled input vectors under Git-ignored artifacts/")
        down.add_argument("--summary", type=Path)
        down.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=4)
        if operation == "verify":
            down.add_argument("--report", type=Path, required=True)
            down.add_argument("--reexecute", action="store_true")
        else:
            down.add_argument("--output", type=Path, required=True)
        down.set_defaults(handler=command_k2048_probes, operation=operation)

    for operation in ("plan", "run", "verify"):
        dense = subparsers.add_parser(f"gemma-k128-dense-{operation}", help=f"{operation.title()} disjoint dense full-matrix K128 holdout")
        for flag, default in (("source-summary", "results/gemma3_270m_attention_output_summary.json"),
                              ("probe-plan", "results/gemma3_270m_k1024_probes_plan.json"), ("probe-bundle", "artifacts/gemma3_270m_k1024_probes_inputs.json"),
                              ("probe-report", "artifacts/gemma3_270m_k1024_probes_report.json"), ("model-plan", "results/gemma3_270m_attention_output_plan.json"),
                              ("model-bundle", "artifacts/gemma3_270m_attention_output_predictions.json")):
            dense.add_argument("--" + flag, type=Path, default=Path(default))
        dense.add_argument("--plan", type=Path, default=Path("results/gemma3_270m_k128_dense_plan.json"))
        dense.add_argument("--bundle", type=Path, required=True, help="Dense tensor-rich input/weight/prediction bundle under artifacts/")
        dense.add_argument("--summary", type=Path)
        if operation == "verify":
            dense.add_argument("--report", type=Path, required=True)
            dense.add_argument("--reexecute", action="store_true")
        else:
            dense.add_argument("--output", type=Path, required=True)
        dense.set_defaults(handler=command_dense_k128, operation=operation)

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

    reduction_backend_parser = subparsers.add_parser("gemma-reduction-backend", help="Profile controlled Gemma-shape reductions and bind exact CUDA symbols to the Nsight suite")
    reduction_backend_parser.add_argument("--program", type=Path, required=True)
    reduction_backend_parser.add_argument("--reduction", type=Path, required=True)
    reduction_backend_parser.add_argument("--nsight-suite", type=Path, required=True)
    reduction_backend_parser.add_argument("--sequence-length", type=int, default=30)
    reduction_backend_parser.add_argument("--max-tensor-elements", type=int, default=200000000)
    reduction_backend_parser.add_argument("--output", type=Path, required=True)
    reduction_backend_parser.set_defaults(handler=command_gemma_reduction_backend)

    reduction_backend_verify_parser = subparsers.add_parser("gemma-reduction-backend-verify", help="Verify controlled Gemma-shape CUDA symbol and reduction-candidate bindings")
    reduction_backend_verify_parser.add_argument("--program", type=Path, required=True)
    reduction_backend_verify_parser.add_argument("--reduction", type=Path, required=True)
    reduction_backend_verify_parser.add_argument("--nsight-suite", type=Path, required=True)
    reduction_backend_verify_parser.add_argument("binding", type=Path)
    reduction_backend_verify_parser.set_defaults(handler=command_gemma_reduction_backend_verify)

    wmma_probe_parser = subparsers.add_parser("gemma-wmma-probe", help="Probe operand-position sensitivity inside the exact Gemma query-projection WMMA kernel")
    wmma_probe_parser.add_argument("--program", type=Path, required=True)
    wmma_probe_parser.add_argument("--reduction", type=Path, required=True)
    wmma_probe_parser.add_argument("--nsight-suite", type=Path, required=True)
    wmma_probe_parser.add_argument("--backend", type=Path, required=True)
    wmma_probe_parser.add_argument("--output", type=Path, required=True)
    wmma_probe_parser.set_defaults(handler=command_gemma_wmma_probe)

    wmma_probe_verify_parser = subparsers.add_parser("gemma-wmma-probe-verify", help="Verify WMMA operand-position probe commitments and negative semantic boundaries")
    wmma_probe_verify_parser.add_argument("--program", type=Path, required=True)
    wmma_probe_verify_parser.add_argument("--reduction", type=Path, required=True)
    wmma_probe_verify_parser.add_argument("--nsight-suite", type=Path, required=True)
    wmma_probe_verify_parser.add_argument("--backend", type=Path, required=True)
    wmma_probe_verify_parser.add_argument("certificate", type=Path)
    wmma_probe_verify_parser.add_argument("--reexecute", action="store_true")
    wmma_probe_verify_parser.set_defaults(handler=command_gemma_wmma_probe_verify)

    wmma_accumulator_parser = subparsers.add_parser("gemma-wmma-accumulator-probe", help="Map single-small-addend retention across WMMA K16 lane triplets")
    wmma_accumulator_parser.add_argument("--program", type=Path, required=True)
    wmma_accumulator_parser.add_argument("--reduction", type=Path, required=True)
    wmma_accumulator_parser.add_argument("--nsight-suite", type=Path, required=True)
    wmma_accumulator_parser.add_argument("--backend", type=Path, required=True)
    wmma_accumulator_parser.add_argument("--output", type=Path, required=True)
    wmma_accumulator_parser.set_defaults(handler=command_gemma_wmma_accumulator_probe)

    wmma_accumulator_verify_parser = subparsers.add_parser("gemma-wmma-accumulator-probe-verify", help="Verify WMMA single-addend retention records and optional CUDA replay")
    wmma_accumulator_verify_parser.add_argument("--program", type=Path, required=True)
    wmma_accumulator_verify_parser.add_argument("--reduction", type=Path, required=True)
    wmma_accumulator_verify_parser.add_argument("--nsight-suite", type=Path, required=True)
    wmma_accumulator_verify_parser.add_argument("--backend", type=Path, required=True)
    wmma_accumulator_verify_parser.add_argument("certificate", type=Path)
    wmma_accumulator_verify_parser.add_argument("--reexecute", action="store_true")
    wmma_accumulator_verify_parser.set_defaults(handler=command_gemma_wmma_accumulator_probe_verify)

    wmma_magnitude_parser = subparsers.add_parser("gemma-wmma-magnitude-probe", help="Probe signed bfloat16 magnitude retention across K16 lower, upper, and split placements")
    wmma_magnitude_parser.add_argument("--program", type=Path, required=True)
    wmma_magnitude_parser.add_argument("--reduction", type=Path, required=True)
    wmma_magnitude_parser.add_argument("--nsight-suite", type=Path, required=True)
    wmma_magnitude_parser.add_argument("--backend", type=Path, required=True)
    wmma_magnitude_parser.add_argument("--output", type=Path, required=True)
    wmma_magnitude_parser.set_defaults(handler=command_gemma_wmma_magnitude_probe)

    wmma_magnitude_verify_parser = subparsers.add_parser("gemma-wmma-magnitude-probe-verify", help="Verify signed-magnitude WMMA retention evidence and optional CUDA replay")
    wmma_magnitude_verify_parser.add_argument("--program", type=Path, required=True)
    wmma_magnitude_verify_parser.add_argument("--reduction", type=Path, required=True)
    wmma_magnitude_verify_parser.add_argument("--nsight-suite", type=Path, required=True)
    wmma_magnitude_verify_parser.add_argument("--backend", type=Path, required=True)
    wmma_magnitude_verify_parser.add_argument("certificate", type=Path)
    wmma_magnitude_verify_parser.add_argument("--reexecute", action="store_true")
    wmma_magnitude_verify_parser.set_defaults(handler=command_gemma_wmma_magnitude_probe_verify)

    wmma_candidate_parser = subparsers.add_parser("gemma-wmma-candidate-search", help="Search shared-exponent K8 half-transition candidates against all WMMA probes")
    wmma_candidate_parser.add_argument("--program", type=Path, required=True)
    wmma_candidate_parser.add_argument("--reduction", type=Path, required=True)
    wmma_candidate_parser.add_argument("--nsight-suite", type=Path, required=True)
    wmma_candidate_parser.add_argument("--backend", type=Path, required=True)
    wmma_candidate_parser.add_argument("--position-probe", type=Path, required=True)
    wmma_candidate_parser.add_argument("--accumulator-probe", type=Path, required=True)
    wmma_candidate_parser.add_argument("--magnitude-probe", type=Path, required=True)
    wmma_candidate_parser.add_argument("--output", type=Path, required=True)
    wmma_candidate_parser.set_defaults(handler=command_gemma_wmma_candidate_search)

    wmma_candidate_verify_parser = subparsers.add_parser("gemma-wmma-candidate-search-verify", help="Recompute and verify the WMMA numeric candidate search")
    wmma_candidate_verify_parser.add_argument("--program", type=Path, required=True)
    wmma_candidate_verify_parser.add_argument("--reduction", type=Path, required=True)
    wmma_candidate_verify_parser.add_argument("--nsight-suite", type=Path, required=True)
    wmma_candidate_verify_parser.add_argument("--backend", type=Path, required=True)
    wmma_candidate_verify_parser.add_argument("--position-probe", type=Path, required=True)
    wmma_candidate_verify_parser.add_argument("--accumulator-probe", type=Path, required=True)
    wmma_candidate_verify_parser.add_argument("--magnitude-probe", type=Path, required=True)
    wmma_candidate_verify_parser.add_argument("search", type=Path)
    wmma_candidate_verify_parser.set_defaults(handler=command_gemma_wmma_candidate_search_verify)

    holdout_plan = subparsers.add_parser("gemma-k16-holdout-plan", help="Freeze dense K16 inputs and predictions before CUDA acquisition")
    holdout_plan.add_argument("search", type=Path)
    holdout_plan.add_argument("--output", type=Path, required=True)
    holdout_plan.set_defaults(handler=command_k16_holdout_plan)

    holdout_run = subparsers.add_parser("gemma-k16-holdout-run", help="Acquire frozen holdout; preserve failures and return nonzero on any mismatch")
    holdout_run.add_argument("search", type=Path)
    holdout_run.add_argument("plan", type=Path)
    holdout_run.add_argument("--output", type=Path, required=True)
    holdout_run.set_defaults(handler=command_k16_holdout_run)

    holdout_verify = subparsers.add_parser("gemma-k16-holdout-verify", help="Verify evidence integrity separately from candidate conformance")
    holdout_verify.add_argument("search", type=Path)
    holdout_verify.add_argument("plan", type=Path)
    holdout_verify.add_argument("report", type=Path)
    holdout_verify.add_argument("--reexecute", action="store_true")
    holdout_verify.set_defaults(handler=command_k16_holdout_verify)

    composition_plan = subparsers.add_parser("gemma-composition-holdout-plan", help="Freeze a serial K8 carry hypothesis before CUDA comparison")
    composition_plan.add_argument("search", type=Path)
    composition_plan.add_argument("--output", type=Path, required=True)
    composition_plan.set_defaults(handler=command_k16_holdout_plan, composition=True)

    composition_run = subparsers.add_parser("gemma-composition-holdout-run", help="Test cross-fragment predictions; retain failures and return nonzero on mismatch")
    composition_run.add_argument("search", type=Path)
    composition_run.add_argument("plan", type=Path)
    composition_run.add_argument("--output", type=Path, required=True)
    composition_run.set_defaults(handler=command_k16_holdout_run, composition=True)

    composition_verify = subparsers.add_parser("gemma-composition-holdout-verify", help="Verify composition evidence, optionally replaying CUDA without refitting")
    composition_verify.add_argument("search", type=Path)
    composition_verify.add_argument("plan", type=Path)
    composition_verify.add_argument("report", type=Path)
    composition_verify.add_argument("--reexecute", action="store_true")
    composition_verify.set_defaults(handler=command_k16_holdout_verify, composition=True)

    carry_diagnosis = subparsers.add_parser("gemma-carry-diagnosis", help="Compare four carry-rounding variants on development counterexamples, not validation data")
    carry_diagnosis.add_argument("search", type=Path)
    carry_diagnosis.add_argument("plan", type=Path)
    carry_diagnosis.add_argument("report", type=Path)
    carry_action = carry_diagnosis.add_mutually_exclusive_group(required=True)
    carry_action.add_argument("--output", type=Path)
    carry_action.add_argument("--diagnosis", type=Path, help="Recompute and verify an existing diagnosis")
    carry_diagnosis.set_defaults(handler=command_carry_diagnosis)

    for operation in ("plan", "run", "verify"):
        revision = subparsers.add_parser(f"gemma-carry-revision-{operation}", help=f"{operation.title()} the frozen float32 carry revision's fresh holdout")
        revision.add_argument("search", type=Path)
        revision.add_argument("diagnosis", type=Path)
        if operation != "plan":
            revision.add_argument("plan", type=Path)
        if operation == "verify":
            revision.add_argument("report", type=Path)
            revision.add_argument("--reexecute", action="store_true")
        else:
            revision.add_argument("--output", type=Path, required=True)
        revision.set_defaults(handler=command_carry_revision, operation=operation)

    for operation in ("plan", "run", "verify"):
        product_holdout = subparsers.add_parser(f"gemma-product-holdout-{operation}", help=f"{operation.title()} frozen non-unit-product predictions at three projection shapes")
        product_holdout.add_argument("search", type=Path)
        product_holdout.add_argument("diagnosis", type=Path)
        if operation != "plan":
            product_holdout.add_argument("plan", type=Path)
        if operation == "verify":
            product_holdout.add_argument("report", type=Path)
            product_holdout.add_argument("--reexecute", action="store_true")
        else:
            product_holdout.add_argument("--output", type=Path, required=True)
        product_holdout.set_defaults(handler=command_carry_revision, operation=operation, product=True)

    for operation in ("plan", "run", "verify"):
        matched_products = subparsers.add_parser(f"gemma-matched-products-{operation}", help=f"{operation.title()} controlled identical-product comparisons across shapes and rescalings")
        matched_products.add_argument("search", type=Path)
        matched_products.add_argument("diagnosis", type=Path)
        if operation != "plan":
            matched_products.add_argument("plan", type=Path)
        if operation == "verify":
            matched_products.add_argument("report", type=Path)
            matched_products.add_argument("--reexecute", action="store_true")
        else:
            matched_products.add_argument("--output", type=Path, required=True)
        matched_products.set_defaults(handler=command_carry_revision, operation=operation, matched=True)

    for operation in ("plan", "run", "verify"):
        query_reduction = subparsers.add_parser(f"gemma-query-reduction-{operation}", help=f"{operation.title()} development-only query counterexample deletion")
        for name in ("search", "diagnosis", "source_plan", "source_report"):
            query_reduction.add_argument(name, type=Path)
        if operation != "plan":
            query_reduction.add_argument("plan", type=Path)
        if operation == "verify":
            query_reduction.add_argument("report", type=Path)
            query_reduction.add_argument("--reexecute", action="store_true")
        else:
            query_reduction.add_argument("--output", type=Path, required=True)
        if operation == "run":
            query_reduction.add_argument("--journal", type=Path, required=True)
        query_reduction.set_defaults(handler=command_query_reduction, operation=operation)

    alignment_diagnosis = subparsers.add_parser("gemma-operand-alignment-diagnosis", help="Evaluate operand-exponent alignment on preserved development cases")
    for name in ("search", "carry_diagnosis", "source_plan", "source_report", "reduction_plan", "reduction_report"):
        alignment_diagnosis.add_argument(name, type=Path)
    alignment_result = alignment_diagnosis.add_mutually_exclusive_group(required=True)
    alignment_result.add_argument("--output", type=Path)
    alignment_result.add_argument("--diagnosis", type=Path)
    alignment_diagnosis.set_defaults(handler=command_operand_alignment_diagnosis)

    for operation in ("plan", "run", "verify"):
        alignment_holdout = subparsers.add_parser(f"gemma-operand-alignment-{operation}", help=f"{operation.title()} fresh frozen operand-alignment query holdout")
        alignment_holdout.add_argument("diagnosis", type=Path)
        alignment_holdout.add_argument("source_report", type=Path)
        if operation != "plan":
            alignment_holdout.add_argument("plan", type=Path)
        if operation == "verify":
            alignment_holdout.add_argument("report", type=Path)
            alignment_holdout.add_argument("--reexecute", action="store_true")
        else:
            alignment_holdout.add_argument("--output", type=Path, required=True)
        alignment_holdout.set_defaults(handler=command_operand_alignment_holdout, operation=operation)

    for operation in ("plan", "run", "verify"):
        wide_query = subparsers.add_parser(f"gemma-wide-query-{operation}", help=f"{operation.title()} distinct-row full-width query holdout")
        wide_query.add_argument("source_plan", type=Path)
        wide_query.add_argument("source_report", type=Path)
        if operation != "plan":
            wide_query.add_argument("plan", type=Path)
        if operation == "verify":
            wide_query.add_argument("report", type=Path)
            wide_query.add_argument("--reexecute", action="store_true")
        else:
            wide_query.add_argument("--output", type=Path, required=True)
        wide_query.set_defaults(handler=command_wide_query, operation=operation)

    split_k_diagnosis = subparsers.add_parser("gemma-split-k-diagnosis", help="Compare explicit split-K partition and intermediate-rounding hypotheses on development evidence")
    for name in ("search", "carry_diagnosis", "product_plan", "product_report", "matched_plan", "matched_report"):
        split_k_diagnosis.add_argument(name, type=Path)
    split_k_result = split_k_diagnosis.add_mutually_exclusive_group(required=True)
    split_k_result.add_argument("--output", type=Path)
    split_k_result.add_argument("--diagnosis", type=Path)
    split_k_diagnosis.set_defaults(handler=command_split_k_diagnosis)

    for operation in ("plan", "run", "verify"):
        split_merge = subparsers.add_parser(f"gemma-split-merge-{operation}", help=f"{operation.title()} prospective discrimination of frozen split-K merge candidates")
        split_merge.add_argument("diagnosis", type=Path)
        if operation != "plan":
            split_merge.add_argument("plan", type=Path)
        if operation == "verify":
            split_merge.add_argument("report", type=Path)
            split_merge.add_argument("--reexecute", action="store_true")
        else:
            split_merge.add_argument("--output", type=Path, required=True)
        split_merge.set_defaults(handler=command_split_merge, operation=operation)

    for operation in ("plan", "run", "verify"):
        dense_split = subparsers.add_parser(f"gemma-dense-split-{operation}", help=f"{operation.title()} fresh dense composed split-K holdout")
        for name in ("diagnosis", "merge_plan", "merge_report"):
            dense_split.add_argument(name, type=Path)
        if operation != "plan":
            dense_split.add_argument("plan", type=Path)
        if operation == "verify":
            dense_split.add_argument("report", type=Path)
            dense_split.add_argument("--reexecute", action="store_true")
        else:
            dense_split.add_argument("--output", type=Path, required=True)
        dense_split.set_defaults(handler=command_dense_split, operation=operation)

    primitive_gate_parser = subparsers.add_parser("gemma-ir-primitive-gate", help="Build a gate separating independently tested indexing primitives from unresolved floating-point semantics")
    primitive_gate_parser.add_argument("program", type=Path)
    primitive_gate_parser.add_argument("certificate", type=Path)
    primitive_gate_parser.add_argument("bfloat16_certificate", type=Path)
    primitive_gate_parser.add_argument("reduction_certificate", type=Path)
    primitive_gate_parser.add_argument("reduction_backend", type=Path)
    primitive_gate_parser.add_argument("nsight_suite", type=Path)
    primitive_gate_parser.add_argument("wmma_probe", type=Path)
    primitive_gate_parser.add_argument("wmma_accumulator_probe", type=Path)
    primitive_gate_parser.add_argument("wmma_magnitude_probe", type=Path)
    primitive_gate_parser.add_argument("wmma_candidate_search", type=Path)
    primitive_gate_parser.add_argument("--output", type=Path, required=True)
    primitive_gate_parser.set_defaults(handler=command_gemma_ir_primitive_gate)

    primitive_gate_verify_parser = subparsers.add_parser("gemma-ir-primitive-gate-verify", help="Verify an IR primitive-qualification gate and its negative activation boundaries")
    primitive_gate_verify_parser.add_argument("program", type=Path)
    primitive_gate_verify_parser.add_argument("certificate", type=Path)
    primitive_gate_verify_parser.add_argument("bfloat16_certificate", type=Path)
    primitive_gate_verify_parser.add_argument("reduction_certificate", type=Path)
    primitive_gate_verify_parser.add_argument("reduction_backend", type=Path)
    primitive_gate_verify_parser.add_argument("nsight_suite", type=Path)
    primitive_gate_verify_parser.add_argument("wmma_probe", type=Path)
    primitive_gate_verify_parser.add_argument("wmma_accumulator_probe", type=Path)
    primitive_gate_verify_parser.add_argument("wmma_magnitude_probe", type=Path)
    primitive_gate_verify_parser.add_argument("wmma_candidate_search", type=Path)
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
