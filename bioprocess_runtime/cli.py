from __future__ import annotations

import argparse
import json
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
