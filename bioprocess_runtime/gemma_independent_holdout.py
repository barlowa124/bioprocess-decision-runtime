from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import gemma_independent_run as runner
from . import gemma_first_layer_holdout as first_holdout
from . import gemma_two_layers_holdout as sampler
from .gemma_checkpoint import CheckpointStore, _load, _publish
from .gemma_rotary_slice import _sha, _seal, _check_hash
from .serialization import canonical_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
CASE_IDS = ("distinct_tokens", "repeated_motif")
BASELINE_PREDICTION = "27df8b72a8891abd0326220760d3eaf2a6b18c94bd3bd06b389c329e536a7b75"
BASELINE_REPORT = "737297c6fe982a4eb4773ddb262c31d2604095513ae5d1727e0e8fca05ac67fe"
PINNED_FILES = {
    "declaration": ("results/gemma3_270m_independent_holdout_declaration.json", "08b19852ce09df3009666cc4671f66127e8e4d4b216d2b4930685e5604242ee1"),
    "revision2": ("results/gemma3_270m_independent_holdout_binding_v2.json", "0998f5f709f89b8fdf4df5659d072ba93fec5b165b7cbd821a748c0d4888c0fd"),
    "revision3": ("results/gemma3_270m_independent_holdout_binding_v3.json", "f0cebc8a243e5a76a83524a96484f050c40fb93e9ed5680766c47fe90a8d7b36"),
    "replay_receipt": ("results/gemma3_270m_independent_baseline_replay_v3.json", "35704a04fbdc8a6fc2f06ce2f0b99ac5009b01d6a0cb06c56bf25543b32695e2"),
}
SCOPE = "Two predeclared raw-token full-target cases with fixed v3 arithmetic/runtime/checkpoint. Both complete predictions must be frozen before any held-out native forward. Three boundary-hooked full forwards per case, not all internal native states. Checkpoint continuation is disclosed; no refit, resampling, replacement, language-task, clinical, unrestricted or hardware qualification."


def code_sha256():
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Held-out runner source changed after import")
    return _sha({"module": SOURCE_SHA256, "runner": runner.code_sha256(), "tokenizer": first_holdout._code_sha(), "sampler": sampler._code_sha()})


def validate_declaration(declaration, revision2, revision3, program, baseline_ids, model_path, priors, global_provider):
    if declaration["program_sha256"] != program["program_sha256"] or declaration["target"] != "selected_token_id" or declaration["instruction_count"] != 533 or declaration["position_ids"] != list(range(30)) or declaration["baseline_sequence_sha256"] != _sha(baseline_ids):
        raise ValueError("Declared program/target/positions/baseline mismatch")
    if revision2["superseded_bindings"] != {key: declaration[key] for key in ("runner_code_sha256", "global_provider_sha256")} or revision3["superseded_bindings"] != revision2["bindings"]:
        raise ValueError("Binding revision ancestry mismatch")
    bindings = revision3["bindings"]
    evidence = runner.engine.vocabulary_evidence()
    if bindings != {"runner_code_sha256": runner.code_sha256(), "global_provider_sha256": global_provider["provider_sha256"], "gemv_arithmetic_code_sha256": evidence["arithmetic_code_sha256"], "gemv_component_report_sha256": evidence["report_sha256"], "gemv_component_replay_sha256": evidence["replay_sha256"]}:
        raise ValueError("Current source/provider differs from declared v3 binding")
    if declaration["generator"]["module_sha256"] != sampler.SOURCE_SHA256 or declaration["generator"]["seed"] != sampler.SEED or declaration["generator"]["rule"] != sampler.GENERATOR:
        raise ValueError("Declared sampler source/seed/rule changed")
    metadata, pool = first_holdout.tokenizer_context(model_path, program, baseline_ids)
    if metadata != declaration["tokenizer"]:
        raise ValueError("Tokenizer assets/metadata differ from declaration")
    prior_cases = []
    for name in ("first_layer", "two_layers"):
        prior = priors[name]
        _check_hash(prior, "protocol_sha256")
        if prior["protocol_sha256"] != declaration["prior_protocols"][name] or [case["case_id"] for case in prior["cases"]] != list(CASE_IDS):
            raise ValueError("Prior protocol/case identity mismatch")
        for case in prior["cases"]:
            _check_hash(case, "case_sha256")
            runner.first._tokens(program, case["input_token_ids"])
        prior_cases.extend(prior["cases"])
    excluded = sorted({value for case in prior_cases for value in case["input_token_ids"][0]})
    excluded_set = set(excluded)
    final_pool = [value for value in pool if value not in excluded_set]
    if [case["case_sha256"] for case in prior_cases] != declaration["prior_case_sha256"] or excluded != declaration["prior_excluded_ids"] or _sha(excluded) != declaration["prior_excluded_ids_sha256"] or len(final_pool) != declaration["final_pool_count"] or _sha(final_pool) != declaration["final_pool_sha256"]:
        raise ValueError("Prior exclusions or final pool changed")
    cases = sampler._cases(final_pool)
    if cases != declaration["cases"] or declaration["required_case_ids"] != list(CASE_IDS) or revision3["case_sha256"] != revision2["case_sha256"] or revision3["case_sha256"] != {case["case_id"]: case["case_sha256"] for case in cases}:
        raise ValueError("Declared cases changed; no resampling allowed")
    return cases


class Environment:
    def __init__(self, root=ROOT, model_path=None):
        self.root = root.resolve()
        self.model_path = (model_path or root / ".models/gemma-3-270m-it").resolve()
        self.code = code_sha256()
        self.paths = {key: root / value[0] for key, value in PINNED_FILES.items()}
        if runner.file_hashes(self.paths) != {key: value[1] for key, value in PINNED_FILES.items()}:
            raise ValueError("Frozen declaration/revision/replay receipt changed")
        declaration, revision2, revision3, receipt = (_load(self.paths[name]) for name in PINNED_FILES)
        replay_paths = {"prediction": root / "artifacts/gemma3_270m_independent_baseline_prediction_v3.json", "report": root / "artifacts/gemma3_270m_independent_baseline_report_v3.json",
                        "replay_log": root / "gemma_independent_baseline_v3_replay.log", "supervisor_log": root / "gemma_independent_baseline_v3_replay_supervisor.log",
                        "checkpoint": root / "artifacts/gemma3_270m_independent_baseline_replay_checkpoints_v3/checkpoint-000533-14aa6dabda9d421b7538e7033083c238df8abaf04cd075033663472075a6f24b.json"}
        if runner.file_hashes(replay_paths) != receipt["source_file_sha256"] or any(receipt.get(key) is not True for key in ("checkpoint_matches_original_execution", "cli_valid", "cli_match", "reexecute_requested")) or receipt["replay_exit_code"] != 0 or receipt["runner_code_sha256"] != runner.code_sha256():
            raise ValueError("Complete passing baseline replay receipt required")
        self.paths.update(replay_paths)
        self.paths.update(program=root / "results/gemma3_270m_execution_ir.json", global_provider=root / "results/gemma3_270m_independent_global_rotary_v3.json", first_layer=root / "results/gemma3_270m_first_layer_holdout_protocol.json", two_layers=root / "results/gemma3_270m_two_layers_holdout_protocol.json")
        self.file_hashes = runner.file_hashes(self.paths)
        self.context = runner.load_context(root, self.model_path)
        self.global_provider = _load(self.paths["global_provider"])
        baseline, report = _load(self.paths["prediction"]), _load(self.paths["report"])
        runner.check_prediction(self.context, baseline, self.global_provider)
        _check_hash(report, "report_sha256")
        if baseline["prediction_sha256"] != BASELINE_PREDICTION or report["report_sha256"] != BASELINE_REPORT or report.get("match") is not True or baseline["execution"]["target"] != "selected_token_id":
            raise ValueError("Pinned complete passing full baseline required")
        recounted = runner.comparison_report(self.context, baseline, report["observations"], report["acquisition_guards"])
        if canonical_json(recounted) != canonical_json(report):
            raise ValueError("Baseline native report recount differs")
        priors = {name: _load(self.paths[name]) for name in ("first_layer", "two_layers")}
        cases = validate_declaration(declaration, revision2, revision3, self.context.program, baseline["execution"]["input_token_ids"], self.model_path, priors, self.global_provider)
        self.assets = first_holdout._assets(self.model_path)
        self.protocol = _seal({"kind": "independent_full_target_holdout_protocol_v1", "code_sha256": self.code, "scope": SCOPE, "cases": cases,
                               "source_file_sha256": self.file_hashes, "runner_code_sha256": runner.code_sha256(), "runner_sources": self.context.sources,
                               "runtime": self.context.runtime, "baseline_prediction_sha256": BASELINE_PREDICTION, "baseline_report_sha256": BASELINE_REPORT,
                               "required_forward_count": 6, "qualified": False, "hardware_semantics_established": False}, "protocol_sha256")
        self.guard()

    def guard(self):
        if self.code != code_sha256() or runner.file_hashes(self.paths) != self.file_hashes or first_holdout._assets(self.model_path) != self.assets:
            raise ValueError("Held-out source/declaration/tokenizer changed")
        if runner.load_vocabulary_evidence(self.root, self.context.runtime) != self.context.sources["vocabulary_evidence"]:
            raise ValueError("Vocabulary evidence changed")

    def model(self):
        from .cli import _holdout_model
        return _holdout_model(self.model_path)


def check_case(environment, case, prediction):
    runner.check_prediction(environment.context, prediction, environment.global_provider)
    value = prediction["execution"]
    if value["input_token_ids"] != case["input_token_ids"] or value["target"] != "selected_token_id" or value["vocabulary_candidate_enabled"] is not True or value["state_retention"] != "boundaries":
        raise ValueError("Case prediction input/target/options mismatch")
    if value["status"] == "complete_uncompared" and value["completed_instruction_count"] != 533:
        raise ValueError("Partial target cannot open native gate")


def prediction_path(directory, case_id, replay=False):
    return directory / (case_id + (".replay" if replay else "") + ".prediction.json")


def predict_all(environment, directory, *, resume=False, workers=4, replay=False):
    environment.guard()
    predictions, reused = [], []
    for case in environment.protocol["cases"]:
        runner.require_frozen(directory / "protocol.json", environment.protocol)
        path = prediction_path(directory, case["case_id"], replay)
        if path.exists():
            if not resume:
                raise ValueError("Prediction already exists; explicit resume required")
            prediction = _load(path)
            reused.append(case["case_id"])
        else:
            checkpoint = directory / (case["case_id"] + (".replay" if replay else "") + ".checkpoints")
            if checkpoint.exists() and not resume:
                raise ValueError("Existing case checkpoint requires resume")
            model = environment.model()
            prediction = runner.predict(environment.context, model, case["input_token_ids"], environment.global_provider, workers=workers, vocabulary_candidate=True,
                                        checkpoint_dir=checkpoint, resume=checkpoint.exists(), progress=lambda event, name=case["case_id"]: print(json.dumps({"case_id": name, "replay": replay, **event}), flush=True))
            del model
            check_case(environment, case, prediction)
            environment.guard()
            _publish(path, prediction)
        check_case(environment, case, prediction)
        predictions.append(prediction)
        environment.guard()
    return predictions, reused


def plan_body(environment, predictions, reused=()):
    if list(reused) != [name for name in CASE_IDS if name in reused]:
        raise ValueError("Invalid restored-case provenance")
    cases = environment.protocol["cases"]
    if len(predictions) != len(cases) or len(cases) != 2 or [case["case_id"] for case in cases] != list(CASE_IDS):
        raise ValueError("Both original cases required in declared order")
    for case, prediction in zip(cases, predictions):
        check_case(environment, case, prediction)
    return _seal({"kind": "independent_full_target_holdout_plan_v1", "protocol_sha256": environment.protocol["protocol_sha256"], "code_sha256": environment.code,
                  "cases": [{"case_id": case["case_id"], "case_sha256": case["case_sha256"], "prediction_sha256": prediction["prediction_sha256"],
                             "status": prediction["execution"]["status"], "abstention": prediction["execution"]["abstention"]} for case, prediction in zip(cases, predictions)],
                  "prediction_complete": all(prediction["execution"]["status"] == "complete_uncompared" for prediction in predictions),
                  "required_forward_count": 6, "restored_completed_cases": list(reused), "qualified": False}, "plan_sha256")


def frozen_predictions(environment, directory, plan):
    runner.require_frozen(directory / "protocol.json", environment.protocol)
    runner.require_frozen(directory / "plan.json", plan)
    _check_hash(plan, "plan_sha256")
    predictions = [_load(prediction_path(directory, name)) for name in CASE_IDS]
    if canonical_json(plan_body(environment, predictions, plan["restored_completed_cases"])) != canonical_json(plan):
        raise ValueError("Frozen two-case plan/prediction mismatch")
    for name, prediction in zip(CASE_IDS, predictions):
        runner.require_frozen(prediction_path(directory, name), prediction)
    environment.guard()
    return predictions


def acquire_all(environment, directory, plan, *, resume=False):
    predictions = frozen_predictions(environment, directory, plan)
    if not plan["prediction_complete"]:
        return _seal({"kind": "independent_full_target_holdout_report_v1", "plan_sha256": plan["plan_sha256"], "status": "blocked_by_prediction_failure",
                      "match": False, "cases": plan["cases"], "recorded_forward_count": 0, "fresh_forward_count": 0, "qualified": False}, "report_sha256")
    reports, reused = [], []
    for case, prediction in zip(environment.protocol["cases"], predictions):
        frozen_predictions(environment, directory, plan)
        path = directory / (case["case_id"] + ".report.json")
        if path.exists():
            if not resume:
                raise ValueError("Native case report exists; explicit resume required")
            report = _load(path)
            reused.append(case["case_id"])
        else:
            model = environment.model()
            report = runner.acquire(environment.context, model, prediction, environment.global_provider, prediction_path(directory, case["case_id"]))
            del model
            frozen_predictions(environment, directory, plan)
            _publish(path, report)
        _check_hash(report, "report_sha256")
        if canonical_json(runner.comparison_report(environment.context, prediction, report["observations"], report["acquisition_guards"])) != canonical_json(report):
            raise ValueError("Native case report recount mismatch")
        reports.append(report)
    return _seal({"kind": "independent_full_target_holdout_report_v1", "plan_sha256": plan["plan_sha256"], "status": "matched" if all(report["match"] for report in reports) else "mismatched",
                  "match": all(report["match"] for report in reports), "cases": [{"case_id": name, "report_sha256": report["report_sha256"], "match": report["match"], "selected_token_id": report["selected_token_id"], "mismatch_count": report["aggregate_mismatch_count"]} for name, report in zip(CASE_IDS, reports)],
                  "recorded_forward_count": sum(report["original_forward_count"] for report in reports), "fresh_forward_count": 3 * (2 - len(reused)), "restored_native_cases": reused,
                  "qualified": False, "hardware_semantics_established": False, "scope": SCOPE}, "report_sha256")


def replay_all(environment, directory, plan, *, resume=False, workers=4):
    originals = frozen_predictions(environment, directory, plan)
    if not plan["prediction_complete"]:
        raise ValueError("Incomplete predictions block replay/native acquisition")
    reports = [_load(directory / (name + ".report.json")) for name in CASE_IDS]
    for original, report in zip(originals, reports):
        _check_hash(report, "report_sha256")
        if canonical_json(runner.comparison_report(environment.context, original, report["observations"], report["acquisition_guards"])) != canonical_json(report):
            raise ValueError("Original report is not intact")
    fresh, reused = predict_all(environment, directory, resume=resume, workers=workers, replay=True)
    matched = all(runner.same_numerical_prediction(old, new) for old, new in zip(originals, fresh))
    replay_plan = _seal({"kind": "independent_holdout_replay_predictions_v1", "plan_sha256": plan["plan_sha256"], "prediction_sha256": [value["prediction_sha256"] for value in fresh],
                         "match": matched, "restored_completed_cases": reused}, "replay_plan_sha256")
    path = directory / "replay_predictions.json"
    if path.exists():
        saved = _load(path)
        _check_hash(saved, "replay_plan_sha256")
        if saved["prediction_sha256"] != replay_plan["prediction_sha256"] or saved["plan_sha256"] != plan["plan_sha256"] or saved["match"] is not matched:
            raise ValueError("Replay prediction freeze differs")
    else:
        _publish(path, replay_plan)
        saved = replay_plan

    def replay_guard():
        frozen_predictions(environment, directory, plan)
        runner.require_frozen(directory / "replay_predictions.json", saved)
        for case, value, report in zip(environment.protocol["cases"], fresh, reports):
            runner.require_frozen(prediction_path(directory, case["case_id"], True), value)
            runner.require_frozen(directory / (case["case_id"] + ".report.json"), report)

    replay_guard()
    if not matched:
        return _seal({"kind": "independent_holdout_replay_v1", "plan_sha256": plan["plan_sha256"], "replay_prediction_plan_sha256": saved["replay_plan_sha256"], "valid": False, "match": False, "native_forward_count": 0, "failure": "fresh_prediction_mismatch", "qualified": False}, "replay_sha256")
    native, restored_native = [], []
    for name, original, report in zip(CASE_IDS, originals, reports):
        replay_guard()
        path = directory / (name + ".replay.report.json")
        if path.exists():
            if not resume:
                raise ValueError("Replay report exists; explicit resume required")
            observed = _load(path)
            restored_native.append(name)
        else:
            model = environment.model()
            observed = runner.acquire(environment.context, model, original, environment.global_provider, prediction_path(directory, name))
            del model
            replay_guard()
            _publish(path, observed)
        replay_guard()
        _check_hash(observed, "report_sha256")
        native.append(canonical_json(observed) == canonical_json(report))
    environment.guard()
    return _seal({"kind": "independent_holdout_replay_v1", "plan_sha256": plan["plan_sha256"], "valid": all(native), "match": all(native) and all(report["match"] for report in reports),
                  "recorded_native_forward_count": 6, "fresh_native_forward_count": 3 * (2 - len(restored_native)), "restored_native_cases": restored_native,
                  "replay_prediction_plan_sha256": saved["replay_plan_sha256"], "prediction_sha256": [value["prediction_sha256"] for value in fresh], "native_report_match": dict(zip(CASE_IDS, native)),
                  "qualified": False, "hardware_semantics_established": False, "scope": SCOPE}, "replay_sha256")


def main():
    parser = argparse.ArgumentParser(description="Fixed-case gated independent full-target holdout")
    parser.add_argument("operation", choices=("plan", "compare", "verify", "run"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if (ROOT / "artifacts").resolve() not in directory.parents or not 1 <= args.workers <= 4:
        raise ValueError("Use a dedicated ignored artifacts subdirectory and one to four workers")
    existing = directory.exists()
    if args.resume and not existing or args.operation in ("plan", "run") and existing and not args.resume or args.operation in ("compare", "verify") and not existing:
        raise ValueError("New run directory or explicit existing-run resume required")
    environment = Environment()
    binding = {"protocol_sha256": environment.protocol["protocol_sha256"], "code_sha256": environment.code, "purpose": "exclusive full-target holdout orchestration lease"}
    with CheckpointStore(directory, binding, resume=existing) as lease:
        if not existing:
            lease.save({"completed_instruction_count": 0}, {})
            _publish(directory / "protocol.json", environment.protocol)
        runner.require_frozen(directory / "protocol.json", environment.protocol)
        plan_path = directory / "plan.json"
        if args.operation in ("plan", "run") and not plan_path.exists():
            predictions, reused = predict_all(environment, directory, resume=args.resume, workers=args.workers)
            plan = plan_body(environment, predictions, reused)
            environment.guard()
            _publish(plan_path, plan)
            print(json.dumps({"plan_sha256": plan["plan_sha256"], "prediction_complete": plan["prediction_complete"], "restored_completed_cases": reused}), flush=True)
        else:
            plan = _load(plan_path)
        frozen_predictions(environment, directory, plan)
        if args.operation == "plan":
            return 0 if plan["prediction_complete"] else 1
        output = directory / ("replay.json" if args.operation == "verify" else "report.json")
        if output.exists():
            raise ValueError("Completed aggregate output already exists; preserve it")
        result = replay_all(environment, directory, plan, resume=args.resume, workers=args.workers) if args.operation == "verify" else acquire_all(environment, directory, plan, resume=args.resume)
        environment.guard()
        _publish(output, result)
        print(json.dumps({key: value for key, value in result.items() if key not in ("cases", "scope")}), flush=True)
        return 0 if result["match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
