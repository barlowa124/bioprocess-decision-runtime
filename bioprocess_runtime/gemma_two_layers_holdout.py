from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import gemma_two_layers as two
from . import gemma_first_layer_holdout as holdout
from .gemma_rotary_slice import _sha, _seal, _check_hash
from .serialization import canonical_json

SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
BASELINE_PLAN_SHA256 = "49d121756cca33eb1d139061093d844aa851791323cc7f5b34f19a0870b78fc4"
BASELINE_REPORT_SHA256 = "9b177ee28872ef11411307ce015843233562b2bdeeeabedc1bfc6caa9dacf5bc"
PRIOR_PROTOCOL_SHA256 = "37af351612e293750f60678176980cb4b735467756b20641a909cf71a4c31905"
SEED = 0x8E2F6B41
CASE_IDS = ("distinct_tokens", "repeated_motif")
GENERATOR = "sorted final allowed IDs; xorshift32 shifts 13,17,5 masked uint32; partial Fisher-Yates indices 0..34, j=index+word()%(pool_count-index); first30 distinct; last5 cycled6; deterministic index sampling, not a statistical-uniformity claim"
SCOPE = "Prospective two predeclared 30-token RAW-ID experimental numerical cases for the fixed Gemma3 270m checkpoint, connected input_ids to hidden.2 through the unchanged 63-node/66-state dependency cone. Each case uses all 29 fresh checkpoint parameter snapshots, no stored hidden.1, boundary or intermediate predictions. Excludes all IDs from the declared baseline and both declared previous first-layer holdouts only, not unknown historical experiments or model training. Reuses empirical rsqrt/exp/GELU and fixed-position local rotary specifications. Separate prefix-stopped and second-layer-stopped native forwards, not a same-invocation whole trace or pointer/launch bridge proof. Twelve RMS norms cover 1620 FP32 scalar positions and two softmax outputs cover 7200 FP32 positions per case, not all FP32 internals. Raw tokens are not fresh prompts, natural-language or clinical task-performance evidence. No refit, resampling, replacement, unrestricted-domain, hardware, full-model, later-layer, final norm or logits qualification. The previous fixed-baseline input-case scope does not apply to these new raw-token cases."
GUARDS = (*two.GUARDS, "providers_unchanged", "protocol_unchanged")
require_frozen = holdout.require_frozen
write_new = holdout.write_new


def _code_sha() -> str:
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Two-layer holdout source changed after import")
    return _sha({"module": SOURCE_SHA256, "two_layers": two._code_sha(), "first_layer_holdout": holdout._code_sha()})


def _flags(connected: bool = False) -> dict[str, bool]:
    return {**dict.fromkeys(holdout.FALSE_FLAGS, False), **two._flags(connected),
            "fresh_prompt_holdout": False, "natural_language_performance_qualified": False,
            "prospective_two_layer_raw_token_cases": True, "excluded_from_declared_prior_cases": True}


class _CachedTwoSources:
    def __init__(self, source: two.TwoLayerSources):
        self.source = source
        self.context = None
        self.model_path = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.source, name)

    def validate(self, model_path: Path) -> tuple[Any, ...]:
        if self.context is None:
            self.context = self.source.validate(model_path)
            self.model_path = model_path.resolve()
        if model_path.resolve() != self.model_path:
            raise ValueError("Validation cache is local to one operation and model path")
        return self.context


@dataclass(frozen=True)
class BaselineSources:
    two_sources: two.TwoLayerSources
    two_layer_plan: dict[str, Any]
    two_layer_bundle: dict[str, Any]
    two_layer_report: dict[str, Any]

    @property
    def program(self) -> dict[str, Any]:
        return self.two_sources.program

    @property
    def runtime(self) -> dict[str, Any]:
        return self.two_sources.runtime

    def commitments(self) -> dict[str, Any]:
        return {"two_sources": self.two_sources.commitments(), "two_layer_plan_payload_sha256": _sha(self.two_layer_plan),
                "two_layer_bundle_payload_sha256": _sha(self.two_layer_bundle), "two_layer_report_payload_sha256": _sha(self.two_layer_report)}

    def validate(self, model_path: Path) -> tuple[Any, ...]:
        code, guard = _code_sha(), _sha(self.commitments())
        for payload, field, digest in ((self.two_layer_plan, "plan_sha256", BASELINE_PLAN_SHA256),
                                       (self.two_layer_report, "report_sha256", BASELINE_REPORT_SHA256)):
            _check_hash(payload, field)
            if payload[field] != digest:
                raise ValueError("Requires the pinned connected two-layer baseline plan/report")
        cached = _CachedTwoSources(self.two_sources)
        checked = two.verify_two_layers(cached, self.two_layer_plan, self.two_layer_bundle, self.two_layer_report, model_path)
        if any(checked.get(key) is not True for key in ("valid", "two_layers_match", "connected_two_layers_independently_recomputed")):
            raise ValueError("Requires passing connected two-layer baseline and full prerequisite lineage")
        report = self.two_layer_report
        if report.get("native_coverage_complete") is not True or report.get("original_forward_count") != 12 or report.get("required_forward_count") != 12 or any(value is not True for value in report["checks"].values()) or any(value is not True for value in report["acquisition_guards"].values()):
            raise ValueError("Requires complete twelve-forward baseline coverage with no failed checks")
        context = cached.validate(model_path)
        if context[0] != self.two_layer_plan["input_token_ids"]:
            raise ValueError("Declared baseline token binding differs from validated sources")
        _prior_exclusions(self)
        _kernel_expectations(self)
        if code != _code_sha() or guard != _sha(self.commitments()):
            raise ValueError("Baseline source changed during validation")
        return context


def _kernel_expectations(sources: BaselineSources) -> dict[str, Any]:
    trace = sources.two_layer_report["kernel_trace"]
    if len(trace) != 3 or any(pair != trace[0] for pair in trace) or set(trace[0]) != set(two.STAGE_IDS):
        raise ValueError("Connected baseline must pin three stable paired kernel sets")
    for stage, module in (("first", two.first), ("second", two.second)):
        roles, symbols = module._native_kernel_roles(sources.program), trace[0][stage]
        if len(roles) != 22 or set(symbols) != set(roles) or any(not isinstance(names, list) or not names or names != sorted(set(names)) or any(not isinstance(name, str) or not name.strip() for name in names) for names in symbols.values()):
            raise ValueError("Both baseline stages require all 22 valid kernel role sets")
    if trace[0] != sources.two_layer_plan["expected_kernel_sets"]:
        raise ValueError("Connected baseline report/plan kernel sets disagree")
    return copy.deepcopy(trace[0])


def _prior_exclusions(sources: BaselineSources) -> dict[str, Any]:
    prior = sources.two_sources.second_sources.protocol
    _check_hash(prior, "protocol_sha256")
    if prior["protocol_sha256"] != PRIOR_PROTOCOL_SHA256 or [case["case_id"] for case in prior["cases"]] != list(holdout.CASE_IDS):
        raise ValueError("Both declared previous first-layer holdout cases required")
    for case in prior["cases"]:
        two.first._tokens(sources.program, case["input_token_ids"])
        _check_hash(case, "case_sha256")
    ids = sorted({value for case in prior["cases"] for value in case["input_token_ids"][0]})
    return {"prior_protocol_sha256": prior["protocol_sha256"], "prior_protocol_payload_sha256": _sha(prior),
            "prior_case_sha256": {case["case_id"]: case["case_sha256"] for case in prior["cases"]},
            "prior_token_ids": ids, "prior_token_ids_sha256": _sha(ids)}


def _cases(pool: list[int]) -> list[dict[str, Any]]:
    if len(pool) < 35 or any(type(value) is not int for value in pool) or pool != sorted(set(pool)):
        raise ValueError("Expected sorted unique final pool of at least 35 integer IDs")
    selected, state = list(pool), SEED
    for index in range(35):
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        state &= 0xFFFFFFFF
        other = index + state % (len(selected) - index)
        selected[index], selected[other] = selected[other], selected[index]
    return [_seal({"case_id": name, "input_token_ids": [ids]}, "case_sha256")
            for name, ids in zip(CASE_IDS, (selected[:30], selected[30:35] * 6))]


def _protocol_body(sources: BaselineSources, model_path: Path, context: tuple[Any, ...]) -> dict[str, Any]:
    ids, providers, profiles = context
    metadata, pool = holdout.tokenizer_context(model_path, sources.program, ids)
    prior = _prior_exclusions(sources)
    excluded = set(prior["prior_token_ids"])
    final_pool = [value for value in pool if value not in excluded]
    metadata.update(base_allowed_pool_count=metadata["allowed_pool_count"], base_allowed_pool_sha256=metadata["allowed_pool_sha256"],
                    allowed_pool_count=len(final_pool), allowed_pool_sha256=_sha(final_pool), prior_first_layer_holdout_exclusions=prior)
    return {"schema_version": 1, "artifact_kind": "gemma_two_layers_holdout_protocol", "scope": SCOPE,
            "seed": SEED, "generator_rule": GENERATOR, "cases": _cases(final_pool), "required_case_ids": list(CASE_IDS), "tokenizer": metadata,
            "sources": sources.commitments(), "code_sha256": _code_sha(), "two_layers_code_sha256": two._code_sha(),
            "first_layer_holdout_code_sha256": holdout._code_sha(),
            "capture_code_sha256": {"first": two.first_capture._code_sha(), "second": two.second_capture._code_sha()},
            "runtime": copy.deepcopy(sources.runtime), "program_sha256": sources.program["program_sha256"], "program_payload_sha256": _sha(sources.program),
            "model_binding": copy.deepcopy(sources.two_layer_plan["model_binding"]), "providers": providers.commitments(), "profiles": copy.deepcopy(profiles),
            "expected_kernel_sets": _kernel_expectations(sources), "coverage_per_case": two._coverage(sources.program),
            "case_count": 2, "repetitions_per_case": 3, "required_forward_count": 24, "native_scope": two.NATIVE_SCOPE,
            "native_protocol": ["first_traced_stop_0", "first_plain_stop_0", "second_traced_stop_1", "second_plain_stop_1"],
            "kernel_role_scope_count_per_case": 44,
            "observation_rule": "freeze protocol before any new prediction; freeze ALL complete case predictions before ANY native call; fixed case order, three repetitions of four separate stopped forwards per case; preserve failures, no refit/resampling/replacement/output-driven selection",
            "epoch_scope": "prospective_raw_token_cases_excluded_from_declared_baseline_and_both_prior_first_layer_holdouts_only",
            **_flags()}


def build_holdout_protocol(sources: BaselineSources, model_path: Path) -> dict[str, Any]:
    code, guard = _code_sha(), _sha(sources.commitments())
    result = _seal(_protocol_body(sources, model_path, sources.validate(model_path)), "protocol_sha256")
    if code != _code_sha() or guard != _sha(sources.commitments()):
        raise ValueError("Source changed during protocol declaration")
    return result


def _protocol_check(sources: BaselineSources, protocol: dict[str, Any], model_path: Path, context: tuple[Any, ...] | None = None) -> tuple[Any, ...]:
    _check_hash(protocol, "protocol_sha256")
    context = sources.validate(model_path) if context is None else context
    if canonical_json(protocol) != canonical_json(_seal(_protocol_body(sources, model_path, context), "protocol_sha256")):
        raise ValueError("Protocol differs from exact current tokenizer/source/exclusions/seed/case regeneration")
    return context


def _plan_body(protocol: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    if set(bundle) != {"protocol_sha256", "cases"} or bundle["protocol_sha256"] != protocol["protocol_sha256"] or [item["case_id"] for item in bundle["cases"]] != list(CASE_IDS) or [item["case_id"] for item in protocol["cases"]] != list(CASE_IDS):
        raise ValueError("Every declared case is required exactly once in original order")
    cases = []
    for declared, outcome in zip(protocol["cases"], bundle["cases"]):
        if set(outcome) != {"case_id", "parameter_snapshots", "execution", "prediction_failure"} or (outcome["execution"] is None) != (outcome["prediction_failure"] is not None):
            raise ValueError("Each case requires fresh snapshots and either complete execution or preserved domain failure")
        item = {**copy.deepcopy(declared), "input_ids": two.first._descriptor(two.first._array(declared["input_token_ids"], [1, 30], "torch.int64")),
                "parameter_snapshots": {name: {key: value for key, value in snapshot.items() if key != "bits"} for name, snapshot in outcome["parameter_snapshots"].items()},
                "prediction_failure": copy.deepcopy(outcome["prediction_failure"])}
        execution = outcome["execution"]
        if execution is not None:
            coverage = protocol["coverage_per_case"]["states"]
            item.update(state_descriptors={name: two.first._descriptor(two.first._array(value, coverage[name]["shape"], coverage[name]["dtype"])) for name, value in execution["state_bits"].items()},
                        trace_root=execution["records"][-1]["record_hash"], execution_sha256=_sha(execution),
                        scalar_stages_sha256=_sha(execution["scalar_stages"]), softmax_f32_sha256=_sha(execution["softmax_f32_bits"]),
                        bridge=copy.deepcopy(execution["bridge"]), subtrace_roots=copy.deepcopy(execution["subtrace_roots"]))
        cases.append(item)
    return {**{key: copy.deepcopy(value) for key, value in protocol.items() if key not in ("cases", "artifact_kind", "tokenizer", "seed", "generator_rule")},
            "artifact_kind": "gemma_two_layers_holdout_plan", "cases": cases, "bundle_sha256": _sha(bundle),
            "prediction_complete": all(item["prediction_failure"] is None for item in cases),
            "parameter_snapshot_membership": "fresh full-checkpoint binding and selected embedding row indexing; offline integrity does not revalidate embedding membership; reexecute requires fresh model"}


def _validate_snapshots(sources: BaselineSources, ids: Any, snapshots: dict[str, Any], context: tuple[Any, ...]) -> None:
    _, providers, profiles = context
    left, right = two._split_snapshots(sources.program, snapshots)
    parameters = two.first.check_parameter_snapshots(sources.program, ids, left)
    two.second.check_parameter_snapshots(sources.program, right)
    two.first._profiles(profiles)
    providers.validate(sources.runtime, parameters[two.first.FREQUENCY])


def _predict(sources: BaselineSources, model: Any, protocol: dict[str, Any], context: tuple[Any, ...], workers: int) -> tuple[dict[str, Any], dict[str, Any]]:
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Holdout workers must be 1..4")
    code, guard, declaration = _code_sha(), _sha(sources.commitments()), _sha(protocol)
    _, providers, profiles = context
    outcomes = []
    for case in protocol["cases"]:
        ids = case["input_token_ids"]
        snapshots = two._snapshot_model(sources.two_sources, model, ids)
        _validate_snapshots(sources, ids, snapshots, context)
        execution, failure = None, None
        try:
            execution = two.execute_two_layers(sources.program, ids, snapshots, providers, profiles, sources.runtime, workers)
        except (ValueError, ArithmeticError) as error:
            if not holdout._domain_failure(error):
                raise
            failure = holdout._failure(error, case["case_id"])
        if canonical_json(two._snapshot_model(sources.two_sources, model, ids)) != canonical_json(snapshots):
            raise ValueError("Live checkpoint snapshots changed during case prediction")
        _validate_snapshots(sources, ids, snapshots, context)
        if code != _code_sha() or guard != _sha(sources.commitments()) or declaration != _sha(protocol):
            raise ValueError("Source or protocol changed during prediction")
        outcomes.append({"case_id": case["case_id"], "parameter_snapshots": snapshots, "execution": execution, "prediction_failure": failure})
    bundle = {"protocol_sha256": protocol["protocol_sha256"], "cases": outcomes}
    return _seal(_plan_body(protocol, bundle), "plan_sha256"), bundle


def build_holdout_plan(sources: BaselineSources, model: Any, protocol: dict[str, Any], model_path: Path, protocol_path: Path, workers: int = 4, frozen_inputs: tuple[Path, ...] = ()) -> tuple[dict[str, Any], dict[str, Any]]:
    require_frozen(protocol_path, protocol)
    paths = (protocol_path, *frozen_inputs)
    files = two._file_guard(paths)
    context = _protocol_check(sources, protocol, model_path)
    result = _predict(sources, model, protocol, context, workers)
    require_frozen(protocol_path, protocol)
    _protocol_check(sources, protocol, model_path, context)
    if files != two._file_guard(paths):
        raise ValueError("Frozen source/protocol files changed during prediction")
    return result


def _plan_check(sources: BaselineSources, protocol: dict[str, Any], plan: dict[str, Any], bundle: dict[str, Any], context: tuple[Any, ...]) -> None:
    _check_hash(plan, "plan_sha256")
    expected = _seal(_plan_body(protocol, bundle), "plan_sha256")
    if canonical_json(plan) != canonical_json(expected):
        raise ValueError("Frozen plan/bundle/case/header binding mismatch")
    _, providers, profiles = context
    for case, outcome in zip(protocol["cases"], bundle["cases"]):
        _validate_snapshots(sources, case["input_token_ids"], outcome["parameter_snapshots"], context)
        failure = outcome["prediction_failure"]
        if failure is None:
            two.check_execution(sources.program, case["input_token_ids"], outcome["parameter_snapshots"], providers, profiles, sources.runtime, outcome["execution"])
        elif not isinstance(failure, dict) or set(failure) != {"case_id", "type", "message", "kind"} or failure["case_id"] != case["case_id"] or failure["kind"] != "declared_domain_prediction_failure_not_native_evidence" or not isinstance(failure["message"], str) or failure["type"] not in ("ValueError", "ArithmeticError", "OverflowError", "ZeroDivisionError", "FloatingPointError") or (failure["type"] == "ValueError" and failure["message"] not in holdout.DOMAIN_ERRORS):
            raise ValueError("Malformed domain prediction failure; infrastructure failures cannot qualify")


def check_holdout_plan(sources: BaselineSources, protocol: dict[str, Any], plan: dict[str, Any], bundle: dict[str, Any], model_path: Path) -> None:
    code, guard = _code_sha(), _sha(sources.commitments())
    _plan_check(sources, protocol, plan, bundle, _protocol_check(sources, protocol, model_path))
    if code != _code_sha() or guard != _sha(sources.commitments()):
        raise ValueError("Source changed during plan integrity verification")


def _case_comparison(plan: dict[str, Any], outcome: dict[str, Any], pairs: list[dict[str, Any]]) -> dict[str, Any]:
    view = {"coverage": plan["coverage_per_case"], **{key: plan[key] for key in ("runtime", "capture_code_sha256", "expected_kernel_sets")}}
    result = two._native_comparison(view, {"execution": outcome["execution"], "parameter_snapshots": outcome["parameter_snapshots"]}, pairs)
    order = []
    for stage in two.STAGE_IDS:
        execution = outcome["execution"]["stage_executions"][stage]
        if stage == "second":
            order.extend("second:" + name for name in two.second.ROOTS)
            order.extend("cross_invocation_bridge:" + name for name in two.second.ROOTS)
        for record in execution["records"]:
            payload = record["payload"]
            for name in payload["outputs"]:
                if name in execution["scalar_stages"]:
                    order.extend(stage + ":" + name + ":" + scalar for scalar in two.first.STAGES)
                if payload["opcode"] == "SOFTMAX":
                    order.append(stage + ":softmax_f32_bits")
                order.append(stage + ":" + name)
        order.extend(stage + ":plain:" + name for name in (("hidden.1",) if stage == "first" else ("hidden.1", "hidden.2")))
    divergence = next(({"repetition": index, "state": name, **comparison[name]["first_divergence"]} for index, comparison in enumerate(result["comparisons"]) for name in order if comparison[name]["first_divergence"] is not None), None)
    if divergence is not None:
        result["first_divergence"] = divergence
    return result


def _partial_comparison(plan: dict[str, Any], outcome: dict[str, Any], pairs: list[dict[str, Any]], error: Exception) -> dict[str, Any]:
    counts, retained, divergence = {}, [], None
    for index, repetition in enumerate(pairs):
        try:
            checked = _case_comparison(plan, outcome, [repetition] * 3)
            comparison = checked["comparisons"][0]
            retained.append({"repetition": index, "comparisons": comparison, "checks": checked["checks"]})
            for name, value in comparison.items():
                counts[name] = counts.get(name, 0) + value["mismatch_count"]
            if divergence is None and checked["first_divergence"] is not None:
                divergence = {**checked["first_divergence"], "repetition": index}
        except (KeyError, TypeError, ValueError, RuntimeError, IndexError, AttributeError):
            if divergence is None:
                divergence = {"repetition": index, "check": "native_coverage"}
    return {"two_layers_match": False, "native_coverage_complete": False, "mismatch_counts": counts,
            "aggregate_mismatch_count": sum(counts.values()), "retained_repetition_comparisons": retained,
            "mismatch_count_scope": "available complete paired scopes only; unavailable observations are not zero-mismatch evidence",
            "first_divergence": divergence or {"check": "native_coverage"},
            "abstention": {"type": type(error).__name__, "message": str(error), "kind": "incomplete_native_evidence_not_numerical_proof"}}


def holdout_report(protocol: dict[str, Any], plan: dict[str, Any], bundle: dict[str, Any], observations: list[dict[str, Any]], acquisition_guards: dict[str, bool] | None = None) -> dict[str, Any]:
    _check_hash(protocol, "protocol_sha256")
    _check_hash(plan, "plan_sha256")
    if protocol["scope"] != SCOPE or any(protocol.get(key) is not value for key, value in _flags().items()) or canonical_json(plan) != canonical_json(_seal(_plan_body(protocol, bundle), "plan_sha256")):
        raise ValueError("Recount requires own frozen raw-token protocol and complete bound case plan")
    guards = dict.fromkeys(GUARDS, True) if acquisition_guards is None else acquisition_guards
    guarded = set(guards) == set(GUARDS) and all(value is True for value in guards.values())
    cases, attempts, forwards = [], 0, 0
    if not plan["prediction_complete"]:
        if observations:
            raise ValueError("Incomplete predictions block ALL native observations")
        cases = [{"case_id": item["case_id"], "prediction_failure": item["prediction_failure"], "case_matches": False,
                  "native_coverage_complete": False, "mismatch_counts": {}, "aggregate_mismatch_count": 0,
                  "original_forward_count": 0, "capture_attempt_count": 0, "first_divergence": {"check": "prediction_incomplete"}} for item in bundle["cases"]]
    else:
        if not isinstance(observations, list) or [item["case_id"] for item in observations] != list(CASE_IDS):
            raise ValueError("Native evidence requires both declared cases in original order")
        for declared, outcome, observed in zip(protocol["cases"], bundle["cases"], observations):
            if set(observed) != {"case_id", "input_token_ids", "pairs"} or canonical_json(observed["input_token_ids"]) != canonical_json(declared["input_token_ids"]):
                raise ValueError("Native raw-token case binding mismatch")
            records = [record for repetition in observed["pairs"] for stage in two.STAGE_IDS for record in repetition.get(stage, {}).values()]
            count = sum(isinstance(record, dict) and "capture_failure" not in record for record in records)
            try:
                result = _case_comparison(plan, outcome, observed["pairs"])
            except (KeyError, TypeError, ValueError, RuntimeError, IndexError, AttributeError) as error:
                result = _partial_comparison(plan, outcome, observed["pairs"], error)
            cases.append({"case_id": declared["case_id"], **result, "case_matches": result["two_layers_match"] and count == 12,
                          "original_forward_count": count, "capture_attempt_count": len(records)})
            attempts += len(records)
            forwards += count
    coverage = {"all_predictions_complete": plan["prediction_complete"], "both_cases_present": len(cases) == 2,
                "all_native_coverage_complete": all(item["native_coverage_complete"] for item in cases),
                "all_24_calls_successful": forwards == 24 and attempts == 24, "acquisition_guards": guarded}
    matches = all(coverage.values()) and all(item["case_matches"] for item in cases)
    divergence = next(({"case_id": item["case_id"], **(item.get("first_divergence") or {"check": "native_count"})} for item in cases if not item["case_matches"]), None)
    if not plan["prediction_complete"]:
        failure = next(item["prediction_failure"] for item in bundle["cases"] if item["prediction_failure"] is not None)
        divergence = {"case_id": failure["case_id"], "check": "prediction_failure", "type": failure["type"], "message": failure["message"]}
    if divergence is None and not guarded:
        divergence = {"check": next((key for key, value in guards.items() if value is not True), "acquisition_guards")}
    return _seal({"schema_version": 1, "artifact_kind": "gemma_two_layers_holdout_report", "scope": SCOPE,
                  "protocol_sha256": protocol["protocol_sha256"], "plan_sha256": plan["plan_sha256"], "bundle_sha256": _sha(bundle),
                  "sources": plan["sources"], "code_sha256": plan["code_sha256"], "observations": observations, "cases": cases,
                  "required_case_ids": list(CASE_IDS), "coverage_per_case": plan["coverage_per_case"], "coverage_checks": coverage,
                  "prediction_complete": plan["prediction_complete"], "native_coverage_complete": coverage["all_native_coverage_complete"] and forwards == 24,
                  "required_forward_count": 24, "original_forward_count": forwards, "capture_attempt_count": attempts,
                  "forward_count_scope": "successfully returned records only; failed attempts may have partially executed; zero mismatches with incomplete counts do not establish agreement",
                  "mismatch_counts": {item["case_id"]: item["mismatch_counts"] for item in cases},
                  "aggregate_mismatch_count": sum(item["aggregate_mismatch_count"] for item in cases),
                  "first_divergence": divergence, "two_layer_holdout_matches": matches, "acquisition_guards": guards,
                  "native_scope": two.NATIVE_SCOPE, "kernel_role_scope_count_per_case": 44,
                  "unique_predicted_states_per_case": 66, "native_state_observations_per_paired_scope": 70,
                  "comparison_scopes_per_paired_scope": 115,
                  "bridge_evidence": "four cross-invocation value comparisons, not same-invocation pointer or launch correspondence",
                  **_flags(matches)}, "report_sha256")


def _frozen(protocol: dict[str, Any], plan: dict[str, Any], bundle: dict[str, Any], protocol_path: Path, plan_path: Path, bundle_path: Path) -> None:
    for path, payload in ((protocol_path, protocol), (plan_path, plan), (bundle_path, bundle)):
        require_frozen(path, payload)


def _live_snapshots(sources: BaselineSources, model: Any, protocol: dict[str, Any], bundle: dict[str, Any]) -> None:
    for case, outcome in zip(protocol["cases"], bundle["cases"]):
        if canonical_json(two._snapshot_model(sources.two_sources, model, case["input_token_ids"])) != canonical_json(outcome["parameter_snapshots"]):
            raise ValueError("All 29 fresh checkpoint/embedding snapshots for ALL cases must match frozen predictions")


def acquire_holdout(sources: BaselineSources, model: Any, protocol: dict[str, Any], plan: dict[str, Any], bundle: dict[str, Any], model_path: Path, protocol_path: Path, plan_path: Path, bundle_path: Path, frozen_inputs: tuple[Path, ...] = ()) -> dict[str, Any]:
    _frozen(protocol, plan, bundle, protocol_path, plan_path, bundle_path)
    if not plan["prediction_complete"]:
        raise ValueError("Incomplete predictions block ALL native acquisition")
    paths = (protocol_path, plan_path, bundle_path, *frozen_inputs)
    files, code, guard = two._file_guard(paths), _code_sha(), _sha(sources.commitments())
    context = _protocol_check(sources, protocol, model_path)
    _plan_check(sources, protocol, plan, bundle, context)
    _live_snapshots(sources, model, protocol, bundle)
    for case, outcome in zip(protocol["cases"], bundle["cases"]):
        _validate_snapshots(sources, case["input_token_ids"], outcome["parameter_snapshots"], context)
    _protocol_check(sources, protocol, model_path, context)
    _frozen(protocol, plan, bundle, protocol_path, plan_path, bundle_path)
    if files != two._file_guard(paths) or code != _code_sha() or guard != _sha(sources.commitments()):
        raise ValueError("Frozen source/checkpoint preparation changed before ANY native call")
    observations = []
    for case in protocol["cases"]:
        pairs = []
        for _ in range(3):
            repetition = {}
            for stage, capture in (("first", two.first_capture.capture_first_layer), ("second", two.second_capture.capture_second_layer)):
                pair = {}
                for kind, traced in (("traced", True), ("untraced", False)):
                    try:
                        pair[kind] = capture(model, case["input_token_ids"], traced)
                    except Exception as error:
                        pair[kind] = {"capture_failure": {"type": type(error).__name__, "message": str(error), "kind": "capture_abstention_not_numerical_evidence"}}
                repetition[stage] = pair
            pairs.append(repetition)
        observations.append({"case_id": case["case_id"], "input_token_ids": copy.deepcopy(case["input_token_ids"]), "pairs": pairs})
    guards = dict.fromkeys(GUARDS, False)
    try:
        _live_snapshots(sources, model, protocol, bundle)
        guards["checkpoint_unchanged"] = True
    except (ValueError, RuntimeError):
        pass
    try:
        guards["source_unchanged"] = code == _code_sha() and guard == _sha(sources.commitments())
        for case, outcome in zip(protocol["cases"], bundle["cases"]):
            _validate_snapshots(sources, case["input_token_ids"], outcome["parameter_snapshots"], context)
        guards["providers_unchanged"] = True
        _protocol_check(sources, protocol, model_path, context)
        guards["protocol_unchanged"] = True
    except (ValueError, RuntimeError, OSError):
        pass
    try:
        _frozen(protocol, plan, bundle, protocol_path, plan_path, bundle_path)
        guards["frozen_files_unchanged"] = files == two._file_guard(paths)
    except (OSError, ValueError):
        pass
    return holdout_report(protocol, plan, bundle, observations, guards)


def verify_holdout(sources: BaselineSources, protocol: dict[str, Any], plan: dict[str, Any], bundle: dict[str, Any], report: dict[str, Any], model_path: Path) -> dict[str, Any]:
    result = {"mode": "integrity_protocol_regeneration_full_source_snapshots_constituent_and_global_ledger_bridges_coverage_native_recount_only",
              "connected_numerical_recomputation_performed": False, "embedding_membership_revalidated_with_model": False,
              "previous_lineage_validation": "full_source_integrity_and_short_RMS_recomputation_not_previous_matmuls", **_flags()}
    try:
        code, guard = _code_sha(), _sha(sources.commitments())
        check_holdout_plan(sources, protocol, plan, bundle, model_path)
        _check_hash(report, "report_sha256")
        expected = holdout_report(protocol, plan, bundle, report["observations"], report["acquisition_guards"])
        valid = canonical_json(expected) == canonical_json(report) and code == _code_sha() and guard == _sha(sources.commitments())
        matches = valid and expected["two_layer_holdout_matches"]
        return {**result, "valid": valid, "two_layer_holdout_matches": matches, "prediction_complete": plan["prediction_complete"],
                "native_coverage_complete": expected["native_coverage_complete"], "first_divergence": expected["first_divergence"], **_flags(matches)}
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError, IndexError, OSError) as error:
        return {**result, "valid": False, "two_layer_holdout_matches": False, "reason": str(error)}


def reexecute_holdout(sources: BaselineSources, model: Any, protocol: dict[str, Any], plan: dict[str, Any], bundle: dict[str, Any], report: dict[str, Any], model_path: Path, protocol_path: Path, plan_path: Path, bundle_path: Path, workers: int = 4, frozen_inputs: tuple[Path, ...] = ()) -> dict[str, Any]:
    _frozen(protocol, plan, bundle, protocol_path, plan_path, bundle_path)
    paths = (protocol_path, plan_path, bundle_path, *frozen_inputs)
    files = two._file_guard(paths)
    checked = verify_holdout(sources, protocol, plan, bundle, report, model_path)
    if not checked["valid"]:
        return checked
    regenerated_plan, regenerated_bundle = build_holdout_plan(sources, model, protocol, model_path, protocol_path, workers, frozen_inputs)
    same = canonical_json(regenerated_plan) == canonical_json(plan) and canonical_json(regenerated_bundle) == canonical_json(bundle)
    if files != two._file_guard(paths):
        raise ValueError("Frozen files changed during ALL-case rebuild; no native calls allowed")
    replay = None
    if same:
        replay = acquire_holdout(sources, model, protocol, plan, bundle, model_path, protocol_path, plan_path, bundle_path, frozen_inputs) if plan["prediction_complete"] else holdout_report(protocol, plan, bundle, [])
    exact = same and canonical_json(replay) == canonical_json(report)
    matches = exact and replay is not None and replay["two_layer_holdout_matches"]
    return {"valid": exact, "mode": "fresh_ALL_cases_29_snapshots_and_63_nodes_before_ANY_24_separate_native_forwards" if plan["prediction_complete"] else "prediction_failure_reexecution_only_no_native",
            "predictions_recomputed_exact": same, "reexecution_exact": exact, "prediction_complete": plan["prediction_complete"],
            "connected_numerical_recomputation_performed": True, "embedding_membership_revalidated_with_model": True,
            "two_layer_holdout_matches": matches, "replay_report": replay if not exact else None, **_flags(matches)}


def holdout_summary(plan: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in report.items() if key not in ("observations", "cases", "report_sha256")}
    body["cases"] = [{key: value for key, value in case.items() if key != "comparisons"} for case in report["cases"]]
    body.update(source_report_sha256=report["report_sha256"], providers=plan["providers"], profiles=plan["profiles"])
    return _seal(body, "summary_sha256")
