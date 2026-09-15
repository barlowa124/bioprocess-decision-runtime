from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

from . import gemma_first_layer as first
from . import gemma_first_layer_capture as capture
from .gemma_rotary_slice import _sha, _seal, _check_hash
from .serialization import canonical_json

SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
BASELINE_PLAN_SHA256 = "d1f494eeb379eb9c694da26e6c503789702e9de11652492168d2584176a13636"
SEED = 0xC4D18A73
CASE_IDS = ("distinct_tokens", "repeated_motif")
ASSETS = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json")
GENERATOR = "sorted allowed IDs; xorshift32 shifts 13,17,5 masked uint32; partial Fisher-Yates indices 0..34, j=index+word()%(pool_count-index); first30 distinct; last5 cycled6; deterministic index sampling, not a statistical-uniformity claim"
SCOPE = "Prospective predeclared two 30-token RAW-ID numerical cases for the fixed Gemma3 270m checkpoint, input_ids to hidden.1 through the unchanged 34-node/36-state dependency cone. Excludes every token ID in the declared previously observed baseline, not unknown historical runs. Reuses empirical primitive specifications, never intermediate activations. Not natural-language, fresh-prompt, task-performance, clinical, unrestricted-domain, full-first-layer or hardware qualification. No later layer, final model norm or logits."
FALSE_FLAGS = tuple(dict.fromkeys((*first.FALSE_FLAGS, "generic_domain_qualified", "unknown_historical_independence", "task_performance_qualified", "clinical_semantics_established")))
DOMAIN_ERRORS = frozenset(("Lookup accepts positive normal float32 encodings only", "Rsqrt exponent has not passed complete domain validation", "GELU lookup accepts finite bfloat16 encodings only", "Exponential specification accepts nonpositive non-NaN float32 encodings only", "NaN and infinity inputs are outside finite bfloat16 semantics", "NaN and infinity are outside finite float32 semantics", "Nonfinite float32 carry is outside this experiment", "Subnormal operands are outside the alignment hypothesis"))


def _code_sha() -> str:
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Holdout source changed after import")
    return _sha({"module": SOURCE_SHA256, "first_layer": first._code_sha(), "capture": capture._code_sha()})


def _flags() -> dict[str, Any]:
    return {**{name: False for name in FALSE_FLAGS}, "prospective_raw_token_cases": True,
            "excluded_from_declared_baseline": True, "empirical_primitive_data_reused": True}


class _CachedFirstLayerSources:
    def __init__(self, source: first.FirstLayerSources):
        self.source = source
        self.context = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.source, name)

    def validate(self) -> tuple[Any, ...]:
        if self.context is None:
            self.context = self.source.validate()
        return self.context


@dataclass(frozen=True)
class BaselineSources:
    first_sources: first.FirstLayerSources
    baseline_plan: dict[str, Any]
    baseline_bundle: dict[str, Any]
    baseline_report: dict[str, Any]

    def commitments(self) -> dict[str, Any]:
        return {"first_sources": self.first_sources.commitments(), "baseline_plan_sha256": _sha(self.baseline_plan),
                "baseline_bundle_sha256": _sha(self.baseline_bundle), "baseline_report_sha256": _sha(self.baseline_report)}

    def validate(self) -> tuple[Any, ...]:
        code, guard = _code_sha(), _sha(self.commitments())
        if self.baseline_plan.get("plan_sha256") != BASELINE_PLAN_SHA256 or self.baseline_plan.get("scope") != first.SCOPE:
            raise ValueError("Holdout requires the declared fixed-case baseline plan and unchanged scope")
        cached_source = _CachedFirstLayerSources(self.first_sources)
        checked = first.verify_first_layer(cached_source, self.baseline_plan, self.baseline_bundle, self.baseline_report)
        if any(checked.get(name) is not True for name in ("valid", "first_layer_matches", "connected_first_layer_independently_recomputed")):
            raise ValueError("Holdout requires passing full baseline source, trace, snapshot and native checks")
        context = cached_source.validate()
        if canonical_json(context[0]) != canonical_json(self.baseline_plan["input_token_ids"]):
            raise ValueError("Declared baseline tokens differ from root source tokens")
        roles = first._native_kernel_roles(self.first_sources.program)
        kernels = self.baseline_report["observations"][0]["traced"]["kernels"]
        if len(roles) != 22 or set(kernels) != set(roles) or any(not names or names != sorted(set(names)) for names in kernels.values()):
            raise ValueError("Baseline must bind all 22 distinct kernel-role symbol sets")
        if any(pair["traced"]["kernels"] != kernels for pair in self.baseline_report["observations"]) or len(self.baseline_report["observations"]) != 3:
            raise ValueError("Baseline kernel sets are not repeat stable")
        if code != _code_sha() or guard != _sha(self.commitments()):
            raise ValueError("Baseline source changed during validation")
        return context


def _assets(model_path: Path) -> dict[str, str]:
    if not model_path.is_dir():
        raise ValueError("Tokenizer must be an existing local model directory")
    assets = {name: hashlib.sha256((model_path / name).read_bytes()).hexdigest() for name in ASSETS if (model_path / name).is_file()}
    if not assets:
        raise ValueError("No whitelisted local tokenizer assets")
    return assets


def tokenizer_context(model_path: Path, program: dict[str, Any], baseline_ids: list[list[int]]) -> tuple[dict[str, Any], list[int]]:
    from transformers import AutoTokenizer
    assets = _assets(model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, use_fast=True, trust_remote_code=False)
    values = list(tokenizer.get_vocab().values())
    special = list(tokenizer.all_special_ids)
    if any(type(value) is not int for value in values + special):
        raise ValueError("Tokenizer vocabulary and special IDs must be integers")
    vocabulary = sorted(set(values))
    baseline = sorted(set(first._tokens(program, baseline_ids).reshape(-1).tolist()))
    excluded = set(special) | set(baseline)
    pool = [value for value in vocabulary if 0 <= value < program["configuration"]["vocabulary_size"] and value not in excluded]
    if len(pool) < 35:
        raise ValueError("Tokenizer requires at least 35 allowed distinct embedding-domain IDs")
    if assets != _assets(model_path):
        raise ValueError("Local tokenizer assets changed while loading")
    return {"assets_sha256": assets, "tokenizer_class": type(tokenizer).__module__ + "." + type(tokenizer).__qualname__,
            "package_versions": {name: version(name) for name in ("transformers", "tokenizers")},
            "vocabulary_id_set_sha256": _sha(vocabulary), "vocabulary_id_count": len(vocabulary),
            "all_special_ids": special, "all_special_ids_sha256": _sha(special),
            "baseline_token_ids": baseline, "baseline_token_ids_sha256": _sha(baseline),
            "declared_baseline_sequence_sha256": _sha(baseline_ids), "allowed_pool_sha256": _sha(pool), "allowed_pool_count": len(pool),
            "embedding_vocabulary_size": program["configuration"]["vocabulary_size"],
            "loading": "CPU tokenizer only; local_files_only=True; use_fast=True; trust_remote_code=False; no decode or private settings"}, pool


def _cases(pool: list[int]) -> list[dict[str, Any]]:
    if len(pool) < 35 or pool != sorted(set(pool)):
        raise ValueError("Expected sorted unique pool of at least 35 IDs")
    selected, state = list(pool), SEED
    for index in range(35):
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        state &= 0xFFFFFFFF
        other = index + state % (len(selected) - index)
        selected[index], selected[other] = selected[other], selected[index]
    return [{"case_id": name, "input_token_ids": [ids], "case_sha256": _sha({"case_id": name, "input_token_ids": [ids]})}
            for name, ids in zip(CASE_IDS, (selected[:30], selected[30:35] * 6))]


def _protocol_body(baseline: BaselineSources, model_path: Path, context: tuple[Any, ...]) -> dict[str, Any]:
    ids, providers, profiles = context
    source = baseline.first_sources
    metadata, pool = tokenizer_context(model_path, source.program, ids)
    return {"schema_version": 1, "artifact_kind": "gemma_first_layer_holdout_protocol", "scope": SCOPE, "seed": SEED,
            "generator_rule": GENERATOR, "cases": _cases(pool), "required_case_ids": list(CASE_IDS), "tokenizer": metadata,
            "sources": baseline.commitments(), "code_sha256": _code_sha(), "first_layer_code_sha256": first._code_sha(),
            "capture_code_sha256": capture._code_sha(), "runtime": copy.deepcopy(source.runtime),
            "program_sha256": source.program["program_sha256"], "program_payload_sha256": _sha(source.program),
            "model_binding": copy.deepcopy(baseline.baseline_plan["model_binding"]), "providers": providers.commitments(), "profiles": copy.deepcopy(profiles),
            "baseline_kernel_symbols": copy.deepcopy(baseline.baseline_report["observations"][0]["traced"]["kernels"]),
            "coverage_per_case": first._coverage(source.program), "case_count": 2, "repetitions_per_case": 3, "original_forward_count": 12,
            "observation_rule": "all cases in declared order, three traced/plain pairs each; freeze protocol before prediction and ALL predictions before ANY native forward; retain all failures; no refit, resampling, replacement or output-based selection",
            "epoch_scope": "new_raw_token_cases_predeclared_before_their_acquisition; baseline_was_previously_observed",
            **_flags()}


def build_holdout_protocol(baseline: BaselineSources, model_path: Path) -> dict[str, Any]:
    code, guard = _code_sha(), _sha(baseline.commitments())
    result = _seal(_protocol_body(baseline, model_path, baseline.validate()), "protocol_sha256")
    if code != _code_sha() or guard != _sha(baseline.commitments()):
        raise ValueError("Source changed during protocol declaration")
    return result


def _protocol_check(baseline: BaselineSources, protocol: dict[str, Any], model_path: Path, context: tuple[Any, ...] | None = None) -> tuple[Any, ...]:
    _check_hash(protocol, "protocol_sha256")
    context = baseline.validate() if context is None else context
    expected = _seal(_protocol_body(baseline, model_path, context), "protocol_sha256")
    if canonical_json(protocol) != canonical_json(expected):
        raise ValueError("Holdout protocol differs from exact current source/tokenizer/seed/case regeneration; no adaptive selection")
    return context


def require_frozen(path: Path, payload: dict[str, Any]) -> None:
    if not path.is_file() or canonical_json(json.loads(path.read_text(encoding="utf-8"))) != canonical_json(payload):
        raise ValueError("A matching immutable protocol or prediction artifact must already exist: " + str(path))


def write_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _failure(error: ValueError | ArithmeticError, case_id: str) -> dict[str, str]:
    return {"case_id": case_id, "type": type(error).__name__, "message": str(error), "kind": "declared_domain_prediction_failure_not_native_evidence"}


def _domain_failure(error: ValueError | ArithmeticError) -> bool:
    return isinstance(error, ArithmeticError) or str(error) in DOMAIN_ERRORS


def _plan_body(protocol: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    if set(bundle) != {"protocol_sha256", "cases"} or bundle["protocol_sha256"] != protocol["protocol_sha256"] or [item["case_id"] for item in bundle["cases"]] != list(CASE_IDS) or [item["case_id"] for item in protocol["cases"]] != list(CASE_IDS):
        raise ValueError("Plan requires every declared case exactly once in original order")
    if any((item["execution"] is None) != (item["prediction_failure"] is not None) for item in bundle["cases"]):
        raise ValueError("Each case must preserve either a complete execution or an explicit prediction failure")
    cases = []
    for declared, outcome in zip(protocol["cases"], bundle["cases"]):
        snapshots = outcome["parameter_snapshots"]
        item = {**copy.deepcopy(declared), "input_ids": first._descriptor(first._array(declared["input_token_ids"], [1, 30], "torch.int64")),
                "parameter_snapshots": {name: {key: value for key, value in snapshot.items() if key != "bits"} for name, snapshot in snapshots.items()},
                "prediction_failure": outcome["prediction_failure"]}
        execution = outcome["execution"]
        if execution is not None:
            item.update(state_descriptors={name: first._descriptor(first._array(value, protocol["coverage_per_case"]["states"][name]["shape"], protocol["coverage_per_case"]["states"][name]["dtype"])) for name, value in execution["state_bits"].items()},
                        trace_root=execution["records"][-1]["record_hash"], execution_sha256=_sha(execution))
        cases.append(item)
    return {"schema_version": 1, "artifact_kind": "gemma_first_layer_holdout_plan", "scope": SCOPE,
            "protocol_sha256": protocol["protocol_sha256"], "cases": cases, "bundle_sha256": _sha(bundle),
            "prediction_complete": all(item["prediction_failure"] is None for item in cases),
            **{key: copy.deepcopy(protocol[key]) for key in ("sources", "code_sha256", "runtime", "program_sha256", "program_payload_sha256", "model_binding", "providers", "profiles", "coverage_per_case", "required_case_ids", "baseline_kernel_symbols", "case_count", "repetitions_per_case", "original_forward_count", "observation_rule", "epoch_scope")},
            "parameter_snapshot_membership": "fresh full-checkpoint binding with selected embedding rows; default verification does not revalidate membership without model",
            "connected_first_layer_independently_recomputed": False, **_flags()}


def _predict(baseline: BaselineSources, model: Any, protocol: dict[str, Any], context: tuple[Any, ...], workers: int) -> tuple[dict[str, Any], dict[str, Any]]:
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Holdout workers must be 1..4")
    source, (_, providers, profiles) = baseline.first_sources, context
    code, guard = _code_sha(), _sha(baseline.commitments())
    outcomes = []
    for case in protocol["cases"]:
        ids = case["input_token_ids"]
        snapshots = first._snapshot_model(source, model, ids)
        parameters = first.check_parameter_snapshots(source.program, ids, snapshots)
        first._profiles(profiles)
        providers.validate(source.runtime, parameters[first.FREQUENCY])
        execution, failure = None, None
        try:
            execution = first.execute_first_layer(source.program, ids, snapshots, providers, profiles, source.runtime, workers)
        except (ValueError, ArithmeticError) as error:
            if not _domain_failure(error):
                raise
            failure = _failure(error, case["case_id"])
        if canonical_json(first._snapshot_model(source, model, ids)) != canonical_json(snapshots):
            raise ValueError("Live checkpoint parameter snapshots changed during case prediction")
        providers.validate(source.runtime, parameters[first.FREQUENCY])
        if code != _code_sha() or guard != _sha(baseline.commitments()):
            raise ValueError("Source lineage changed during case prediction")
        outcomes.append({"case_id": case["case_id"], "parameter_snapshots": snapshots, "execution": execution, "prediction_failure": failure})
    bundle = {"protocol_sha256": protocol["protocol_sha256"], "cases": outcomes}
    return _seal(_plan_body(protocol, bundle), "plan_sha256"), bundle


def build_holdout_plan(baseline: BaselineSources, model: Any, protocol: dict[str, Any], model_path: Path, protocol_path: Path, workers: int = 4) -> tuple[dict[str, Any], dict[str, Any]]:
    require_frozen(protocol_path, protocol)
    context = _protocol_check(baseline, protocol, model_path)
    result = _predict(baseline, model, protocol, context, workers)
    require_frozen(protocol_path, protocol)
    _protocol_check(baseline, protocol, model_path, context)
    return result


def _plan_check(baseline: BaselineSources, protocol: dict[str, Any], plan: dict[str, Any], bundle: dict[str, Any], context: tuple[Any, ...]) -> None:
    _check_hash(plan, "plan_sha256")
    if set(bundle) != {"protocol_sha256", "cases"} or bundle["protocol_sha256"] != protocol["protocol_sha256"] or not isinstance(bundle["cases"], list) or [item["case_id"] for item in bundle["cases"]] != list(CASE_IDS):
        raise ValueError("Holdout prediction bundle must contain every original case in order")
    source, (_, providers, profiles) = baseline.first_sources, context
    for case, outcome in zip(protocol["cases"], bundle["cases"]):
        if set(outcome) != {"case_id", "parameter_snapshots", "execution", "prediction_failure"}:
            raise ValueError("Unexpected holdout prediction outcome material")
        parameters = first.check_parameter_snapshots(source.program, case["input_token_ids"], outcome["parameter_snapshots"])
        first._profiles(profiles)
        providers.validate(source.runtime, parameters[first.FREQUENCY])
        failure = outcome["prediction_failure"]
        if failure is None:
            first.check_execution(source.program, case["input_token_ids"], outcome["parameter_snapshots"], providers, profiles, source.runtime, outcome["execution"])
        else:
            if outcome["execution"] is not None or not isinstance(failure, dict) or set(failure) != {"case_id", "type", "message", "kind"} or failure["case_id"] != case["case_id"] or failure["kind"] != "declared_domain_prediction_failure_not_native_evidence" or not isinstance(failure["message"], str):
                raise ValueError("Malformed preserved prediction failure")
            if failure["type"] not in ("ValueError", "ArithmeticError", "OverflowError", "ZeroDivisionError", "FloatingPointError") or (failure["type"] == "ValueError" and failure["message"] not in DOMAIN_ERRORS):
                raise ValueError("Infrastructure errors are not declared-domain prediction failures")
    if canonical_json(plan) != canonical_json(_seal(_plan_body(protocol, bundle), "plan_sha256")):
        raise ValueError("Frozen holdout prediction plan/bundle binding mismatch")


def check_holdout_plan(baseline: BaselineSources, protocol: dict[str, Any], plan: dict[str, Any], bundle: dict[str, Any], model_path: Path) -> None:
    context = _protocol_check(baseline, protocol, model_path)
    _plan_check(baseline, protocol, plan, bundle, context)


def _case_comparison(plan: dict[str, Any], outcome: dict[str, Any], pairs: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(pairs, list) or len(pairs) != 3:
        raise ValueError("Each case requires all three original traced/plain pairs")
    execution, coverage = outcome["execution"], plan["coverage_per_case"]
    comparisons, controls, kernel_sets = [], [], []
    checks = {name: True for name in ("capture_checks", "geometry", "runtime", "capture_code", "baseline_kernel_symbols", "same_cuda_device")}
    for pair in pairs:
        if set(pair) != {"traced", "untraced"}:
            raise ValueError("Native pair coverage mismatch")
        traced, plain = pair["traced"], pair["untraced"]
        if any("capture_failure" in record for record in (traced, plain)):
            raise ValueError("Native capture abstained; no complete numerical evidence")
        if set(traced["state_bits"]) != set(coverage["states"]) or set(traced["geometry"]) != set(coverage["states"]) or set(plain["state_bits"]) != {"hidden.1"} or set(plain["geometry"]) != {"hidden.1"}:
            raise ValueError("Native state/geometry coverage mismatch")
        if any(plain.get(key) for key in ("scalar_stages", "softmax_f32_bits", "kernels")):
            raise ValueError("Plain control must use only minimal hidden.1 hook")
        result = {}
        for name, declaration in coverage["states"].items():
            result[name] = first._comparison(execution["state_bits"][name], traced["state_bits"][name], declaration["shape"], declaration["dtype"])
            checks["geometry"] &= first._geometry(traced["geometry"][name], declaration["shape"], declaration["dtype"])
        if set(traced["scalar_stages"]) != set(execution["scalar_stages"]):
            raise ValueError("Native RMS scalar coverage mismatch")
        for name, stages in execution["scalar_stages"].items():
            native = traced["scalar_stages"][name]
            if set(native) != {*first.STAGES, "mean_input_metadata"}:
                raise ValueError("Native RMS auxiliary stage fields mismatch")
            for key in first.STAGES:
                result[name + ":" + key] = first._comparison(stages[key], native[key], [len(stages[key])], "torch.float32")
            metadata, shape = native["mean_input_metadata"], coverage["states"][name]["shape"]
            strides = metadata.get("input_strides")
            checks["geometry"] &= metadata.get("input_shape") == shape and metadata.get("input_dtype") == "torch.float32" and metadata.get("axes") == [-1] and metadata.get("keepdim") is True and metadata.get("alignment_mod16") == 0 and isinstance(strides, list) and len(strides) == len(shape) and strides[-1] == 1 and all(type(stride) is int and stride > 0 and stride % 4 == 0 for stride in strides[:-1])
        result["softmax_f32_bits"] = first._comparison(execution["softmax_f32_bits"], traced["softmax_f32_bits"], [1, 4, 30, 30], "torch.float32")
        controls.append(first._comparison(execution["state_bits"]["hidden.1"], plain["state_bits"]["hidden.1"], [1, 30, 640], "torch.bfloat16"))
        checks["geometry"] &= first._geometry(plain["geometry"]["hidden.1"], [1, 30, 640], "torch.bfloat16")
        checks["same_cuda_device"] &= len({value["device"] for record in (traced, plain) for value in record["geometry"].values()}) == 1
        for record in (traced, plain):
            required = {"token_commitment_unchanged", "stopped_at_decoder_tuple", "decoder_tuple_length_one", "no_later_layer", "no_final_norm", "no_lm_head", "all_needed_states_present", "code_unchanged", "runtime_unchanged"}
            required.update({"original_operations_once", "all_operand_links_unchanged", "rms_scalar_stages_complete", "softmax_f32_complete"} if record is traced else {"plain_minimal_control"})
            checks["capture_checks"] &= isinstance(record.get("checks"), dict) and required.issubset(record["checks"]) and all(value is True for value in record["checks"].values())
            checks["runtime"] &= canonical_json(record["runtime_before"]) == canonical_json(plan["runtime"]) == canonical_json(record["runtime_after"])
            checks["capture_code"] &= record["code_before"] == capture._code_sha() == record["code_after"]
        checks["baseline_kernel_symbols"] &= traced["kernels"] == plan["baseline_kernel_symbols"]
        kernel_sets.append(traced["kernels"])
        comparisons.append(result)
    checks["repeat_stable_kernel_sets"] = all(item == kernel_sets[0] for item in kernel_sets)
    counts = {name: sum(item[name]["mismatch_count"] for item in comparisons) for name in comparisons[0]}
    counts["plain_hidden.1"] = sum(item["mismatch_count"] for item in controls)
    first_divergence = next(({"repetition": index, "state": name, **result["first_divergence"]} for index, comparison in enumerate(comparisons) for name, result in comparison.items() if result["first_divergence"] is not None), None)
    if first_divergence is None:
        first_divergence = next(({"repetition": index, "state": "plain_hidden.1", **item["first_divergence"]} for index, item in enumerate(controls) if item["first_divergence"] is not None), None)
    if first_divergence is None:
        first_divergence = next(({"check": name} for name, passed in checks.items() if not passed), None)
    return {"case_id": outcome["case_id"], "comparisons": comparisons, "untraced_controls": controls, "checks": checks,
            "mismatch_counts": counts, "first_divergence": first_divergence, "native_coverage_complete": True,
            "case_matches": all(checks.values()) and not any(counts.values())}


def holdout_report(protocol: dict[str, Any], plan: dict[str, Any], bundle: dict[str, Any], observations: list[dict[str, Any]]) -> dict[str, Any]:
    _check_hash(protocol, "protocol_sha256")
    _check_hash(plan, "plan_sha256")
    if plan["scope"] != SCOPE or plan["protocol_sha256"] != protocol["protocol_sha256"] or canonical_json(plan) != canonical_json(_seal(_plan_body(protocol, bundle), "plan_sha256")):
        raise ValueError("Native recount requires frozen holdout scope and predictions")
    if not plan["prediction_complete"]:
        if observations:
            raise ValueError("Incomplete predictions block every native observation")
        cases = [{"case_id": item["case_id"], "prediction_failure": item["prediction_failure"], "case_matches": False, "native_coverage_complete": False} for item in bundle["cases"]]
    else:
        if not isinstance(observations, list) or [item["case_id"] for item in observations] != list(CASE_IDS):
            raise ValueError("Native observations must contain both original cases in order")
        cases = []
        for declared, outcome, observed in zip(protocol["cases"], bundle["cases"], observations):
            if set(observed) != {"case_id", "input_token_ids", "pairs"} or observed["input_token_ids"] != declared["input_token_ids"]:
                raise ValueError("Native case input binding mismatch")
            try:
                result = _case_comparison(plan, outcome, observed["pairs"])
            except (ValueError, KeyError, TypeError, IndexError) as error:
                result = {"case_id": declared["case_id"], "case_matches": False, "native_coverage_complete": False,
                          "abstention": {"type": type(error).__name__, "message": str(error), "kind": "unsupported_or_incomplete_native_evidence"},
                          "first_divergence": {"check": "native_coverage"}}
            cases.append(result)
    matches = plan["prediction_complete"] and len(cases) == 2 and all(item["case_matches"] for item in cases)
    first_divergence = next(({"case_id": item["case_id"], **item.get("first_divergence", {"check": "prediction_incomplete"})} for item in cases if not item["case_matches"]), None)
    counts = {item["case_id"]: item.get("mismatch_counts") for item in cases}
    capture_count = sum(len(pair) for item in observations for pair in item["pairs"])
    forward_count = sum(isinstance(record, dict) and "capture_failure" not in record for item in observations for pair in item["pairs"] for record in pair.values())
    return _seal({"schema_version": 1, "artifact_kind": "gemma_first_layer_holdout_report", "scope": SCOPE,
                  "protocol_sha256": protocol["protocol_sha256"], "plan_sha256": plan["plan_sha256"], "bundle_sha256": _sha(bundle),
                  "observations": observations, "cases": cases, "required_case_ids": list(CASE_IDS), "coverage_per_case": plan["coverage_per_case"],
                  "prediction_complete": plan["prediction_complete"], "native_coverage_complete": all(item["native_coverage_complete"] for item in cases),
                  "original_forward_count": forward_count, "capture_attempt_count": capture_count, "required_forward_count": 12, "mismatch_counts": counts,
                  "forward_count_scope": "successfully returned capture records only; failed attempts may have partially executed; completeness requires all native checks",
                  "aggregate_mismatch_count": sum(sum(value.values()) for value in counts.values() if value is not None),
                  "first_divergence": first_divergence, "holdout_matches": matches,
                  "connected_first_layer_independently_recomputed": matches,
                  "kernel_provenance": "all 22 baseline distinct-symbol sets per case, not launch order or arithmetic-internal proof", **_flags()}, "report_sha256")


def acquire_holdout(baseline: BaselineSources, model: Any, protocol: dict[str, Any], plan: dict[str, Any], bundle: dict[str, Any], model_path: Path, protocol_path: Path, plan_path: Path, bundle_path: Path) -> dict[str, Any]:
    for path, payload in ((protocol_path, protocol), (plan_path, plan), (bundle_path, bundle)):
        require_frozen(path, payload)
    context = _protocol_check(baseline, protocol, model_path)
    _plan_check(baseline, protocol, plan, bundle, context)
    if not plan["prediction_complete"]:
        raise ValueError("Incomplete prediction blocks native acquisition; preserve the failed plan and bundle")
    code, guard = _code_sha(), _sha(baseline.commitments())
    for case, outcome in zip(protocol["cases"], bundle["cases"]):
        if canonical_json(first._snapshot_model(baseline.first_sources, model, case["input_token_ids"])) != canonical_json(outcome["parameter_snapshots"]):
            raise ValueError("Fresh acquisition checkpoint/embedding membership differs from frozen predictions")
    observations = []
    for case in protocol["cases"]:
        pairs = []
        for _ in range(3):
            pair = {}
            for name, traced in (("traced", True), ("untraced", False)):
                try:
                    pair[name] = capture.capture_first_layer(model, case["input_token_ids"], traced)
                except ValueError as error:
                    pair[name] = {"capture_failure": {"type": type(error).__name__, "message": str(error), "kind": "capture_abstention_not_numerical_evidence"}}
            pairs.append(pair)
        observations.append({"case_id": case["case_id"], "input_token_ids": copy.deepcopy(case["input_token_ids"]), "pairs": pairs})
    for case, outcome in zip(protocol["cases"], bundle["cases"]):
        if canonical_json(first._snapshot_model(baseline.first_sources, model, case["input_token_ids"])) != canonical_json(outcome["parameter_snapshots"]):
            raise ValueError("Checkpoint changed across native acquisition")
    if code != _code_sha() or guard != _sha(baseline.commitments()):
        raise ValueError("Source changed across native acquisition")
    _protocol_check(baseline, protocol, model_path, context)
    for path, payload in ((protocol_path, protocol), (plan_path, plan), (bundle_path, bundle)):
        require_frozen(path, payload)
    return holdout_report(protocol, plan, bundle, observations)


def verify_holdout(baseline: BaselineSources, protocol: dict[str, Any], plan: dict[str, Any], bundle: dict[str, Any], report: dict[str, Any], model_path: Path) -> dict[str, Any]:
    result = {"mode": "integrity_protocol_regeneration_trace_snapshot_coverage_native_recount_only",
              "connected_numerical_recomputation_performed": False, "embedding_membership_revalidated_with_model": False,
              "previous_lineage_validation": "full_source_integrity_and_short_RMS_recomputation_not_all_previous_matmuls", **_flags()}
    try:
        code, guard = _code_sha(), _sha(baseline.commitments())
        check_holdout_plan(baseline, protocol, plan, bundle, model_path)
        _check_hash(report, "report_sha256")
        expected = holdout_report(protocol, plan, bundle, report["observations"])
        valid = canonical_json(expected) == canonical_json(report) and code == _code_sha() and guard == _sha(baseline.commitments())
        return {**result, "valid": valid, "holdout_matches": valid and expected["holdout_matches"], "prediction_complete": plan["prediction_complete"],
                "connected_first_layer_independently_recomputed": valid and expected["holdout_matches"], "first_divergence": expected["first_divergence"]}
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError, IndexError) as error:
        return {**result, "valid": False, "holdout_matches": False, "connected_first_layer_independently_recomputed": False, "reason": str(error)}


def reexecute_holdout(baseline: BaselineSources, model: Any, protocol: dict[str, Any], plan: dict[str, Any], bundle: dict[str, Any], report: dict[str, Any], model_path: Path, protocol_path: Path, plan_path: Path, bundle_path: Path, workers: int = 4) -> dict[str, Any]:
    for path, payload in ((protocol_path, protocol), (plan_path, plan), (bundle_path, bundle)):
        require_frozen(path, payload)
    checked = verify_holdout(baseline, protocol, plan, bundle, report, model_path)
    if not checked["valid"]:
        return checked
    regenerated_plan, regenerated_bundle = build_holdout_plan(baseline, model, protocol, model_path, protocol_path, workers)
    same = canonical_json(regenerated_plan) == canonical_json(plan) and canonical_json(regenerated_bundle) == canonical_json(bundle)
    replay = None
    if same and plan["prediction_complete"]:
        replay = acquire_holdout(baseline, model, protocol, plan, bundle, model_path, protocol_path, plan_path, bundle_path)
    exact = same and (canonical_json(replay) == canonical_json(report) if plan["prediction_complete"] else not report["observations"])
    return {"valid": exact, "mode": "all_cases_fresh_prediction_before_any_twelve_native_forward_replay" if plan["prediction_complete"] else "prediction_failure_reexecution_only_no_native",
            "predictions_recomputed_exact": same, "reexecution_exact": exact, "prediction_complete": plan["prediction_complete"],
            "connected_numerical_recomputation_performed": True, "embedding_membership_revalidated_with_model": True,
            "holdout_matches": exact and replay is not None and replay["holdout_matches"],
            "connected_first_layer_independently_recomputed": exact and replay is not None and replay["holdout_matches"], **_flags()}


def holdout_summary(plan: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in report.items() if key not in ("observations", "cases", "report_sha256")}
    body["cases"] = [{key: value for key, value in item.items() if key not in ("comparisons", "untraced_controls")} for item in report["cases"]]
    body.update(source_report_sha256=report["report_sha256"], sources=plan["sources"], code_sha256=plan["code_sha256"], providers=plan["providers"])
    return _seal(body, "summary_sha256")
