from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import gemma_first_layer as first
from . import gemma_second_layer as second
from . import gemma_first_layer_capture as first_capture
from . import gemma_second_layer_capture as second_capture
from .gemma_rotary_slice import _sha, _seal, _check_hash
from .operational_semantics import append_chain_record, verify_trace_chain
from .serialization import canonical_json

SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
SECOND_PLAN_SHA256 = "affd51af91c0e6d6c938a167e73b90830b65e9bc30b5968e251b476621fb82de"
SECOND_REPORT_SHA256 = "2479b53c366fb6b1f85fe0f9b0f035f4b45f52458db727b5c22df62741643649"
NATIVE_SCOPE = "separate_prefix_and_second_layer_stopped_forwards"
SCOPE = "Fixed previously observed Gemma3 270m checkpoint and 30-token baseline only: fresh input_ids-to-hidden.2 connected independent prediction, 63 dependency-cone instructions and 66 unique produced states. Both actual decoder layers 0/1 use local RoPE and sliding window 512; first full-attention index is 5. Fresh checkpoint snapshots for both stages, not stored hidden.1 or intermediate predictions. Empirical rsqrt/exp/GELU and checked fixed-position local rotary provider specifications remain reused. Native evidence consists of separate original prefix-stopped and second-layer-stopped invocations, not a same-invocation whole trace or pointer/launch bridge proof. Twelve RMS norms contribute 1620 FP32 scalar positions and two softmax outputs contribute 7200 FP32 positions per paired scope, not all FP32 internals. Previous first-layer holdouts and layer-one evidence are prerequisites, not new two-layer cases. No fresh-prompt/holdout, future layer 2, final norm, logits, full-layer/model, hardware or unrestricted qualification."
FALSE_FLAGS = tuple(sorted((set(first.FALSE_FLAGS) | set(second.FALSE_FLAGS) | {
    "stored_boundary_predictions_used", "native_same_invocation_fullcoverage", "same_invocation_two_layer_trace",
    "fresh_two_layer_holdout", "all_65_prefix_instructions_claimed"}) - {
    "connected_two_layers_independently_recomputed", "connected_first_layer_independently_recomputed"}))
STAGE_IDS = ("first", "second")
GUARDS = ("checkpoint_unchanged", "source_unchanged", "frozen_files_unchanged")


def _flags(connected: bool = False) -> dict[str, bool]:
    return {**dict.fromkeys(FALSE_FLAGS, False), "connected_two_layers_independently_recomputed": connected,
            "connected_first_layer_independently_recomputed": connected, "empirical_primitive_data_reused": True,
            "empirical_providers_reused": True}


def _code_sha() -> str:
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Two-layer source changed after import")
    return _sha({"module": SOURCE_SHA256, "first": first._code_sha(), "second": second._code_sha(),
                 "first_capture": first_capture._code_sha(), "second_capture": second_capture._code_sha()})


def two_layer_instructions(program: dict[str, Any]) -> list[dict[str, Any]]:
    stages = first.first_layer_instructions(program) + second.second_layer_instructions(program)
    needed, selected = {"hidden.2"}, []
    for node in reversed(program["instructions"]):
        if needed.intersection(node["outputs"]):
            selected.append(node)
            needed.difference_update(node["outputs"])
            needed.update(node["inputs"])
    selected.reverse()
    if needed != {"input_ids"} or selected != stages or [node["id"] for node in selected] != [f"i{index:04d}" for index in range(65) if index not in (3, 5)]:
        raise ValueError("Two-layer exact 63-node backward closure/root/order mismatch")
    available = {"input_ids"}
    for node in selected:
        if any(name not in available for name in node["inputs"]) or available.intersection(node["outputs"]):
            raise ValueError("Disconnected or duplicate two-layer state write")
        available.update(node["outputs"])
    if len(available) != 67:
        raise ValueError("Two-layer cone must produce exactly 66 unique states")
    return selected


def _parameter_names(program: dict[str, Any]) -> tuple[set[str], set[str]]:
    nodes = two_layer_instructions(program)
    left = {name for node in nodes[:34] for name in node["parameter_refs"]}
    right = {name for node in nodes[34:] for name in node["parameter_refs"]}
    if len(left) != 16 or len(right) != 13 or left & right or len(left | right) != 29:
        raise ValueError("Fresh parameter references must be disjoint exact 16/13 union")
    return left, right


def _split_snapshots(program: dict[str, Any], snapshots: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    left, right = _parameter_names(program)
    if set(snapshots) != left | right:
        raise ValueError("Exactly 29 fresh parameter snapshots required; no stored roots")
    return ({name: snapshots[name] for name in sorted(left)}, {name: snapshots[name] for name in sorted(right)})


def _compose(program: dict[str, Any], token_ids: Any, snapshots: dict[str, Any], providers: first.Providers,
             profiles: dict[str, Any], runtime: dict[str, Any], stages: dict[str, Any]) -> dict[str, Any]:
    nodes = two_layer_instructions(program)
    if set(stages) != set(STAGE_IDS):
        raise ValueError("Both fresh constituent executions are required")
    left, right = _split_snapshots(program, snapshots)
    a, b = stages["first"], stages["second"]
    first.check_execution(program, token_ids, left, providers, profiles, runtime, a)
    roots = {name: a["state_bits"][name] for name in second.ROOTS}
    second.check_execution(program, roots, right, providers, profiles, runtime, b)
    merged = dict(a["state_bits"])
    for name, value in b["state_bits"].items():
        if name in second.ROOTS:
            if canonical_json(value) != canonical_json(roots[name]):
                raise ValueError("Second-stage root is not the fresh first-stage output")
        elif name in merged:
            raise ValueError("Duplicate global write")
        else:
            merged[name] = value
    states = {"input_ids": first._tokens(program, token_ids)}
    hashes = {"input_ids": "ROOT:" + first._descriptor(states["input_ids"])["sha256"]}
    records, bridge = [], {}
    fresh_records = a["records"] + b["records"]
    for index, (node, record) in enumerate(zip(nodes, fresh_records)):
        stage_id = "first" if index < 34 else "second"
        payload = copy.deepcopy(record["payload"])
        if payload["instruction_id"] != node["id"] or set(payload["inputs"]) != set(node["inputs"]) or set(payload["outputs"]) != set(node["outputs"]) or set(payload["parameters"]) != set(node["parameter_refs"]):
            raise ValueError("Fresh constituent instruction/operand coverage mismatch")
        for name, entry in payload["inputs"].items():
            if name not in states or entry["descriptor"] != first._descriptor(states[name]):
                raise ValueError("Global input descriptor mismatch")
            entry["producer_record_hash"] = hashes[name]
        for name in node["outputs"]:
            if name in states:
                raise ValueError("Duplicate global producer")
            states[name] = first._array(merged[name], first._shape(program, name), program["tensors"][name]["dtype"])
            if payload["outputs"][name] != first._descriptor(states[name]):
                raise ValueError("Global output descriptor mismatch")
        payload.update(stage_id=stage_id, original_fresh_stage_record_hash=record["record_hash"])
        global_record = append_chain_record(records, payload)
        for name in node["outputs"]:
            hashes[name] = global_record["record_hash"]
            if name in second.ROOTS:
                descriptor = first._descriptor(states[name])
                bridge[name] = {"producer_instruction_id": node["id"], "producer_record_hash": hashes[name],
                                "descriptor": descriptor, "second_stage_root_hash": "ROOT:" + descriptor["sha256"]}
    if len(records) != 63 or len(merged) != 66 or set(bridge) != set(second.ROOTS) or not verify_trace_chain(records)["valid"]:
        raise ValueError("Incomplete global chain or bridge")
    scalar = {**a["scalar_stages"], **b["scalar_stages"]}
    if len(scalar) != 12 or sum(len(values) for stages_ in scalar.values() for values in stages_.values()) != 1620:
        raise ValueError("Twelve RMS norms and 1620 scalar positions required")
    return {"stage_executions": stages, "state_bits": merged, "scalar_stages": scalar,
            "softmax_f32_bits": {stage: stages[stage]["softmax_f32_bits"] for stage in STAGE_IDS},
            "records": records, "bridge": bridge,
            "subtrace_roots": {stage: stages[stage]["records"][-1]["record_hash"] for stage in STAGE_IDS}}


def execute_two_layers(program: dict[str, Any], token_ids: Any, parameter_snapshots: dict[str, Any], providers: first.Providers,
                       profiles: dict[str, Any], runtime: dict[str, Any], workers: int = 4) -> dict[str, Any]:
    code = _code_sha()
    if type(workers) is not int or not 1 <= workers <= 4 or type(providers) is not first.Providers:
        raise ValueError("Invalid two-layer workers/provider container")
    left, right = _split_snapshots(program, parameter_snapshots)
    first._tokens(program, token_ids)
    first.check_parameter_snapshots(program, token_ids, left)
    second.check_parameter_snapshots(program, right)
    first_execution = first.execute_first_layer(program, token_ids, left, providers, profiles, runtime, workers)
    guard = _sha(first_execution)
    roots = {name: first_execution["state_bits"][name] for name in second.ROOTS}
    second_execution = second.execute_second_layer(program, roots, right, providers, profiles, runtime, workers)
    if guard != _sha(first_execution):
        raise ValueError("Second execution mutated the fresh first-stage frame")
    result = _compose(program, token_ids, parameter_snapshots, providers, profiles, runtime,
                      {"first": first_execution, "second": second_execution})
    if code != _code_sha():
        raise ValueError("Two-layer arithmetic source changed during execution")
    return result


def check_execution(program: dict[str, Any], token_ids: Any, snapshots: dict[str, Any], providers: first.Providers,
                    profiles: dict[str, Any], runtime: dict[str, Any], execution: dict[str, Any]) -> None:
    expected = _compose(program, token_ids, snapshots, providers, profiles, runtime, execution["stage_executions"])
    if canonical_json(execution) != canonical_json(expected):
        raise ValueError("Global fresh ledger/bridge/states/auxiliary/frame mismatch; integrity only, no numerical recompute")


def _coverage(program: dict[str, Any]) -> dict[str, Any]:
    nodes = two_layer_instructions(program)
    _parameter_names(program)
    return {"target": "hidden.2", "root_inputs": ["input_ids"], "instruction_ids": [node["id"] for node in nodes],
            "instruction_count": 63, "state_count": 66, "new_state_count": 66, "parameter_count": 29,
            "states": {name: {"shape": first._shape(program, name), "dtype": program["tensors"][name]["dtype"]} for node in nodes for name in node["outputs"]},
            "excluded_prefix_nodes": [{"id": "i0003", "outputs": ["rotary.global.cosine", "rotary.global.sine"], "reason": "Global RoPE outside hidden.2 backward cone"},
                                      {"id": "i0005", "outputs": ["mask.full"], "reason": "Full mask outside hidden.2 backward cone"}],
            "stage_instruction_counts": [34, 29], "stage_new_state_counts": [36, 30], "cross_stage_root_count": 4,
            "rms_norm_count": 12, "rms_scalar_positions": 1620, "softmax_fp32_positions": 7200,
            "layer_indices": [0, 1], "attention_types": ["sliding_attention", "sliding_attention"],
            "rotary_profiles": ["local", "local"], "sliding_window": 512, "first_full_attention_index": 5}


class _CachedSecondSources:
    def __init__(self, source: second.SecondLayerSources):
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
            raise ValueError("Cached source validation is local to one model path and operation")
        return self.context


@dataclass(frozen=True)
class TwoLayerSources:
    second_sources: second.SecondLayerSources
    second_plan: dict[str, Any]
    second_bundle: dict[str, Any]
    second_report: dict[str, Any]

    @property
    def program(self) -> dict[str, Any]:
        return self.second_sources.program

    @property
    def runtime(self) -> dict[str, Any]:
        return self.second_sources.runtime

    def commitments(self) -> dict[str, Any]:
        return {"second_sources": self.second_sources.commitments(), "second_plan_payload_sha256": _sha(self.second_plan),
                "second_bundle_payload_sha256": _sha(self.second_bundle), "second_report_payload_sha256": _sha(self.second_report),
                "program_payload_sha256": _sha(self.program)}

    def validate(self, model_path: Path) -> tuple[Any, ...]:
        code, guard = _code_sha(), _sha(self.commitments())
        two_layer_instructions(self.program)
        for payload, field, digest in ((self.second_plan, "plan_sha256", SECOND_PLAN_SHA256), (self.second_report, "report_sha256", SECOND_REPORT_SHA256)):
            _check_hash(payload, field)
            if payload[field] != digest:
                raise ValueError("Two-layer sources require the declared passing latest layer-one evidence")
        cached = _CachedSecondSources(self.second_sources)
        checked = second.verify_second_layer(cached, self.second_plan, self.second_bundle, self.second_report, model_path)
        if any(checked.get(name) is not True for name in ("valid", "second_layer_matches", "layer_one_independently_recomputed")):
            raise ValueError("Two-layer prerequisite full source graph/holdouts/layer-one verification failed")
        ids, providers, profiles, old_roots, old_bindings = cached.validate(model_path)
        del old_roots, old_bindings
        if ids != self.second_sources.baseline.baseline_plan["input_token_ids"] or len(ids[0]) != 30:
            raise ValueError("Two-layer public scope is the previously observed baseline thirty tokens")
        _kernel_expectations(self)
        if code != _code_sha() or guard != _sha(self.commitments()):
            raise ValueError("Two-layer sources changed during validation")
        return ids, providers, profiles


def _kernel_expectations(sources: TwoLayerSources) -> dict[str, Any]:
    first_sets = [pair["traced"]["kernels"] for pair in sources.second_sources.baseline.baseline_report["observations"]]
    second_sets = sources.second_report["kernel_trace"]
    expected = {}
    for stage, sets, roles in (("first", first_sets, first._native_kernel_roles(sources.program)),
                               ("second", second_sets, second._native_kernel_roles(sources.program))):
        if len(sets) != 3 or len(roles) != 22 or any(value != sets[0] for value in sets) or set(sets[0]) != set(roles):
            raise ValueError("Both passing sources must pin all 22 stable native kernel role sets")
        if any(not isinstance(names, list) or not names or names != sorted(set(names)) or any(not isinstance(name, str) or not name.strip() for name in names) for names in sets[0].values()):
            raise ValueError("Malformed source kernel symbol set")
        expected[stage] = copy.deepcopy(sets[0])
    return expected


def _snapshot_model(sources: TwoLayerSources, model: Any, ids: Any) -> dict[str, Any]:
    second._model_context(sources.second_sources, model)
    left = first._snapshot_model(sources.second_sources.baseline.first_sources, model, ids)
    right = second._snapshot_model(sources.second_sources, model)
    left_names, right_names = _parameter_names(sources.program)
    if set(left) != left_names or set(right) != right_names or set(left) & set(right):
        raise ValueError("Fresh snapshot helper results must be the exact disjoint 16/13 reference union")
    second._model_context(sources.second_sources, model)
    return {**left, **right}


def _source_header(sources: TwoLayerSources, context: tuple[Any, ...]) -> dict[str, Any]:
    ids, providers, profiles = context
    if ids != sources.second_sources.baseline.baseline_plan["input_token_ids"]:
        raise ValueError("Source header requires the declared baseline tokens")
    return {"scope": SCOPE, "program_sha256": sources.program["program_sha256"], "coverage": _coverage(sources.program),
            "sources": sources.commitments(), "code_sha256": _code_sha(), "runtime": copy.deepcopy(sources.runtime),
            "capture_code_sha256": {"first": first_capture._code_sha(), "second": second_capture._code_sha()},
            "model_binding": copy.deepcopy(sources.second_sources.baseline.baseline_plan["model_binding"]),
            "input_token_ids": copy.deepcopy(ids), "input_ids": first._descriptor(first._tokens(sources.program, ids)),
            "providers": providers.commitments(), "profiles": copy.deepcopy(profiles), "expected_kernel_sets": _kernel_expectations(sources)}


def _plan_body(sources: TwoLayerSources, context: tuple[Any, ...], bundle: dict[str, Any]) -> dict[str, Any]:
    execution = bundle["execution"]
    return {"schema_version": 1, "artifact_kind": "gemma_two_layers_plan", **_source_header(sources, context),
            "parameter_snapshots": {name: {key: value for key, value in snapshot.items() if key != "bits"} for name, snapshot in bundle["parameter_snapshots"].items()},
            "parameter_snapshot_membership": "fresh full-checkpoint binding and selected embedding row indexing; offline integrity cannot revalidate embedding membership; reexecute requires fresh model",
            "state_descriptors": {name: first._descriptor(first._array(value, first._shape(sources.program, name), sources.program["tensors"][name]["dtype"])) for name, value in execution["state_bits"].items()},
            "scalar_stages_sha256": _sha(execution["scalar_stages"]), "softmax_f32_sha256": _sha(execution["softmax_f32_bits"]),
            "bundle_sha256": _sha(bundle), "trace_root": execution["records"][-1]["record_hash"],
            "bridge": copy.deepcopy(execution["bridge"]), "subtrace_roots": copy.deepcopy(execution["subtrace_roots"]),
            "prediction_complete": True, "repetitions": 3, "original_forward_count": 12, "prefix_stopped_forward_count": 6,
            "second_layer_stopped_forward_count": 6, "native_scope": NATIVE_SCOPE,
            "native_protocol": ["first_traced_stop_0", "first_plain_stop_0", "second_traced_stop_1", "second_plain_stop_1"],
            "kernel_role_scope_count": 44, "kernel_provenance": "22 role symbol sets per stage, 44 role-scopes not 44 unique kernels",
            "epoch_scope": "previously_observed_baseline_only; source_holdouts_are_prerequisites_not_new_two_layer_cases", **_flags()}


def _predict(sources: TwoLayerSources, model: Any, context: tuple[Any, ...], workers: int) -> tuple[dict[str, Any], dict[str, Any]]:
    code, guard = _code_sha(), _sha(sources.commitments())
    ids, providers, profiles = context
    if ids != sources.second_sources.baseline.baseline_plan["input_token_ids"]:
        raise ValueError("Public prediction rejects different token IDs")
    snapshots = _snapshot_model(sources, model, ids)
    execution = execute_two_layers(sources.program, ids, snapshots, providers, profiles, sources.runtime, workers)
    if canonical_json(_snapshot_model(sources, model, ids)) != canonical_json(snapshots):
        raise ValueError("Fresh full checkpoint/embedding snapshots changed across both-stage forecast")
    bundle = {"parameter_snapshots": snapshots, "execution": execution}
    plan = _seal(_plan_body(sources, context, bundle), "plan_sha256")
    if code != _code_sha() or guard != _sha(sources.commitments()):
        raise ValueError("Source changed across two-layer forecast")
    return plan, bundle


def build_two_layers_plan(sources: TwoLayerSources, model: Any, model_path: Path, workers: int = 4,
                          input_token_ids: Any = None) -> tuple[dict[str, Any], dict[str, Any]]:
    context = sources.validate(model_path)
    if input_token_ids is not None and canonical_json(input_token_ids) != canonical_json(context[0]):
        raise ValueError("Public two-layer plan is restricted to baseline token IDs")
    return _predict(sources, model, context, workers)


def _check_plan(sources: TwoLayerSources, plan: dict[str, Any], bundle: dict[str, Any], context: tuple[Any, ...]) -> None:
    _check_hash(plan, "plan_sha256")
    if set(bundle) != {"parameter_snapshots", "execution"}:
        raise ValueError("Unexpected two-layer bundle roots or boundary paths")
    ids, providers, profiles = context
    check_execution(sources.program, ids, bundle["parameter_snapshots"], providers, profiles, sources.runtime, bundle["execution"])
    if canonical_json(plan) != canonical_json(_seal(_plan_body(sources, context, bundle), "plan_sha256")):
        raise ValueError("Two-layer source/snapshot/global-chain/bridge/header binding mismatch")


def check_two_layers_plan(sources: TwoLayerSources, plan: dict[str, Any], bundle: dict[str, Any], model_path: Path) -> None:
    code, guard = _code_sha(), _sha(sources.commitments())
    _check_plan(sources, plan, bundle, sources.validate(model_path))
    if code != _code_sha() or guard != _sha(sources.commitments()):
        raise ValueError("Sources changed during two-layer integrity verification")


def _native_comparison(plan: dict[str, Any], bundle: dict[str, Any], observations: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(observations, list) or len(observations) != 3:
        raise ValueError("Exactly three paired scopes, twelve original forwards required")
    comparisons, kernel_trace, counts = [], [], {}
    checks = dict.fromkeys(("capture_checks", "geometry", "runtime", "capture_code", "source_kernel_sets", "same_cuda_device"), True)
    execution = bundle["execution"]
    for repetition in observations:
        if set(repetition) != set(STAGE_IDS):
            raise ValueError("Both separate invocation pairs required")
        compared, kernels, devices = {}, {}, set()
        for stage in STAGE_IDS:
            pair = repetition[stage]
            if set(pair) != {"traced", "untraced"}:
                raise ValueError("Separate original traced/plain pair fields mismatch")
            traced, plain = pair["traced"], pair["untraced"]
            predicted = execution["stage_executions"][stage]
            names = set(predicted["state_bits"])
            plain_names = {"hidden.1"} if stage == "first" else {"hidden.1", "hidden.2"}
            if set(traced["state_bits"]) != names or set(traced["geometry"]) != names or set(plain["state_bits"]) != plain_names or set(plain["geometry"]) != plain_names:
                raise ValueError("Native all-state/root/geometry coverage incomplete")
            if any(plain.get(key) for key in ("scalar_stages", "softmax_f32_bits", "kernels")):
                raise ValueError("Plain control must use only existing minimal hidden hooks")
            if set(traced["scalar_stages"]) != set(predicted["scalar_stages"]):
                raise ValueError("Native twelve RMS norm coverage incomplete")
            for name in predicted["state_bits"]:
                declaration = plan["coverage"]["states"][name]
                shape, dtype = declaration["shape"], declaration["dtype"]
                compared[stage + ":" + name] = first._comparison(predicted["state_bits"][name], traced["state_bits"][name], shape, dtype)
                checks["geometry"] &= first._geometry(traced["geometry"][name], shape, dtype)
            for name, stages in predicted["scalar_stages"].items():
                native = traced["scalar_stages"][name]
                if set(native) != {*first.STAGES, "mean_input_metadata"}:
                    raise ValueError("Native RMS scalar fields incomplete")
                for scalar in first.STAGES:
                    compared[stage + ":" + name + ":" + scalar] = first._comparison(stages[scalar], native[scalar], [len(stages[scalar])], "torch.float32")
                metadata = native["mean_input_metadata"]
                shape = plan["coverage"]["states"][name]["shape"]
                strides = metadata.get("input_strides")
                checks["geometry"] &= metadata.get("input_shape") == shape and metadata.get("input_dtype") == "torch.float32" and metadata.get("axes") == [-1] and metadata.get("keepdim") is True and metadata.get("alignment_mod16") == 0 and isinstance(strides, list) and len(strides) == len(shape) and strides[-1] == 1 and all(type(stride) is int and stride > 0 and stride % 4 == 0 for stride in strides[:-1])
            compared[stage + ":softmax_f32_bits"] = first._comparison(predicted["softmax_f32_bits"], traced["softmax_f32_bits"], [1, 4, 30, 30], "torch.float32")
            for name in sorted(plain_names):
                compared[stage + ":plain:" + name] = first._comparison(predicted["state_bits"][name], plain["state_bits"][name], [1, 30, 640], "torch.bfloat16")
                checks["geometry"] &= first._geometry(plain["geometry"][name], [1, 30, 640], "torch.bfloat16")
            capture_code = first_capture._code_sha() if stage == "first" else second_capture._code_sha()
            for kind, record in pair.items():
                if "capture_failure" in record:
                    raise ValueError("Native capture abstained")
                required = {"token_commitment_unchanged", "stopped_at_decoder_tuple", "decoder_tuple_length_one", "no_later_layer", "no_final_norm", "no_lm_head", "all_needed_states_present", "code_unchanged", "runtime_unchanged"}
                required.update({"original_operations_once", "all_operand_links_unchanged", "rms_scalar_stages_complete", "softmax_f32_complete"} if kind == "traced" else {"plain_minimal_control"})
                if stage == "second":
                    required.update({"target_layer_one", "local_rotary_and_sliding_mask"})
                checks["capture_checks"] &= isinstance(record.get("checks"), dict) and required.issubset(record["checks"]) and all(value is True for value in record["checks"].values())
                checks["runtime"] &= canonical_json(record["runtime_before"]) == canonical_json(plan["runtime"]) == canonical_json(record["runtime_after"])
                checks["capture_code"] &= record["code_before"] == record["code_after"] == capture_code == plan["capture_code_sha256"][stage]
                devices.update(value["device"] for value in record["geometry"].values())
            symbols = traced["kernels"]
            checks["source_kernel_sets"] &= isinstance(symbols, dict) and len(symbols) == 22 and symbols == plan["expected_kernel_sets"][stage] and all(isinstance(names_, list) and names_ and names_ == sorted(set(names_)) and all(isinstance(name, str) and name.strip() for name in names_) for names_ in symbols.values())
            kernels[stage] = symbols
        checks["same_cuda_device"] &= len(devices) == 1
        for name in second.ROOTS:
            declaration = plan["coverage"]["states"][name]
            compared["cross_invocation_bridge:" + name] = first._comparison(repetition["first"]["traced"]["state_bits"][name], repetition["second"]["traced"]["state_bits"][name], declaration["shape"], declaration["dtype"])
        comparisons.append(compared)
        kernel_trace.append(kernels)
        for name, item in compared.items():
            counts[name] = counts.get(name, 0) + item["mismatch_count"]
    checks["repeat_stable_kernel_sets"] = all(value == kernel_trace[0] for value in kernel_trace)
    divergence = next(({"repetition": index, "state": name, **item["first_divergence"]} for index, compared in enumerate(comparisons) for name, item in compared.items() if item["first_divergence"] is not None), None)
    if divergence is None:
        divergence = next(({"check": name} for name, passed in checks.items() if not passed), None)
    return {"comparisons": comparisons, "checks": checks, "kernel_trace": kernel_trace, "mismatch_counts": counts,
            "aggregate_mismatch_count": sum(counts.values()), "first_divergence": divergence,
            "native_coverage_complete": True, "two_layers_match": all(checks.values()) and not any(counts.values())}


def two_layers_report(plan: dict[str, Any], bundle: dict[str, Any], observations: list[dict[str, Any]],
                      acquisition_guards: dict[str, bool] | None = None) -> dict[str, Any]:
    _check_hash(plan, "plan_sha256")
    execution = bundle["execution"]
    if plan["scope"] != SCOPE or plan["native_scope"] != NATIVE_SCOPE or _sha(bundle) != plan["bundle_sha256"] or any(plan.get(key) is not value for key, value in _flags().items()):
        raise ValueError("Comparison requires the frozen bounded two-layer plan")
    if len(execution["records"]) != 63 or len(execution["state_bits"]) != 66 or set(execution["stage_executions"]) != set(STAGE_IDS) or not verify_trace_chain(execution["records"], plan["trace_root"])["valid"]:
        raise ValueError("Complete connected ledger required before native comparison")
    guards = dict.fromkeys(GUARDS, True) if acquisition_guards is None else acquisition_guards
    try:
        result = _native_comparison(plan, bundle, observations)
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as error:
        result = {"two_layers_match": False, "native_coverage_complete": False, "mismatch_counts": {}, "aggregate_mismatch_count": 0,
                  "first_divergence": {"check": "native_coverage"}, "abstention": {"type": type(error).__name__, "message": str(error)}}
    observed = observations if isinstance(observations, list) else []
    counts = {stage: sum(isinstance(pair, dict) and isinstance(pair.get(stage), dict) and isinstance(pair[stage].get(kind), dict) and "capture_failure" not in pair[stage][kind] for pair in observed for kind in ("traced", "untraced")) for stage in STAGE_IDS}
    guarded = set(guards) == set(GUARDS) and all(value is True for value in guards.values())
    matches = result["two_layers_match"] and guarded and counts == {"first": 6, "second": 6}
    if not guarded and result["first_divergence"] is None:
        result["first_divergence"] = {"check": "acquisition_guards"}
    return _seal({"schema_version": 1, "artifact_kind": "gemma_two_layers_report", "scope": SCOPE,
                  "plan_sha256": plan["plan_sha256"], "bundle_sha256": _sha(bundle), "coverage": plan["coverage"],
                  "observations": observations, **result, "acquisition_guards": guards, "two_layers_match": matches,
                  "prediction_complete": True, "required_forward_count": 12, "original_forward_count": sum(counts.values()),
                  "prefix_stopped_forward_count": counts["first"], "second_layer_stopped_forward_count": counts["second"],
                  "native_scope": NATIVE_SCOPE, "kernel_role_scope_count": 44,
                  "bridge_evidence": "value agreement across separate original invocations, not same-invocation pointer or launch correspondence",
                  **_flags(matches)}, "report_sha256")


def _file_guard(paths: tuple[Path, ...]) -> dict[str, str]:
    return {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths if not path.is_dir()}


def _acquire(sources: TwoLayerSources, model: Any, plan: dict[str, Any], bundle: dict[str, Any], context: tuple[Any, ...],
             plan_path: Path, bundle_path: Path, frozen_inputs: tuple[Path, ...] = ()) -> dict[str, Any]:
    paths = (plan_path, bundle_path, *frozen_inputs)
    for path, payload in ((plan_path, plan), (bundle_path, bundle)):
        second.holdout.require_frozen(path, payload)
    code, guard, files = _code_sha(), _sha(sources.commitments()), _file_guard(paths)
    if canonical_json(_snapshot_model(sources, model, context[0])) != canonical_json(bundle["parameter_snapshots"]):
        raise ValueError("All 29 fresh acquisition snapshots must match frozen predictions")
    observations = []
    for _ in range(3):
        repetition = {}
        for stage, capture in (("first", first_capture.capture_first_layer), ("second", second_capture.capture_second_layer)):
            pair = {}
            for kind, traced in (("traced", True), ("untraced", False)):
                try:
                    pair[kind] = capture(model, context[0], traced)
                except (ValueError, RuntimeError) as error:
                    pair[kind] = {"capture_failure": {"type": type(error).__name__, "message": str(error), "kind": "capture_abstention_not_numerical_evidence"}}
            repetition[stage] = pair
        observations.append(repetition)
    guards = {}
    try:
        guards["checkpoint_unchanged"] = canonical_json(_snapshot_model(sources, model, context[0])) == canonical_json(bundle["parameter_snapshots"])
    except (ValueError, RuntimeError):
        guards["checkpoint_unchanged"] = False
    try:
        guards["source_unchanged"] = code == _code_sha() and guard == _sha(sources.commitments())
    except ValueError:
        guards["source_unchanged"] = False
    try:
        guards["frozen_files_unchanged"] = files == _file_guard(paths)
        for path, payload in ((plan_path, plan), (bundle_path, bundle)):
            second.holdout.require_frozen(path, payload)
    except (OSError, ValueError):
        guards["frozen_files_unchanged"] = False
    return two_layers_report(plan, bundle, observations, guards)


def acquire_two_layers(sources: TwoLayerSources, model: Any, plan: dict[str, Any], bundle: dict[str, Any], model_path: Path,
                       plan_path: Path, bundle_path: Path, workers: int = 4, frozen_inputs: tuple[Path, ...] = ()) -> dict[str, Any]:
    paths = (plan_path, bundle_path, *frozen_inputs)
    files = _file_guard(paths)
    for path, payload in ((plan_path, plan), (bundle_path, bundle)):
        second.holdout.require_frozen(path, payload)
    context = sources.validate(model_path)
    _check_plan(sources, plan, bundle, context)
    regenerated_plan, regenerated_bundle = _predict(sources, model, context, workers)
    if canonical_json(regenerated_plan) != canonical_json(plan) or canonical_json(regenerated_bundle) != canonical_json(bundle):
        raise ValueError("Both fresh stages must reproduce the entire frozen prediction before any native forward")
    if files != _file_guard(paths):
        raise ValueError("Frozen inputs changed during two-layer rebuild; no native calls allowed")
    return _acquire(sources, model, plan, bundle, context, plan_path, bundle_path, frozen_inputs)


def verify_two_layers(sources: TwoLayerSources, plan: dict[str, Any], bundle: dict[str, Any], report: dict[str, Any], model_path: Path) -> dict[str, Any]:
    result = {"mode": "integrity_full_source_snapshots_stage_and_global_hashchains_bridge_coverage_native_recount_only",
              "connected_numerical_recomputation_performed": False, "actual_layer0_numerical_recompute": False,
              "actual_layer1_numerical_recompute": False, "embedding_membership_revalidated_with_model": False,
              "previous_lineage_validation": "full_source_integrity_and_short_RMS_recomputation_not_previous_matmuls", **_flags()}
    try:
        code, guard = _code_sha(), _sha(sources.commitments())
        check_two_layers_plan(sources, plan, bundle, model_path)
        _check_hash(report, "report_sha256")
        expected = two_layers_report(plan, bundle, report["observations"], report["acquisition_guards"])
        valid = canonical_json(report) == canonical_json(expected) and code == _code_sha() and guard == _sha(sources.commitments())
        matches = valid and expected["two_layers_match"]
        return {**result, "valid": valid, "two_layers_match": matches, "prediction_complete": valid,
                "mismatch_counts": expected["mismatch_counts"], "first_divergence": expected["first_divergence"], **_flags(matches)}
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError, IndexError) as error:
        return {**result, "valid": False, "two_layers_match": False, "reason": str(error)}


def reexecute_two_layers(sources: TwoLayerSources, model: Any, plan: dict[str, Any], bundle: dict[str, Any], report: dict[str, Any],
                         model_path: Path, plan_path: Path, bundle_path: Path, workers: int = 4, frozen_inputs: tuple[Path, ...] = ()) -> dict[str, Any]:
    paths = (plan_path, bundle_path, *frozen_inputs)
    files = _file_guard(paths)
    for path, payload in ((plan_path, plan), (bundle_path, bundle)):
        second.holdout.require_frozen(path, payload)
    checked = verify_two_layers(sources, plan, bundle, report, model_path)
    if not checked["valid"]:
        return checked
    context = sources.validate(model_path)
    regenerated_plan, regenerated_bundle = _predict(sources, model, context, workers)
    same = canonical_json(regenerated_plan) == canonical_json(plan) and canonical_json(regenerated_bundle) == canonical_json(bundle)
    if files != _file_guard(paths):
        raise ValueError("Frozen inputs changed during two-layer replay rebuild; no native calls allowed")
    replay = _acquire(sources, model, plan, bundle, context, plan_path, bundle_path, frozen_inputs) if same else None
    exact = same and canonical_json(replay) == canonical_json(report)
    return {"valid": exact, "mode": "fresh_all_29_snapshots_both_stages_all_63_nodes_before_twelve_separate_original_forwards",
            "predictions_recomputed_exact": same, "reexecution_exact": exact, "connected_numerical_recomputation_performed": True,
            "actual_layer0_numerical_recompute": True, "actual_layer1_numerical_recompute": True,
            "embedding_membership_revalidated_with_model": True, "two_layers_match": exact and replay["two_layers_match"],
            "replay_report": replay if not exact else None, **_flags(exact and replay["two_layers_match"])}


def two_layers_summary(plan: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in report.items() if key not in ("observations", "comparisons", "report_sha256")}
    body.update(source_report_sha256=report["report_sha256"], sources=plan["sources"], code_sha256=plan["code_sha256"],
                trace_root=plan["trace_root"], bridge=plan["bridge"], subtrace_roots=plan["subtrace_roots"],
                providers=plan["providers"], profiles=plan["profiles"])
    return _seal(body, "summary_sha256")
