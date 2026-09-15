from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import gemma_mlp_down as down
from .gemma_attention_entry import _bits, _descriptor, _state_array
from .gemma_k2048_probes import probe_pool
from .gemma_rotary_slice import _sha, _seal, _check_hash
from .gemma_rsqrt_lookup import _runtime
from .serialization import canonical_json

SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
SEED = 0xA73C5E19
INPUT_SHAPE = (1, 30, 2048)
WEIGHT_SHAPE = (640, 2048)
OUTPUT_SHAPE = (1, 30, 640)
SUPPORTED_ID = down.SUPPORTED_ID
KERNEL_PROVENANCE = down.KERNEL_PROVENANCE
SCOPE = "Empirical synthetic fixed-shape dense non-repeated K2048 holdout of the unchanged K192/BF16-partial/sequential-FP32 candidate; all 19200 outputs, not all possible values, a new model-prompt holdout, full-layer qualification, or hardware proof. Native fused partials and registers are unobserved. Kernel provenance is distinct-symbol-set evidence only, not launch order."
SOURCE_MODE = "passing_v2_down_integrity_lineage_and_geometry_only_no_down_numerical_recomputation"
VERIFY_MODE = "integrity_generator_disjointness_geometry_and_recount_only_no_dot_recomputation_no_cuda"
FALSE_FLAGS = ("prediction_based_case_selection", "candidate_refitting_allowed", "original_model_revalidated", "fresh_model_prompt_holdout", "hardware_partitioning_established", "hardware_semantics_established", "split_boundaries_observed", "intermediate_values_observed", "native_fused_partials_observed", "native_registers_observed", "generic_domain_qualified", "full_first_layer_qualified", "global_exactness_activation_allowed", "qualification_promotion_allowed", "prefix_independently_recomputed", "model_down_predictions_recomputed")
SCHEDULE = [{"call": 2 * repetition + index, "repetition": repetition, "mode": mode, "operator": "torch.nn.functional.linear", "bias": None} for repetition in range(3) for index, mode in enumerate(("untraced", "traced"))]
GEOMETRY = {"input": (INPUT_SHAPE, [61440, 2048, 1]), "weight": (WEIGHT_SHAPE, [2048, 1]), "output": (OUTPUT_SHAPE, [19200, 640, 1])}


def _code_sha() -> str:
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Dense K2048 source changed after import")
    return _sha({"module": SOURCE_SHA256, "unchanged_down": down._code_sha()})


class _CachedDownSources:
    def __init__(self, source: down.DownSources):
        self.source = source
        self.values = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.source, name)

    def material(self) -> tuple[np.ndarray, dict[str, Any]]:
        if self.values is None:
            self.values = self.source.material()
        return self.values


@dataclass(frozen=True)
class DenseSources:
    down: down.DownSources
    model_plan: dict[str, Any]
    model_bundle: dict[str, Any]
    model_report: dict[str, Any]

    def check(self) -> tuple[dict[str, Any], list[str]]:
        code = _code_sha()
        cached = _CachedDownSources(self.down)
        checked = down.verify_down(cached, self.model_plan, self.model_bundle, self.model_report)
        if checked.get("valid") is not True or checked.get("down_matches") is not True or checked.get("mismatch_counts") != [0, 0, 0]:
            raise ValueError("Dense K2048 requires passing v2 model-down lineage")
        if self.model_plan.get("kernel_provenance") != KERNEL_PROVENANCE or self.model_plan.get("metadata_correction_development_regression") is not True or self.model_plan.get("value_count") != 19200:
            raise ValueError("Dense K2048 requires explicit v2 distinct-symbol-set provenance")
        inputs, weights, _ = down.check_down_plan(cached, self.model_plan, self.model_bundle)
        _, candidate = cached.material()
        down._supported_candidate(candidate)
        if canonical_json(self.model_plan["candidate"]) != canonical_json(candidate) or canonical_json(self.model_plan["runtime"]) != canonical_json(self.down.probe_plan["runtime"]):
            raise ValueError("Model-down candidate/runtime differs from controlled source")
        left, pool = probe_pool()
        excluded = {_sha(left), *(_sha(item["right_bits"]) for item in pool), *self.down.probe_plan["excluded_backend_vector_hashes"], *(_sha(row) for row in inputs[0].tolist()), *(_sha(row) for row in weights.tolist())}
        if _code_sha() != code:
            raise ValueError("Dense sources changed during validation")
        return candidate, sorted(excluded)

    def commitments(self) -> dict[str, Any]:
        return {"down_sources": self.down.commitments(), "model_plan_sha256": self.model_plan["plan_sha256"],
                "model_bundle_sha256": _sha(self.model_bundle), "model_report_sha256": self.model_report["report_sha256"],
                "model_summary_sha256": down.down_summary(self.model_plan, self.model_report)["summary_sha256"], "model_verification_mode": SOURCE_MODE}


def dense_vectors() -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    state = SEED

    def word() -> int:
        nonlocal state
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        return state & 0xFFFFFFFF

    def scalar() -> int:
        bits = word()
        return ((bits >> 16) & 0x8000) | ((122 + bits % 11) << 7) | (1 + word() % 127)

    def scale(bits: int, power: int) -> int:
        exponent = ((bits >> 7) & 255) + power
        if not 1 <= exponent <= 254:
            raise ValueError("Dense exact scaling escaped finite normals")
        return (bits & 0x807F) | (exponent << 7)

    inputs = np.asarray([[scalar() for _ in range(2048)] for _ in range(30)], dtype=np.uint16)
    weights, columns = [], []
    for column in range(640):
        if column < 320:
            values = [scalar() for _ in range(2048)]
            record = {"column": column, "family": "dense", "anchor_row": None, "pairing": None, "perturbed_coordinate": None}
        else:
            index, values = column - 320, [0] * 2048
            anchor = (index // 2) % 30
            pairing = "adjacent" if index % 2 == 0 else "across_halves"
            for pair in range(1024):
                a, b = (2 * pair, 2 * pair + 1) if pairing == "adjacent" else (pair, pair + 1024)
                power = int(word() % 7) - 3
                values[a] = scale(int(inputs[anchor, b]), power)
                values[b] = scale(int(inputs[anchor, a]), power) ^ 0x8000
            start = ((index // 2) % 11) * 192
            length = min(192, 2048 - start)
            coordinate = start + (0, 7, 8, 63, 64, 127, 128, 191)[(index // 22) % 8] % length
            values[coordinate] += -1 if values[coordinate] & 127 == 127 else 1
            record = {"column": column, "family": "row_anchored_cancellation", "anchor_row": anchor, "pairing": pairing, "perturbed_coordinate": coordinate}
        weights.append(values)
        columns.append(record)
    return inputs[None, ...], np.asarray(weights, dtype=np.uint16), columns


def _validate_vectors(inputs: np.ndarray, weights: np.ndarray, excluded: list[str]) -> tuple[list[str], list[str]]:
    if inputs.dtype != np.uint16 or weights.dtype != np.uint16 or inputs.shape != INPUT_SHAPE or weights.shape != WEIGHT_SHAPE:
        raise ValueError("Dense K2048 tensor geometry mismatch")
    for tensor in (inputs, weights):
        exponent = (tensor >> 7) & 255
        if not np.all((exponent > 0) & (exponent < 255) & ((tensor & 127) > 0)):
            raise ValueError("Dense coefficients must be finite-normal nonzero and non-unit")
    input_hashes = [_sha(row) for row in inputs[0].tolist()]
    weight_hashes = [_sha(row) for row in weights.tolist()]
    if len(set(input_hashes + weight_hashes)) != 670 or set(excluded) & set(input_hashes + weight_hashes):
        raise ValueError("Dense vectors repeat or overlap declared full-vector exclusions; no resampling")
    return input_hashes, weight_hashes


def _plan_body(sources: DenseSources, candidate: dict[str, Any], excluded: list[str], inputs: np.ndarray, weights: np.ndarray, columns: list[dict[str, Any]], prediction: np.ndarray, bundle: dict[str, Any]) -> dict[str, Any]:
    down._supported_candidate(candidate)
    input_hashes, weight_hashes = _validate_vectors(inputs, weights, excluded)
    return {"schema_version": 1, "scope": SCOPE, "sources": sources.commitments(), "code_sha256": _code_sha(), "candidate": candidate,
            "seed": SEED, "generator": "xorshift32_integer_bf16_dense_and_single_coordinate_perturbed_exact_cancellation", "columns": columns,
            "input": _descriptor(inputs), "weight": _descriptor(weights), "prediction": _descriptor(prediction),
            "input_shape": list(INPUT_SHAPE), "weight_shape": list(WEIGHT_SHAPE), "output_shape": list(OUTPUT_SHAPE),
            "input_row_hashes": input_hashes, "weight_vector_hashes": weight_hashes, "excluded_vector_hashes": excluded,
            "exclusion_scope": "complete_original_K2048_probe_pool_left_and_256_rights_declared_backend_vectors_30_model_product_rows_640_model_down_weights_only",
            "input_rows_distinct": True, "weight_vectors_distinct": True, "vectors_disjoint_from_declared_sources": True,
            "bundle_sha256": _sha(bundle), "value_count": 19200, "repetitions": 3, "complete_matrix_compared": True,
            "runtime": sources.model_plan["runtime"], "expected_kernel_names": down._kernel_symbols(sources.down.probe_plan["expected_kernel_names"]),
            "kernel_provenance": KERNEL_PROVENANCE, "acquisition_schedule": SCHEDULE, "linear_call_count": 6, "hidden_warmup_calls": 0,
            "linear_implementation": "torch.nn.functional.linear is torch._C._nn.linear", "bias": None,
            **{key: False for key in FALSE_FLAGS}}


def build_dense_plan(sources: DenseSources, workers: int = 1) -> tuple[dict[str, Any], dict[str, Any]]:
    code = _code_sha()
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Dense prediction requires one to four CPU workers")
    candidate, excluded = sources.check()
    commitments = canonical_json(sources.commitments())
    inputs, weights, columns = dense_vectors()
    _validate_vectors(inputs, weights, excluded)
    prediction = down.project_down_bits(inputs, weights, candidate, workers)
    bundle = {"input_bits": inputs.tolist(), "weight_bits": weights.tolist(), "prediction_bits": prediction.tolist()}
    plan = _seal(_plan_body(sources, candidate, excluded, inputs, weights, columns, prediction, bundle), "plan_sha256")
    if _code_sha() != code or canonical_json(sources.commitments()) != commitments:
        raise ValueError("Dense source changed during prediction")
    return plan, bundle


def check_dense_plan(sources: DenseSources, plan: dict[str, Any], bundle: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    code = _code_sha()
    _check_hash(plan, "plan_sha256")
    candidate, excluded = sources.check()
    if set(bundle) != {"input_bits", "weight_bits", "prediction_bits"}:
        raise ValueError("Unexpected dense bundle payload")
    inputs = _state_array(bundle["input_bits"], list(INPUT_SHAPE))
    weights = _state_array(bundle["weight_bits"], list(WEIGHT_SHAPE))
    prediction = _state_array(bundle["prediction_bits"], list(OUTPUT_SHAPE))
    expected_inputs, expected_weights, columns = dense_vectors()
    if not np.array_equal(inputs, expected_inputs) or not np.array_equal(weights, expected_weights):
        raise ValueError("Dense operands differ from fixed generator; no refitting")
    expected = _seal(_plan_body(sources, candidate, excluded, inputs, weights, columns, prediction, bundle), "plan_sha256")
    if canonical_json(plan) != canonical_json(expected) or _code_sha() != code:
        raise ValueError("Dense source, runtime, scope, exclusion or geometry commitment mismatch")
    return inputs, weights, prediction


def _snapshot(tensor: Any) -> dict[str, Any]:
    return {"tensor": _descriptor(_bits(tensor)), "shape": list(tensor.shape), "strides": list(tensor.stride()),
            "dtype": str(tensor.dtype), "alignment_mod16": tensor.data_ptr() % 16, "device": str(tensor.device), "contiguous": tensor.is_contiguous()}


def _snapshot_matches(record: dict[str, Any], descriptor: dict[str, Any], role: str) -> bool:
    shape, strides = GEOMETRY[role]
    return down._geometry_matches(record, shape, strides) and record.get("contiguous") is True and canonical_json(record.get("tensor")) == canonical_json(descriptor)


def dense_report(plan: dict[str, Any], prediction: np.ndarray, observations: list[dict[str, Any]], runtime: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(observations, list) or len(observations) != 3:
        raise ValueError("Dense holdout requires three untraced/traced pairs")
    if plan.get("kernel_provenance") != KERNEL_PROVENANCE or plan.get("acquisition_schedule") != SCHEDULE:
        raise ValueError("Dense schedule or distinct-symbol-set scope mismatch")
    expected_symbols = down._kernel_symbols(plan["expected_kernel_names"])
    mismatches, checks, hashes = [], [], {mode: [] for mode in ("untraced", "traced")}
    for repetition, pair in enumerate(observations):
        actuals = {}
        current = {}
        devices = set()
        for index, mode in enumerate(("untraced", "traced")):
            record = pair[mode]
            actual = _state_array(record["output_bits"], list(OUTPUT_SHAPE))
            actuals[mode] = actual
            descriptor = _descriptor(actual)
            hashes[mode].append(descriptor["sha256"])
            current[mode + "_schedule_matches"] = record["call"] == 2 * repetition + index and record["mode"] == mode
            current[mode + "_source_operands_match"] = all(_snapshot_matches(record[phase][role], plan[role], role) for phase in ("before", "after") for role in ("input", "weight"))
            current[mode + "_output_geometry_matches"] = _snapshot_matches(record["output"], descriptor, "output")
            current[mode + "_operand_identity_unchanged"] = record["operand_identity_unchanged"] is True
            current[mode + "_linear_identity_matches"] = record["linear_identity_matches"] is True
            current[mode + "_source_matches"] = record["code_before"] == record["code_after"] == plan["code_sha256"]
            current[mode + "_runtime_matches"] = all(canonical_json(record[key]) == canonical_json(plan["runtime"]) for key in ("runtime_before", "runtime_after"))
            devices.update(record[phase][role]["device"] for phase in ("before", "after") for role in ("input", "weight"))
            devices.add(record["output"]["device"])
            if mode == "traced":
                current["kernel_names_match"] = down._kernel_symbols(record["kernel_names"]) == expected_symbols
            elif record.get("kernel_names") != []:
                raise ValueError("Untraced control must not claim profiled kernels")
            for coordinate in np.argwhere(actual != prediction):
                row, column = int(coordinate[1]), int(coordinate[2])
                metadata = plan["columns"][column]
                mismatches.append({"repetition": repetition, "mode": mode, "row": row, "column": column, "family": metadata["family"], "pairing": metadata["pairing"],
                                   "predicted_bits": int(prediction[0, row, column]), "observed_bits": int(actual[0, row, column])})
        current["same_cuda_device"] = len(devices) == 1
        current["untraced_output_matches"] = np.array_equal(actuals["untraced"], actuals["traced"])
        checks.append(current)
    runtime_match = canonical_json(runtime) == canonical_json(plan["runtime"])
    repeated = all(len(set(values)) == 1 for values in hashes.values())
    counts = {mode: [sum(item["mode"] == mode and item["repetition"] == repetition for item in mismatches) for repetition in range(3)] for mode in hashes}
    families = {family: {mode: [sum(item["mode"] == mode and item["repetition"] == repetition and item["family"] == family for item in mismatches) for repetition in range(3)] for mode in hashes} for family in ("dense", "row_anchored_cancellation")}
    pairings = {pairing: {mode: [sum(item["mode"] == mode and item["repetition"] == repetition and item["pairing"] == pairing for item in mismatches) for repetition in range(3)] for mode in hashes} for pairing in ("adjacent", "across_halves")}
    compatible = runtime_match and all(item["kernel_names_match"] and item["untraced_runtime_matches"] and item["traced_runtime_matches"] for item in checks)
    passed = compatible and repeated and not mismatches and all(all(item.values()) for item in checks)
    first = next((key for key in checks[0] if not all(item[key] for item in checks)), None) or (mismatches[0] if mismatches else None) or (None if runtime_match and repeated else "runtime_or_repeat_stability")
    return _seal({"schema_version": 1, "scope": SCOPE, "plan_sha256": plan["plan_sha256"], "runtime": runtime, "runtime_matches_plan": runtime_match,
                  "kernel_provenance": KERNEL_PROVENANCE, "observations": observations, "checks": checks, "observed_output_hashes": hashes,
                  "repeated_outputs_identical": repeated, "mismatch_counts": counts["traced"], "untraced_mismatch_counts": counts["untraced"],
                  "family_mismatch_counts": families, "pairing_mismatch_counts": pairings, "mismatches": mismatches, "first_divergence": first,
                  "candidate_passes_dense_holdout": passed, "scope_abstained": not compatible, "status": "abstained" if not compatible else "passed" if passed else "failed",
                  "value_count": 19200, "complete_matrix_compared": True, "acquisition_schedule": SCHEDULE, "linear_call_count": 6,
                  **{key: False for key in FALSE_FLAGS}}, "report_sha256")


def _capture_linear(left: Any, right: Any, call: int, mode: str) -> dict[str, Any]:
    import torch
    from torch.profiler import profile, ProfilerActivity

    original = torch.nn.functional.linear
    if original is not torch._C._nn.linear:
        raise ValueError("Original F.linear implementation was replaced")
    code, runtime = _code_sha(), _runtime()
    identities = [(id(value), value.data_ptr(), value._version) for value in (left, right)]
    before = {"input": _snapshot(left), "weight": _snapshot(right)}
    if mode == "traced":
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as captured:
            output = original(left, right)
            torch.cuda.synchronize()
        names = down._kernel_symbols(sorted({event.name for event in captured.events() if event.device_type == torch.autograd.DeviceType.CUDA}))
    elif mode == "untraced":
        output = original(left, right)
        torch.cuda.synchronize()
        names = []
    else:
        raise ValueError("Unknown dense call mode")
    after = {"input": _snapshot(left), "weight": _snapshot(right)}
    return {"call": call, "mode": mode, "before": before, "after": after, "output": _snapshot(output), "output_bits": _bits(output).tolist(), "kernel_names": names,
            "operand_identity_unchanged": identities == [(id(value), value.data_ptr(), value._version) for value in (left, right)],
            "linear_identity_matches": torch.nn.functional.linear is original and original is torch._C._nn.linear,
            "code_before": code, "code_after": _code_sha(), "runtime_before": runtime, "runtime_after": _runtime()}


def acquire_dense(sources: DenseSources, plan: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    import torch

    code = _code_sha()
    inputs, weights, prediction = check_dense_plan(sources, plan, bundle)
    commitments = canonical_json(sources.commitments())
    if canonical_json(_runtime()) != canonical_json(plan["runtime"]):
        raise ValueError("Scope abstention: dense runtime differs from frozen source")
    if torch.nn.functional.linear is not torch._C._nn.linear:
        raise ValueError("Original F.linear implementation was replaced")
    left = torch.from_numpy(inputs.copy()).view(torch.bfloat16).to("cuda").contiguous()
    right = torch.from_numpy(weights.copy()).view(torch.bfloat16).to("cuda").contiguous()
    for role, value in (("input", left), ("weight", right)):
        if not _snapshot_matches(_snapshot(value), plan[role], role):
            raise ValueError("Actual native dense operand differs from frozen geometry/hash")
    observations = []
    with torch.no_grad():
        for repetition in range(3):
            plain = _capture_linear(left, right, 2 * repetition, "untraced")
            traced = _capture_linear(left, right, 2 * repetition + 1, "traced")
            observations.append({"untraced": plain, "traced": traced})
    if _code_sha() != code or canonical_json(sources.commitments()) != commitments:
        raise ValueError("Dense source changed during acquisition")
    return dense_report(plan, prediction, observations, _runtime())


def verify_dense(sources: DenseSources, plan: dict[str, Any], bundle: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    try:
        _, _, prediction = check_dense_plan(sources, plan, bundle)
        expected = dense_report(plan, prediction, report["observations"], report["runtime"])
        return {"valid": canonical_json(report) == canonical_json(expected), "mode": VERIFY_MODE, "predictions_recomputed": False,
                "model_verification_mode": SOURCE_MODE, "candidate_passes_dense_holdout": expected["candidate_passes_dense_holdout"],
                "scope_abstained": expected["scope_abstained"], "mismatch_counts": expected["mismatch_counts"], "untraced_mismatch_counts": expected["untraced_mismatch_counts"],
                **{key: False for key in FALSE_FLAGS}}
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError, StopIteration) as error:
        return {"valid": False, "mode": VERIFY_MODE, "predictions_recomputed": False, "reason": str(error)}


def dense_summary(plan: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in report.items() if key not in ("observations", "mismatches", "report_sha256")}
    body.update({"source_report_sha256": report["report_sha256"], "sources": plan["sources"], "code_sha256": plan["code_sha256"], "candidate": plan["candidate"], "seed": SEED,
                 "input_row_count": 30, "weight_vector_count": 640, "vectors_disjoint_from_declared_sources": plan["vectors_disjoint_from_declared_sources"],
                 "kernel_names": [down._kernel_symbols(pair["traced"]["kernel_names"]) for pair in report["observations"]]})
    return _seal(body, "summary_sha256")
