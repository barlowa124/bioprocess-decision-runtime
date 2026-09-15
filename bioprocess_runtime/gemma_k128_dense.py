from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .gemma_attention_entry import _bits, _descriptor, _state_array
from .gemma_k1024_probes import probe_pool
from .gemma_output_survivor import selected_candidate, survivor_dot, _code_sha as survivor_code_sha
from .gemma_reduction_backend import _profile_call
from .gemma_rotary_slice import _sha, _seal, _check_hash
from .gemma_rsqrt_lookup import _runtime
from .serialization import canonical_json

SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
SEED = 0xE4715A2D
CANDIDATE_ID = "k128:bfloat16_rne:sequential_float32_rne"
SCOPE = "Prospective dense non-repeated K1024 holdout of the frozen K128/BF16-partial/sequential-FP32 survivor; complete 30x640 matrix compared, not unrestricted inputs, original-model holdout, or hardware partition proof."


def _code_sha() -> str:
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Dense K128 source changed after import")
    return _sha({"module": SOURCE_SHA256, "survivor": survivor_code_sha()})


@dataclass(frozen=True)
class DenseSources:
    source_summary: dict[str, Any]
    probe_plan: dict[str, Any]
    probe_bundle: dict[str, Any]
    probe_report: dict[str, Any]
    model_plan: dict[str, Any]
    model_bundle: dict[str, Any]

    def check(self) -> tuple[dict[str, Any], list[str]]:
        candidate = selected_candidate(self.source_summary, self.probe_plan, self.probe_bundle, self.probe_report)
        if candidate["id"] != CANDIDATE_ID:
            raise ValueError("Dense holdout requires the frozen K128 survivor")
        _check_hash(self.model_plan, "plan_sha256")
        if self.source_summary["plan_sha256"] != self.model_plan["plan_sha256"] or self.model_plan["bundle_sha256"] != _sha(self.model_bundle):
            raise ValueError("Model-case exclusion source commitment mismatch")
        inputs = _state_array(self.model_bundle["predictions"]["concatenated"], [1, 30, 1024])
        weights = _state_array(self.model_bundle["weight_bits"], [640, 1024])
        if canonical_json(_descriptor(inputs)) != canonical_json(self.model_plan["predictions"]["concatenated"]) or canonical_json(_descriptor(weights)) != canonical_json(self.model_plan["weight"]):
            raise ValueError("Model-case exclusion vectors differ from committed tensors")
        left, pool = probe_pool()
        excluded = {_sha(left), *(_sha(item["right_bits"]) for item in pool), *(_sha(row) for row in inputs[0].tolist()), *(_sha(row) for row in weights.tolist())}
        return candidate, sorted(excluded)

    def commitments(self) -> dict[str, Any]:
        return {"source_summary_sha256": self.source_summary["summary_sha256"], "probe_plan_sha256": self.probe_plan["plan_sha256"],
                "probe_report_sha256": self.probe_report["report_sha256"], "model_plan_sha256": self.model_plan["plan_sha256"], "model_bundle_sha256": self.model_plan["bundle_sha256"]}


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
        return ((bits >> 16) & 0x8000) | ((122 + bits % 11) << 7) | (1 + (word() % 127))

    inputs = np.asarray([[scalar() for _ in range(1024)] for _ in range(30)], dtype=np.uint16)
    weights, records = [], []
    for column in range(640):
        if column < 320:
            values = [scalar() for _ in range(1024)]
            record = {"column": column, "family": "dense", "anchor_row": None, "perturbed_coordinate": None}
        else:
            index, values = column - 320, []
            anchor = index % 30
            for coordinate in range(0, 1024, 2):
                power = int(word() % 7) - 3
                for source, negate in ((int(inputs[anchor, coordinate + 1]), False), (int(inputs[anchor, coordinate]), True)):
                    exponent = ((source >> 7) & 255) + power
                    values.append(((source ^ (0x8000 if negate else 0)) & 0x807F) | (exponent << 7))
            coordinate = (index % 8) * 128 + (0, 63, 64, 127)[(index // 8) % 4]
            values[coordinate] += -1 if values[coordinate] & 127 == 127 else 1
            record = {"column": column, "family": "row_anchored_cancellation", "anchor_row": anchor, "perturbed_coordinate": coordinate}
        weights.append(values)
        records.append(record)
    return inputs[None, ...], np.asarray(weights, dtype=np.uint16), records


def _validate_vectors(inputs: np.ndarray, weights: np.ndarray, excluded: list[str]) -> tuple[list[str], list[str]]:
    if inputs.dtype != np.uint16 or weights.dtype != np.uint16 or inputs.shape != (1, 30, 1024) or weights.shape != (640, 1024):
        raise ValueError("Dense K128 tensor geometry mismatch")
    for tensor in (inputs, weights):
        exponent = (tensor >> 7) & 255
        if not np.all((exponent > 0) & (exponent < 255) & ((tensor & 127) > 0)):
            raise ValueError("Dense holdout requires nonzero finite-normal non-unit coefficients")
    input_hashes = [_sha(row) for row in inputs[0].tolist()]
    weight_hashes = [_sha(row) for row in weights.tolist()]
    if len(set(input_hashes)) != 30 or len(set(weight_hashes)) != 640 or set(input_hashes) & set(weight_hashes) or set(excluded) & set(input_hashes + weight_hashes):
        raise ValueError("Dense vectors repeat or overlap the declared exclusion set")
    return input_hashes, weight_hashes


def _plan_body(sources: DenseSources, candidate: dict[str, Any], excluded: list[str], inputs: np.ndarray, weights: np.ndarray, columns: list[dict[str, Any]], predicted: np.ndarray, bundle: dict[str, Any]) -> dict[str, Any]:
    input_hashes, weight_hashes = _validate_vectors(inputs, weights, excluded)
    return {"schema_version": 1, "scope": SCOPE, "sources": sources.commitments(), "candidate": candidate, "seed": SEED,
            "code_sha256": _code_sha(), "input": _descriptor(inputs), "weight": _descriptor(weights), "prediction": _descriptor(predicted),
            "input_row_hashes": input_hashes, "weight_vector_hashes": weight_hashes, "excluded_vector_hashes": excluded,
            "columns": columns, "bundle_sha256": _sha(bundle), "value_count": 19200, "repetitions": 3,
            "runtime": sources.probe_plan["runtime"], "expected_kernel_names": sources.probe_plan["expected_kernel_names"],
            "complete_matrix_compared": True, "input_rows_distinct": True, "weight_vectors_distinct": True,
            "vectors_disjoint_from_declared_sources": True, "prediction_based_case_selection": False,
            "candidate_refitting_allowed": False, "original_model_revalidated": False, "hardware_partitioning_established": False,
            "full_first_layer_qualified": False, "global_exactness_activation_allowed": False}


def build_dense_plan(sources: DenseSources) -> tuple[dict[str, Any], dict[str, Any]]:
    code = _code_sha()
    candidate, excluded = sources.check()
    inputs, weights, columns = dense_vectors()
    _validate_vectors(inputs, weights, excluded)
    rows = weights.tolist()
    predicted = np.asarray([[[survivor_dot(row, weight, candidate) for weight in rows] for row in inputs[0].tolist()]], dtype=np.uint16)
    bundle = {"input_bits": inputs.tolist(), "weight_bits": weights.tolist(), "prediction_bits": predicted.tolist()}
    if _code_sha() != code:
        raise ValueError("Dense K128 source changed during prediction")
    return _seal(_plan_body(sources, candidate, excluded, inputs, weights, columns, predicted, bundle), "plan_sha256"), bundle


def check_dense_plan(sources: DenseSources, plan: dict[str, Any], bundle: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _check_hash(plan, "plan_sha256")
    candidate, excluded = sources.check()
    inputs = _state_array(bundle["input_bits"], [1, 30, 1024])
    weights = _state_array(bundle["weight_bits"], [640, 1024])
    predicted = _state_array(bundle["prediction_bits"], [1, 30, 640])
    expected_inputs, expected_weights, columns = dense_vectors()
    if not np.array_equal(inputs, expected_inputs) or not np.array_equal(weights, expected_weights):
        raise ValueError("Dense operands do not match the frozen generator")
    expected = _seal(_plan_body(sources, candidate, excluded, inputs, weights, columns, predicted, bundle), "plan_sha256")
    if canonical_json(plan) != canonical_json(expected):
        raise ValueError("Dense holdout commitments, disjointness, profile, or scope mismatch")
    return inputs, weights, predicted


def dense_report(plan: dict[str, Any], predicted: np.ndarray, observations: list[dict[str, Any]], runtime: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(observations, list) or len(observations) != 3:
        raise ValueError("Dense holdout requires three repetitions")
    mismatches, hashes = [], []
    for repetition, observation in enumerate(observations):
        actual = _state_array(observation["output_bits"], [1, 30, 640])
        for name, strides in (("input", [30720, 1024, 1]), ("weight", [1024, 1])):
            record = observation[name]
            if record["strides"] != strides or any(record["tensor"].get(key) != plan[name][key] for key in ("kind", "shape", "dtype", "layout", "numel", "sha256")) or not str(record["tensor"].get("device", "")).startswith("cuda:"):
                raise ValueError("Dense acquisition input/weight geometry mismatch")
        if not isinstance(observation["kernel_names"], list) or not observation["kernel_names"] or any(not isinstance(name, str) or not name for name in observation["kernel_names"]):
            raise ValueError("Missing dense projection CUDA provenance")
        for coordinate in np.argwhere(actual != predicted):
            row, column = int(coordinate[1]), int(coordinate[2])
            mismatches.append({"repetition": repetition, "row": row, "column": column, "family": plan["columns"][column]["family"],
                               "predicted_bits": int(predicted[0, row, column]), "observed_bits": int(actual[0, row, column])})
        hashes.append(_descriptor(actual)["sha256"])
    runtime_match = canonical_json(runtime) == canonical_json(plan["runtime"])
    kernels_match = all(item["kernel_names"] == plan["expected_kernel_names"] for item in observations)
    repeated = len(set(hashes)) == 1
    counts = [sum(item["repetition"] == repetition for item in mismatches) for repetition in range(3)]
    families = {family: [sum(item["repetition"] == repetition and item["family"] == family for item in mismatches) for repetition in range(3)] for family in ("dense", "row_anchored_cancellation")}
    return _seal({"schema_version": 1, "scope": SCOPE, "plan_sha256": plan["plan_sha256"], "observations": observations, "runtime": runtime,
                  "runtime_matches_plan": runtime_match, "kernel_names_match": kernels_match, "repeated_outputs_identical": repeated,
                  "mismatch_counts": counts, "family_mismatch_counts": families, "mismatches": mismatches, "first_divergence": mismatches[0] if mismatches else None,
                  "candidate_passes_dense_holdout": not mismatches and runtime_match and kernels_match and repeated,
                  "value_count": 19200, "complete_matrix_compared": True, "original_model_revalidated": False,
                  "hardware_partitioning_established": False, "full_first_layer_qualified": False, "global_exactness_activation_allowed": False}, "report_sha256")


def acquire_dense(sources: DenseSources, plan: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    import torch

    inputs, weights, predicted = check_dense_plan(sources, plan, bundle)
    if canonical_json(_runtime()) != canonical_json(plan["runtime"]):
        raise ValueError("Dense acquisition runtime differs from the frozen plan")
    left = torch.from_numpy(inputs).view(torch.bfloat16).to("cuda")
    right = torch.from_numpy(weights).view(torch.bfloat16).to("cuda")
    if left.data_ptr() % 16 or right.data_ptr() % 16:
        raise ValueError("Dense CUDA operands must be aligned")
    from .operational_semantics import tensor_descriptor
    geometry = {"input": {"tensor": tensor_descriptor(left), "strides": list(left.stride())}, "weight": {"tensor": tensor_descriptor(right), "strides": list(right.stride())}}
    observations = []
    with torch.no_grad():
        for _ in range(3):
            output, names = _profile_call(lambda: torch.nn.functional.linear(left, right))
            observations.append({**geometry, "output_bits": _bits(output).tolist(), "kernel_names": names})
    _code_sha()
    return dense_report(plan, predicted, observations, _runtime())


def verify_dense(sources: DenseSources, plan: dict[str, Any], bundle: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    try:
        _, _, predicted = check_dense_plan(sources, plan, bundle)
        expected = dense_report(plan, predicted, report["observations"], report["runtime"])
        return {"valid": canonical_json(report) == canonical_json(expected), "mode": "integrity_generator_disjointness_and_geometry_checks",
                "candidate_passes_dense_holdout": expected["candidate_passes_dense_holdout"], "mismatch_counts": expected["mismatch_counts"],
                "global_exactness_activation_allowed": False}
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError) as error:
        return {"valid": False, "reason": str(error)}


def dense_summary(plan: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in report.items() if key not in ("observations", "mismatches", "report_sha256")}
    body.update({"source_report_sha256": report["report_sha256"], "sources": plan["sources"], "candidate": plan["candidate"], "seed": SEED,
                 "input_row_count": 30, "weight_vector_count": 640, "vectors_disjoint_from_declared_sources": plan["vectors_disjoint_from_declared_sources"],
                 "kernel_names": [item["kernel_names"] for item in report["observations"]],
                 "observed_output_hashes": [_descriptor(_state_array(item["output_bits"], [1, 30, 640]))["sha256"] for item in report["observations"]]})
    return _seal(body, "summary_sha256")
