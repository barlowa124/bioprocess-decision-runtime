from __future__ import annotations

import hashlib
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from .gemma_attention_entry import _bits, _descriptor, _state_array
from .gemma_float_semantics import decode_finite_bfloat16, encode_bfloat16_rne
from .gemma_ir_interpreter import _projection_arithmetic_commitment
from .gemma_reduction_backend import _profile_call
from .gemma_rotary_slice import _sha, _seal, _check_hash
from .gemma_rsqrt_lookup import _runtime
from .gemma_wmma_candidate import _operand_aligned_accumulator, _merge_split_partials, OPERAND_ALIGNMENT_PROFILE
from .serialization import canonical_json

SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
SEED = 0xD8A91F03
WIDTHS = tuple(range(64, 1025, 64))
FORMATS = ("bfloat16_rne", "float32")
MERGES = ("exact", "sequential_float32_rne", "pairwise_float32_rne", "sequential_bfloat16_rne")
FAMILIES = ("boundary_cancellation", "merge_order", "dense_regions", "fragment_carry")
POOL_PER_FAMILY = 64
SELECT_PER_FAMILY = 32
CASE_COUNT = len(FAMILIES) * SELECT_PER_FAMILY
SCOPE = "Prospective candidate-selected K1024 split/merge discrimination on controlled BF16 [30,1024] x [640,1024] projections; repeated rows/columns and synthetic operands, not original-model or hardware partition qualification."


def _code_sha() -> str:
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("K1024 probe source changed after import")
    return _sha({"module": SOURCE_SHA256, "arithmetic": _projection_arithmetic_commitment()})


def candidates() -> list[dict[str, Any]]:
    return [{"id": f"k{width}:{format}:{merge}", "chunk_size": width, "partial_format": format, "merge": merge,
             "partitions": [[start, min(start + width, 1024)] for start in range(0, 1024, width)]}
            for width in WIDTHS for format in FORMATS for merge in MERGES]


def candidate_predictions(left: list[int], right: list[int]) -> list[int]:
    if len(left) != 1024 or len(right) != 1024:
        raise ValueError("Controlled split candidates require K1024")
    output = []
    for width in WIDTHS:
        partials = [_operand_aligned_accumulator(left[start:start + width], right[start:start + width]) for start in range(0, 1024, width)]
        for format in FORMATS:
            values = [decode_finite_bfloat16(encode_bfloat16_rne(value))[0] for value in partials] if format == "bfloat16_rne" else partials
            output.extend(_merge_split_partials(values, merge) for merge in MERGES)
    return output


def probe_pool() -> tuple[list[int], list[dict[str, Any]]]:
    state = SEED

    def word() -> int:
        nonlocal state
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        return state & 0xFFFFFFFF

    powers = [int(word() % 9) - 4 for _ in range(1024)]
    left = [(127 + power) << 7 for power in powers]
    pool, seen = [], set()

    def target(exponent: int, negative: bool = False) -> int:
        return (0x8000 if negative else 0) | ((127 + exponent) << 7) | (word() % 128)

    def distinct(count: int, upper: int) -> list[int]:
        positions = []
        while len(positions) < count:
            position = word() % upper
            if position not in positions:
                positions.append(position)
        return positions

    for family in FAMILIES:
        for _ in range(POOL_PER_FAMILY):
            small = int(word() % 11) - 5
            if family == "boundary_cancellation":
                boundary = (1 + word() % 15) * 64
                positions = [boundary - 1, boundary]
                positions.append(next(position for position in distinct(3, 1024) if position not in positions))
                large = target(small + 13 + word() % 8)
                products = [large, large ^ 0x8000, target(small)]
            elif family == "merge_order":
                positions = [block * 64 + word() % 64 for block in distinct(4, 16)]
                large = target(small + 26 + word() % 15)
                products = [target(small), large, target(small), large ^ 0x8000]
            elif family == "dense_regions":
                positions = distinct(64, 1024)
                products = [target(int(word() % 13) - 6, bool(word() & 1)) for _ in positions]
            else:
                block, other = distinct(2, 16)
                positions = [block * 64, block * 64 + 7, block * 64 + 8, block * 64 + 15, other * 64 + 31]
                large = target(small + 20 + word() % 8)
                products = [large, target(small), large ^ 0x8000, target(small, True), target(small)]
            right = [0] * 1024
            for position, product in zip(positions, products):
                exponent = ((product >> 7) & 255) - powers[position]
                if not 1 <= exponent <= 254:
                    raise ValueError("Probe operand escaped the finite-normal domain")
                right[position] = (product & 0x807F) | (exponent << 7)
                if decode_finite_bfloat16(left[position])[0] * decode_finite_bfloat16(right[position])[0] != decode_finite_bfloat16(product)[0]:
                    raise ValueError("Probe factorization changed the exact product")
            commitment = _sha(right)
            if commitment in seen:
                raise ValueError("Duplicate generated K1024 probe")
            seen.add(commitment)
            pool.append({"pool_index": len(pool), "family": family, "right_bits": right})
    return left, pool


def select_probes(pool: list[dict[str, Any]], predictions: list[list[int]]) -> tuple[list[int], int, list[list[int]]]:
    count = len(candidates())
    if len(predictions) != len(pool) or any(len(row) != count for row in predictions):
        raise ValueError("Malformed candidate prediction pool")
    pairs = [(left, right) for left in range(count) for right in range(left + 1, count)]
    masks = [sum(1 << index for index, (left, right) in enumerate(pairs) if row[left] != row[right]) for row in predictions]
    selected, used, covered = [], {family: 0 for family in FAMILIES}, 0
    for _ in range(CASE_COUNT):
        eligible = [index for index, item in enumerate(pool) if index not in selected and used[item["family"]] < SELECT_PER_FAMILY]
        if not eligible:
            raise ValueError("Insufficient family coverage for controlled probes")
        chosen = max(eligible, key=lambda index: ((masks[index] & ~covered).bit_count(), masks[index].bit_count(), -index))
        selected.append(chosen)
        used[pool[chosen]["family"]] += 1
        covered |= masks[chosen]
    groups = {}
    for candidate in range(count):
        signature = tuple(predictions[index][candidate] for index in selected)
        groups.setdefault(signature, []).append(candidate)
    return selected, covered.bit_count(), list(groups.values())


def _source(source: dict[str, Any]) -> None:
    _check_hash(source, "summary_sha256")
    names = source["projection_cuda_events"]
    if source["mismatch_counts"]["head_output"] != [0, 0, 0] or source["projection_value_count"] != 19200 or len(names) != 3 or not all(item == names[0] for item in names) or not any("splitKreduce" in item for item in names[0]):
        raise ValueError("Expected intact K1024 split-projection source observations")


def build_probe_plan(source: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    _source(source)
    code = _code_sha()
    left, pool = probe_pool()
    predictions = [candidate_predictions(left, item["right_bits"]) for item in pool]
    selected, separated_pairs, groups = select_probes(pool, predictions)
    cases = [pool[index] for index in selected]
    bundle = {"left_bits": left, "cases": cases}
    body = {"schema_version": 1, "scope": SCOPE, "source_summary_sha256": source["summary_sha256"], "source_report_sha256": source["source_report_sha256"],
            "seed": SEED, "code_sha256": code, "partial_arithmetic_profile": dict(OPERAND_ALIGNMENT_PROFILE), "candidates": candidates(),
            "candidate_count": len(candidates()), "pool_per_family": POOL_PER_FAMILY, "selected_per_family": SELECT_PER_FAMILY,
            "pool_commitment_sha256": _sha(pool), "pool_prediction_sha256": _sha(predictions), "selected_pool_indices": selected,
            "candidate_pairs_separated": separated_pairs, "prediction_equivalence_classes": groups,
            "selection": "greedy new candidate-pair separation, then total separation, then lower pool index; family quotas enforced",
            "input_shape": [1, 30, 1024], "weight_shape": [640, 1024], "left_bits_sha256": _sha(left), "bundle_sha256": _sha(bundle),
            "cases": [{"pool_index": item["pool_index"], "family": item["family"], "right_bits_sha256": _sha(item["right_bits"]), "prediction_bits": predictions[item["pool_index"]]} for item in cases],
            "case_count": CASE_COUNT, "matrix_value_count": 19200, "column_assignment": "column modulo 128 indexes selected case", "identical_input_rows": 30,
            "expected_kernel_names": source["projection_cuda_events"][0], "runtime": source["runtime"], "repetitions": 3,
            "acquisition_schedule": "one warm-up plus one profiled full projection per repetition",
            "candidate_selection_used_only_predictions": True, "candidate_refitting_allowed": False, "original_model_revalidated": False,
            "split_boundaries_observed": False, "intermediate_values_observed": False, "hardware_semantics_established": False,
            "global_exactness_activation_allowed": False}
    if _code_sha() != code:
        raise ValueError("K1024 source changed during probe prediction")
    return _seal(body, "plan_sha256"), bundle


def check_probe_plan(source: dict[str, Any], plan: dict[str, Any], bundle: dict[str, Any]) -> None:
    _source(source)
    _check_hash(plan, "plan_sha256")
    if plan["code_sha256"] != _code_sha() or plan["source_summary_sha256"] != source["summary_sha256"] or plan["source_report_sha256"] != source["source_report_sha256"] or plan["bundle_sha256"] != _sha(bundle):
        raise ValueError("Controlled K1024 source/bundle commitment mismatch")
    expected = {"schema_version": 1, "scope": SCOPE, "seed": SEED, "partial_arithmetic_profile": dict(OPERAND_ALIGNMENT_PROFILE), "candidates": candidates(),
                "candidate_count": 128, "pool_per_family": POOL_PER_FAMILY, "selected_per_family": SELECT_PER_FAMILY,
                "selection": "greedy new candidate-pair separation, then total separation, then lower pool index; family quotas enforced",
                "input_shape": [1, 30, 1024], "weight_shape": [640, 1024], "case_count": CASE_COUNT, "matrix_value_count": 19200,
                "column_assignment": "column modulo 128 indexes selected case", "identical_input_rows": 30,
                "expected_kernel_names": source["projection_cuda_events"][0], "runtime": source["runtime"], "repetitions": 3,
                "acquisition_schedule": "one warm-up plus one profiled full projection per repetition",
                "candidate_selection_used_only_predictions": True, "candidate_refitting_allowed": False, "original_model_revalidated": False,
                "split_boundaries_observed": False, "intermediate_values_observed": False, "hardware_semantics_established": False, "global_exactness_activation_allowed": False}
    if any(canonical_json(plan.get(key)) != canonical_json(value) for key, value in expected.items()):
        raise ValueError("K1024 candidate grid, geometry, or qualification scope changed")
    left, pool = probe_pool()
    if bundle["left_bits"] != left or plan["left_bits_sha256"] != _sha(left) or plan["pool_commitment_sha256"] != _sha(pool) or len(plan["cases"]) != CASE_COUNT or len(bundle["cases"]) != CASE_COUNT:
        raise ValueError("K1024 source generator commitments differ")
    indices = plan["selected_pool_indices"]
    if len(indices) != CASE_COUNT or len(set(indices)) != CASE_COUNT or any(type(index) is not int or not 0 <= index < len(pool) for index in indices):
        raise ValueError("Invalid selected probe indices")
    if bundle["cases"] != [pool[index] for index in indices] or any(sum(item["family"] == family for item in bundle["cases"]) != SELECT_PER_FAMILY for family in FAMILIES):
        raise ValueError("K1024 selected inputs or family coverage mismatch")
    for item, declaration in zip(bundle["cases"], plan["cases"]):
        if declaration["pool_index"] != item["pool_index"] or declaration["family"] != item["family"] or declaration["right_bits_sha256"] != _sha(item["right_bits"]) or len(declaration["prediction_bits"]) != 128 or any(type(bits) is not int or not 0 <= bits <= 65535 for bits in declaration["prediction_bits"]):
            raise ValueError("Probe prediction/input binding mismatch")
    predictions = [item["prediction_bits"] for item in plan["cases"]]
    groups = {}
    for candidate in range(128):
        groups.setdefault(tuple(row[candidate] for row in predictions), []).append(candidate)
    separated = sum(any(row[left] != row[right] for row in predictions) for left in range(128) for right in range(left + 1, 128))
    if plan["prediction_equivalence_classes"] != list(groups.values()) or plan["candidate_pairs_separated"] != separated:
        raise ValueError("K1024 discrimination accounting mismatch")


def _matrix_inputs(bundle: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(bundle["left_bits"], dtype=np.uint16)
    right = np.asarray([bundle["cases"][column % CASE_COUNT]["right_bits"] for column in range(640)], dtype=np.uint16)
    return np.tile(left, (1, 30, 1)), right


def probe_report(plan: dict[str, Any], bundle: dict[str, Any], observations: list[dict[str, Any]], runtime: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(observations, list) or len(observations) != 3:
        raise ValueError("Expected three controlled K1024 repetitions")
    left, right = _matrix_inputs(bundle)
    arrays = []
    for observation in observations:
        arrays.append(_state_array(observation["output_bits"], [1, 30, 640]))
        for name, values in (("input", left), ("weight", right)):
            descriptor = _descriptor(values)
            geometry = observation[name]
            if any(geometry["tensor"].get(key) != descriptor[key] for key in ("kind", "shape", "dtype", "layout", "numel", "sha256")) or geometry["strides"] != ([30720, 1024, 1] if name == "input" else [1024, 1]) or not str(geometry["tensor"].get("device", "")).startswith("cuda:"):
                raise ValueError("Controlled K1024 tensor geometry/value mismatch")
        if not observation["kernel_names"] or any(not isinstance(name, str) for name in observation["kernel_names"]):
            raise ValueError("Missing controlled projection CUDA provenance")
    results = []
    for candidate_index, candidate in enumerate(plan["candidates"]):
        expected = np.asarray([plan["cases"][column % CASE_COUNT]["prediction_bits"][candidate_index] for column in range(640)], dtype=np.uint16)[None, None, :]
        mismatches, first, case_indices = [], [], []
        for actual in arrays:
            different = actual != expected
            indices = np.argwhere(different)
            mismatches.append(int(np.count_nonzero(different)))
            first.append({"coordinate": indices[0].tolist(), "predicted_bits": int(expected[0, 0, indices[0, 2]]), "observed_bits": int(actual[tuple(indices[0])])} if len(indices) else None)
            case_indices.append(sorted({int(column % CASE_COUNT) for column in np.where(np.any(different, axis=(0, 1)))[0]}))
        results.append({"candidate_id": candidate["id"], "mismatch_counts": mismatches, "mismatch_case_indices": case_indices, "first_mismatches": first})
    same_runtime = canonical_json(runtime) == canonical_json(plan["runtime"])
    same_kernels = all(item["kernel_names"] == plan["expected_kernel_names"] for item in observations)
    repeated = all(np.array_equal(array, arrays[0]) for array in arrays)
    duplicate_coordinates_match = all(np.array_equal(array, np.tile(array[0, 0, :CASE_COUNT], (1, 30, 5))) for array in arrays)
    survivors = [item["candidate_id"] for item in results if item["mismatch_counts"] == [0, 0, 0]]
    return _seal({"schema_version": 1, "scope": SCOPE, "plan_sha256": plan["plan_sha256"], "runtime": runtime,
                  "observations": observations, "candidate_results": results, "surviving_candidates": survivors,
                  "minimum_total_mismatches": min(sum(item["mismatch_counts"]) for item in results),
                  "runtime_matches_plan": same_runtime, "kernel_names_match_source_shape": same_kernels,
                  "repeated_outputs_identical": repeated, "duplicate_coordinates_match": duplicate_coordinates_match,
                  "survivors_supported_in_declared_scope": bool(survivors) and same_runtime and same_kernels and repeated and duplicate_coordinates_match,
                  "unique_survivor_in_frozen_grid": len(survivors) == 1, "case_count": CASE_COUNT, "matrix_value_count": 19200,
                  "split_boundaries_observed": False, "intermediate_values_observed": False, "original_model_revalidated": False,
                  "hardware_semantics_established": False, "global_exactness_activation_allowed": False}, "report_sha256")


def acquire_probes(source: dict[str, Any], plan: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    import torch

    check_probe_plan(source, plan, bundle)
    if canonical_json(_runtime()) != canonical_json(plan["runtime"]):
        raise ValueError("Controlled K1024 runtime differs from the plan")
    left_bits, right_bits = _matrix_inputs(bundle)
    left = torch.from_numpy(left_bits).view(torch.bfloat16).to("cuda")
    right = torch.from_numpy(right_bits).view(torch.bfloat16).to("cuda")
    if left.data_ptr() % 16 or right.data_ptr() % 16:
        raise ValueError("Controlled K1024 tensors must be aligned")
    geometry = {"input": {"tensor": _descriptor(left_bits), "strides": list(left.stride())}, "weight": {"tensor": _descriptor(right_bits), "strides": list(right.stride())}}
    geometry["input"]["tensor"]["device"] = str(left.device)
    geometry["weight"]["tensor"]["device"] = str(right.device)
    observations = []
    with torch.no_grad():
        for _ in range(3):
            output, names = _profile_call(lambda: torch.nn.functional.linear(left, right))
            if output.dtype != torch.bfloat16 or list(output.shape) != [1, 30, 640]:
                raise ValueError("Controlled projection output type mismatch")
            observations.append({**geometry, "output_bits": _bits(output).tolist(), "kernel_names": names})
    _code_sha()
    return probe_report(plan, bundle, observations, _runtime())


def verify_probes(source: dict[str, Any], plan: dict[str, Any], bundle: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    try:
        check_probe_plan(source, plan, bundle)
        expected = probe_report(plan, bundle, report["observations"], report["runtime"])
        return {"valid": canonical_json(expected) == canonical_json(report), "mode": "integrity_only",
                "surviving_candidates": expected["surviving_candidates"], "survivors_supported_in_declared_scope": expected["survivors_supported_in_declared_scope"],
                "unique_survivor_in_frozen_grid": expected["unique_survivor_in_frozen_grid"], "global_exactness_activation_allowed": False}
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError) as error:
        return {"valid": False, "reason": str(error)}


def probe_summary(plan: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in report.items() if key not in ("observations", "candidate_results", "report_sha256")}
    body.update({"source_report_sha256": report["report_sha256"], "code_sha256": plan["code_sha256"], "candidate_count": 128,
                 "candidate_pairs_separated": plan["candidate_pairs_separated"], "prediction_equivalence_classes": plan["prediction_equivalence_classes"],
                 "candidate_results": [{key: value for key, value in item.items() if key not in ("mismatch_case_indices", "first_mismatches")} for item in report["candidate_results"]],
                 "kernel_names": [item["kernel_names"] for item in report["observations"]],
                 "observed_output_hashes": [_descriptor(_state_array(item["output_bits"], [1, 30, 640]))["sha256"] for item in report["observations"]]})
    return _seal(body, "summary_sha256")
