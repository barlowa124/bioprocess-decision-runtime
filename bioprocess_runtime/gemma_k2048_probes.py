from __future__ import annotations

import hashlib
from concurrent.futures import ProcessPoolExecutor
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
SEED = 0xB632D9E1
K = 2048
WIDTHS = tuple(range(64, K + 1, 64))
FORMATS = ("bfloat16_rne", "float32")
MERGES = ("exact", "sequential_float32_rne", "pairwise_float32_rne", "sequential_bfloat16_rne")
FAMILIES = ("boundary_cancellation", "merge_order", "dense_regions", "fragment_order")
POOL_PER_FAMILY = 64
PER_FAMILY = 32
CASES = 128
SCOPE = "Prospective controlled K2048 down-projection candidate-grid discrimination; non-unit synthetic operands, repeated rows/columns, WMMA-derived partial arithmetic tested as an unqualified hypothesis for the recorded Ampere backend; not model or hardware qualification."


def _code_sha() -> str:
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("K2048 source changed after import")
    return _sha({"module": SOURCE_SHA256, "partial_and_merge_arithmetic": _projection_arithmetic_commitment()})


def candidates() -> list[dict[str, Any]]:
    return [{"id": f"k{width}:{format}:{merge}", "chunk_size": width, "partial_format": format, "merge": merge,
             "partitions": [[start, min(start + width, K)] for start in range(0, K, width)]}
            for width in WIDTHS for format in FORMATS for merge in MERGES]


def candidate_predictions(left: list[int], right: list[int]) -> list[int]:
    if len(left) != K or len(right) != K:
        raise ValueError("Down-projection grid requires K2048")
    predicted = []
    for width in WIDTHS:
        partials = [_operand_aligned_accumulator(left[start:start + width], right[start:start + width]) for start in range(0, K, width)]
        for format in FORMATS:
            values = [decode_finite_bfloat16(encode_bfloat16_rne(value))[0] for value in partials] if format == "bfloat16_rne" else partials
            predicted.extend(_merge_split_partials(values, merge) for merge in MERGES)
    return predicted


def _worker_init(expected_code: str) -> None:
    if _code_sha() != expected_code:
        raise ValueError("K2048 prediction worker source differs")


def _worker_predictions(pair: tuple[list[int], list[int]]) -> list[int]:
    return candidate_predictions(*pair)


def _scale(bits: int, power: int) -> int:
    exponent = ((bits >> 7) & 255) + power
    if not 1 <= exponent <= 254:
        raise ValueError("Controlled operand scaling escaped finite normals")
    return (bits & 0x807F) | (exponent << 7)


def probe_pool() -> tuple[list[int], list[dict[str, Any]]]:
    state = SEED

    def word() -> int:
        nonlocal state
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        return state & 0xFFFFFFFF

    def scalar(exponent: int) -> int:
        return ((word() >> 16) & 0x8000) | ((127 + exponent) << 7) | (1 + word() % 127)

    def distinct(count: int, limit: int) -> list[int]:
        result = []
        while len(result) < count:
            index = word() % limit
            if index not in result:
                result.append(index)
        return result

    left = [scalar(int(word() % 9) - 4) for _ in range(K)]
    pool, seen = [], set()
    for family in FAMILIES:
        for _ in range(POOL_PER_FAMILY):
            right = [0] * K
            small = int(word() % 11) - 5
            if family == "boundary_cancellation":
                boundary = (1 + word() % 31) * 64
                a, b = boundary - 1, boundary
                extra = next(index for index in distinct(3, K) if index not in (a, b))
                power = small + 18 + int(word() % 8)
                right[a], right[b], right[extra] = _scale(left[b], power), _scale(left[a], power) ^ 0x8000, scalar(small)
            elif family == "merge_order":
                a, b, c, d = [block * 64 + word() % 64 for block in distinct(4, 32)]
                power = small + 30 + int(word() % 12)
                right[a], right[c] = scalar(small), scalar(small)
                right[b], right[d] = _scale(left[d], power), _scale(left[b], power) ^ 0x8000
            elif family == "dense_regions":
                for position in distinct(128, K):
                    right[position] = scalar(int(word() % 13) - 6)
            else:
                block, other = distinct(2, 32)
                a, b = block * 64, block * 64 + 7
                power = small + 24 + int(word() % 8)
                right[a], right[b] = _scale(left[b], power), _scale(left[a], power) ^ 0x8000
                for position in (block * 64 + 8, block * 64 + 15, block * 64 + 31, block * 64 + 32, other * 64 + 63):
                    right[position] = scalar(small)
            digest = _sha(right)
            if digest in seen:
                raise ValueError("Duplicate K2048 probe vector")
            seen.add(digest)
            pool.append({"pool_index": len(pool), "family": family, "right_bits": right})
    return left, pool


def select_probes(pool: list[dict[str, Any]], predictions: list[list[int]]) -> tuple[list[int], int, list[list[int]]]:
    count = len(candidates())
    if len(pool) != len(predictions) or any(len(values) != count for values in predictions):
        raise ValueError("Malformed K2048 prediction pool")
    pairs = [(left, right) for left in range(count) for right in range(left + 1, count)]
    masks = [sum(1 << index for index, (a, b) in enumerate(pairs) if values[a] != values[b]) for values in predictions]
    selected, quotas, covered = [], {family: 0 for family in FAMILIES}, 0
    for _ in range(CASES):
        eligible = [index for index, item in enumerate(pool) if index not in selected and quotas[item["family"]] < PER_FAMILY]
        if not eligible:
            raise ValueError("Insufficient K2048 family coverage")
        index = max(eligible, key=lambda item: ((masks[item] & ~covered).bit_count(), masks[item].bit_count(), -item))
        selected.append(index)
        quotas[pool[index]["family"]] += 1
        covered |= masks[index]
    groups = {}
    for candidate in range(count):
        groups.setdefault(tuple(predictions[index][candidate] for index in selected), []).append(candidate)
    return selected, covered.bit_count(), list(groups.values())


def _sources(binding: dict[str, Any], product: dict[str, Any]) -> tuple[list[str], list[str]]:
    _check_hash(binding, "binding_sha256")
    _check_hash(product, "summary_sha256")
    records = [record for record in binding["records"] if record.get("role") == "mlp_down_projection"]
    if not records or product.get("activation_and_product_match") is not True or product.get("down_projection_executed") is not False or binding["program_sha256"] != product["sources"]["program_sha256"]:
        raise ValueError("Missing valid K2048 provenance/product source")
    names = records[0]["cuda_kernel_names"]
    if not any("ampere_bf16_s16816" in name for name in names) or not any("splitKreduce" in name for name in names):
        raise ValueError("Expected recorded Ampere plus splitKreduce down-projection backend")
    for record in records:
        _check_hash(record, "record_sha256")
        if record["cuda_kernel_names"] != names or record["input_shape"] != [30, K] or record["weight_shape"] != [640, K]:
            raise ValueError("Inconsistent recorded down-projection shape/kernel")
    return names, sorted({record[key] for record in records for key in ("left_vector_sha256", "right_vector_sha256")})


def build_plan(binding: dict[str, Any], product: dict[str, Any], workers: int = 1) -> tuple[dict[str, Any], dict[str, Any]]:
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Expected one to four deterministic CPU workers")
    names, excluded = _sources(binding, product)
    code = _code_sha()
    left, pool = probe_pool()
    if set(excluded) & {_sha(left), *(_sha(item["right_bits"]) for item in pool)}:
        raise ValueError("K2048 generated vectors overlap declared prior backend vectors")
    pairs = [(left, item["right_bits"]) for item in pool]
    if workers == 1:
        predictions = [_worker_predictions(pair) for pair in pairs]
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init, initargs=(code,)) as executor:
            predictions = list(executor.map(_worker_predictions, pairs))
    selected, separated, groups = select_probes(pool, predictions)
    bundle = {"left_bits": left, "cases": [pool[index] for index in selected]}
    body = {"schema_version": 1, "scope": SCOPE, "binding_sha256": binding["binding_sha256"], "product_summary_sha256": product["summary_sha256"],
            "product_report_sha256": product["source_report_sha256"], "program_sha256": binding["program_sha256"], "runtime": product["runtime"],
            "seed": SEED, "code_sha256": code, "candidates": candidates(), "candidate_count": 256, "partial_profile": dict(OPERAND_ALIGNMENT_PROFILE),
            "input_shape": [1, 30, K], "weight_shape": [640, K], "pool_size": len(pool), "pool_sha256": _sha(pool), "pool_predictions_sha256": _sha(predictions),
            "selected_pool_indices": selected, "candidate_pairs_separated": separated, "prediction_equivalence_classes": groups,
            "cases": [{"pool_index": index, "family": pool[index]["family"], "right_bits_sha256": _sha(pool[index]["right_bits"]), "prediction_bits": predictions[index]} for index in selected],
            "left_bits_sha256": _sha(left), "excluded_backend_vector_hashes": excluded, "bundle_sha256": _sha(bundle), "case_count": CASES,
            "matrix_value_count": 19200, "column_assignment": "column modulo 128 indexes selected case", "identical_input_rows": 30,
            "expected_kernel_names": names, "repetitions": 3, "selection": "greedy candidate-pair separation with 32 cases per family and deterministic ties",
            "acquisition_schedule": "one warm-up and one profiled projection per repetition", "candidate_selection_uses_only_predictions": True,
            "partial_arithmetic_transfer_prequalified": False, "candidate_refitting_allowed": False, "split_boundaries_observed": False,
            "intermediate_values_observed": False, "original_model_revalidated": False, "hardware_semantics_established": False,
            "global_exactness_activation_allowed": False}
    if _code_sha() != code:
        raise ValueError("K2048 source changed during prediction")
    return _seal(body, "plan_sha256"), bundle


def check_plan(binding: dict[str, Any], product: dict[str, Any], plan: dict[str, Any], bundle: dict[str, Any]) -> None:
    names, excluded = _sources(binding, product)
    _check_hash(plan, "plan_sha256")
    expected = {"schema_version": 1, "scope": SCOPE, "binding_sha256": binding["binding_sha256"], "product_summary_sha256": product["summary_sha256"],
                "product_report_sha256": product["source_report_sha256"], "program_sha256": binding["program_sha256"], "runtime": product["runtime"],
                "seed": SEED, "code_sha256": _code_sha(), "candidates": candidates(), "candidate_count": 256, "partial_profile": dict(OPERAND_ALIGNMENT_PROFILE),
                "input_shape": [1, 30, K], "weight_shape": [640, K], "case_count": CASES, "matrix_value_count": 19200,
                "column_assignment": "column modulo 128 indexes selected case", "identical_input_rows": 30, "expected_kernel_names": names,
                "excluded_backend_vector_hashes": excluded, "bundle_sha256": _sha(bundle), "repetitions": 3,
                "selection": "greedy candidate-pair separation with 32 cases per family and deterministic ties",
                "acquisition_schedule": "one warm-up and one profiled projection per repetition", "candidate_selection_uses_only_predictions": True,
                "partial_arithmetic_transfer_prequalified": False, "candidate_refitting_allowed": False, "split_boundaries_observed": False,
                "intermediate_values_observed": False, "original_model_revalidated": False, "hardware_semantics_established": False, "global_exactness_activation_allowed": False}
    if any(canonical_json(plan.get(key)) != canonical_json(value) for key, value in expected.items()):
        raise ValueError("K2048 profile/source/scope mismatch")
    left, pool = probe_pool()
    indices = plan["selected_pool_indices"]
    if len(indices) != CASES or len(set(indices)) != CASES or any(type(index) is not int or not 0 <= index < len(pool) for index in indices):
        raise ValueError("Invalid K2048 selected pool indices")
    if plan["pool_size"] != len(pool) or plan["pool_sha256"] != _sha(pool) or plan["left_bits_sha256"] != _sha(left) or bundle != {"left_bits": left, "cases": [pool[index] for index in indices]} or len(plan["cases"]) != CASES:
        raise ValueError("K2048 generator/selected-input mismatch")
    if set(excluded) & {_sha(left), *(_sha(item["right_bits"]) for item in pool)} or any(sum(pool[index]["family"] == family for index in indices) != PER_FAMILY for family in FAMILIES):
        raise ValueError("K2048 exclusion/family coverage mismatch")
    for index, record in zip(indices, plan["cases"]):
        if record["pool_index"] != index or record["family"] != pool[index]["family"] or record["right_bits_sha256"] != _sha(pool[index]["right_bits"]) or len(record["prediction_bits"]) != 256 or any(type(bits) is not int or not 0 <= bits <= 65535 for bits in record["prediction_bits"]):
            raise ValueError("K2048 prediction/input binding mismatch")
    rows, groups = [item["prediction_bits"] for item in plan["cases"]], {}
    for candidate in range(256):
        groups.setdefault(tuple(row[candidate] for row in rows), []).append(candidate)
    separated = sum(any(row[left] != row[right] for row in rows) for left in range(256) for right in range(left + 1, 256))
    if plan["prediction_equivalence_classes"] != list(groups.values()) or plan["candidate_pairs_separated"] != separated:
        raise ValueError("K2048 discrimination accounting mismatch")


def _matrices(bundle: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return np.tile(np.asarray(bundle["left_bits"], dtype=np.uint16), (1, 30, 1)), np.asarray([bundle["cases"][column % CASES]["right_bits"] for column in range(640)], dtype=np.uint16)


def report_from_observations(plan: dict[str, Any], bundle: dict[str, Any], observations: list[dict[str, Any]], runtime: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(observations, list) or len(observations) != 3:
        raise ValueError("Expected three controlled K2048 observations")
    left, right = _matrices(bundle)
    arrays = []
    for observation in observations:
        arrays.append(_state_array(observation["output_bits"], [1, 30, 640]))
        for name, values in (("input", left), ("weight", right)):
            descriptor, geometry = _descriptor(values), observation[name]
            if any(geometry["tensor"].get(key) != descriptor[key] for key in ("kind", "shape", "dtype", "layout", "numel", "sha256")) or geometry["strides"] != ([61440, 2048, 1] if name == "input" else [2048, 1]) or not str(geometry["tensor"].get("device", "")).startswith("cuda:"):
                raise ValueError("K2048 operand geometry/value mismatch")
        if not isinstance(observation["kernel_names"], list) or not observation["kernel_names"] or any(not isinstance(name, str) or not name for name in observation["kernel_names"]):
            raise ValueError("Missing K2048 CUDA provenance")
    results = []
    for index, candidate in enumerate(plan["candidates"]):
        expected = np.asarray([plan["cases"][column % CASES]["prediction_bits"][index] for column in range(640)], dtype=np.uint16)[None, None, :]
        counts, cases, first = [], [], []
        for actual in arrays:
            different = actual != expected
            coordinates = np.argwhere(different)
            counts.append(int(np.count_nonzero(different)))
            cases.append(sorted({int(column % CASES) for column in np.where(np.any(different, axis=(0, 1)))[0]}))
            first.append({"coordinate": coordinates[0].tolist(), "predicted_bits": int(expected[0, 0, coordinates[0, 2]]), "observed_bits": int(actual[tuple(coordinates[0])])} if len(coordinates) else None)
        results.append({"candidate_id": candidate["id"], "mismatch_counts": counts, "mismatch_case_indices": cases, "first_mismatches": first})
    survivors = [item["candidate_id"] for item in results if item["mismatch_counts"] == [0, 0, 0]]
    runtime_match = canonical_json(runtime) == canonical_json(plan["runtime"])
    kernel_match = all(item["kernel_names"] == plan["expected_kernel_names"] for item in observations)
    repeated = all(np.array_equal(array, arrays[0]) for array in arrays)
    duplicate = all(np.array_equal(array, np.tile(array[0, 0, :CASES], (1, 30, 5))) for array in arrays)
    return _seal({"schema_version": 1, "scope": SCOPE, "plan_sha256": plan["plan_sha256"], "observations": observations, "runtime": runtime,
                  "candidate_results": results, "surviving_candidates": survivors, "minimum_total_mismatches": min(sum(item["mismatch_counts"]) for item in results),
                  "runtime_matches_plan": runtime_match, "kernel_names_match": kernel_match, "repeated_outputs_identical": repeated, "duplicate_coordinates_match": duplicate,
                  "survivors_supported_in_declared_scope": bool(survivors) and runtime_match and kernel_match and repeated and duplicate,
                  "unique_survivor_in_frozen_grid": len(survivors) == 1, "case_count": CASES, "matrix_value_count": 19200,
                  "split_boundaries_observed": False, "intermediate_values_observed": False, "original_model_revalidated": False,
                  "hardware_semantics_established": False, "global_exactness_activation_allowed": False}, "report_sha256")


def acquire_probes(binding: dict[str, Any], product: dict[str, Any], plan: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    import torch
    from .operational_semantics import tensor_descriptor

    check_plan(binding, product, plan, bundle)
    if canonical_json(_runtime()) != canonical_json(plan["runtime"]):
        raise ValueError("K2048 acquisition runtime differs from the plan")
    left_bits, right_bits = _matrices(bundle)
    left, right = torch.from_numpy(left_bits).view(torch.bfloat16).to("cuda"), torch.from_numpy(right_bits).view(torch.bfloat16).to("cuda")
    if left.data_ptr() % 16 or right.data_ptr() % 16:
        raise ValueError("K2048 controlled operands must be aligned")
    geometry = {"input": {"tensor": tensor_descriptor(left), "strides": list(left.stride())}, "weight": {"tensor": tensor_descriptor(right), "strides": list(right.stride())}}
    observations = []
    with torch.no_grad():
        for _ in range(3):
            output, names = _profile_call(lambda: torch.nn.functional.linear(left, right))
            observations.append({**geometry, "output_bits": _bits(output).tolist(), "kernel_names": names})
    _code_sha()
    return report_from_observations(plan, bundle, observations, _runtime())


def verify_probes(binding: dict[str, Any], product: dict[str, Any], plan: dict[str, Any], bundle: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    try:
        check_plan(binding, product, plan, bundle)
        expected = report_from_observations(plan, bundle, report["observations"], report["runtime"])
        return {"valid": canonical_json(expected) == canonical_json(report), "mode": "integrity_generator_geometry_and_grid_accounting",
                "surviving_candidates": expected["surviving_candidates"], "survivors_supported_in_declared_scope": expected["survivors_supported_in_declared_scope"],
                "unique_survivor_in_frozen_grid": expected["unique_survivor_in_frozen_grid"], "global_exactness_activation_allowed": False}
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError) as error:
        return {"valid": False, "reason": str(error)}


def probe_summary(plan: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in report.items() if key not in ("observations", "candidate_results", "report_sha256")}
    body.update({"source_report_sha256": report["report_sha256"], "code_sha256": plan["code_sha256"], "candidate_count": 256,
                 "candidate_pairs_separated": plan["candidate_pairs_separated"], "prediction_equivalence_classes": plan["prediction_equivalence_classes"],
                 "candidate_results": [{key: value for key, value in result.items() if key not in ("mismatch_case_indices", "first_mismatches")} for result in report["candidate_results"]],
                 "kernel_names": [item["kernel_names"] for item in report["observations"]],
                 "observed_output_hashes": [_descriptor(_state_array(item["output_bits"], [1, 30, 640]))["sha256"] for item in report["observations"]]})
    return _seal(body, "summary_sha256")
