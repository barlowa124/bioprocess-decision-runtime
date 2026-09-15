from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
import hashlib
from typing import Any

from .gemma_float_semantics import decode_finite_bfloat16, encode_bfloat16_rne
from .gemma_reduction_backend import verify_gemma_reduction_backend_binding
from .gemma_wmma_accumulator_probe import verify_wmma_accumulator_probe_certificate
from .gemma_wmma_magnitude_probe import verify_wmma_magnitude_probe_certificate
from .gemma_wmma_probe import verify_wmma_probe_certificate
from .serialization import canonical_json


FROZEN_CANDIDATE = (25, "toward_zero", "lower_then_upper")
HOLDOUT_SEED = 0x6D2B79F5
HOLDOUT_FAMILIES = ("dense_narrow", "dense_wide", "paired_cancellation", "rounding_boundary")

PRECISION_BITS = tuple(range(16, 32))
ROUNDING_MODES = ("toward_zero", "nearest_even")
HALF_ORDERS = ("lower_then_upper", "upper_then_lower")


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _scale_power_of_two(value: Fraction, exponent: int) -> Fraction:
    if exponent >= 0:
        return Fraction(value.numerator << exponent, value.denominator)
    return Fraction(value.numerator, value.denominator << -exponent)


def _at_least_power_of_two(value: Fraction, exponent: int) -> bool:
    if exponent >= 0:
        return value.numerator >= value.denominator << exponent
    return value.numerator << -exponent >= value.denominator


def _floor_log2(value: Fraction) -> int:
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if not _at_least_power_of_two(value, exponent):
        exponent -= 1
    while _at_least_power_of_two(value, exponent + 1):
        exponent += 1
    return exponent


def _round_units(value: Fraction, mode: str) -> int:
    negative = value < 0
    magnitude = abs(value)
    quotient, remainder = divmod(magnitude.numerator, magnitude.denominator)
    if mode == "nearest_even":
        comparison = remainder * 2 - magnitude.denominator
        if comparison > 0 or (comparison == 0 and quotient % 2 == 1):
            quotient += 1
    elif mode != "toward_zero":
        raise ValueError("Unknown shared-exponent rounding mode")
    return -quotient if negative else quotient


def _shared_exponent_sum(
    values: list[Fraction], precision_bits: int, rounding_mode: str
) -> Fraction:
    nonzero = [abs(value) for value in values if value != 0]
    if not nonzero:
        return Fraction(0)
    maximum_exponent = max(_floor_log2(value) for value in nonzero)
    quantum = _scale_power_of_two(
        Fraction(1), maximum_exponent - (precision_bits - 1)
    )
    return sum(
        (_round_units(value / quantum, rounding_mode) * quantum for value in values),
        Fraction(0),
    )


@lru_cache(maxsize=None)
def _decoded_k16(value_bits: tuple[int, ...]) -> tuple[Fraction, ...]:
    return tuple(decode_finite_bfloat16(bits)[0] for bits in value_bits)


@lru_cache(maxsize=None)
def _cached_k16_half_transition_bits(
    value_bits: tuple[int, ...],
    precision_bits: int,
    rounding_mode: str,
    half_order: str,
) -> int:
    values = _decoded_k16(value_bits)
    halves = (range(0, 8), range(8, 16))
    if half_order == "upper_then_lower":
        halves = tuple(reversed(halves))
    elif half_order != "lower_then_upper":
        raise ValueError("Unknown K16 half order")
    accumulator = Fraction(0)
    for indices in halves:
        accumulator = _shared_exponent_sum(
            [accumulator, *(values[index] for index in indices)],
            precision_bits,
            rounding_mode,
        )
    return encode_bfloat16_rne(accumulator)


def k16_half_transition_bits(
    value_bits: list[int],
    precision_bits: int,
    rounding_mode: str,
    half_order: str,
) -> int:
    if len(value_bits) != 16:
        raise ValueError("K16 half-transition requires exactly 16 bfloat16 values")
    return _cached_k16_half_transition_bits(
        tuple(value_bits), precision_bits, rounding_mode, half_order
    )


def _record_vector(record: dict[str, Any]) -> list[int]:
    values = [0] * 16
    value_bits = record.get("value_bits")
    if value_bits is None:
        value_bits = [*record["large_value_bits"], record["small_value_bits"]]
    for position, value in zip(record["positions"], value_bits):
        lane = position % 16
        if values[lane] != 0:
            raise ValueError("Probe assigns multiple values to one K16 lane")
        values[lane] = int(value, 16)
    return values


def _datasets(
    position_probe: dict[str, Any],
    accumulator_probe: dict[str, Any],
    magnitude_probe: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    return {
        "four_term_within_fragment": [
            record
            for record in position_probe["records"]
            if record["family"] != "cross_fragment"
        ],
        "three_term_exhaustive": accumulator_probe["records"],
        "signed_magnitude": magnitude_probe["records"],
    }


def build_wmma_candidate_search(
    program: dict[str, Any],
    reduction_certificate: dict[str, Any],
    nsight_suite: dict[str, Any],
    backend_binding: dict[str, Any],
    position_probe: dict[str, Any],
    accumulator_probe: dict[str, Any],
    magnitude_probe: dict[str, Any],
) -> dict[str, Any]:
    if not verify_gemma_reduction_backend_binding(
        program, reduction_certificate, nsight_suite, backend_binding
    )["valid"]:
        raise ValueError("Reduction backend binding is invalid")
    if not verify_wmma_probe_certificate(
        program, reduction_certificate, nsight_suite, backend_binding, position_probe
    )["valid"]:
        raise ValueError("WMMA position probe is invalid")
    if not verify_wmma_accumulator_probe_certificate(
        program,
        reduction_certificate,
        nsight_suite,
        backend_binding,
        accumulator_probe,
    )["valid"]:
        raise ValueError("WMMA accumulator probe is invalid")
    if not verify_wmma_magnitude_probe_certificate(
        program,
        reduction_certificate,
        nsight_suite,
        backend_binding,
        magnitude_probe,
    )["valid"]:
        raise ValueError("WMMA magnitude probe is invalid")
    datasets = _datasets(position_probe, accumulator_probe, magnitude_probe)
    prepared = {
        name: [
            (_record_vector(record), int(record["actual_result_bits"], 16))
            for record in records
        ]
        for name, records in datasets.items()
    }
    profiles = []
    for precision_bits in PRECISION_BITS:
        for rounding_mode in ROUNDING_MODES:
            for half_order in HALF_ORDERS:
                matches = {}
                for name, records in prepared.items():
                    matches[name] = sum(
                        k16_half_transition_bits(
                            values, precision_bits, rounding_mode, half_order
                        )
                        == actual
                        for values, actual in records
                    )
                total_matches = sum(matches.values())
                total_records = sum(len(records) for records in prepared.values())
                profiles.append(
                    {
                        "precision_bits": precision_bits,
                        "rounding_mode": rounding_mode,
                        "half_order": half_order,
                        "dataset_matches": matches,
                        "total_matches": total_matches,
                        "total_records": total_records,
                        "matches_all": total_matches == total_records,
                    }
                )
    profiles.sort(
        key=lambda item: (
            -item["total_matches"],
            item["precision_bits"],
            item["rounding_mode"],
            item["half_order"],
        )
    )
    complete = [
        {
            "precision_bits": profile["precision_bits"],
            "rounding_mode": profile["rounding_mode"],
            "half_order": profile["half_order"],
        }
        for profile in profiles
        if profile["matches_all"]
    ]
    best_matches = profiles[0]["total_matches"]
    best_profiles = [
        profile for profile in profiles if profile["total_matches"] == best_matches
    ]
    reference_profile = best_profiles[0]
    mismatch_frontier = []
    for dataset_name, records in datasets.items():
        for record in records:
            predicted = k16_half_transition_bits(
                _record_vector(record),
                reference_profile["precision_bits"],
                reference_profile["rounding_mode"],
                reference_profile["half_order"],
            )
            actual = int(record["actual_result_bits"], 16)
            if predicted != actual:
                mismatch_frontier.append(
                    {
                        "dataset": dataset_name,
                        "probe_index": record["probe_index"],
                        "placement": record.get("placement"),
                        "family": record.get("family"),
                        "magnitude_index": record.get("magnitude_index"),
                        "negative": record.get("negative"),
                        "exponent_field": record.get("exponent_field"),
                        "fraction_field": record.get("fraction_field"),
                        "predicted_result_bits": f"0x{predicted:04x}",
                        "actual_result_bits": f"0x{actual:04x}",
                    }
                )
    body = {
        "schema_version": 1,
        "scope": "Search of shared-exponent K8 half-transition candidates against three controlled WMMA probe datasets; candidate fit does not establish CUTLASS, WMMA, tensor-core, or hardware semantics.",
        "reduction_backend_binding_sha256": backend_binding["binding_sha256"],
        "position_probe_sha256": position_probe["certificate_sha256"],
        "accumulator_probe_sha256": accumulator_probe["certificate_sha256"],
        "magnitude_probe_sha256": magnitude_probe["certificate_sha256"],
        "dataset_record_counts": {
            name: len(records) for name, records in datasets.items()
        },
        "searched_precision_bits": list(PRECISION_BITS),
        "searched_rounding_modes": list(ROUNDING_MODES),
        "searched_half_orders": list(HALF_ORDERS),
        "profile_count": len(profiles),
        "profiles": profiles,
        "best_total_matches": best_matches,
        "best_profiles": best_profiles,
        "mismatch_reference_profile": {
            "precision_bits": reference_profile["precision_bits"],
            "rounding_mode": reference_profile["rounding_mode"],
            "half_order": reference_profile["half_order"],
        },
        "mismatch_frontier_count": len(mismatch_frontier),
        "mismatch_frontier": mismatch_frontier,
        "complete_matching_profiles": complete,
        "unique_all_matching_candidate_in_search_space": len(complete) == 1,
        "complete_numeric_transition_established": False,
        "wmma_accumulator_mapping_identified": False,
        "reduction_order_semantics_established": False,
        "hardware_instruction_semantics_established": False,
    }
    return {**body, "search_sha256": _sha256(body)}


def verify_wmma_candidate_search(
    program: dict[str, Any],
    reduction_certificate: dict[str, Any],
    nsight_suite: dict[str, Any],
    backend_binding: dict[str, Any],
    position_probe: dict[str, Any],
    accumulator_probe: dict[str, Any],
    magnitude_probe: dict[str, Any],
    search: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(search, dict):
        return {"valid": False}
    try:
        expected = build_wmma_candidate_search(
            program,
            reduction_certificate,
            nsight_suite,
            backend_binding,
            position_probe,
            accumulator_probe,
            magnitude_probe,
        )
        search_hash_valid = _sha256(
            {key: value for key, value in search.items() if key != "search_sha256"}
        ) == search.get("search_sha256")
    except (TypeError, ValueError, RuntimeError):
        return {"valid": False}
    exact_match = search == expected
    return {
        "valid": search_hash_valid and exact_match,
        "search_hash_valid": search_hash_valid,
        "derived_search_exact_match": exact_match,
        "profile_count": expected["profile_count"],
        "best_total_matches": expected["best_total_matches"],
        "total_records": sum(expected["dataset_record_counts"].values()),
        "complete_matching_profile_count": len(
            expected["complete_matching_profiles"]
        ),
        "unique_all_matching_candidate_in_search_space": expected[
            "unique_all_matching_candidate_in_search_space"
        ],
        "complete_numeric_transition_established": expected[
            "complete_numeric_transition_established"
        ],
    }


def build_k16_holdout_plan(search: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(search, dict):
        raise ValueError("Candidate search must be a mapping")
    body = {key: value for key, value in search.items() if key != "search_sha256"}
    selected = dict(zip(("precision_bits", "rounding_mode", "half_order"), FROZEN_CANDIDATE))
    if _sha256(body) != search.get("search_sha256") or search.get("complete_matching_profiles") != [selected]:
        raise ValueError("Holdout requires the frozen 25-bit candidate search commitment")
    state = HOLDOUT_SEED

    def next_word() -> int:
        nonlocal state
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        return state & 0xFFFFFFFF

    cases = []
    for family in HOLDOUT_FAMILIES:
        for index in range(256):
            values = []
            for lane in range(16):
                word = next_word()
                sign = (word >> 16) & 0x8000
                exponent = 122 + word % 11 if family == "dense_narrow" else 90 + word % 71
                values.append(sign | (exponent << 7) | ((word >> 8) & 127))
            if family == "paired_cancellation":
                values = [item for bits in values[:8] for item in (bits, bits ^ 0x8000)]
                values[index % 16] = 0x3F80 | (next_word() & 127)
            elif family == "rounding_boundary":
                values = [((next_word() >> 16) & 0x8000) | ((128 + next_word() % 12) << 7) | (next_word() & 127) for _ in range(16)]
                values[0], values[1] = 0x4E80, 0xCE80
            predicted = k16_half_transition_bits(values, *FROZEN_CANDIDATE)
            cases.append({"index": len(cases), "family": family, "value_bits": values,
                          "predicted_bits": predicted})
    body = {
        "schema_version": 1,
        "scope": "Prospective dense K16 holdout for a fixed fitted candidate; not independent hardware validation.",
        "search_sha256": search["search_sha256"],
        "candidate": selected,
        "seed": HOLDOUT_SEED,
        "input_shape": [30, 640],
        "weight_shape": [1024, 640],
        "active_k_indices": list(range(16)),
        "input_value_bits": 0x3F80,
        "cases": cases,
        "predictions_sha256": _sha256([case["predicted_bits"] for case in cases]),
        "cross_fragment_composition_tested": False,
        "candidate_refitting_allowed": False,
    }
    return {**body, "plan_sha256": _sha256(body)}


def _holdout_report(plan: dict[str, Any], observed: list[list[int]], kernels: list[list[str]],
                    environment: dict[str, Any], families: tuple[str, ...] = HOLDOUT_FAMILIES,
                    scope: str = "Held-out bit comparison of the unchanged K16 candidate; integrity checks do not replay CUDA.") -> dict[str, Any]:
    if not isinstance(observed, list) or not isinstance(kernels, list) or len(observed) != 3 or len(kernels) != 3:
        raise ValueError("Holdout requires three separately profiled repetitions")
    if not isinstance(environment, dict):
        raise ValueError("Invalid holdout environment")
    if any(not isinstance(row, list) or len(row) != 1024 or any(type(bits) is not int or not 0 <= bits <= 65535 for bits in row) for row in observed):
        raise ValueError("Invalid held-out output bits")
    if any(not isinstance(names, list) or not names or any(not isinstance(name, str) or not name for name in names) for names in kernels):
        raise ValueError("Missing CUDA kernel observations")
    predicted = [case["predicted_bits"] for case in plan["cases"]]
    mismatches = [
        {"repetition": repetition, "index": index, "family": plan["cases"][index]["family"],
         "predicted_bits": expected, "observed_bits": actual}
        for repetition, row in enumerate(observed)
        for index, (expected, actual) in enumerate(zip(predicted, row)) if expected != actual
    ]
    body = {
        "schema_version": 1,
        "scope": scope,
        "plan_sha256": plan["plan_sha256"],
        "search_sha256": plan["search_sha256"],
        "predictions_sha256": plan["predictions_sha256"],
        "environment": environment,
        "observed_bits": observed,
        "cuda_kernel_names": kernels,
        "repetitions": 3,
        "case_count": len(predicted),
        "repeated_outputs_identical": all(row == observed[0] for row in observed),
        "kernel_names_stable": all(names == kernels[0] for names in kernels),
        "matching_cases_by_family": {
            family: [sum(row[case["index"]] == case["predicted_bits"] for case in plan["cases"] if case["family"] == family) for row in observed]
            for family in families
        },
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "candidate_passes_holdout": not mismatches,
        "candidate_refitted": False,
        "hardware_semantics_established": False,
        "global_exactness_activation_allowed": False,
    }
    return {**body, "report_sha256": _sha256(body)}


def acquire_k16_holdout(plan: dict[str, Any], search: dict[str, Any]) -> dict[str, Any]:
    if plan != build_k16_holdout_plan(search):
        raise ValueError("Holdout plan differs from the frozen specification")
    vectors = [case["value_bits"] + [0] * 624 for case in plan["cases"]]
    observed, kernels, environment = _acquire_holdout_vectors(vectors)
    return _holdout_report(plan, observed, kernels, environment)


def _acquire_holdout_vectors(vectors: list[list[int]]) -> tuple[list[list[int]], list[list[str]], dict[str, Any]]:
    import torch
    from .gemma_reduction_backend import _profile_call

    if len(vectors) != 1024 or any(len(row) != 640 for row in vectors):
        raise ValueError("Controlled projection requires 1024 rows of 640 values")
    if any(type(bits) is not int or not 0 <= bits <= 65535 for row in vectors for bits in row):
        raise ValueError("Controlled projection values must be bfloat16 bit patterns")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for held-out acquisition")
    weights_bits = torch.tensor(vectors, dtype=torch.uint16)
    weights = weights_bits.view(torch.bfloat16).cuda()
    inputs = torch.ones((30, 640), dtype=torch.bfloat16, device="cuda")
    observed, kernels = [], []
    with torch.no_grad():
        for _ in range(3):
            result, names = _profile_call(lambda: torch.nn.functional.linear(inputs, weights))
            bits = result.cpu().view(torch.uint16)
            if not all(torch.equal(bits[0], row) for row in bits):
                raise RuntimeError("Equal input rows produced different held-out output bits")
            observed.append(bits[0].tolist())
            kernels.append(names)
    environment = {
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    return observed, kernels, environment


def verify_k16_holdout(plan: dict[str, Any], search: dict[str, Any], report: dict[str, Any],
                       reexecute: bool = False) -> dict[str, Any]:
    try:
        if plan != build_k16_holdout_plan(search):
            return {"valid": False, "reason": "Frozen plan mismatch"}
        expected = _holdout_report(plan, report["observed_bits"], report["cuda_kernel_names"], report["environment"])
        integrity = report == expected
        replay = acquire_k16_holdout(plan, search) == report if reexecute and integrity else None
        return {
            "valid": integrity and (not reexecute or replay is True),
            "mode": "cuda_replay" if reexecute else "integrity_only",
            "reexecution_exact": replay,
            "candidate_passes_holdout": expected["candidate_passes_holdout"],
            "mismatch_count": expected["mismatch_count"],
            "matching_cases_by_family": expected["matching_cases_by_family"],
            "repeated_outputs_identical": expected["repeated_outputs_identical"],
            "kernel_names_stable": expected["kernel_names_stable"],
            "global_exactness_activation_allowed": False,
        }
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError) as error:
        return {"valid": False, "reason": str(error)}


COMPOSITION_SEED = 0xA713CF29
COMPOSITION_FAMILIES = ("adjacent_carry", "distant_cancellation", "k128_boundary", "dense_k640")
COMPOSITION_SCOPE = "Held-out test of a new serial K8 composition hypothesis with exact rational carry and final bfloat16 RNE; not a change to the fitted local K16 candidate or a hardware claim."
CARRY_ROUNDING_PROFILES = (("nearest_even", 8), ("nearest_even", 16), ("toward_zero", 8), ("toward_zero", 16))


def _float32_carry(value: Fraction, mode: str) -> Fraction:
    from .gemma_reduction_semantics import encode_float32_rne, decode_finite_float32

    if mode not in ("nearest_even", "toward_zero"):
        raise ValueError("Unsupported carry rounding mode")
    bits = encode_float32_rne(value)
    if bits & 0x7F800000 == 0x7F800000:
        raise ValueError("Nonfinite float32 carry is outside this experiment")
    rounded = decode_finite_float32(bits)[0]
    if mode == "toward_zero" and abs(rounded) > abs(value):
        bits -= 1
        rounded = decode_finite_float32(bits)[0]
    return rounded


def serial_k8_float32_carry_bits(value_bits: list[int], mode: str, interval: int) -> int:
    if not value_bits or len(value_bits) % 16:
        raise ValueError("Composition requires a nonempty multiple of 16 values")
    if (mode, interval) not in CARRY_ROUNDING_PROFILES or type(interval) is not int:
        raise ValueError("Unsupported carry rounding profile")
    accumulator = Fraction(0)
    for start in range(0, len(value_bits), 8):
        terms = [decode_finite_bfloat16(bits)[0] for bits in value_bits[start:start + 8]]
        accumulator = _shared_exponent_sum([accumulator, *terms], 25, "toward_zero")
        if (start + 8) % interval == 0:
            accumulator = _float32_carry(accumulator, mode)
    return encode_bfloat16_rne(accumulator)


def serial_k8_product_bits(left_bits: list[int], right_bits: list[int]) -> int:
    if not left_bits or len(left_bits) != len(right_bits) or len(left_bits) % 16:
        raise ValueError("Product composition requires equal nonempty multiples of 16")
    accumulator = Fraction(0)
    for start in range(0, len(left_bits), 8):
        products = [decode_finite_bfloat16(left)[0] * decode_finite_bfloat16(right)[0]
                    for left, right in zip(left_bits[start:start + 8], right_bits[start:start + 8])]
        accumulator = _shared_exponent_sum([accumulator, *products], 25, "toward_zero")
        accumulator = _float32_carry(accumulator, "toward_zero")
    return encode_bfloat16_rne(accumulator)


def _operand_aligned_accumulator(left_bits: list[int], right_bits: list[int]) -> Fraction:
    if not left_bits or len(left_bits) != len(right_bits) or len(left_bits) % 16:
        raise ValueError("Operand-aligned composition requires equal nonempty multiples of 16")
    operands = []
    for a, b in zip(left_bits, right_bits):
        left, right = decode_finite_bfloat16(a)[0], decode_finite_bfloat16(b)[0]
        if (a & 0x7F80 == 0 and left) or (b & 0x7F80 == 0 and right):
            raise ValueError("Subnormal operands are outside the alignment hypothesis")
        operands.append((left, right))
    accumulator = Fraction(0)
    for start in range(0, len(operands), 8):
        pairs = operands[start:start + 8]
        exponents = [_floor_log2(abs(a)) + _floor_log2(abs(b)) for a, b in pairs if a and b]
        if accumulator:
            exponents.append(_floor_log2(abs(accumulator)))
        if exponents:
            quantum = _scale_power_of_two(Fraction(1), max(exponents) - 24)
            terms = [accumulator, *(a * b for a, b in pairs)]
            accumulator = sum((_round_units(term / quantum, "toward_zero") * quantum for term in terms), Fraction(0))
        accumulator = _float32_carry(accumulator, "toward_zero")
    return accumulator


def operand_aligned_product_bits(left_bits: list[int], right_bits: list[int]) -> int:
    return encode_bfloat16_rne(_operand_aligned_accumulator(left_bits, right_bits))


def serial_k8_composition_bits(value_bits: list[int]) -> int:
    if not value_bits or len(value_bits) % 16:
        raise ValueError("Composition requires a nonempty multiple of 16 values")
    accumulator = Fraction(0)
    for start in range(0, len(value_bits), 8):
        terms = [decode_finite_bfloat16(bits)[0] for bits in value_bits[start:start + 8]]
        accumulator = _shared_exponent_sum([accumulator, *terms], 25, "toward_zero")
    return encode_bfloat16_rne(accumulator)


def _composition_vectors(seed: int = COMPOSITION_SEED) -> list[tuple[str, list[int]]]:
    if type(seed) is not int or not 0 < seed <= 0xFFFFFFFF:
        raise ValueError("Composition seed must be a nonzero uint32")
    state = seed

    def word() -> int:
        nonlocal state
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        return state & 0xFFFFFFFF

    def finite_bits(low: int, span: int) -> int:
        value = word()
        return ((value >> 16) & 0x8000) | ((low + value % span) << 7) | ((value >> 8) & 127)

    vectors = []
    for family in COMPOSITION_FAMILIES:
        for index in range(256):
            values = [0] * 640
            if family == "adjacent_carry":
                values[:16] = [finite_bits(122, 16) for _ in range(16)]
                values[16:32] = [finite_bits(114, 30) for _ in range(16)]
            elif family == "distant_cancellation":
                first = index % 16
                last_tile = 1 + (index // 16) % 39
                values[first] = 0x4E80
                values[last_tile * 16 + (first + 5) % 16] = 0xCE80
                values[last_tile * 16 + (first + 9) % 16] = finite_bits(126, 14)
                values[(first + 1) % 16] = finite_bits(126, 14)
            elif family == "k128_boundary":
                boundary = (1 + index % 4) * 128
                values[boundary - 16:boundary + 16] = [finite_bits(120, 25) for _ in range(32)]
                values[boundary - 8] = 0x4E80
                values[boundary] = 0xCE80
            else:
                values = [finite_bits(90, 71) for _ in range(640)]
            vectors.append((family, values))
    return vectors


def build_composition_holdout_plan(search: dict[str, Any]) -> dict[str, Any]:
    local_plan = build_k16_holdout_plan(search)
    cases = [
        {"index": index, "family": family, "input_bits_sha256": _sha256(values),
         "active_k16_blocks": sorted({lane // 16 for lane, bits in enumerate(values) if bits != 0}),
         "predicted_bits": serial_k8_composition_bits(values)}
        for index, (family, values) in enumerate(_composition_vectors())
    ]
    body = {
        "schema_version": 1,
        "scope": COMPOSITION_SCOPE,
        "search_sha256": search["search_sha256"],
        "candidate": local_plan["candidate"],
        "generator": "xorshift32-composition-v1",
        "seed": COMPOSITION_SEED,
        "composition": {
            "initial_accumulator": "exact rational zero",
            "half_order": "ascending K in contiguous groups of 8, including zero-only groups",
            "half_transition": "quantize each term and incoming accumulator toward zero at 25 shared-exponent bits; sum exactly",
            "inter_half_state": "exact rational sum; no additional float32 conversion",
            "final_encoding": "bfloat16 round to nearest ties to even",
        },
        "input_shape": [30, 640],
        "weight_shape": [1024, 640],
        "input_value_bits": 0x3F80,
        "cases": cases,
        "predictions_sha256": _sha256([case["predicted_bits"] for case in cases]),
        "cross_fragment_composition_tested": True,
        "candidate_refitting_allowed": False,
        "full_vectors_committed": False,
    }
    return {**body, "plan_sha256": _sha256(body)}


def diagnose_float32_carry(search: dict[str, Any], plan: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    verification = verify_composition_holdout(plan, search, report)
    if not verification["valid"]:
        raise ValueError("Development evidence failed integrity verification")
    vectors = _composition_vectors()
    profiles = []
    for mode, interval in CARRY_ROUNDING_PROFILES:
        predicted = [serial_k8_float32_carry_bits(values, mode, interval) for _, values in vectors]
        mismatches = [
            {"repetition": repetition, "index": index, "family": vectors[index][0],
             "predicted_bits": expected, "observed_bits": actual}
            for repetition, observed in enumerate(report["observed_bits"])
            for index, (expected, actual) in enumerate(zip(predicted, observed)) if expected != actual
        ]
        profiles.append({
            "mode": mode, "interval": interval,
            "predictions_sha256": _sha256(predicted),
            "mismatch_count": len(mismatches),
            "mismatches": mismatches,
            "matches_all_development_records": not mismatches,
            "matching_cases_by_family": {
                family: [sum(predicted[index] == row[index] for index, (name, _) in enumerate(vectors) if name == family) for row in report["observed_bits"]]
                for family in COMPOSITION_FAMILIES
            },
        })
    body = {
        "schema_version": 1,
        "scope": "Post-failure development comparison of four float32 carry-rounding variants; fitting these observed outputs is not held-out validation.",
        "search_sha256": search["search_sha256"],
        "source_plan_sha256": plan["plan_sha256"],
        "source_report_sha256": report["report_sha256"],
        "source_evidence_mode": "integrity_only",
        "source_case_count": 1024, "source_repetitions": 3,
        "baseline_mismatch_count": report["mismatch_count"],
        "profiles": profiles,
        "all_matching_development_profiles": [{"mode": item["mode"], "interval": item["interval"]} for item in profiles if item["matches_all_development_records"]],
        "held_out_validation_established": False,
        "hardware_semantics_established": False,
        "global_exactness_activation_allowed": False,
    }
    return {**body, "diagnosis_sha256": _sha256(body)}


def verify_carry_diagnosis(search: dict[str, Any], plan: dict[str, Any], report: dict[str, Any], diagnosis: dict[str, Any]) -> dict[str, Any]:
    try:
        expected = diagnose_float32_carry(search, plan, report)
        return {"valid": diagnosis == expected, "mode": "software_recomputation",
                "baseline_mismatch_count": expected["baseline_mismatch_count"],
                "profile_mismatch_counts": [{"mode": item["mode"], "interval": item["interval"], "mismatch_count": item["mismatch_count"]} for item in expected["profiles"]],
                "all_matching_development_profiles": expected["all_matching_development_profiles"],
                "held_out_validation_established": False}
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError) as error:
        return {"valid": False, "reason": str(error)}


CARRY_REVISION_SEED = 0xC4B82F17
CARRY_REVISION = ("toward_zero", 8)
CARRY_REVISION_SCOPE = "Fresh same-generator holdout of frozen per-K8 float32-toward-zero carry revision; development cases excluded by vector hash; not unrestricted WMMA or hardware qualification."


def build_carry_revision_plan(search: dict[str, Any], diagnosis: dict[str, Any]) -> dict[str, Any]:
    local_plan = build_k16_holdout_plan(search)
    if not isinstance(diagnosis, dict):
        raise ValueError("Carry diagnosis must be a mapping")
    body = {key: value for key, value in diagnosis.items() if key != "diagnosis_sha256"}
    if _sha256(body) != diagnosis.get("diagnosis_sha256") or diagnosis.get("search_sha256") != search["search_sha256"]:
        raise ValueError("Carry diagnosis commitment mismatch")
    if diagnosis.get("all_matching_development_profiles") != [{"mode": CARRY_REVISION[0], "interval": CARRY_REVISION[1]}]:
        raise ValueError("Diagnosis does not support the frozen carry revision")
    development_hashes = {_sha256(values) for _, values in _composition_vectors()}
    cases = []
    for index, (family, values) in enumerate(_composition_vectors(CARRY_REVISION_SEED)):
        vector_hash = _sha256(values)
        if vector_hash in development_hashes:
            raise ValueError("Fresh holdout overlaps the development inputs")
        cases.append({"index": index, "family": family, "input_bits_sha256": vector_hash,
                      "predicted_bits": serial_k8_float32_carry_bits(values, *CARRY_REVISION)})
    body = {
        "schema_version": 1, "scope": CARRY_REVISION_SCOPE,
        "search_sha256": search["search_sha256"],
        "diagnosis_sha256": diagnosis["diagnosis_sha256"],
        "source_diagnosis_verification": "integrity_only; use gemma-carry-diagnosis to recompute development fit",
        "candidate": local_plan["candidate"],
        "carry_revision": {"mode": CARRY_REVISION[0], "interval": CARRY_REVISION[1]},
        "generator": "xorshift32-composition-v1", "seed": CARRY_REVISION_SEED,
        "development_seed": COMPOSITION_SEED,
        "input_shape": [30, 640], "weight_shape": [1024, 640], "input_value_bits": 0x3F80,
        "cases": cases, "predictions_sha256": _sha256([case["predicted_bits"] for case in cases]),
        "development_vectors_disjoint": True,
        "candidate_refitting_allowed": False,
        "full_vectors_committed": False,
        "exceptional_values_qualified": False,
        "hardware_semantics_established": False,
    }
    return {**body, "plan_sha256": _sha256(body)}


def acquire_carry_revision_holdout(plan: dict[str, Any], search: dict[str, Any], diagnosis: dict[str, Any]) -> dict[str, Any]:
    if plan != build_carry_revision_plan(search, diagnosis):
        raise ValueError("Carry revision plan differs from frozen specification")
    vectors = [values for _, values in _composition_vectors(CARRY_REVISION_SEED)]
    if any(_sha256(values) != case["input_bits_sha256"] for values, case in zip(vectors, plan["cases"])):
        raise ValueError("Carry revision input hash mismatch")
    observed, kernels, environment = _acquire_holdout_vectors(vectors)
    return _holdout_report(plan, observed, kernels, environment, COMPOSITION_FAMILIES, CARRY_REVISION_SCOPE)


def verify_carry_revision_holdout(plan: dict[str, Any], search: dict[str, Any], diagnosis: dict[str, Any], report: dict[str, Any], reexecute: bool = False) -> dict[str, Any]:
    try:
        if plan != build_carry_revision_plan(search, diagnosis):
            return {"valid": False, "reason": "Frozen revision plan mismatch"}
        expected = _holdout_report(plan, report["observed_bits"], report["cuda_kernel_names"], report["environment"], COMPOSITION_FAMILIES, CARRY_REVISION_SCOPE)
        integrity = report == expected
        replay = acquire_carry_revision_holdout(plan, search, diagnosis) == report if reexecute and integrity else None
        return {"valid": integrity and (not reexecute or replay is True),
                "mode": "cuda_replay" if reexecute else "integrity_only",
                "reexecution_exact": replay,
                "candidate_passes_holdout": expected["candidate_passes_holdout"],
                "mismatch_count": expected["mismatch_count"],
                "matching_cases_by_family": expected["matching_cases_by_family"],
                "kernel_names_stable": expected["kernel_names_stable"],
                "repeated_outputs_identical": expected["repeated_outputs_identical"],
                "global_exactness_activation_allowed": False}
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError) as error:
        return {"valid": False, "reason": str(error)}


PRODUCT_SEED = 0x58D91A37
PRODUCT_SHAPES = (("query", 30, 1024, 640), ("key_value", 30, 256, 640), ("mlp_down", 30, 640, 2048))
PRODUCT_FAMILIES = ("dense_narrow", "dense_wide", "paired_cancellation", "boundary_cancellation")
PRODUCT_CASES_PER_SHAPE = 128
PRODUCT_SCOPE = "Prospective exact-product/per-K8 carry comparison on controlled projection shapes; only first 128 output columns tested, repeated input rows, zero-padded other weights; no gate or hardware qualification."


def _product_vectors(seed: int = PRODUCT_SEED) -> list[tuple[list[int], list[tuple[str, list[int]]]]]:
    if type(seed) is not int or not 0 < seed <= 0xFFFFFFFF:
        raise ValueError("Product seed must be a nonzero uint32")
    state = seed

    def word() -> int:
        nonlocal state
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        return state & 0xFFFFFFFF

    def value(low: int, span: int) -> int:
        bits = word()
        return ((bits >> 16) & 0x8000) | ((low + bits % span) << 7) | (1 + (bits >> 8) % 127)

    vectors = []
    for _, _, _, inner in PRODUCT_SHAPES:
        if type(inner) is not int or inner <= 128 or inner % 128:
            raise ValueError("Product shape inner dimension must exceed 128 and be divisible by 128")
        left = [item for _ in range(inner // 2) for item in [value(126, 3)] * 2]
        cases = []
        for family in PRODUCT_FAMILIES:
            for index in range(32):
                if family == "dense_narrow":
                    right = [value(118, 21) for _ in range(inner)]
                elif family == "dense_wide":
                    right = [value(90, 71) for _ in range(inner)]
                elif family == "paired_cancellation":
                    right = [item for _ in range(inner // 2) for bits in [value(120, 25)] for item in (bits, bits ^ 0x8000)]
                    right[index * 2] = value(120, 25)
                else:
                    boundary = 128 * (1 + index % (inner // 128 - 1))
                    right = [0] * inner
                    right[boundary - 16:boundary + 16] = [value(120, 25) for _ in range(32)]
                    first, second = boundary - 2, boundary
                    right[first] = left[second] + (30 << 7)
                    right[second] = (left[first] + (30 << 7)) ^ 0x8000
                cases.append((family, right))
        vectors.append((left, cases))
    return vectors


def build_product_holdout_plan(search: dict[str, Any], diagnosis: dict[str, Any]) -> dict[str, Any]:
    revision = build_carry_revision_plan(search, diagnosis)
    shapes = []
    for (role, rows, outputs, inner), (left, vectors) in zip(PRODUCT_SHAPES, _product_vectors()):
        cases = [{"index": index, "family": family, "right_bits_sha256": _sha256(right),
                  "predicted_bits": serial_k8_product_bits(left, right)}
                 for index, (family, right) in enumerate(vectors)]
        shapes.append({"role": role, "input_shape": [rows, inner], "weight_shape": [outputs, inner],
                       "left_bits_sha256": _sha256(left), "cases": cases})
    body = {
        "schema_version": 1, "scope": PRODUCT_SCOPE,
        "search_sha256": search["search_sha256"], "diagnosis_sha256": diagnosis["diagnosis_sha256"],
        "source_revision_plan_sha256": revision["plan_sha256"],
        "profile": {"product": "exact rational product of finite bfloat16 operands",
                    "shared_exponent_bits": 25, "quantization": "toward_zero",
                    "carry_conversion": "float32 toward zero after every K8, including zero-only groups",
                    "initial_accumulator": "zero", "final_conversion": "bfloat16 RNE"},
        "seed": PRODUCT_SEED, "generator": "xorshift32-nonunit-products-v1",
        "tested_columns_per_shape": PRODUCT_CASES_PER_SHAPE,
        "untested_weight_rows": "zero-filled to retain the declared projection shape",
        "input_rows": "identical copies of non-unit vector; adjacent K entries paired",
        "shapes": shapes,
        "predictions_sha256": _sha256([[case["predicted_bits"] for case in shape["cases"]] for shape in shapes]),
        "candidate_refitting_allowed": False,
        "full_vectors_committed": False,
        "global_exactness_activation_allowed": False,
    }
    return {**body, "plan_sha256": _sha256(body)}


def _product_report(plan: dict[str, Any], acquisitions: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(acquisitions, list) or len(acquisitions) != len(plan["shapes"]):
        raise ValueError("Expected all planned controlled shapes")
    results = []
    for shape, acquired in zip(plan["shapes"], acquisitions):
        if not isinstance(acquired, dict) or acquired.get("role") != shape["role"]:
            raise ValueError("Product shape identity mismatch")
        observed, kernels = acquired.get("observed_bits"), acquired.get("kernel_names")
        if not isinstance(observed, list) or len(observed) != 3 or any(
            not isinstance(row, list) or len(row) != PRODUCT_CASES_PER_SHAPE or
            any(type(bits) is not int or not 0 <= bits <= 65535 for bits in row) for row in observed
        ):
            raise ValueError("Invalid product observations")
        if not isinstance(kernels, list) or len(kernels) != 3 or any(
            not isinstance(names, list) or not names or any(not isinstance(name, str) or not name for name in names) for names in kernels
        ):
            raise ValueError("Missing product kernel observations")
        if not isinstance(acquired.get("environment"), dict):
            raise ValueError("Missing acquisition environment")
        mismatches = [{"repetition": repetition, "index": case["index"], "family": case["family"],
                       "predicted_bits": case["predicted_bits"], "observed_bits": row[case["index"]]}
                      for repetition, row in enumerate(observed) for case in shape["cases"] if row[case["index"]] != case["predicted_bits"]]
        results.append({"role": shape["role"], "mismatch_count": len(mismatches), "mismatches": mismatches,
                        "repeated_outputs_identical": all(row == observed[0] for row in observed),
                        "kernel_names_stable": all(names == kernels[0] for names in kernels),
                        "matching_cases_by_family": {family: [sum(row[case["index"]] == case["predicted_bits"] for case in shape["cases"] if case["family"] == family) for row in observed] for family in PRODUCT_FAMILIES}})
    body = {
        "schema_version": 1, "scope": PRODUCT_SCOPE,
        "plan_sha256": plan["plan_sha256"], "predictions_sha256": plan["predictions_sha256"],
        "acquisitions": acquisitions, "shape_results": results,
        "tested_case_count": len(plan["shapes"]) * PRODUCT_CASES_PER_SHAPE,
        "repetitions": 3,
        "candidate_passes_holdout": all(item["mismatch_count"] == 0 for item in results),
        "mismatch_count": sum(item["mismatch_count"] for item in results),
        "candidate_refitted": False,
        "hardware_semantics_established": False,
        "global_exactness_activation_allowed": False,
    }
    return {**body, "report_sha256": _sha256(body)}


def acquire_product_holdout(plan: dict[str, Any], search: dict[str, Any], diagnosis: dict[str, Any]) -> dict[str, Any]:
    if plan != build_product_holdout_plan(search, diagnosis):
        raise ValueError("Product holdout plan differs from frozen specification")
    return _product_report(plan, _acquire_product_shapes(plan, _product_vectors()))


def _acquire_product_shapes(plan: dict[str, Any], vectors: list[tuple[list[int], list[tuple[str, list[int]]]]]) -> list[dict[str, Any]]:
    import torch
    from .gemma_reduction_backend import _profile_call

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for product holdout")
    if len(vectors) != len(plan["shapes"]):
        raise ValueError("Product acquisition shape count mismatch")
    acquisitions = []
    with torch.no_grad():
        for (left, cases), shape in zip(vectors, plan["shapes"]):
            role = shape["role"]
            rows, inner = shape["input_shape"]
            outputs = shape["weight_shape"][0]
            if len(cases) != PRODUCT_CASES_PER_SHAPE or len(left) != inner or any(len(right) != inner for _, right in cases):
                raise ValueError("Product acquisition dimensions mismatch")
            if _sha256(left) != shape["left_bits_sha256"] or any(_sha256(right) != case["right_bits_sha256"] for (_, right), case in zip(cases, shape["cases"])):
                raise ValueError("Regenerated product vector hash mismatch")
            inputs = torch.tensor(left, dtype=torch.uint16).view(torch.bfloat16).cuda().reshape(1, inner).expand(rows, inner).contiguous()
            weight_bits = torch.zeros((outputs, inner), dtype=torch.uint16)
            weight_bits[:PRODUCT_CASES_PER_SHAPE] = torch.tensor([right for _, right in cases], dtype=torch.uint16)
            weights = weight_bits.view(torch.bfloat16).cuda()
            observed, kernels = [], []
            for _ in range(3):
                result, names = _profile_call(lambda: torch.nn.functional.linear(inputs, weights))
                bits = result.cpu().view(torch.uint16)
                if not all(torch.equal(bits[0], row) for row in bits):
                    raise RuntimeError("Repeated product input rows gave unequal output bits")
                observed.append(bits[0, :PRODUCT_CASES_PER_SHAPE].tolist())
                kernels.append(names)
            acquisitions.append({"role": role, "observed_bits": observed, "kernel_names": kernels,
                                 "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
                                                 "device": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
                                                 "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                                                 "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}})
    return acquisitions


def verify_product_holdout(plan: dict[str, Any], search: dict[str, Any], diagnosis: dict[str, Any], report: dict[str, Any], reexecute: bool = False) -> dict[str, Any]:
    try:
        if plan != build_product_holdout_plan(search, diagnosis):
            return {"valid": False, "reason": "Frozen product plan mismatch"}
        expected = _product_report(plan, report["acquisitions"])
        integrity = report == expected
        replay = acquire_product_holdout(plan, search, diagnosis) == report if reexecute and integrity else None
        return {"valid": integrity and (not reexecute or replay is True), "mode": "cuda_replay" if reexecute else "integrity_only",
                "reexecution_exact": replay, "candidate_passes_holdout": expected["candidate_passes_holdout"],
                "tested_case_count": expected["tested_case_count"], "mismatch_count": expected["mismatch_count"],
                "shape_results": [{key: value for key, value in item.items() if key != "mismatches"} for item in expected["shape_results"]],
                "global_exactness_activation_allowed": False}
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError) as error:
        return {"valid": False, "reason": str(error)}


MATCHED_PRODUCT_SCOPE = "Controlled matched-product diagnosis reusing prior query vectors across shapes and exact power-of-two rescalings; not fresh holdout validation or hardware qualification."


def _scale_normal_bfloat_bits(bits: int, shift: int) -> int:
    decode_finite_bfloat16(bits)
    if type(shift) is not int:
        raise ValueError("Scale exponent must be an integer")
    if bits & 0x7FFF == 0:
        return bits
    exponent = (bits >> 7) & 255
    if exponent == 0 or not 1 <= exponent + shift <= 254:
        raise ValueError("Exact rescaling is restricted to normal finite operands")
    return (bits & 0x807F) | ((exponent + shift) << 7)


def _matched_product_vectors() -> list[tuple[list[int], list[tuple[str, list[int]]]]]:
    left, cases = _product_vectors()[0]
    vectors = []
    for _, _, _, inner in PRODUCT_SHAPES:
        for shift in (0, 1):
            transformed_left = [_scale_normal_bfloat_bits(bits, shift) for bits in left] + [0] * (inner - 640)
            transformed_cases = [(family, [_scale_normal_bfloat_bits(bits, -shift) for bits in right] + [0] * (inner - 640)) for family, right in cases]
            vectors.append((transformed_left, transformed_cases))
    return vectors


def build_matched_product_plan(search: dict[str, Any], diagnosis: dict[str, Any]) -> dict[str, Any]:
    original = build_product_holdout_plan(search, diagnosis)
    source_left, source_cases = _product_vectors()[0]
    source_products = [[decode_finite_bfloat16(a)[0] * decode_finite_bfloat16(b)[0] for a, b in zip(source_left, right)] for _, right in source_cases]
    configurations = [(role, rows, outputs, inner, shift) for role, rows, outputs, inner in PRODUCT_SHAPES for shift in (0, 1)]
    shapes = []
    for (role, rows, outputs, inner, shift), (left, vectors) in zip(configurations, _matched_product_vectors()):
        cases = []
        for index, (family, right) in enumerate(vectors):
            products = [decode_finite_bfloat16(a)[0] * decode_finite_bfloat16(b)[0] for a, b in zip(left, right)]
            if products != source_products[index] + [Fraction(0)] * (inner - 640):
                raise ValueError("Rescaling or padding changed exact products")
            cases.append({"index": index, "family": family, "right_bits_sha256": _sha256(right),
                          "predicted_bits": serial_k8_product_bits(left, right)})
        shapes.append({"role": f"{role}:shift{shift}", "base_role": role, "rescaling_exponent": shift,
                       "input_shape": [rows, inner], "weight_shape": [outputs, inner],
                       "left_bits_sha256": _sha256(left), "cases": cases})
    predicted = [[case["predicted_bits"] for case in shape["cases"]] for shape in shapes]
    body = {
        "schema_version": 1, "scope": MATCHED_PRODUCT_SCOPE,
        "search_sha256": search["search_sha256"], "diagnosis_sha256": diagnosis["diagnosis_sha256"],
        "source_product_plan_sha256": original["plan_sha256"],
        "source_query_vectors_reused": True,
        "exact_products_identical_ignoring_trailing_zeros": True,
        "source_product_vector_length": 640,
        "exact_products_sha256": _sha256([[[value.numerator, value.denominator] for value in row] for row in source_products]),
        "rescaling": "input multiplied by 2^shift, weights by 2^-shift with exact normal bfloat16 exponent edits",
        "longer_inner_dimensions": "pad both operands with positive zeros",
        "candidate_profile": original["profile"],
        "shapes": shapes, "tested_columns_per_shape": PRODUCT_CASES_PER_SHAPE,
        "predictions_sha256": _sha256(predicted),
        "all_candidate_predictions_identical": all(row == predicted[0] for row in predicted),
        "candidate_refitting_allowed": False,
        "fresh_holdout_validation_established": False,
        "global_exactness_activation_allowed": False,
    }
    return {**body, "plan_sha256": _sha256(body)}


def _matched_product_report(plan: dict[str, Any], acquisitions: list[dict[str, Any]]) -> dict[str, Any]:
    body = _product_report(plan, acquisitions)
    body.pop("report_sha256")
    body["scope"] = MATCHED_PRODUCT_SCOPE
    body["candidate_passes_controlled_comparison"] = body.pop("candidate_passes_holdout")
    body["fresh_holdout_validation_established"] = False
    comparisons = []
    for reference, target, kind in ((0, 1, "rescaling"), (2, 3, "rescaling"), (4, 5, "rescaling"), (0, 2, "output_width"), (1, 3, "output_width"), (0, 4, "shape_and_zero_padding"), (1, 5, "shape_and_zero_padding")):
        before, after = acquisitions[reference], acquisitions[target]
        differences = [{"repetition": repetition, "index": index, "reference_bits": a, "target_bits": b}
                       for repetition in range(3) for index, (a, b) in enumerate(zip(before["observed_bits"][repetition], after["observed_bits"][repetition])) if a != b]
        comparisons.append({"kind": kind, "reference": before["role"], "target": after["role"],
                            "difference_count": len(differences), "differences": differences,
                            "kernel_name_sequences_equal": before["kernel_names"] == after["kernel_names"]})
    body["comparisons"] = comparisons
    return {**body, "report_sha256": _sha256(body)}


def acquire_matched_products(plan: dict[str, Any], search: dict[str, Any], diagnosis: dict[str, Any]) -> dict[str, Any]:
    if plan != build_matched_product_plan(search, diagnosis):
        raise ValueError("Matched-product plan differs from frozen specification")
    return _matched_product_report(plan, _acquire_product_shapes(plan, _matched_product_vectors()))


def verify_matched_products(plan: dict[str, Any], search: dict[str, Any], diagnosis: dict[str, Any], report: dict[str, Any], reexecute: bool = False) -> dict[str, Any]:
    try:
        if plan != build_matched_product_plan(search, diagnosis):
            return {"valid": False, "reason": "Frozen matched-product plan mismatch"}
        expected = _matched_product_report(plan, report["acquisitions"])
        integrity = report == expected
        replay = acquire_matched_products(plan, search, diagnosis) == report if reexecute and integrity else None
        return {"valid": integrity and (not reexecute or replay is True), "mode": "cuda_replay" if reexecute else "integrity_only",
                "reexecution_exact": replay, "candidate_passes_controlled_comparison": expected["candidate_passes_controlled_comparison"],
                "candidate_mismatch_count": expected["mismatch_count"],
                "comparisons": [{key: value for key, value in item.items() if key != "differences"} for item in expected["comparisons"]],
                "fresh_holdout_validation_established": False, "global_exactness_activation_allowed": False}
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError) as error:
        return {"valid": False, "reason": str(error)}


QUERY_REDUCTION_SCOPE = "Adaptive development-only deletion of query counterexample terms at original columns and fixed full shape; no candidate refit, fresh validation, or hardware qualification."
QUERY_CHUNKS = (320, 160, 80, 40, 20, 10, 5, 2, 1)


def build_query_reduction_plan(search: dict[str, Any], diagnosis: dict[str, Any], source_plan: dict[str, Any], source_report: dict[str, Any]) -> dict[str, Any]:
    for certificate, field in ((source_plan, "plan_sha256"), (source_report, "report_sha256")):
        if not isinstance(certificate, dict) or _sha256({k: v for k, v in certificate.items() if k != field}) != certificate.get(field):
            raise ValueError("Query source commitment mismatch")
    verified = verify_product_holdout(source_plan, search, diagnosis, source_report)
    if not verified["valid"] or not source_report["shape_results"][0]["repeated_outputs_identical"]:
        raise ValueError("Query reduction requires intact repeatable source evidence")
    query = source_report["acquisitions"][0]
    indices = [case["index"] for case in source_plan["shapes"][0]["cases"] if case["predicted_bits"] != query["observed_bits"][0][case["index"]]]
    if not indices:
        raise ValueError("No query counterexamples to reduce")
    body = {"schema_version": 1, "scope": QUERY_REDUCTION_SCOPE,
            "source_plan_sha256": source_plan["plan_sha256"], "source_report_sha256": source_report["report_sha256"],
            "case_indices": indices, "input_shape": [30, 640], "weight_shape": [1024, 640],
            "chunk_sizes": list(QUERY_CHUNKS), "max_trials_per_case": 256,
            "deletion_protocol": "ascending contiguous chunks at each width; repeat singleton passes until no deletion accepted or budget reached",
            "acceptance": "three identical outputs differ from fixed prediction; original kernels, environment, and other 127 tested columns unchanged",
            "background": "original first 128 weight rows, other weight rows zero; restore all source rows before each independent case",
            "candidate_profile": source_plan["profile"],
            "candidate_refitting_allowed": False, "fresh_holdout_validation_established": False,
            "global_exactness_activation_allowed": False}
    return {**body, "plan_sha256": _sha256(body)}


def _query_reduction_engine(plan: dict[str, Any], baseline: dict[str, Any], observe: Any, persist: Any = None) -> dict[str, Any]:
    left, source_cases = _product_vectors()[0]
    trials, final_cases = [], []
    for index in plan["case_indices"]:
        current = list(source_cases[index][1])
        current_observed = [row[index] for row in baseline["observed_bits"]]
        count, minimal = 0, False
        for width in plan["chunk_sizes"]:
            while True:
                changed, clean = False, True
                for start in range(0, 640, width):
                    proposal = current[:]
                    proposal[start:start + width] = [0] * len(proposal[start:start + width])
                    if proposal == current:
                        continue
                    if count >= plan["max_trials_per_case"]:
                        clean = False
                        break
                    predicted = serial_k8_product_bits(left, proposal)
                    spec = {"case_index": index, "chunk_size": width, "start": start,
                            "remaining_indices": [k for k, bits in enumerate(proposal) if bits & 0x7FFF],
                            "right_bits_sha256": _sha256(proposal), "predicted_bits": predicted}
                    if persist is not None:
                        persist({"plan_sha256": plan["plan_sha256"], "baseline": baseline, "trials": trials, "pending": spec})
                    observed = observe(index, proposal)
                    bits = observed.get("bits")
                    if not isinstance(bits, list) or len(bits) != 3 or any(type(item) is not int or not 0 <= item <= 65535 for item in bits):
                        raise ValueError("Malformed query trial outputs")
                    stable = (observed.get("kernel_names") == baseline["kernel_names"] and
                              observed.get("environment") == baseline["environment"] and
                              observed.get("background_differences") == [] and bits[0] == bits[1] == bits[2])
                    accepted = stable and bits[0] != predicted
                    trials.append({**spec, "observations": observed, "controlled_conditions_preserved": stable, "accepted": accepted})
                    count += 1
                    clean = clean and stable
                    if accepted:
                        current, current_observed = proposal, bits
                        changed = True
                    if persist is not None:
                        persist({"plan_sha256": plan["plan_sha256"], "baseline": baseline, "trials": trials, "pending": None})
                if width != 1 or not changed or count >= plan["max_trials_per_case"]:
                    minimal = width == 1 and not changed and clean
                    break
            if count >= plan["max_trials_per_case"]:
                break
        exact = sum((decode_finite_bfloat16(a)[0] * decode_finite_bfloat16(b)[0] for a, b in zip(left, current)), Fraction(0))
        final_cases.append({"case_index": index, "family": source_cases[index][0], "trial_count": count,
                            "terms": [{"position": k, "left_bits": left[k], "right_bits": bits} for k, bits in enumerate(current) if bits & 0x7FFF],
                            "right_bits_sha256": _sha256(current), "predicted_bits": serial_k8_product_bits(left, current),
                            "observed_bits": current_observed, "exact_sum_rne_bits": encode_bfloat16_rne(exact),
                            "singleton_deletion_irreducible_on_observed_trials": minimal,
                            "cardinality_minimal_proven": False, "budget_exhausted": count >= plan["max_trials_per_case"]})
    body = {"schema_version": 1, "scope": QUERY_REDUCTION_SCOPE, "plan_sha256": plan["plan_sha256"],
            "baseline": baseline, "trials": trials, "final_cases": final_cases,
            "trial_count": len(trials), "candidate_refitted": False,
            "fresh_holdout_validation_established": False, "hardware_semantics_established": False,
            "global_exactness_activation_allowed": False}
    return {**body, "report_sha256": _sha256(body)}


def acquire_query_reduction(plan: dict[str, Any], search: dict[str, Any], diagnosis: dict[str, Any], source_plan: dict[str, Any], source_report: dict[str, Any], persist: Any) -> dict[str, Any]:
    if canonical_json(plan) != canonical_json(build_query_reduction_plan(search, diagnosis, source_plan, source_report)) or not callable(persist):
        raise ValueError("Frozen query reduction plan and journal callback required")
    left, source_cases = _product_vectors()[0]
    shape = source_plan["shapes"][0]
    persist({"plan_sha256": plan["plan_sha256"], "pending": "baseline", "trials": []})
    baseline = _acquire_product_shapes({"shapes": [shape]}, [(left, source_cases)])[0]
    persist({"plan_sha256": plan["plan_sha256"], "baseline": baseline, "pending": None, "trials": []})
    if baseline != source_report["acquisitions"][0]:
        raise ValueError("Source query baseline did not replay exactly; preserved in journal")

    def observe(index: int, right: list[int]) -> dict[str, Any]:
        cases = source_cases[:]
        cases[index] = (cases[index][0], right)
        descriptors = [dict(item) for item in shape["cases"]]
        descriptors[index]["right_bits_sha256"] = _sha256(right)
        actual = _acquire_product_shapes({"shapes": [{**shape, "cases": descriptors}]}, [(left, cases)])[0]
        return {"bits": [row[index] for row in actual["observed_bits"]],
                "kernel_names": actual["kernel_names"], "environment": actual["environment"],
                "background_differences": [{"repetition": repetition, "index": k, "bits": bits}
                                           for repetition, row in enumerate(actual["observed_bits"]) for k, bits in enumerate(row)
                                           if k != index and bits != baseline["observed_bits"][repetition][k]]}

    return _query_reduction_engine(plan, baseline, observe, persist)


def verify_query_reduction(plan: dict[str, Any], search: dict[str, Any], diagnosis: dict[str, Any], source_plan: dict[str, Any], source_report: dict[str, Any], report: dict[str, Any], reexecute: bool = False) -> dict[str, Any]:
    try:
        if canonical_json(plan) != canonical_json(build_query_reduction_plan(search, diagnosis, source_plan, source_report)):
            return {"valid": False, "reason": "Frozen query reduction plan mismatch"}
        if canonical_json(report["baseline"]) != canonical_json(source_report["acquisitions"][0]):
            return {"valid": False, "reason": "Query source baseline mismatch"}
        cursor = 0

        def observe(index: int, right: list[int]) -> dict[str, Any]:
            nonlocal cursor
            trial = report["trials"][cursor]
            cursor += 1
            if trial["case_index"] != index or trial["right_bits_sha256"] != _sha256(right):
                raise ValueError("Query trial ordering or input mismatch")
            return trial["observations"]

        expected = _query_reduction_engine(plan, report["baseline"], observe)
        integrity = canonical_json(report) == canonical_json(expected) and cursor == len(report["trials"])
        replay = acquire_query_reduction(plan, search, diagnosis, source_plan, source_report, lambda pending: None) == report if reexecute and integrity else None
        return {"valid": integrity and (not reexecute or replay is True), "mode": "cuda_replay" if reexecute else "integrity_only",
                "reexecution_exact": replay, "trial_count": expected["trial_count"], "final_cases": expected["final_cases"],
                "fresh_holdout_validation_established": False, "global_exactness_activation_allowed": False}
    except (KeyError, IndexError, TypeError, ValueError, RuntimeError, AttributeError) as error:
        return {"valid": False, "reason": str(error)}


OPERAND_ALIGNMENT_PROFILE = {"alignment_exponent": "max(nonzero operand exponent sums, normalized incoming accumulator exponent)",
                             "precision_bits": 25, "quantization": "toward_zero", "half_order": "lower_then_upper",
                             "carry_conversion": "float32 toward zero after every K8", "final_conversion": "bfloat16 RNE",
                             "operand_domain": "zero and finite normal bfloat16; signed-zero propagation unqualified"}


def diagnose_operand_alignment(search: dict[str, Any], carry_diagnosis: dict[str, Any], source_plan: dict[str, Any], source_report: dict[str, Any], reduction_plan: dict[str, Any], reduction_report: dict[str, Any]) -> dict[str, Any]:
    verified = verify_query_reduction(reduction_plan, search, carry_diagnosis, source_plan, source_report, reduction_report)
    if not verified["valid"]:
        raise ValueError("Operand alignment requires intact source reduction evidence")
    left, source_cases = _product_vectors()[0]
    records = []

    def record(phase: str, index: int, column: int, values: list[int], baseline: int, observed: list[int]) -> None:
        records.append({"phase": phase, "index": index, "source_case_index": column,
                        "right_bits_sha256": _sha256(values), "baseline_predicted_bits": baseline,
                        "predicted_bits": operand_aligned_product_bits(left, values), "observed_bits": observed})

    for index, (_, values) in enumerate(source_cases):
        record("original_query", index, index, values, source_plan["shapes"][0]["cases"][index]["predicted_bits"],
               [row[index] for row in source_report["acquisitions"][0]["observed_bits"]])
    for index, trial in enumerate(reduction_report["trials"]):
        column = trial["case_index"]
        remaining = set(trial["remaining_indices"])
        values = [bits if k in remaining else 0 for k, bits in enumerate(source_cases[column][1])]
        record("deletion_trial", index, column, values, trial["predicted_bits"], trial["observations"]["bits"])
    for index, item in enumerate(reduction_report["final_cases"]):
        values = [0] * 640
        for term in item["terms"]:
            values[term["position"]] = term["right_bits"]
        record("final_reduction", index, item["case_index"], values, item["predicted_bits"], item["observed_bits"])
    summaries = {}
    for phase in ("original_query", "deletion_trial", "final_reduction"):
        selected = [item for item in records if item["phase"] == phase]
        summaries[phase] = {"case_count": len(selected),
                            "baseline_mismatches": [sum(item["baseline_predicted_bits"] != item["observed_bits"][r] for item in selected) for r in range(3)],
                            "revised_mismatches": [sum(item["predicted_bits"] != item["observed_bits"][r] for item in selected) for r in range(3)]}
    body = {"schema_version": 1, "scope": "Development-only evaluation of a separate operand-exponent alignment hypothesis; original, adaptive, and final sets overlap and are not independent validation.",
            "source_product_plan_sha256": source_plan["plan_sha256"], "source_product_report_sha256": source_report["report_sha256"],
            "source_reduction_plan_sha256": reduction_plan["plan_sha256"], "source_reduction_report_sha256": reduction_report["report_sha256"],
            "profile": dict(OPERAND_ALIGNMENT_PROFILE), "records": records, "phase_summaries": summaries,
            "all_development_predictions_match": all(item["predicted_bits"] == bits for item in records for bits in item["observed_bits"]),
            "fresh_holdout_validation_established": False, "hardware_semantics_established": False,
            "global_exactness_activation_allowed": False}
    return {**body, "diagnosis_sha256": _sha256(body)}


OPERAND_ALIGNMENT_SEEDS = tuple(0x92A37B41 + index * 0x01020307 for index in range(8))
OPERAND_ALIGNMENT_SCOPE = "Fresh same-generator query holdout of frozen operand-exponent alignment; eight new left vectors, 128 tested columns per full projection, repeated rows and zero-padded remaining weights; no hardware or global qualification."


def _operand_alignment_vectors() -> list[tuple[list[int], list[tuple[str, list[int]]]]]:
    return [_product_vectors(seed)[0] for seed in OPERAND_ALIGNMENT_SEEDS]


def build_operand_alignment_holdout_plan(diagnosis: dict[str, Any], source_report: dict[str, Any]) -> dict[str, Any]:
    for certificate, field in ((diagnosis, "diagnosis_sha256"), (source_report, "report_sha256")):
        if not isinstance(certificate, dict) or _sha256({k: v for k, v in certificate.items() if k != field}) != certificate.get(field):
            raise ValueError("Alignment source commitment mismatch")
    if canonical_json(diagnosis["profile"]) != canonical_json(OPERAND_ALIGNMENT_PROFILE) or diagnosis["all_development_predictions_match"] is not True:
        raise ValueError("Diagnosis does not support the frozen alignment hypothesis")
    if source_report["report_sha256"] != diagnosis["source_product_report_sha256"]:
        raise ValueError("Alignment diagnosis refers to different source observations")
    development_lefts = [left for left, _ in _product_vectors()]
    development_lefts += [[0x3F80] * 640, [_scale_normal_bfloat_bits(bits, 1) for bits in development_lefts[0]]]
    excluded = {_sha256(left) for left in development_lefts}
    seen, shapes = set(), []
    for seed, (left, vectors) in zip(OPERAND_ALIGNMENT_SEEDS, _operand_alignment_vectors()):
        left_hash = _sha256(left)
        if left_hash in excluded or left_hash in seen:
            raise ValueError("Fresh alignment inputs overlap prior left vectors or each other")
        seen.add(left_hash)
        cases = [{"index": index, "family": family, "right_bits_sha256": _sha256(right),
                  "predicted_bits": operand_aligned_product_bits(left, right)} for index, (family, right) in enumerate(vectors)]
        shapes.append({"role": f"query:seed{seed}", "seed": seed, "input_shape": [30, 640], "weight_shape": [1024, 640],
                       "left_bits_sha256": left_hash, "cases": cases})
    body = {"schema_version": 1, "scope": OPERAND_ALIGNMENT_SCOPE,
            "diagnosis_sha256": diagnosis["diagnosis_sha256"], "source_product_report_sha256": source_report["report_sha256"],
            "source_diagnosis_verification": "integrity_only; recompute development fit with gemma-operand-alignment-diagnosis",
            "profile": dict(OPERAND_ALIGNMENT_PROFILE), "generator": "xorshift32-nonunit-products-v1", "seeds": list(OPERAND_ALIGNMENT_SEEDS),
            "development_left_hashes": sorted(excluded), "inputs_disjoint_from_development_by_left_hash": True,
            "shapes": shapes, "tested_columns_per_shape": 128,
            "predictions_sha256": _sha256([[case["predicted_bits"] for case in shape["cases"]] for shape in shapes]),
            "expected_kernel_names": source_report["acquisitions"][0]["kernel_names"][0],
            "expected_environment": source_report["acquisitions"][0]["environment"],
            "candidate_refitting_allowed": False, "full_vectors_committed": False,
            "global_exactness_activation_allowed": False}
    return {**body, "plan_sha256": _sha256(body)}


def _operand_alignment_holdout_report(plan: dict[str, Any], acquisitions: list[dict[str, Any]]) -> dict[str, Any]:
    body = _product_report(plan, acquisitions)
    body.pop("report_sha256")
    body["scope"] = OPERAND_ALIGNMENT_SCOPE
    body["source_kernel_environment_match"] = all(
        canonical_json(item["environment"]) == canonical_json(plan["expected_environment"]) and
        all(names == plan["expected_kernel_names"] for names in item["kernel_names"]) for item in acquisitions)
    body["candidate_passes_within_declared_scope"] = body["candidate_passes_holdout"] and body["source_kernel_environment_match"]
    return {**body, "report_sha256": _sha256(body)}


def acquire_operand_alignment_holdout(plan: dict[str, Any], diagnosis: dict[str, Any], source_report: dict[str, Any]) -> dict[str, Any]:
    if canonical_json(plan) != canonical_json(build_operand_alignment_holdout_plan(diagnosis, source_report)):
        raise ValueError("Frozen alignment plan mismatch")
    return _operand_alignment_holdout_report(plan, _acquire_product_shapes(plan, _operand_alignment_vectors()))


def verify_operand_alignment_holdout(plan: dict[str, Any], diagnosis: dict[str, Any], source_report: dict[str, Any], report: dict[str, Any], reexecute: bool = False) -> dict[str, Any]:
    try:
        if canonical_json(plan) != canonical_json(build_operand_alignment_holdout_plan(diagnosis, source_report)):
            return {"valid": False, "reason": "Frozen alignment plan mismatch"}
        expected = _operand_alignment_holdout_report(plan, report["acquisitions"])
        integrity = canonical_json(expected) == canonical_json(report)
        replay = canonical_json(acquire_operand_alignment_holdout(plan, diagnosis, source_report)) == canonical_json(report) if reexecute and integrity else None
        return {"valid": integrity and (not reexecute or replay is True), "mode": "cuda_replay" if reexecute else "integrity_only",
                "reexecution_exact": replay, "candidate_passes_holdout": expected["candidate_passes_holdout"],
                "candidate_passes_within_declared_scope": expected["candidate_passes_within_declared_scope"],
                "source_kernel_environment_match": expected["source_kernel_environment_match"],
                "tested_case_count": expected["tested_case_count"], "mismatch_count": expected["mismatch_count"],
                "all_repetitions_identical": all(item["repeated_outputs_identical"] for item in expected["shape_results"]),
                "global_exactness_activation_allowed": False}
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError) as error:
        return {"valid": False, "reason": str(error)}


WIDE_QUERY_SEED = 0x5EA9B731
WIDE_QUERY_ROWS = (0, 7, 15, 29)
WIDE_QUERY_FAMILIES = ("dense_narrow", "dense_wide", "row0_paired_cancellation", "row0_boundary_cancellation")
WIDE_QUERY_SCOPE = "Prospective query-shape holdout with 30 distinct unpaired input rows and all 1024 weight rows active; only output rows 0,7,15,29 compared; no unrestricted or hardware qualification."


def _wide_query_vectors(seed: int = WIDE_QUERY_SEED) -> tuple[list[list[int]], list[tuple[str, list[int]]]]:
    if type(seed) is not int or not 0 < seed <= 0xFFFFFFFF:
        raise ValueError("Wide projection seed must be a nonzero uint32")
    state = seed

    def word() -> int:
        nonlocal state
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        return state & 0xFFFFFFFF

    def value(low: int, span: int) -> int:
        bits = word()
        return ((bits >> 16) & 0x8000) | ((low + bits % span) << 7) | (1 + (bits >> 8) % 127)

    inputs = [[value(125, 5) for _ in range(640)] for _ in range(30)]
    weights = []
    anchor = inputs[0]
    for family in WIDE_QUERY_FAMILIES:
        for index in range(256):
            if family == "dense_narrow":
                right = [value(118, 21) for _ in range(640)]
            elif family == "dense_wide":
                right = [value(90, 71) for _ in range(640)]
            elif family == "row0_paired_cancellation":
                right = []
                for k in range(0, 640, 2):
                    shift = word() % 23 - 6
                    right.extend((_scale_normal_bfloat_bits(anchor[k + 1], shift), _scale_normal_bfloat_bits(anchor[k], shift) ^ 0x8000))
                right[index * 2] = value(120, 25)
            else:
                boundary = 128 * (1 + index % 4)
                right = [0] * 640
                right[boundary - 16:boundary + 16] = [value(120, 25) for _ in range(32)]
                right[boundary - 2] = _scale_normal_bfloat_bits(anchor[boundary], 30)
                right[boundary] = _scale_normal_bfloat_bits(anchor[boundary - 2], 30) ^ 0x8000
            weights.append((family, right))
    return inputs, weights


def build_wide_query_plan(source_plan: dict[str, Any], source_report: dict[str, Any]) -> dict[str, Any]:
    for certificate, field in ((source_plan, "plan_sha256"), (source_report, "report_sha256")):
        if not isinstance(certificate, dict) or _sha256({k: v for k, v in certificate.items() if k != field}) != certificate.get(field):
            raise ValueError("Wide query source commitment mismatch")
    shapes = source_plan.get("shapes")
    if source_plan.get("seeds") != list(OPERAND_ALIGNMENT_SEEDS) or not isinstance(shapes, list) or len(shapes) != 8 or any(
        shape.get("input_shape") != [30, 640] or shape.get("weight_shape") != [1024, 640] or
        canonical_json([case["index"] for case in shape["cases"]]) != canonical_json(list(range(128))) for shape in shapes
    ):
        raise ValueError("Wide query source must cover the complete preceding alignment experiment")
    expected = _operand_alignment_holdout_report(source_plan, source_report["acquisitions"])
    if canonical_json(expected) != canonical_json(source_report) or expected["candidate_passes_within_declared_scope"] is not True:
        raise ValueError("Wide query requires intact passing source alignment evidence")
    if canonical_json(source_plan["profile"]) != canonical_json(OPERAND_ALIGNMENT_PROFILE):
        raise ValueError("Wide query profile differs from frozen operand alignment")
    inputs, weights = _wide_query_vectors()
    input_hashes = [_sha256(row) for row in inputs]
    excluded = set(source_plan["development_left_hashes"]) | {shape["left_bits_sha256"] for shape in source_plan["shapes"]}
    if len(set(input_hashes)) != 30 or excluded.intersection(input_hashes):
        raise ValueError("Wide query input rows repeat or overlap prior input rows")
    if any(all(row[k] == row[k + 1] for k in range(0, 640, 2)) for row in inputs) or any(not any(right) for _, right in weights):
        raise ValueError("Wide query must have unpaired inputs and no zero-only weight rows")
    predictions = [[operand_aligned_product_bits(inputs[row], right) for _, right in weights] for row in WIDE_QUERY_ROWS]
    body = {"schema_version": 1, "scope": WIDE_QUERY_SCOPE, "source_plan_sha256": source_plan["plan_sha256"],
            "source_report_sha256": source_report["report_sha256"],
            "source_verification": "integrity_only; use gemma-operand-alignment-verify for source recomputation or CUDA replay",
            "profile": dict(OPERAND_ALIGNMENT_PROFILE), "generator": "xorshift32-wide-query-v1", "seed": WIDE_QUERY_SEED,
            "input_shape": [30, 640], "weight_shape": [1024, 640], "selected_output_rows": list(WIDE_QUERY_ROWS),
            "input_bits_sha256": _sha256(inputs), "input_row_hashes": input_hashes,
            "weight_bits_sha256": _sha256([right for _, right in weights]),
            "weight_row_hashes": [_sha256(right) for _, right in weights], "column_families": [family for family, _ in weights],
            "predicted_bits": predictions, "predictions_sha256": _sha256(predictions), "tested_case_count": 4096,
            "expected_kernel_names": source_plan["expected_kernel_names"], "expected_environment": source_plan["expected_environment"],
            "input_rows_disjoint_from_prior_left_hashes": True, "complete_output_compared": False,
            "candidate_refitting_allowed": False, "full_vectors_committed": False, "global_exactness_activation_allowed": False}
    return {**body, "plan_sha256": _sha256(body)}


def _wide_query_report(plan: dict[str, Any], observations: list[list[list[int]]], kernels: list[list[str]], environment: dict[str, Any]) -> dict[str, Any]:
    columns = len(plan["column_families"])
    if not isinstance(observations, list) or len(observations) != 3 or any(
        not isinstance(rows, list) or len(rows) != 4 or any(not isinstance(row, list) or len(row) != columns or
        any(type(bits) is not int or not 0 <= bits <= 65535 for bits in row) for row in rows) for rows in observations
    ):
        raise ValueError("Expected three repetitions of four complete output rows")
    if not isinstance(kernels, list) or len(kernels) != 3 or any(not isinstance(names, list) or not names or any(not isinstance(name, str) or not name for name in names) for names in kernels):
        raise ValueError("Missing wide-query kernel observations")
    if not isinstance(environment, dict):
        raise ValueError("Missing wide-query environment")
    mismatches = [{"repetition": repetition, "row": row, "column": column, "family": plan["column_families"][column],
                   "predicted_bits": predicted, "observed_bits": rows[position][column]}
                  for repetition, rows in enumerate(observations) for position, row in enumerate(WIDE_QUERY_ROWS)
                  for column, predicted in enumerate(plan["predicted_bits"][position]) if rows[position][column] != predicted]
    scope_match = canonical_json(environment) == canonical_json(plan["expected_environment"]) and all(names == plan["expected_kernel_names"] for names in kernels)
    summaries = [{"row": row, "matching_cases": [sum(a == b for a, b in zip(rows[position], plan["predicted_bits"][position])) for rows in observations],
                  "matching_cases_by_family": {family: [sum(rows[position][column] == plan["predicted_bits"][position][column] for column, item in enumerate(plan["column_families"]) if item == family) for rows in observations] for family in WIDE_QUERY_FAMILIES}}
                 for position, row in enumerate(WIDE_QUERY_ROWS)]
    body = {"schema_version": 1, "scope": plan.get("scope", WIDE_QUERY_SCOPE), "plan_sha256": plan["plan_sha256"],
            "predictions_sha256": plan["predictions_sha256"], "observed_bits": observations,
            "cuda_kernel_names": kernels, "environment": environment, "row_summaries": summaries,
            "mismatch_count": len(mismatches), "mismatches": mismatches, "tested_case_count": 4 * columns, "repetitions": 3,
            "repeated_outputs_identical": all(rows == observations[0] for rows in observations),
            "source_kernel_environment_match": scope_match, "candidate_passes_holdout": not mismatches,
            "candidate_passes_within_declared_scope": not mismatches and scope_match,
            "complete_output_compared": False, "candidate_refitted": False, "hardware_semantics_established": False,
            "global_exactness_activation_allowed": False}
    return {**body, "report_sha256": _sha256(body)}


def acquire_wide_query_holdout(plan: dict[str, Any], source_plan: dict[str, Any], source_report: dict[str, Any]) -> dict[str, Any]:
    if canonical_json(plan) != canonical_json(build_wide_query_plan(source_plan, source_report)):
        raise ValueError("Frozen wide-query plan mismatch")
    import torch
    from .gemma_reduction_backend import _profile_call

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the wide query holdout")
    input_bits, cases = _wide_query_vectors()
    weight_bits = [right for _, right in cases]
    if _sha256(input_bits) != plan["input_bits_sha256"] or _sha256(weight_bits) != plan["weight_bits_sha256"]:
        raise ValueError("Regenerated wide-query matrix commitment mismatch")
    with torch.no_grad():
        inputs = torch.tensor(input_bits, dtype=torch.uint16).view(torch.bfloat16).cuda()
        weights = torch.tensor(weight_bits, dtype=torch.uint16).view(torch.bfloat16).cuda()
        observations, kernels = [], []
        for _ in range(3):
            result, names = _profile_call(lambda: torch.nn.functional.linear(inputs, weights))
            observations.append(result.cpu()[list(WIDE_QUERY_ROWS)].view(torch.uint16).tolist())
            kernels.append(names)
    environment = {"torch": torch.__version__, "cuda": torch.version.cuda,
                   "device": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
                   "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                   "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
    return _wide_query_report(plan, observations, kernels, environment)


def verify_wide_query_holdout(plan: dict[str, Any], source_plan: dict[str, Any], source_report: dict[str, Any], report: dict[str, Any], reexecute: bool = False) -> dict[str, Any]:
    try:
        if canonical_json(plan) != canonical_json(build_wide_query_plan(source_plan, source_report)):
            return {"valid": False, "reason": "Frozen wide-query plan mismatch"}
        expected = _wide_query_report(plan, report["observed_bits"], report["cuda_kernel_names"], report["environment"])
        integrity = canonical_json(report) == canonical_json(expected)
        replay = canonical_json(acquire_wide_query_holdout(plan, source_plan, source_report)) == canonical_json(report) if reexecute and integrity else None
        return {"valid": integrity and (not reexecute or replay is True), "mode": "cuda_replay" if reexecute else "integrity_only",
                "reexecution_exact": replay, "candidate_passes_holdout": expected["candidate_passes_holdout"],
                "candidate_passes_within_declared_scope": expected["candidate_passes_within_declared_scope"],
                "row_summaries": expected["row_summaries"], "mismatch_count": expected["mismatch_count"],
                "repeated_outputs_identical": expected["repeated_outputs_identical"], "tested_case_count": 4096,
                "complete_output_compared": False, "global_exactness_activation_allowed": False}
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError) as error:
        return {"valid": False, "reason": str(error)}


SPLIT_K_CHUNKS = tuple(range(64, 641, 64))
SPLIT_K_PARTIAL_FORMATS = ("bfloat16_rne", "float32")
SPLIT_K_REDUCTIONS = ("exact", "sequential_float32_rne", "pairwise_float32_rne", "sequential_bfloat16_rne")


def _merge_split_partials(partials: list[Fraction], mode: str) -> int:
    if not partials or mode not in SPLIT_K_REDUCTIONS:
        raise ValueError("A nonempty partial list and supported reduction are required")
    if mode == "exact":
        return encode_bfloat16_rne(sum(partials, Fraction(0)))
    if mode == "pairwise_float32_rne":
        values = list(partials)
        while len(values) > 1:
            values = [_float32_carry(values[k] + values[k + 1], "nearest_even") if k + 1 < len(values) else values[k]
                      for k in range(0, len(values), 2)]
        return encode_bfloat16_rne(values[0])
    accumulator = Fraction(0)
    for value in partials:
        total = accumulator + value
        accumulator = decode_finite_bfloat16(encode_bfloat16_rne(total))[0] if mode == "sequential_bfloat16_rne" else _float32_carry(total, "nearest_even")
    return encode_bfloat16_rne(accumulator)


def _split_partitions(left: list[int], right: list[int], chunk: int) -> list[Fraction]:
    if type(chunk) is not int or chunk not in SPLIT_K_CHUNKS or len(left) != 640 or len(right) != 640:
        raise ValueError("Split-K comparison requires K640 and a declared contiguous partition width")
    return [_operand_aligned_accumulator(left[start:start + chunk], right[start:start + chunk]) for start in range(0, 640, chunk)]


def split_k_candidate_bits(left: list[int], right: list[int], chunk: int, partial_format: str, reduction: str) -> int:
    if partial_format not in SPLIT_K_PARTIAL_FORMATS:
        raise ValueError("Unsupported split-K intermediate format")
    partials = _split_partitions(left, right, chunk)
    if partial_format == "bfloat16_rne":
        partials = [decode_finite_bfloat16(encode_bfloat16_rne(value))[0] for value in partials]
    return _merge_split_partials(partials, reduction)


def diagnose_split_k(search: dict[str, Any], carry_diagnosis: dict[str, Any], product_plan: dict[str, Any], product_report: dict[str, Any], matched_plan: dict[str, Any], matched_report: dict[str, Any]) -> dict[str, Any]:
    for certificate, field in ((product_plan, "plan_sha256"), (product_report, "report_sha256"), (matched_plan, "plan_sha256"), (matched_report, "report_sha256")):
        if not isinstance(certificate, dict) or _sha256({k: v for k, v in certificate.items() if k != field}) != certificate.get(field):
            raise ValueError("Split-K source commitment mismatch")
    if not verify_product_holdout(product_plan, search, carry_diagnosis, product_report)["valid"] or not verify_matched_products(matched_plan, search, carry_diagnosis, matched_report)["valid"]:
        raise ValueError("Split-K comparison requires intact source evidence")
    cohorts = (("original_key_value", _product_vectors()[1], product_report["acquisitions"][1]),
               ("matched_key_value", _matched_product_vectors()[2], matched_report["acquisitions"][2]))
    records, operands, provenance = [], [], []
    for name, (left, vectors), acquired in cohorts:
        if not all(any("splitKreduce" in kernel for kernel in names) for names in acquired["kernel_names"]):
            raise ValueError("Source lacks a recorded split-K reduction launch")
        provenance.append({"cohort": name, "kernel_names": acquired["kernel_names"], "environment": acquired["environment"]})
        for index, (family, right) in enumerate(vectors):
            records.append({"cohort": name, "index": index, "family": family, "left_bits_sha256": _sha256(left),
                            "right_bits_sha256": _sha256(right), "observed_bits": [row[index] for row in acquired["observed_bits"]]})
            operands.append((left, right))
    candidates = []
    for chunk in SPLIT_K_CHUNKS:
        partitions = [_split_partitions(left, right, chunk) for left, right in operands]
        for partial_format in SPLIT_K_PARTIAL_FORMATS:
            converted = [[decode_finite_bfloat16(encode_bfloat16_rne(value))[0] for value in row] for row in partitions] if partial_format == "bfloat16_rne" else partitions
            for reduction in SPLIT_K_REDUCTIONS:
                predictions = [_merge_split_partials(values, reduction) for values in converted]
                mismatches = [[index for index, record in enumerate(records) if record["observed_bits"][repetition] != predictions[index]] for repetition in range(3)]
                candidates.append({"chunk_size": chunk, "partial_format": partial_format, "reduction": reduction,
                                   "partitions": [[start, min(start + chunk, 640)] for start in range(0, 640, chunk)],
                                   "predicted_bits": predictions, "predictions_sha256": _sha256(predictions), "mismatch_count": sum(len(indices) for indices in mismatches),
                                   "mismatches_by_cohort": {name: [sum(predictions[k] != record["observed_bits"][r] for k, record in enumerate(records) if record["cohort"] == name) for r in range(3)] for name, _, _ in cohorts},
                                   "mismatch_record_indices_by_repetition": mismatches})
    def summary(item: dict[str, Any]) -> dict[str, Any]:
        return {key: item[key] for key in ("chunk_size", "partial_format", "reduction", "mismatch_count", "mismatches_by_cohort")}

    minimum = min(item["mismatch_count"] for item in candidates)
    matching = [summary(item) for item in candidates if item["mismatch_count"] == 0]
    body = {"schema_version": 1, "scope": "Development-only split-K comparison on original and matched key/value controlled projections; candidate boundaries and intermediate values are hypothetical, not observed.",
            "source_product_plan_sha256": product_plan["plan_sha256"], "source_product_report_sha256": product_report["report_sha256"],
            "source_matched_plan_sha256": matched_plan["plan_sha256"], "source_matched_report_sha256": matched_report["report_sha256"],
            "input_shape": [30, 640], "weight_shape": [256, 640], "tested_columns_per_cohort": 128,
            "partial_accumulation_profile": dict(OPERAND_ALIGNMENT_PROFILE),
            "search_space": {"chunk_sizes": list(SPLIT_K_CHUNKS), "partial_formats": list(SPLIT_K_PARTIAL_FORMATS), "reductions": list(SPLIT_K_REDUCTIONS)},
            "records": records, "provenance": provenance, "candidates": candidates, "candidate_count": len(candidates),
            "minimum_mismatch_count": minimum, "best_candidates": [summary(item) for item in candidates if item["mismatch_count"] == minimum],
            "all_matching_candidates": matching, "unique_all_matching_candidate_in_search_space": len(matching) == 1,
            "split_boundaries_observed": False, "intermediate_values_observed": False,
            "fresh_holdout_validation_established": False, "hardware_semantics_established": False,
            "complete_numeric_transition_established": False, "global_exactness_activation_allowed": False}
    return {**body, "diagnosis_sha256": _sha256(body)}


SPLIT_MERGE_SEED = 0x71C4E92B
SPLIT_MERGE_FINALISTS = ("exact", "sequential_float32_rne", "pairwise_float32_rne")
SPLIT_MERGE_SCOPE = "Prospective discrimination among three frozen K64/bfloat16-partial merge candidates; candidate-selected inputs, no observed-output fitting, no unrestricted split-K or hardware qualification."


def _split_merge_vectors() -> list[tuple[list[int], list[int]]]:
    state = SPLIT_MERGE_SEED

    def word() -> int:
        nonlocal state
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        return state & 0xFFFFFFFF

    vectors, seen = [], set()
    for _ in range(100000):
        positions = []
        while len(positions) < 4:
            position = word() % 10
            if position not in positions:
                positions.append(position)
        small_exponent = word() % 17 - 8
        large_exponent = small_exponent + 25 + word() % 16
        large = ((large_exponent + 127) << 7) | ((word() & 1) << 15)
        small = ((small_exponent + 127) << 7) | ((word() & 1) << 15)
        partials = [0] * 10
        for position, bits in zip(positions, (large, large ^ 0x8000, small, small)):
            partials[position] = bits
        values = [decode_finite_bfloat16(bits)[0] for bits in partials]
        if len({_merge_split_partials(values, mode) for mode in SPLIT_MERGE_FINALISTS}) != 3:
            continue
        weights = [0] * 640
        for position in positions:
            weights[position * 64 + word() % 64] = _scale_normal_bfloat_bits(partials[position], -1)
        commitment = _sha256(weights)
        if commitment not in seen:
            seen.add(commitment)
            vectors.append((partials, weights))
        if len(vectors) == 256:
            return vectors
    raise ValueError("Could not construct 256 distinct discriminating split-K cases")


def build_split_merge_plan(diagnosis: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(diagnosis, dict) or _sha256({k: v for k, v in diagnosis.items() if k != "diagnosis_sha256"}) != diagnosis.get("diagnosis_sha256"):
        raise ValueError("Split-K diagnosis commitment mismatch")
    finalists = [{key: item[key] for key in ("chunk_size", "partial_format", "reduction")} for item in diagnosis["all_matching_candidates"]]
    expected = [{"chunk_size": 64, "partial_format": "bfloat16_rne", "reduction": mode} for mode in SPLIT_MERGE_FINALISTS]
    if canonical_json(finalists) != canonical_json(expected) or canonical_json(diagnosis["partial_accumulation_profile"]) != canonical_json(OPERAND_ALIGNMENT_PROFILE):
        raise ValueError("Diagnosis does not support the frozen split-K finalist set")
    left = [0x4000] * 640
    if _sha256(left) in {record["left_bits_sha256"] for record in diagnosis["records"]}:
        raise ValueError("Split-K holdout input overlaps development left vectors")
    cases = []
    for index, (partial_bits, right) in enumerate(_split_merge_vectors()):
        partials = [decode_finite_bfloat16(bits)[0] for bits in partial_bits]
        if _split_partitions(left, right, 64) != partials:
            raise ValueError("Constructed exact products differ from the intended partials")
        predictions = {mode: _merge_split_partials(partials, mode) for mode in SPLIT_MERGE_FINALISTS}
        cases.append({"index": index, "right_bits_sha256": _sha256(right), "hypothetical_partial_bits": partial_bits, "predicted_bits": predictions})
    body = {"schema_version": 1, "scope": SPLIT_MERGE_SCOPE, "diagnosis_sha256": diagnosis["diagnosis_sha256"],
            "source_verification": "integrity_only; use gemma-split-k-diagnosis for source recomputation",
            "generator": "xorshift32-discriminating-split-merges-v1", "seed": SPLIT_MERGE_SEED,
            "input_shape": [30, 640], "weight_shape": [256, 640], "input_value_bits": 0x4000,
            "left_bits_sha256": _sha256(left), "input_pairs_disjoint_from_development": True,
            "candidate_selection": "retain inputs only when all three frozen candidates predict different bits; no CUDA selection",
            "finalists": expected, "partial_accumulation_profile": dict(OPERAND_ALIGNMENT_PROFILE),
            "cases": cases, "predictions_sha256": _sha256([case["predicted_bits"] for case in cases]),
            "expected_kernel_names": diagnosis["provenance"][0]["kernel_names"][0],
            "expected_environment": diagnosis["provenance"][0]["environment"],
            "candidate_refitting_allowed": False, "hypothetical_partials_observed": False,
            "global_exactness_activation_allowed": False}
    return {**body, "plan_sha256": _sha256(body)}


def _split_merge_report(plan: dict[str, Any], observed: list[list[int]], kernels: list[list[str]], environment: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(observed, list) or len(observed) != 3 or any(not isinstance(row, list) or len(row) != 256 or any(type(bits) is not int or not 0 <= bits <= 65535 for bits in row) for row in observed):
        raise ValueError("Expected three repetitions of 256 split-K outputs")
    if not isinstance(kernels, list) or len(kernels) != 3 or any(not isinstance(names, list) or not names or any(not isinstance(name, str) or not name for name in names) for names in kernels) or not isinstance(environment, dict):
        raise ValueError("Missing split-K kernel or environment observations")
    candidates = []
    for mode in SPLIT_MERGE_FINALISTS:
        mismatches = [[index for index, bits in enumerate(row) if bits != plan["cases"][index]["predicted_bits"][mode]] for row in observed]
        candidates.append({"reduction": mode, "mismatch_count": sum(len(indices) for indices in mismatches), "mismatch_indices_by_repetition": mismatches})
    survivors = [item["reduction"] for item in candidates if item["mismatch_count"] == 0]
    scope_match = canonical_json(environment) == canonical_json(plan["expected_environment"]) and all(names == plan["expected_kernel_names"] for names in kernels)
    body = {"schema_version": 1, "scope": SPLIT_MERGE_SCOPE, "plan_sha256": plan["plan_sha256"],
            "predictions_sha256": plan["predictions_sha256"], "observed_bits": observed, "cuda_kernel_names": kernels,
            "environment": environment, "candidates": candidates, "surviving_candidates": survivors,
            "unique_surviving_candidate_in_frozen_set": len(survivors) == 1, "source_kernel_environment_match": scope_match,
            "any_candidate_passes_within_declared_scope": bool(survivors) and scope_match,
            "repeated_outputs_identical": all(row == observed[0] for row in observed), "tested_case_count": 256,
            "candidate_refitted": False, "hypothetical_partials_observed": False, "split_boundaries_observed": False,
            "hardware_semantics_established": False, "global_exactness_activation_allowed": False}
    return {**body, "report_sha256": _sha256(body)}


def acquire_split_merge_holdout(plan: dict[str, Any], diagnosis: dict[str, Any]) -> dict[str, Any]:
    if canonical_json(plan) != canonical_json(build_split_merge_plan(diagnosis)):
        raise ValueError("Frozen split-K merge plan mismatch")
    import torch
    from .gemma_reduction_backend import _profile_call

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for split-K acquisition")
    vectors = [right for _, right in _split_merge_vectors()]
    left = [plan["input_value_bits"]] * 640
    if _sha256(left) != plan["left_bits_sha256"] or len(vectors) != len(plan["cases"]) or any(_sha256(right) != case["right_bits_sha256"] for right, case in zip(vectors, plan["cases"])):
        raise ValueError("Regenerated split-K operand commitment mismatch")
    with torch.no_grad():
        inputs = torch.tensor(left, dtype=torch.uint16).view(torch.bfloat16).cuda().reshape(1, 640).expand(30, 640).contiguous()
        weights = torch.tensor(vectors, dtype=torch.uint16).view(torch.bfloat16).cuda()
        observed, kernels = [], []
        for _ in range(3):
            result, names = _profile_call(lambda: torch.nn.functional.linear(inputs, weights))
            bits = result.cpu().view(torch.uint16)
            if not all(torch.equal(bits[0], row) for row in bits):
                raise ValueError("Repeated split-K input rows produced different output bits")
            observed.append(bits[0].tolist())
            kernels.append(names)
    environment = {"torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(),
                   "capability": list(torch.cuda.get_device_capability()),
                   "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                   "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
    return _split_merge_report(plan, observed, kernels, environment)


def verify_split_merge_holdout(plan: dict[str, Any], diagnosis: dict[str, Any], report: dict[str, Any], reexecute: bool = False) -> dict[str, Any]:
    try:
        if canonical_json(plan) != canonical_json(build_split_merge_plan(diagnosis)):
            return {"valid": False, "reason": "Frozen split-K merge plan mismatch"}
        expected = _split_merge_report(plan, report["observed_bits"], report["cuda_kernel_names"], report["environment"])
        integrity = canonical_json(report) == canonical_json(expected)
        replay = canonical_json(acquire_split_merge_holdout(plan, diagnosis)) == canonical_json(report) if reexecute and integrity else None
        return {"valid": integrity and (not reexecute or replay is True), "mode": "cuda_replay" if reexecute else "integrity_only",
                "reexecution_exact": replay, "surviving_candidates": expected["surviving_candidates"],
                "candidate_mismatch_counts": {item["reduction"]: item["mismatch_count"] for item in expected["candidates"]},
                "any_candidate_passes_within_declared_scope": expected["any_candidate_passes_within_declared_scope"],
                "repeated_outputs_identical": expected["repeated_outputs_identical"], "source_kernel_environment_match": expected["source_kernel_environment_match"],
                "global_exactness_activation_allowed": False}
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError) as error:
        return {"valid": False, "reason": str(error)}


DENSE_SPLIT_SEED = 0xC957246B
DENSE_SPLIT_PROFILE = (64, "bfloat16_rne", "sequential_float32_rne")
DENSE_SPLIT_COLUMNS = tuple(family * 256 + index * 4 + index % 4 for family in range(4) for index in range(64))
DENSE_SPLIT_SCOPE = "Fresh composed split-K key/value holdout with dense and row0-anchored cancellation families, 30 distinct non-unit input rows, all 256 weight rows active; only output rows 0,7,15,29 compared; no hardware or global qualification."


def _dense_split_vectors() -> tuple[list[list[int]], list[tuple[str, list[int]]]]:
    inputs, weights = _wide_query_vectors(DENSE_SPLIT_SEED)
    return inputs, [weights[column] for column in DENSE_SPLIT_COLUMNS]


def build_dense_split_plan(diagnosis: dict[str, Any], merge_plan: dict[str, Any], merge_report: dict[str, Any]) -> dict[str, Any]:
    verified = verify_split_merge_holdout(merge_plan, diagnosis, merge_report)
    if not verified["valid"] or not verified["any_candidate_passes_within_declared_scope"] or verified["surviving_candidates"] != [DENSE_SPLIT_PROFILE[2]]:
        raise ValueError("Dense split-K requires the verified frozen sequential-merge survivor")
    inputs, weights = _dense_split_vectors()
    original_lefts = [left for left, _ in _product_vectors()]
    prior_rows = _wide_query_vectors()[0] + [left for left, _ in _operand_alignment_vectors()] + original_lefts
    prior_rows.append([_scale_normal_bfloat_bits(bits, 1) for bits in original_lefts[0]])
    prior_hashes = {_sha256(row) for row in prior_rows} | {record["left_bits_sha256"] for record in diagnosis["records"]}
    prior_hashes.update((_sha256([0x3F80] * 640), _sha256([0x4000] * 640)))
    input_hashes = [_sha256(row) for row in inputs]
    if len(set(input_hashes)) != 30 or prior_hashes.intersection(input_hashes):
        raise ValueError("Dense split-K rows repeat or overlap prior inputs")
    if len(weights) != 256 or any(not any(right) for _, right in weights):
        raise ValueError("Dense split-K requires 256 active weight rows")
    predicted = [[split_k_candidate_bits(inputs[row], right, *DENSE_SPLIT_PROFILE) for _, right in weights] for row in WIDE_QUERY_ROWS]
    body = {"schema_version": 1, "scope": DENSE_SPLIT_SCOPE, "diagnosis_sha256": diagnosis["diagnosis_sha256"],
            "source_merge_plan_sha256": merge_plan["plan_sha256"], "source_merge_report_sha256": merge_report["report_sha256"],
            "source_verification": "software replay of merge evidence; diagnosis is integrity-only unless separately recomputed",
            "partial_accumulation_profile": dict(OPERAND_ALIGNMENT_PROFILE),
            "split_profile": {"chunk_size": DENSE_SPLIT_PROFILE[0], "partial_format": DENSE_SPLIT_PROFILE[1], "reduction": DENSE_SPLIT_PROFILE[2]},
            "generator": "xorshift32-wide-query-v1-stratified-columns", "seed": DENSE_SPLIT_SEED,
            "selected_generator_columns": list(DENSE_SPLIT_COLUMNS), "selected_output_rows": list(WIDE_QUERY_ROWS),
            "input_shape": [30, 640], "weight_shape": [256, 640], "input_bits_sha256": _sha256(inputs),
            "input_row_hashes": input_hashes, "excluded_input_row_hashes": sorted(prior_hashes),
            "weight_bits_sha256": _sha256([right for _, right in weights]), "weight_row_hashes": [_sha256(right) for _, right in weights],
            "column_families": [family for family, _ in weights], "predicted_bits": predicted, "predictions_sha256": _sha256(predicted),
            "tested_case_count": 1024, "expected_kernel_names": merge_plan["expected_kernel_names"], "expected_environment": merge_plan["expected_environment"],
            "input_rows_disjoint_from_prior_left_hashes": True, "candidate_refitting_allowed": False,
            "complete_output_compared": False, "full_vectors_committed": False, "global_exactness_activation_allowed": False}
    return {**body, "plan_sha256": _sha256(body)}


def _dense_split_report(plan: dict[str, Any], observed: list[list[list[int]]], kernels: list[list[str]], environment: dict[str, Any]) -> dict[str, Any]:
    body = _wide_query_report(plan, observed, kernels, environment)
    body.pop("report_sha256")
    body["split_boundaries_observed"] = False
    body["intermediate_values_observed"] = False
    return {**body, "report_sha256": _sha256(body)}


def acquire_dense_split_holdout(plan: dict[str, Any], diagnosis: dict[str, Any], merge_plan: dict[str, Any], merge_report: dict[str, Any]) -> dict[str, Any]:
    if canonical_json(plan) != canonical_json(build_dense_split_plan(diagnosis, merge_plan, merge_report)):
        raise ValueError("Frozen dense split-K plan mismatch")
    import torch
    from .gemma_reduction_backend import _profile_call

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for dense split-K acquisition")
    inputs, cases = _dense_split_vectors()
    weights = [right for _, right in cases]
    if _sha256(inputs) != plan["input_bits_sha256"] or _sha256(weights) != plan["weight_bits_sha256"]:
        raise ValueError("Regenerated dense split-K operand commitment mismatch")
    with torch.no_grad():
        left = torch.tensor(inputs, dtype=torch.uint16).view(torch.bfloat16).cuda()
        right = torch.tensor(weights, dtype=torch.uint16).view(torch.bfloat16).cuda()
        observed, kernels = [], []
        for _ in range(3):
            output, names = _profile_call(lambda: torch.nn.functional.linear(left, right))
            observed.append(output.cpu()[list(WIDE_QUERY_ROWS)].view(torch.uint16).tolist())
            kernels.append(names)
    environment = {"torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(),
                   "capability": list(torch.cuda.get_device_capability()),
                   "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                   "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
    return _dense_split_report(plan, observed, kernels, environment)


def verify_dense_split_holdout(plan: dict[str, Any], diagnosis: dict[str, Any], merge_plan: dict[str, Any], merge_report: dict[str, Any], report: dict[str, Any], reexecute: bool = False) -> dict[str, Any]:
    try:
        if canonical_json(plan) != canonical_json(build_dense_split_plan(diagnosis, merge_plan, merge_report)):
            return {"valid": False, "reason": "Frozen dense split-K plan mismatch"}
        expected = _dense_split_report(plan, report["observed_bits"], report["cuda_kernel_names"], report["environment"])
        integrity = canonical_json(report) == canonical_json(expected)
        replay = canonical_json(acquire_dense_split_holdout(plan, diagnosis, merge_plan, merge_report)) == canonical_json(report) if reexecute and integrity else None
        return {"valid": integrity and (not reexecute or replay is True), "mode": "cuda_replay" if reexecute else "integrity_only",
                "reexecution_exact": replay, "candidate_passes_within_declared_scope": expected["candidate_passes_within_declared_scope"],
                "source_kernel_environment_match": expected["source_kernel_environment_match"], "mismatch_count": expected["mismatch_count"],
                "row_summaries": expected["row_summaries"], "tested_case_count": expected["tested_case_count"],
                "repeated_outputs_identical": expected["repeated_outputs_identical"], "complete_output_compared": False,
                "global_exactness_activation_allowed": False}
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError) as error:
        return {"valid": False, "reason": str(error)}


def acquire_composition_holdout(plan: dict[str, Any], search: dict[str, Any]) -> dict[str, Any]:
    if plan != build_composition_holdout_plan(search):
        raise ValueError("Composition plan differs from frozen specification")
    vectors = [values for _, values in _composition_vectors()]
    if len(vectors) != len(plan["cases"]) or any(
        _sha256(values) != case["input_bits_sha256"]
        for values, case in zip(vectors, plan["cases"])
    ):
        raise ValueError("Regenerated composition vectors do not match committed hashes")
    observed, kernels, environment = _acquire_holdout_vectors(vectors)
    return _holdout_report(plan, observed, kernels, environment, COMPOSITION_FAMILIES, COMPOSITION_SCOPE)


def verify_composition_holdout(plan: dict[str, Any], search: dict[str, Any], report: dict[str, Any],
                               reexecute: bool = False) -> dict[str, Any]:
    try:
        if plan != build_composition_holdout_plan(search):
            return {"valid": False, "reason": "Frozen composition plan mismatch"}
        expected = _holdout_report(plan, report["observed_bits"], report["cuda_kernel_names"], report["environment"], COMPOSITION_FAMILIES, COMPOSITION_SCOPE)
        integrity = report == expected
        replay = acquire_composition_holdout(plan, search) == report if reexecute and integrity else None
        return {
            "valid": integrity and (not reexecute or replay is True),
            "mode": "cuda_replay" if reexecute else "integrity_only",
            "reexecution_exact": replay,
            "candidate_passes_holdout": expected["candidate_passes_holdout"],
            "mismatch_count": expected["mismatch_count"],
            "matching_cases_by_family": expected["matching_cases_by_family"],
            "repeated_outputs_identical": expected["repeated_outputs_identical"],
            "kernel_names_stable": expected["kernel_names_stable"],
            "global_exactness_activation_allowed": False,
        }
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError) as error:
        return {"valid": False, "reason": str(error)}
