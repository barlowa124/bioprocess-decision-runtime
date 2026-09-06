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
