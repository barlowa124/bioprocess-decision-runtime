from __future__ import annotations

from fractions import Fraction
import hashlib
import platform
import struct
from typing import Any

from .gemma_float_semantics import (
    decode_finite_bfloat16,
    encode_bfloat16_rne,
)
from .serialization import canonical_json

try:
    import torch
    from torch.nn import functional
except ImportError:
    torch = None
    functional = None


FLOAT32_SIGN_MASK = 0x80000000
FLOAT32_EXPONENT_MASK = 0x7F800000
FLOAT32_FRACTION_MASK = 0x007FFFFF
FLOAT32_INFINITY = 0x7F800000


def _require_torch() -> None:
    if torch is None or functional is None:
        raise RuntimeError("Reduction conformance capture requires PyTorch")


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _scale_power_of_two(value: Fraction, exponent: int) -> Fraction:
    if exponent >= 0:
        return Fraction(value.numerator << exponent, value.denominator)
    return Fraction(value.numerator, value.denominator << -exponent)


def _round_nearest_even(value: Fraction) -> int:
    quotient, remainder = divmod(value.numerator, value.denominator)
    comparison = remainder * 2 - value.denominator
    if comparison > 0 or (comparison == 0 and quotient % 2 == 1):
        return quotient + 1
    return quotient


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


def decode_finite_float32(bits: int) -> tuple[Fraction, bool]:
    if not isinstance(bits, int) or isinstance(bits, bool) or not 0 <= bits <= 0xFFFFFFFF:
        raise ValueError("Float32 bits must be an unsigned 32-bit integer")
    exponent = (bits & FLOAT32_EXPONENT_MASK) >> 23
    fraction = bits & FLOAT32_FRACTION_MASK
    if exponent == 0xFF:
        raise ValueError("NaN and infinity are outside finite float32 semantics")
    negative = bool(bits & FLOAT32_SIGN_MASK)
    if exponent == 0:
        magnitude = _scale_power_of_two(Fraction(fraction), -149)
    else:
        magnitude = _scale_power_of_two(Fraction((1 << 23) + fraction), exponent - 150)
    return (-magnitude if negative else magnitude), negative and magnitude == 0


def encode_float32_rne(value: Fraction, negative_zero: bool = False) -> int:
    if not isinstance(value, Fraction):
        value = Fraction(value)
    negative = value < 0 or (value == 0 and negative_zero)
    magnitude = abs(value)
    sign = FLOAT32_SIGN_MASK if negative else 0
    if magnitude == 0:
        return sign
    exponent = _floor_log2(magnitude)
    if exponent >= -126:
        significand = _round_nearest_even(
            _scale_power_of_two(magnitude, 23 - exponent)
        )
        if significand == 1 << 24:
            significand = 1 << 23
            exponent += 1
        if exponent > 127:
            return sign | FLOAT32_INFINITY
        return sign | ((exponent + 127) << 23) | (significand - (1 << 23))
    subnormal = _round_nearest_even(_scale_power_of_two(magnitude, 149))
    if subnormal == 0:
        return sign
    if subnormal >= 1 << 23:
        return sign | 0x00800000
    return sign | subnormal


def _float32_add_bits(left_bits: int, right_bits: int) -> int:
    left, left_negative_zero = decode_finite_float32(left_bits)
    right, right_negative_zero = decode_finite_float32(right_bits)
    result = left + right
    return encode_float32_rne(
        result,
        negative_zero=result == 0 and left_negative_zero and right_negative_zero,
    )


def _bfloat16_values(bits: list[int]) -> list[Fraction]:
    return [decode_finite_bfloat16(value)[0] for value in bits]


def _exact_products(left: list[int], right: list[int]) -> list[Fraction]:
    return [
        left_value * right_value
        for left_value, right_value in zip(
            _bfloat16_values(left), _bfloat16_values(right)
        )
    ]


def exact_sum_then_bfloat16(left: list[int], right: list[int]) -> int:
    return encode_bfloat16_rne(sum(_exact_products(left, right), Fraction(0)))


def sequential_float32_then_bfloat16(left: list[int], right: list[int]) -> int:
    accumulator_bits = 0
    for product in _exact_products(left, right):
        product_bits = encode_float32_rne(product)
        accumulator_bits = _float32_add_bits(accumulator_bits, product_bits)
    accumulator, negative_zero = decode_finite_float32(accumulator_bits)
    return encode_bfloat16_rne(accumulator, negative_zero)


def pairwise_float32_then_bfloat16(left: list[int], right: list[int]) -> int:
    values = [encode_float32_rne(product) for product in _exact_products(left, right)]
    while len(values) > 1:
        values = [
            _float32_add_bits(values[index], values[index + 1])
            if index + 1 < len(values)
            else values[index]
            for index in range(0, len(values), 2)
        ]
    result, negative_zero = decode_finite_float32(values[0])
    return encode_bfloat16_rne(result, negative_zero)


def block_float32_then_bfloat16(
    left: list[int], right: list[int], block_size: int = 16
) -> int:
    products = _exact_products(left, right)
    block_results = []
    for start in range(0, len(products), block_size):
        accumulator_bits = 0
        for product in products[start : start + block_size]:
            accumulator_bits = _float32_add_bits(
                accumulator_bits, encode_float32_rne(product)
            )
        block_results.append(accumulator_bits)
    accumulator_bits = 0
    for block_result in block_results:
        accumulator_bits = _float32_add_bits(accumulator_bits, block_result)
    result, negative_zero = decode_finite_float32(accumulator_bits)
    return encode_bfloat16_rne(result, negative_zero)


def sequential_bfloat16(left: list[int], right: list[int]) -> int:
    accumulator = 0
    for left_bits, right_bits in zip(left, right):
        product = encode_bfloat16_rne(
            decode_finite_bfloat16(left_bits)[0]
            * decode_finite_bfloat16(right_bits)[0]
        )
        accumulator = encode_bfloat16_rne(
            decode_finite_bfloat16(accumulator)[0]
            + decode_finite_bfloat16(product)[0]
        )
    return accumulator


def _candidate_results(left: list[int], right: list[int]) -> dict[str, str]:
    functions = {
        "exact_sum_then_bfloat16": exact_sum_then_bfloat16,
        "sequential_float32_then_bfloat16": sequential_float32_then_bfloat16,
        "pairwise_float32_then_bfloat16": pairwise_float32_then_bfloat16,
        "block16_float32_then_bfloat16": block_float32_then_bfloat16,
        "sequential_bfloat16": sequential_bfloat16,
    }
    return {
        name: f"0x{function(left, right):04x}"
        for name, function in functions.items()
    }


def _torch_bfloat16_vector(bits: list[int], device: Any) -> Any:
    return torch.tensor(bits, dtype=torch.uint16).view(torch.bfloat16).to(device)


def _result_bits(value: Any) -> str:
    return f"0x{int(value.detach().cpu().reshape(1).view(torch.uint16).item()):04x}"


def _vectors() -> list[dict[str, Any]]:
    vectors = []
    for length in (4, 8, 16, 32, 64, 128, 256, 640, 2048):
        repetitions = (length + 3) // 4
        cancellation = ([0x4E80, 0x3F80, 0xCE80, 0x3F80] * repetitions)[:length]
        grouped = ([0x4E80, 0xCE80, 0x3F80, 0x3F80] * repetitions)[:length]
        state = 0x13579BDF ^ length
        pseudo = []
        for _ in range(length):
            state = (1664525 * state + 1013904223) & 0xFFFFFFFF
            sign = 0x8000 if state & 1 else 0
            exponent = 120 + ((state >> 8) % 15)
            fraction = (state >> 16) & 0x7F
            pseudo.append(sign | (exponent << 7) | fraction)
        vectors.extend(
            (
                {
                    "case": f"cancellation_{length}",
                    "left": cancellation,
                    "right": [0x3F80] * length,
                },
                {
                    "case": f"grouped_{length}",
                    "left": grouped,
                    "right": [0x3F80] * length,
                },
                {
                    "case": f"pseudo_{length}",
                    "left": pseudo,
                    "right": list(reversed(pseudo)),
                },
            )
        )
    return vectors


def _torch_dot_results(left: list[int], right: list[int], device: Any) -> dict[str, str]:
    left_tensor = _torch_bfloat16_vector(left, device)
    right_tensor = _torch_bfloat16_vector(right, device)
    length = len(left)
    return {
        "LINEAR": _result_bits(
            functional.linear(left_tensor.reshape(1, length), right_tensor.reshape(1, length))
        ),
        "MATMUL_QK": _result_bits(
            torch.matmul(
                left_tensor.reshape(1, 1, 1, length),
                right_tensor.reshape(1, 1, 1, length).transpose(2, 3),
            )
        ),
        "MATMUL_AV": _result_bits(
            torch.matmul(
                left_tensor.reshape(1, 1, 1, length),
                right_tensor.reshape(1, 1, length, 1),
            )
        ),
    }


def build_reduction_characterization_certificate() -> dict[str, Any]:
    _require_torch()
    cuda_device = torch.device("cuda") if torch.cuda.is_available() else None
    records = []
    for vector in _vectors():
        candidates = _candidate_results(vector["left"], vector["right"])
        cpu = _torch_dot_results(vector["left"], vector["right"], torch.device("cpu"))
        cuda = (
            _torch_dot_results(vector["left"], vector["right"], cuda_device)
            if cuda_device is not None
            else None
        )
        for primitive in ("LINEAR", "MATMUL_QK", "MATMUL_AV"):
            body = {
                "case": vector["case"],
                "inner_dimension": len(vector["left"]),
                "primitive": primitive,
                "left_sha256": _sha256(vector["left"]),
                "right_sha256": _sha256(vector["right"]),
                "candidate_results": candidates,
                "cpu_result": cpu[primitive],
                "cuda_result": None if cuda is None else cuda[primitive],
                "cpu_matching_candidates": sorted(
                    name for name, result in candidates.items() if result == cpu[primitive]
                ),
                "cuda_matching_candidates": (
                    None
                    if cuda is None
                    else sorted(
                        name for name, result in candidates.items() if result == cuda[primitive]
                    )
                ),
                "candidate_results_distinct": len(set(candidates.values())),
            }
            records.append({**body, "record_sha256": _sha256(body)})
    candidate_names = sorted(records[0]["candidate_results"])
    match_counts = {
        backend: {
            primitive: {
                candidate: sum(
                    candidate
                    in (record[f"{backend}_matching_candidates"] or [])
                    for record in records
                    if record["primitive"] == primitive
                )
                for candidate in candidate_names
            }
            for primitive in ("LINEAR", "MATMUL_QK", "MATMUL_AV")
        }
        for backend in ("cpu", "cuda")
    }
    body = {
        "schema_version": 1,
        "scope": "Bounded characterization of bfloat16 LINEAR, MATMUL_QK, and MATMUL_AV outputs against explicit exact-sum, sequential, pairwise, block-16, and bfloat16 reduction candidates; no candidate is promoted to deployed reduction semantics.",
        "input_domain": "finite bfloat16 vectors with inner dimensions 4, 8, 16, 32, 64, 128, 256, 640, and 2048",
        "candidate_profiles": candidate_names,
        "record_count": len(records),
        "records": records,
        "match_counts": match_counts,
        "discriminating_record_count": sum(
            record["candidate_results_distinct"] > 1 for record in records
        ),
        "environment": {
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cuda_available": cuda_device is not None,
            "cuda_device_name": (
                torch.cuda.get_device_name(cuda_device)
                if cuda_device is not None
                else None
            ),
            "cuda_device_capability": (
                list(torch.cuda.get_device_capability(cuda_device))
                if cuda_device is not None
                else None
            ),
        },
        "unique_reduction_profile_identified": False,
        "reduction_order_semantics_established": False,
        "tensor_core_semantics_established": False,
        "hardware_instruction_semantics_established": False,
    }
    return {**body, "certificate_sha256": _sha256(body)}


def verify_reduction_characterization_certificate(
    certificate: dict[str, Any], reexecute: bool = True
) -> dict[str, Any]:
    if not isinstance(certificate, dict):
        return {"valid": False}
    body = {
        key: value for key, value in certificate.items() if key != "certificate_sha256"
    }
    try:
        certificate_hash_valid = _sha256(body) == certificate.get("certificate_sha256")
    except (TypeError, ValueError):
        return {"valid": False}
    records = certificate.get("records")

    def record_valid(record: Any) -> bool:
        if not isinstance(record, dict):
            return False
        try:
            record_body = {
                key: value for key, value in record.items() if key != "record_sha256"
            }
            candidates = record.get("candidate_results")
            return bool(
                record.get("primitive") in {"LINEAR", "MATMUL_QK", "MATMUL_AV"}
                and isinstance(record.get("inner_dimension"), int)
                and record.get("inner_dimension") > 0
                and isinstance(candidates, dict)
                and set(candidates)
                == {
                    "exact_sum_then_bfloat16",
                    "sequential_float32_then_bfloat16",
                    "pairwise_float32_then_bfloat16",
                    "block16_float32_then_bfloat16",
                    "sequential_bfloat16",
                }
                and record.get("cpu_matching_candidates")
                == sorted(
                    name
                    for name, result in candidates.items()
                    if result == record.get("cpu_result")
                )
                and (
                    record.get("cuda_matching_candidates") is None
                    if record.get("cuda_result") is None
                    else record.get("cuda_matching_candidates")
                    == sorted(
                        name
                        for name, result in candidates.items()
                        if result == record.get("cuda_result")
                    )
                )
                and record.get("candidate_results_distinct")
                == len(set(candidates.values()))
                and record.get("record_sha256") == _sha256(record_body)
            )
        except (TypeError, ValueError, AttributeError):
            return False

    records_valid = bool(
        isinstance(records, list)
        and records
        and all(record_valid(record) for record in records)
    )
    candidate_names = certificate.get("candidate_profiles")
    primitives = ("LINEAR", "MATMUL_QK", "MATMUL_AV")
    calculated_match_counts = {}
    if (
        isinstance(candidate_names, list)
        and candidate_names
        and all(isinstance(name, str) for name in candidate_names)
        and isinstance(records, list)
        and all(isinstance(record, dict) for record in records)
    ):
        calculated_match_counts = {
            backend: {
                primitive: {
                    candidate: sum(
                        candidate
                        in (record.get(f"{backend}_matching_candidates") or [])
                        for record in records
                        if record.get("primitive") == primitive
                    )
                    for candidate in candidate_names
                }
                for primitive in primitives
            }
            for backend in ("cpu", "cuda")
        }
    claims_consistent = bool(
        certificate.get("schema_version") == 1
        and candidate_names
        == sorted(
            {
                "exact_sum_then_bfloat16",
                "sequential_float32_then_bfloat16",
                "pairwise_float32_then_bfloat16",
                "block16_float32_then_bfloat16",
                "sequential_bfloat16",
            }
        )
        and certificate.get("match_counts") == calculated_match_counts
        and certificate.get("record_count")
        == (len(records) if isinstance(records, list) else -1)
        and certificate.get("discriminating_record_count")
        == (
            sum(record.get("candidate_results_distinct", 0) > 1 for record in records)
            if isinstance(records, list) and all(isinstance(record, dict) for record in records)
            else -1
        )
        and certificate.get("unique_reduction_profile_identified") is False
        and certificate.get("reduction_order_semantics_established") is False
        and certificate.get("tensor_core_semantics_established") is False
        and certificate.get("hardware_instruction_semantics_established") is False
    )
    reexecution_exact = True
    if reexecute:
        try:
            reexecution_exact = certificate == build_reduction_characterization_certificate()
        except (RuntimeError, TypeError, ValueError):
            reexecution_exact = False
    valid = all(
        (
            certificate_hash_valid,
            records_valid,
            claims_consistent,
            reexecution_exact,
        )
    )
    return {
        "valid": valid,
        "certificate_hash_valid": certificate_hash_valid,
        "records_valid": records_valid,
        "claims_consistent": claims_consistent,
        "reexecution_performed": reexecute,
        "reexecution_exact": reexecution_exact,
        "record_count": len(records) if isinstance(records, list) else 0,
        "discriminating_record_count": certificate.get("discriminating_record_count"),
        "unique_reduction_profile_identified": certificate.get(
            "unique_reduction_profile_identified"
        ),
        "reduction_order_semantics_established": certificate.get(
            "reduction_order_semantics_established"
        ),
    }
