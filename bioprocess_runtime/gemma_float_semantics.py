from __future__ import annotations

from fractions import Fraction
import hashlib
import itertools
import platform
import re
from typing import Any

from .serialization import canonical_json

try:
    import torch
except ImportError:
    torch = None


BFLOAT16_SIGN_MASK = 0x8000
BFLOAT16_EXPONENT_MASK = 0x7F80
BFLOAT16_FRACTION_MASK = 0x007F
BFLOAT16_INFINITY = 0x7F80


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("Bfloat16 conformance capture requires PyTorch")


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _scale_power_of_two(value: Fraction, exponent: int) -> Fraction:
    if exponent >= 0:
        return Fraction(value.numerator << exponent, value.denominator)
    return Fraction(value.numerator, value.denominator << -exponent)


def _round_nearest_even(value: Fraction) -> int:
    if value < 0:
        raise ValueError("Round-to-nearest-even input must be nonnegative")
    quotient, remainder = divmod(value.numerator, value.denominator)
    comparison = remainder * 2 - value.denominator
    if comparison > 0 or (comparison == 0 and quotient % 2 == 1):
        return quotient + 1
    return quotient


def _at_least_power_of_two(value: Fraction, exponent: int) -> bool:
    if exponent >= 0:
        return value.numerator >= (value.denominator << exponent)
    return (value.numerator << -exponent) >= value.denominator


def _floor_log2(value: Fraction) -> int:
    if value <= 0:
        raise ValueError("Logarithm input must be positive")
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if not _at_least_power_of_two(value, exponent):
        exponent -= 1
    while _at_least_power_of_two(value, exponent + 1):
        exponent += 1
    return exponent


def decode_finite_bfloat16(bits: int) -> tuple[Fraction, bool]:
    if not isinstance(bits, int) or isinstance(bits, bool) or not 0 <= bits <= 0xFFFF:
        raise ValueError("Bfloat16 bits must be an unsigned 16-bit integer")
    exponent = (bits & BFLOAT16_EXPONENT_MASK) >> 7
    fraction = bits & BFLOAT16_FRACTION_MASK
    if exponent == 0xFF:
        raise ValueError("NaN and infinity inputs are outside finite bfloat16 semantics")
    negative = bool(bits & BFLOAT16_SIGN_MASK)
    if exponent == 0:
        magnitude = _scale_power_of_two(Fraction(fraction), -133)
    else:
        magnitude = _scale_power_of_two(Fraction(128 + fraction), exponent - 134)
    return (-magnitude if negative else magnitude), negative and magnitude == 0


def encode_bfloat16_rne(value: Fraction, negative_zero: bool = False) -> int:
    if not isinstance(value, Fraction):
        value = Fraction(value)
    negative = value < 0 or (value == 0 and negative_zero)
    magnitude = abs(value)
    sign = BFLOAT16_SIGN_MASK if negative else 0
    if magnitude == 0:
        return sign
    exponent = _floor_log2(magnitude)
    if exponent >= -126:
        significand = _round_nearest_even(
            _scale_power_of_two(magnitude, 7 - exponent)
        )
        if significand == 256:
            significand = 128
            exponent += 1
        if exponent > 127:
            return sign | BFLOAT16_INFINITY
        return sign | ((exponent + 127) << 7) | (significand - 128)
    subnormal = _round_nearest_even(_scale_power_of_two(magnitude, 133))
    if subnormal == 0:
        return sign
    if subnormal >= 128:
        return sign | 0x0080
    return sign | subnormal


def bfloat16_add_bits(left_bits: int, right_bits: int) -> int:
    left, left_negative_zero = decode_finite_bfloat16(left_bits)
    right, right_negative_zero = decode_finite_bfloat16(right_bits)
    result = left + right
    negative_zero = result == 0 and left_negative_zero and right_negative_zero
    return encode_bfloat16_rne(result, negative_zero)


def bfloat16_multiply_bits(left_bits: int, right_bits: int) -> int:
    left, left_negative_zero = decode_finite_bfloat16(left_bits)
    right, right_negative_zero = decode_finite_bfloat16(right_bits)
    result = left * right
    left_negative = left < 0 or left_negative_zero
    right_negative = right < 0 or right_negative_zero
    return encode_bfloat16_rne(
        result, negative_zero=result == 0 and left_negative != right_negative
    )


def _torch_bfloat16_bits(bits: int, device: Any) -> Any:
    return torch.tensor([bits], dtype=torch.uint16).view(torch.bfloat16).to(device)


def _result_bits(value: Any) -> int:
    return int(value.detach().cpu().view(torch.uint16).item())


def _representative_finite_patterns() -> tuple[int, ...]:
    return (
        0x0000,
        0x8000,
        0x0001,
        0x8001,
        0x007F,
        0x807F,
        0x0080,
        0x8080,
        0x3E80,
        0xBE80,
        0x3F00,
        0xBF00,
        0x3F80,
        0xBF80,
        0x3FC0,
        0xBFC0,
        0x4000,
        0xC000,
        0x4040,
        0xC040,
        0x7F7F,
        0xFF7F,
    )


def _operation_record(
    operation: str,
    left_bits: int,
    right_bits: int,
    oracle_bits: int,
    cpu_bits: int,
    cuda_bits: int | None,
) -> dict[str, Any]:
    body = {
        "operation": operation,
        "left_bits": f"0x{left_bits:04x}",
        "right_bits": f"0x{right_bits:04x}",
        "oracle_result_bits": f"0x{oracle_bits:04x}",
        "cpu_result_bits": f"0x{cpu_bits:04x}",
        "cuda_result_bits": None if cuda_bits is None else f"0x{cuda_bits:04x}",
        "cpu_exact": cpu_bits == oracle_bits,
        "cuda_exact": None if cuda_bits is None else cuda_bits == oracle_bits,
    }
    return {**body, "record_sha256": _sha256(body)}


def build_bfloat16_semantics_certificate() -> dict[str, Any]:
    _require_torch()
    finite_roundtrip_count = 0
    for bits in range(0x10000):
        if bits & BFLOAT16_EXPONENT_MASK == BFLOAT16_EXPONENT_MASK:
            continue
        value, negative_zero = decode_finite_bfloat16(bits)
        if encode_bfloat16_rne(value, negative_zero) != bits:
            raise RuntimeError(f"Finite bfloat16 round-trip failed for 0x{bits:04x}")
        finite_roundtrip_count += 1

    cuda_device = torch.device("cuda") if torch.cuda.is_available() else None
    records = []
    patterns = _representative_finite_patterns()
    for left_bits, right_bits in itertools.product(patterns, repeat=2):
        left_cpu = _torch_bfloat16_bits(left_bits, torch.device("cpu"))
        right_cpu = _torch_bfloat16_bits(right_bits, torch.device("cpu"))
        for operation, oracle, cpu_result in (
            (
                "ADD",
                bfloat16_add_bits(left_bits, right_bits),
                left_cpu + right_cpu,
            ),
            (
                "MUL",
                bfloat16_multiply_bits(left_bits, right_bits),
                left_cpu * right_cpu,
            ),
        ):
            cuda_bits = None
            if cuda_device is not None:
                left_cuda = _torch_bfloat16_bits(left_bits, cuda_device)
                right_cuda = _torch_bfloat16_bits(right_bits, cuda_device)
                cuda_result = (
                    left_cuda + right_cuda if operation == "ADD" else left_cuda * right_cuda
                )
                cuda_bits = _result_bits(cuda_result)
            records.append(
                _operation_record(
                    operation,
                    left_bits,
                    right_bits,
                    oracle,
                    _result_bits(cpu_result),
                    cuda_bits,
                )
            )

    scale_patterns = (0x3D80, 0x3E80, 0x3F00, 0x3F80, 0x4000)
    for value_bits, scale_bits in itertools.product(patterns, scale_patterns):
        value_cpu = _torch_bfloat16_bits(value_bits, torch.device("cpu"))
        scale_cpu = _torch_bfloat16_bits(scale_bits, torch.device("cpu"))
        scale_value, _ = decode_finite_bfloat16(scale_bits)
        python_scalar = float(scale_value)
        oracle = bfloat16_multiply_bits(value_bits, scale_bits)
        tensor_cuda_bits = None
        python_cuda_bits = None
        if cuda_device is not None:
            value_cuda = _torch_bfloat16_bits(value_bits, cuda_device)
            tensor_cuda_bits = _result_bits(
                value_cuda * _torch_bfloat16_bits(scale_bits, cuda_device)
            )
            python_cuda_bits = _result_bits(value_cuda * python_scalar)
        records.append(
            _operation_record(
                "SCALE_TENSOR",
                value_bits,
                scale_bits,
                oracle,
                _result_bits(value_cpu * scale_cpu),
                tensor_cuda_bits,
            )
        )
        records.append(
            _operation_record(
                "SCALE_PYTHON",
                value_bits,
                scale_bits,
                oracle,
                _result_bits(value_cpu * python_scalar),
                python_cuda_bits,
            )
        )

    operation_counts = {
        operation: sum(record["operation"] == operation for record in records)
        for operation in ("ADD", "MUL", "SCALE_TENSOR", "SCALE_PYTHON")
    }
    body = {
        "schema_version": 1,
        "scope": "Independent exact-rational software semantics and bounded CPU/CUDA conformance for finite-input IEEE-754 bfloat16 round-to-nearest-even ADD, MUL, and tensor SCALE; not exhaustive binary-operation, reduction, transcendental, or hardware-instruction qualification.",
        "format": {
            "bits": 16,
            "exponent_bits": 8,
            "fraction_bits": 7,
            "bias": 127,
            "rounding": "round to nearest, ties to even",
            "input_domain": "finite bfloat16 bit patterns",
            "nan_and_infinity_inputs_supported": False,
        },
        "software_semantics": {
            "ADD": "decode exact dyadic rationals, add exactly, then encode once with bfloat16 RNE",
            "MUL": "decode exact dyadic rationals, multiply exactly, then encode once with bfloat16 RNE",
            "SCALE_TENSOR": "bfloat16 MUL semantics with a bfloat16 scalar tensor",
            "SCALE_PYTHON": "bfloat16 MUL semantics with an exactly bfloat16-representable Python scalar",
        },
        "software_semantics_sha256": _sha256(
            {
                "decode": "finite IEEE-754 bfloat16 to exact Fraction",
                "encode": "exact Fraction to IEEE-754 bfloat16 RNE",
                "add": "exact rational addition followed by one encode",
                "multiply": "exact rational multiplication followed by one encode",
            }
        ),
        "finite_pattern_roundtrip_count": finite_roundtrip_count,
        "finite_pattern_roundtrip_complete": finite_roundtrip_count == 65280,
        "representative_input_patterns": [f"0x{bits:04x}" for bits in patterns],
        "operation_counts": operation_counts,
        "records": records,
        "cpu_all_exact": all(record["cpu_exact"] for record in records),
        "environment": {
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
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
        "cuda_available": cuda_device is not None,
        "cuda_device": torch.cuda.get_device_name(cuda_device) if cuda_device is not None else None,
        "cuda_all_exact": (
            all(record["cuda_exact"] is True for record in records)
            if cuda_device is not None
            else None
        ),
        "complete_binary_truth_tables_established": False,
        "reduction_order_semantics_established": False,
        "hardware_instruction_semantics_established": False,
    }
    return {**body, "certificate_sha256": _sha256(body)}


def _parse_bfloat16_hex(value: Any) -> int:
    if not isinstance(value, str) or re.fullmatch(r"0x[0-9a-f]{4}", value) is None:
        raise ValueError("Invalid canonical bfloat16 hexadecimal value")
    return int(value, 16)


def _record_oracle_bits(record: dict[str, Any]) -> str:
    left = _parse_bfloat16_hex(record.get("left_bits"))
    right = _parse_bfloat16_hex(record.get("right_bits"))
    operation = record.get("operation")
    if operation == "ADD":
        result = bfloat16_add_bits(left, right)
    elif operation in {"MUL", "SCALE_TENSOR", "SCALE_PYTHON"}:
        result = bfloat16_multiply_bits(left, right)
    else:
        raise ValueError("Unknown bfloat16 operation")
    return f"0x{result:04x}"


def verify_bfloat16_semantics_certificate(
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
            return bool(
                record.get("operation")
                in {"ADD", "MUL", "SCALE_TENSOR", "SCALE_PYTHON"}
                and record.get("oracle_result_bits") == _record_oracle_bits(record)
                and record.get("cpu_exact") is True
                and record.get("oracle_result_bits") == record.get("cpu_result_bits")
                and (
                    record.get("cuda_exact") is None
                    or (
                        record.get("cuda_exact") is True
                        and record.get("oracle_result_bits")
                        == record.get("cuda_result_bits")
                    )
                )
                and record.get("record_sha256") == _sha256(record_body)
            )
        except (TypeError, ValueError, AttributeError):
            return False

    records_valid = bool(
        isinstance(records, list) and records and all(record_valid(record) for record in records)
    )
    operations = ("ADD", "MUL", "SCALE_TENSOR", "SCALE_PYTHON")
    calculated_counts = {
        operation: sum(
            isinstance(record, dict) and record.get("operation") == operation
            for record in records
        )
        for operation in operations
    } if isinstance(records, list) else {}
    claims_consistent = bool(
        certificate.get("schema_version") == 1
        and certificate.get("finite_pattern_roundtrip_count") == 65280
        and certificate.get("finite_pattern_roundtrip_complete") is True
        and certificate.get("operation_counts") == calculated_counts
        and all(count > 0 for count in calculated_counts.values())
        and certificate.get("cpu_all_exact") is True
        and isinstance(certificate.get("environment"), dict)
        and isinstance(certificate["environment"].get("platform"), str)
        and isinstance(certificate["environment"].get("torch_version"), str)
        and (
            certificate.get("cuda_all_exact") is True
            if certificate.get("cuda_available") is True
            else certificate.get("cuda_all_exact") is None
        )
        and certificate.get("complete_binary_truth_tables_established") is False
        and certificate.get("reduction_order_semantics_established") is False
        and certificate.get("hardware_instruction_semantics_established") is False
    )
    reexecution_exact = True
    if reexecute:
        try:
            reexecution_exact = certificate == build_bfloat16_semantics_certificate()
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
        "finite_pattern_roundtrip_count": certificate.get(
            "finite_pattern_roundtrip_count"
        ),
        "operation_counts": calculated_counts,
        "cuda_available": certificate.get("cuda_available"),
        "cuda_all_exact": certificate.get("cuda_all_exact"),
    }
