from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .gemma_rotary_slice import _sha

SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
PROFILE = "gemv_k640_stride8_fp32_rne_halves_bf16_v1"
SCOPE = "Candidate for fixed-runtime M1/N262144/K640 BF16 GEMV; selected after baseline observation. Integer operations only. Normal operands/products/carries or signed zero; no hardware or unrestricted-domain proof."


class UnsupportedGemvArithmetic(ValueError):
    pass


def code_sha256():
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("GEMV specification source changed")
    return _sha({"source": SOURCE_SHA256, "profile": PROFILE, "scope": SCOPE})


def _normal_or_zero(bits, width):
    shift = 23 if width == 32 else 7
    exponent = (bits >> shift) & 255
    fraction = bits & ((1 << shift) - 1)
    if np.any(exponent == 255) or np.any((exponent == 0) & (fraction != 0)):
        raise UnsupportedGemvArithmetic("GEMV accepts normal finite values and signed zero only")


def add_float32_bits(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if left.dtype != np.uint32 or right.dtype != np.uint32 or left.shape != right.shape:
        raise ValueError("Expected equally shaped uint32 FP32 bit arrays")
    _normal_or_zero(left, 32)
    _normal_or_zero(right, 32)
    swap = (left & 0x7FFFFFFF) < (right & 0x7FFFFFFF)
    large = np.where(swap, right, left).astype(np.uint64)
    small = np.where(swap, left, right).astype(np.uint64)
    exponent = ((large >> 23) & 255).astype(np.int64)
    small_exponent = ((small >> 23) & 255).astype(np.int64)
    large_significand = np.where(exponent != 0, (large & 0x7FFFFF) | 0x800000, 0).astype(np.uint64) << np.uint64(3)
    small_significand = np.where(small_exponent != 0, (small & 0x7FFFFF) | 0x800000, 0).astype(np.uint64) << np.uint64(3)
    distance = np.minimum(exponent - small_exponent, 31).astype(np.uint64)
    discarded = small_significand & ((np.uint64(1) << distance) - np.uint64(1))
    aligned = (small_significand >> distance) | (discarded != 0).astype(np.uint64)
    same_sign = ((large ^ small) & 0x80000000) == 0
    significand = np.where(same_sign, large_significand + aligned, large_significand - aligned)
    carry = significand >= (1 << 27)
    significand = np.where(carry, (significand >> np.uint64(1)) | (significand & np.uint64(1)), significand)
    exponent += carry.astype(np.int64)
    for shift in (16, 8, 4, 2, 1):
        normalize = (significand != 0) & (significand < (1 << (26 - shift)))
        significand = np.where(normalize, significand << np.uint64(shift), significand)
        exponent -= normalize.astype(np.int64) * shift
    normalize = (significand != 0) & (significand < (1 << 26))
    significand = np.where(normalize, significand << np.uint64(1), significand)
    exponent -= normalize.astype(np.int64)
    nonzero = significand != 0
    if np.any(nonzero & ((exponent <= 0) | (exponent >= 255))):
        raise UnsupportedGemvArithmetic("GEMV FP32 carry outside normal finite domain")
    rounded = (significand >> np.uint64(3)) + (((significand & 7) > 4) | (((significand & 7) == 4) & (((significand >> np.uint64(3)) & 1) != 0))).astype(np.uint64)
    carry = rounded >= (1 << 24)
    rounded = np.where(carry, rounded >> np.uint64(1), rounded)
    exponent += carry.astype(np.int64)
    if np.any(nonzero & (exponent >= 255)):
        raise UnsupportedGemvArithmetic("GEMV FP32 rounding overflow")
    zero_sign = (left & right & 0x80000000).astype(np.uint64)
    bits = (large & 0x80000000) | (exponent.astype(np.uint64) << np.uint64(23)) | (rounded & 0x7FFFFF)
    return np.where(nonzero, bits, zero_sign).astype(np.uint32)


def multiply_bfloat16_to_float32_bits(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if left.dtype != np.uint16 or right.dtype != np.uint16:
        raise ValueError("Expected uint16 BF16 bit arrays")
    _normal_or_zero(left, 16)
    _normal_or_zero(right, 16)
    left, right = np.broadcast_arrays(left.astype(np.uint32), right.astype(np.uint32))
    left_exponent, right_exponent = (left >> 7) & 255, (right >> 7) & 255
    product = ((left & 127) | 128) * ((right & 127) | 128)
    high = product >= (1 << 15)
    exponent = left_exponent.astype(np.int64) + right_exponent.astype(np.int64) - 127 + high.astype(np.int64)
    zero = (left_exponent == 0) | (right_exponent == 0)
    if np.any(~zero & ((exponent <= 0) | (exponent >= 255))):
        raise UnsupportedGemvArithmetic("GEMV product outside normal finite FP32 domain")
    significand = product << np.where(high, 8, 9).astype(np.uint32)
    sign = ((left ^ right) & 0x8000) << 16
    return np.where(zero, sign, sign | (exponent.astype(np.uint32) << 23) | (significand & 0x7FFFFF)).astype(np.uint32)


def round_bfloat16_bits(bits):
    bits = np.asarray(bits)
    if bits.dtype != np.uint32:
        raise ValueError("Expected uint32 FP32 bits")
    _normal_or_zero(bits, 32)
    result = ((bits.astype(np.uint64) + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    _normal_or_zero(result, 16)
    return result


def project_bits(left, weights, block_size=1024):
    left, weights = np.asarray(left), np.asarray(weights)
    if left.dtype != np.uint16 or weights.dtype != np.uint16 or left.shape != (1, 640) or weights.ndim != 2 or weights.shape[1] != 640 or weights.shape[0] == 0:
        raise ValueError("GEMV candidate requires one K640 row and a nonempty K640 weight matrix")
    if type(block_size) is not int or block_size < 1:
        raise ValueError("Positive integer block size required")
    code = code_sha256()
    output = np.empty((1, weights.shape[0]), dtype=np.uint16)
    for start in range(0, weights.shape[0], block_size):
        products = multiply_bfloat16_to_float32_bits(left, weights[start:start + block_size])
        lanes = np.zeros((len(products), 8), dtype=np.uint32)
        for index in range(0, 640, 8):
            lanes = add_float32_bits(lanes, products[:, index:index + 8])
        for half in (4, 2, 1):
            lanes = add_float32_bits(lanes[:, :half], lanes[:, half:half * 2])
        output[0, start:start + len(products)] = round_bfloat16_bits(lanes[:, 0])
    if code_sha256() != code:
        raise ValueError("GEMV source changed during prediction")
    return output
