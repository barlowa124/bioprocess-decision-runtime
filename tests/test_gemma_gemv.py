from __future__ import annotations

import unittest
from fractions import Fraction

import numpy as np

from bioprocess_runtime import gemma_gemv as gemv
from bioprocess_runtime.gemma_float_semantics import decode_finite_bfloat16, encode_bfloat16_rne
from bioprocess_runtime.gemma_reduction_semantics import _float32_add_bits, decode_finite_float32, encode_float32_rne


class GemvArithmeticTests(unittest.TestCase):
    def test_fp32_add_matches_exact_rational_reference(self):
        rng = np.random.default_rng(61023)
        left = rng.integers(0, 2**32, 1600, dtype=np.uint32)
        right = rng.integers(0, 2**32, 1600, dtype=np.uint32)
        pairs = [(0,0), (0x80000000,0x80000000), (0,0x80000000), (0x3F800000,0xBF800000),
                 (0x3F800000,0x33800000), (0x3F800001,0x33800000), (0x3F800000,0xB3800000),
                 (0x3F800000,0xBF7FFFFF), (0x3F800001,0xBF800000), (0x7F000000,0x00800000)]
        pairs.extend((int(a),int(b)) for a,b in zip(left,right) if 0 < ((int(a) >> 23) & 255) < 255 and 0 < ((int(b) >> 23) & 255) < 255)
        valid, expected = [], []
        for a,b in pairs:
            result = _float32_add_bits(a,b)
            exponent = (result >> 23) & 255
            if exponent == 255 or exponent == 0 and result & 0x7FFFFF:
                with self.assertRaises(gemv.UnsupportedGemvArithmetic):
                    gemv.add_float32_bits(np.uint32(a),np.uint32(b))
            else:
                valid.append((a,b))
                expected.append(result)
        a,b = np.asarray(valid,dtype=np.uint32).T
        np.testing.assert_array_equal(gemv.add_float32_bits(a,b),np.asarray(expected,dtype=np.uint32))

    def test_fp32_cancellation_and_alignment_boundaries(self):
        pairs = []
        for exponent in (32,96,127,160,224):
            for fraction in (0,1,2,0x3FFFFF,0x7FFFFE,0x7FFFFF):
                base = (exponent << 23) | fraction
                pairs.extend((base,(base + delta) ^ 0x80000000) for delta in (-1,0,1))
                pairs.extend((base,((exponent-distance) << 23) | 0x800001) for distance in (1,2,23,24,25,26,27,31) if exponent > distance)
        for a,b in pairs:
            self.assertEqual(int(gemv.add_float32_bits(np.uint32(a),np.uint32(b))),_float32_add_bits(a,b))

    def test_product_matches_exact_reference_and_signed_zero(self):
        rng = np.random.default_rng(82716)
        left = ((rng.integers(64,191,1000,dtype=np.uint16) << 7) | rng.integers(0,128,1000,dtype=np.uint16) | (rng.integers(0,2,1000,dtype=np.uint16) << 15))
        right = ((127 << 7) | rng.integers(0,128,1000,dtype=np.uint16) | (rng.integers(0,2,1000,dtype=np.uint16) << 15))
        expected = [encode_float32_rne(decode_finite_bfloat16(int(a))[0] * decode_finite_bfloat16(int(b))[0]) for a,b in zip(left,right)]
        np.testing.assert_array_equal(gemv.multiply_bfloat16_to_float32_bits(left,right),np.asarray(expected,dtype=np.uint32))
        np.testing.assert_array_equal(gemv.multiply_bfloat16_to_float32_bits(np.array([0,0x8000,0x8000],dtype=np.uint16),np.array([0xBF80,0xBF80,0x3F80],dtype=np.uint16)),np.array([0x80000000,0,0x80000000],dtype=np.uint32))

    def test_bfloat_rounding_ties_sign_and_carry(self):
        bits = np.array([0,0x80000000,0x3F808000,0x3F818000,0x3FFF8000,0xBF808000,0xBF818000,0x00800000],dtype=np.uint32)
        expected = [encode_bfloat16_rne(*decode_finite_float32(int(value))) for value in bits]
        np.testing.assert_array_equal(gemv.round_bfloat16_bits(bits),expected)

    def test_unsupported_values_and_shapes_fail_closed(self):
        for bits in (1,0x80000001,0x7F800000,0x7FC00001):
            with self.assertRaises(gemv.UnsupportedGemvArithmetic):
                gemv.add_float32_bits(np.uint32(bits),np.uint32(0))
        for a,b in ((0x7F7FFFFF,0x7F7FFFFF),(0x00800001,0x80800000)):
            with self.assertRaises(gemv.UnsupportedGemvArithmetic):
                gemv.add_float32_bits(np.uint32(a),np.uint32(b))
        for a,b in ((1,0x3F80),(0x7F80,0x3F80),(0x0080,0x0080),(0x7F00,0x7F00)):
            with self.assertRaises(gemv.UnsupportedGemvArithmetic):
                gemv.multiply_bfloat16_to_float32_bits(np.uint16(a),np.uint16(b))
        with self.assertRaises(ValueError):
            gemv.add_float32_bits(np.array([0],dtype=np.uint16),np.array([0],dtype=np.uint32))
        with self.assertRaises(ValueError):
            gemv.project_bits(np.zeros((2,640),dtype=np.uint16),np.zeros((3,640),dtype=np.uint16))

    def test_projection_matches_rational_stride8_halving_specification(self):
        rng = np.random.default_rng(273814)
        left = ((rng.integers(118,132,(1,640),dtype=np.uint16) << 7) | rng.integers(0,128,(1,640),dtype=np.uint16) | (rng.integers(0,2,(1,640),dtype=np.uint16) << 15))
        weights = ((rng.integers(116,130,(6,640),dtype=np.uint16) << 7) | rng.integers(0,128,(6,640),dtype=np.uint16) | (rng.integers(0,2,(6,640),dtype=np.uint16) << 15))
        expected = []
        for row in weights:
            lanes = [0] * 8
            for index,(a,b) in enumerate(zip(left[0],row)):
                product = decode_finite_bfloat16(int(a))[0] * decode_finite_bfloat16(int(b))[0]
                lanes[index % 8] = _float32_add_bits(lanes[index % 8],encode_float32_rne(product))
            for half in (4,2,1):
                lanes = [_float32_add_bits(lanes[index],lanes[index+half]) for index in range(half)]
            expected.append(encode_bfloat16_rne(*decode_finite_float32(lanes[0])))
        for block in (1,4,32):
            np.testing.assert_array_equal(gemv.project_bits(left,weights,block),np.array([expected],dtype=np.uint16))
        self.assertEqual(len(gemv.code_sha256()),64)


if __name__ == "__main__":
    unittest.main()
