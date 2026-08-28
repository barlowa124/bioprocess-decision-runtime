from __future__ import annotations

import hashlib
from typing import Any

from .serialization import canonical_json

try:
    import z3
except ModuleNotFoundError:
    z3 = None


def _require_z3() -> None:
    if z3 is None:
        raise RuntimeError("Z3 is required; install the proof extra")


def _ripple_add(left: Any, right: Any, width: int) -> Any:
    value = left
    carry = right
    for _ in range(width):
        value, carry = value ^ carry, (value & carry) << 1
    return z3.Extract(width - 1, 0, value)


def _shift_add_multiply(left: Any, right: Any, input_width: int) -> Any:
    output_width = input_width * 2
    multiplicand = z3.ZeroExt(input_width, left)
    result = z3.BitVecVal(0, output_width)
    for bit in range(input_width):
        term = multiplicand << bit
        selected = z3.If(z3.Extract(bit, bit, right) == 1, term, z3.BitVecVal(0, output_width))
        result = _ripple_add(result, selected, output_width)
    return result


def _proof(name: str, proposition: Any, scope: dict[str, Any]) -> dict[str, Any]:
    solver = z3.Solver()
    solver.add(z3.Not(proposition))
    result = solver.check()
    smt = solver.to_smt2()
    return {
        "name": name,
        "method": "counterexample search for negated universally quantified property",
        "solver_result": str(result),
        "proved": result == z3.unsat,
        "scope": scope,
        "smt2_sha256": hashlib.sha256(smt.encode("utf-8")).hexdigest(),
        "counterexample": str(solver.model()) if result == z3.sat else None,
    }


def build_formal_proof_certificate() -> dict[str, Any]:
    _require_z3()
    proofs = []

    width = 8
    left = z3.BitVec("add_left", width)
    right = z3.BitVec("add_right", width)
    proofs.append(
        _proof(
            "ripple_add_equals_modular_addition",
            _ripple_add(left, right, width) == left + right,
            {"input": "all pairs of 8-bit bitvectors", "arithmetic": "modulo 2^8"},
        )
    )

    multiply_width = 4
    multiplicand = z3.BitVec("multiplicand", multiply_width)
    multiplier = z3.BitVec("multiplier", multiply_width)
    direct_product = z3.ZeroExt(multiply_width, multiplicand) * z3.ZeroExt(multiply_width, multiplier)
    proofs.append(
        _proof(
            "shift_add_equals_modular_multiplication",
            _shift_add_multiply(multiplicand, multiplier, multiply_width) == direct_product,
            {"input": "all pairs of 4-bit unsigned bitvectors", "output": "8-bit modular product"},
        )
    )

    dot_width = 3
    inputs = [z3.BitVec(f"dot_input_{index}", dot_width) for index in range(3)]
    weights = [z3.BitVec(f"dot_weight_{index}", dot_width) for index in range(3)]
    accumulator_width = dot_width * 2 + 2
    direct_terms = [
        z3.ZeroExt(accumulator_width - dot_width, value) * z3.ZeroExt(accumulator_width - dot_width, weight)
        for value, weight in zip(inputs, weights)
    ]
    direct_dot = sum(direct_terms, z3.BitVecVal(0, accumulator_width))
    reference_dot = z3.BitVecVal(0, accumulator_width)
    for value, weight in zip(inputs, weights):
        product = _shift_add_multiply(value, weight, dot_width)
        product = z3.ZeroExt(accumulator_width - product.size(), product)
        reference_dot = _ripple_add(reference_dot, product, accumulator_width)
    proofs.append(
        _proof(
            "three_term_shift_add_dot_equals_direct_dot",
            reference_dot == direct_dot,
            {
                "input": "all six assignments of three 3-bit unsigned inputs and three 3-bit unsigned weights",
                "accumulator": f"{accumulator_width}-bit modular",
            },
        )
    )

    first = z3.BitVec("argmax_first", 8)
    second = z3.BitVec("argmax_second", 8)
    third = z3.BitVec("argmax_third", 8)
    selected = z3.If(
        z3.And(z3.UGE(first, second), z3.UGE(first, third)),
        z3.BitVecVal(0, 2),
        z3.If(z3.And(z3.UGT(second, first), z3.UGE(second, third)), z3.BitVecVal(1, 2), z3.BitVecVal(2, 2)),
    )
    selected_value = z3.If(selected == 0, first, z3.If(selected == 1, second, third))
    maximal = z3.And(z3.UGE(selected_value, first), z3.UGE(selected_value, second), z3.UGE(selected_value, third))
    first_tie = z3.Implies(z3.And(first == selected_value, z3.Or(second == selected_value, third == selected_value)), selected == 0)
    second_tie = z3.Implies(z3.And(second == selected_value, third == selected_value, first != selected_value), selected == 1)
    proofs.append(
        _proof(
            "first_index_argmax_is_maximal_and_stable_on_ties",
            z3.And(maximal, first_tie, second_tie),
            {"input": "all triples of 8-bit unsigned logits", "tie_rule": "lowest index"},
        )
    )

    query_position = z3.Int("query_position")
    key_position = z3.Int("key_position")
    sequence_length = z3.Int("sequence_length")
    window = z3.Int("window")
    domain = z3.And(
        sequence_length > 0,
        window > 0,
        query_position >= 0,
        query_position < sequence_length,
        key_position >= 0,
        key_position < sequence_length,
    )
    direct_allowed = z3.And(key_position <= query_position, key_position > query_position - window)
    distance_allowed = z3.And(key_position <= query_position, query_position - key_position < window)
    proofs.append(
        _proof(
            "sliding_causal_mask_equivalence",
            z3.ForAll(
                [query_position, key_position, sequence_length, window],
                z3.Implies(domain, direct_allowed == distance_allowed),
            ),
            {"input": "all integer sequence lengths, positive windows, and in-range query/key positions"},
        )
    )

    key_value_heads = z3.Int("key_value_heads")
    groups = z3.Int("groups")
    query_head = z3.Int("query_head")
    mapped_head = query_head / groups
    grouped_query_domain = z3.And(key_value_heads > 0, groups > 0, query_head >= 0, query_head < key_value_heads * groups)
    grouped_query_mapping = z3.And(
        mapped_head >= 0,
        mapped_head < key_value_heads,
        mapped_head * groups <= query_head,
        query_head < (mapped_head + 1) * groups,
    )
    proofs.append(
        _proof(
            "grouped_query_head_mapping_is_total_and_in_range",
            z3.ForAll(
                [key_value_heads, groups, query_head],
                z3.Implies(grouped_query_domain, grouped_query_mapping),
            ),
            {"input": "all positive key/value-head and group counts and every valid query-head index"},
        )
    )

    rotate_width = 8
    rotate_left = z3.BitVec("rotate_left", rotate_width)
    rotate_right = z3.BitVec("rotate_right", rotate_width)
    rotated_once_left, rotated_once_right = -rotate_right, rotate_left
    rotated_twice_left, rotated_twice_right = -rotated_once_right, rotated_once_left
    proofs.append(
        _proof(
            "rotate_half_encoding_double_application_is_negation",
            z3.And(rotated_twice_left == -rotate_left, rotated_twice_right == -rotate_right),
            {
                "input": "all pairs of 8-bit coordinates",
                "arithmetic": "modulo 2^8",
                "model": "Algebraic SMT encoding sanity check, not verification of a PyTorch or CUDA circuit.",
            },
        )
    )

    bfloat16 = z3.FPSort(8, 8)
    bfloat_value = z3.FP("bfloat_value", bfloat16)
    widened = z3.fpToFP(z3.RNE(), bfloat_value, z3.Float32())
    narrowed = z3.fpToFP(z3.RNE(), widened, bfloat16)
    proofs.append(
        _proof(
            "bfloat16_float32_round_trip_preserves_non_nan_bits",
            z3.Implies(
                z3.Not(z3.fpIsNaN(bfloat_value)),
                z3.fpToIEEEBV(narrowed) == z3.fpToIEEEBV(bfloat_value),
            ),
            {
                "input": "all non-NaN IEEE-754 bfloat16 bit patterns",
                "rounding": "round to nearest, ties to even",
                "model": "SMT-LIB IEEE-754 abstract theory, not a PyTorch or CUDA kernel claim.",
            },
        )
    )

    bfloat_one = z3.FPVal(1.0, bfloat16)
    multiplied_by_one = z3.fpMul(z3.RNE(), bfloat_value, bfloat_one)
    proofs.append(
        _proof(
            "bfloat16_multiplication_by_one_preserves_non_nan_bits",
            z3.Implies(
                z3.Not(z3.fpIsNaN(bfloat_value)),
                z3.fpToIEEEBV(multiplied_by_one) == z3.fpToIEEEBV(bfloat_value),
            ),
            {
                "input": "all non-NaN IEEE-754 bfloat16 bit patterns",
                "rounding": "round to nearest, ties to even",
                "model": "SMT-LIB IEEE-754 abstract theory, not a PyTorch or CUDA kernel claim.",
            },
        )
    )

    composition_input = z3.BitVec("composition_input", 8)
    direct_composition = (composition_input + z3.BitVecVal(7, 8)) * z3.BitVecVal(3, 8) + z3.BitVecVal(5, 8)
    first_stage = _ripple_add(composition_input, z3.BitVecVal(7, 8), 8)
    multiplied = z3.Extract(7, 0, _shift_add_multiply(first_stage, z3.BitVecVal(3, 8), 8))
    reference_composition = _ripple_add(multiplied, z3.BitVecVal(5, 8), 8)
    proofs.append(
        _proof(
            "verified_primitives_compose",
            reference_composition == direct_composition,
            {"input": "all 8-bit bitvectors", "program": "((x + 7) * 3) + 5 modulo 2^8"},
        )
    )

    body = {
        "scope": "Universal SMT proofs over explicitly stated bitvector, integer, and selected IEEE-754 bfloat16 formulas; not complete softmax, GELU, RMSNorm, or full-transformer proofs.",
        "solver": {"name": "Z3", "version": z3.get_version_string()},
        "proofs": proofs,
        "proved": sum(proof["proved"] for proof in proofs),
        "total": len(proofs),
        "unresolved": [
            "Complete IEEE-754 and bfloat16 operator equivalence beyond the proved properties",
            "Transcendental softmax, GELU, trigonometric RoPE, and reciprocal-square-root semantics",
            "full Gemma equivalence for unbounded token sequences",
            "CUDA SASS instruction-level semantic equivalence",
        ],
    }
    body["certificate_sha256"] = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    return body


def verify_formal_proof_certificate(certificate: dict[str, Any]) -> dict[str, Any]:
    stored_body = {key: value for key, value in certificate.items() if key != "certificate_sha256"}
    integrity_valid = hashlib.sha256(canonical_json(stored_body).encode("utf-8")).hexdigest() == certificate.get(
        "certificate_sha256"
    )
    recomputed = build_formal_proof_certificate()
    claim_fields = ("name", "method", "solver_result", "proved", "scope", "counterexample")
    recorded_claims = [{key: proof[key] for key in claim_fields} for proof in certificate.get("proofs", [])]
    recomputed_claims = [{key: proof[key] for key in claim_fields} for proof in recomputed["proofs"]]
    claims_match = recorded_claims == recomputed_claims
    metadata_fields = ("scope", "solver", "proved", "total", "unresolved")
    metadata_match = all(certificate.get(key) == recomputed[key] for key in metadata_fields)
    return {
        "valid": bool(
            integrity_valid and claims_match and metadata_match and recomputed["proved"] == recomputed["total"]
        ),
        "integrity_valid": integrity_valid,
        "reexecution_claims_match": claims_match,
        "reexecution_metadata_match": metadata_match,
        "exact_serialization_expected": False,
        "exact_serialization_reason": "Z3-generated SMT-LIB identifiers are not stable across constructions in one process.",
        "proved": recomputed["proved"],
        "total": recomputed["total"],
        "recomputed_certificate_sha256": recomputed["certificate_sha256"],
    }
