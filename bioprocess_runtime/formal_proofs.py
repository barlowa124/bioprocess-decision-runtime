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


def _counterexample(name: str, equality: Any, scope: dict[str, Any]) -> dict[str, Any]:
    solver = z3.Solver()
    solver.add(z3.Not(equality))
    result = solver.check()
    return {
        "name": name,
        "method": "SMT witness search against a proposed universal equality",
        "solver_result": str(result),
        "counterexample_found": result == z3.sat,
        "scope": scope,
        "witness": str(solver.model()) if result == z3.sat else None,
    }


def build_formal_proof_certificate() -> dict[str, Any]:
    _require_z3()
    proofs = []
    counterexamples = []

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

    second_bfloat = z3.FP("second_bfloat", bfloat16)
    bfloat_inputs_valid = z3.And(z3.Not(z3.fpIsNaN(bfloat_value)), z3.Not(z3.fpIsNaN(second_bfloat)))
    add_forward = z3.fpAdd(z3.RNE(), bfloat_value, second_bfloat)
    add_reverse = z3.fpAdd(z3.RNE(), second_bfloat, bfloat_value)
    proofs.append(
        _proof(
            "bfloat16_addition_is_bit_commutative_when_result_is_not_nan",
            z3.Implies(
                z3.And(bfloat_inputs_valid, z3.Not(z3.fpIsNaN(add_forward))),
                z3.fpToIEEEBV(add_forward) == z3.fpToIEEEBV(add_reverse),
            ),
            {
                "input": "all non-NaN IEEE-754 bfloat16 pairs whose sum is not NaN",
                "rounding": "round to nearest, ties to even",
                "model": "SMT-LIB IEEE-754 abstract theory, not a PyTorch or CUDA kernel claim.",
            },
        )
    )
    multiply_forward = z3.fpMul(z3.RNE(), bfloat_value, second_bfloat)
    multiply_reverse = z3.fpMul(z3.RNE(), second_bfloat, bfloat_value)
    proofs.append(
        _proof(
            "bfloat16_multiplication_is_bit_commutative_when_result_is_not_nan",
            z3.Implies(
                z3.And(bfloat_inputs_valid, z3.Not(z3.fpIsNaN(multiply_forward))),
                z3.fpToIEEEBV(multiply_forward) == z3.fpToIEEEBV(multiply_reverse),
            ),
            {
                "input": "all non-NaN IEEE-754 bfloat16 pairs whose product is not NaN",
                "rounding": "round to nearest, ties to even",
                "model": "SMT-LIB IEEE-754 abstract theory, not a PyTorch or CUDA kernel claim.",
            },
        )
    )
    negated_twice = z3.fpNeg(z3.fpNeg(bfloat_value))
    proofs.append(
        _proof(
            "bfloat16_double_negation_preserves_all_bits",
            z3.fpToIEEEBV(negated_twice) == z3.fpToIEEEBV(bfloat_value),
            {
                "input": "all IEEE-754 bfloat16 bit patterns, including infinities, signed zeros, and NaNs",
                "model": "SMT-LIB IEEE-754 abstract theory, not a PyTorch or CUDA kernel claim.",
            },
        )
    )

    tensor = z3.DeclareSort("AbstractTensor")
    stage_names = (
        "input_norm",
        "qkv_projection",
        "qk_norm",
        "rotary",
        "repeat_kv",
        "attention_scores",
        "causal_mask",
        "softmax",
        "value_aggregation",
        "attention_output_projection",
        "pre_feedforward_norm",
        "gated_gelu_mlp",
    )
    reference_stages = {name: z3.Function(f"reference_{name}", tensor, tensor) for name in stage_names}
    deployed_stages = {name: z3.Function(f"deployed_{name}", tensor, tensor) for name in stage_names}
    reference_residual = z3.Function("reference_residual", tensor, tensor, tensor)
    deployed_residual = z3.Function("deployed_residual", tensor, tensor, tensor)
    stage_value = z3.Const("stage_value", tensor)
    residual_left = z3.Const("residual_left", tensor)
    residual_right = z3.Const("residual_right", tensor)
    stage_axioms = [
        z3.ForAll([stage_value], reference_stages[name](stage_value) == deployed_stages[name](stage_value))
        for name in stage_names
    ]
    stage_axioms.append(
        z3.ForAll(
            [residual_left, residual_right],
            reference_residual(residual_left, residual_right)
            == deployed_residual(residual_left, residual_right),
        )
    )

    def layer_expression(stages: dict[str, Any], residual: Any, value: Any) -> Any:
        attention = stages["input_norm"](value)
        for name in stage_names[1:10]:
            attention = stages[name](attention)
        post_attention = residual(value, attention)
        feedforward = stages["pre_feedforward_norm"](post_attention)
        feedforward = stages["gated_gelu_mlp"](feedforward)
        return residual(post_attention, feedforward)

    layer_input = z3.Const("layer_input", tensor)
    reference_layer = layer_expression(reference_stages, reference_residual, layer_input)
    deployed_layer = layer_expression(deployed_stages, deployed_residual, layer_input)
    proofs.append(
        _proof(
            "gemma_layer_composition_is_equal_if_every_stage_is_extensionally_equal",
            z3.Implies(z3.And(stage_axioms), z3.ForAll([layer_input], reference_layer == deployed_layer)),
            {
                "input": "all values of an abstract tensor sort",
                "condition": "Every named Gemma stage and residual operator is extensionally equal across implementations.",
                "conclusion": "One complete attention-plus-MLP decoder layer is extensionally equal.",
                "boundary": "Conditional architecture-level congruence; it does not prove the stage premises or arbitrary-layer induction.",
            },
        )
    )

    layer_index = z3.Int("layer_index")
    reference_transition = z3.Function("reference_layer_transition", tensor, z3.IntSort(), tensor)
    deployed_transition = z3.Function("deployed_layer_transition", tensor, z3.IntSort(), tensor)
    induction_reference_state = z3.Const("induction_reference_state", tensor)
    induction_deployed_state = z3.Const("induction_deployed_state", tensor)
    transition_premise = z3.ForAll(
        [stage_value, layer_index],
        reference_transition(stage_value, layer_index) == deployed_transition(stage_value, layer_index),
    )
    induction_step = z3.ForAll(
        [induction_reference_state, induction_deployed_state, layer_index],
        z3.Implies(
            induction_reference_state == induction_deployed_state,
            reference_transition(induction_reference_state, layer_index)
            == deployed_transition(induction_deployed_state, layer_index),
        ),
    )
    proofs.append(
        _proof(
            "layerwise_equivalence_induction_step_is_valid_for_every_layer_index",
            z3.Implies(transition_premise, induction_step),
            {
                "input": "all abstract hidden states and integer layer indices",
                "condition": "Reference and deployed layer transitions are pointwise equal at every index.",
                "conclusion": "Equality of incoming states implies equality after the indexed layer.",
                "boundary": "Proves the induction step; base equality and every concrete transition premise must still be discharged.",
            },
        )
    )

    reference_final_norm = z3.Function("reference_final_norm", tensor, tensor)
    deployed_final_norm = z3.Function("deployed_final_norm", tensor, tensor)
    reference_lm_head = z3.Function("reference_lm_head", tensor, tensor)
    deployed_lm_head = z3.Function("deployed_lm_head", tensor, tensor)
    final_hidden = z3.Const("final_hidden", tensor)
    finalization_premise = z3.And(
        z3.ForAll([stage_value], reference_final_norm(stage_value) == deployed_final_norm(stage_value)),
        z3.ForAll([stage_value], reference_lm_head(stage_value) == deployed_lm_head(stage_value)),
    )
    proofs.append(
        _proof(
            "gemma_final_normalization_and_lm_head_compose_under_extensional_equality",
            z3.Implies(
                finalization_premise,
                z3.ForAll(
                    [final_hidden],
                    reference_lm_head(reference_final_norm(final_hidden))
                    == deployed_lm_head(deployed_final_norm(final_hidden)),
                ),
            ),
            {
                "input": "all values of an abstract final-hidden-state sort",
                "condition": "Final normalization and vocabulary projection are each extensionally equal.",
                "conclusion": "The complete vocabulary-logit outputs are equal.",
                "boundary": "Does not prove either operator premise or deployed argmax implementation.",
            },
        )
    )

    exponential_values = [z3.Real(f"exp_value_{index}") for index in range(3)]
    exponential_total = sum(exponential_values)
    probabilities = [value / exponential_total for value in exponential_values]
    positive_exponentials = z3.And(*(value > 0 for value in exponential_values))
    proofs.append(
        _proof(
            "abstract_three_way_softmax_is_positive_normalized_and_bounded",
            z3.Implies(
                positive_exponentials,
                z3.And(
                    sum(probabilities) == 1,
                    *(probability > 0 for probability in probabilities),
                    *(probability < 1 for probability in probabilities),
                ),
            ),
            {
                "input": "all triples of positive exact-real exponential outputs",
                "condition": "Exponentiation is abstracted to positive values.",
                "boundary": "Proves normalization algebra, not exp implementation, overflow handling, or floating-point reduction.",
            },
        )
    )
    shift_scale = z3.Real("softmax_shift_scale")
    shifted_total = sum(shift_scale * value for value in exponential_values)
    shifted_probabilities = [shift_scale * value / shifted_total for value in exponential_values]
    proofs.append(
        _proof(
            "abstract_softmax_is_invariant_to_common_positive_exponential_scale",
            z3.Implies(
                z3.And(positive_exponentials, shift_scale > 0),
                z3.And(*(left == right for left, right in zip(probabilities, shifted_probabilities))),
            ),
            {
                "input": "all triples of positive exact-real exponential outputs and positive common scales",
                "connection": "Models exact-real softmax invariance under an additive logit shift when exp(x+c)=exp(c)exp(x).",
                "boundary": "The exponential identity and floating-point implementation are premises, not proved here.",
            },
        )
    )

    rms_first = z3.Real("rms_first")
    rms_second = z3.Real("rms_second")
    rms_epsilon = z3.Real("rms_epsilon")
    rms_root = z3.Real("rms_root")
    mean_square = (rms_first * rms_first + rms_second * rms_second) / 2
    normalized_mean_square = mean_square / (rms_root * rms_root)
    proofs.append(
        _proof(
            "abstract_rms_normalization_has_mean_square_below_one_with_positive_epsilon",
            z3.Implies(
                z3.And(rms_epsilon > 0, rms_root > 0, rms_root * rms_root == mean_square + rms_epsilon),
                z3.And(normalized_mean_square >= 0, normalized_mean_square < 1),
            ),
            {
                "input": "all pairs of exact-real coordinates and positive epsilon/root satisfying the RMS equation",
                "boundary": "Proves an exact-real invariant, not rsqrt approximation, casting, weighting, or kernel rounding.",
            },
        )
    )

    gelu_input = z3.Real("gelu_input")
    tanh_value = z3.Real("gelu_tanh_value")
    abstract_gelu = gelu_input * (1 + tanh_value) / 2
    proofs.append(
        _proof(
            "abstract_gelu_tanh_factor_bounds_output_by_input_and_zero",
            z3.Implies(
                z3.And(tanh_value >= -1, tanh_value <= 1),
                z3.And(
                    z3.Implies(gelu_input >= 0, z3.And(abstract_gelu >= 0, abstract_gelu <= gelu_input)),
                    z3.Implies(gelu_input <= 0, z3.And(abstract_gelu <= 0, abstract_gelu >= gelu_input)),
                ),
            ),
            {
                "input": "all exact-real inputs and abstract tanh outputs in [-1,1]",
                "boundary": "Proves the gating bound, not the tanh approximation or floating-point implementation.",
            },
        )
    )

    rope_first = z3.Real("rope_first")
    rope_second = z3.Real("rope_second")
    rope_cosine = z3.Real("rope_cosine")
    rope_sine = z3.Real("rope_sine")
    rotated_first = rope_first * rope_cosine - rope_second * rope_sine
    rotated_second = rope_first * rope_sine + rope_second * rope_cosine
    proofs.append(
        _proof(
            "abstract_rope_pair_preserves_squared_norm_under_trigonometric_identity",
            z3.Implies(
                rope_cosine * rope_cosine + rope_sine * rope_sine == 1,
                rotated_first * rotated_first + rotated_second * rotated_second
                == rope_first * rope_first + rope_second * rope_second,
            ),
            {
                "input": "all exact-real coordinate pairs and sine/cosine values satisfying cos^2+sin^2=1",
                "boundary": "Proves rotation algebra, not sin/cos evaluation, scaling, casting, or kernel rounding.",
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

    third_bfloat = z3.FP("third_bfloat", bfloat16)
    left_associated = z3.fpAdd(z3.RNE(), z3.fpAdd(z3.RNE(), bfloat_value, second_bfloat), third_bfloat)
    right_associated = z3.fpAdd(z3.RNE(), bfloat_value, z3.fpAdd(z3.RNE(), second_bfloat, third_bfloat))
    finite_three = z3.And(
        z3.Not(z3.fpIsNaN(bfloat_value)),
        z3.Not(z3.fpIsNaN(second_bfloat)),
        z3.Not(z3.fpIsNaN(third_bfloat)),
        z3.Not(z3.fpIsInf(bfloat_value)),
        z3.Not(z3.fpIsInf(second_bfloat)),
        z3.Not(z3.fpIsInf(third_bfloat)),
        z3.Not(z3.fpIsNaN(left_associated)),
        z3.Not(z3.fpIsNaN(right_associated)),
    )
    counterexamples.append(
        _counterexample(
            "bfloat16_addition_is_not_associative",
            z3.Implies(
                finite_three,
                z3.fpToIEEEBV(left_associated) == z3.fpToIEEEBV(right_associated),
            ),
            {
                "input": "finite non-NaN bfloat16 triples with non-NaN intermediate results",
                "importance": "Reduction order cannot be ignored when comparing eager and fused kernels.",
            },
        )
    )
    fused = z3.fpFMA(z3.RNE(), bfloat_value, second_bfloat, third_bfloat)
    unfused = z3.fpAdd(z3.RNE(), z3.fpMul(z3.RNE(), bfloat_value, second_bfloat), third_bfloat)
    finite_fma = z3.And(
        finite_three,
        z3.Not(z3.fpIsNaN(fused)),
        z3.Not(z3.fpIsNaN(unfused)),
    )
    counterexamples.append(
        _counterexample(
            "bfloat16_fused_multiply_add_can_differ_from_separate_operations",
            z3.Implies(finite_fma, z3.fpToIEEEBV(fused) == z3.fpToIEEEBV(unfused)),
            {
                "input": "finite non-NaN bfloat16 triples with non-NaN results",
                "importance": "Fused and unfused implementation paths are not universally interchangeable.",
            },
        )
    )

    body = {
        "scope": "Universal SMT proofs over explicitly stated bitvector, integer, selected IEEE-754 bfloat16, and conditional architecture-composition formulas; not complete transcendental or full-transformer implementation proofs.",
        "solver": {"name": "Z3", "version": z3.get_version_string()},
        "proofs": proofs,
        "proved": sum(proof["proved"] for proof in proofs),
        "total": len(proofs),
        "counterexamples": counterexamples,
        "counterexamples_found": sum(item["counterexample_found"] for item in counterexamples),
        "unresolved": [
            "Complete IEEE-754 and bfloat16 operator equivalence beyond the proved properties",
            "Transcendental softmax, GELU, trigonometric RoPE, and reciprocal-square-root semantics",
            "Discharge of every layer-stage premise and induction across arbitrary layers and token sequences",
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
    counterexample_fields = ("name", "method", "solver_result", "counterexample_found", "scope")
    recorded_counterexamples = [
        {key: item[key] for key in counterexample_fields} for item in certificate.get("counterexamples", [])
    ]
    recomputed_counterexamples = [
        {key: item[key] for key in counterexample_fields} for item in recomputed["counterexamples"]
    ]
    counterexamples_match = recorded_counterexamples == recomputed_counterexamples
    metadata_fields = ("scope", "solver", "proved", "total", "counterexamples_found", "unresolved")
    metadata_match = all(certificate.get(key) == recomputed[key] for key in metadata_fields)
    return {
        "valid": bool(
            integrity_valid
            and claims_match
            and counterexamples_match
            and metadata_match
            and recomputed["proved"] == recomputed["total"]
        ),
        "integrity_valid": integrity_valid,
        "reexecution_claims_match": claims_match,
        "reexecution_counterexamples_match": counterexamples_match,
        "reexecution_metadata_match": metadata_match,
        "exact_serialization_expected": False,
        "exact_serialization_reason": "Z3-generated identifiers and satisfying witness models are not stable across constructions.",
        "proved": recomputed["proved"],
        "total": recomputed["total"],
        "recomputed_certificate_sha256": recomputed["certificate_sha256"],
    }
