from __future__ import annotations

import hashlib
from typing import Any

from .formal_proofs import _require_z3, _ripple_add, _shift_add_multiply
from .serialization import canonical_json

try:
    import z3
except ModuleNotFoundError:
    z3 = None


def sass_mov(source: Any) -> Any:
    return source


def sass_iadd3(first: Any, second: Any, third: Any, width: int) -> Any:
    return z3.Extract(width - 1, 0, first + second + third)


def sass_imad(first: Any, second: Any, addend: Any, width: int) -> Any:
    return z3.Extract(width - 1, 0, first * second + addend)


def sass_lop3(first: Any, second: Any, third: Any, lut: int, width: int) -> Any:
    output_bits = []
    for bit in range(width):
        index = z3.Concat(z3.Extract(bit, bit, third), z3.Extract(bit, bit, second), z3.Extract(bit, bit, first))
        selected = z3.BitVecVal((lut >> 7) & 1, 1)
        for value in reversed(range(7)):
            selected = z3.If(index == value, z3.BitVecVal((lut >> value) & 1, 1), selected)
        output_bits.append(selected)
    return z3.Concat(*reversed(output_bits))


def sass_sel(first: Any, second: Any, predicate: Any) -> Any:
    return z3.If(predicate, first, second)


def sass_isetp_ge_unsigned(first: Any, second: Any) -> Any:
    return z3.UGE(first, second)


def sass_predicate_compare(first: Any, second: Any, relation: str, unsigned: bool, prior: Any) -> Any:
    relations = {
        ("eq", False): first == second,
        ("ne", False): first != second,
        ("ge", False): first >= second,
        ("gt", False): first > second,
        ("lt", False): first < second,
        ("ge", True): z3.UGE(first, second),
        ("gt", True): z3.UGT(first, second),
        ("lt", True): z3.ULT(first, second),
    }
    return z3.And(relations[(relation, unsigned)], prior)


def sass_shf_right_unsigned_high(low: Any, high: Any, shift: Any, width: int) -> Any:
    combined = z3.Concat(high, low)
    shifted = z3.LShR(combined, z3.ZeroExt(width, shift))
    return z3.Extract(width - 1, 0, shifted)


def sass_fp_add(first: Any, second: Any) -> Any:
    return z3.fpAdd(z3.RNE(), first, second)


def sass_fp_multiply(first: Any, second: Any) -> Any:
    return z3.fpMul(z3.RNE(), first, second)


def sass_fp_fma(first: Any, second: Any, third: Any) -> Any:
    return z3.fpFMA(z3.RNE(), first, second, third)


def sass_nop(state: Any) -> Any:
    return state


def sass_branch(target: Any) -> Any:
    return target


def sass_exit() -> Any:
    return z3.BoolVal(True)


def sass_memory_load_little_endian(memory: Any, address: Any, byte_count: int) -> Any:
    return z3.Concat(
        *[
            z3.Select(memory, address + z3.BitVecVal(index, address.size()))
            for index in reversed(range(byte_count))
        ]
    )


def sass_memory_store_little_endian(memory: Any, address: Any, value: Any, byte_count: int) -> Any:
    result = memory
    for index in range(byte_count):
        result = z3.Store(
            result,
            address + z3.BitVecVal(index, address.size()),
            z3.Extract(index * 8 + 7, index * 8, value),
        )
    return result


def sass_uldc64(constant_memory: Any, address: Any) -> Any:
    return sass_memory_load_little_endian(constant_memory, address, 8)


def sass_ldg(global_memory: Any, address: Any, byte_count: int) -> Any:
    return sass_memory_load_little_endian(global_memory, address, byte_count)


def sass_ldg_transition(global_memory: Any, address: Any, byte_count: int) -> tuple[Any, Any]:
    return global_memory, sass_ldg(global_memory, address, byte_count)


def sass_stg(global_memory: Any, address: Any, value: Any, byte_count: int) -> Any:
    return sass_memory_store_little_endian(global_memory, address, value, byte_count)


def sass_ldg32(global_memory: Any, address: Any) -> Any:
    return sass_ldg(global_memory, address, 4)


def sass_ldg32_transition(global_memory: Any, address: Any) -> tuple[Any, Any]:
    return sass_ldg_transition(global_memory, address, 4)


def sass_stg32(global_memory: Any, address: Any, value: Any) -> Any:
    return sass_stg(global_memory, address, value, 4)


def sass_imad_wide_unsigned(first: Any, second: Any, addend: Any, input_width: int) -> Any:
    output_width = input_width * 2
    return z3.Extract(
        output_width - 1,
        0,
        z3.ZeroExt(input_width, first) * z3.ZeroExt(input_width, second) + addend,
    )


def sass_wide_shift_add(base: Any, index: Any, shift: int, index_width: int) -> Any:
    return z3.Extract(
        base.size() - 1,
        0,
        base + (z3.ZeroExt(base.size() - index_width, index) << shift),
    )


def sass_ulea_low(base: Any, index: Any, shift: int, index_width: int) -> Any:
    wide = sass_wide_shift_add(base, index, shift, index_width)
    return z3.Extract(base.size() // 2 - 1, 0, wide)


def sass_ulea_high_without_carry(base: Any, index: Any, shift: int, index_width: int) -> Any:
    wide = sass_wide_shift_add(base, index, shift, index_width)
    return z3.Extract(base.size() - 1, base.size() // 2, wide)


def sass_ldgsts128(global_memory: Any, shared_memory: Any, global_address: Any, shared_address: Any) -> tuple[Any, Any]:
    value = sass_memory_load_little_endian(global_memory, global_address, 16)
    return global_memory, sass_memory_store_little_endian(shared_memory, shared_address, value, 16)


PROPOSED_MEMORY_WIDTHS = {
    "LDG.E": 4,
    "LDG.E.64": 8,
    "LDG.E.LTC128B.128": 16,
    "STG.E.64": 8,
    "STG.E.128": 16,
    "LDGSTS.E.BYPASS.LTC128B.128": 16,
}


PROPOSED_MEMORY_OPERAND_ROLES = {
    "LDG": [["candidate_read", "global"]],
    "STG": [["candidate_write", "global"]],
    "LDGSTS": [["candidate_write", "shared"], ["candidate_read", "global"]],
}


PROPOSED_SEMANTICS_OPCODES = {
    "BRA",
    "BRA.U",
    "EXIT",
    "FADD",
    "FFMA",
    "FMUL",
    "FSEL",
    "IADD3",
    "IMAD",
    "IMAD.IADD",
    "IMAD.MOV",
    "IMAD.MOV.U32",
    "IMAD.SHL.U32",
    "IMAD.U32",
    "IMAD.WIDE.U32",
    "ISETP.EQ.AND",
    "ISETP.GE.AND",
    "ISETP.GE.U32.AND",
    "ISETP.GT.AND",
    "ISETP.GT.U32.AND",
    "ISETP.LT.AND",
    "ISETP.LT.U32.AND",
    "ISETP.NE.AND",
    "LDG",
    "LDG.E",
    "LDG.E.64",
    "LDG.E.LTC128B.128",
    "LDGSTS",
    "LDGSTS.E.BYPASS.LTC128B.128",
    "LOP3.LUT",
    "MOV",
    "NOP",
    "SEL",
    "SHF.R.U32.HI",
    "STG",
    "STG.E.64",
    "STG.E.128",
    "UIMAD.WIDE.U32",
    "ULDC.64",
    "ULEA",
    "ULEA.HI",
}


def _prove(name: str, equality: Any, scope: dict[str, Any]) -> dict[str, Any]:
    solver = z3.Solver()
    solver.add(z3.Not(equality))
    result = solver.check()
    return {
        "name": name,
        "solver_result": str(result),
        "proved": result == z3.unsat,
        "scope": scope,
        "counterexample": str(solver.model()) if result == z3.sat else None,
    }


def build_sass_semantics_certificate() -> dict[str, Any]:
    _require_z3()
    proofs = []
    width = 8
    first = z3.BitVec("sass_first", width)
    second = z3.BitVec("sass_second", width)
    third = z3.BitVec("sass_third", width)
    predicate = z3.Bool("sass_predicate")

    proofs.append(
        _prove(
            "mov_is_identity",
            sass_mov(first) == first,
            {"opcode": "MOV", "input": "all 8-bit values"},
        )
    )
    independent_sum = _ripple_add(_ripple_add(first, second, width), third, width)
    proofs.append(
        _prove(
            "iadd3_matches_ripple_carry_sum",
            sass_iadd3(first, second, third, width) == independent_sum,
            {"opcode": "IADD3", "input": "all triples of 8-bit values", "flags": "carry and predicate outputs excluded"},
        )
    )
    independent_product = z3.Extract(width - 1, 0, _shift_add_multiply(first, second, width))
    independent_imad = _ripple_add(independent_product, third, width)
    proofs.append(
        _prove(
            "imad_matches_shift_add_multiply_accumulate",
            sass_imad(first, second, third, width) == independent_imad,
            {"opcode": "IMAD", "input": "all triples of 8-bit unsigned values", "flags": "base modular form only"},
        )
    )
    proofs.append(
        _prove(
            "lop3_lut_0x96_is_three_input_xor",
            sass_lop3(first, second, third, 0x96, width) == (first ^ second ^ third),
            {"opcode": "LOP3.LUT", "lut": "0x96", "input": "all triples of 8-bit values", "extension": "bitwise and therefore width-independent"},
        )
    )
    majority = (first & second) | (first & third) | (second & third)
    proofs.append(
        _prove(
            "lop3_lut_0xe8_is_three_input_majority",
            sass_lop3(first, second, third, 0xE8, width) == majority,
            {"opcode": "LOP3.LUT", "lut": "0xe8", "input": "all triples of 8-bit values", "extension": "bitwise and therefore width-independent"},
        )
    )
    proofs.append(
        _prove(
            "sel_obeys_predicate",
            z3.And(
                z3.Implies(predicate, sass_sel(first, second, predicate) == first),
                z3.Implies(z3.Not(predicate), sass_sel(first, second, predicate) == second),
            ),
            {"opcode": "SEL", "input": "all pairs of 8-bit values and both predicate values"},
        )
    )
    proofs.append(
        _prove(
            "isetp_ge_u32_matches_unsigned_order",
            sass_isetp_ge_unsigned(first, second) == z3.Not(z3.ULT(first, second)),
            {"opcode": "ISETP.GE.U32", "input": "all pairs of 8-bit values", "extension": "unsigned-order identity is width-independent"},
        )
    )

    comparison_cases = (
        ("eq", False, "ISETP.EQ.AND", first == second),
        ("ne", False, "ISETP.NE.AND", first != second),
        ("ge", False, "ISETP.GE.AND", first >= second),
        ("gt", False, "ISETP.GT.AND", first > second),
        ("lt", False, "ISETP.LT.AND", first < second),
        ("ge", True, "ISETP.GE.U32.AND", z3.UGE(first, second)),
        ("gt", True, "ISETP.GT.U32.AND", z3.UGT(first, second)),
        ("lt", True, "ISETP.LT.U32.AND", z3.ULT(first, second)),
    )
    for relation, unsigned, opcode, direct in comparison_cases:
        proofs.append(
            _prove(
                f"{opcode.lower().replace('.', '_')}_matches_comparison_and_prior_predicate",
                sass_predicate_compare(first, second, relation, unsigned, predicate) == z3.And(direct, predicate),
                {
                    "opcode": opcode,
                    "input": "all pairs of 8-bit values and both prior-predicate values",
                    "boundary": "Predicate output only; condition-code side effects and opcode modifiers are excluded.",
                },
            )
        )

    proofs.append(
        _prove(
            "imad_move_forms_are_identity_encodings",
            z3.And(sass_mov(first) == first, sass_mov(second) == second),
            {
                "opcodes": ["IMAD.MOV", "IMAD.MOV.U32"],
                "input": "all 8-bit source values",
                "boundary": "Proposed move pseudo-form only; encoded source selection and modifiers are premises.",
            },
        )
    )
    shift_amount = z3.BitVec("sass_shift", width)
    shf = sass_shf_right_unsigned_high(first, second, shift_amount, width)
    proofs.append(
        _prove(
            "shf_r_u32_hi_selects_low_at_zero_and_high_at_word_width",
            z3.And(
                z3.Implies(shift_amount == 0, shf == first),
                z3.Implies(shift_amount == width, shf == second),
            ),
            {
                "opcode": "SHF.R.U32.HI",
                "input": "all pairs of 8-bit words at shift values zero and eight",
                "boundary": "Reduced-width funnel-shift model; hardware shift masking and other modifiers are excluded.",
            },
        )
    )
    proofs.append(
        _prove(
            "imad_iadd_and_u32_share_modular_multiply_add_core",
            sass_imad(first, second, third, width) == independent_imad,
            {
                "opcodes": ["IMAD.IADD", "IMAD.U32"],
                "input": "all triples of 8-bit values",
                "boundary": "Core low-word equation only; high-word, carry, sign-extension, and modifiers are excluded.",
            },
        )
    )
    proofs.append(
        _prove(
            "imad_shl_power_of_two_form_matches_left_shift",
            sass_imad(first, z3.BitVecVal(8, width), z3.BitVecVal(0, width), width) == first << 3,
            {
                "opcode": "IMAD.SHL.U32",
                "input": "all 8-bit values for the checked multiply-by-eight form",
                "boundary": "One power-of-two form; arbitrary immediates and modifiers are not proved.",
            },
        )
    )

    float32 = z3.Float32()
    float_first = z3.FP("sass_float_first", float32)
    float_second = z3.FP("sass_float_second", float32)
    float_third = z3.FP("sass_float_third", float32)
    add_forward = sass_fp_add(float_first, float_second)
    add_reverse = sass_fp_add(float_second, float_first)
    multiply_forward = sass_fp_multiply(float_first, float_second)
    multiply_reverse = sass_fp_multiply(float_second, float_first)
    valid_add = z3.And(z3.Not(z3.fpIsNaN(float_first)), z3.Not(z3.fpIsNaN(float_second)), z3.Not(z3.fpIsNaN(add_forward)))
    valid_multiply = z3.And(
        z3.Not(z3.fpIsNaN(float_first)), z3.Not(z3.fpIsNaN(float_second)), z3.Not(z3.fpIsNaN(multiply_forward))
    )
    proofs.append(
        _prove(
            "fadd_rne_is_bit_commutative_for_non_nan_results",
            z3.Implies(valid_add, z3.fpToIEEEBV(add_forward) == z3.fpToIEEEBV(add_reverse)),
            {
                "opcode": "FADD",
                "input": "all non-NaN float32 pairs whose RNE sum is not NaN",
                "boundary": "Abstract IEEE-754 RNE model; FTZ, saturation, and hardware conformance are excluded.",
            },
        )
    )
    proofs.append(
        _prove(
            "fmul_rne_is_bit_commutative_for_non_nan_results",
            z3.Implies(valid_multiply, z3.fpToIEEEBV(multiply_forward) == z3.fpToIEEEBV(multiply_reverse)),
            {
                "opcode": "FMUL",
                "input": "all non-NaN float32 pairs whose RNE product is not NaN",
                "boundary": "Abstract IEEE-754 RNE model; FTZ, saturation, and hardware conformance are excluded.",
            },
        )
    )
    proofs.append(
        _prove(
            "ffma_proposed_semantics_is_single_rounding_ieee_fma",
            sass_fp_fma(float_first, float_second, float_third)
            == z3.fpFMA(z3.RNE(), float_first, float_second, float_third),
            {
                "opcode": "FFMA",
                "input": "all float32 triples",
                "boundary": "Definition check in abstract IEEE-754 RNE theory; modifiers and hardware conformance are excluded.",
            },
        )
    )
    proofs.append(
        _prove(
            "fsel_and_sel_share_predicate_selection_equation",
            z3.And(
                z3.Implies(predicate, sass_sel(first, second, predicate) == first),
                z3.Implies(z3.Not(predicate), sass_sel(first, second, predicate) == second),
            ),
            {"opcodes": ["FSEL", "SEL"], "input": "all abstract operand bit patterns and predicate values"},
        )
    )
    control_state = z3.BitVec("sass_control_state", width)
    branch_target = z3.BitVec("sass_branch_target", width)
    proofs.append(
        _prove(
            "nop_preserves_state_branch_selects_target_and_exit_terminates",
            z3.And(sass_nop(control_state) == control_state, sass_branch(branch_target) == branch_target, sass_exit()),
            {
                "opcodes": ["NOP", "BRA", "BRA.U", "EXIT"],
                "input": "all 8-bit abstract states and branch targets",
                "boundary": "Abstract control-state equations; reconvergence, predication, PC width, and hardware behavior are excluded.",
            },
        )
    )

    address_width = 8
    address = z3.BitVec("sass_memory_address", address_width)
    other_address = z3.BitVec("sass_other_memory_address", address_width)
    constant_memory = z3.Array("sass_constant_memory", z3.BitVecSort(address_width), z3.BitVecSort(8))
    global_memory = z3.Array("sass_global_memory", z3.BitVecSort(address_width), z3.BitVecSort(8))
    shared_memory = z3.Array("sass_shared_memory", z3.BitVecSort(address_width), z3.BitVecSort(8))
    value32 = z3.BitVec("sass_memory_value32", 32)
    explicit_uldc64 = z3.Concat(
        *[
            z3.Select(constant_memory, address + z3.BitVecVal(index, address_width))
            for index in reversed(range(8))
        ]
    )
    proofs.append(
        _prove(
            "uldc64_matches_little_endian_constant_memory_read",
            sass_uldc64(constant_memory, address) == explicit_uldc64,
            {
                "opcode": "ULDC.64",
                "input": "all 8-bit abstract addresses and byte-array constant memories",
                "boundary": "Proposed 64-bit little-endian read only; constant-bank selection, alignment, faults, caching, and hardware behavior excluded.",
            },
        )
    )
    ldg_memory, ldg_value = sass_ldg32_transition(global_memory, address)
    proofs.append(
        _prove(
            "ldg32_reads_little_endian_and_preserves_abstract_global_memory",
            z3.And(ldg_memory == global_memory, ldg_value == sass_memory_load_little_endian(global_memory, address, 4)),
            {
                "opcode": "LDG",
                "input": "all 8-bit abstract addresses and byte-array global memories",
                "boundary": "Proposed 32-bit read transition only; modifiers, alignment, faults, caching, ordering, and hardware behavior excluded.",
            },
        )
    )
    stored_global = sass_stg32(global_memory, address, value32)
    proofs.append(
        _prove(
            "stg32_then_ldg32_at_same_address_returns_stored_bits",
            sass_ldg32(stored_global, address) == value32,
            {
                "opcodes": ["STG", "LDG"],
                "input": "all 8-bit abstract addresses, 32-bit values, and byte-array global memories",
                "boundary": "Sequential proposed little-endian byte-array model; concurrency, alignment, faults, caches, ordering, and hardware behavior excluded.",
            },
        )
    )
    nonoverlap = z3.And(
        *[
            other_address != address + z3.BitVecVal(index, address_width)
            for index in range(4)
        ]
    )
    proofs.append(
        _prove(
            "stg32_preserves_nonoverlapping_abstract_global_byte",
            z3.Implies(nonoverlap, z3.Select(stored_global, other_address) == z3.Select(global_memory, other_address)),
            {
                "opcode": "STG",
                "input": "all 8-bit abstract addresses, nonoverlapping byte addresses, 32-bit values, and global memories",
                "boundary": "Single-threaded proposed byte-array update only; concurrency and hardware behavior excluded.",
            },
        )
    )
    unchanged_global, copied_shared = sass_ldgsts128(global_memory, shared_memory, address, other_address)
    proofs.append(
        _prove(
            "ldgsts128_preserves_global_and_copies_128_bits_to_shared",
            z3.And(
                unchanged_global == global_memory,
                sass_memory_load_little_endian(copied_shared, other_address, 16)
                == sass_memory_load_little_endian(global_memory, address, 16),
            ),
            {
                "opcode": "LDGSTS",
                "input": "all 8-bit abstract global/shared addresses and byte-array memories",
                "boundary": "Proposed sequential 128-bit global-read/shared-write equation; async behavior, predicates, barriers, alignment, faults, ordering, and hardware behavior excluded.",
            },
        )
    )
    for byte_count, load_opcode, store_opcode in (
        (8, "LDG.E.64", "STG.E.64"),
        (16, "LDG.E.LTC128B.128", "STG.E.128"),
    ):
        value = z3.BitVec(f"sass_memory_value_{byte_count * 8}", byte_count * 8)
        preserved_memory, loaded_value = sass_ldg_transition(global_memory, address, byte_count)
        proofs.append(
            _prove(
                f"ldg_{byte_count * 8}_reads_little_endian_and_preserves_global_memory",
                z3.And(
                    preserved_memory == global_memory,
                    loaded_value == sass_memory_load_little_endian(global_memory, address, byte_count),
                ),
                {
                    "opcode": load_opcode,
                    "input": f"all 8-bit abstract addresses and {byte_count * 8}-bit global-memory values",
                    "boundary": "Width-specific proposed read equation; cache/eviction modifiers are uninterpreted premises and alignment, faults, ordering, concurrency, and hardware behavior are excluded.",
                },
            )
        )
        stored = sass_stg(global_memory, address, value, byte_count)
        proofs.append(
            _prove(
                f"stg_{byte_count * 8}_then_matching_ldg_returns_stored_bits",
                sass_ldg(stored, address, byte_count) == value,
                {
                    "opcodes": [store_opcode, load_opcode],
                    "input": f"all 8-bit abstract addresses, {byte_count * 8}-bit values, and global memories",
                    "boundary": "Sequential width-specific proposed byte-array equation; modifiers, alignment, faults, caches, ordering, concurrency, and hardware behavior are excluded.",
                },
            )
        )

    wide_input_width = 8
    wide_first = z3.BitVec("sass_wide_first", wide_input_width)
    wide_second = z3.BitVec("sass_wide_second", wide_input_width)
    wide_addend = z3.BitVec("sass_wide_addend", wide_input_width * 2)
    independent_wide_product = _shift_add_multiply(wide_first, wide_second, wide_input_width)
    independent_wide_sum = _ripple_add(independent_wide_product, wide_addend, wide_input_width * 2)
    proofs.append(
        _prove(
            "imad_wide_unsigned_matches_shift_add_widened_product",
            sass_imad_wide_unsigned(wide_first, wide_second, wide_addend, wide_input_width)
            == independent_wide_sum,
            {
                "opcodes": ["IMAD.WIDE.U32", "UIMAD.WIDE.U32"],
                "input": "all pairs of 8-bit unsigned factors and 16-bit addends",
                "boundary": "Reduced-width proposed unsigned multiply-add equation; signed forms, carry modifiers, register pairing, and hardware behavior excluded.",
            },
        )
    )
    wide_base = z3.BitVec("sass_wide_base", 16)
    wide_index = z3.BitVec("sass_wide_index", 8)
    proofs.append(
        _prove(
            "ulea_separate_low_and_high_results_recompose_wide_shift_add",
            z3.Concat(
                sass_ulea_high_without_carry(wide_base, wide_index, 3, 8),
                sass_ulea_low(wide_base, wide_index, 3, 8),
            )
            == sass_wide_shift_add(wide_base, wide_index, 3, 8),
            {
                "opcodes": ["ULEA", "ULEA.HI"],
                "input": "all 16-bit bases and 8-bit unsigned indices for fixed shift three",
                "boundary": "Reduced-width proposed separate low/high result equations; .X predicate carry, sign extension, arbitrary shifts, register encoding, and hardware behavior excluded.",
            },
        )
    )

    body = {
        "scope": "Proposed bitvector, abstract IEEE-754, control, and byte-array memory semantics for selected opcode forms observed in attested CUDA functions; not NVIDIA-certified SASS semantics.",
        "solver": {"name": "Z3", "version": z3.get_version_string()},
        "proofs": proofs,
        "proved": sum(item["proved"] for item in proofs),
        "total": len(proofs),
        "covered_base_opcodes": sorted(PROPOSED_SEMANTICS_OPCODES),
        "proposed_memory_operand_roles": PROPOSED_MEMORY_OPERAND_ROLES,
        "proposed_memory_widths_bytes": PROPOSED_MEMORY_WIDTHS,
        "excluded": [
            "Opcode modifiers not explicitly named in each obligation",
            "register width and type variants beyond the stated formulas",
            "Memory forms beyond the declared fixed-width abstract byte-array equations",
            "Barrier, warp, reconvergence, and complete control-flow semantics",
            "NVIDIA hardware conformance to these proposed equations",
        ],
    }
    body["certificate_sha256"] = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    return body


def sass_image_coverage(opcode_histogram: dict[str, int]) -> dict[str, Any]:
    exact_covered = PROPOSED_SEMANTICS_OPCODES
    total = sum(opcode_histogram.values())
    covered = sum(count for opcode, count in opcode_histogram.items() if opcode in exact_covered)
    return {
        "total_instruction_lines": total,
        "covered_instruction_lines": covered,
        "coverage_fraction": covered / total if total else 0.0,
        "exact_covered_opcodes_present": sorted(exact_covered.intersection(opcode_histogram)),
        "scope": "Syntactic exact-base-opcode coverage only; no modifier folding or hardware-semantic claim.",
    }


def verify_sass_semantics_certificate(certificate: dict[str, Any]) -> dict[str, Any]:
    stored = {key: value for key, value in certificate.items() if key != "certificate_sha256"}
    integrity_valid = hashlib.sha256(canonical_json(stored).encode("utf-8")).hexdigest() == certificate.get(
        "certificate_sha256"
    )
    recomputed = build_sass_semantics_certificate()
    fields = (
        "scope",
        "solver",
        "proofs",
        "proved",
        "total",
        "covered_base_opcodes",
        "proposed_memory_operand_roles",
        "proposed_memory_widths_bytes",
        "excluded",
    )
    claims_match = all(certificate.get(key) == recomputed[key] for key in fields)
    return {
        "valid": bool(integrity_valid and claims_match and recomputed["proved"] == recomputed["total"]),
        "integrity_valid": integrity_valid,
        "reexecution_claims_match": claims_match,
        "proved": recomputed["proved"],
        "total": recomputed["total"],
    }
