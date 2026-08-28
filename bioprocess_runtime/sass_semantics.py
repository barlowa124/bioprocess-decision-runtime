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

    body = {
        "scope": "Proposed bitvector operational semantics for a small base-opcode subset observed in the extracted CUDA image; not NVIDIA-certified SASS semantics.",
        "solver": {"name": "Z3", "version": z3.get_version_string()},
        "proofs": proofs,
        "proved": sum(item["proved"] for item in proofs),
        "total": len(proofs),
        "covered_base_opcodes": ["MOV", "IADD3", "IMAD", "LOP3.LUT", "SEL", "ISETP.GE.U32"],
        "excluded": [
            "Opcode modifiers not explicitly named in each obligation",
            "register width and type variants beyond the stated formulas",
            "predication, condition-code, carry, memory, barrier, warp, and control-flow semantics",
            "NVIDIA hardware conformance to these proposed equations",
        ],
    }
    body["certificate_sha256"] = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    return body


def sass_image_coverage(opcode_histogram: dict[str, int]) -> dict[str, Any]:
    exact_covered = {"MOV", "IADD3", "IMAD", "LOP3.LUT", "SEL", "ISETP.GE.U32"}
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
    fields = ("scope", "solver", "proofs", "proved", "total", "covered_base_opcodes", "excluded")
    claims_match = all(certificate.get(key) == recomputed[key] for key in fields)
    return {
        "valid": bool(integrity_valid and claims_match and recomputed["proved"] == recomputed["total"]),
        "integrity_valid": integrity_valid,
        "reexecution_claims_match": claims_match,
        "proved": recomputed["proved"],
        "total": recomputed["total"],
    }
