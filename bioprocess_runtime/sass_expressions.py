from __future__ import annotations

import ctypes
import hashlib
import re
import subprocess
from collections import Counter, deque
from pathlib import Path
from typing import Any

from .attention_bounds import verify_attention_logical_bounds_certificate
from .nsight_attestation import (
    _instruction_summary,
    _parse_cuobjdump_sass,
    verify_nsight_launch_certificate,
)
from .sass_memory import (
    _AttentionParams,
    _base_opcode,
    _basic_blocks,
    _branch_target,
    _call_string_address_taint_slices,
    _cfg_successors,
    _constant_offsets,
    _derive_parameter_base,
    _destination_register_count,
    _field_for_offset,
    _registers,
    _source_registers_for_opcode,
)
from .sass_semantics import (
    sass_iadd3,
    sass_imad,
    sass_imad_wide_signed,
    sass_imad_wide_unsigned,
    sass_lop3,
    sass_mov,
    sass_uldc,
)
from .serialization import canonical_json

try:
    import z3
except ModuleNotFoundError:
    z3 = None


EXPRESSION_OPCODE_SEMANTICS = {
    "MOV": ["mov_is_identity"],
    "UMOV": ["uniform_move_shares_proposed_identity_equation"],
    "S2R": [
        "register_transfer_shares_proposed_bit_copy_equation",
        "register_transfer_full_32bit_definition_instance",
    ],
    "S2UR": [
        "register_transfer_shares_proposed_bit_copy_equation",
        "register_transfer_full_32bit_definition_instance",
    ],
    "R2UR": [
        "register_transfer_shares_proposed_bit_copy_equation",
        "register_transfer_full_32bit_definition_instance",
    ],
    "IADD3": ["iadd3_matches_ripple_carry_sum"],
    "UIADD3": [
        "uniform_iadd3_matches_ripple_carry_sum",
        "uniform_iadd3_full_32bit_definition_instance",
    ],
    "IMAD": ["imad_matches_shift_add_multiply_accumulate"],
    "IMAD.IADD": ["imad_iadd_and_u32_share_modular_multiply_add_core"],
    "IMAD.U32": ["imad_iadd_and_u32_share_modular_multiply_add_core"],
    "UIMAD": [
        "uniform_imad_matches_shift_add_multiply_accumulate",
        "uniform_imad_full_32bit_definition_instance",
    ],
    "IMAD.WIDE": [
        "imad_wide_signed_matches_twos_complement_shift_add_product",
        "imad_wide_signed_full_32x32_to_64_definition_instance",
    ],
    "IMAD.WIDE.U32": [
        "imad_wide_unsigned_matches_shift_add_widened_product",
        "imad_wide_unsigned_full_32x32_to_64_definition_instance",
    ],
    "UIMAD.WIDE": [
        "imad_wide_signed_matches_twos_complement_shift_add_product",
        "imad_wide_signed_full_32x32_to_64_definition_instance",
    ],
    "UIMAD.WIDE.U32": [
        "imad_wide_unsigned_matches_shift_add_widened_product",
        "imad_wide_unsigned_full_32x32_to_64_definition_instance",
    ],
    "ULDC": ["uldc_matches_little_endian_constant_memory_read"],
    "ULDC.64": ["uldc64_matches_little_endian_constant_memory_read"],
    "ULDC.U8": ["uldc_u8_matches_little_endian_constant_memory_read"],
}


CONDITIONAL_EXPRESSION_OPCODE_SEMANTICS = {
    ("LOP3.LUT", "0x96"): ["lop3_lut_0x96_is_three_input_xor"],
    ("LOP3.LUT", "0xe8"): ["lop3_lut_0xe8_is_three_input_majority"],
}


UNIFORM_VALUE_OPCODES = {
    "UMOV",
    "UIADD3",
    "UIMAD",
    "UIMAD.WIDE",
    "UIMAD.WIDE.U32",
}


CONSTANT_LOAD_OPCODES = {"ULDC", "ULDC.64", "ULDC.U8"}


TRANSFER_VALUE_OPCODES = {"S2R", "S2UR", "R2UR"}


EXPRESSION_OPCODE_OPERAND_COUNTS = {
    "MOV": 2,
    "UMOV": 2,
    "S2R": 2,
    "S2UR": 2,
    "R2UR": 2,
    "IADD3": 4,
    "UIADD3": 4,
    "IMAD": 4,
    "UIMAD": 4,
    "IMAD.IADD": 4,
    "IMAD.U32": 4,
    "IMAD.WIDE": 4,
    "IMAD.WIDE.U32": 4,
    "UIMAD.WIDE": 4,
    "UIMAD.WIDE.U32": 4,
    "ULDC": 2,
    "ULDC.64": 2,
    "ULDC.U8": 2,
}


def _semantic_requirement(opcode: str, operands: str) -> list[str] | None:
    tokens = [token.strip().lower() for token in operands.split(",")]
    if opcode in EXPRESSION_OPCODE_SEMANTICS:
        shape_matches = (
            len(tokens) == EXPRESSION_OPCODE_OPERAND_COUNTS[opcode]
            and all(tokens)
            and not any(
                re.fullmatch(r"-(?:ur|r)\d+", token) for token in tokens[1:]
            )
        )
        if opcode not in (
            UNIFORM_VALUE_OPCODES | CONSTANT_LOAD_OPCODES | TRANSFER_VALUE_OPCODES
        ):
            shape_matches = bool(shape_matches and re.fullmatch(r"r\d+", tokens[0]))
        if opcode in UNIFORM_VALUE_OPCODES:
            shape_matches = bool(
                shape_matches
                and re.fullmatch(r"ur\d+", tokens[0])
                and all(
                    re.fullmatch(r"(?:ur\d+|urz|-?(?:0x[0-9a-f]+|\d+))", token)
                    for token in tokens[1:]
                )
            )
        if opcode in CONSTANT_LOAD_OPCODES:
            shape_matches = bool(
                shape_matches
                and re.fullmatch(r"ur\d+", tokens[0])
                and re.fullmatch(r"c\[0x0\]\[0x[0-9a-f]+\]", tokens[1])
            )
        if opcode in TRANSFER_VALUE_OPCODES:
            transfer_patterns = {
                "S2R": (r"r\d+", r"sr_[a-z0-9_]+(?:\.[a-z0-9_]+)*"),
                "S2UR": (r"ur\d+", r"sr_[a-z0-9_]+(?:\.[a-z0-9_]+)*"),
                "R2UR": (r"ur\d+", r"r\d+"),
            }
            shape_matches = bool(
                shape_matches
                and re.fullmatch(transfer_patterns[opcode][0], tokens[0])
                and re.fullmatch(transfer_patterns[opcode][1], tokens[1])
            )
        if shape_matches:
            return EXPRESSION_OPCODE_SEMANTICS[opcode]
        return None
    if opcode == "LOP3.LUT" and len(tokens) == 6:
        literal = tokens[4]
        predicate = tokens[5]
        requirement = CONDITIONAL_EXPRESSION_OPCODE_SEMANTICS.get((opcode, literal))
        value_operand = r"(?:u?r\d+|u?rz|-?(?:0x[0-9a-f]+|\d+))"
        if (
            requirement
            and all(re.fullmatch(value_operand, token) for token in tokens[1:4])
            and re.fullmatch(r"!?p(?:t|\d+)", predicate)
        ):
            return requirement
    return None


def _proof_record(record: dict[str, Any]) -> dict[str, Any]:
    body = {
        "name": record["name"],
        "proved": record["proved"],
        "solver_result": record["solver_result"],
        "scope": record["scope"],
    }
    return {
        **body,
        "proof_record_sha256": hashlib.sha256(
            canonical_json(body).encode("utf-8")
        ).hexdigest(),
    }


def _build_expression_semantics_snapshot(
    certificate: dict[str, Any]
) -> dict[str, Any]:
    certificate_body = {
        key: value for key, value in certificate.items() if key != "certificate_sha256"
    }
    if hashlib.sha256(canonical_json(certificate_body).encode("utf-8")).hexdigest() != certificate.get(
        "certificate_sha256"
    ):
        raise ValueError("Invalid SASS semantics certificate hash")
    proof_by_name = {record["name"]: record for record in certificate["proofs"]}
    obligation_names = sorted(
        {
            name
            for names in (
                list(EXPRESSION_OPCODE_SEMANTICS.values())
                + list(CONDITIONAL_EXPRESSION_OPCODE_SEMANTICS.values())
            )
            for name in names
        }
    )
    if any(name not in proof_by_name for name in obligation_names):
        raise ValueError("SASS semantics certificate lacks a required expression obligation")
    if any(
        proof_by_name[name].get("proved") is not True
        or proof_by_name[name].get("solver_result") != "unsat"
        for name in obligation_names
    ):
        raise ValueError("Required expression semantics obligations are not proved")
    obligations = [_proof_record(proof_by_name[name]) for name in obligation_names]
    body = {
        "registry_version": 1,
        "sass_semantics_certificate_sha256": certificate["certificate_sha256"],
        "exact_opcode_obligations": EXPRESSION_OPCODE_SEMANTICS,
        "exact_opcode_operand_counts": EXPRESSION_OPCODE_OPERAND_COUNTS,
        "uniform_value_opcodes": sorted(UNIFORM_VALUE_OPCODES),
        "constant_load_opcodes": sorted(CONSTANT_LOAD_OPCODES),
        "transfer_value_opcodes": sorted(TRANSFER_VALUE_OPCODES),
        "conditional_opcode_obligations": [
            {
                "opcode": opcode,
                "operand_literal": literal,
                "obligation_names": names,
            }
            for (opcode, literal), names in sorted(
                CONDITIONAL_EXPRESSION_OPCODE_SEMANTICS.items()
            )
        ],
        "obligations": obligations,
    }
    body["snapshot_sha256"] = hashlib.sha256(
        canonical_json(body).encode("utf-8")
    ).hexdigest()
    return body


def _verify_expression_semantics_snapshot(snapshot: dict[str, Any]) -> bool:
    body = {key: value for key, value in snapshot.items() if key != "snapshot_sha256"}
    hash_valid = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest() == snapshot.get(
        "snapshot_sha256"
    )
    obligation_by_name = {
        record.get("name"): record for record in snapshot.get("obligations", [])
    }
    expected_names = {
        name
        for names in (
            list(EXPRESSION_OPCODE_SEMANTICS.values())
            + list(CONDITIONAL_EXPRESSION_OPCODE_SEMANTICS.values())
        )
        for name in names
    }
    records_valid = (
        len(snapshot.get("obligations", [])) == len(expected_names)
        and set(obligation_by_name) == expected_names
        and all(
            record.get("proved") is True
            and record.get("solver_result") == "unsat"
            and record.get("proof_record_sha256")
            == hashlib.sha256(
                canonical_json(
                    {
                        key: value
                        for key, value in record.items()
                        if key != "proof_record_sha256"
                    }
                ).encode("utf-8")
            ).hexdigest()
            for record in obligation_by_name.values()
        )
    )
    registry_valid = (
        snapshot.get("exact_opcode_obligations") == EXPRESSION_OPCODE_SEMANTICS
        and snapshot.get("exact_opcode_operand_counts")
        == EXPRESSION_OPCODE_OPERAND_COUNTS
        and snapshot.get("uniform_value_opcodes") == sorted(UNIFORM_VALUE_OPCODES)
        and snapshot.get("constant_load_opcodes") == sorted(CONSTANT_LOAD_OPCODES)
        and snapshot.get("transfer_value_opcodes") == sorted(TRANSFER_VALUE_OPCODES)
        and snapshot.get("conditional_opcode_obligations")
        == [
            {
                "opcode": opcode,
                "operand_literal": literal,
                "obligation_names": names,
            }
            for (opcode, literal), names in sorted(
                CONDITIONAL_EXPRESSION_OPCODE_SEMANTICS.items()
            )
        ]
    )
    return bool(hash_valid and records_valid and registry_valid)


def _instruction_semantic_binding(
    opcode: str, operands: str, snapshot: dict[str, Any] | None
) -> dict[str, Any] | None:
    obligation_names = _semantic_requirement(opcode, operands)
    if not obligation_names or not snapshot:
        return None
    proof_by_name = {
        record["name"]: record for record in snapshot["obligations"]
    }
    return {
        "opcode": opcode,
        "obligation_names": obligation_names,
        "proof_record_sha256": [
            proof_by_name[name]["proof_record_sha256"] for name in obligation_names
        ],
        "sass_semantics_certificate_sha256": snapshot[
            "sass_semantics_certificate_sha256"
        ],
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "binding_scope": "Proposed-semantics proof-record reference only.",
        "proof_premises_established_for_instruction": False,
        "hardware_instruction_semantics_established": False,
    }


SPECIAL_REGISTER_COORDINATES = {
    "SR_TID.X": ("block_size", 0),
    "SR_TID.Y": ("block_size", 1),
    "SR_TID.Z": ("block_size", 2),
    "SR_CTAID.X": ("grid_size", 0),
    "SR_CTAID.Y": ("grid_size", 1),
    "SR_CTAID.Z": ("grid_size", 2),
}


def _parse_launch_dimensions(value: str) -> list[int]:
    match = re.fullmatch(r"\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", value)
    if not match:
        raise ValueError(f"Invalid launch dimensions: {value}")
    dimensions = [int(match.group(index)) for index in range(1, 4)]
    if any(dimension <= 0 for dimension in dimensions):
        raise ValueError("Launch dimensions must be positive")
    return dimensions


def _launch_coordinate_registers(
    block_size: list[int], grid_size: list[int]
) -> dict[str, dict[str, Any]]:
    dimensions = {"block_size": block_size, "grid_size": grid_size}
    return {
        register: {
            "symbolic": True,
            "bit_width": 32,
            "minimum_inclusive": 0,
            "maximum_exclusive": dimensions[source][axis],
            "launch_dimension": source,
            "axis": "XYZ"[axis],
            "coordinate_correspondence_established": False,
            "hardware_acquisition_established": False,
        }
        for register, (source, axis) in SPECIAL_REGISTER_COORDINATES.items()
    }


def _build_launch_coordinate_domains(
    nsight_certificate: dict[str, Any]
) -> dict[str, Any]:
    details = nsight_certificate["details"]
    block_size = _parse_launch_dimensions(details["block_size"])
    grid_size = _parse_launch_dimensions(details["grid_size"])
    try:
        reported_block_volume = int(
            details["metrics"]["Block Size"]["value"].replace(",", "")
        )
        reported_grid_volume = int(
            details["metrics"]["Grid Size"]["value"].replace(",", "")
        )
    except (AttributeError, KeyError, ValueError) as error:
        raise ValueError("Nsight launch-volume metrics are missing or invalid") from error
    body = {
        "scope": "Launch-coordinate domain assumptions derived from retained Nsight grid/block dimensions; SR naming correspondence, runtime values, and hardware acquisition are not established.",
        "nsight_certificate_sha256": nsight_certificate["certificate_sha256"],
        "block_size": block_size,
        "grid_size": grid_size,
        "reported_block_volume": reported_block_volume,
        "reported_grid_volume": reported_grid_volume,
        "dimension_products_match_reported_metrics": block_size[0]
        * block_size[1]
        * block_size[2]
        == reported_block_volume
        and grid_size[0] * grid_size[1] * grid_size[2] == reported_grid_volume,
        "registers": _launch_coordinate_registers(block_size, grid_size),
    }
    body["domain_sha256"] = hashlib.sha256(
        canonical_json(body).encode("utf-8")
    ).hexdigest()
    return body


def _verify_launch_coordinate_domains(domains: dict[str, Any]) -> bool:
    body = {key: value for key, value in domains.items() if key != "domain_sha256"}
    hash_valid = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest() == domains.get(
        "domain_sha256"
    )
    block_size = domains.get("block_size", [])
    grid_size = domains.get("grid_size", [])
    dimensions_valid = (
        len(block_size) == 3
        and len(grid_size) == 3
        and all(isinstance(value, int) and value > 0 for value in block_size + grid_size)
    )
    return bool(
        hash_valid
        and dimensions_valid
        and bool(re.fullmatch(r"[0-9a-f]{64}", domains.get("nsight_certificate_sha256", "")))
        and domains.get("dimension_products_match_reported_metrics") is True
        and block_size[0] * block_size[1] * block_size[2]
        == domains.get("reported_block_volume")
        and grid_size[0] * grid_size[1] * grid_size[2]
        == domains.get("reported_grid_volume")
        and domains.get("registers")
        == _launch_coordinate_registers(block_size, grid_size)
    )


DESIRED_ACCESS = {
    "query_ptr": "candidate_read",
    "key_ptr": "candidate_read",
    "value_ptr": "candidate_read",
    "output_ptr": "candidate_write",
    "output_accum_ptr": "candidate_write",
}


class _NodeRegistry:
    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}

    def add(self, body: dict[str, Any]) -> str:
        identifier = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
        self.nodes.setdefault(identifier, {"node_sha256": identifier, **body})
        return identifier


def _special_registers(value: str) -> list[str]:
    return re.findall(r"\bSR_[A-Z0-9_]+(?:\.[A-Z0-9_]+)*\b", value)


def _immediate_value(token: str) -> int | None:
    try:
        return int(token, 0)
    except ValueError:
        return None


def _ordered_semantic_operands(
    opcode: str,
    operands: str,
    source_node_by_register: dict[str, str],
    special_node_by_register: dict[str, str],
    parameter_base: int,
) -> list[dict[str, Any]]:
    tokens = [token.strip() for token in operands.split(",")][1:]
    predicate_roles = _predicate_operand_roles(opcode, operands)
    descriptors = []
    for index, token in enumerate(tokens):
        normalized = token.upper()
        operand_index = index + 1
        predicate = _numbered_predicate(token)
        if operand_index in predicate_roles["outputs"] and predicate:
            descriptors.append(
                {
                    "kind": "predicate_output",
                    "predicate": predicate,
                    "negated": token.strip().lstrip("@").startswith("!"),
                    "width_bits": 1,
                }
            )
            continue
        if predicate:
            descriptors.append(
                {
                    "kind": "predicate",
                    "predicate": predicate,
                    "negated": token.strip().lstrip("@").startswith("!"),
                    "source_node": source_node_by_register.get(predicate),
                    "width_bits": 1,
                }
            )
            continue
        if normalized.lstrip("-") in {"RZ", "URZ"}:
            descriptors.append(
                {
                    "kind": "zero",
                    "width_bits": 64 if ".WIDE" in opcode and index == len(tokens) - 1 else 32,
                    "text": token,
                }
            )
            continue
        immediate = _immediate_value(token)
        if immediate is not None:
            descriptors.append(
                {
                    "kind": "immediate",
                    "width_bits": (
                        64 if ".WIDE" in opcode and index == len(tokens) - 1 else 32
                    ),
                    "value": immediate
                    & (0xFFFFFFFFFFFFFFFF if ".WIDE" in opcode and index == len(tokens) - 1 else 0xFFFFFFFF),
                    "text": token,
                }
            )
            continue
        constant = re.fullmatch(r"c\[0x0\]\[(0x[0-9a-fA-F]+)\]", token)
        if constant:
            constant_offset = int(constant.group(1), 16)
            parameter_offset = constant_offset - parameter_base
            descriptors.append(
                {
                    "kind": "constant_memory",
                    "constant_offset": constant_offset,
                    "parameter_field": (
                        _field_for_offset(parameter_offset)
                        if 0 <= parameter_offset < ctypes.sizeof(_AttentionParams)
                        else None
                    ),
                    "width_bits": {"ULDC.U8": 8, "ULDC": 32, "ULDC.64": 64}.get(
                        opcode
                    ),
                    "text": token,
                }
            )
            continue
        special = re.fullmatch(r"SR_[A-Z0-9_]+(?:\.[A-Z0-9_]+)*", normalized)
        if special:
            descriptors.append(
                {
                    "kind": "special_register",
                    "register": normalized,
                    "source_node": special_node_by_register.get(normalized),
                    "width_bits": 32,
                }
            )
            continue
        register = re.fullmatch(r"(-?)(UR|R)(\d+)", normalized)
        if register:
            name = f"{register.group(2)}{register.group(3)}"
            if ".WIDE" in opcode and index == len(tokens) - 1:
                high = f"{register.group(2)}{int(register.group(3)) + 1}"
                descriptors.append(
                    {
                        "kind": "register_pair",
                        "registers": [name, high],
                        "source_nodes": [
                            source_node_by_register.get(name),
                            source_node_by_register.get(high),
                        ],
                        "width_bits": 64,
                        "unary": "negate" if register.group(1) else None,
                    }
                )
            else:
                descriptors.append(
                    {
                        "kind": "register",
                        "register": name,
                        "source_node": source_node_by_register.get(name),
                        "width_bits": 32,
                        "unary": "negate" if register.group(1) else None,
                    }
                )
            continue
        descriptors.append({"kind": "unsupported", "text": token})
    return descriptors


def _numbered_predicate(token: str) -> str | None:
    normalized = token.strip().upper().lstrip("@!")
    return normalized if re.fullmatch(r"(?:UP|P)\d+", normalized) else None


def _predicate_operand_roles(opcode: str, operands: str) -> dict[str, list[int]]:
    tokens = [token.strip() for token in operands.split(",")]
    output_indexes = []
    source_indexes = []
    if opcode in {"IADD3", "UIADD3"} and len(tokens) >= 4:
        output_indexes = [
            index
            for index in range(1, len(tokens) - 3)
            if _numbered_predicate(tokens[index])
        ]
    elif opcode in {"LEA", "ULEA"} and len(tokens) >= 4:
        output_indexes = [
            index
            for index in range(1, len(tokens) - 3)
            if _numbered_predicate(tokens[index])
        ]
    if opcode in {"IADD3.X", "UIADD3.X"}:
        source_indexes = [
            index
            for index in range(max(1, len(tokens) - 2), len(tokens))
            if _numbered_predicate(tokens[index])
        ]
    elif opcode in {
        "LEA.HI.X",
        "LEA.HI.X.SX32",
        "ULEA.HI.X",
        "ULEA.HI.X.SX32",
    }:
        source_indexes = [
            len(tokens) - 1
        ] if len(tokens) > 1 and _numbered_predicate(tokens[-1]) else []
    return {"outputs": output_indexes, "sources": source_indexes}


def _instruction_predicate_sources(instruction: dict[str, Any]) -> list[str]:
    roles = _predicate_operand_roles(
        instruction["opcode"], instruction["operands"]
    )
    tokens = [token.strip() for token in instruction["operands"].split(",")]
    sources = [
        _numbered_predicate(tokens[index]) for index in roles["sources"]
    ]
    guard = _numbered_predicate(instruction.get("predicate") or "")
    return sorted({source for source in [*sources, guard] if source})


def _instruction_predicate_outputs(instruction: dict[str, Any]) -> list[str]:
    roles = _predicate_operand_roles(
        instruction["opcode"], instruction["operands"]
    )
    tokens = [token.strip() for token in instruction["operands"].split(",")]
    return [
        predicate
        for index in roles["outputs"]
        if (predicate := _numbered_predicate(tokens[index]))
    ]


NON_DEFINING_BASE_OPCODES = {
    "ST",
    "STG",
    "STS",
    "RED",
    "BRA",
    "CALL",
    "RET",
    "EXIT",
}


def _definition_key(
    block_index: int, call_stack: tuple[int, ...], instruction_index: int, output_index: int
) -> str:
    return hashlib.sha256(
        canonical_json(
            {
                "block_index": block_index,
                "call_stack": list(call_stack),
                "instruction_index": instruction_index,
                "output_index": output_index,
            }
        ).encode("utf-8")
    ).hexdigest()


def _join_definitions(
    left: dict[str, frozenset[str]], right: dict[str, frozenset[str]]
) -> dict[str, frozenset[str]]:
    return {
        register: left.get(register, frozenset()) | right.get(register, frozenset())
        for register in left.keys() | right.keys()
    }


def _transfer_definitions(
    instruction: dict[str, Any],
    instruction_index: int,
    context: tuple[int, tuple[int, ...]],
    state: dict[str, frozenset[str]],
) -> dict[str, frozenset[str]]:
    destination = re.match(r"((?:UR|R)\d+)\b", instruction["operands"])
    if not destination or _base_opcode(instruction["opcode"]) in NON_DEFINING_BASE_OPCODES:
        return state
    output = dict(state)
    destination_register = destination.group(1)
    prefix = "UR" if destination_register.startswith("UR") else "R"
    first = int(destination_register[len(prefix) :])
    block_index, call_stack = context
    register_output_count = _destination_register_count(instruction["opcode"])
    output_names = [
        f"{prefix}{first + output_index}"
        for output_index in range(register_output_count)
    ] + _instruction_predicate_outputs(instruction)
    for output_index, output_name in enumerate(output_names):
        definition = frozenset(
            {_definition_key(block_index, call_stack, instruction_index, output_index)}
        )
        if instruction.get("predicate"):
            definition |= state.get(output_name, frozenset())
        output[output_name] = definition
    return output


def _context_targets(
    instructions: list[dict[str, Any]],
    blocks: list[list[int]],
    instruction_to_block: dict[int, int],
    block_successors: list[list[int]],
    offset_to_block: dict[int, int],
    context: tuple[int, tuple[int, ...]],
    maximum_call_depth: int,
) -> tuple[set[tuple[int, tuple[int, ...]]], bool, bool, int]:
    block_index, call_stack = context
    terminator_index = blocks[block_index][-1]
    terminator = instructions[terminator_index]
    base_opcode = _base_opcode(terminator["opcode"])
    next_block = (
        instruction_to_block[terminator_index + 1]
        if terminator_index + 1 < len(instructions)
        else None
    )
    targets: set[tuple[int, tuple[int, ...]]] = set()
    overflow = False
    unresolved_return = False
    observed_depth = len(call_stack)
    if base_opcode == "CALL":
        target_block = offset_to_block.get(_branch_target(terminator))
        if target_block is not None and next_block is not None:
            if len(call_stack) < maximum_call_depth:
                targets.add((target_block, (*call_stack, next_block)))
                observed_depth = len(call_stack) + 1
            else:
                overflow = True
                targets.add((target_block, (*call_stack[1:], next_block)))
        if terminator.get("predicate") and next_block is not None:
            targets.add((next_block, call_stack))
    elif base_opcode == "RET":
        if call_stack:
            targets.add((call_stack[-1], call_stack[:-1]))
        else:
            unresolved_return = True
        if terminator.get("predicate") and next_block is not None:
            targets.add((next_block, call_stack))
    else:
        targets.update((successor, call_stack) for successor in block_successors[block_index])
    return targets, overflow, unresolved_return, observed_depth


def _call_string_expression_snapshots(
    instructions: list[dict[str, Any]],
    parameter_base: int,
    maximum_call_depth: int = 4,
    semantics_snapshot: dict[str, Any] | None = None,
    launch_coordinate_domains: dict[str, Any] | None = None,
) -> tuple[dict[int, list[dict[str, Any]]], _NodeRegistry, dict[str, Any]]:
    if semantics_snapshot and not _verify_expression_semantics_snapshot(semantics_snapshot):
        raise ValueError("Invalid expression semantics snapshot")
    if launch_coordinate_domains and not _verify_launch_coordinate_domains(
        launch_coordinate_domains
    ):
        raise ValueError("Invalid launch-coordinate domains")
    instruction_successors, _ = _cfg_successors(instructions)
    blocks, instruction_to_block, block_successors = _basic_blocks(
        instructions, instruction_successors
    )
    offset_to_block = {
        instructions[instruction_index]["offset"]: block_index
        for block_index, block in enumerate(blocks)
        for instruction_index in block
    }
    states: dict[tuple[int, tuple[int, ...]], dict[str, frozenset[str]]] = {(0, ()): {}}
    queue = deque([(0, ())])
    queued = {(0, ())}
    iterations = 0
    transitions = 0
    overflow_contexts: set[tuple[int, tuple[int, ...]]] = set()
    unresolved_return_contexts: set[tuple[int, tuple[int, ...]]] = set()
    maximum_observed_depth = 0
    while queue:
        context = queue.popleft()
        queued.remove(context)
        block_index, _ = context
        iterations += 1
        output = states[context]
        for instruction_index in blocks[block_index]:
            output = _transfer_definitions(
                instructions[instruction_index], instruction_index, context, output
            )
        targets, overflow, unresolved_return, observed_depth = _context_targets(
            instructions,
            blocks,
            instruction_to_block,
            block_successors,
            offset_to_block,
            context,
            maximum_call_depth,
        )
        transitions += len(targets)
        if overflow:
            overflow_contexts.add(context)
        if unresolved_return:
            unresolved_return_contexts.add(context)
        maximum_observed_depth = max(maximum_observed_depth, observed_depth)
        for target in targets:
            joined = output if target not in states else _join_definitions(states[target], output)
            if states.get(target) != joined:
                states[target] = joined
                if target not in queued:
                    queue.append(target)
                    queued.add(target)
    definition_specs: dict[str, dict[str, Any]] = {}
    raw_snapshots: dict[int, list[dict[str, Any]]] = {}
    for context, entry_state in sorted(states.items()):
        block_index, call_stack = context
        state = entry_state
        for instruction_index in blocks[block_index]:
            instruction = instructions[instruction_index]
            memories = re.findall(r"\[([^]]+)\]", instruction["operands"])
            if _base_opcode(instruction["opcode"]) in {
                "LDG",
                "STG",
                "LDGSTS",
                "ATOM",
                "ATOMS",
                "RED",
                "LDS",
                "STS",
                "LDSM",
                "LD",
                "ST",
            } and memories:
                raw_snapshots.setdefault(instruction["offset"], []).append(
                    {
                        "block_index": block_index,
                        "call_stack": list(call_stack),
                        "definitions": {
                            register: sorted(
                                state.get(register, {f"entry:{register}"})
                            )
                            for memory in memories
                            for register in _registers(memory)
                        },
                    }
                )
            destination = re.match(r"((?:UR|R)\d+)\b", instruction["operands"])
            if destination and _base_opcode(instruction["opcode"]) not in NON_DEFINING_BASE_OPCODES:
                remaining = instruction["operands"][destination.end() :]
                source_names = [
                    *_source_registers_for_opcode(instruction["opcode"], remaining),
                    *_instruction_predicate_sources(instruction),
                ]
                source_definitions = {
                    name: sorted(state.get(name, {f"entry:{name}"}))
                    for name in sorted(set(source_names))
                }
                destination_register = destination.group(1)
                prefix = "UR" if destination_register.startswith("UR") else "R"
                first = int(destination_register[len(prefix) :])
                register_output_count = _destination_register_count(
                    instruction["opcode"]
                )
                output_names = [
                    f"{prefix}{first + output_index}"
                    for output_index in range(register_output_count)
                ] + _instruction_predicate_outputs(instruction)
                for output_index, output_name in enumerate(output_names):
                    key = _definition_key(
                        block_index, call_stack, instruction_index, output_index
                    )
                    definition_specs[key] = {
                        "block_index": block_index,
                        "call_stack": list(call_stack),
                        "instruction_index": instruction_index,
                        "instruction_offset": instruction["offset"],
                        "opcode": instruction["opcode"],
                        "predicate": instruction.get("predicate"),
                        "output_index": output_index,
                        "output_name": output_name,
                        "output_kind": (
                            "predicate"
                            if output_index >= register_output_count
                            else "register"
                        ),
                        "output_width_bits": (
                            1 if output_index >= register_output_count else 32
                        ),
                        "source_definitions": source_definitions,
                        "special_registers": sorted(set(_special_registers(remaining))),
                    }
            state = _transfer_definitions(instruction, instruction_index, context, state)
    registry = _NodeRegistry()
    materialized: dict[str, str] = {}

    def materialize(key: str, active: frozenset[str] = frozenset()) -> str:
        if key.startswith("entry:"):
            name = key.split(":", 1)[1]
            return registry.add(
                {
                    "kind": "entry_register",
                    "register": name,
                    "value_kind": (
                        "predicate"
                        if re.fullmatch(r"(?:UP|P)\d+", name)
                        else "register"
                    ),
                    "width_bits": (
                        1 if re.fullmatch(r"(?:UP|P)\d+", name) else 32
                    ),
                }
            )
        if key in active:
            spec = definition_specs[key]
            return registry.add(
                {
                    "kind": "cyclic_reaching_definition",
                    "instruction_offset": spec["instruction_offset"],
                    "output_index": spec["output_index"],
                    "output_name": spec["output_name"],
                    "output_kind": spec["output_kind"],
                    "width_bits": spec["output_width_bits"],
                    "block_index": spec["block_index"],
                    "call_stack": spec["call_stack"],
                }
            )
        if key in materialized:
            return materialized[key]
        spec = definition_specs[key]
        source_nodes = []
        source_node_by_register = {}
        for register, definitions in sorted(spec["source_definitions"].items()):
            alternatives = sorted(materialize(item, active | {key}) for item in definitions)
            source_node = (
                alternatives[0]
                if len(alternatives) == 1
                else registry.add(
                    {
                        "kind": "reaching_definition_join",
                        "register": register,
                        "value_kind": (
                            "predicate"
                            if re.fullmatch(r"(?:UP|P)\d+", register)
                            else "register"
                        ),
                        "width_bits": (
                            1
                            if re.fullmatch(r"(?:UP|P)\d+", register)
                            else 32
                        ),
                        "source_nodes": alternatives,
                    }
                )
            )
            source_nodes.append(source_node)
            source_node_by_register[register] = source_node
        special_node_by_register = {
            register: registry.add(
                {
                    "kind": "special_register",
                    "register": register,
                    "launch_domain_assumption": (
                        launch_coordinate_domains.get("registers", {}).get(register)
                        if launch_coordinate_domains
                        else None
                    ),
                }
            )
            for register in spec["special_registers"]
        }
        source_nodes.extend(special_node_by_register.values())
        parameter_fields = []
        for constant_offset in _constant_offsets(instructions[spec["instruction_index"]]):
            parameter_offset = constant_offset - parameter_base
            field = (
                _field_for_offset(parameter_offset)
                if 0 <= parameter_offset < ctypes.sizeof(_AttentionParams)
                else None
            )
            if field:
                parameter_fields.append(field)
        definition_node = registry.add(
            {
                "kind": "instruction_definition",
                "instruction_offset": spec["instruction_offset"],
                "opcode": spec["opcode"],
                "operands": instructions[spec["instruction_index"]]["operands"],
                "predicate": spec["predicate"],
                "block_index": spec["block_index"],
                "call_stack": spec["call_stack"],
                "output_name": spec["output_name"],
                "output_kind": spec["output_kind"],
                "output_width_bits": spec["output_width_bits"],
                "source_registers": [
                    *sorted(spec["source_definitions"]),
                    *spec["special_registers"],
                ],
                "source_nodes": source_nodes,
                "ordered_semantic_operands": _ordered_semantic_operands(
                    spec["opcode"],
                    instructions[spec["instruction_index"]]["operands"],
                    source_node_by_register,
                    special_node_by_register,
                    parameter_base,
                ),
                "parameter_fields": sorted(set(parameter_fields)),
                "semantic_binding": _instruction_semantic_binding(
                    spec["opcode"],
                    instructions[spec["instruction_index"]]["operands"],
                    semantics_snapshot,
                ),
            }
        )
        output_node = registry.add(
            {
                "kind": "instruction_output",
                "definition_node": definition_node,
                "output_index": spec["output_index"],
                "output_name": spec["output_name"],
                "output_kind": spec["output_kind"],
                "width_bits": spec["output_width_bits"],
            }
        )
        materialized[key] = output_node
        return output_node

    snapshots: dict[int, list[dict[str, Any]]] = {}
    for offset, contexts in raw_snapshots.items():
        snapshots[offset] = [
            {
                **context,
                "definitions": {
                    register: sorted(materialize(key) for key in definitions)
                    for register, definitions in context["definitions"].items()
                },
            }
            for context in contexts
        ]
    reachable_blocks = {block_index for block_index, _ in states}
    graph = {
        "maximum_call_depth": maximum_call_depth,
        "maximum_observed_call_depth": maximum_observed_depth,
        "call_context_count": len(states),
        "reachable_basic_block_count": len(reachable_blocks),
        "reachable_instruction_count": sum(len(blocks[index]) for index in reachable_blocks),
        "fixed_point_context_iterations": iterations,
        "context_transition_count": transitions,
        "abstracted_call_overflow_count": len(overflow_contexts),
        "call_overflow_abstraction": "Drop oldest return site and retain the newest at the fixed depth.",
        "unresolved_return_context_count": len(unresolved_return_contexts),
        "ambiguous_reaching_definition_nodes": sum(
            node["kind"] == "reaching_definition_join" for node in registry.nodes.values()
        ),
        "cyclic_reaching_definition_nodes": sum(
            node["kind"] == "cyclic_reaching_definition" for node in registry.nodes.values()
        ),
        "special_register_nodes": sum(
            node["kind"] == "special_register" for node in registry.nodes.values()
        ),
        "launch_domain_bound_special_register_nodes": sum(
            node["kind"] == "special_register"
            and node.get("launch_domain_assumption") is not None
            for node in registry.nodes.values()
        ),
        "predicate_definition_nodes": sum(
            node["kind"] == "instruction_definition"
            and node.get("output_kind") == "predicate"
            for node in registry.nodes.values()
        ),
        "predicate_source_edges": sum(
            bool(re.fullmatch(r"(?:UP|P)\d+", source))
            for node in registry.nodes.values()
            if node["kind"] == "instruction_definition"
            for source in node.get("source_registers", [])
        ),
        "entry_predicate_nodes": sum(
            node["kind"] == "entry_register"
            and node.get("value_kind") == "predicate"
            for node in registry.nodes.values()
        ),
        "predicate_join_nodes": sum(
            node["kind"] == "reaching_definition_join"
            and node.get("value_kind") == "predicate"
            for node in registry.nodes.values()
        ),
    }
    return snapshots, registry, graph


def _reachable_nodes(roots: list[str], registry: _NodeRegistry) -> list[dict[str, Any]]:
    pending = list(roots)
    visited = set()
    while pending:
        identifier = pending.pop()
        if identifier in visited:
            continue
        visited.add(identifier)
        node = registry.nodes[identifier]
        pending.extend(node.get("source_nodes", []))
        if node.get("definition_node"):
            pending.append(node["definition_node"])
    return [registry.nodes[identifier] for identifier in sorted(visited)]


class _FormulaRegistry:
    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}

    def add(self, body: dict[str, Any]) -> str:
        identifier = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
        self.nodes.setdefault(
            identifier, {"formula_node_sha256": identifier, **body}
        )
        return identifier


def _build_partial_symbolic_formula(
    roots: list[str], expression_nodes: list[dict[str, Any]]
) -> dict[str, Any]:
    expression_by_id = {node["node_sha256"]: node for node in expression_nodes}
    registry = _FormulaRegistry()
    lowered: dict[str, str] = {}

    def opaque(node_id: str, reason: str, width_bits: int = 32) -> str:
        return registry.add(
            {
                "kind": "opaque_leaf",
                "expression_node_sha256": node_id,
                "reason": reason,
                "width_bits": width_bits,
            }
        )

    def operand(descriptor: dict[str, Any]) -> str:
        kind = descriptor["kind"]
        if kind == "zero":
            return registry.add(
                {"kind": "bitvector_literal", "width_bits": descriptor["width_bits"], "value": 0}
            )
        if kind == "immediate":
            return registry.add(
                {
                    "kind": "bitvector_literal",
                    "width_bits": descriptor["width_bits"],
                    "value": descriptor["value"],
                }
            )
        if kind == "register":
            source = descriptor.get("source_node")
            if not source:
                return registry.add(
                    {
                        "kind": "opaque_operand",
                        "descriptor": descriptor,
                        "width_bits": descriptor.get("width_bits") or 32,
                    }
                )
            value = lower(source)
            if descriptor.get("unary") == "negate":
                return registry.add(
                    {
                        "kind": "opaque_operation",
                        "opcode": "unmodeled_bvneg",
                        "width_bits": 32,
                        "arguments": [value],
                    }
                )
            return value
        if kind == "register_pair":
            source_nodes = descriptor.get("source_nodes", [])
            if len(source_nodes) != 2 or not all(source_nodes):
                return registry.add(
                    {
                        "kind": "opaque_operand",
                        "descriptor": descriptor,
                        "width_bits": descriptor.get("width_bits") or 32,
                    }
                )
            value = registry.add(
                {
                    "kind": "opaque_operation",
                    "opcode": "unmodeled_register_pair_concatenation",
                    "width_bits": 64,
                    "arguments": [lower(source_nodes[1]), lower(source_nodes[0])],
                }
            )
            if descriptor.get("unary") == "negate":
                return registry.add(
                    {
                        "kind": "opaque_operation",
                        "opcode": "unmodeled_bvneg",
                        "width_bits": 64,
                        "arguments": [value],
                    }
                )
            return value
        if kind == "predicate":
            source = descriptor.get("source_node")
            return (
                lower(source)
                if source
                else registry.add(
                    {
                        "kind": "opaque_operand",
                        "descriptor": descriptor,
                        "width_bits": 1,
                    }
                )
            )
        if kind == "special_register" and descriptor.get("source_node"):
            return lower(descriptor["source_node"])
        return registry.add(
                    {
                        "kind": "opaque_operand",
                        "descriptor": descriptor,
                        "width_bits": descriptor.get("width_bits") or 32,
                    }
                )

    def lower(node_id: str) -> str:
        if node_id in lowered:
            return lowered[node_id]
        node = expression_by_id[node_id]
        kind = node["kind"]
        if kind == "instruction_output":
            definition_id = node["definition_node"]
            definition = expression_by_id[definition_id]
            value = lower(definition_id)
            output_index = node["output_index"]
            value_width = registry.nodes[value].get("width_bits", 0)
            if node.get("output_kind") == "predicate" and value_width == 1:
                result = value
            elif value_width > 32 and output_index * 32 + 31 < value_width:
                result = registry.add(
                    {
                        "kind": "opaque_operation",
                        "expression_node_sha256": node_id,
                        "opcode": "unmodeled_register_pair_projection",
                        "width_bits": 32,
                        "source_width_bits": value_width,
                        "low_bit": output_index * 32,
                        "high_bit": output_index * 32 + 31,
                        "arguments": [value],
                    }
                )
            elif 0 < value_width < 32 and output_index == 0:
                result = registry.add(
                    {
                        "kind": "opaque_operation",
                        "expression_node_sha256": node_id,
                        "opcode": "unmodeled_register_extension",
                        "width_bits": 32,
                        "source_width_bits": value_width,
                        "arguments": [value],
                    }
                )
            elif value_width == 32 and output_index == 0:
                result = value
            else:
                result = opaque(node_id, "unsupported_instruction_output")
        elif kind == "instruction_definition":
            if not node.get("semantic_binding"):
                result = registry.add(
                    {
                        "kind": "opaque_operation",
                        "expression_node_sha256": node_id,
                        "opcode": node.get("opcode"),
                        "width_bits": node.get("output_width_bits")
                        or max(
                            32,
                            _destination_register_count(node.get("opcode", "")) * 32,
                        ),
                        "arguments": [
                            operand(item)
                            for item in node.get("ordered_semantic_operands", [])
                            if item.get("kind") != "predicate_output"
                        ],
                    }
                )
            else:
                descriptors = node.get("ordered_semantic_operands", [])
                opcode = node["opcode"]
                if opcode in CONSTANT_LOAD_OPCODES and len(descriptors) == 1:
                    descriptor = descriptors[0]
                    result = (
                        registry.add(
                            {
                                "kind": "constant_memory_read",
                                "width_bits": descriptor["width_bits"],
                                "constant_offset": descriptor["constant_offset"],
                                "parameter_field": descriptor["parameter_field"],
                            }
                        )
                        if descriptor["kind"] == "constant_memory"
                        else opaque(node_id, "unsupported_constant_memory_operand")
                    )
                elif opcode == "LOP3.LUT" and len(descriptors) == 5:
                    lut = descriptors[3]
                    result = (
                        registry.add(
                            {
                                "kind": "lop3",
                                "width_bits": 32,
                                "lut": lut["value"],
                                "arguments": [operand(item) for item in descriptors[:3]],
                            }
                        )
                        if lut["kind"] == "immediate"
                        else opaque(node_id, "unsupported_lop3_lut_operand")
                    )
                else:
                    arguments = [operand(item) for item in descriptors]
                    if opcode in {"MOV", "UMOV", "S2R", "S2UR", "R2UR"} and len(arguments) == 1:
                        result = registry.add(
                            {"kind": "identity", "width_bits": 32, "arguments": arguments}
                        )
                    elif opcode in {"IADD3", "UIADD3"} and len(arguments) == 3:
                        result = registry.add(
                            {"kind": "bvadd_mod", "width_bits": 32, "arguments": arguments}
                        )
                    elif opcode in {"IMAD", "IMAD.IADD", "IMAD.U32", "UIMAD"} and len(arguments) == 3:
                        result = registry.add(
                            {"kind": "bvmul_add_mod", "width_bits": 32, "arguments": arguments}
                        )
                    elif ".WIDE" in opcode and len(arguments) == 3:
                        result = registry.add(
                            {
                                "kind": "bvmul_add_mod",
                                "width_bits": 64,
                                "factor_width_bits": 32,
                                "factor_signedness": "unsigned" if ".U32" in opcode else "signed",
                                "arguments": arguments,
                            }
                        )
                    else:
                        result = opaque(
                            node_id,
                            f"unsupported_lowering:{opcode}",
                            max(32, _destination_register_count(opcode) * 32),
                        )
        elif kind == "reaching_definition_join":
            result = registry.add(
                {
                    "kind": "opaque_join",
                    "expression_node_sha256": node_id,
                    "width_bits": node.get("width_bits", 32),
                    "arguments": [lower(source) for source in node.get("source_nodes", [])],
                }
            )
        elif kind == "special_register":
            result = registry.add(
                {
                    "kind": "opaque_symbol",
                    "symbol": node["register"],
                    "width_bits": 32,
                    "launch_domain_assumption": node.get("launch_domain_assumption"),
                    "coordinate_correspondence_established": False,
                }
            )
        else:
            result = opaque(node_id, kind, node.get("width_bits", 32))
        lowered[node_id] = result
        return result

    formula_roots = [lower(root) for root in roots]
    nodes = [registry.nodes[identifier] for identifier in sorted(registry.nodes)]
    body = {
        "root_nodes": formula_roots,
        "formula_nodes": nodes,
        "node_count": len(nodes),
        "lowered_operation_node_count": sum(
            node["kind"]
            not in {
                "opaque_leaf",
                "opaque_operand",
                "opaque_symbol",
                "opaque_operation",
                "opaque_join",
                "bitvector_literal",
            }
            for node in nodes
        ),
        "opaque_node_count": sum(
            node["kind"] in {
                "opaque_leaf",
                "opaque_operand",
                "opaque_symbol",
                "opaque_operation",
                "opaque_join",
            }
            for node in nodes
        ),
        "closed_formula": not any(
            node["kind"] in {
                "opaque_leaf",
                "opaque_operand",
                "opaque_symbol",
                "opaque_operation",
                "opaque_join",
            }
            for node in nodes
        ),
    }
    body["formula_sha256"] = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    return body


LOWERED_FORMULA_KINDS = {
    "identity",
    "bvadd_mod",
    "bvmul_add_mod",
    "constant_memory_read",
    "lop3",
}


OPAQUE_FORMULA_KINDS = {
    "opaque_leaf",
    "opaque_operand",
    "opaque_symbol",
    "opaque_operation",
    "opaque_join",
}


def _type_check_partial_formula(formula: dict[str, Any]) -> dict[str, Any]:
    nodes = {node["formula_node_sha256"]: node for node in formula.get("formula_nodes", [])}
    errors = []
    widths: dict[str, int] = {}
    active: set[str] = set()

    def width(node_id: str) -> int:
        if node_id in widths:
            return widths[node_id]
        if node_id not in nodes:
            errors.append(f"missing_node:{node_id}")
            return 0
        if node_id in active:
            errors.append(f"formula_cycle:{node_id}")
            return 0
        active.add(node_id)
        node = nodes[node_id]
        kind = node.get("kind")
        result_width = node.get("width_bits")
        arguments = node.get("arguments", [])
        argument_widths = [width(argument) for argument in arguments]
        if not isinstance(result_width, int) or result_width <= 0:
            errors.append(f"invalid_width:{node_id}")
            result_width = 0
        if kind == "bitvector_literal":
            if arguments or not isinstance(node.get("value"), int) or not 0 <= node.get("value", -1) < 1 << result_width:
                errors.append(f"invalid_literal:{node_id}")
        elif kind == "identity":
            if len(argument_widths) != 1 or argument_widths[0] != result_width:
                errors.append(f"invalid_identity:{node_id}")
        elif kind == "bvadd_mod":
            if len(argument_widths) != 3 or any(item != result_width for item in argument_widths):
                errors.append(f"invalid_add:{node_id}")
        elif kind == "bvmul_add_mod":
            if result_width == 32:
                if len(argument_widths) != 3 or any(item != 32 for item in argument_widths):
                    errors.append(f"invalid_imad32:{node_id}")
            elif result_width == 64:
                if len(argument_widths) != 3 or argument_widths != [32, 32, 64]:
                    errors.append(f"invalid_imad64:{node_id}")
                if node.get("factor_width_bits") != 32:
                    errors.append(f"invalid_factor_width:{node_id}")
                if node.get("factor_signedness") not in {"signed", "unsigned"}:
                    errors.append(f"invalid_signedness:{node_id}")
            else:
                errors.append(f"invalid_imad_width:{node_id}")
        elif kind == "constant_memory_read":
            if arguments or result_width not in {8, 32, 64} or not isinstance(node.get("constant_offset"), int):
                errors.append(f"invalid_constant_read:{node_id}")
        elif kind == "lop3":
            if len(argument_widths) != 3 or any(item != 32 for item in argument_widths):
                errors.append(f"invalid_lop3:{node_id}")
            if node.get("lut") not in {0x96, 0xE8}:
                errors.append(f"invalid_lut:{node_id}")
        elif kind in OPAQUE_FORMULA_KINDS:
            if kind == "opaque_join" and any(item != result_width for item in argument_widths):
                errors.append(f"invalid_opaque_join:{node_id}")
            if kind == "opaque_symbol" and node.get("launch_domain_assumption"):
                domain = node["launch_domain_assumption"]
                minimum = domain.get("minimum_inclusive")
                maximum = domain.get("maximum_exclusive")
                if not (
                    isinstance(minimum, int)
                    and isinstance(maximum, int)
                    and 0 <= minimum < maximum <= 1 << result_width
                ):
                    errors.append(f"invalid_symbol_domain:{node_id}")
        else:
            errors.append(f"unknown_kind:{node_id}:{kind}")
        active.remove(node_id)
        widths[node_id] = result_width
        return result_width

    root_widths = [width(root) for root in formula.get("root_nodes", [])]
    for node_id in sorted(nodes):
        width(node_id)
    if any(root not in nodes for root in formula.get("root_nodes", [])):
        errors.append("missing_root")
    if any(item != 32 for item in root_widths):
        errors.append("invalid_root_width")
    return {
        "well_typed": not errors,
        "errors": sorted(set(errors)),
        "root_widths": root_widths,
        "node_widths_sha256": hashlib.sha256(
            canonical_json(dict(sorted(widths.items()))).encode("utf-8")
        ).hexdigest(),
    }


def _translate_partial_formula_to_z3(
    formula: dict[str, Any]
) -> tuple[list[Any], list[Any]]:
    if z3 is None:
        raise RuntimeError("Install the proof dependencies to translate partial formulas")
    type_check = _type_check_partial_formula(formula)
    if not type_check["well_typed"]:
        raise ValueError("Cannot translate an ill-typed partial formula")
    nodes = {node["formula_node_sha256"]: node for node in formula["formula_nodes"]}
    translated: dict[str, Any] = {}
    symbols: dict[str, Any] = {}
    constrained_symbols: set[str] = set()
    assumptions = []

    def lower(node_id: str) -> Any:
        if node_id in translated:
            return translated[node_id]
        node = nodes[node_id]
        kind = node["kind"]
        width_bits = node["width_bits"]
        arguments = [lower(argument) for argument in node.get("arguments", [])]
        if kind == "bitvector_literal":
            value = z3.BitVecVal(node["value"], width_bits)
        elif kind == "identity":
            value = arguments[0]
        elif kind == "bvadd_mod":
            value = sass_iadd3(arguments[0], arguments[1], arguments[2], width_bits)
        elif kind == "bvmul_add_mod" and width_bits == 32:
            value = sass_imad(arguments[0], arguments[1], arguments[2], 32)
        elif kind == "bvmul_add_mod" and node["factor_signedness"] == "unsigned":
            value = sass_imad_wide_unsigned(arguments[0], arguments[1], arguments[2], 32)
        elif kind == "bvmul_add_mod":
            value = sass_imad_wide_signed(arguments[0], arguments[1], arguments[2], 32)
        elif kind == "constant_memory_read":
            memory = z3.Array(
                f"formula_memory_{node_id}", z3.BitVecSort(32), z3.BitVecSort(8)
            )
            address = z3.BitVecVal(node["constant_offset"] & 0xFFFFFFFF, 32)
            value = sass_uldc(memory, address, width_bits // 8)
        elif kind == "lop3":
            value = sass_lop3(arguments[0], arguments[1], arguments[2], node["lut"], 32)
        else:
            if kind == "opaque_symbol":
                symbol = node["symbol"]
                value = symbols.setdefault(
                    symbol, z3.BitVec(f"formula_symbol_{symbol}", width_bits)
                )
                domain = node.get("launch_domain_assumption")
                if domain and symbol not in constrained_symbols:
                    assumptions.extend(
                        [
                            z3.UGE(value, domain["minimum_inclusive"]),
                            z3.ULT(value, domain["maximum_exclusive"]),
                        ]
                    )
                    constrained_symbols.add(symbol)
            else:
                value = z3.BitVec(f"formula_opaque_{node_id}", width_bits)
        translated[node_id] = value
        return value

    return [lower(root) for root in formula["root_nodes"]], assumptions


def _local_lowering_obligations(formula: dict[str, Any]) -> list[dict[str, Any]]:
    if z3 is None:
        return []
    obligations = []
    for node in formula["formula_nodes"]:
        if node["kind"] not in LOWERED_FORMULA_KINDS:
            continue
        width_bits = node["width_bits"]
        arguments = [
            z3.BitVec(f"local_{node['formula_node_sha256']}_{index}", 32 if node["kind"] == "bvmul_add_mod" and width_bits == 64 and index < 2 else width_bits)
            for index, _ in enumerate(node.get("arguments", []))
        ]
        if node["kind"] == "identity":
            implementation = arguments[0]
            reference = sass_mov(arguments[0])
        elif node["kind"] == "bvadd_mod":
            implementation = z3.Extract(width_bits - 1, 0, arguments[0] + arguments[1] + arguments[2])
            reference = sass_iadd3(arguments[0], arguments[1], arguments[2], width_bits)
        elif node["kind"] == "bvmul_add_mod" and width_bits == 32:
            implementation = z3.Extract(31, 0, arguments[0] * arguments[1] + arguments[2])
            reference = sass_imad(arguments[0], arguments[1], arguments[2], 32)
        elif node["kind"] == "bvmul_add_mod" and node["factor_signedness"] == "unsigned":
            implementation = z3.Extract(
                63,
                0,
                z3.ZeroExt(32, arguments[0]) * z3.ZeroExt(32, arguments[1]) + arguments[2],
            )
            reference = sass_imad_wide_unsigned(arguments[0], arguments[1], arguments[2], 32)
        elif node["kind"] == "bvmul_add_mod":
            implementation = z3.Extract(
                63,
                0,
                z3.SignExt(32, arguments[0]) * z3.SignExt(32, arguments[1]) + arguments[2],
            )
            reference = sass_imad_wide_signed(arguments[0], arguments[1], arguments[2], 32)
        elif node["kind"] == "constant_memory_read":
            memory = z3.Array(
                f"local_memory_{node['formula_node_sha256']}",
                z3.BitVecSort(32),
                z3.BitVecSort(8),
            )
            address = z3.BitVecVal(node["constant_offset"] & 0xFFFFFFFF, 32)
            implementation = sass_uldc(memory, address, width_bits // 8)
            reference_bytes = [
                z3.Select(memory, address + index)
                for index in range(width_bits // 8)
            ]
            reference = (
                reference_bytes[0]
                if len(reference_bytes) == 1
                else z3.Concat(*reversed(reference_bytes))
            )
        else:
            implementation = sass_lop3(arguments[0], arguments[1], arguments[2], node["lut"], 32)
            reference = (
                arguments[0] ^ arguments[1] ^ arguments[2]
                if node["lut"] == 0x96
                else (arguments[0] & arguments[1])
                | (arguments[0] & arguments[2])
                | (arguments[1] & arguments[2])
            )
        solver = z3.Solver()
        solver.add(implementation != reference)
        result = solver.check()
        obligations.append(
            {
                "formula_node_sha256": node["formula_node_sha256"],
                "formula_kind": node["kind"],
                "solver_result": str(result),
                "proved": result == z3.unsat,
                "scope": "Local definitional equivalence to the proposed formula operator only; instruction premises and hardware behavior are excluded.",
            }
        )
    return obligations


def _analyze_partial_symbolic_formula(formula: dict[str, Any]) -> dict[str, Any]:
    type_check = _type_check_partial_formula(formula)
    obligations = _local_lowering_obligations(formula) if type_check["well_typed"] else []
    translated_roots = 0
    assumption_count = 0
    if type_check["well_typed"] and z3 is not None:
        roots, assumptions = _translate_partial_formula_to_z3(formula)
        translated_roots = len(roots)
        assumption_count = len(assumptions)
    body = {
        "type_check": type_check,
        "z3_available": z3 is not None,
        "translated_root_count": translated_roots,
        "launch_domain_assumption_count": assumption_count,
        "local_lowering_obligations": obligations,
        "proved_local_lowering_obligations": sum(item["proved"] for item in obligations),
        "total_local_lowering_obligations": len(obligations),
        "all_local_lowering_obligations_proved": all(
            item["proved"] for item in obligations
        ),
        "local_equivalence_scope": "Definitional correspondence between lowered formula-node operators and proposed equations; not instruction-premise, SASS, or hardware equivalence.",
    }
    body["analysis_sha256"] = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    return body


def _analyze_partial_formula_intervals(formula: dict[str, Any]) -> dict[str, Any]:
    type_check = _type_check_partial_formula(formula)
    if not type_check["well_typed"]:
        raise ValueError("Cannot analyze intervals for an ill-typed partial formula")
    nodes = {node["formula_node_sha256"]: node for node in formula["formula_nodes"]}
    intervals: dict[str, dict[str, Any]] = {}

    def record(
        node_id: str,
        minimum: int,
        maximum: int,
        basis: str,
        bound_depends_on_assumption: bool,
    ) -> dict[str, Any]:
        width_bits = nodes[node_id]["width_bits"]
        maximum_value = (1 << width_bits) - 1
        body = {
            "formula_node_sha256": node_id,
            "width_bits": width_bits,
            "minimum_inclusive": minimum,
            "maximum_inclusive": maximum,
            "basis": basis,
            "bound_depends_on_assumption": bound_depends_on_assumption,
            "exact": minimum == maximum,
            "full_width_unknown": minimum == 0 and maximum == maximum_value,
        }
        return {
            **body,
            "interval_sha256": hashlib.sha256(
                canonical_json(body).encode("utf-8")
            ).hexdigest(),
        }

    def full(node_id: str, basis: str) -> dict[str, Any]:
        return record(
            node_id,
            0,
            (1 << nodes[node_id]["width_bits"]) - 1,
            basis,
            False,
        )

    def analyze(node_id: str) -> dict[str, Any]:
        if node_id in intervals:
            return intervals[node_id]
        node = nodes[node_id]
        kind = node["kind"]
        arguments = [analyze(argument) for argument in node.get("arguments", [])]
        width_bits = node["width_bits"]
        modulus = 1 << width_bits
        bound_depends_on_assumption = any(
            argument["bound_depends_on_assumption"] for argument in arguments
        )
        if kind == "bitvector_literal":
            result = record(node_id, node["value"], node["value"], "literal", False)
        elif kind == "opaque_symbol" and node.get("launch_domain_assumption"):
            domain = node["launch_domain_assumption"]
            result = record(
                node_id,
                domain["minimum_inclusive"],
                domain["maximum_exclusive"] - 1,
                "launch_domain_assumption",
                True,
            )
        elif kind == "identity":
            result = record(
                node_id,
                arguments[0]["minimum_inclusive"],
                arguments[0]["maximum_inclusive"],
                "identity",
                bound_depends_on_assumption,
            )
        elif kind == "opaque_join" and arguments:
            result = record(
                node_id,
                min(argument["minimum_inclusive"] for argument in arguments),
                max(argument["maximum_inclusive"] for argument in arguments),
                "reaching_definition_join_hull",
                bound_depends_on_assumption,
            )
        elif kind == "bvadd_mod":
            if all(argument["exact"] for argument in arguments):
                value = sum(argument["minimum_inclusive"] for argument in arguments) % modulus
                result = record(node_id, value, value, "exact_modular_addition", bound_depends_on_assumption)
            else:
                minimum = sum(argument["minimum_inclusive"] for argument in arguments)
                maximum = sum(argument["maximum_inclusive"] for argument in arguments)
                result = (
                    record(
                        node_id,
                        minimum,
                        maximum,
                        "non_wrapping_modular_addition",
                        bound_depends_on_assumption,
                    )
                    if maximum < modulus
                    else full(node_id, "addition_wrap_or_unknown")
                )
        elif kind == "bvmul_add_mod":
            if all(argument["exact"] for argument in arguments):
                factors = [argument["minimum_inclusive"] for argument in arguments[:2]]
                if width_bits == 64 and node.get("factor_signedness") == "signed":
                    factors = [
                        value - (1 << 32) if value >= 1 << 31 else value
                        for value in factors
                    ]
                value = (factors[0] * factors[1] + arguments[2]["minimum_inclusive"]) % modulus
                result = record(node_id, value, value, "exact_modular_multiply_add", bound_depends_on_assumption)
            else:
                signed_nonnegative = not (
                    width_bits == 64
                    and node.get("factor_signedness") == "signed"
                    and any(argument["maximum_inclusive"] >= 1 << 31 for argument in arguments[:2])
                )
                minimum = (
                    arguments[0]["minimum_inclusive"]
                    * arguments[1]["minimum_inclusive"]
                    + arguments[2]["minimum_inclusive"]
                )
                maximum = (
                    arguments[0]["maximum_inclusive"]
                    * arguments[1]["maximum_inclusive"]
                    + arguments[2]["maximum_inclusive"]
                )
                result = (
                    record(
                        node_id,
                        minimum,
                        maximum,
                        "non_wrapping_modular_multiply_add",
                        bound_depends_on_assumption,
                    )
                    if signed_nonnegative and maximum < modulus
                    else full(node_id, "multiply_add_wrap_signed_or_unknown")
                )
        elif kind == "lop3" and all(argument["exact"] for argument in arguments):
            first, second, third = (
                argument["minimum_inclusive"] for argument in arguments
            )
            value = (
                first ^ second ^ third
                if node["lut"] == 0x96
                else (first & second) | (first & third) | (second & third)
            )
            result = record(node_id, value, value, "exact_lop3", bound_depends_on_assumption)
        else:
            result = full(node_id, f"unbounded_{kind}")
        intervals[node_id] = result
        return result

    root_intervals = [analyze(root) for root in formula["root_nodes"]]
    for node_id in sorted(nodes):
        analyze(node_id)
    records = [intervals[node_id] for node_id in sorted(intervals)]
    body = {
        "scope": "Unsigned conservative intervals over proposed partial formula operators; launch-coordinate ranges are assumptions and opaque values remain full-width.",
        "intervals": records,
        "root_intervals": root_intervals,
        "node_count": len(records),
        "bounded_node_count": sum(not item["full_width_unknown"] for item in records),
        "exact_node_count": sum(item["exact"] for item in records),
        "assumption_bounded_node_count": sum(
            not item["full_width_unknown"] and item["bound_depends_on_assumption"]
            for item in records
        ),
        "all_nodes_analyzed": len(records) == len(nodes),
        "all_root_intervals_full_width_unknown": all(
            item["full_width_unknown"] for item in root_intervals
        ),
        "assumption_conditioned_nontrivial_root_interval_count": sum(
            not item["full_width_unknown"] and item["bound_depends_on_assumption"]
            for item in root_intervals
        ),
        "effective_address_bounds_established": False,
        "hardware_semantics_established": False,
    }
    body["analysis_sha256"] = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    return body


def _verify_interval_record(record: dict[str, Any]) -> bool:
    body = {key: value for key, value in record.items() if key != "interval_sha256"}
    width_bits = record.get("width_bits")
    minimum = record.get("minimum_inclusive")
    maximum = record.get("maximum_inclusive")
    return bool(
        isinstance(width_bits, int)
        and width_bits > 0
        and isinstance(minimum, int)
        and isinstance(maximum, int)
        and 0 <= minimum <= maximum < 1 << width_bits
        and record.get("exact") == (minimum == maximum)
        and record.get("full_width_unknown")
        == (minimum == 0 and maximum == (1 << width_bits) - 1)
        and isinstance(record.get("bound_depends_on_assumption"), bool)
        and hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
        == record.get("interval_sha256")
    )


def _analyze_partial_formula_blockers(formula: dict[str, Any]) -> dict[str, Any]:
    nodes = {node["formula_node_sha256"]: node for node in formula["formula_nodes"]}

    def label(node: dict[str, Any]) -> str:
        if node["kind"] == "opaque_operation":
            return node.get("opcode", "opaque_operation")
        if node["kind"] == "opaque_symbol":
            return f"opaque_symbol:{node.get('symbol')}"
        if node["kind"] == "opaque_leaf":
            return f"opaque_leaf:{node.get('reason')}"
        if node["kind"] == "opaque_operand":
            return f"opaque_operand:{node.get('descriptor', {}).get('kind')}"
        return node["kind"]

    frontier = []
    for root_index, root in enumerate(formula["root_nodes"]):
        queue = deque([(root, 0)])
        visited = set()
        while queue:
            node_id, depth = queue.popleft()
            if node_id in visited:
                continue
            visited.add(node_id)
            node = nodes[node_id]
            if node["kind"] in OPAQUE_FORMULA_KINDS:
                body = {
                    "root_index": root_index,
                    "root_formula_node_sha256": root,
                    "formula_node_sha256": node_id,
                    "depth": depth,
                    "kind": node["kind"],
                    "label": label(node),
                    "expression_node_sha256": node.get("expression_node_sha256"),
                    "width_bits": node["width_bits"],
                    "argument_count": len(node.get("arguments", [])),
                }
                frontier.append(
                    {
                        **body,
                        "blocker_sha256": hashlib.sha256(
                            canonical_json(body).encode("utf-8")
                        ).hexdigest(),
                    }
                )
                continue
            queue.extend(
                (argument, depth + 1) for argument in node.get("arguments", [])
            )
    opaque_nodes = [
        node for node in formula["formula_nodes"] if node["kind"] in OPAQUE_FORMULA_KINDS
    ]
    histogram = Counter(label(node) for node in opaque_nodes)
    body = {
        "scope": "First opaque node on each traversed root-to-leaf path plus a full opaque-node label histogram; this prioritizes missing proposed semantics but establishes none.",
        "root_frontier": frontier,
        "root_count": len(formula["root_nodes"]),
        "roots_with_blockers": len({item["root_index"] for item in frontier}),
        "root_frontier_blocker_count": len(frontier),
        "all_opaque_node_count": len(opaque_nodes),
        "all_opaque_label_histogram": dict(sorted(histogram.items())),
        "all_roots_blocked": bool(formula["root_nodes"])
        and len({item["root_index"] for item in frontier})
        == len(formula["root_nodes"]),
        "effective_address_formula_closed": False,
        "hardware_semantics_established": False,
    }
    body["analysis_sha256"] = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    return body


def _verify_blocker_record(record: dict[str, Any]) -> bool:
    body = {key: value for key, value in record.items() if key != "blocker_sha256"}
    return bool(
        isinstance(record.get("root_index"), int)
        and record["root_index"] >= 0
        and isinstance(record.get("depth"), int)
        and record["depth"] >= 0
        and isinstance(record.get("width_bits"), int)
        and record["width_bits"] > 0
        and isinstance(record.get("argument_count"), int)
        and record["argument_count"] >= 0
        and record.get("kind") in OPAQUE_FORMULA_KINDS
        and bool(re.fullmatch(r"[0-9a-f]{64}", record.get("formula_node_sha256", "")))
        and bool(
            re.fullmatch(
                r"[0-9a-f]{64}", record.get("root_formula_node_sha256", "")
            )
        )
        and hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
        == record.get("blocker_sha256")
    )


def _unsupported_expression_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        node
        for node in nodes
        if node["kind"] in {
            "entry_register",
            "special_register",
            "reaching_definition_join",
            "cyclic_reaching_definition",
        }
        or (
            node["kind"] == "instruction_definition"
            and node.get("semantic_binding") is None
        )
    ]


def build_sass_expression_certificate(
    cuobjdump: str,
    cubin: Path,
    kernel: str,
    sass_memory_certificate: dict[str, Any],
    logical_bounds_certificate: dict[str, Any],
    sass_semantics_certificate: dict[str, Any],
    nsight_certificate: dict[str, Any],
) -> dict[str, Any]:
    completed = subprocess.run(
        [cuobjdump, "--dump-sass", "--function", kernel, str(cubin)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr or completed.stdout)
    instructions = _parse_cuobjdump_sass(completed.stdout)
    summary = _instruction_summary(instructions)
    parameter_base = _derive_parameter_base(instructions)["base_constant_offset"]
    semantics_snapshot = _build_expression_semantics_snapshot(
        sass_semantics_certificate
    )
    launch_coordinate_domains = _build_launch_coordinate_domains(nsight_certificate)
    slices, call_graph = _call_string_address_taint_slices(instructions, parameter_base)
    snapshots, registry, expression_graph = _call_string_expression_snapshots(
        instructions,
        parameter_base,
        semantics_snapshot=semantics_snapshot,
        launch_coordinate_domains=launch_coordinate_domains,
    )
    selections = []
    for field, access_class in DESIRED_ACCESS.items():
        candidates = [
            (memory_slice, operand)
            for memory_slice in slices
            for operand in memory_slice["address_operands"]
            if access_class == operand["access_class"] and field in operand["source_parameter_fields"]
        ]
        if not candidates:
            selections.append({"field": field, "available": False, "reason": "No matching bounded call-string operand"})
            continue
        selection = None
        for memory_slice, operand in sorted(
            candidates,
            key=lambda item: (
                len(item[1]["source_parameter_fields"]),
                item[0]["instruction_offset"],
            ),
        ):
            contexts = snapshots[memory_slice["instruction_offset"]]
            for context in contexts:
                roots = []
                for register in operand["address_registers"]:
                    alternatives = context["definitions"].get(register, [])
                    roots.append(
                        alternatives[0]
                        if len(alternatives) == 1
                        else registry.add(
                            {
                                "kind": "reaching_definition_join",
                                "register": register,
                                "source_nodes": alternatives,
                            }
                        )
                    )
                nodes = _reachable_nodes(roots, registry)
                represented_fields = sorted(
                    {
                        parameter_field
                        for node in nodes
                        for parameter_field in node.get("parameter_fields", [])
                    }
                )
                unsupported_nodes = _unsupported_expression_nodes(nodes)
                instruction_nodes = [
                    node for node in nodes if node["kind"] == "instruction_definition"
                ]
                proof_backed_nodes = [
                    node for node in instruction_nodes if node.get("semantic_binding")
                ]
                unmodeled_opcodes = Counter(
                    node["opcode"]
                    for node in instruction_nodes
                    if not node.get("semantic_binding")
                )
                partial_formula = _build_partial_symbolic_formula(roots, nodes)
                partial_formula_analysis = _analyze_partial_symbolic_formula(partial_formula)
                partial_formula_interval_analysis = _analyze_partial_formula_intervals(
                    partial_formula
                )
                partial_formula_blocker_analysis = _analyze_partial_formula_blockers(
                    partial_formula
                )
                candidate_selection = {
                    "field": field,
                    "available": True,
                    "instruction_offset": memory_slice["instruction_offset"],
                    "memory_opcode": memory_slice["opcode"],
                    "access_class": operand["access_class"],
                    "memory_space": operand["memory_space"],
                    "address_registers": operand["address_registers"],
                    "available_call_context_count": len(contexts),
                    "selection_criterion": "Fewest aggregate source parameter fields, then lowest instruction offset, then first bounded call context containing the target field.",
                    "selected_block_index": context["block_index"],
                    "selected_call_stack": context["call_stack"],
                    "root_nodes": roots,
                    "node_count": len(nodes),
                    "ambiguous_reaching_definition_node_count": sum(
                        node["kind"] == "reaching_definition_join" for node in nodes
                    ),
                    "cyclic_reaching_definition_node_count": sum(
                        node["kind"] == "cyclic_reaching_definition" for node in nodes
                    ),
                    "special_register_leaf_count": sum(
                        node["kind"] == "special_register" for node in nodes
                    ),
                    "predicate_definition_node_count": sum(
                        node["kind"] == "instruction_definition"
                        and node.get("output_kind") == "predicate"
                        for node in nodes
                    ),
                    "predicate_source_edge_count": sum(
                        bool(re.fullmatch(r"(?:UP|P)\d+", source))
                        for node in nodes
                        if node["kind"] == "instruction_definition"
                        for source in node.get("source_registers", [])
                    ),
                    "entry_predicate_leaf_count": sum(
                        node["kind"] == "entry_register"
                        and node.get("value_kind") == "predicate"
                        for node in nodes
                    ),
                    "predicate_join_node_count": sum(
                        node["kind"] == "reaching_definition_join"
                        and node.get("value_kind") == "predicate"
                        for node in nodes
                    ),
                    "coordinate_special_register_leaf_count": sum(
                        node["kind"] == "special_register"
                        and node.get("register") in SPECIAL_REGISTER_COORDINATES
                        for node in nodes
                    ),
                    "launch_domain_bound_special_register_leaf_count": sum(
                        node["kind"] == "special_register"
                        and node.get("launch_domain_assumption") is not None
                        for node in nodes
                    ),
                    "instruction_definition_node_count": len(instruction_nodes),
                    "proof_backed_instruction_node_count": len(proof_backed_nodes),
                    "unmodeled_instruction_node_count": len(instruction_nodes)
                    - len(proof_backed_nodes),
                    "unmodeled_opcode_histogram": dict(sorted(unmodeled_opcodes.items())),
                    "referenced_semantics_obligations": sorted(
                        {
                            name
                            for node in proof_backed_nodes
                            for name in node["semantic_binding"]["obligation_names"]
                        }
                    ),
                    "represented_parameter_fields": represented_fields,
                    "target_field_represented": field in represented_fields,
                    "unsupported_or_entry_node_count": len(unsupported_nodes),
                    "expression_nodes": nodes,
                    "partial_symbolic_formula": partial_formula,
                    "partial_formula_analysis": partial_formula_analysis,
                    "partial_formula_interval_analysis": partial_formula_interval_analysis,
                    "partial_formula_blocker_analysis": partial_formula_blocker_analysis,
                    "partial_formula_root_frontier_blocker_count": partial_formula_blocker_analysis[
                        "root_frontier_blocker_count"
                    ],
                    "partial_formula_all_roots_blocked": partial_formula_blocker_analysis[
                        "all_roots_blocked"
                    ],
                    "partial_formula_bounded_interval_node_count": partial_formula_interval_analysis[
                        "bounded_node_count"
                    ],
                    "partial_formula_exact_interval_node_count": partial_formula_interval_analysis[
                        "exact_node_count"
                    ],
                    "partial_formula_assumption_bounded_interval_node_count": partial_formula_interval_analysis[
                        "assumption_bounded_node_count"
                    ],
                    "partial_formula_well_typed": partial_formula_analysis["type_check"][
                        "well_typed"
                    ],
                    "partial_formula_local_lowering_obligation_count": partial_formula_analysis[
                        "total_local_lowering_obligations"
                    ],
                    "partial_formula_lowered_operation_node_count": partial_formula[
                        "lowered_operation_node_count"
                    ],
                    "partial_formula_opaque_node_count": partial_formula[
                        "opaque_node_count"
                    ],
                    "closed_supported_formula": not unsupported_nodes
                    and partial_formula["closed_formula"],
                }
                selection = candidate_selection
                if candidate_selection["target_field_represented"]:
                    break
            if selection and selection["target_field_represented"]:
                break
        selections.append(selection)
    expression_graph["ambiguous_reaching_definition_nodes"] = sum(
        node["kind"] == "reaching_definition_join" for node in registry.nodes.values()
    )
    expression_graph["cyclic_reaching_definition_nodes"] = sum(
        node["kind"] == "cyclic_reaching_definition" for node in registry.nodes.values()
    )
    expression_graph["special_register_nodes"] = sum(
        node["kind"] == "special_register" for node in registry.nodes.values()
    )
    expression_graph["launch_domain_bound_special_register_nodes"] = sum(
        node["kind"] == "special_register"
        and node.get("launch_domain_assumption") is not None
        for node in registry.nodes.values()
    )
    checks = {
        "cubin_hash_matches": hashlib.sha256(cubin.read_bytes()).hexdigest()
        == sass_memory_certificate["cubin_sha256"],
        "sass_hash_matches": summary["canonical_sha256"]
        == sass_memory_certificate["sass"]["canonical_sha256"],
        "kernel_matches": kernel == sass_memory_certificate["kernel_name"],
        "sass_memory_certificate_hash_valid": hashlib.sha256(
            canonical_json(
                {
                    key: value
                    for key, value in sass_memory_certificate.items()
                    if key != "certificate_sha256"
                }
            ).encode("utf-8")
        ).hexdigest()
        == sass_memory_certificate.get("certificate_sha256"),
        "sass_memory_certificate_checks_pass": sass_memory_certificate.get("all_checks_pass")
        is True
        and all(sass_memory_certificate.get("checks", {}).values()),
        "sass_semantics_certificate_hash_valid": hashlib.sha256(
            canonical_json(
                {
                    key: value
                    for key, value in sass_semantics_certificate.items()
                    if key != "certificate_sha256"
                }
            ).encode("utf-8")
        ).hexdigest()
        == sass_semantics_certificate.get("certificate_sha256"),
        "sass_semantics_certificate_claims_all_proved": sass_semantics_certificate.get("proved")
        == sass_semantics_certificate.get("total")
        == len(sass_semantics_certificate.get("proofs", []))
        and all(
            proof.get("proved") is True and proof.get("solver_result") == "unsat"
            for proof in sass_semantics_certificate.get("proofs", [])
        ),
        "sass_semantics_hash_matches_sass_memory": sass_semantics_certificate.get(
            "certificate_sha256"
        )
        == sass_memory_certificate.get("sass_semantics_certificate_sha256"),
        "expression_semantics_snapshot_valid": _verify_expression_semantics_snapshot(
            semantics_snapshot
        ),
        "nsight_certificate_valid": verify_nsight_launch_certificate(
            nsight_certificate
        )["valid"],
        "nsight_hash_matches_sass_memory": nsight_certificate.get("certificate_sha256")
        == sass_memory_certificate.get("nsight_certificate_sha256"),
        "launch_coordinate_domains_valid": _verify_launch_coordinate_domains(
            launch_coordinate_domains
        ),
        "all_observed_coordinate_special_registers_have_launch_domains": all(
            node.get("launch_domain_assumption") is not None
            for selection in selections
            for node in selection.get("expression_nodes", [])
            if node.get("kind") == "special_register"
            and node.get("register") in SPECIAL_REGISTER_COORDINATES
        ),
        "logical_bounds_certificate_valid": verify_attention_logical_bounds_certificate(
            logical_bounds_certificate
        )["valid"],
        "expression_contexts_match_taint_contexts": expression_graph["call_context_count"]
        == call_graph["call_context_count"],
        "expression_call_bound_matches_taint_bound": expression_graph["maximum_call_depth"]
        == call_graph["maximum_call_depth"],
        "all_target_skeletons_available": all(selection["available"] for selection in selections),
        "all_target_fields_represented": all(
            selection.get("target_field_represented", False) for selection in selections
        ),
        "all_partial_formulas_well_typed": all(
            selection.get("partial_formula_well_typed") is True
            for selection in selections
        ),
        "all_local_lowering_obligations_proved": all(
            selection.get("partial_formula_analysis", {}).get(
                "all_local_lowering_obligations_proved"
            )
            is True
            for selection in selections
        ),
        "all_partial_formula_roots_translated": all(
            selection.get("partial_formula_analysis", {}).get("translated_root_count")
            == len(selection.get("partial_symbolic_formula", {}).get("root_nodes", []))
            for selection in selections
        ),
        "all_partial_formula_intervals_analyzed": all(
            selection.get("partial_formula_interval_analysis", {}).get(
                "all_nodes_analyzed"
            )
            is True
            for selection in selections
        ),
        "interval_evidence_boundaries_preserved": all(
            selection.get("partial_formula_interval_analysis", {}).get(
                "effective_address_bounds_established"
            )
            is False
            and selection.get("partial_formula_interval_analysis", {}).get(
                "hardware_semantics_established"
            )
            is False
            for selection in selections
        ),
        "all_root_blocker_frontiers_recorded": all(
            selection.get("partial_formula_blocker_analysis", {}).get(
                "all_roots_blocked"
            )
            is True
            for selection in selections
        ),
        "blocker_evidence_boundaries_preserved": all(
            selection.get("partial_formula_blocker_analysis", {}).get(
                "effective_address_formula_closed"
            )
            is False
            and selection.get("partial_formula_blocker_analysis", {}).get(
                "hardware_semantics_established"
            )
            is False
            for selection in selections
        ),
        "all_materialized_predicate_sources_resolved": all(
            descriptor.get("source_node") is not None
            and any(
                candidate.get("node_sha256") == descriptor.get("source_node")
                and candidate.get("kind") != "entry_register"
                for candidate in selection.get("expression_nodes", [])
            )
            for selection in selections
            for node in selection.get("expression_nodes", [])
            if node.get("kind") == "instruction_definition"
            for descriptor in node.get("ordered_semantic_operands", [])
            if descriptor.get("kind") == "predicate"
        ),
        "predicate_producer_consumer_links_present": any(
            selection.get("predicate_definition_node_count", 0) > 0
            and selection.get("predicate_source_edge_count", 0) > 0
            for selection in selections
        ),
    }
    body = {
        "scope": "One hash-consed bounded-call-string reaching-definition address-expression DAG selected per target field; selected instruction nodes bind exact opcode and retained operand text to proved records in the proposed SASS-semantics certificate. Typed partial formulas retain conservative unsigned interval records, with launch dimensions used only as symbolic assumptions. SR naming correspondence, concrete values, acquisition semantics, hardware behavior, closed formulas, effective-address bounds, and logical correspondence are not established. Ambiguous joins, cyclic definitions, unsupported operations, and entry registers remain explicit.",
        "kernel_name": kernel,
        "cubin_sha256": hashlib.sha256(cubin.read_bytes()).hexdigest(),
        "sass_canonical_sha256": summary["canonical_sha256"],
        "sass_memory_certificate_sha256": sass_memory_certificate["certificate_sha256"],
        "nsight_certificate_sha256": nsight_certificate["certificate_sha256"],
        "launch_coordinate_domains": launch_coordinate_domains,
        "sass_semantics_certificate_sha256": sass_semantics_certificate[
            "certificate_sha256"
        ],
        "expression_opcode_semantics": semantics_snapshot,
        "logical_bounds_certificate_sha256": logical_bounds_certificate["certificate_sha256"],
        "call_string_context_summary": call_graph,
        "expression_reaching_definition_summary": expression_graph,
        "selections": selections,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "selected_sass_address_expression_dags_established": all(
            selection.get("available") and selection.get("target_field_represented")
            for selection in selections
        ),
        "bounded_call_string_expression_reaching_definitions_established": True,
        "bounded_predicate_reaching_definitions_established": True,
        "predicate_producer_consumer_dependencies_established": checks[
            "all_materialized_predicate_sources_resolved"
        ]
        and checks["predicate_producer_consumer_links_present"],
        "predicate_values_established": False,
        "predicate_carry_equations_established": False,
        "predicate_hardware_semantics_established": False,
        "partial_proposed_symbolic_formulas_established": all(
            bool(selection.get("partial_symbolic_formula", {}).get("formula_nodes"))
            for selection in selections
        ),
        "partial_formula_ordered_operands_preserved": True,
        "partial_formula_well_typed": all(
            selection.get("partial_formula_well_typed") is True
            for selection in selections
        ),
        "typed_z3_translation_established": all(
            selection.get("partial_formula_analysis", {}).get("translated_root_count")
            == len(selection.get("partial_symbolic_formula", {}).get("root_nodes", []))
            for selection in selections
        ),
        "local_proposed_operator_lowering_equivalence_established": all(
            selection.get("partial_formula_analysis", {}).get(
                "all_local_lowering_obligations_proved"
            )
            is True
            for selection in selections
        ),
        "partial_formula_interval_analysis_established": all(
            selection.get("partial_formula_interval_analysis", {}).get(
                "all_nodes_analyzed"
            )
            is True
            for selection in selections
        ),
        "all_selected_root_intervals_full_width_unknown": all(
            selection.get("partial_formula_interval_analysis", {}).get(
                "all_root_intervals_full_width_unknown"
            )
            is True
            for selection in selections
        ),
        "assumption_conditioned_effective_address_bounds_established": False,
        "root_opaque_blocker_frontiers_established": all(
            selection.get("partial_formula_blocker_analysis", {}).get(
                "all_roots_blocked"
            )
            is True
            for selection in selections
        ),
        "all_selected_roots_have_opaque_blockers": all(
            selection.get("partial_formula_all_roots_blocked") is True
            for selection in selections
        ),
        "proposed_semantics_proof_bindings_established": any(
            selection.get("proof_backed_instruction_node_count", 0) > 0
            for selection in selections
        ),
        "proof_premises_established_for_bound_instructions": False,
        "special_register_launch_domain_assumptions_bound": all(
            selection.get("launch_domain_bound_special_register_leaf_count")
            == selection.get("coordinate_special_register_leaf_count")
            for selection in selections
        ),
        "special_register_coordinate_correspondence_established": False,
        "special_register_concrete_values_established": False,
        "special_register_hardware_acquisition_established": False,
        "all_expression_instruction_semantics_bound": all(
            selection.get("unmodeled_instruction_node_count") == 0
            for selection in selections
        ),
        "expression_call_string_depth_overflow_free": expression_graph[
            "abstracted_call_overflow_count"
        ]
        == 0,
        "unbounded_context_sensitive_expression_reaching_definitions_established": False,
        "hardware_instruction_semantics_established": False,
        "closed_supported_sass_formulas_established": all(
            selection.get("closed_supported_formula", False) for selection in selections
        ),
        "sass_effective_address_formula_bound": False,
        "sass_to_logical_stride_correspondence_established": False,
        "sass_effective_address_bounds_established": False,
        "kernel_memory_safety_established": False,
    }
    body["certificate_sha256"] = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    return body


def build_sass_expression_summary(certificate: dict[str, Any]) -> dict[str, Any]:
    verification = verify_sass_expression_certificate(certificate)
    if not verification["valid"]:
        raise ValueError("Cannot summarize an invalid SASS expression certificate")
    selections = [
        {
            key: value
            for key, value in selection.items()
            if key not in {
                "expression_nodes",
                "partial_symbolic_formula",
                "partial_formula_analysis",
                "partial_formula_interval_analysis",
                "partial_formula_blocker_analysis",
            }
        }
        | {
            "expression_graph_sha256": hashlib.sha256(
                canonical_json(selection.get("expression_nodes", [])).encode("utf-8")
            ).hexdigest(),
            "partial_formula_sha256": selection.get("partial_symbolic_formula", {}).get(
                "formula_sha256"
            ),
            "partial_formula_root_nodes": selection.get("partial_symbolic_formula", {}).get(
                "root_nodes", []
            ),
            "partial_formula_node_count": selection.get("partial_symbolic_formula", {}).get(
                "node_count"
            ),
            "partial_formula_closed": selection.get("partial_symbolic_formula", {}).get(
                "closed_formula"
            ),
            "partial_formula_analysis_sha256": selection.get(
                "partial_formula_analysis", {}
            ).get("analysis_sha256"),
            "partial_formula_all_local_lowering_obligations_proved": selection.get(
                "partial_formula_analysis", {}
            ).get("all_local_lowering_obligations_proved"),
            "partial_formula_translated_root_count": selection.get(
                "partial_formula_analysis", {}
            ).get("translated_root_count"),
            "partial_formula_launch_domain_assumption_count": selection.get(
                "partial_formula_analysis", {}
            ).get("launch_domain_assumption_count"),
            "partial_formula_interval_analysis_sha256": selection.get(
                "partial_formula_interval_analysis", {}
            ).get("analysis_sha256"),
            "partial_formula_root_intervals": selection.get(
                "partial_formula_interval_analysis", {}
            ).get("root_intervals", []),
            "partial_formula_all_root_intervals_full_width_unknown": selection.get(
                "partial_formula_interval_analysis", {}
            ).get("all_root_intervals_full_width_unknown"),
            "partial_formula_assumption_conditioned_nontrivial_root_interval_count": selection.get(
                "partial_formula_interval_analysis", {}
            ).get("assumption_conditioned_nontrivial_root_interval_count"),
            "partial_formula_blocker_analysis_sha256": selection.get(
                "partial_formula_blocker_analysis", {}
            ).get("analysis_sha256"),
            "partial_formula_root_blocker_frontier": selection.get(
                "partial_formula_blocker_analysis", {}
            ).get("root_frontier", []),
            "partial_formula_all_opaque_label_histogram": selection.get(
                "partial_formula_blocker_analysis", {}
            ).get("all_opaque_label_histogram", {}),
        }
        for selection in certificate["selections"]
    ]
    body = {
        "scope": certificate["scope"],
        "source_certificate_sha256": certificate["certificate_sha256"],
        "kernel_name": certificate["kernel_name"],
        "cubin_sha256": certificate["cubin_sha256"],
        "sass_canonical_sha256": certificate["sass_canonical_sha256"],
        "sass_memory_certificate_sha256": certificate["sass_memory_certificate_sha256"],
        "nsight_certificate_sha256": certificate["nsight_certificate_sha256"],
        "launch_coordinate_domains": certificate["launch_coordinate_domains"],
        "sass_semantics_certificate_sha256": certificate[
            "sass_semantics_certificate_sha256"
        ],
        "expression_opcode_semantics": certificate["expression_opcode_semantics"],
        "logical_bounds_certificate_sha256": certificate["logical_bounds_certificate_sha256"],
        "expression_reaching_definition_summary": certificate[
            "expression_reaching_definition_summary"
        ],
        "selections": selections,
        "total_selections": len(selections),
        "available_selections": sum(selection["available"] for selection in selections),
        "closed_supported_formulas": sum(
            selection.get("closed_supported_formula", False) for selection in selections
        ),
        "selected_sass_address_expression_dags_established": certificate[
            "selected_sass_address_expression_dags_established"
        ],
        "bounded_call_string_expression_reaching_definitions_established": certificate[
            "bounded_call_string_expression_reaching_definitions_established"
        ],
        "bounded_predicate_reaching_definitions_established": certificate[
            "bounded_predicate_reaching_definitions_established"
        ],
        "predicate_producer_consumer_dependencies_established": certificate[
            "predicate_producer_consumer_dependencies_established"
        ],
        "predicate_values_established": False,
        "predicate_carry_equations_established": False,
        "predicate_hardware_semantics_established": False,
        "partial_proposed_symbolic_formulas_established": certificate[
            "partial_proposed_symbolic_formulas_established"
        ],
        "partial_formula_ordered_operands_preserved": certificate[
            "partial_formula_ordered_operands_preserved"
        ],
        "partial_formula_well_typed": certificate["partial_formula_well_typed"],
        "typed_z3_translation_established": certificate[
            "typed_z3_translation_established"
        ],
        "local_proposed_operator_lowering_equivalence_established": certificate[
            "local_proposed_operator_lowering_equivalence_established"
        ],
        "partial_formula_interval_analysis_established": certificate[
            "partial_formula_interval_analysis_established"
        ],
        "all_selected_root_intervals_full_width_unknown": certificate[
            "all_selected_root_intervals_full_width_unknown"
        ],
        "assumption_conditioned_effective_address_bounds_established": False,
        "root_opaque_blocker_frontiers_established": certificate[
            "root_opaque_blocker_frontiers_established"
        ],
        "all_selected_roots_have_opaque_blockers": certificate[
            "all_selected_roots_have_opaque_blockers"
        ],
        "proposed_semantics_proof_bindings_established": certificate[
            "proposed_semantics_proof_bindings_established"
        ],
        "proof_premises_established_for_bound_instructions": False,
        "special_register_launch_domain_assumptions_bound": certificate[
            "special_register_launch_domain_assumptions_bound"
        ],
        "special_register_coordinate_correspondence_established": False,
        "special_register_concrete_values_established": False,
        "special_register_hardware_acquisition_established": False,
        "all_expression_instruction_semantics_bound": certificate[
            "all_expression_instruction_semantics_bound"
        ],
        "expression_call_string_depth_overflow_free": certificate[
            "expression_call_string_depth_overflow_free"
        ],
        "unbounded_context_sensitive_expression_reaching_definitions_established": False,
        "hardware_instruction_semantics_established": False,
        "closed_supported_sass_formulas_established": certificate[
            "closed_supported_sass_formulas_established"
        ],
        "sass_effective_address_formula_bound": False,
        "sass_to_logical_stride_correspondence_established": False,
        "sass_effective_address_bounds_established": False,
        "kernel_memory_safety_established": False,
    }
    body["summary_sha256"] = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    return body


def verify_sass_expression_summary(summary: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in summary.items() if key != "summary_sha256"}
    hash_valid = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest() == summary.get(
        "summary_sha256"
    )
    selections = summary.get("selections", [])
    source_certificate_link_valid = bool(
        re.fullmatch(r"[0-9a-f]{64}", summary.get("source_certificate_sha256", ""))
    )
    counts_consistent = (
        summary.get("total_selections") == len(selections)
        and summary.get("available_selections") == sum(selection["available"] for selection in selections)
        and summary.get("closed_supported_formulas")
        == sum(selection.get("closed_supported_formula", False) for selection in selections)
        and summary.get("closed_supported_sass_formulas_established")
        == all(selection.get("closed_supported_formula", False) for selection in selections)
        and all(
            selection.get("closed_supported_formula")
            == (
                selection.get("unsupported_or_entry_node_count") == 0
                and selection.get("partial_formula_closed") is True
            )
            and selection.get("instruction_definition_node_count")
            == selection.get("proof_backed_instruction_node_count")
            + selection.get("unmodeled_instruction_node_count")
            and selection.get("unmodeled_instruction_node_count")
            == sum(selection.get("unmodeled_opcode_histogram", {}).values())
            and bool(
                re.fullmatch(
                    r"[0-9a-f]{64}", selection.get("partial_formula_sha256", "")
                )
            )
            and selection.get("partial_formula_node_count", 0)
            >= len(selection.get("partial_formula_root_nodes", []))
            and selection.get("partial_formula_closed")
            == (selection.get("partial_formula_opaque_node_count") == 0)
            and selection.get("partial_formula_well_typed") is True
            and selection.get("partial_formula_all_local_lowering_obligations_proved")
            is True
            and selection.get("partial_formula_local_lowering_obligation_count")
            == selection.get("partial_formula_lowered_operation_node_count")
            and selection.get("partial_formula_translated_root_count")
            == len(selection.get("partial_formula_root_nodes", []))
            and isinstance(
                selection.get("partial_formula_launch_domain_assumption_count"), int
            )
            and bool(
                re.fullmatch(
                    r"[0-9a-f]{64}",
                    selection.get("partial_formula_analysis_sha256", ""),
                )
            )
            and bool(
                re.fullmatch(
                    r"[0-9a-f]{64}",
                    selection.get("partial_formula_interval_analysis_sha256", ""),
                )
            )
            and len(selection.get("partial_formula_root_intervals", []))
            == len(selection.get("partial_formula_root_nodes", []))
            and all(
                root == interval.get("formula_node_sha256")
                for root, interval in zip(
                    selection.get("partial_formula_root_nodes", []),
                    selection.get("partial_formula_root_intervals", []),
                    strict=True,
                )
            )
            and all(
                _verify_interval_record(record)
                for record in selection.get("partial_formula_root_intervals", [])
            )
            and selection.get("partial_formula_all_root_intervals_full_width_unknown")
            == all(
                record.get("full_width_unknown")
                for record in selection.get("partial_formula_root_intervals", [])
            )
            and selection.get(
                "partial_formula_assumption_conditioned_nontrivial_root_interval_count"
            )
            == sum(
                not record.get("full_width_unknown")
                and record.get("bound_depends_on_assumption")
                for record in selection.get("partial_formula_root_intervals", [])
            )
            and 0
            <= selection.get("partial_formula_exact_interval_node_count", -1)
            <= selection.get("partial_formula_bounded_interval_node_count", -1)
            <= selection.get("partial_formula_node_count", -1)
            and 0
            <= selection.get(
                "partial_formula_assumption_bounded_interval_node_count", -1
            )
            <= selection.get("partial_formula_bounded_interval_node_count", -1)
            and bool(
                re.fullmatch(
                    r"[0-9a-f]{64}",
                    selection.get("partial_formula_blocker_analysis_sha256", ""),
                )
            )
            and selection.get("partial_formula_root_frontier_blocker_count")
            == len(selection.get("partial_formula_root_blocker_frontier", []))
            and all(
                _verify_blocker_record(record)
                and record["root_index"]
                < len(selection.get("partial_formula_root_nodes", []))
                and record["root_formula_node_sha256"]
                == selection["partial_formula_root_nodes"][record["root_index"]]
                for record in selection.get(
                    "partial_formula_root_blocker_frontier", []
                )
            )
            and selection.get("partial_formula_all_roots_blocked")
            == (
                len(
                    {
                        record["root_index"]
                        for record in selection.get(
                            "partial_formula_root_blocker_frontier", []
                        )
                    }
                )
                == len(selection.get("partial_formula_root_nodes", []))
            )
            and sum(
                selection.get("partial_formula_all_opaque_label_histogram", {}).values()
            )
            == selection.get("partial_formula_opaque_node_count")
            and all(
                isinstance(selection.get(field), int)
                and selection.get(field) >= 0
                for field in (
                    "predicate_definition_node_count",
                    "predicate_source_edge_count",
                    "entry_predicate_leaf_count",
                    "predicate_join_node_count",
                )
            )
            for selection in selections
        )
        and summary.get("all_expression_instruction_semantics_bound")
        == all(
            selection.get("unmodeled_instruction_node_count") == 0
            for selection in selections
        )
        and summary.get("proposed_semantics_proof_bindings_established")
        == any(
            selection.get("proof_backed_instruction_node_count", 0) > 0
            for selection in selections
        )
        and summary.get("selected_sass_address_expression_dags_established")
        == all(
            selection.get("available")
            and selection.get("target_field_represented")
            for selection in selections
        )
        and summary.get("bounded_predicate_reaching_definitions_established") is True
        and summary.get("predicate_producer_consumer_dependencies_established")
        == any(
            selection.get("predicate_definition_node_count", 0) > 0
            and selection.get("predicate_source_edge_count", 0) > 0
            for selection in selections
        )
        and summary.get("partial_proposed_symbolic_formulas_established")
        == all(bool(selection.get("partial_formula_sha256")) for selection in selections)
        and summary.get("partial_formula_ordered_operands_preserved") is True
        and summary.get("partial_formula_well_typed")
        == all(selection.get("partial_formula_well_typed") is True for selection in selections)
        and summary.get("typed_z3_translation_established")
        == all(
            selection.get("partial_formula_translated_root_count")
            == len(selection.get("partial_formula_root_nodes", []))
            for selection in selections
        )
        and summary.get("local_proposed_operator_lowering_equivalence_established")
        == all(
            selection.get("partial_formula_all_local_lowering_obligations_proved")
            is True
            for selection in selections
        )
        and summary.get("partial_formula_interval_analysis_established")
        == all(
            bool(selection.get("partial_formula_interval_analysis_sha256"))
            for selection in selections
        )
        and summary.get("all_selected_root_intervals_full_width_unknown")
        == all(
            selection.get("partial_formula_all_root_intervals_full_width_unknown")
            is True
            for selection in selections
        )
        and summary.get("assumption_conditioned_effective_address_bounds_established")
        is False
        and summary.get("root_opaque_blocker_frontiers_established")
        == all(
            selection.get("partial_formula_all_roots_blocked") is True
            for selection in selections
        )
        and summary.get("all_selected_roots_have_opaque_blockers")
        == all(
            selection.get("partial_formula_all_roots_blocked") is True
            for selection in selections
        )
        and _verify_expression_semantics_snapshot(
            summary.get("expression_opcode_semantics", {})
        )
        and summary.get("sass_semantics_certificate_sha256")
        == summary.get("expression_opcode_semantics", {}).get(
            "sass_semantics_certificate_sha256"
        )
        and _verify_launch_coordinate_domains(
            summary.get("launch_coordinate_domains", {})
        )
        and summary.get("nsight_certificate_sha256")
        == summary.get("launch_coordinate_domains", {}).get(
            "nsight_certificate_sha256"
        )
        and summary.get("special_register_launch_domain_assumptions_bound")
        == all(
            selection.get("launch_domain_bound_special_register_leaf_count")
            == selection.get("coordinate_special_register_leaf_count")
            for selection in selections
        )
    )
    boundaries_preserved = (
        summary.get("selected_sass_address_expression_dags_established") is True
        and summary.get("bounded_call_string_expression_reaching_definitions_established") is True
        and summary.get("bounded_predicate_reaching_definitions_established") is True
        and summary.get("predicate_producer_consumer_dependencies_established") is True
        and summary.get("predicate_values_established") is False
        and summary.get("predicate_carry_equations_established") is False
        and summary.get("predicate_hardware_semantics_established") is False
        and summary.get("partial_proposed_symbolic_formulas_established") is True
        and summary.get("partial_formula_ordered_operands_preserved") is True
        and summary.get("partial_formula_well_typed") is True
        and summary.get("typed_z3_translation_established") is True
        and summary.get("local_proposed_operator_lowering_equivalence_established") is True
        and summary.get("partial_formula_interval_analysis_established") is True
        and summary.get("assumption_conditioned_effective_address_bounds_established") is False
        and summary.get("root_opaque_blocker_frontiers_established") is True
        and summary.get("all_selected_roots_have_opaque_blockers") is True
        and summary.get("proposed_semantics_proof_bindings_established") is True
        and summary.get("proof_premises_established_for_bound_instructions") is False
        and summary.get("special_register_launch_domain_assumptions_bound") is True
        and summary.get("special_register_coordinate_correspondence_established") is False
        and summary.get("special_register_concrete_values_established") is False
        and summary.get("special_register_hardware_acquisition_established") is False
        and summary.get("hardware_instruction_semantics_established") is False
        and summary.get("expression_call_string_depth_overflow_free")
        == (summary.get("expression_reaching_definition_summary", {}).get("abstracted_call_overflow_count") == 0)
        and summary.get("unbounded_context_sensitive_expression_reaching_definitions_established") is False
        and summary.get("sass_effective_address_formula_bound") is False
        and summary.get("sass_to_logical_stride_correspondence_established") is False
        and summary.get("sass_effective_address_bounds_established") is False
        and summary.get("kernel_memory_safety_established") is False
    )
    return {
        "valid": bool(
            hash_valid
            and source_certificate_link_valid
            and counts_consistent
            and boundaries_preserved
            and selections
        ),
        "summary_hash_valid": hash_valid,
        "source_certificate_link_valid": source_certificate_link_valid,
        "counts_consistent": counts_consistent,
        "boundaries_preserved": boundaries_preserved,
    }


def _semantic_binding_consistent(
    node: dict[str, Any], snapshot: dict[str, Any]
) -> bool:
    requirement = _semantic_requirement(node.get("opcode", ""), node.get("operands", ""))
    binding = node.get("semantic_binding")
    if requirement is None:
        return binding is None
    if not binding:
        return False
    proof_by_name = {
        record["name"]: record for record in snapshot.get("obligations", [])
    }
    if any(name not in proof_by_name for name in requirement):
        return False
    return (
        binding.get("opcode") == node.get("opcode")
        and binding.get("obligation_names") == requirement
        and binding.get("proof_record_sha256")
        == [proof_by_name[name]["proof_record_sha256"] for name in requirement]
        and binding.get("sass_semantics_certificate_sha256")
        == snapshot.get("sass_semantics_certificate_sha256")
        and binding.get("snapshot_sha256") == snapshot.get("snapshot_sha256")
        and binding.get("binding_scope")
        == "Proposed-semantics proof-record reference only."
        and binding.get("proof_premises_established_for_instruction") is False
        and binding.get("hardware_instruction_semantics_established") is False
    )


def _selection_graph_consistent(
    selection: dict[str, Any],
    snapshot: dict[str, Any],
    launch_coordinate_domains: dict[str, Any],
) -> bool:
    nodes = selection.get("expression_nodes", [])
    identifiers = {node.get("node_sha256") for node in nodes}
    references = {
        reference
        for node in nodes
        for reference in (
            [*node.get("source_nodes", [])]
            + ([node["definition_node"]] if node.get("definition_node") else [])
        )
    }
    represented_fields = sorted(
        {
            field
            for node in nodes
            for field in node.get("parameter_fields", [])
        }
    )
    instruction_nodes = [
        node for node in nodes if node.get("kind") == "instruction_definition"
    ]
    special_nodes = [node for node in nodes if node.get("kind") == "special_register"]
    proof_backed_nodes = [node for node in instruction_nodes if node.get("semantic_binding")]
    unmodeled_opcodes = Counter(
        node["opcode"] for node in instruction_nodes if not node.get("semantic_binding")
    )
    referenced_obligations = sorted(
        {
            name
            for node in proof_backed_nodes
            for name in node["semantic_binding"]["obligation_names"]
        }
    )
    expected_partial_formula = _build_partial_symbolic_formula(
        selection.get("root_nodes", []), nodes
    )
    expected_partial_formula_analysis = _analyze_partial_symbolic_formula(
        expected_partial_formula
    )
    expected_partial_formula_interval_analysis = _analyze_partial_formula_intervals(
        expected_partial_formula
    )
    expected_partial_formula_blocker_analysis = _analyze_partial_formula_blockers(
        expected_partial_formula
    )
    node_by_id = {node["node_sha256"]: node for node in nodes}
    predicate_descriptors = [
        descriptor
        for node in instruction_nodes
        for descriptor in node.get("ordered_semantic_operands", [])
        if descriptor.get("kind") == "predicate"
    ]
    return bool(
        selection.get("available")
        and selection.get("node_count") == len(nodes)
        and set(selection.get("root_nodes", [])) <= identifiers
        and references <= identifiers
        and selection.get("represented_parameter_fields") == represented_fields
        and selection.get("target_field_represented")
        == (selection.get("field") in represented_fields)
        and selection.get("ambiguous_reaching_definition_node_count")
        == sum(node.get("kind") == "reaching_definition_join" for node in nodes)
        and selection.get("cyclic_reaching_definition_node_count")
        == sum(node.get("kind") == "cyclic_reaching_definition" for node in nodes)
        and selection.get("special_register_leaf_count") == len(special_nodes)
        and selection.get("predicate_definition_node_count")
        == sum(
            node.get("output_kind") == "predicate"
            for node in instruction_nodes
        )
        and selection.get("predicate_source_edge_count")
        == sum(
            bool(re.fullmatch(r"(?:UP|P)\d+", source))
            for node in instruction_nodes
            for source in node.get("source_registers", [])
        )
        and selection.get("entry_predicate_leaf_count")
        == sum(
            node.get("kind") == "entry_register"
            and node.get("value_kind") == "predicate"
            for node in nodes
        )
        and selection.get("predicate_join_node_count")
        == sum(
            node.get("kind") == "reaching_definition_join"
            and node.get("value_kind") == "predicate"
            for node in nodes
        )
        and all(
            descriptor.get("source_node") in node_by_id
            and node_by_id[descriptor["source_node"]].get("kind")
            != "entry_register"
            for descriptor in predicate_descriptors
        )
        and selection.get("coordinate_special_register_leaf_count")
        == sum(
            node.get("register") in SPECIAL_REGISTER_COORDINATES
            for node in special_nodes
        )
        and selection.get("launch_domain_bound_special_register_leaf_count")
        == sum(node.get("launch_domain_assumption") is not None for node in special_nodes)
        and all(
            node.get("launch_domain_assumption")
            == launch_coordinate_domains.get("registers", {}).get(node.get("register"))
            for node in special_nodes
        )
        and selection.get("instruction_definition_node_count") == len(instruction_nodes)
        and selection.get("proof_backed_instruction_node_count") == len(proof_backed_nodes)
        and selection.get("unmodeled_instruction_node_count")
        == len(instruction_nodes) - len(proof_backed_nodes)
        and selection.get("unmodeled_opcode_histogram")
        == dict(sorted(unmodeled_opcodes.items()))
        and selection.get("referenced_semantics_obligations")
        == referenced_obligations
        and selection.get("partial_symbolic_formula") == expected_partial_formula
        and selection.get("partial_formula_lowered_operation_node_count")
        == expected_partial_formula["lowered_operation_node_count"]
        and selection.get("partial_formula_opaque_node_count")
        == expected_partial_formula["opaque_node_count"]
        and selection.get("partial_formula_analysis")
        == expected_partial_formula_analysis
        and selection.get("partial_formula_well_typed")
        == expected_partial_formula_analysis["type_check"]["well_typed"]
        and selection.get("partial_formula_local_lowering_obligation_count")
        == expected_partial_formula_analysis["total_local_lowering_obligations"]
        and selection.get("partial_formula_interval_analysis")
        == expected_partial_formula_interval_analysis
        and selection.get("partial_formula_bounded_interval_node_count")
        == expected_partial_formula_interval_analysis["bounded_node_count"]
        and selection.get("partial_formula_exact_interval_node_count")
        == expected_partial_formula_interval_analysis["exact_node_count"]
        and selection.get("partial_formula_assumption_bounded_interval_node_count")
        == expected_partial_formula_interval_analysis["assumption_bounded_node_count"]
        and selection.get("partial_formula_blocker_analysis")
        == expected_partial_formula_blocker_analysis
        and selection.get("partial_formula_root_frontier_blocker_count")
        == expected_partial_formula_blocker_analysis["root_frontier_blocker_count"]
        and selection.get("partial_formula_all_roots_blocked")
        == expected_partial_formula_blocker_analysis["all_roots_blocked"]
        and all(
            _semantic_binding_consistent(node, snapshot) for node in instruction_nodes
        )
    )


def verify_sass_expression_certificate(certificate: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in certificate.items() if key != "certificate_sha256"}
    hash_valid = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest() == certificate.get(
        "certificate_sha256"
    )
    checks_consistent = certificate.get("all_checks_pass") == all(certificate.get("checks", {}).values())
    semantics_snapshot = certificate.get("expression_opcode_semantics", {})
    semantics_snapshot_valid = _verify_expression_semantics_snapshot(
        semantics_snapshot
    ) and certificate.get("sass_semantics_certificate_sha256") == semantics_snapshot.get(
        "sass_semantics_certificate_sha256"
    )
    launch_coordinate_domains = certificate.get("launch_coordinate_domains", {})
    launch_coordinate_domains_valid = _verify_launch_coordinate_domains(
        launch_coordinate_domains
    ) and certificate.get("nsight_certificate_sha256") == launch_coordinate_domains.get(
        "nsight_certificate_sha256"
    )
    boundaries_preserved = (
        certificate.get("selected_sass_address_expression_dags_established") is True
        and certificate.get("bounded_call_string_expression_reaching_definitions_established") is True
        and certificate.get("bounded_predicate_reaching_definitions_established") is True
        and certificate.get("predicate_producer_consumer_dependencies_established") is True
        and certificate.get("predicate_values_established") is False
        and certificate.get("predicate_carry_equations_established") is False
        and certificate.get("predicate_hardware_semantics_established") is False
        and certificate.get("partial_proposed_symbolic_formulas_established") is True
        and certificate.get("partial_formula_ordered_operands_preserved") is True
        and certificate.get("partial_formula_well_typed") is True
        and certificate.get("typed_z3_translation_established") is True
        and certificate.get("local_proposed_operator_lowering_equivalence_established") is True
        and certificate.get("partial_formula_interval_analysis_established") is True
        and certificate.get("assumption_conditioned_effective_address_bounds_established") is False
        and certificate.get("root_opaque_blocker_frontiers_established") is True
        and certificate.get("all_selected_roots_have_opaque_blockers") is True
        and certificate.get("proposed_semantics_proof_bindings_established") is True
        and certificate.get("proof_premises_established_for_bound_instructions") is False
        and certificate.get("special_register_launch_domain_assumptions_bound") is True
        and certificate.get("special_register_coordinate_correspondence_established") is False
        and certificate.get("special_register_concrete_values_established") is False
        and certificate.get("special_register_hardware_acquisition_established") is False
        and certificate.get("hardware_instruction_semantics_established") is False
        and certificate.get("expression_call_string_depth_overflow_free")
        == (
            certificate.get("expression_reaching_definition_summary", {}).get(
                "abstracted_call_overflow_count"
            )
            == 0
        )
        and certificate.get("unbounded_context_sensitive_expression_reaching_definitions_established") is False
        and certificate.get("sass_effective_address_formula_bound") is False
        and certificate.get("sass_to_logical_stride_correspondence_established") is False
        and certificate.get("sass_effective_address_bounds_established") is False
        and certificate.get("kernel_memory_safety_established") is False
    )
    closed_formulas_consistent = all(
        selection.get("unsupported_or_entry_node_count")
        == len(_unsupported_expression_nodes(selection.get("expression_nodes", [])))
        and selection.get("closed_supported_formula")
        == (
            len(_unsupported_expression_nodes(selection.get("expression_nodes", []))) == 0
            and selection.get("partial_symbolic_formula", {}).get("closed_formula") is True
        )
        for selection in certificate.get("selections", [])
    ) and certificate.get("closed_supported_sass_formulas_established") == all(
        selection.get("closed_supported_formula", False)
        for selection in certificate.get("selections", [])
    )
    selections = certificate.get("selections", [])
    selection_graphs_consistent = (
        all(
            _selection_graph_consistent(
                selection, semantics_snapshot, launch_coordinate_domains
            )
            for selection in selections
        )
        and certificate.get("all_expression_instruction_semantics_bound")
        == all(
            selection.get("unmodeled_instruction_node_count") == 0
            for selection in selections
        )
        and certificate.get("proposed_semantics_proof_bindings_established")
        == any(
            selection.get("proof_backed_instruction_node_count", 0) > 0
            for selection in selections
        )
        and certificate.get("special_register_launch_domain_assumptions_bound")
        == all(
            selection.get("launch_domain_bound_special_register_leaf_count")
            == selection.get("coordinate_special_register_leaf_count")
            for selection in selections
        )
        and certificate.get("bounded_predicate_reaching_definitions_established") is True
        and certificate.get("predicate_producer_consumer_dependencies_established")
        == any(
            selection.get("predicate_definition_node_count", 0) > 0
            and selection.get("predicate_source_edge_count", 0) > 0
            for selection in selections
        )
        and certificate.get("selected_sass_address_expression_dags_established")
        == all(
            selection.get("available")
            and selection.get("target_field_represented")
            for selection in selections
        )
        and certificate.get("partial_proposed_symbolic_formulas_established")
        == all(
            bool(selection.get("partial_symbolic_formula", {}).get("formula_nodes"))
            for selection in selections
        )
        and certificate.get("partial_formula_well_typed")
        == all(selection.get("partial_formula_well_typed") is True for selection in selections)
        and certificate.get("typed_z3_translation_established")
        == all(
            selection.get("partial_formula_analysis", {}).get("translated_root_count")
            == len(selection.get("partial_symbolic_formula", {}).get("root_nodes", []))
            for selection in selections
        )
        and certificate.get("local_proposed_operator_lowering_equivalence_established")
        == all(
            selection.get("partial_formula_analysis", {}).get(
                "all_local_lowering_obligations_proved"
            )
            is True
            for selection in selections
        )
        and certificate.get("partial_formula_interval_analysis_established")
        == all(
            selection.get("partial_formula_interval_analysis", {}).get(
                "all_nodes_analyzed"
            )
            is True
            for selection in selections
        )
        and certificate.get("all_selected_root_intervals_full_width_unknown")
        == all(
            selection.get("partial_formula_interval_analysis", {}).get(
                "all_root_intervals_full_width_unknown"
            )
            is True
            for selection in selections
        )
        and certificate.get("root_opaque_blocker_frontiers_established")
        == all(
            selection.get("partial_formula_blocker_analysis", {}).get(
                "all_roots_blocked"
            )
            is True
            for selection in selections
        )
        and certificate.get("all_selected_roots_have_opaque_blockers")
        == all(
            selection.get("partial_formula_blocker_analysis", {}).get(
                "all_roots_blocked"
            )
            is True
            for selection in selections
        )
    )
    node_hashes_valid = all(
        node["node_sha256"]
        == hashlib.sha256(
            canonical_json({key: value for key, value in node.items() if key != "node_sha256"}).encode("utf-8")
        ).hexdigest()
        for selection in certificate.get("selections", [])
        for node in selection.get("expression_nodes", [])
    )
    return {
        "valid": bool(
            hash_valid
            and checks_consistent
            and boundaries_preserved
            and semantics_snapshot_valid
            and launch_coordinate_domains_valid
            and closed_formulas_consistent
            and selection_graphs_consistent
            and node_hashes_valid
            and certificate.get("all_checks_pass")
        ),
        "certificate_hash_valid": hash_valid,
        "checks_consistent": checks_consistent,
        "boundaries_preserved": boundaries_preserved,
        "semantics_snapshot_valid": semantics_snapshot_valid,
        "launch_coordinate_domains_valid": launch_coordinate_domains_valid,
        "closed_formulas_consistent": closed_formulas_consistent,
        "selection_graphs_consistent": selection_graphs_consistent,
        "node_hashes_valid": node_hashes_valid,
    }
