from __future__ import annotations

import ctypes
import hashlib
import re
import subprocess
from collections import Counter, deque
from pathlib import Path
from typing import Any

from .attention_parameters import verify_attention_parameter_certificate
from .kernel_signatures import _AttentionParams, _layout
from .nsight_attestation import _instruction_summary, _parse_cuobjdump_sass, verify_nsight_launch_certificate
from .sass_semantics import (
    PROPOSED_MEMORY_OPERAND_ROLES,
    PROPOSED_MEMORY_WIDTHS,
    verify_sass_semantics_certificate,
)
from .serialization import canonical_json


GLOBAL_MEMORY_BASES = {"LDG", "STG", "LDGSTS", "ATOM", "RED"}
SHARED_MEMORY_BASES = {"LDS", "STS", "LDSM", "ATOMS"}
GENERIC_MEMORY_BASES = {"LD", "ST"}
MEMORY_BASES = GLOBAL_MEMORY_BASES | SHARED_MEMORY_BASES | GENERIC_MEMORY_BASES
LOAD_BASES = {"LDG", "LDS", "LDSM", "LD"}


TARGET_POINTER_FIELDS = {
    "query_ptr": 0,
    "key_ptr": 8,
    "value_ptr": 16,
    "output_ptr": 64,
    "output_accum_ptr": 72,
}


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _constant_offsets(instruction: dict[str, Any]) -> list[int]:
    return [int(value, 16) for value in re.findall(r"c\[0x0\]\[0x([0-9a-fA-F]+)\]", instruction["operands"])]


def _derive_parameter_base(instructions: list[dict[str, Any]]) -> dict[str, Any]:
    offsets = {
        offset
        for instruction in instructions
        if instruction["opcode"].startswith("ULDC.64")
        for offset in _constant_offsets(instruction)
    }
    candidates = sorted(
        base
        for base in offsets
        if base % 16 == 0 and all(base + field_offset in offsets for field_offset in TARGET_POINTER_FIELDS.values())
    )
    if len(candidates) != 1:
        raise ValueError(f"Expected one attention parameter base candidate; found {candidates}")
    return {
        "base_constant_offset": candidates[0],
        "derivation": "Unique 16-byte-aligned constant-space offset for which ULDC.64 loads at all five source-reconstructed pointer-field offsets are observed.",
        "candidate_count": len(candidates),
    }


def _field_for_offset(parameter_offset: int) -> str | None:
    fields = _layout(_AttentionParams)["fields"]
    return next(
        (
            field["name"]
            for field in fields
            if field["offset"] <= parameter_offset < field["offset"] + field["size_bytes"]
        ),
        None,
    )


def _base_opcode(opcode: str) -> str:
    return opcode.split(".", 1)[0]


def _registers(value: str) -> list[str]:
    return re.findall(r"\b(?:UR|R)\d+\b", value)


def _next_register(register: str, increment: int = 1) -> str:
    prefix = "UR" if register.startswith("UR") else "R"
    return f"{prefix}{int(register[len(prefix):]) + increment}"


def _memory_width_bytes(opcode: str) -> int:
    if re.search(r"(?:^|\.)128(?:\.|$)", opcode):
        return 16
    if re.search(r"(?:^|\.)64(?:\.|$)", opcode):
        return 8
    if re.search(r"(?:^|\.)U16(?:\.|$)", opcode):
        return 2
    return 4


def _destination_register_count(opcode: str) -> int:
    if re.search(r"(?:^|\.)128(?:\.|$)", opcode):
        return 4
    if re.search(r"(?:^|\.)64(?:\.|$)", opcode):
        return 2
    if _base_opcode(opcode) == "LDSM":
        matrices = re.search(r"\.(\d+)$", opcode)
        if matrices:
            return int(matrices.group(1))
    return 1


def _address_operand_specs(opcode: str, memories: list[str]) -> list[tuple[str, str, str]]:
    base_opcode = _base_opcode(opcode)
    if base_opcode == "LDGSTS" and len(memories) == 2:
        return [
            ("candidate_write", "shared", memories[0]),
            ("candidate_read", "global", memories[1]),
        ]
    classifications = {
        "LDG": ("candidate_read", "global"),
        "LDS": ("candidate_read", "shared"),
        "LDSM": ("candidate_read", "shared"),
        "LD": ("candidate_read", "generic"),
        "STG": ("candidate_write", "global"),
        "STS": ("candidate_write", "shared"),
        "ST": ("candidate_write", "generic"),
        "ATOM": ("candidate_read_write", "global"),
        "ATOMS": ("candidate_read_write", "shared"),
        "RED": ("candidate_read_write", "global"),
    }
    access_class, memory_space = classifications[base_opcode]
    return [(access_class, memory_space, memory) for memory in memories]


def _memory_slice(instruction: dict[str, Any], taint: dict[str, set[str]]) -> dict[str, Any] | None:
    memories = re.findall(r"\[([^]]+)\]", instruction["operands"])
    if _base_opcode(instruction["opcode"]) not in MEMORY_BASES or not memories:
        return None
    address_operands = [
        {
            "operand_index": index,
            "access_class": access_class,
            "memory_space": memory_space,
            "address_registers": sorted(set(_registers(memory))),
            "source_parameter_fields": sorted(
                {field for register in _registers(memory) for field in taint.get(register, set())}
            ),
        }
        for index, (access_class, memory_space, memory) in enumerate(
            _address_operand_specs(instruction["opcode"], memories)
        )
    ]
    return {
        "instruction_offset": instruction["offset"],
        "opcode": instruction["opcode"],
        "address_registers": sorted(
            {register for operand in address_operands for register in operand["address_registers"]}
        ),
        "source_parameter_fields": sorted(
            {field for operand in address_operands for field in operand["source_parameter_fields"]}
        ),
        "address_operands": address_operands,
    }


def _transfer_taint(
    instruction: dict[str, Any], taint: dict[str, set[str]], parameter_base: int
) -> dict[str, set[str]]:
    result = {register: set(fields) for register, fields in taint.items()}
    opcode = instruction["opcode"]
    base_opcode = _base_opcode(opcode)
    operands = instruction["operands"]
    destination = re.match(r"((?:UR|R)\d+)\b", operands)
    if not destination or base_opcode in {"ST", "STG", "STS", "ATOM", "ATOMS", "RED"} or opcode.startswith(("BRA", "CALL", "RET", "EXIT")):
        return result
    destination_register = destination.group(1)
    source_fields: set[str] = set()
    if base_opcode not in LOAD_BASES:
        remaining = operands[destination.end() :]
        source_fields = {field for register in _registers(remaining) for field in taint.get(register, set())}
        for constant_offset in _constant_offsets(instruction):
            parameter_offset = constant_offset - parameter_base
            if 0 <= parameter_offset < ctypes.sizeof(_AttentionParams):
                field = _field_for_offset(parameter_offset)
                if field:
                    source_fields.add(field)
    for increment in range(_destination_register_count(opcode)):
        written_register = _next_register(destination_register, increment)
        written_fields = set(source_fields)
        if instruction.get("predicate"):
            written_fields.update(taint.get(written_register, set()))
        result[written_register] = written_fields
    return result


def _address_taint_slices(instructions: list[dict[str, Any]], parameter_base: int) -> list[dict[str, Any]]:
    taint: dict[str, set[str]] = {}
    slices = []
    for instruction in instructions:
        memory_slice = _memory_slice(instruction, taint)
        if memory_slice:
            slices.append(memory_slice)
        taint = _transfer_taint(instruction, taint, parameter_base)
    return slices


def _branch_target(instruction: dict[str, Any]) -> int | None:
    targets = re.findall(r"0x([0-9a-fA-F]+)", instruction["operands"])
    return int(targets[-1], 16) if targets else None


def _barrier_register(instruction: dict[str, Any]) -> str | None:
    match = re.search(r"\bB\d+\b", instruction["operands"])
    return match.group(0) if match else None


def _cfg_successors(instructions: list[dict[str, Any]]) -> tuple[list[list[int]], dict[str, int]]:
    offset_to_index = {instruction["offset"]: index for index, instruction in enumerate(instructions)}
    barrier_stacks: dict[str, list[int]] = {}
    reconvergence_targets: dict[int, int] = {}
    break_targets: dict[int, int] = {}
    matched_bssy_bsync = 0
    unmatched_barrier_controls = 0
    for index, instruction in enumerate(instructions):
        base_opcode = _base_opcode(instruction["opcode"])
        barrier = _barrier_register(instruction)
        if base_opcode == "BSSY" and barrier:
            target = _branch_target(instruction)
            if target in offset_to_index:
                barrier_stacks.setdefault(barrier, []).append(offset_to_index[target])
            else:
                unmatched_barrier_controls += 1
        elif base_opcode == "BREAK" and barrier:
            if barrier_stacks.get(barrier):
                break_targets[index] = barrier_stacks[barrier][-1]
            else:
                unmatched_barrier_controls += 1
        elif base_opcode == "BSYNC" and barrier:
            if barrier_stacks.get(barrier):
                reconvergence_targets[index] = barrier_stacks[barrier].pop()
                matched_bssy_bsync += 1
            else:
                unmatched_barrier_controls += 1
    call_fallthroughs = [
        index + 1
        for index, instruction in enumerate(instructions[:-1])
        if _base_opcode(instruction["opcode"]) == "CALL"
    ]
    successors: list[list[int]] = []
    unresolved_targets = 0
    unresolved_indirect_transfers = 0
    direct_branches = 0
    direct_calls = 0
    returns = 0
    bssy_count = 0
    bsync_count = 0
    break_count = 0
    for index, instruction in enumerate(instructions):
        opcode = instruction["opcode"]
        next_index = index + 1 if index + 1 < len(instructions) else None
        edges: set[int] = set()
        if _base_opcode(opcode) == "BRA":
            direct_branches += 1
            target = _branch_target(instruction)
            if target in offset_to_index:
                edges.add(offset_to_index[target])
            else:
                unresolved_targets += 1
            if instruction.get("predicate") and next_index is not None:
                edges.add(next_index)
        elif _base_opcode(opcode) == "CALL":
            direct_calls += 1
            target = _branch_target(instruction)
            if target in offset_to_index:
                edges.add(offset_to_index[target])
            else:
                unresolved_targets += 1
            if next_index is not None:
                edges.add(next_index)
        elif _base_opcode(opcode) == "RET":
            returns += 1
            edges.update(call_fallthroughs)
            if instruction.get("predicate") and next_index is not None:
                edges.add(next_index)
        elif opcode == "EXIT":
            if instruction.get("predicate") and next_index is not None:
                edges.add(next_index)
        elif _base_opcode(opcode) == "BSSY":
            bssy_count += 1
            if next_index is not None:
                edges.add(next_index)
        elif _base_opcode(opcode) == "BSYNC":
            bsync_count += 1
            if index in reconvergence_targets:
                edges.add(reconvergence_targets[index])
            elif next_index is not None:
                edges.add(next_index)
        elif _base_opcode(opcode) == "BREAK":
            break_count += 1
            if index in break_targets:
                edges.add(break_targets[index])
            if instruction.get("predicate") and next_index is not None:
                edges.add(next_index)
        elif _base_opcode(opcode) == "BRX":
            unresolved_indirect_transfers += 1
            if instruction.get("predicate") and next_index is not None:
                edges.add(next_index)
        else:
            if next_index is not None:
                edges.add(next_index)
        successors.append(sorted(edges))
    return successors, {
        "direct_branches": direct_branches,
        "direct_calls": direct_calls,
        "direct_call_fallthroughs": len(call_fallthroughs),
        "returns": returns,
        "context_insensitive_return_edges": returns * len(call_fallthroughs),
        "bssy_instructions": bssy_count,
        "bsync_instructions": bsync_count,
        "break_instructions": break_count,
        "lexically_matched_bssy_bsync": matched_bssy_bsync,
        "unmatched_barrier_controls": unmatched_barrier_controls,
        "unresolved_direct_targets": unresolved_targets,
        "unresolved_indirect_transfers": unresolved_indirect_transfers,
        "edge_count": sum(len(edges) for edges in successors),
    }


def _join_taint(left: dict[str, set[str]], right: dict[str, set[str]]) -> dict[str, set[str]]:
    return {
        register: set(left.get(register, set())) | set(right.get(register, set()))
        for register in left.keys() | right.keys()
    }


def _basic_blocks(
    instructions: list[dict[str, Any]], instruction_successors: list[list[int]]
) -> tuple[list[list[int]], dict[int, int], list[list[int]]]:
    leaders = {0}
    for index, successors in enumerate(instruction_successors):
        if successors != ([index + 1] if index + 1 < len(instructions) else []):
            if index + 1 < len(instructions):
                leaders.add(index + 1)
            leaders.update(successors)
    starts = sorted(leaders)
    blocks = [
        list(range(start, starts[position + 1] if position + 1 < len(starts) else len(instructions)))
        for position, start in enumerate(starts)
    ]
    instruction_to_block = {
        instruction_index: block_index
        for block_index, block in enumerate(blocks)
        for instruction_index in block
    }
    block_successors = [
        sorted({instruction_to_block[successor] for successor in instruction_successors[block[-1]]})
        for block in blocks
    ]
    return blocks, instruction_to_block, block_successors


def _cfg_address_taint_slices(
    instructions: list[dict[str, Any]], parameter_base: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    instruction_successors, graph = _cfg_successors(instructions)
    blocks, instruction_to_block, block_successors = _basic_blocks(instructions, instruction_successors)
    states: list[dict[str, set[str]] | None] = [None] * len(blocks)
    states[0] = {}
    queue = deque([0])
    queued = {0}
    iterations = 0
    while queue:
        block_index = queue.popleft()
        queued.remove(block_index)
        iterations += 1
        output = states[block_index] or {}
        for instruction_index in blocks[block_index]:
            output = _transfer_taint(instructions[instruction_index], output, parameter_base)
        for successor in block_successors[block_index]:
            joined = output if states[successor] is None else _join_taint(states[successor] or {}, output)
            if states[successor] != joined:
                states[successor] = joined
                if successor not in queued:
                    queue.append(successor)
                    queued.add(successor)
    slices = []
    for block_index, block in enumerate(blocks):
        if states[block_index] is None:
            continue
        state = states[block_index] or {}
        for instruction_index in block:
            memory_slice = _memory_slice(instructions[instruction_index], state)
            if memory_slice:
                slices.append(memory_slice)
            state = _transfer_taint(instructions[instruction_index], state, parameter_base)
    graph.update(
        {
            "basic_block_count": len(blocks),
            "reachable_basic_block_count": sum(state is not None for state in states),
            "reachable_instruction_count": sum(
                len(block) for block, state in zip(blocks, states) if state is not None
            ),
            "fixed_point_block_iterations": iterations,
        }
    )
    return slices, graph


def _call_string_address_taint_slices(
    instructions: list[dict[str, Any]], parameter_base: int, maximum_call_depth: int = 4
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    instruction_successors, _ = _cfg_successors(instructions)
    blocks, instruction_to_block, block_successors = _basic_blocks(instructions, instruction_successors)
    offset_to_block = {
        instructions[instruction_index]["offset"]: block_index
        for block_index, block in enumerate(blocks)
        for instruction_index in block
    }
    states: dict[tuple[int, tuple[int, ...]], dict[str, set[str]]] = {(0, ()): {}}
    queue = deque([(0, ())])
    queued = {(0, ())}
    iterations = 0
    transitions = 0
    truncated_calls = 0
    unresolved_returns = 0
    maximum_observed_depth = 0
    while queue:
        context = queue.popleft()
        queued.remove(context)
        block_index, call_stack = context
        iterations += 1
        output = states[context]
        for instruction_index in blocks[block_index]:
            output = _transfer_taint(instructions[instruction_index], output, parameter_base)
        terminator_index = blocks[block_index][-1]
        terminator = instructions[terminator_index]
        base_opcode = _base_opcode(terminator["opcode"])
        next_block = (
            instruction_to_block[terminator_index + 1]
            if terminator_index + 1 < len(instructions)
            else None
        )
        targets: set[tuple[int, tuple[int, ...]]] = set()
        if base_opcode == "CALL":
            target = _branch_target(terminator)
            target_block = offset_to_block.get(target)
            if target_block is not None and next_block is not None:
                if len(call_stack) < maximum_call_depth:
                    targets.add((target_block, (*call_stack, next_block)))
                    maximum_observed_depth = max(maximum_observed_depth, len(call_stack) + 1)
                else:
                    truncated_calls += 1
                    targets.add((target_block, (*call_stack[1:], next_block)))
            if terminator.get("predicate") and next_block is not None:
                targets.add((next_block, call_stack))
        elif base_opcode == "RET":
            if call_stack:
                targets.add((call_stack[-1], call_stack[:-1]))
            else:
                unresolved_returns += 1
            if terminator.get("predicate") and next_block is not None:
                targets.add((next_block, call_stack))
        else:
            targets.update((successor, call_stack) for successor in block_successors[block_index])
        transitions += len(targets)
        for target_context in targets:
            joined = (
                output
                if target_context not in states
                else _join_taint(states[target_context], output)
            )
            if states.get(target_context) != joined:
                states[target_context] = joined
                if target_context not in queued:
                    queue.append(target_context)
                    queued.add(target_context)
    slice_map: dict[tuple[int, str, tuple[str, ...]], dict[str, Any]] = {}
    for (block_index, call_stack), entry_state in states.items():
        state = entry_state
        for instruction_index in blocks[block_index]:
            memory_slice = _memory_slice(instructions[instruction_index], state)
            if memory_slice:
                key = (
                    memory_slice["instruction_offset"],
                    memory_slice["opcode"],
                    tuple(memory_slice["address_registers"]),
                )
                aggregate = slice_map.setdefault(
                    key,
                    {
                        **memory_slice,
                        "source_parameter_fields": [],
                        "address_operands": [
                            {**operand, "source_parameter_fields": [], "call_context_count": 0}
                            for operand in memory_slice["address_operands"]
                        ],
                        "call_context_count": 0,
                    },
                )
                aggregate["source_parameter_fields"] = sorted(
                    set(aggregate["source_parameter_fields"]) | set(memory_slice["source_parameter_fields"])
                )
                for aggregate_operand, context_operand in zip(
                    aggregate["address_operands"], memory_slice["address_operands"]
                ):
                    aggregate_operand["source_parameter_fields"] = sorted(
                        set(aggregate_operand["source_parameter_fields"])
                        | set(context_operand["source_parameter_fields"])
                    )
                    aggregate_operand["call_context_count"] += 1
                aggregate["call_context_count"] += 1
            state = _transfer_taint(instructions[instruction_index], state, parameter_base)
    slices = list(slice_map.values())
    reachable_blocks = {block_index for block_index, call_stack in states}
    return slices, {
        "maximum_call_depth": maximum_call_depth,
        "maximum_observed_call_depth": maximum_observed_depth,
        "call_context_count": len(states),
        "reachable_basic_block_count": len(reachable_blocks),
        "reachable_instruction_count": sum(len(blocks[index]) for index in reachable_blocks),
        "fixed_point_context_iterations": iterations,
        "context_transition_count": transitions,
        "abstracted_call_overflow_count": truncated_calls,
        "call_overflow_abstraction": "Drop oldest return site and retain the newest at the fixed depth.",
        "unresolved_return_context_count": unresolved_returns,
    }


def _opcode_text_access_classification(slices: list[dict[str, Any]]) -> dict[str, Any]:
    by_field: dict[str, Counter[str]] = {}
    by_class: Counter[str] = Counter()
    linked_operands = 0
    for memory_slice in slices:
        for operand in memory_slice["address_operands"]:
            fields = operand["source_parameter_fields"]
            if fields:
                linked_operands += 1
            by_class[operand["access_class"]] += 1
            for field in fields:
                by_field.setdefault(field, Counter())[operand["access_class"]] += 1
    return {
        "scope": "Exact opcode-text operand classification only; labels do not establish NVIDIA instruction or access semantics.",
        "address_operand_count_by_class": dict(sorted(by_class.items())),
        "parameter_linked_address_operand_count": linked_operands,
        "parameter_field_links_by_class": {
            field: dict(sorted(counts.items())) for field, counts in sorted(by_field.items())
        },
    }


def build_sass_memory_certificate(
    cuobjdump: str,
    cubin: Path,
    kernel: str,
    nsight_certificate: dict[str, Any],
    attention_certificate: dict[str, Any],
    sass_semantics_certificate: dict[str, Any],
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
    base = _derive_parameter_base(instructions)
    parameter_loads = []
    for instruction in instructions:
        for constant_offset in _constant_offsets(instruction):
            parameter_offset = constant_offset - base["base_constant_offset"]
            if 0 <= parameter_offset < ctypes.sizeof(_AttentionParams):
                parameter_loads.append(
                    {
                        "instruction_offset": instruction["offset"],
                        "opcode": instruction["opcode"],
                        "constant_offset": constant_offset,
                        "parameter_byte_offset": parameter_offset,
                        "source_field": _field_for_offset(parameter_offset),
                        "destination_registers": _registers(instruction["operands"].split(",", 1)[0]),
                    }
                )
    memory_instructions = [
        instruction for instruction in instructions if _base_opcode(instruction["opcode"]) in MEMORY_BASES
    ]
    memory_by_space = {
        "global_or_global_to_shared": [
            instruction
            for instruction in memory_instructions
            if _base_opcode(instruction["opcode"]) in GLOBAL_MEMORY_BASES
        ],
        "shared": [
            instruction
            for instruction in memory_instructions
            if _base_opcode(instruction["opcode"]) in SHARED_MEMORY_BASES
        ],
        "generic": [
            instruction
            for instruction in memory_instructions
            if _base_opcode(instruction["opcode"]) in GENERIC_MEMORY_BASES
        ],
    }
    slices = _address_taint_slices(instructions, base["base_constant_offset"])
    cfg_slices, cfg_graph = _cfg_address_taint_slices(instructions, base["base_constant_offset"])
    call_string_slices, call_string_graph = _call_string_address_taint_slices(
        instructions, base["base_constant_offset"]
    )
    target_loads = {
        field: [
            load
            for load in parameter_loads
            if load["parameter_byte_offset"] == field_offset and load["opcode"].startswith("ULDC.64")
        ]
        for field, field_offset in TARGET_POINTER_FIELDS.items()
    }
    linked_counts = Counter(field for item in slices for field in item["source_parameter_fields"])
    cfg_linked_counts = Counter(field for item in cfg_slices for field in item["source_parameter_fields"])
    call_string_linked_counts = Counter(
        field for item in call_string_slices for field in item["source_parameter_fields"]
    )
    access_classification = _opcode_text_access_classification(call_string_slices)
    field_accesses = access_classification["parameter_field_links_by_class"]
    real_role_records = []
    for instruction in instructions:
        base_opcode = _base_opcode(instruction["opcode"])
        if base_opcode not in PROPOSED_MEMORY_OPERAND_ROLES:
            continue
        memories = re.findall(r"\[([^]]+)\]", instruction["operands"])
        actual_roles = [list(role[:2]) for role in _address_operand_specs(instruction["opcode"], memories)]
        expected_roles = PROPOSED_MEMORY_OPERAND_ROLES[base_opcode]
        real_role_records.append(
            {
                "instruction_offset": instruction["offset"],
                "opcode": instruction["opcode"],
                "base_opcode": base_opcode,
                "actual_roles": actual_roles,
                "expected_roles": expected_roles,
                "matches": actual_roles == expected_roles,
            }
        )
    real_role_counts = Counter(record["base_opcode"] for record in real_role_records)
    real_width_records = [
        {
            "opcode": instruction["opcode"],
            "instruction_offset": instruction["offset"],
            "decoded_width_bytes": _memory_width_bytes(instruction["opcode"]),
            "proposed_width_bytes": PROPOSED_MEMORY_WIDTHS[instruction["opcode"]],
            "matches": _memory_width_bytes(instruction["opcode"])
            == PROPOSED_MEMORY_WIDTHS[instruction["opcode"]],
        }
        for instruction in instructions
        if instruction["opcode"] in PROPOSED_MEMORY_WIDTHS
    ]
    real_width_counts = Counter(record["opcode"] for record in real_width_records)
    checks = {
        "cubin_hash_matches_attestation": _file_sha256(cubin) == nsight_certificate["cupti_module"]["cubin_sha256"],
        "kernel_matches_attestation": kernel == nsight_certificate["details"]["kernel_name"],
        "sass_hash_matches_attestation": summary["canonical_sha256"]
        == nsight_certificate["loaded_cubin_function_sass"]["canonical_sha256"],
        "instruction_count_matches_attestation": summary["instruction_count"]
        == nsight_certificate["loaded_cubin_function_sass"]["instruction_count"],
        "nsight_certificate_valid": verify_nsight_launch_certificate(nsight_certificate)["valid"],
        "attention_parameter_certificate_valid": verify_attention_parameter_certificate(attention_certificate)["valid"],
        "sass_semantics_certificate_valid": verify_sass_semantics_certificate(sass_semantics_certificate)["valid"],
        "real_opcode_operands_match_proposed_role_table": bool(real_role_records)
        and all(record["matches"] for record in real_role_records)
        and set(real_role_counts) == set(PROPOSED_MEMORY_OPERAND_ROLES)
        and sass_semantics_certificate["proposed_memory_operand_roles"] == PROPOSED_MEMORY_OPERAND_ROLES,
        "real_opcode_widths_match_proposed_width_table": bool(real_width_records)
        and all(record["matches"] for record in real_width_records)
        and set(real_width_counts) == set(PROPOSED_MEMORY_WIDTHS)
        and sass_semantics_certificate["proposed_memory_widths_bytes"] == PROPOSED_MEMORY_WIDTHS,
        "parameter_base_unique": base["candidate_count"] == 1,
        "target_pointer_fields_loaded": all(target_loads.values()),
        "global_memory_operations_present": bool(memory_by_space["global_or_global_to_shared"]),
        "dependency_barriers_excluded": all(_base_opcode(item["opcode"]) != "LDGDEPBAR" for item in memory_instructions),
        "linear_syntactic_address_links_present": all(linked_counts[field] > 0 for field in TARGET_POINTER_FIELDS),
        "cfg_syntactic_address_links_present": all(cfg_linked_counts[field] > 0 for field in TARGET_POINTER_FIELDS),
        "call_string_syntactic_address_links_present": all(
            call_string_linked_counts[field] > 0 for field in TARGET_POINTER_FIELDS
        ),
        "call_string_overflow_abstraction_recorded": call_string_graph["call_overflow_abstraction"]
        == "Drop oldest return site and retain the newest at the fixed depth.",
        "call_string_returns_resolved": call_string_graph["unresolved_return_context_count"] == 0,
        "target_fields_have_opcode_text_access_classes": all(field in field_accesses for field in TARGET_POINTER_FIELDS),
        "qkv_have_read_only_opcode_text_links": all(
            set(field_accesses[field]) == {"candidate_read"}
            for field in ("query_ptr", "key_ptr", "value_ptr")
        ),
        "output_fields_have_candidate_write_links": all(
            field_accesses[field].get("candidate_write", 0) > 0
            for field in ("output_ptr", "output_accum_ptr")
        ),
        "opcode_text_access_classes_bounded": set(access_classification["address_operand_count_by_class"])
        <= {"candidate_read", "candidate_write", "candidate_read_write"},
        "direct_branch_targets_resolved": cfg_graph["unresolved_direct_targets"] == 0,
        "context_insensitive_return_edges_present": cfg_graph["context_insensitive_return_edges"]
        == cfg_graph["returns"] * cfg_graph["direct_call_fallthroughs"],
        "barrier_controls_lexically_matched": cfg_graph["lexically_matched_bssy_bsync"]
        == cfg_graph["bssy_instructions"]
        == cfg_graph["bsync_instructions"]
        and cfg_graph["unmatched_barrier_controls"] == 0,
        "no_indirect_transfers_observed": cfg_graph["unresolved_indirect_transfers"] == 0,
    }
    body = {
        "scope": "Syntactic SASS parameter-to-address provenance with linear, context-insensitive, and bounded call-string fixed points plus lexical barrier-token edges; unbounded call stacks, hardware reconvergence, predicate truth, instruction semantics, access direction, bounds, and hardware behavior remain incomplete.",
        "kernel_name": kernel,
        "cubin_sha256": _file_sha256(cubin),
        "sass_semantics_certificate_sha256": sass_semantics_certificate["certificate_sha256"],
        "real_opcode_width_check": {
            "instruction_count": len(real_width_records),
            "instruction_count_by_opcode": dict(sorted(real_width_counts.items())),
            "mismatch_count": sum(not record["matches"] for record in real_width_records),
            "mismatches": [record for record in real_width_records if not record["matches"]],
            "proposed_width_table_bytes": PROPOSED_MEMORY_WIDTHS,
        },
        "real_opcode_operand_role_check": {
            "instruction_count": len(real_role_records),
            "instruction_count_by_base_opcode": dict(sorted(real_role_counts.items())),
            "mismatch_count": sum(not record["matches"] for record in real_role_records),
            "mismatches": [record for record in real_role_records if not record["matches"]],
            "proposed_role_table": PROPOSED_MEMORY_OPERAND_ROLES,
        },
        "sass": summary,
        "parameter_layout_size_bytes": ctypes.sizeof(_AttentionParams),
        "parameter_base": base,
        "parameter_base_alignment_assumption_bytes": 16,
        "target_pointer_loads": target_loads,
        "parameter_load_count": len(parameter_loads),
        "parameter_fields_observed": sorted({load["source_field"] for load in parameter_loads if load["source_field"]}),
        "memory_opcode_histogram": dict(sorted(Counter(item["opcode"] for item in memory_instructions).items())),
        "memory_instruction_count": len(memory_instructions),
        "memory_spaces": {
            name: {
                "instruction_count": len(items),
                "opcode_histogram": dict(sorted(Counter(item["opcode"] for item in items).items())),
            }
            for name, items in memory_by_space.items()
        },
        "linear_analysis": {
            "syntactic_memory_address_slice_count": len(slices),
            "address_slices_with_parameter_fields": sum(bool(item["source_parameter_fields"]) for item in slices),
            "syntactic_memory_links_by_field": dict(sorted(linked_counts.items())),
        },
        "direct_cfg_analysis": {
            **cfg_graph,
            "syntactic_memory_address_slice_count": len(cfg_slices),
            "address_slices_with_parameter_fields": sum(bool(item["source_parameter_fields"]) for item in cfg_slices),
            "syntactic_memory_links_by_field": dict(sorted(cfg_linked_counts.items())),
        },
        "bounded_call_string_analysis": {
            **call_string_graph,
            "syntactic_memory_address_slice_count": len(call_string_slices),
            "address_slices_with_parameter_fields": sum(
                bool(item["source_parameter_fields"]) for item in call_string_slices
            ),
            "syntactic_memory_links_by_field": dict(sorted(call_string_linked_counts.items())),
        },
        "opcode_text_access_classification": access_classification,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "linear_text_baseline_retained": True,
        "direct_branch_cfg_reaching_definitions_established": True,
        "context_insensitive_call_return_edges_established": True,
        "bounded_call_string_dataflow_established": True,
        "call_string_depth_overflow_free": call_string_graph["abstracted_call_overflow_count"] == 0,
        "unbounded_context_sensitive_call_return_dataflow_established": False,
        "lexical_barrier_reconvergence_edges_established": True,
        "hardware_reconvergence_semantics_established": False,
        "predicate_truth_modeled": False,
        "complete_control_flow_dataflow_established": False,
        "opcode_text_access_classification_established": True,
        "real_opcode_operands_match_proposed_memory_role_table": True,
        "real_opcode_widths_match_proposed_memory_width_table": True,
        "memory_access_direction_established": False,
        "memory_bounds_established": False,
        "sass_instruction_semantics_established": False,
        "hardware_conformance_established": False,
    }
    body["certificate_sha256"] = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    return body


def verify_sass_memory_certificate(
    certificate: dict[str, Any],
    cuobjdump: str | None = None,
    cubin: Path | None = None,
    nsight_certificate: dict[str, Any] | None = None,
    attention_certificate: dict[str, Any] | None = None,
    sass_semantics_certificate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body = {key: value for key, value in certificate.items() if key != "certificate_sha256"}
    hash_valid = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest() == certificate.get(
        "certificate_sha256"
    )
    checks_consistent = certificate.get("all_checks_pass") == all(certificate.get("checks", {}).values())
    boundaries_preserved = (
        certificate.get("linear_text_baseline_retained") is True
        and certificate.get("direct_branch_cfg_reaching_definitions_established") is True
        and certificate.get("context_insensitive_call_return_edges_established") is True
        and certificate.get("bounded_call_string_dataflow_established") is True
        and certificate.get("call_string_depth_overflow_free")
        == (certificate.get("bounded_call_string_analysis", {}).get("abstracted_call_overflow_count") == 0)
        and certificate.get("unbounded_context_sensitive_call_return_dataflow_established") is False
        and certificate.get("lexical_barrier_reconvergence_edges_established") is True
        and certificate.get("hardware_reconvergence_semantics_established") is False
        and certificate.get("predicate_truth_modeled") is False
        and certificate.get("complete_control_flow_dataflow_established") is False
        and certificate.get("opcode_text_access_classification_established") is True
        and certificate.get("real_opcode_operands_match_proposed_memory_role_table") is True
        and certificate.get("real_opcode_widths_match_proposed_memory_width_table") is True
        and certificate.get("memory_access_direction_established") is False
        and certificate.get("memory_bounds_established") is False
        and certificate.get("sass_instruction_semantics_established") is False
        and certificate.get("hardware_conformance_established") is False
    )
    histogram = certificate.get("memory_opcode_histogram", {})
    spaces = certificate.get("memory_spaces", {})
    opcode_invariants_valid = (
        sum(histogram.values()) == certificate.get("memory_instruction_count")
        and all(_base_opcode(opcode) in MEMORY_BASES for opcode in histogram)
        and all(_base_opcode(opcode) != "LDGDEPBAR" for opcode in histogram)
        and sum(space.get("instruction_count", 0) for space in spaces.values())
        == certificate.get("memory_instruction_count")
    )
    access = certificate.get("opcode_text_access_classification", {})
    access_labels = set(access.get("address_operand_count_by_class", {}))
    role_check = certificate.get("real_opcode_operand_role_check", {})
    width_check = certificate.get("real_opcode_width_check", {})
    access_classification_valid = (
        access_labels <= {"candidate_read", "candidate_write", "candidate_read_write"}
        and all(
            field in access.get("parameter_field_links_by_class", {}) for field in TARGET_POINTER_FIELDS
        )
        and role_check.get("mismatch_count") == 0
        and role_check.get("mismatches") == []
        and role_check.get("proposed_role_table") == PROPOSED_MEMORY_OPERAND_ROLES
        and role_check.get("instruction_count")
        == sum(role_check.get("instruction_count_by_base_opcode", {}).values())
        and width_check.get("mismatch_count") == 0
        and width_check.get("mismatches") == []
        and width_check.get("proposed_width_table_bytes") == PROPOSED_MEMORY_WIDTHS
        and width_check.get("instruction_count")
        == sum(width_check.get("instruction_count_by_opcode", {}).values())
    )
    replay_available = all(
        value is not None
        for value in (cuobjdump, cubin, nsight_certificate, attention_certificate, sass_semantics_certificate)
    )
    input_certificates_valid = False
    replay_matches = False
    if replay_available:
        input_certificates_valid = (
            verify_nsight_launch_certificate(nsight_certificate)["valid"]
            and verify_attention_parameter_certificate(attention_certificate)["valid"]
            and verify_sass_semantics_certificate(sass_semantics_certificate)["valid"]
        )
    if replay_available:
        rebuilt = build_sass_memory_certificate(
            cuobjdump,
            cubin,
            certificate["kernel_name"],
            nsight_certificate,
            attention_certificate,
            sass_semantics_certificate,
        )
        replay_matches = rebuilt == certificate
    return {
        "valid": bool(
            hash_valid
            and checks_consistent
            and boundaries_preserved
            and certificate.get("all_checks_pass")
            and opcode_invariants_valid
            and access_classification_valid
            and input_certificates_valid
            and replay_matches
        ),
        "certificate_hash_valid": hash_valid,
        "checks_consistent": checks_consistent,
        "boundaries_preserved": boundaries_preserved,
        "opcode_invariants_valid": opcode_invariants_valid,
        "access_classification_valid": access_classification_valid,
        "input_certificates_valid": input_certificates_valid,
        "replay_available": replay_available,
        "replay_matches": replay_matches,
    }
