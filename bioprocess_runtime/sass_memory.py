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


def _memory_slice(instruction: dict[str, Any], taint: dict[str, set[str]]) -> dict[str, Any] | None:
    memories = re.findall(r"\[([^]]+)\]", instruction["operands"])
    if _base_opcode(instruction["opcode"]) not in MEMORY_BASES or not memories:
        return None
    address_registers = sorted({register for memory in memories for register in _registers(memory)})
    return {
        "instruction_offset": instruction["offset"],
        "opcode": instruction["opcode"],
        "address_registers": address_registers,
        "source_parameter_fields": sorted(
            {field for register in address_registers for field in taint.get(register, set())}
        ),
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


def _cfg_address_taint_slices(
    instructions: list[dict[str, Any]], parameter_base: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    instruction_successors, graph = _cfg_successors(instructions)
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


def build_sass_memory_certificate(
    cuobjdump: str,
    cubin: Path,
    kernel: str,
    nsight_certificate: dict[str, Any],
    attention_certificate: dict[str, Any],
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
    checks = {
        "cubin_hash_matches_attestation": _file_sha256(cubin) == nsight_certificate["cupti_module"]["cubin_sha256"],
        "kernel_matches_attestation": kernel == nsight_certificate["details"]["kernel_name"],
        "sass_hash_matches_attestation": summary["canonical_sha256"]
        == nsight_certificate["loaded_cubin_function_sass"]["canonical_sha256"],
        "instruction_count_matches_attestation": summary["instruction_count"]
        == nsight_certificate["loaded_cubin_function_sass"]["instruction_count"],
        "nsight_certificate_valid": verify_nsight_launch_certificate(nsight_certificate)["valid"],
        "attention_parameter_certificate_valid": verify_attention_parameter_certificate(attention_certificate)["valid"],
        "parameter_base_unique": base["candidate_count"] == 1,
        "target_pointer_fields_loaded": all(target_loads.values()),
        "global_memory_operations_present": bool(memory_by_space["global_or_global_to_shared"]),
        "dependency_barriers_excluded": all(_base_opcode(item["opcode"]) != "LDGDEPBAR" for item in memory_instructions),
        "linear_syntactic_address_links_present": all(linked_counts[field] > 0 for field in TARGET_POINTER_FIELDS),
        "cfg_syntactic_address_links_present": all(cfg_linked_counts[field] > 0 for field in TARGET_POINTER_FIELDS),
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
        "scope": "Syntactic SASS parameter-to-address provenance with fixed-point direct branches, context-insensitive return over-approximation, and lexical barrier-token reconvergence edges; context-sensitive calls, hardware reconvergence, predicate truth, instruction semantics, access direction, bounds, and hardware behavior remain incomplete.",
        "kernel_name": kernel,
        "cubin_sha256": _file_sha256(cubin),
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
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "linear_text_baseline_retained": True,
        "direct_branch_cfg_reaching_definitions_established": True,
        "context_insensitive_call_return_edges_established": True,
        "context_sensitive_call_return_dataflow_established": False,
        "lexical_barrier_reconvergence_edges_established": True,
        "hardware_reconvergence_semantics_established": False,
        "predicate_truth_modeled": False,
        "complete_control_flow_dataflow_established": False,
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
        and certificate.get("context_sensitive_call_return_dataflow_established") is False
        and certificate.get("lexical_barrier_reconvergence_edges_established") is True
        and certificate.get("hardware_reconvergence_semantics_established") is False
        and certificate.get("predicate_truth_modeled") is False
        and certificate.get("complete_control_flow_dataflow_established") is False
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
    replay_available = all(value is not None for value in (cuobjdump, cubin, nsight_certificate, attention_certificate))
    input_certificates_valid = False
    replay_matches = False
    if replay_available:
        input_certificates_valid = verify_nsight_launch_certificate(nsight_certificate)["valid"] and verify_attention_parameter_certificate(attention_certificate)["valid"]
    if replay_available:
        rebuilt = build_sass_memory_certificate(
            cuobjdump,
            cubin,
            certificate["kernel_name"],
            nsight_certificate,
            attention_certificate,
        )
        replay_matches = rebuilt == certificate
    return {
        "valid": bool(
            hash_valid
            and checks_consistent
            and boundaries_preserved
            and certificate.get("all_checks_pass")
            and opcode_invariants_valid
            and input_certificates_valid
            and replay_matches
        ),
        "certificate_hash_valid": hash_valid,
        "checks_consistent": checks_consistent,
        "boundaries_preserved": boundaries_preserved,
        "opcode_invariants_valid": opcode_invariants_valid,
        "input_certificates_valid": input_certificates_valid,
        "replay_available": replay_available,
        "replay_matches": replay_matches,
    }
