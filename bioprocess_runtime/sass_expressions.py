from __future__ import annotations

import ctypes
import hashlib
import re
import subprocess
from collections import deque
from pathlib import Path
from typing import Any

from .attention_bounds import verify_attention_logical_bounds_certificate
from .nsight_attestation import _instruction_summary, _parse_cuobjdump_sass
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
from .serialization import canonical_json


SUPPORTED_EXPRESSION_OPCODES = {
    "MOV",
    "IMAD",
    "IMAD.WIDE",
    "IMAD.WIDE.U32",
    "IMAD.IADD",
    "IMAD.SHL.U32",
    "IADD3",
    "LEA",
    "LEA.HI",
    "LEA.HI.X",
    "ULEA",
    "ULEA.HI",
    "ULEA.HI.X",
    "ULDC",
    "ULDC.64",
}


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
    for output_index in range(_destination_register_count(instruction["opcode"])):
        register = f"{prefix}{first + output_index}"
        definition = frozenset(
            {_definition_key(block_index, call_stack, instruction_index, output_index)}
        )
        if instruction.get("predicate"):
            definition |= state.get(register, frozenset())
        output[register] = definition
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
    instructions: list[dict[str, Any]], parameter_base: int, maximum_call_depth: int = 4
) -> tuple[dict[int, list[dict[str, Any]]], _NodeRegistry, dict[str, Any]]:
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
                source_definitions = {
                    register: sorted(state.get(register, {f"entry:{register}"}))
                    for register in _source_registers_for_opcode(
                        instruction["opcode"], remaining
                    )
                }
                destination_register = destination.group(1)
                for output_index in range(_destination_register_count(instruction["opcode"])):
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
                        "source_definitions": source_definitions,
                    }
            state = _transfer_definitions(instruction, instruction_index, context, state)
    registry = _NodeRegistry()
    materialized: dict[str, str] = {}

    def materialize(key: str, active: frozenset[str] = frozenset()) -> str:
        if key.startswith("entry:"):
            return registry.add({"kind": "entry_register", "register": key.split(":", 1)[1]})
        if key in active:
            spec = definition_specs[key]
            return registry.add(
                {
                    "kind": "cyclic_reaching_definition",
                    "instruction_offset": spec["instruction_offset"],
                    "output_index": spec["output_index"],
                    "block_index": spec["block_index"],
                    "call_stack": spec["call_stack"],
                }
            )
        if key in materialized:
            return materialized[key]
        spec = definition_specs[key]
        source_nodes = []
        for register, definitions in sorted(spec["source_definitions"].items()):
            alternatives = sorted(materialize(item, active | {key}) for item in definitions)
            source_nodes.append(
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
                "source_registers": sorted(spec["source_definitions"]),
                "source_nodes": source_nodes,
                "parameter_fields": sorted(set(parameter_fields)),
            }
        )
        output_node = registry.add(
            {
                "kind": "instruction_output",
                "definition_node": definition_node,
                "output_index": spec["output_index"],
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


def _unsupported_expression_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        node
        for node in nodes
        if node["kind"] in {
            "entry_register",
            "reaching_definition_join",
            "cyclic_reaching_definition",
        }
        or (
            node["kind"] == "instruction_definition"
            and node["opcode"] not in SUPPORTED_EXPRESSION_OPCODES
        )
    ]


def build_sass_expression_certificate(
    cuobjdump: str,
    cubin: Path,
    kernel: str,
    sass_memory_certificate: dict[str, Any],
    logical_bounds_certificate: dict[str, Any],
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
    slices, call_graph = _call_string_address_taint_slices(instructions, parameter_base)
    snapshots, registry, expression_graph = _call_string_expression_snapshots(
        instructions, parameter_base
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
                    "represented_parameter_fields": represented_fields,
                    "target_field_represented": field in represented_fields,
                    "unsupported_or_entry_node_count": len(unsupported_nodes),
                    "expression_nodes": nodes,
                    "closed_supported_formula": not unsupported_nodes,
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
    }
    body = {
        "scope": "One hash-consed bounded-call-string reaching-definition address-expression DAG selected per target field; ambiguous joins, cyclic definitions, unsupported operations, and entry registers remain explicit, and no closed SASS formula or logical correspondence is claimed.",
        "kernel_name": kernel,
        "cubin_sha256": hashlib.sha256(cubin.read_bytes()).hexdigest(),
        "sass_canonical_sha256": summary["canonical_sha256"],
        "sass_memory_certificate_sha256": sass_memory_certificate["certificate_sha256"],
        "logical_bounds_certificate_sha256": logical_bounds_certificate["certificate_sha256"],
        "call_string_context_summary": call_graph,
        "expression_reaching_definition_summary": expression_graph,
        "selections": selections,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "selected_sass_address_expression_dags_established": True,
        "bounded_call_string_expression_reaching_definitions_established": True,
        "expression_call_string_depth_overflow_free": expression_graph[
            "abstracted_call_overflow_count"
        ]
        == 0,
        "unbounded_context_sensitive_expression_reaching_definitions_established": False,
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
            if key != "expression_nodes"
        }
        | {
            "expression_graph_sha256": hashlib.sha256(
                canonical_json(selection.get("expression_nodes", [])).encode("utf-8")
            ).hexdigest()
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
        "selected_sass_address_expression_dags_established": True,
        "bounded_call_string_expression_reaching_definitions_established": certificate[
            "bounded_call_string_expression_reaching_definitions_established"
        ],
        "expression_call_string_depth_overflow_free": certificate[
            "expression_call_string_depth_overflow_free"
        ],
        "unbounded_context_sensitive_expression_reaching_definitions_established": False,
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
    counts_consistent = (
        summary.get("total_selections") == len(selections)
        and summary.get("available_selections") == sum(selection["available"] for selection in selections)
        and summary.get("closed_supported_formulas")
        == sum(selection.get("closed_supported_formula", False) for selection in selections)
        and summary.get("closed_supported_sass_formulas_established")
        == all(selection.get("closed_supported_formula", False) for selection in selections)
        and all(
            selection.get("closed_supported_formula")
            == (selection.get("unsupported_or_entry_node_count") == 0)
            for selection in selections
        )
    )
    boundaries_preserved = (
        summary.get("selected_sass_address_expression_dags_established") is True
        and summary.get("bounded_call_string_expression_reaching_definitions_established") is True
        and summary.get("expression_call_string_depth_overflow_free")
        == (summary.get("expression_reaching_definition_summary", {}).get("abstracted_call_overflow_count") == 0)
        and summary.get("unbounded_context_sensitive_expression_reaching_definitions_established") is False
        and summary.get("sass_effective_address_formula_bound") is False
        and summary.get("sass_to_logical_stride_correspondence_established") is False
        and summary.get("sass_effective_address_bounds_established") is False
        and summary.get("kernel_memory_safety_established") is False
    )
    return {
        "valid": bool(hash_valid and counts_consistent and boundaries_preserved and selections),
        "summary_hash_valid": hash_valid,
        "counts_consistent": counts_consistent,
        "boundaries_preserved": boundaries_preserved,
    }


def _selection_graph_consistent(selection: dict[str, Any]) -> bool:
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
    )


def verify_sass_expression_certificate(certificate: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in certificate.items() if key != "certificate_sha256"}
    hash_valid = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest() == certificate.get(
        "certificate_sha256"
    )
    checks_consistent = certificate.get("all_checks_pass") == all(certificate.get("checks", {}).values())
    boundaries_preserved = (
        certificate.get("selected_sass_address_expression_dags_established") is True
        and certificate.get("bounded_call_string_expression_reaching_definitions_established") is True
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
        == (len(_unsupported_expression_nodes(selection.get("expression_nodes", []))) == 0)
        for selection in certificate.get("selections", [])
    ) and certificate.get("closed_supported_sass_formulas_established") == all(
        selection.get("closed_supported_formula", False)
        for selection in certificate.get("selections", [])
    )
    selection_graphs_consistent = all(
        _selection_graph_consistent(selection)
        for selection in certificate.get("selections", [])
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
            and closed_formulas_consistent
            and selection_graphs_consistent
            and node_hashes_valid
            and certificate.get("all_checks_pass")
        ),
        "certificate_hash_valid": hash_valid,
        "checks_consistent": checks_consistent,
        "boundaries_preserved": boundaries_preserved,
        "closed_formulas_consistent": closed_formulas_consistent,
        "selection_graphs_consistent": selection_graphs_consistent,
        "node_hashes_valid": node_hashes_valid,
    }
