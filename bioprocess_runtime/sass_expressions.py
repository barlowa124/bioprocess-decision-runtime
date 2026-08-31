from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path
from typing import Any

from .attention_bounds import verify_attention_logical_bounds_certificate
from .nsight_attestation import _instruction_summary, _parse_cuobjdump_sass
from .sass_memory import (
    TARGET_POINTER_FIELDS,
    _address_operand_specs,
    _base_opcode,
    _call_string_address_taint_slices,
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


def _linear_expression_snapshots(
    instructions: list[dict[str, Any]], parameter_base: int
) -> tuple[dict[int, dict[str, str]], _NodeRegistry]:
    registry = _NodeRegistry()
    registers: dict[str, str] = {}
    snapshots: dict[int, dict[str, str]] = {}
    for instruction in instructions:
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
            snapshots[instruction["offset"]] = {
                register: registers.get(
                    register,
                    registry.add({"kind": "entry_register", "register": register}),
                )
                for memory in memories
                for register in _registers(memory)
            }
        destination = re.match(r"((?:UR|R)\d+)\b", instruction["operands"])
        if not destination or _base_opcode(instruction["opcode"]) in {
            "ST",
            "STG",
            "STS",
            "ATOM",
            "ATOMS",
            "RED",
            "BRA",
            "CALL",
            "RET",
            "EXIT",
        }:
            continue
        destination_register = destination.group(1)
        remaining = instruction["operands"][destination.end() :]
        source_nodes = [
            registers.get(register, registry.add({"kind": "entry_register", "register": register}))
            for register in _source_registers_for_opcode(instruction["opcode"], remaining)
        ]
        parameter_fields = []
        for constant_offset in _constant_offsets(instruction):
            parameter_offset = constant_offset - parameter_base
            field = _field_for_offset(parameter_offset) if 0 <= parameter_offset < 264 else None
            if field:
                parameter_fields.append(field)
        node = registry.add(
            {
                "kind": "instruction_definition",
                "instruction_offset": instruction["offset"],
                "opcode": instruction["opcode"],
                "predicate": instruction.get("predicate"),
                "source_nodes": source_nodes,
                "parameter_fields": sorted(set(parameter_fields)),
            }
        )
        for increment in range(_destination_register_count(instruction["opcode"])):
            prefix = "UR" if destination_register.startswith("UR") else "R"
            register = f"{prefix}{int(destination_register[len(prefix):]) + increment}"
            registers[register] = registry.add(
                {
                    "kind": "instruction_output",
                    "definition_node": node,
                    "output_index": increment,
                }
            )
    return snapshots, registry


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
        if node["kind"] == "entry_register"
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
    snapshots, registry = _linear_expression_snapshots(instructions, parameter_base)
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
        for memory_slice, operand in sorted(candidates, key=lambda item: item[0]["instruction_offset"]):
            roots = [
                snapshots[memory_slice["instruction_offset"]][register]
                for register in operand["address_registers"]
            ]
            nodes = _reachable_nodes(roots, registry)
            represented_fields = sorted(
                {parameter_field for node in nodes for parameter_field in node.get("parameter_fields", [])}
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
                "root_nodes": roots,
                "node_count": len(nodes),
                "represented_parameter_fields": represented_fields,
                "target_field_represented": field in represented_fields,
                "unsupported_or_entry_node_count": len(unsupported_nodes),
                "expression_nodes": nodes,
                "closed_supported_formula": not unsupported_nodes,
            }
            selection = candidate_selection
            if candidate_selection["target_field_represented"]:
                break
        selections.append(selection)
    checks = {
        "cubin_hash_matches": hashlib.sha256(cubin.read_bytes()).hexdigest()
        == sass_memory_certificate["cubin_sha256"],
        "sass_hash_matches": summary["canonical_sha256"]
        == sass_memory_certificate["sass"]["canonical_sha256"],
        "kernel_matches": kernel == sass_memory_certificate["kernel_name"],
        "logical_bounds_certificate_valid": verify_attention_logical_bounds_certificate(
            logical_bounds_certificate
        )["valid"],
        "all_target_skeletons_available": all(selection["available"] for selection in selections),
        "all_target_fields_represented": all(
            selection.get("target_field_represented", False) for selection in selections
        ),
    }
    body = {
        "scope": "One hash-consed linear-definition address-expression DAG selected per target field from bounded call-string memory operands; unsupported entry/register operations remain explicit and no closed SASS formula or logical correspondence is claimed.",
        "kernel_name": kernel,
        "cubin_sha256": hashlib.sha256(cubin.read_bytes()).hexdigest(),
        "sass_canonical_sha256": summary["canonical_sha256"],
        "sass_memory_certificate_sha256": sass_memory_certificate["certificate_sha256"],
        "logical_bounds_certificate_sha256": logical_bounds_certificate["certificate_sha256"],
        "call_string_context_summary": call_graph,
        "selections": selections,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "selected_sass_address_expression_dags_established": True,
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
        "selections": selections,
        "total_selections": len(selections),
        "available_selections": sum(selection["available"] for selection in selections),
        "closed_supported_formulas": sum(
            selection.get("closed_supported_formula", False) for selection in selections
        ),
        "selected_sass_address_expression_dags_established": True,
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


def verify_sass_expression_certificate(certificate: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in certificate.items() if key != "certificate_sha256"}
    hash_valid = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest() == certificate.get(
        "certificate_sha256"
    )
    checks_consistent = certificate.get("all_checks_pass") == all(certificate.get("checks", {}).values())
    boundaries_preserved = (
        certificate.get("selected_sass_address_expression_dags_established") is True
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
            and node_hashes_valid
            and certificate.get("all_checks_pass")
        ),
        "certificate_hash_valid": hash_valid,
        "checks_consistent": checks_consistent,
        "boundaries_preserved": boundaries_preserved,
        "closed_formulas_consistent": closed_formulas_consistent,
        "node_hashes_valid": node_hashes_valid,
    }
