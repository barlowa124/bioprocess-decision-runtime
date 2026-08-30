from __future__ import annotations

import ctypes
import hashlib
import re
import subprocess
from collections import Counter
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
LOAD_BASES = {"LDG", "LDS", "LD"}


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


def _next_register(register: str) -> str:
    prefix = "UR" if register.startswith("UR") else "R"
    return f"{prefix}{int(register[len(prefix):]) + 1}"


def _address_taint_slices(instructions: list[dict[str, Any]], parameter_base: int) -> list[dict[str, Any]]:
    taint: dict[str, set[str]] = {}
    slices = []
    for instruction in instructions:
        opcode = instruction["opcode"]
        base_opcode = _base_opcode(opcode)
        operands = instruction["operands"]
        memories = re.findall(r"\[([^]]+)\]", operands)
        if base_opcode in MEMORY_BASES and memories:
            address_registers = sorted({register for memory in memories for register in _registers(memory)})
            fields = sorted({field for register in address_registers for field in taint.get(register, set())})
            slices.append(
                {
                    "instruction_offset": instruction["offset"],
                    "opcode": opcode,
                    "address_registers": address_registers,
                    "source_parameter_fields": fields,
                }
            )
        destination = re.match(r"((?:UR|R)\d+)\b", operands)
        if not destination or base_opcode in {"ST", "STG", "STS", "ATOM", "ATOMS", "RED"} or opcode.startswith(("BRA", "CALL", "RET", "EXIT")):
            continue
        destination_register = destination.group(1)
        if base_opcode in LOAD_BASES:
            taint[destination_register] = set()
            if ".64" in opcode:
                taint[_next_register(destination_register)] = set()
            continue
        remaining = operands[destination.end() :]
        source_fields = {field for register in _registers(remaining) for field in taint.get(register, set())}
        for constant_offset in _constant_offsets(instruction):
            parameter_offset = constant_offset - parameter_base
            if 0 <= parameter_offset < ctypes.sizeof(_AttentionParams):
                field = _field_for_offset(parameter_offset)
                if field:
                    source_fields.add(field)
        taint[destination_register] = source_fields
        if ".64" in opcode:
            taint[_next_register(destination_register)] = set(source_fields)
    return slices


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
    target_loads = {
        field: [
            load
            for load in parameter_loads
            if load["parameter_byte_offset"] == field_offset and load["opcode"].startswith("ULDC.64")
        ]
        for field, field_offset in TARGET_POINTER_FIELDS.items()
    }
    linked_counts = Counter(field for item in slices for field in item["source_parameter_fields"])
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
        "syntactic_address_links_present": all(linked_counts[field] > 0 for field in TARGET_POINTER_FIELDS),
    }
    body = {
        "scope": "Syntactic SASS constant-space and register-text provenance from source-reconstructed attention parameter fields to memory address operands; not control-flow-complete dataflow, instruction semantics, access-direction, bounds, or hardware proof.",
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
        "syntactic_memory_address_slice_count": len(slices),
        "address_slices_with_parameter_fields": sum(bool(item["source_parameter_fields"]) for item in slices),
        "syntactic_memory_links_by_field": dict(sorted(linked_counts.items())),
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "linear_text_taint_only": True,
        "control_flow_dataflow_established": False,
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
        certificate.get("linear_text_taint_only") is True
        and certificate.get("control_flow_dataflow_established") is False
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
