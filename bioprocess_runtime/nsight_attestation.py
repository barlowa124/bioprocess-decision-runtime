from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from .serialization import canonical_json


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_ncu_import(ncu: str, report: Path, arguments: list[str]) -> str:
    completed = subprocess.run(
        [ncu, "--import", str(report), *arguments], check=True, capture_output=True, text=True, timeout=120
    )
    return completed.stdout


def _parse_details(output: str) -> dict[str, Any]:
    rows = list(csv.DictReader(io.StringIO(output)))
    if not rows:
        raise ValueError("Nsight details page contained no metric rows")
    identities = {
        (
            row["Process ID"],
            row["Kernel Name"],
            row["Context"],
            row["Stream"],
            row["Block Size"],
            row["Grid Size"],
            row["CC"],
        )
        for row in rows
    }
    if len(identities) != 1:
        raise ValueError("Expected exactly one profiled launch identity")
    process_id, kernel, context, stream, block, grid, capability = identities.pop()
    metrics = {
        row["Metric Name"]: {"value": row["Metric Value"], "unit": row["Metric Unit"]}
        for row in rows
        if row["Metric Name"]
    }
    return {
        "process_id": int(process_id),
        "kernel_name": kernel,
        "context": int(context),
        "stream": int(stream),
        "block_size": block,
        "grid_size": grid,
        "compute_capability": capability,
        "metrics": metrics,
    }


def _normalize_operands(opcode: str, operands: str, base_address: int) -> str:
    value = operands.strip().rstrip(";").strip()
    if opcode == "BRA":
        match = re.fullmatch(r"0x([0-9a-fA-F]+)", value)
        if match:
            target = int(match.group(1), 16)
            if target >= base_address:
                value = f"0x{target - base_address:x}"
    return re.sub(r"\s+", "", value)


def _parse_nsight_sass(output: str) -> list[dict[str, Any]]:
    raw = []
    expression = re.compile(r"^0x([0-9a-fA-F]+)\s+(?:(@[!A-Z0-9]+)\s+)?([A-Z][A-Z0-9_.]*)\s*(.*?)\s*$")
    for line in output.splitlines():
        match = expression.match(line.strip())
        if match:
            raw.append((int(match.group(1), 16), match.group(2), match.group(3), match.group(4)))
    if not raw:
        raise ValueError("Nsight source page contained no SASS instructions")
    base = raw[0][0]
    return [
        {
            "offset": address - base,
            "predicate": predicate,
            "opcode": opcode,
            "operands": _normalize_operands(opcode, operands, base),
        }
        for address, predicate, opcode, operands in raw
    ]


def _parse_cuobjdump_sass(output: str) -> list[dict[str, Any]]:
    expression = re.compile(
        r"/\*([0-9a-fA-F]+)\*/\s+(?:(@[!A-Z0-9]+)\s+)?([A-Z][A-Z0-9_.]*)\s*(.*?)\s*;"
    )
    instructions = []
    for match in expression.finditer(output):
        opcode = match.group(3)
        instructions.append(
            {
                "offset": int(match.group(1), 16),
                "predicate": match.group(2),
                "opcode": opcode,
                "operands": _normalize_operands(opcode, match.group(4), 0),
            }
        )
    if not instructions:
        raise ValueError("cuobjdump output contained no SASS instructions")
    return instructions


def _instruction_summary(instructions: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "instruction_count": len(instructions),
        "first_offset": instructions[0]["offset"],
        "last_offset": instructions[-1]["offset"],
        "opcode_histogram": dict(sorted(Counter(item["opcode"] for item in instructions).items())),
        "canonical_sha256": hashlib.sha256(canonical_json(instructions).encode("utf-8")).hexdigest(),
    }


def build_nsight_launch_certificate(
    report_path: Path,
    binding_path: Path,
    cuda_summary_path: Path,
    cupti_report_path: Path,
    cupti_artifact_directory: Path,
) -> dict[str, Any]:
    ncu = shutil.which("ncu")
    cuobjdump = shutil.which("cuobjdump")
    if ncu is None or cuobjdump is None:
        raise RuntimeError("ncu and cuobjdump are required")
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    binding_body = {key: value for key, value in binding.items() if key != "binding_sha256"}
    binding_valid = hashlib.sha256(canonical_json(binding_body).encode("utf-8")).hexdigest() == binding.get(
        "binding_sha256"
    )
    cuda_summary = json.loads(cuda_summary_path.read_text(encoding="utf-8"))
    cupti_report = json.loads(cupti_report_path.read_text(encoding="utf-8"))
    details_output = _run_ncu_import(ncu, report_path, ["--page", "details", "--print-kernel-base", "mangled", "--csv"])
    source_output = _run_ncu_import(
        ncu, report_path, ["--page", "source", "--print-source", "sass", "--print-kernel-base", "mangled"]
    )
    session_output = _run_ncu_import(ncu, report_path, ["--page", "session"])
    details = _parse_details(details_output)
    launch_instructions = _parse_nsight_sass(source_output)
    static_hash = cuda_summary["profiled_symbol_disassembly"]["embedded_image_sha256"]
    matching_modules = [module for module in cupti_report["modules"] if module["cubin_sha256"] == static_hash]
    if len(matching_modules) != 1:
        raise ValueError("Expected one CUPTI module matching the statically extracted image")
    cubin_path = cupti_artifact_directory / matching_modules[0]["artifact"]
    kernel = details["kernel_name"]
    static_output = subprocess.run(
        [cuobjdump, "--dump-sass", "--function", kernel, str(cubin_path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout
    static_instructions = _parse_cuobjdump_sass(static_output)
    launch_summary = _instruction_summary(launch_instructions)
    static_summary = _instruction_summary(static_instructions)
    compact_session = re.sub(r"\s+", "", session_output)
    checks = {
        "binding_hash_valid": binding_valid,
        "process_id_matches": details["process_id"] == binding["process_id"],
        "kernel_matches_profile": kernel == cuda_summary["profiled_symbol_disassembly"]["profiled_kernel_name"],
        "model_state_matches_profile": binding["model_state_sha256"]
        == cuda_summary["execution_binding"]["model_state_sha256"],
        "input_matches_profile": binding["input_ids_tensor"]["sha256"]
        == cuda_summary["execution_binding"]["input_ids_tensor"]["sha256"],
        "output_matches_profile": binding["output_logits_tensor"]["sha256"]
        == cuda_summary["execution_binding"]["output_logits_tensor"]["sha256"],
        "selected_token_matches_profile": binding["selected_token_id"]
        == cuda_summary["execution_binding"]["selected_token_id"],
        "binding_nvtx_range_matches": binding["nvtx_range"] == "gemma_bound_forward",
        "nvtx_filter_recorded": "--nvtx-includegemma_bound_forward/" in compact_session,
        "kernel_filter_recorded": f"--kernel-name{kernel}" in compact_session,
        "single_launch_filter_recorded": "--launch-count1" in compact_session,
        "cupti_module_hash_matches_static_image": _file_sha256(cubin_path) == static_hash,
        "launch_sass_matches_loaded_cubin_function": launch_instructions == static_instructions,
    }
    body = {
        "scope": "Launch-specific Nsight SASS bound by process ID and NVTX range to a Gemma execution, then compared with the matching CUPTI-loaded cubin function; not hardware-semantic proof.",
        "report_sha256": _file_sha256(report_path),
        "binding_sha256": binding["binding_sha256"],
        "session_output_sha256": hashlib.sha256(session_output.encode("utf-8")).hexdigest(),
        "source_output_sha256": hashlib.sha256(source_output.encode("utf-8")).hexdigest(),
        "details": details,
        "execution_binding": binding_body,
        "cupti_module": {key: matching_modules[0][key] for key in ("module_id", "cubin_size", "cubin_sha256")},
        "launch_sass": launch_summary,
        "loaded_cubin_function_sass": static_summary,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "limitations": [
            "Nsight and CUPTI are NVIDIA tooling and are not independent hardware implementations.",
            "Matching SASS establishes instruction-text identity, not correctness of instruction semantics.",
            "Only one selected kernel launch is attested by this certificate.",
        ],
    }
    body["certificate_sha256"] = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    return body


def verify_nsight_launch_certificate(certificate: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in certificate.items() if key != "certificate_sha256"}
    hash_valid = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest() == certificate.get(
        "certificate_sha256"
    )
    checks_consistent = certificate.get("all_checks_pass") == all(certificate.get("checks", {}).values())
    sass_claim_consistent = (
        certificate.get("checks", {}).get("launch_sass_matches_loaded_cubin_function")
        == (
            certificate.get("launch_sass", {}).get("canonical_sha256")
            == certificate.get("loaded_cubin_function_sass", {}).get("canonical_sha256")
        )
    )
    return {
        "valid": bool(hash_valid and checks_consistent and sass_claim_consistent and certificate.get("all_checks_pass")),
        "certificate_hash_valid": hash_valid,
        "checks_consistent": checks_consistent,
        "sass_claim_consistent": sass_claim_consistent,
    }
