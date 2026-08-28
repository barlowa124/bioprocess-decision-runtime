from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
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


def _normalize_opcode(opcode: str) -> str:
    aliases = {
        "F2FP.BF16.F32.PACK_AB": "F2FP.BF16.PACK_AB",
        "F2FP.F16.F32.PACK_AB": "F2FP.PACK_AB",
    }
    return aliases.get(opcode, opcode)


def _normalize_operands(opcode: str, operands: str, base_address: int) -> str:
    value = operands.strip().rstrip(";").strip().replace(".reuse", "")
    if opcode.startswith("BRA") or opcode == "BSSY" or opcode.startswith("CALL."):
        targets = list(re.finditer(r"0x([0-9a-fA-F]+)", value))
        if targets:
            target_match = targets[-1]
            target = int(target_match.group(1), 16)
            if target >= base_address:
                value = value[: target_match.start()] + f"0x{target - base_address:x}" + value[target_match.end() :]
    if opcode == "BRX":
        match = re.fullmatch(r"(R\d+)[,\s]+(-?0x[0-9a-fA-F]+)", value)
        if match:
            value = f"{match.group(1)},{match.group(2)}"
    if opcode.startswith("RET."):
        match = re.fullmatch(r"(R\d+)[,\s]+0x([0-9a-fA-F]+)", value)
        if match:
            target = int(match.group(2), 16)
            value = f"{match.group(1)},0x{target - base_address:x}" if target >= base_address else f"{match.group(1)},0x{target:x}"
    if opcode.startswith("LDGSTS"):
        memories = re.findall(r"\[[^]]+\]", value)
        predicate = re.search(r"(?:^|,)\s*(P\d+)\s*(?:,|$)", value)
        if len(memories) == 2:
            value = ",".join([*memories, predicate.group(1) if predicate else ""])
    elif opcode.startswith("LDG"):
        destination = re.match(r"(R\d+)", value)
        memory = re.search(r"\[[^]]+\]", value)
        predicate = re.search(r"(?:^|,)\s*(P\d+)\s*(?:,|$)", value)
        if destination and memory and predicate:
            value = ",".join((destination.group(1), memory.group(0), predicate.group(1)))
    return re.sub(r"\s+", "", value).rstrip(",")


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
            "opcode": _normalize_opcode(opcode),
            "operands": _normalize_operands(_normalize_opcode(opcode), operands, base),
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
                "opcode": _normalize_opcode(opcode),
                "operands": _normalize_operands(_normalize_opcode(opcode), match.group(4), 0),
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


def capture_nsight_launch(
    kernel: str,
    model_path: Path,
    prompt: str,
    report_base: Path,
    binding_path: Path,
    request_path: Path,
    module_patterns: tuple[str, ...] = (),
) -> dict[str, Any]:
    ncu = shutil.which("ncu")
    if ncu is None:
        raise RuntimeError("ncu is required")
    request = {
        "scope": "Pre-execution request for one exact mangled kernel inside the Gemma NVTX range.",
        "kernel_name": kernel,
        "kernel_name_base": "mangled",
        "nvtx_range": "gemma_bound_forward/",
        "launch_count": 1,
        "section": "LaunchStats",
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "report_name": report_base.name,
        "binding_name": binding_path.name,
        "module_nvtx_patterns": list(module_patterns),
    }
    request["request_sha256"] = hashlib.sha256(canonical_json(request).encode("utf-8")).hexdigest()
    request_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_text(json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    command = [
        ncu,
        "--target-processes",
        "all",
        "--nvtx",
        "--nvtx-include",
        request["nvtx_range"],
        "--kernel-name-base",
        request["kernel_name_base"],
        "--kernel-name",
        request["kernel_name"],
        "--launch-count",
        str(request["launch_count"]),
        "--section",
        request["section"],
        "--export",
        str(report_base),
        "--force-overwrite",
        sys.executable,
        "-m",
        "bioprocess_runtime",
        "gemma-nsight-target",
        "--model-path",
        str(model_path),
        "--prompt",
        prompt,
        "--output",
        str(binding_path),
    ]
    for pattern in module_patterns:
        command.extend(("--module-nvtx-pattern", pattern))
    completed = subprocess.run(command, capture_output=True, text=True, timeout=280)
    if completed.returncode != 0:
        raise RuntimeError(f"Nsight capture failed with return code {completed.returncode}")
    report_path = report_base.with_suffix(".ncu-rep")
    if not report_path.is_file() or not binding_path.is_file():
        raise RuntimeError("Nsight capture did not produce the expected report and binding")
    return request


def _parse_elf_function_symbols(output: str) -> set[str]:
    return {
        parts[-1]
        for line in output.splitlines()
        if (parts := line.split()) and parts[0] == "STT_FUNC" and len(parts) >= 4
    }


def _locate_cupti_module_function(
    cuobjdump: str,
    kernel: str,
    cupti_report: dict[str, Any],
    artifact_directory: Path,
    preferred_hash: str | None = None,
) -> tuple[dict[str, Any], Path, str, list[dict[str, Any]]]:
    modules = cupti_report["modules"]
    if preferred_hash is not None:
        modules = sorted(modules, key=lambda item: item["cubin_sha256"] != preferred_hash)
    matches = []
    for module in modules:
        cubin_path = artifact_directory / module["artifact"]
        completed = subprocess.run(
            [cuobjdump, "--dump-sass", "--function", kernel, str(cubin_path)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        function_sections = re.findall(r"^\s*Function\s*:\s*(\S+)\s*$", completed.stdout, re.MULTILINE)
        if completed.returncode == 0 and function_sections == [kernel]:
            try:
                instructions = _parse_cuobjdump_sass(completed.stdout)
            except ValueError:
                continue
            matches.append((module, cubin_path, completed.stdout, instructions))
            if module["cubin_sha256"] == preferred_hash:
                break
    if len(matches) != 1:
        raise ValueError(f"Expected one CUPTI-loaded module containing {kernel!r}; found {len(matches)}")
    return matches[0]


def build_nsight_launch_certificate(
    report_path: Path,
    binding_path: Path,
    cuda_summary_path: Path,
    cupti_report_path: Path,
    cupti_artifact_directory: Path,
    preferred_cupti_hash: str | None = None,
    capture_request_path: Path | None = None,
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
    kernel = details["kernel_name"]
    selected_disassembly = cuda_summary.get("profiled_symbol_disassembly", {})
    preferred_hash = preferred_cupti_hash or (
        selected_disassembly.get("embedded_image_sha256")
        if selected_disassembly.get("profiled_kernel_name") == kernel
        else None
    )
    matching_module, cubin_path, static_output, static_instructions = _locate_cupti_module_function(
        cuobjdump, kernel, cupti_report, cupti_artifact_directory, preferred_hash
    )
    profile_counts = cuda_summary.get("profile", {}).get("kernel_launch_counts")
    if profile_counts is None:
        profile_counts = {
            selected_disassembly.get("profiled_kernel_name"): selected_disassembly.get("observed_launch_count")
        }
    launch_summary = _instruction_summary(launch_instructions)
    static_summary = _instruction_summary(static_instructions)
    compact_session = re.sub(r"\s+", "", session_output)
    exact_kernel_filter_visible = f"--kernel-name{kernel}" in compact_session or f"--kernel-name={kernel}" in compact_session
    capture_request = None
    capture_request_valid = False
    if capture_request_path is not None and capture_request_path.is_file():
        capture_request = json.loads(capture_request_path.read_text(encoding="utf-8"))
        request_body = {key: value for key, value in capture_request.items() if key != "request_sha256"}
        capture_request_valid = (
            hashlib.sha256(canonical_json(request_body).encode("utf-8")).hexdigest()
            == capture_request.get("request_sha256")
            and capture_request.get("kernel_name") == kernel
            and capture_request.get("kernel_name_base") == "mangled"
            and capture_request.get("nvtx_range") == "gemma_bound_forward/"
            and capture_request.get("launch_count") == 1
        )
    checks = {
        "binding_hash_valid": binding_valid,
        "process_id_matches": details["process_id"] == binding["process_id"],
        "kernel_matches_profile": kernel in profile_counts,
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
        "exact_kernel_selection_recorded": exact_kernel_filter_visible or capture_request_valid,
        "single_launch_filter_recorded": "--launch-count1" in compact_session,
        "cupti_module_artifact_hash_valid": _file_sha256(cubin_path) == matching_module["cubin_sha256"],
        "launch_sass_matches_loaded_cubin_function": launch_instructions == static_instructions,
    }
    body = {
        "scope": "Launch-specific Nsight SASS bound by process ID and NVTX range to a Gemma execution, then compared with the matching CUPTI-loaded cubin function; not hardware-semantic proof.",
        "report_sha256": _file_sha256(report_path),
        "binding_sha256": binding["binding_sha256"],
        "session_output_sha256": hashlib.sha256(session_output.encode("utf-8")).hexdigest(),
        "source_output_sha256": hashlib.sha256(source_output.encode("utf-8")).hexdigest(),
        "session_exact_kernel_filter_value_visible": exact_kernel_filter_visible,
        "capture_request_sha256": capture_request.get("request_sha256") if capture_request else None,
        "details": details,
        "observed_forward_launch_count": profile_counts[kernel],
        "execution_binding": binding_body,
        "cupti_module": {key: matching_module[key] for key in ("module_id", "cubin_size", "cubin_sha256")},
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


def _kernel_family(name: str) -> str:
    lowered = name.lower()
    if "fmha" in lowered:
        return "fused_attention"
    if "gelu" in lowered:
        return "gelu"
    if "reduce_kernel" in name or "meanops" in lowered or "splitkreduce" in lowered:
        return "reduction"
    if "gemm" in lowered or "cublas" in lowered or "cutlass" in lowered or "gemvx" in lowered:
        return "linear_algebra"
    return "elementwise_and_indexing"


def build_nsight_kernel_suite(
    report_directory: Path,
    cuda_manifest_path: Path,
    cupti_report_path: Path,
    cupti_artifact_directory: Path,
    redact: bool = False,
) -> dict[str, Any]:
    from .sass_semantics import sass_image_coverage

    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        raise RuntimeError("cuobjdump is required")
    manifest = json.loads(cuda_manifest_path.read_text(encoding="utf-8"))
    cupti_report = json.loads(cupti_report_path.read_text(encoding="utf-8"))
    launch_counts = {
        name: count
        for name, count in manifest["profile"]["kernel_launch_counts"].items()
        if not name.startswith("Memcpy")
    }
    symbol_outputs = {}
    for module in cupti_report["modules"]:
        path = cupti_artifact_directory / module["artifact"]
        completed = subprocess.run(
            [cuobjdump, "--dump-elf-symbols", str(path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        symbol_outputs[module["cubin_sha256"]] = _parse_elf_function_symbols(completed.stdout)
    entries = []
    failures = []
    for index, (kernel, launches) in enumerate(launch_counts.items()):
        label = f"k{index:02d}"
        report = report_directory / f"{label}.ncu-rep"
        binding = report_directory / f"{label}_binding.json"
        candidates = [value for value, symbols in symbol_outputs.items() if kernel in symbols]
        if not report.is_file() or not binding.is_file() or len(candidates) != 1:
            failures.append(
                {
                    "index": index,
                    "kernel_name": kernel,
                    "family": _kernel_family(kernel),
                    "observed_forward_launch_count": launches,
                    "reason": "missing report/binding" if not report.is_file() or not binding.is_file() else "module symbol match was not unique",
                    "module_candidates": len(candidates),
                }
            )
            continue
        try:
            certificate = build_nsight_launch_certificate(
                report,
                binding,
                cuda_manifest_path,
                cupti_report_path,
                cupti_artifact_directory,
                candidates[0],
                report_directory / f"{label}_request.json",
            )
            certificate_path = report_directory / f"{label}_certificate.json"
            certificate_path.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            if not certificate["all_checks_pass"]:
                failures.append(
                    {
                        "index": index,
                        "kernel_name": kernel,
                        "family": _kernel_family(kernel),
                        "observed_forward_launch_count": launches,
                        "reason": "certificate checks failed",
                        "failed_checks": [name for name, passed in certificate["checks"].items() if not passed],
                    }
                )
                continue
            semantics_coverage = sass_image_coverage(certificate["launch_sass"]["opcode_histogram"])
            entries.append(
                {
                    "index": index,
                    "kernel_name": kernel,
                    "family": _kernel_family(kernel),
                    "observed_forward_launch_count": launches,
                    "module_id": certificate["cupti_module"]["module_id"],
                    "cubin_sha256": certificate["cupti_module"]["cubin_sha256"],
                    "instruction_count": certificate["launch_sass"]["instruction_count"],
                    "syntactic_proposed_semantics_opcode_lines": semantics_coverage["covered_instruction_lines"],
                    "sass_canonical_sha256": certificate["launch_sass"]["canonical_sha256"],
                    "report_sha256": certificate["report_sha256"],
                    "certificate_sha256": certificate["certificate_sha256"],
                    "session_exact_kernel_filter_value_visible": certificate[
                        "session_exact_kernel_filter_value_visible"
                    ],
                    "capture_request_sha256": certificate["capture_request_sha256"],
                }
            )
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
            failures.append(
                {
                    "index": index,
                    "kernel_name": kernel,
                    "family": _kernel_family(kernel),
                    "observed_forward_launch_count": launches,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
    expected_launches = sum(launch_counts.values())
    attested_launches = sum(entry["observed_forward_launch_count"] for entry in entries)
    family_names = sorted({_kernel_family(name) for name in launch_counts})
    families = {
        family: {
            "expected_distinct": sum(_kernel_family(name) == family for name in launch_counts),
            "attested_distinct": sum(entry["family"] == family for entry in entries),
            "expected_launches": sum(count for name, count in launch_counts.items() if _kernel_family(name) == family),
            "attested_launches": sum(
                entry["observed_forward_launch_count"] for entry in entries if entry["family"] == family
            ),
        }
        for family in family_names
    }
    instruction_lines = sum(entry["instruction_count"] for entry in entries)
    semantics_covered = sum(entry["syntactic_proposed_semantics_opcode_lines"] for entry in entries)
    session_exact_filters = sum(entry["session_exact_kernel_filter_value_visible"] for entry in entries)
    request_backed = sum(entry["capture_request_sha256"] is not None for entry in entries)
    if redact:
        for entry in entries:
            for key in (
                "cubin_sha256",
                "sass_canonical_sha256",
                "report_sha256",
                "certificate_sha256",
                "capture_request_sha256",
            ):
                if entry[key] is not None:
                    entry[key] = "redacted"
    body = {
        "scope": "One NVTX-bounded launch certificate attempted for every distinct non-copy CUDA kernel symbol observed in the recorded Gemma forward; instruction identity is not semantic proof.",
        "privacy": {"redacted": redact},
        "expected_distinct_kernels": len(launch_counts),
        "attested_distinct_kernels": len(entries),
        "session_exact_filter_value_kernels": session_exact_filters,
        "capture_request_backed_kernels": request_backed,
        "distinct_coverage_fraction": len(entries) / len(launch_counts) if launch_counts else 0.0,
        "expected_kernel_launches": expected_launches,
        "attested_kernel_launches": attested_launches,
        "launch_weighted_coverage_fraction": attested_launches / expected_launches if expected_launches else 0.0,
        "distinct_function_instruction_lines": instruction_lines,
        "syntactic_proposed_semantics_opcode_lines": semantics_covered,
        "syntactic_proposed_semantics_opcode_fraction": semantics_covered / instruction_lines if instruction_lines else 0.0,
        "families": families,
        "entries": entries,
        "failures": failures,
        "complete": len(entries) == len(launch_counts),
        "limitations": [
            "Each certificate covers one invocation of a distinct symbol, not every invocation.",
            "Kernel symbols with absent or non-unique CUPTI module matches cannot receive instruction-identity certificates.",
            "Nsight and CUPTI are NVIDIA tooling; instruction identity is not independent hardware-semantic verification.",
        ],
    }
    body["suite_sha256"] = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    return body


def verify_nsight_kernel_suite(
    suite: dict[str, Any], certificate_directory: Path | None = None, cupti_report: dict[str, Any] | None = None
) -> dict[str, Any]:
    body = {key: value for key, value in suite.items() if key != "suite_sha256"}
    hash_valid = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest() == suite.get("suite_sha256")
    entries = suite.get("entries", [])
    failures = suite.get("failures", [])
    expected = [*entries, *failures]
    attested_launches = sum(entry["observed_forward_launch_count"] for entry in entries)
    expected_launches = sum(entry["observed_forward_launch_count"] for entry in expected)
    distinct_consistent = (
        suite.get("attested_distinct_kernels") == len(entries)
        and suite.get("expected_distinct_kernels") == len(expected)
    )
    launch_consistent = (
        suite.get("attested_kernel_launches") == attested_launches
        and suite.get("expected_kernel_launches") == expected_launches
    )
    fractions_consistent = (
        suite.get("distinct_coverage_fraction") == (len(entries) / len(expected) if expected else 0.0)
        and suite.get("launch_weighted_coverage_fraction")
        == (attested_launches / expected_launches if expected_launches else 0.0)
    )
    complete_consistent = suite.get("complete") == (len(entries) == len(expected))
    instruction_lines = sum(entry.get("instruction_count", 0) for entry in entries)
    semantics_covered = sum(entry.get("syntactic_proposed_semantics_opcode_lines", 0) for entry in entries)
    selection_evidence_consistent = (
        suite.get("session_exact_filter_value_kernels")
        == sum(entry.get("session_exact_kernel_filter_value_visible", False) for entry in entries)
        and suite.get("capture_request_backed_kernels")
        == sum(entry.get("capture_request_sha256") is not None for entry in entries)
    )
    semantics_consistent = (
        suite.get("distinct_function_instruction_lines") == instruction_lines
        and suite.get("syntactic_proposed_semantics_opcode_lines") == semantics_covered
        and suite.get("syntactic_proposed_semantics_opcode_fraction")
        == (semantics_covered / instruction_lines if instruction_lines else 0.0)
    )
    family_consistent = set(suite.get("families", {})) == {item["family"] for item in expected} and all(
        claims
        == {
            "expected_distinct": sum(item["family"] == family for item in expected),
            "attested_distinct": sum(item["family"] == family for item in entries),
            "expected_launches": sum(
                item["observed_forward_launch_count"] for item in expected if item["family"] == family
            ),
            "attested_launches": sum(
                item["observed_forward_launch_count"] for item in entries if item["family"] == family
            ),
        }
        for family, claims in suite.get("families", {}).items()
    )
    certificate_results = None
    if certificate_directory is not None:
        certificate_results = []
        for entry in entries:
            path = certificate_directory / f"k{entry['index']:02d}_certificate.json"
            if not path.is_file():
                certificate_results.append(False)
                continue
            certificate = json.loads(path.read_text(encoding="utf-8"))
            certificate_results.append(
                verify_nsight_launch_certificate(certificate)["valid"]
                and certificate["certificate_sha256"] == entry["certificate_sha256"]
                and certificate["details"]["kernel_name"] == entry["kernel_name"]
                and certificate["cupti_module"]["cubin_sha256"] == entry["cubin_sha256"]
                and certificate["launch_sass"]["canonical_sha256"] == entry["sass_canonical_sha256"]
                and certificate["launch_sass"]["instruction_count"] == entry["instruction_count"]
            )
    certificates_valid = None if certificate_results is None else all(certificate_results) and len(certificate_results) == len(entries)
    cupti_links_valid = None
    if cupti_report is not None:
        loaded_hashes = {module["cubin_sha256"] for module in cupti_report.get("modules", [])}
        cupti_links_valid = all(entry["cubin_sha256"] in loaded_hashes for entry in entries)
    return {
        "valid": bool(
            hash_valid
            and distinct_consistent
            and launch_consistent
            and fractions_consistent
            and complete_consistent
            and selection_evidence_consistent
            and semantics_consistent
            and family_consistent
            and certificates_valid is not False
            and cupti_links_valid is not False
        ),
        "suite_hash_valid": hash_valid,
        "per_certificate_verification_performed": certificate_directory is not None,
        "per_certificate_claims_valid": certificates_valid,
        "cupti_relink_performed": cupti_report is not None,
        "cupti_links_valid": cupti_links_valid,
        "distinct_claim_consistent": distinct_consistent,
        "launch_claim_consistent": launch_consistent,
        "coverage_fractions_consistent": fractions_consistent,
        "complete_claim_consistent": complete_consistent,
        "selection_evidence_claims_consistent": selection_evidence_consistent,
        "sass_semantics_claims_consistent": semantics_consistent,
        "family_claims_consistent": family_consistent,
    }


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
