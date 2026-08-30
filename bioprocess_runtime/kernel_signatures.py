from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from .serialization import canonical_json

try:
    import torch
except ModuleNotFoundError:
    torch = None


def build_kernel_signature_certificate(summary: dict[str, Any]) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("PyTorch is required")
    source = Path(torch.__file__).resolve().parent / "include" / "ATen" / "native" / "cuda" / "CUDALoops.cuh"
    text = source.read_text(encoding="utf-8")
    signature = re.search(
        r"__global__\s+void\s+vectorized_elementwise_kernel\s*\(\s*int\s+N\s*,\s*func_t\s+f\s*,\s*array_t\s+data\s*\)",
        text,
    )
    entries = []
    for entry in summary["entries"]:
        gelu_signature = (
            "vectorized_elementwise_kernel" in entry["kernel_name"]
            and "GeluCUDAKernelImpl" in entry["kernel_name"]
            and "St5arrayIPcLy2E" in entry["kernel_name"]
            and entry["parameter_sizes"] == [4, 1, 16]
            and signature is not None
        )
        typed_fields = []
        if gelu_signature:
            matches = {(match["parameter_index"], match["parameter_byte_offset"]): match for match in entry["boundary_pointer_matches"]}
            typed_fields = [
                {"parameter_index": 0, "byte_offset": 0, "size_bytes": 4, "type": "int", "name": "N"},
                {"parameter_index": 1, "byte_offset": 0, "size_bytes": 1, "type": "func_t", "name": "f"},
                {
                    "parameter_index": 2,
                    "byte_offset": 0,
                    "size_bytes": 8,
                    "type": "char*",
                    "name": "data[0]",
                    "observed_boundary_role": matches.get((2, 0), {}).get("role"),
                },
                {
                    "parameter_index": 2,
                    "byte_offset": 8,
                    "size_bytes": 8,
                    "type": "char*",
                    "name": "data[1]",
                    "observed_boundary_role": matches.get((2, 8), {}).get("role"),
                },
            ]
        entries.append(
            {
                "kernel_name": entry["kernel_name"],
                "expected_module": entry["expected_module"],
                "parameter_sizes": entry["parameter_sizes"],
                "typed_signature_established": gelu_signature,
                "typed_fields": typed_fields,
                "evidence": (
                    "Installed CUDALoops.cuh signature, mangled std::array<char*,2> type, driver parameter sizes, and boundary pointer matches."
                    if gelu_signature
                    else "No independently available local definition for the packed parameter type."
                ),
                "full_field_semantics_established": False,
            }
        )
    body = {
        "scope": "Partial kernel-signature typing from installed PyTorch headers, mangled type identity, driver-reported sizes, and observed pointer matches; not full field or access semantics.",
        "source": {
            "name": "ATen/native/cuda/CUDALoops.cuh",
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "expected_signature_found": signature is not None,
        },
        "source_launch_argument_summary_sha256": summary["summary_sha256"],
        "entries": entries,
        "typed_entries": sum(entry["typed_signature_established"] for entry in entries),
        "total_entries": len(entries),
        "all_signatures_typed": all(entry["typed_signature_established"] for entry in entries),
        "complete_field_semantics_established": False,
    }
    body["certificate_sha256"] = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    return body


def verify_kernel_signature_certificate(certificate: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in certificate.items() if key != "certificate_sha256"}
    hash_valid = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest() == certificate.get(
        "certificate_sha256"
    )
    entries = certificate.get("entries", [])
    counts_consistent = (
        certificate.get("typed_entries") == sum(entry["typed_signature_established"] for entry in entries)
        and certificate.get("total_entries") == len(entries)
        and certificate.get("all_signatures_typed") == all(entry["typed_signature_established"] for entry in entries)
    )
    boundaries_preserved = certificate.get("complete_field_semantics_established") is False and all(
        entry["full_field_semantics_established"] is False for entry in entries
    )
    return {
        "valid": bool(hash_valid and counts_consistent and boundaries_preserved),
        "certificate_hash_valid": hash_valid,
        "counts_consistent": counts_consistent,
        "boundaries_preserved": boundaries_preserved,
    }
