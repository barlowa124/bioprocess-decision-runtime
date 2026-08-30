from __future__ import annotations

import ctypes
import hashlib
import json
import re
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Any

from .serialization import canonical_json

try:
    import torch
except ModuleNotFoundError:
    torch = None


PYTORCH_COMMIT = "e2d141dbde55c2a4370fac5165b0561b6af4798b"
CUTLASS_COMMIT = "afa1772203677c5118fcd82537a9c8fefbcc7008"
CUDALOOPS_SOURCE_BLOB = "92b77dfb6aeaf41d7b817df266bd5ba602be5dae"
ATTENTION_SOURCE_BLOB = "d2e53e9dfadbf9103052c07e74b1fcda7be2692f"
REDUCTION_SOURCE_BLOB = "2c25c413ead2fb2b5f5f0c8c9f878ba08250654d"
SOURCE_PATHS = {
    "cudaloops": "aten/src/ATen/native/cuda/CUDALoops.cuh",
    "attention": "aten/src/ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h",
    "reduction": "aten/src/ATen/native/cuda/Reduce.cuh",
}


@lru_cache(maxsize=8)
def _fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "bioprocess-decision-runtime"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _git_blob_sha1(value: bytes) -> str:
    return hashlib.sha1(f"blob {len(value)}\0".encode("ascii") + value).hexdigest()


def _attention_source_fields(text: str) -> list[str]:
    match = re.search(r"struct\s+Params\s*\{(.*?)CUTLASS_DEVICE\s+bool\s+advance_to_block", text, re.DOTALL)
    if not match:
        raise ValueError("Could not locate AttentionKernel::Params")
    body = re.sub(r"//.*", "", match.group(1))
    fields = []
    for statement in body.split(";"):
        value = statement.strip()
        if not value:
            continue
        name = re.search(r"([A-Za-z_]\w*)\s*(?:=[^=].*)?$", value)
        if name:
            fields.append(name.group(1))
    return fields


def _source_evidence(installed_cudaloops: bytes) -> tuple[dict[str, Any], dict[str, bytes]]:
    base = f"https://raw.githubusercontent.com/pytorch/pytorch/{PYTORCH_COMMIT}/"
    remote = {name: _fetch(base + path) for name, path in SOURCE_PATHS.items()}
    api = _fetch(
        f"https://api.github.com/repos/pytorch/pytorch/contents/third_party/cutlass?ref={PYTORCH_COMMIT}"
    )
    cutlass = json.loads(api.decode("utf-8"))
    expected_blobs = {
        "cudaloops": CUDALOOPS_SOURCE_BLOB,
        "attention": ATTENTION_SOURCE_BLOB,
        "reduction": REDUCTION_SOURCE_BLOB,
    }
    files = [
        {
            "name": SOURCE_PATHS[name],
            "git_blob_sha1": _git_blob_sha1(value),
            "expected_git_blob_sha1": expected_blobs[name],
            "git_blob_verified": _git_blob_sha1(value) == expected_blobs[name],
            "sha256": hashlib.sha256(value).hexdigest(),
            "installed_copy_matches": installed_cudaloops.replace(b"\r\n", b"\n") == value.replace(b"\r\n", b"\n")
            if name == "cudaloops"
            else None,
        }
        for name, value in remote.items()
    ]
    evidence = {
        "wheel_version": torch.__version__,
        "wheel_git_commit": torch.version.git_version,
        "expected_pytorch_commit": PYTORCH_COMMIT,
        "wheel_revision_matches": torch.version.git_version == PYTORCH_COMMIT,
        "cutlass_gitlink_commit": cutlass.get("sha"),
        "expected_cutlass_gitlink_commit": CUTLASS_COMMIT,
        "cutlass_gitlink_verified": cutlass.get("sha") == CUTLASS_COMMIT,
        "files": files,
    }
    evidence["all_source_identities_verified"] = (
        evidence["wheel_revision_matches"]
        and evidence["cutlass_gitlink_verified"]
        and all(file["git_blob_verified"] for file in files)
        and files[0]["installed_copy_matches"]
    )
    return evidence, remote


class _PhiloxCudaState(ctypes.Structure):
    _fields_ = [
        ("seed", ctypes.c_uint64),
        ("offset", ctypes.c_uint64),
        ("offset_intragraph", ctypes.c_uint32),
        ("captured", ctypes.c_bool),
    ]


class _AttentionParams(ctypes.Structure):
    _fields_ = [
        ("query_ptr", ctypes.c_void_p),
        ("key_ptr", ctypes.c_void_p),
        ("value_ptr", ctypes.c_void_p),
        ("attn_bias_ptr", ctypes.c_void_p),
        ("seqstart_q_ptr", ctypes.c_void_p),
        ("seqstart_k_ptr", ctypes.c_void_p),
        ("seqlen_k_ptr", ctypes.c_void_p),
        ("causal_diagonal_offset", ctypes.c_uint32),
        ("output_ptr", ctypes.c_void_p),
        ("output_accum_ptr", ctypes.c_void_p),
        ("logsumexp_ptr", ctypes.c_void_p),
        ("window_size", ctypes.c_int32),
        ("scale", ctypes.c_float),
        ("head_dim", ctypes.c_int32),
        ("head_dim_value", ctypes.c_int32),
        ("num_queries", ctypes.c_int32),
        ("num_keys", ctypes.c_int32),
        ("num_keys_absolute", ctypes.c_int32),
        ("custom_mask_type", ctypes.c_uint8),
        ("q_strideM", ctypes.c_int32),
        ("k_strideM", ctypes.c_int32),
        ("v_strideM", ctypes.c_int32),
        ("bias_strideM", ctypes.c_int32),
        ("o_strideM", ctypes.c_int32),
        ("q_strideH", ctypes.c_int32),
        ("k_strideH", ctypes.c_int32),
        ("v_strideH", ctypes.c_int32),
        ("bias_strideH", ctypes.c_int64),
        ("q_strideB", ctypes.c_int64),
        ("k_strideB", ctypes.c_int64),
        ("v_strideB", ctypes.c_int64),
        ("bias_strideB", ctypes.c_int64),
        ("num_batches", ctypes.c_int32),
        ("num_heads", ctypes.c_int32),
        ("use_dropout", ctypes.c_bool),
        ("dropout_batch_head_rng_offset", ctypes.c_uint64),
        ("dropout_prob", ctypes.c_float),
        ("rng_engine_inputs", _PhiloxCudaState),
        ("extragraph_offset", ctypes.c_void_p),
        ("seed", ctypes.c_void_p),
    ]


def _layout(structure: type[ctypes.Structure]) -> dict[str, Any]:
    return {
        "size_bytes": ctypes.sizeof(structure),
        "fields": [
            {"name": name, "offset": getattr(structure, name).offset, "size_bytes": ctypes.sizeof(field_type)}
            for name, field_type in structure._fields_
        ],
    }


def build_kernel_signature_certificate(summary: dict[str, Any]) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("PyTorch is required")
    source = Path(torch.__file__).resolve().parent / "include" / "ATen" / "native" / "cuda" / "CUDALoops.cuh"
    text = source.read_text(encoding="utf-8")
    signature = re.search(
        r"__global__\s+void\s+vectorized_elementwise_kernel\s*\(\s*int\s+N\s*,\s*func_t\s+f\s*,\s*array_t\s+data\s*\)",
        text,
    )
    source_evidence, remote_sources = _source_evidence(text.encode("utf-8"))
    attention_source_fields = _attention_source_fields(remote_sources["attention"].decode("utf-8"))
    ctypes_attention_fields = [name for name, field_type in _AttentionParams._fields_]
    attention_field_order_verified = attention_source_fields == ctypes_attention_fields
    entries = []
    for entry in summary["entries"]:
        gelu_signature = (
            "vectorized_elementwise_kernel" in entry["kernel_name"]
            and "GeluCUDAKernelImpl" in entry["kernel_name"]
            and "St5arrayIPcLy2E" in entry["kernel_name"]
            and entry["parameter_sizes"] == [4, 1, 16]
            and signature is not None
        )
        attention_layout = (
            "fmha_cutlass" in entry["kernel_name"]
            and "AttentionKernel" in entry["kernel_name"]
            and entry["parameter_sizes"] == [ctypes.sizeof(_AttentionParams)]
            and source_evidence["all_source_identities_verified"]
            and attention_field_order_verified
        )
        typed_fields = []
        if gelu_signature:
            matches = {(match["parameter_index"], match["parameter_byte_offset"]): match for match in entry["boundary_pointer_matches"]}
            typed_fields = [
                {"parameter_index": 0, "byte_offset": 0, "size_bytes": 4, "type": "int", "name": "N"},
                {"parameter_index": 1, "byte_offset": 0, "size_bytes": 1, "type": "unresolved func_t closure", "name": "f"},
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
        reconstructed_fields = _layout(_AttentionParams)["fields"] if attention_layout else []
        observed_role_conflicts = []
        if attention_layout:
            field_by_offset = {field["offset"]: field["name"] for field in reconstructed_fields}
            read_pointer_fields = {
                "query_ptr",
                "key_ptr",
                "value_ptr",
                "attn_bias_ptr",
                "seqstart_q_ptr",
                "seqstart_k_ptr",
                "seqlen_k_ptr",
            }
            write_pointer_fields = {"output_ptr", "output_accum_ptr", "logsumexp_ptr"}
            observed_role_conflicts = [
                {
                    "parameter_byte_offset": match["parameter_byte_offset"],
                    "source_field": field_by_offset.get(match["parameter_byte_offset"]),
                    "retrospective_boundary_role": match["role"],
                    "temporal_status": match.get("temporal_status"),
                    "interpretation": "Address equality is temporally ambiguous and is not used as a typed role binding.",
                }
                for match in entry["boundary_pointer_matches"]
                if (
                    match.get("temporal_status", "post_launch_retrospective") == "post_launch_retrospective"
                    and match["role"] == "output"
                    and field_by_offset.get(match["parameter_byte_offset"]) in read_pointer_fields
                )
                or (
                    match.get("temporal_status") == "pre_launch_retained"
                    and match["role"] in {"input", "parameter"}
                    and field_by_offset.get(match["parameter_byte_offset"]) in write_pointer_fields
                )
            ]
        entries.append(
            {
                "kernel_name": entry["kernel_name"],
                "expected_module": entry["expected_module"],
                "parameter_sizes": entry["parameter_sizes"],
                "typed_signature_established": False,
                "partial_parameter_schema_established": gelu_signature,
                "typed_fields": typed_fields,
                "source_layout_reconstructed": attention_layout,
                "reconstructed_layout_size_bytes": ctypes.sizeof(_AttentionParams) if attention_layout else None,
                "reconstructed_fields": reconstructed_fields,
                "observed_role_conflicts": observed_role_conflicts,
                "evidence": (
                    "Verified exact-commit CUDALoops.cuh signature, mangled std::array<char*,2> type, driver parameter sizes, and boundary pointer matches; closure type remains unresolved."
                    if gelu_signature
                    else (
                        "Exact PyTorch source field order, x64 ABI reconstruction, matching 264-byte driver size, and source-revision commitments."
                        if attention_layout
                        else "No independently reconstructable local layout for the packed parameter type."
                    )
                ),
                "compiled_layout_verified": False,
                "full_field_semantics_established": False,
            }
        )
    body = {
        "scope": "Partial kernel-signature typing from installed PyTorch headers, mangled type identity, driver-reported sizes, and observed pointer matches; not full field or access semantics.",
        "source": {
            **source_evidence,
            "expected_gelu_signature_found": signature is not None,
            "attention_source_field_order": attention_source_fields,
            "attention_ctypes_field_order": ctypes_attention_fields,
            "attention_field_order_verified": attention_field_order_verified,
            "abi_assumptions": {
                "pointer_size_bytes": ctypes.sizeof(ctypes.c_void_p),
                "native_ctypes_alignment": True,
                "platform": "Windows x64" if ctypes.sizeof(ctypes.c_void_p) == 8 else "unsupported",
            },
        },
        "source_launch_argument_summary_sha256": summary["summary_sha256"],
        "entries": entries,
        "typed_entries": sum(entry["typed_signature_established"] for entry in entries),
        "partial_parameter_schema_entries": sum(entry["partial_parameter_schema_established"] for entry in entries),
        "source_layout_reconstructed_entries": sum(entry["source_layout_reconstructed"] for entry in entries),
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
        and certificate.get("partial_parameter_schema_entries")
        == sum(entry["partial_parameter_schema_established"] for entry in entries)
        and certificate.get("source_layout_reconstructed_entries")
        == sum(entry["source_layout_reconstructed"] for entry in entries)
        and certificate.get("total_entries") == len(entries)
        and certificate.get("all_signatures_typed") == all(entry["typed_signature_established"] for entry in entries)
    )
    boundaries_preserved = certificate.get("complete_field_semantics_established") is False and all(
        entry["full_field_semantics_established"] is False for entry in entries
    )
    source_path = Path(torch.__file__).resolve().parent / "include" / "ATen" / "native" / "cuda" / "CUDALoops.cuh"
    source_evidence, remote_sources = _source_evidence(source_path.read_bytes())
    recorded_source = certificate.get("source", {})
    recomputed_source_fields = _attention_source_fields(remote_sources["attention"].decode("utf-8"))
    recomputed_ctypes_fields = [name for name, field_type in _AttentionParams._fields_]
    recomputed_field_order_match = recomputed_source_fields == recomputed_ctypes_fields
    source_valid = (
        source_evidence["all_source_identities_verified"]
        and recorded_source.get("wheel_git_commit") == source_evidence["wheel_git_commit"]
        and recorded_source.get("cutlass_gitlink_commit") == source_evidence["cutlass_gitlink_commit"]
        and recorded_source.get("files") == source_evidence["files"]
        and recorded_source.get("attention_source_field_order") == recomputed_source_fields
        and recorded_source.get("attention_ctypes_field_order") == recomputed_ctypes_fields
        and recorded_source.get("attention_field_order_verified") == recomputed_field_order_match
        and recomputed_field_order_match
    )
    expected_attention_layout = _layout(_AttentionParams)
    layouts_valid = all(
        not entry["source_layout_reconstructed"]
        or (
            entry["reconstructed_layout_size_bytes"] == expected_attention_layout["size_bytes"]
            and entry["reconstructed_fields"] == expected_attention_layout["fields"]
        )
        for entry in entries
    )
    return {
        "valid": bool(hash_valid and counts_consistent and boundaries_preserved and source_valid and layouts_valid),
        "certificate_hash_valid": hash_valid,
        "counts_consistent": counts_consistent,
        "boundaries_preserved": boundaries_preserved,
        "source_identities_valid": source_valid,
        "reconstructed_layouts_valid": layouts_valid,
    }
