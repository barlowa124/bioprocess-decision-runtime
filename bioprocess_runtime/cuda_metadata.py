from __future__ import annotations

import ctypes
import hashlib
import os
import re
from pathlib import Path
from typing import Any

from .cupti_attestation import _find_cupti_library
from .launch_arguments import (
    CUPTI_API_ENTER,
    CUPTI_CB_DOMAIN_DRIVER_API,
    CUPTI_LAUNCH_CALLBACKS,
    _CallbackData,
    _LaunchConfig,
    _LaunchKernelExParams,
    _LaunchKernelParams,
)
from .serialization import canonical_json


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _enum_value(text: str, name: str) -> int:
    match = re.search(rf"\b{re.escape(name)}\s*=\s*(\d+)", text)
    if not match:
        raise ValueError(f"Could not find {name}")
    return int(match.group(1))


def _struct_fields(text: str, typedef_name: str) -> list[str]:
    matches = re.findall(r"typedef\s+struct(?:\s+\w+)?\s*\{(.*?)\}\s*(\w+)\s*;", text, re.DOTALL)
    body = next((body for body, name in matches if name == typedef_name), None)
    if body is None:
        raise ValueError(f"Could not find struct {typedef_name}")
    fields = []
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.DOTALL)
    body = re.sub(r"//.*", "", body)
    for statement in body.split(";"):
        value = statement.strip()
        if not value or value.startswith("#"):
            continue
        match = re.search(r"([A-Za-z_]\w*)\s*(?:\[[^]]+\])?\s*$", value)
        if match:
            fields.append(match.group(1))
    return fields


def _ctypes_layout(structure: type[ctypes.Structure]) -> dict[str, Any]:
    return {
        "size": ctypes.sizeof(structure),
        "fields": [
            {"name": name, "offset": getattr(structure, name).offset, "size": ctypes.sizeof(field_type)}
            for name, field_type in structure._fields_
        ],
    }


def build_cuda_metadata_conformance() -> dict[str, Any]:
    cuda_path = os.environ.get("CUDA_PATH")
    if not cuda_path:
        raise RuntimeError("CUDA_PATH is required")
    root = Path(cuda_path)
    cupti_include = root / "extras" / "CUPTI" / "include"
    headers = {
        "cupti_driver_cbid.h": cupti_include / "cupti_driver_cbid.h",
        "generated_cuda_meta.h": cupti_include / "generated_cuda_meta.h",
        "cupti_callbacks.h": cupti_include / "cupti_callbacks.h",
        "cuda.h": root / "include" / "cuda.h",
    }
    texts = {name: path.read_text(encoding="utf-8", errors="replace") for name, path in headers.items()}
    callback_ids = {
        "cuLaunchKernel": _enum_value(texts["cupti_driver_cbid.h"], "CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel"),
        "cuLaunchKernel_ptsz": _enum_value(texts["cupti_driver_cbid.h"], "CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz"),
        "cuLaunchKernelEx": _enum_value(texts["cupti_driver_cbid.h"], "CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx"),
        "cuLaunchKernelEx_ptsz": _enum_value(texts["cupti_driver_cbid.h"], "CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz"),
    }
    header_structs = {
        "cuLaunchKernel_params": _struct_fields(texts["generated_cuda_meta.h"], "cuLaunchKernel_params"),
        "cuLaunchKernelEx_params": _struct_fields(texts["generated_cuda_meta.h"], "cuLaunchKernelEx_params"),
        "CUpti_CallbackData": _struct_fields(texts["cupti_callbacks.h"], "CUpti_CallbackData"),
        "CUlaunchConfig": _struct_fields(texts["cuda.h"], "CUlaunchConfig"),
    }
    expected_fields = {
        "cuLaunchKernel_params": [
            "f",
            "gridDimX",
            "gridDimY",
            "gridDimZ",
            "blockDimX",
            "blockDimY",
            "blockDimZ",
            "sharedMemBytes",
            "hStream",
            "kernelParams",
            "extra",
        ],
        "cuLaunchKernelEx_params": ["config", "f", "kernelParams", "extra"],
        "CUpti_CallbackData": [
            "callbackSite",
            "functionName",
            "functionParams",
            "functionReturnValue",
            "symbolName",
            "context",
            "contextUid",
            "correlationData",
            "correlationId",
        ],
        "CUlaunchConfig": [
            "gridDimX",
            "gridDimY",
            "gridDimZ",
            "blockDimX",
            "blockDimY",
            "blockDimZ",
            "sharedMemBytes",
            "hStream",
            "attrs",
            "numAttrs",
        ],
    }
    expected_x64_layouts = {
        "callback_data": {"size": 72, "offsets": [0, 8, 16, 24, 32, 40, 48, 56, 64]},
        "launch_kernel": {"size": 64, "offsets": [0, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56]},
        "launch_config": {"size": 56, "offsets": [0, 4, 8, 12, 16, 20, 24, 32, 40, 48]},
        "launch_kernel_ex": {"size": 32, "offsets": [0, 8, 16, 24]},
    }
    layouts = {
        "callback_data": _ctypes_layout(_CallbackData),
        "launch_kernel": _ctypes_layout(_LaunchKernelParams),
        "launch_config": _ctypes_layout(_LaunchConfig),
        "launch_kernel_ex": _ctypes_layout(_LaunchKernelExParams),
    }
    selected_library = _find_cupti_library()
    library = ctypes.WinDLL(str(selected_library)) if os.name == "nt" else ctypes.CDLL(str(selected_library))
    library.cuptiGetVersion.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
    library.cuptiGetVersion.restype = ctypes.c_int
    version = ctypes.c_uint32()
    version_result = library.cuptiGetVersion(ctypes.byref(version))
    checks = {
        "callback_ids_match": callback_ids == {name: callback_id for callback_id, name in CUPTI_LAUNCH_CALLBACKS.items()},
        "driver_domain_matches": _enum_value(texts["cupti_callbacks.h"], "CUPTI_CB_DOMAIN_DRIVER_API")
        == CUPTI_CB_DOMAIN_DRIVER_API,
        "api_enter_matches": _enum_value(texts["cupti_callbacks.h"], "CUPTI_API_ENTER") == CUPTI_API_ENTER,
        "header_field_orders_match": all(header_structs[name] == expected for name, expected in expected_fields.items()),
        "ctypes_x64_layouts_match": ctypes.sizeof(ctypes.c_void_p) == 8
        and all(
            layouts[name]["size"] == expected["size"]
            and [field["offset"] for field in layouts[name]["fields"]] == expected["offsets"]
            for name, expected in expected_x64_layouts.items()
        ),
        "cupti_version_query_succeeded": version_result == 0,
    }
    body = {
        "scope": "Local CUDA/CUPTI header, callback-ID, selected-library, and x64 ctypes-layout conformance; not cross-toolkit portability proof.",
        "header_sha256": {name: _file_sha256(path) for name, path in headers.items()},
        "callback_ids": callback_ids,
        "header_struct_fields": header_structs,
        "ctypes_layouts": layouts,
        "selected_cupti_library": {
            "name": selected_library.name,
            "sha256": _file_sha256(selected_library),
            "reported_version": int(version.value),
        },
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }
    body["certificate_sha256"] = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    return body


def verify_cuda_metadata_conformance(certificate: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in certificate.items() if key != "certificate_sha256"}
    hash_valid = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest() == certificate.get(
        "certificate_sha256"
    )
    checks_consistent = certificate.get("all_checks_pass") == all(certificate.get("checks", {}).values())
    return {
        "valid": bool(hash_valid and checks_consistent and certificate.get("all_checks_pass")),
        "certificate_hash_valid": hash_valid,
        "checks_consistent": checks_consistent,
    }
