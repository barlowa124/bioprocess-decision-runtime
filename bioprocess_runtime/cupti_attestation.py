from __future__ import annotations

import ctypes
import hashlib
import os
import sys
from pathlib import Path
from typing import Any

from .serialization import canonical_json


CUPTI_CB_DOMAIN_RESOURCE = 3
CUPTI_CBID_RESOURCE_MODULE_LOADED = 6


class _ResourceHandle(ctypes.Union):
    _fields_ = [("stream", ctypes.c_void_p), ("module", ctypes.c_void_p), ("context", ctypes.c_void_p)]


class _ResourceData(ctypes.Structure):
    _fields_ = [
        ("context", ctypes.c_void_p),
        ("resourceHandle", _ResourceHandle),
        ("resourceDescriptor", ctypes.c_void_p),
    ]


class _ModuleResourceData(ctypes.Structure):
    _fields_ = [
        ("moduleId", ctypes.c_uint32),
        ("cubinSize", ctypes.c_size_t),
        ("pCubin", ctypes.c_void_p),
    ]


def _find_cupti_library() -> Path:
    torch_libraries = [
        Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib",
        Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "torch" / "lib",
    ]
    patterns = ("cupti64_*.dll", "libcupti.so*")
    bundled = [path for root in torch_libraries for pattern in patterns for path in root.glob(pattern)]
    cuda_path = os.environ.get("CUDA_PATH")
    toolkit = []
    if cuda_path:
        toolkit = [
            path
            for pattern in patterns
            for path in (Path(cuda_path) / "extras" / "CUPTI" / "lib64").glob(pattern)
        ]
    candidates = toolkit or bundled
    if not candidates:
        raise RuntimeError("CUPTI library was not found")
    return sorted({path.resolve() for path in candidates})[-1]


class CuptiModuleCapture:
    def __init__(self, output_directory: Path) -> None:
        self.output_directory = output_directory
        self.records: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.library_path = _find_cupti_library()
        self.library = ctypes.WinDLL(str(self.library_path)) if os.name == "nt" else ctypes.CDLL(str(self.library_path))
        callback_factory = ctypes.WINFUNCTYPE if os.name == "nt" else ctypes.CFUNCTYPE
        self.callback_type = callback_factory(None, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p)
        self.callback = self.callback_type(self._callback)
        self.subscriber = ctypes.c_void_p()
        self.library.cuptiSubscribe.argtypes = [ctypes.POINTER(ctypes.c_void_p), self.callback_type, ctypes.c_void_p]
        self.library.cuptiSubscribe.restype = ctypes.c_int
        self.library.cuptiEnableCallback.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]
        self.library.cuptiEnableCallback.restype = ctypes.c_int
        self.library.cuptiUnsubscribe.argtypes = [ctypes.c_void_p]
        self.library.cuptiUnsubscribe.restype = ctypes.c_int
        self.active = False

    def _callback(self, userdata: int, domain: int, callback_id: int, callback_data: int) -> None:
        if domain != CUPTI_CB_DOMAIN_RESOURCE or callback_id != CUPTI_CBID_RESOURCE_MODULE_LOADED:
            return
        try:
            resource = ctypes.cast(callback_data, ctypes.POINTER(_ResourceData)).contents
            module = ctypes.cast(resource.resourceDescriptor, ctypes.POINTER(_ModuleResourceData)).contents
            cubin = ctypes.string_at(module.pCubin, module.cubinSize)
            digest = hashlib.sha256(cubin).hexdigest()
            self.output_directory.mkdir(parents=True, exist_ok=True)
            filename = f"module_{module.moduleId}_{digest[:16]}.cubin"
            (self.output_directory / filename).write_bytes(cubin)
            self.records.append(
                {
                    "module_id": int(module.moduleId),
                    "cubin_size": int(module.cubinSize),
                    "cubin_sha256": digest,
                    "artifact": filename,
                }
            )
        except Exception as exc:
            self.errors.append(f"{type(exc).__name__}: {exc}")

    def start(self) -> None:
        result = self.library.cuptiSubscribe(ctypes.byref(self.subscriber), self.callback, None)
        if result != 0:
            raise RuntimeError(f"cuptiSubscribe failed with result {result}")
        result = self.library.cuptiEnableCallback(
            1, self.subscriber, CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_MODULE_LOADED
        )
        if result != 0:
            self.library.cuptiUnsubscribe(self.subscriber)
            raise RuntimeError(f"cuptiEnableCallback failed with result {result}")
        self.active = True

    def stop(self) -> None:
        if not self.active:
            return
        result = self.library.cuptiUnsubscribe(self.subscriber)
        self.active = False
        if result != 0:
            raise RuntimeError(f"cuptiUnsubscribe failed with result {result}")

    def report(self, execution_binding: dict[str, Any] | None = None) -> dict[str, Any]:
        unique = {record["cubin_sha256"]: record for record in self.records}
        body = {
            "scope": "CUPTI resource-callback capture of cubin values presented during CUDA module-load events; not per-launch function-to-module correlation.",
            "cupti_library": self.library_path.name,
            "module_load_events": len(self.records),
            "unique_cubins": len(unique),
            "modules": sorted(unique.values(), key=lambda item: (item["module_id"], item["cubin_sha256"])),
            "callback_errors": self.errors,
            "execution_binding": execution_binding,
            "limitations": [
                "Modules loaded before subscription are not captured.",
                "A module-load callback does not prove which function or cubin image served each kernel launch.",
                "Captured cubin bytes are local artifacts and are not committed to Git.",
            ],
        }
        body["record_sha256"] = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
        return body


def verify_cupti_module_capture(report: dict[str, Any], artifact_directory: Path | None = None) -> dict[str, Any]:
    body = {key: value for key, value in report.items() if key != "record_sha256"}
    hash_matches = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest() == report.get("record_sha256")
    artifacts_match = None
    if artifact_directory is not None:
        artifacts_match = True
        for module in report.get("modules", []):
            path = artifact_directory / module["artifact"]
            artifacts_match = artifacts_match and path.is_file()
            if path.is_file():
                value = path.read_bytes()
                artifacts_match = artifacts_match and len(value) == module["cubin_size"]
                artifacts_match = artifacts_match and hashlib.sha256(value).hexdigest() == module["cubin_sha256"]
    return {
        "valid": bool(hash_matches and artifacts_match is not False and not report.get("callback_errors")),
        "record_hash_matches": hash_matches,
        "local_artifacts_match": artifacts_match,
        "callback_errors": report.get("callback_errors", []),
    }


def summarize_cupti_module_capture(report: dict[str, Any], cuda_summary: dict[str, Any]) -> dict[str, Any]:
    if not verify_cupti_module_capture(report)["valid"]:
        raise ValueError("Cannot summarize an invalid CUPTI module capture")
    profiled_image_hash = cuda_summary["profiled_symbol_disassembly"]["embedded_image_sha256"]
    matching = [module for module in report["modules"] if module["cubin_sha256"] == profiled_image_hash]
    return {
        "scope": report["scope"],
        "record_sha256": report["record_sha256"],
        "execution_binding": report["execution_binding"],
        "cupti_library": report["cupti_library"],
        "module_load_events": report["module_load_events"],
        "unique_cubins": report["unique_cubins"],
        "captured_modules": [
            {key: module[key] for key in ("module_id", "cubin_size", "cubin_sha256")} for module in report["modules"]
        ],
        "profiled_static_image_binding": {
            "profiled_kernel_name": cuda_summary["profiled_symbol_disassembly"]["profiled_kernel_name"],
            "static_image_sha256": profiled_image_hash,
            "matching_module_loads": [
                {key: module[key] for key in ("module_id", "cubin_size", "cubin_sha256")} for module in matching
            ],
            "matched": len(matching) == 1,
            "claim": "The statically disassembled compatible image value was presented in a CUDA module-load callback during the bound Gemma execution; per-launch function-to-module correlation remains unproved.",
        },
        "limitations": report["limitations"],
    }
