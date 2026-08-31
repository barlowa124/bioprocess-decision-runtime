from __future__ import annotations

import copy
import ctypes
import hashlib
import os
from typing import Any, Callable

from .cupti_attestation import _find_cupti_library
from .serialization import canonical_json


CUPTI_CB_DOMAIN_DRIVER_API = 1
CUPTI_API_ENTER = 0
CUPTI_LAUNCH_CALLBACKS = {307: "cuLaunchKernel", 442: "cuLaunchKernel_ptsz", 652: "cuLaunchKernelEx", 653: "cuLaunchKernelEx_ptsz"}


class _CallbackData(ctypes.Structure):
    _fields_ = [
        ("callbackSite", ctypes.c_uint32),
        ("functionName", ctypes.c_char_p),
        ("functionParams", ctypes.c_void_p),
        ("functionReturnValue", ctypes.c_void_p),
        ("symbolName", ctypes.c_char_p),
        ("context", ctypes.c_void_p),
        ("contextUid", ctypes.c_uint32),
        ("correlationData", ctypes.POINTER(ctypes.c_uint64)),
        ("correlationId", ctypes.c_uint32),
    ]


class _LaunchKernelParams(ctypes.Structure):
    _fields_ = [
        ("function", ctypes.c_void_p),
        ("grid_x", ctypes.c_uint32),
        ("grid_y", ctypes.c_uint32),
        ("grid_z", ctypes.c_uint32),
        ("block_x", ctypes.c_uint32),
        ("block_y", ctypes.c_uint32),
        ("block_z", ctypes.c_uint32),
        ("shared_memory_bytes", ctypes.c_uint32),
        ("stream", ctypes.c_void_p),
        ("kernel_params", ctypes.POINTER(ctypes.c_void_p)),
        ("extra", ctypes.POINTER(ctypes.c_void_p)),
    ]


class _LaunchConfig(ctypes.Structure):
    _fields_ = [
        ("grid_x", ctypes.c_uint32),
        ("grid_y", ctypes.c_uint32),
        ("grid_z", ctypes.c_uint32),
        ("block_x", ctypes.c_uint32),
        ("block_y", ctypes.c_uint32),
        ("block_z", ctypes.c_uint32),
        ("shared_memory_bytes", ctypes.c_uint32),
        ("stream", ctypes.c_void_p),
        ("attributes", ctypes.c_void_p),
        ("attribute_count", ctypes.c_uint32),
    ]


class _LaunchKernelExParams(ctypes.Structure):
    _fields_ = [
        ("config", ctypes.POINTER(_LaunchConfig)),
        ("function", ctypes.c_void_p),
        ("kernel_params", ctypes.POINTER(ctypes.c_void_p)),
        ("extra", ctypes.POINTER(ctypes.c_void_p)),
    ]


def _handle_hash(value: int | None) -> str:
    return hashlib.sha256(str(value or 0).encode("ascii")).hexdigest()


class CuptiLaunchArgumentCapture:
    def __init__(self, kernel_name: str) -> None:
        self.kernel_name = kernel_name
        self.records: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.range_provider: Callable[[], list[str]] = lambda: []
        library_path = _find_cupti_library()
        self.library = ctypes.WinDLL(str(library_path)) if os.name == "nt" else ctypes.CDLL(str(library_path))
        self.driver = ctypes.WinDLL("nvcuda.dll") if os.name == "nt" else ctypes.CDLL("libcuda.so.1")
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
        self.driver.cuFuncGetParamInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.driver.cuFuncGetParamInfo.restype = ctypes.c_int
        self.active = False

    def _parameters(self, function: int, values: Any, symbol: str = "") -> list[dict[str, Any]]:
        if not values:
            return []
        records = []
        for index in range(256):
            offset = ctypes.c_size_t()
            size = ctypes.c_size_t()
            result = self.driver.cuFuncGetParamInfo(function, index, ctypes.byref(offset), ctypes.byref(size))
            if result == 1:
                break
            if result != 0:
                raise RuntimeError(f"cuFuncGetParamInfo failed with result {result} at index {index}")
            if size.value == 0 or size.value > 4096:
                raise ValueError(f"Unexpected kernel parameter size {size.value} at index {index}")
            address = values[index]
            if not address:
                raise ValueError(f"Null kernel parameter storage at index {index}")
            value = ctypes.string_at(address, size.value)
            record = {
                "index": index,
                "device_layout_offset": int(offset.value),
                "size_bytes": int(size.value),
                "value_sha256": hashlib.sha256(value).hexdigest(),
                "aligned_pointer_candidates": [
                    {
                        "byte_offset": byte_offset,
                        "pointer_value": int.from_bytes(
                            value[byte_offset : byte_offset + ctypes.sizeof(ctypes.c_void_p)], "little"
                        ),
                        "pointer_value_sha256": _handle_hash(
                            int.from_bytes(value[byte_offset : byte_offset + ctypes.sizeof(ctypes.c_void_p)], "little")
                        ),
                    }
                    for byte_offset in range(0, size.value - ctypes.sizeof(ctypes.c_void_p) + 1, ctypes.sizeof(ctypes.c_void_p))
                ],
            }
            if index == 0 and "fmha_cutlass" in symbol and size.value == 264:
                from .kernel_signatures import decode_attention_params

                record["typed_decoding"] = decode_attention_params(value)
            records.append(record)
        return records

    def _callback(self, userdata: int, domain: int, callback_id: int, callback_pointer: int) -> None:
        if domain != CUPTI_CB_DOMAIN_DRIVER_API or callback_id not in CUPTI_LAUNCH_CALLBACKS:
            return
        try:
            callback = ctypes.cast(callback_pointer, ctypes.POINTER(_CallbackData)).contents
            if callback.callbackSite != CUPTI_API_ENTER:
                return
            symbol = callback.symbolName.decode("utf-8") if callback.symbolName else ""
            if symbol != self.kernel_name:
                return
            if callback_id in (652, 653):
                parameters = ctypes.cast(callback.functionParams, ctypes.POINTER(_LaunchKernelExParams)).contents
                config = parameters.config.contents
                function = parameters.function
                kernel_params = parameters.kernel_params
                extra = parameters.extra
            else:
                parameters = ctypes.cast(callback.functionParams, ctypes.POINTER(_LaunchKernelParams)).contents
                config = parameters
                function = parameters.function
                kernel_params = parameters.kernel_params
                extra = parameters.extra
            self.records.append(
                {
                    "callback": CUPTI_LAUNCH_CALLBACKS[callback_id],
                    "symbol_name": symbol,
                    "correlation_id": int(callback.correlationId),
                    "context_uid": int(callback.contextUid),
                    "grid": [int(config.grid_x), int(config.grid_y), int(config.grid_z)],
                    "block": [int(config.block_x), int(config.block_y), int(config.block_z)],
                    "shared_memory_bytes": int(config.shared_memory_bytes),
                    "stream_handle_sha256": _handle_hash(config.stream),
                    "function_handle_sha256": _handle_hash(function),
                    "kernel_params_present": bool(kernel_params),
                    "extra_present": bool(extra),
                    "qualified_module_stack": self.range_provider(),
                    "parameters": self._parameters(function, kernel_params, symbol),
                }
            )
        except Exception as exc:
            self.errors.append(f"{type(exc).__name__}: {exc}")

    def start(self) -> None:
        result = self.library.cuptiSubscribe(ctypes.byref(self.subscriber), self.callback, None)
        if result != 0:
            raise RuntimeError(f"cuptiSubscribe failed with result {result}")
        for callback_id in CUPTI_LAUNCH_CALLBACKS:
            result = self.library.cuptiEnableCallback(1, self.subscriber, CUPTI_CB_DOMAIN_DRIVER_API, callback_id)
            if result != 0:
                self.library.cuptiUnsubscribe(self.subscriber)
                raise RuntimeError(f"cuptiEnableCallback failed for {callback_id} with result {result}")
        self.active = True

    def stop(self) -> None:
        if self.active:
            result = self.library.cuptiUnsubscribe(self.subscriber)
            self.active = False
            if result != 0:
                raise RuntimeError(f"cuptiUnsubscribe failed with result {result}")

    def report(
        self, module_report: dict[str, Any], tensor_storage_ranges: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        pointer_modules = {}
        for invocation in module_report.get("invocations", []):
            for role in ("inputs", "parameters", "outputs"):
                for tensor in invocation[role]:
                    pointer_modules.setdefault(tensor["data_pointer_sha256"], []).append(
                        {"module": invocation["module"], "role": role[:-1], "tensor_sha256": tensor["sha256"]}
                    )
        ranges = tensor_storage_ranges or []
        records = []
        for launch in self.records:
            body = copy.deepcopy(launch)
            body["parameter_pointer_matches"] = [
                {
                    "parameter_index": parameter["index"],
                    "parameter_byte_offset": candidate["byte_offset"],
                    "matches": pointer_modules[candidate["pointer_value_sha256"]],
                }
                for parameter in launch["parameters"]
                for candidate in parameter["aligned_pointer_candidates"]
                if candidate["pointer_value_sha256"] in pointer_modules
            ]
            body["parameter_storage_range_matches"] = [
                {
                    "parameter_index": parameter["index"],
                    "parameter_byte_offset": candidate["byte_offset"],
                    "module": tensor_range["module"],
                    "role": tensor_range["role"],
                    "temporal_status": (
                        "post_launch_retrospective"
                        if tensor_range["role"] == "output"
                        or tensor_range["role"].startswith("attention_dispatch_output_")
                        else "pre_launch_retained"
                    ),
                    "tensor_sha256": tensor_range["tensor_sha256"],
                    "storage_offset_bytes": candidate["pointer_value"] - tensor_range["storage_base"],
                    "equals_tensor_data_pointer": candidate["pointer_value"]
                    == tensor_range["tensor_data_pointer"],
                }
                for parameter in launch["parameters"]
                for candidate in parameter["aligned_pointer_candidates"]
                for tensor_range in ranges
                if tensor_range["storage_base"]
                <= candidate["pointer_value"]
                < tensor_range["storage_base"] + tensor_range["storage_nbytes"]
            ]
            for parameter in body["parameters"]:
                for candidate in parameter["aligned_pointer_candidates"]:
                    candidate.pop("pointer_value", None)
            body["launch_sha256"] = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
            records.append(body)
        body = {
            "scope": "CUPTI driver-entry launch geometry and parameter-value commitments for one exact kernel symbol; pointer matches are against active module-boundary tensors, not type-decoded kernel signatures.",
            "kernel_name": self.kernel_name,
            "launches": records,
            "callback_errors": self.errors,
            "module_report_sha256": module_report["report_sha256"],
            "limitations": [
                "Parameter bytes are hashed and aligned pointer-sized windows are treated only as pointer candidates.",
                "Pointer equality or storage-range containment does not establish argument type, access direction, bounds of access, or aliasing semantics.",
                "Packed extra-parameter buffers are recorded as present but are not decoded.",
            ],
        }
        body["report_sha256"] = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
        return body


def redact_launch_argument_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    redacted = copy.deepcopy(artifact)
    redacted["privacy"] = {"redacted": True}
    redacted["model_state_sha256"] = "redacted"
    redacted["selected_token_id"] = "redacted"
    for key in ("input_ids_tensor", "output_logits_tensor"):
        redacted[key]["sha256"] = "redacted"
    module_report = redacted["module_invocation_report"]
    for invocation in module_report["invocations"]:
        for role in ("inputs", "parameters", "outputs"):
            for tensor in invocation[role]:
                tensor["sha256"] = "redacted"
                tensor["data_pointer_sha256"] = "redacted"
                if "storage_base_pointer_sha256" in tensor:
                    tensor["storage_base_pointer_sha256"] = "redacted"
        invocation_body = {key: value for key, value in invocation.items() if key != "invocation_sha256"}
        invocation["invocation_sha256"] = hashlib.sha256(canonical_json(invocation_body).encode("utf-8")).hexdigest()
    module_body = {key: value for key, value in module_report.items() if key != "report_sha256"}
    module_report["report_sha256"] = hashlib.sha256(canonical_json(module_body).encode("utf-8")).hexdigest()
    dispatch_report = redacted.get("attention_dispatch_report")
    if dispatch_report is not None:
        for operation in dispatch_report["operations"]:
            for role in ("inputs", "outputs"):
                for tensor in operation[role]:
                    tensor["sha256"] = "redacted"
                    tensor["data_pointer_sha256"] = "redacted"
                    if "storage_base_pointer_sha256" in tensor:
                        tensor["storage_base_pointer_sha256"] = "redacted"
            operation_body = {key: value for key, value in operation.items() if key != "operation_sha256"}
            operation["operation_sha256"] = hashlib.sha256(canonical_json(operation_body).encode("utf-8")).hexdigest()
        dispatch_body = {key: value for key, value in dispatch_report.items() if key != "report_sha256"}
        dispatch_report["report_sha256"] = hashlib.sha256(canonical_json(dispatch_body).encode("utf-8")).hexdigest()
    launch_report = redacted["launch_argument_report"]
    launch_report["module_report_sha256"] = module_report["report_sha256"]
    for launch in launch_report["launches"]:
        for parameter in launch["parameters"]:
            parameter["value_sha256"] = "redacted"
            if parameter.get("typed_decoding"):
                parameter["typed_decoding"]["pointer_field_sha256"] = {
                    name: "redacted" for name in parameter["typed_decoding"]["pointer_field_sha256"]
                }
                parameter["typed_decoding"]["rng_state_sha256"] = "redacted"
            for candidate in parameter["aligned_pointer_candidates"]:
                candidate["pointer_value_sha256"] = "redacted"
        for match in launch["parameter_pointer_matches"]:
            for tensor in match["matches"]:
                tensor["tensor_sha256"] = "redacted"
        for match in launch.get("parameter_storage_range_matches", []):
            match["tensor_sha256"] = "redacted"
        launch_body = {key: value for key, value in launch.items() if key != "launch_sha256"}
        launch["launch_sha256"] = hashlib.sha256(canonical_json(launch_body).encode("utf-8")).hexdigest()
    launch_body = {key: value for key, value in launch_report.items() if key != "report_sha256"}
    launch_report["report_sha256"] = hashlib.sha256(canonical_json(launch_body).encode("utf-8")).hexdigest()
    artifact_body = {key: value for key, value in redacted.items() if key != "artifact_sha256"}
    redacted["artifact_sha256"] = hashlib.sha256(canonical_json(artifact_body).encode("utf-8")).hexdigest()
    return redacted


def verify_launch_argument_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    artifact_hash_valid = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest() == artifact.get(
        "artifact_sha256"
    )
    from .module_invocation import verify_attention_dispatch_report, verify_module_invocation_report

    module_valid = verify_module_invocation_report(artifact.get("module_invocation_report", {}))["valid"]
    dispatch_report = artifact.get("attention_dispatch_report")
    dispatch_valid = True if dispatch_report is None else verify_attention_dispatch_report(dispatch_report)["valid"]
    launch_valid = verify_launch_argument_report(artifact.get("launch_argument_report", {}))["valid"]
    return {
        "valid": bool(artifact_hash_valid and module_valid and dispatch_valid and launch_valid),
        "artifact_hash_valid": artifact_hash_valid,
        "module_report_valid": module_valid,
        "attention_dispatch_report_valid": dispatch_valid,
        "launch_report_valid": launch_valid,
    }


def build_launch_argument_summary(artifacts: list[tuple[dict[str, Any], str]]) -> dict[str, Any]:
    entries = []
    for artifact, expected_module in artifacts:
        verification = verify_launch_argument_artifact(artifact)
        launches = artifact["launch_argument_report"]["launches"]
        scoped = [
            launch
            for launch in launches
            if launch["qualified_module_stack"] and launch["qualified_module_stack"][-1] == expected_module
        ]
        if len(scoped) != 1:
            raise ValueError(f"Expected one launch scoped to {expected_module!r}; found {len(scoped)}")
        launch = scoped[0]
        matches = []
        for match in launch["parameter_pointer_matches"]:
            for tensor in match["matches"]:
                if tensor["module"] == expected_module:
                    matches.append(
                        {
                            "parameter_index": match["parameter_index"],
                            "parameter_byte_offset": match["parameter_byte_offset"],
                            "role": tensor["role"],
                            "temporal_status": (
                                "post_launch_retrospective"
                                if tensor["role"] == "output"
                                else "pre_launch_retained"
                            ),
                            "tensor_sha256": tensor["tensor_sha256"],
                        }
                    )
        range_matches = [
            match for match in launch.get("parameter_storage_range_matches", []) if match["module"] == expected_module
        ]
        entries.append(
            {
                "kernel_name": artifact["launch_argument_report"]["kernel_name"],
                "expected_module": expected_module,
                "total_symbol_launches": len(launches),
                "module_scoped_launches": len(scoped),
                "callback": launch["callback"],
                "correlation_id": launch["correlation_id"],
                "grid": launch["grid"],
                "block": launch["block"],
                "shared_memory_bytes": launch["shared_memory_bytes"],
                "parameter_count": len(launch["parameters"]),
                "parameter_sizes": [parameter["size_bytes"] for parameter in launch["parameters"]],
                "boundary_pointer_matches": matches,
                "storage_range_matches": range_matches,
                "input_boundary_pointer_match": any(match["role"] == "input" for match in matches),
                "parameter_boundary_pointer_match": any(match["role"] == "parameter" for match in matches),
                "retrospective_output_address_match": any(match["role"] == "output" for match in matches),
                "artifact_sha256": artifact["artifact_sha256"],
                "artifact_valid": verification["valid"],
            }
        )
    body = {
        "scope": "CUPTI driver-entry launch parameter commitments with pre-launch retained input/parameter matches and post-launch retrospective output-address matches; not typed signature or memory-access proof.",
        "entries": entries,
        "valid_entries": sum(entry["artifact_valid"] for entry in entries),
        "total_entries": len(entries),
        "entries_with_input_boundary_match": sum(entry["input_boundary_pointer_match"] for entry in entries),
        "entries_with_parameter_boundary_match": sum(entry["parameter_boundary_pointer_match"] for entry in entries),
        "entries_with_retrospective_output_address_match": sum(entry["retrospective_output_address_match"] for entry in entries),
        "storage_range_match_count": sum(len(entry["storage_range_matches"]) for entry in entries),
        "typed_kernel_signatures_established": False,
        "complete_argument_binding_established": False,
        "limitations": [
            "Aligned 64-bit values inside packed parameters are pointer candidates until signatures are independently typed.",
            "Pointer equality or storage containment does not establish read/write direction, bounds of access, aliasing, or access behavior.",
            "Post-launch output addresses can equal earlier internal allocations through allocator reuse until a typed field resolves the role.",
            "Fused and reduction kernels can consume intermediates not present at the enclosing module boundary.",
        ],
    }
    body["summary_sha256"] = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    return body


def verify_launch_argument_summary(summary: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in summary.items() if key != "summary_sha256"}
    hash_valid = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest() == summary.get("summary_sha256")
    entries = summary.get("entries", [])
    counts_consistent = (
        summary.get("total_entries") == len(entries)
        and summary.get("valid_entries") == sum(entry.get("artifact_valid", False) for entry in entries)
        and summary.get("entries_with_input_boundary_match")
        == sum(entry.get("input_boundary_pointer_match", False) for entry in entries)
        and summary.get("entries_with_parameter_boundary_match")
        == sum(entry.get("parameter_boundary_pointer_match", False) for entry in entries)
        and summary.get("entries_with_retrospective_output_address_match")
        == sum(entry.get("retrospective_output_address_match", False) for entry in entries)
        and summary.get("storage_range_match_count") == sum(len(entry.get("storage_range_matches", [])) for entry in entries)
    )
    boundaries_preserved = (
        summary.get("typed_kernel_signatures_established") is False
        and summary.get("complete_argument_binding_established") is False
    )
    return {
        "valid": bool(hash_valid and counts_consistent and boundaries_preserved),
        "summary_hash_valid": hash_valid,
        "counts_consistent": counts_consistent,
        "boundaries_preserved": boundaries_preserved,
    }


def verify_launch_argument_report(report: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in report.items() if key != "report_sha256"}
    report_hash_valid = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest() == report.get(
        "report_sha256"
    )
    launch_hashes_valid = all(
        hashlib.sha256(
            canonical_json({key: value for key, value in launch.items() if key != "launch_sha256"}).encode("utf-8")
        ).hexdigest()
        == launch.get("launch_sha256")
        for launch in report.get("launches", [])
    )
    return {
        "valid": bool(report_hash_valid and launch_hashes_valid and report.get("launches") and not report.get("callback_errors")),
        "report_hash_valid": report_hash_valid,
        "launch_hashes_valid": launch_hashes_valid,
        "callback_errors": report.get("callback_errors", []),
    }
