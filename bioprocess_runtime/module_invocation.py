from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import shutil
import subprocess
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from .operational_semantics import tensor_descriptor
from .serialization import canonical_json

try:
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
except ModuleNotFoundError:
    torch = None
    TorchDispatchMode = object


def _tensor_values(value: Any) -> list[Any]:
    if torch is not None and isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [tensor for item in value for tensor in _tensor_values(item)]
    if isinstance(value, dict):
        return [tensor for key in sorted(value) for tensor in _tensor_values(value[key])]
    if hasattr(value, "to_tuple"):
        return _tensor_values(value.to_tuple())
    return []


def _invocation_tensor_record(tensor: Any) -> dict[str, Any]:
    descriptor = tensor_descriptor(tensor)
    descriptor["stride"] = list(tensor.stride())
    storage = tensor.untyped_storage()
    descriptor["storage_offset"] = int(tensor.storage_offset())
    descriptor["element_size_bytes"] = int(tensor.element_size())
    descriptor["storage_nbytes"] = int(storage.nbytes())
    descriptor["data_pointer_offset_bytes"] = int(tensor.data_ptr() - storage.data_ptr())
    descriptor["storage_base_pointer_sha256"] = hashlib.sha256(str(storage.data_ptr()).encode("ascii")).hexdigest()
    descriptor["data_pointer_sha256"] = hashlib.sha256(str(tensor.data_ptr()).encode("ascii")).hexdigest()
    return descriptor


class AttentionDispatchCapture(TorchDispatchMode):
    def __init__(self, module_capture: "ModuleNvtxCapture") -> None:
        super().__init__()
        self.module_capture = module_capture
        self.operations: list[dict[str, Any]] = []

    def __torch_dispatch__(self, function: Any, types: tuple[type, ...], arguments: tuple[Any, ...] = (), keyword_arguments: dict[str, Any] | None = None) -> Any:
        keyword_arguments = keyword_arguments or {}
        active_modules = [invocation["module"] for invocation in self.module_capture.open_stack]
        selected = bool(active_modules) and "scaled_dot_product" in str(function)
        schema = getattr(function, "_schema", None)
        schema_arguments = list(schema.arguments) if schema is not None else []
        input_tensors = []
        input_names = []
        if selected:
            for index, argument in enumerate(arguments):
                tensors = _tensor_values(argument)
                input_tensors.extend(tensors)
                name = schema_arguments[index].name if index < len(schema_arguments) else f"argument_{index}"
                input_names.extend(name for tensor in tensors)
            for name, argument in keyword_arguments.items():
                tensors = _tensor_values(argument)
                input_tensors.extend(tensors)
                input_names.extend(name for tensor in tensors)
        output = function(*arguments, **keyword_arguments)
        if selected:
            self.operations.append(
                {
                    "operation": str(function),
                    "qualified_module_stack": active_modules,
                    "schema": str(schema) if schema is not None else None,
                    "input_names": input_names,
                    "input_tensors": input_tensors,
                    "output_tensors": _tensor_values(output),
                }
            )
        return output

    def tensor_storage_ranges(self) -> list[dict[str, Any]]:
        ranges = []
        for operation_index, operation in enumerate(self.operations):
            for role, key in (("input", "input_tensors"), ("output", "output_tensors")):
                for tensor_index, tensor in enumerate(operation[key]):
                    storage = tensor.untyped_storage()
                    ranges.append(
                        {
                            "module": operation["qualified_module_stack"][-1],
                            "role": f"attention_dispatch_{role}_{tensor_index}",
                            "tensor_sha256": tensor_descriptor(tensor)["sha256"],
                            "storage_base": int(storage.data_ptr()),
                            "storage_nbytes": int(storage.nbytes()),
                            "tensor_data_pointer": int(tensor.data_ptr()),
                            "operation_index": operation_index,
                        }
                    )
        return ranges

    def report(self) -> dict[str, Any]:
        operations = []
        for index, operation in enumerate(self.operations):
            body = {
                "index": index,
                "operation": operation["operation"],
                "schema": operation["schema"],
                "qualified_module_stack": operation["qualified_module_stack"],
                "input_names": operation["input_names"],
                "inputs": [_invocation_tensor_record(tensor) for tensor in operation["input_tensors"]],
                "outputs": [_invocation_tensor_record(tensor) for tensor in operation["output_tensors"]],
            }
            body["operation_sha256"] = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
            operations.append(body)
        report = {
            "scope": "Layer-local scaled-dot-product dispatcher input/output commitments; not proof of fused-kernel argument order.",
            "operations": operations,
        }
        report["report_sha256"] = hashlib.sha256(canonical_json(report).encode("utf-8")).hexdigest()
        return report


def verify_attention_dispatch_report(report: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in report.items() if key != "report_sha256"}
    report_hash_valid = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest() == report.get(
        "report_sha256"
    )
    operation_hashes_valid = all(
        hashlib.sha256(
            canonical_json({key: value for key, value in operation.items() if key != "operation_sha256"}).encode(
                "utf-8"
            )
        ).hexdigest()
        == operation.get("operation_sha256")
        for operation in report.get("operations", [])
    )
    return {
        "valid": bool(report_hash_valid and operation_hashes_valid and report.get("operations")),
        "report_hash_valid": report_hash_valid,
        "operation_hashes_valid": operation_hashes_valid,
    }


class ModuleNvtxCapture(AbstractContextManager["ModuleNvtxCapture"]):
    def __init__(self, model: Any, patterns: tuple[str, ...]) -> None:
        if torch is None or not torch.cuda.is_available():
            raise RuntimeError("CUDA-enabled PyTorch is required")
        self.model = model
        self.patterns = tuple(re.compile(pattern) for pattern in patterns)
        self.handles = []
        self.pending: dict[str, list[dict[str, Any]]] = {}
        self.open_stack: list[dict[str, Any]] = []
        self.invocations: list[dict[str, Any]] = []
        self.lifecycle_errors: list[str] = []

    def _selected(self, name: str) -> bool:
        return any(pattern.fullmatch(name) for pattern in self.patterns)

    def _pre_hook(self, name: str):
        def hook(module: Any, arguments: tuple[Any, ...], keyword_arguments: dict[str, Any]) -> None:
            invocation = {
                "module": name,
                "class": type(module).__name__,
                "nvtx_range": f"gemma_module:{name}",
                "input_tensors": _tensor_values((arguments, keyword_arguments)),
                "parameter_tensors": list(module.parameters(recurse=False)),
            }
            torch.cuda.nvtx.range_push(invocation["nvtx_range"])
            self.pending.setdefault(name, []).append(invocation)
            self.open_stack.append(invocation)

        return hook

    def _post_hook(self, name: str):
        def hook(module: Any, arguments: tuple[Any, ...], keyword_arguments: dict[str, Any], output: Any) -> None:
            pending = self.pending.get(name, [])
            if not pending:
                self.lifecycle_errors.append(f"Unpaired post hook for {name}")
                return
            invocation = pending.pop()
            if not self.open_stack or self.open_stack[-1] is not invocation:
                self.lifecycle_errors.append(f"Non-LIFO post hook for {name}")
                return
            invocation["output_tensors"] = _tensor_values(output)
            self.invocations.append(invocation)
            self.open_stack.pop()
            torch.cuda.nvtx.range_pop()

        return hook

    def __enter__(self) -> "ModuleNvtxCapture":
        for name, module in self.model.named_modules():
            if self._selected(name):
                self.handles.append(module.register_forward_pre_hook(self._pre_hook(name), with_kwargs=True))
                self.handles.append(
                    module.register_forward_hook(self._post_hook(name), with_kwargs=True, always_call=True)
                )
        if not self.handles:
            raise ValueError("No modules matched the requested NVTX patterns")
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        while self.open_stack:
            invocation = self.open_stack.pop()
            pending = self.pending.get(invocation["module"], [])
            if invocation in pending:
                pending.remove(invocation)
            torch.cuda.nvtx.range_pop()

    def tensor_storage_ranges(self) -> list[dict[str, Any]]:
        ranges = []
        for invocation in self.invocations:
            for role, key in (("input", "input_tensors"), ("parameter", "parameter_tensors"), ("output", "output_tensors")):
                for tensor in invocation[key]:
                    storage = tensor.untyped_storage()
                    ranges.append(
                        {
                            "module": invocation["module"],
                            "role": role,
                            "tensor_sha256": tensor_descriptor(tensor)["sha256"],
                            "storage_base": int(storage.data_ptr()),
                            "storage_nbytes": int(storage.nbytes()),
                            "tensor_data_pointer": int(tensor.data_ptr()),
                        }
                    )
        return ranges

    def report(self) -> dict[str, Any]:
        records = []
        for index, invocation in enumerate(self.invocations):
            body = {
                "index": index,
                "module": invocation["module"],
                "class": invocation["class"],
                "nvtx_range": invocation["nvtx_range"],
                "inputs": [_invocation_tensor_record(tensor) for tensor in invocation["input_tensors"]],
                "parameters": [_invocation_tensor_record(tensor) for tensor in invocation["parameter_tensors"]],
                "outputs": [_invocation_tensor_record(tensor) for tensor in invocation["output_tensors"]],
            }
            body["invocation_sha256"] = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
            records.append(body)
        report = {
            "scope": "Qualified module NVTX ranges with deferred logical tensor commitments; not CUDA launch-argument extraction.",
            "patterns": [pattern.pattern for pattern in self.patterns],
            "invocations": records,
            "lifecycle_errors": self.lifecycle_errors,
        }
        report["report_sha256"] = hashlib.sha256(canonical_json(report).encode("utf-8")).hexdigest()
        return report


def verify_module_invocation_report(report: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in report.items() if key != "report_sha256"}
    report_hash_valid = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest() == report.get(
        "report_sha256"
    )
    invocation_hashes_valid = all(
        hashlib.sha256(
            canonical_json({key: value for key, value in invocation.items() if key != "invocation_sha256"}).encode(
                "utf-8"
            )
        ).hexdigest()
        == invocation.get("invocation_sha256")
        for invocation in report.get("invocations", [])
    )
    return {
        "valid": bool(
            report_hash_valid
            and invocation_hashes_valid
            and report.get("invocations")
            and not report.get("lifecycle_errors")
        ),
        "report_hash_valid": report_hash_valid,
        "invocation_hashes_valid": invocation_hashes_valid,
    }


def _raw_nvtx_identity(output: str) -> dict[str, Any]:
    rows = list(csv.DictReader(io.StringIO(output)))
    data = [row for row in rows if row.get("Process ID")]
    if len(data) != 1:
        raise ValueError("Expected one Nsight raw launch row")
    row = data[0]
    range_column = next((name for name in row if "Push/Pop_Range" in name), None)
    if range_column is None:
        raise ValueError("Nsight raw output did not include a push/pop NVTX column")
    ranges = re.findall(r'"<default domain>:(.*?):none:none:none:none:none:none"', row[range_column])
    return {
        "process_id": int(row["Process ID"]),
        "kernel_name": row["Kernel Name"],
        "nvtx_ranges": ranges,
        "qualified_modules": [value.removeprefix("gemma_module:") for value in ranges if value.startswith("gemma_module:")],
    }


def build_module_invocation_certificate(
    nsight_report: Path,
    execution_binding_path: Path,
    launch_certificate_path: Path,
    expected_innermost_module: str,
) -> dict[str, Any]:
    ncu = shutil.which("ncu")
    if ncu is None:
        raise RuntimeError("ncu is required")
    binding = json.loads(execution_binding_path.read_text(encoding="utf-8"))
    launch = json.loads(launch_certificate_path.read_text(encoding="utf-8"))
    module_report = binding.get("module_invocation_report")
    if module_report is None:
        raise ValueError("Execution binding has no module invocation report")
    completed = subprocess.run(
        [ncu, "--import", str(nsight_report), "--page", "raw", "--print-kernel-base", "mangled", "--csv"],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    raw_identity = _raw_nvtx_identity(completed.stdout)
    invocation_by_module = {invocation["module"]: invocation for invocation in module_report["invocations"]}
    matched_invocations = [invocation_by_module[name] for name in raw_identity["qualified_modules"] if name in invocation_by_module]
    binding_body = {key: value for key, value in binding.items() if key != "binding_sha256"}
    binding_hash_valid = hashlib.sha256(canonical_json(binding_body).encode("utf-8")).hexdigest() == binding.get(
        "binding_sha256"
    )
    module_report_valid = verify_module_invocation_report(module_report)["valid"]
    checks = {
        "binding_hash_valid": binding_hash_valid,
        "module_invocation_report_valid": module_report_valid,
        "process_id_matches": raw_identity["process_id"] == binding["process_id"],
        "kernel_matches_launch_certificate": raw_identity["kernel_name"] == launch["details"]["kernel_name"],
        "outer_forward_range_present": "gemma_bound_forward" in raw_identity["nvtx_ranges"],
        "expected_module_is_innermost": bool(raw_identity["qualified_modules"])
        and raw_identity["qualified_modules"][-1] == expected_innermost_module,
        "every_reported_module_has_invocation_commitment": len(matched_invocations)
        == len(raw_identity["qualified_modules"]),
        "module_tensor_commitments_present": all(
            invocation["inputs"] and invocation["outputs"] for invocation in matched_invocations
        ),
    }
    body = {
        "scope": "Nsight-reported nested NVTX module stack linked to deferred module input/output tensor commitments for one kernel launch; not extraction of CUDA kernel argument values.",
        "launch_certificate_sha256": launch["certificate_sha256"],
        "execution_binding_sha256": binding["binding_sha256"],
        "module_report_sha256": module_report["report_sha256"],
        "raw_output_sha256": hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest(),
        "raw_identity": raw_identity,
        "expected_innermost_module": expected_innermost_module,
        "matched_module_invocations": matched_invocations,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "limitations": [
            "Module tensor hashes are captured by framework hooks, not decoded from CUDA launch parameters.",
            "A module range can contain multiple kernel launches and intermediate tensors not exposed by the module boundary.",
            "Logical tensor commitments do not prove device-pointer or kernel-argument binding.",
        ],
    }
    body["certificate_sha256"] = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    return body


def verify_module_invocation_certificate(certificate: dict[str, Any]) -> dict[str, Any]:
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


def build_module_invocation_summary(certificates: list[dict[str, Any]]) -> dict[str, Any]:
    entries = [
        {
            "kernel_name": certificate["raw_identity"]["kernel_name"],
            "qualified_modules": certificate["raw_identity"]["qualified_modules"],
            "innermost_module": certificate["expected_innermost_module"],
            "module_invocation_sha256": certificate["matched_module_invocations"][-1]["invocation_sha256"],
            "input_tensor_count": len(certificate["matched_module_invocations"][-1]["inputs"]),
            "output_tensor_count": len(certificate["matched_module_invocations"][-1]["outputs"]),
            "certificate_sha256": certificate["certificate_sha256"],
            "valid": verify_module_invocation_certificate(certificate)["valid"],
        }
        for certificate in certificates
    ]
    body = {
        "scope": "Representative layer-0 module invocation bindings for attention, projection, RMS normalization, and GELU; not complete per-launch argument binding.",
        "entries": entries,
        "valid_entries": sum(entry["valid"] for entry in entries),
        "total_entries": len(entries),
        "complete": bool(entries) and all(entry["valid"] for entry in entries),
        "full_kernel_argument_binding_established": False,
    }
    body["summary_sha256"] = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    return body


def verify_module_invocation_summary(summary: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in summary.items() if key != "summary_sha256"}
    hash_valid = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest() == summary.get("summary_sha256")
    counts_consistent = (
        summary.get("total_entries") == len(summary.get("entries", []))
        and summary.get("valid_entries") == sum(entry.get("valid", False) for entry in summary.get("entries", []))
        and summary.get("complete")
        == (bool(summary.get("entries")) and all(entry.get("valid", False) for entry in summary.get("entries", [])))
    )
    semantic_boundary = summary.get("full_kernel_argument_binding_established") is False
    return {
        "valid": bool(hash_valid and counts_consistent and semantic_boundary),
        "summary_hash_valid": hash_valid,
        "counts_consistent": counts_consistent,
        "semantic_boundary_preserved": semantic_boundary,
    }
