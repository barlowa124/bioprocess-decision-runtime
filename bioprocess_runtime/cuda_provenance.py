from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from .operational_semantics import tensor_descriptor
from .serialization import canonical_json

try:
    import torch
    from torch.profiler import ProfilerActivity, profile
except ModuleNotFoundError:
    torch = None
    ProfilerActivity = None
    profile = None


def _require_cuda() -> None:
    if torch is None or profile is None or not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled PyTorch installation and GPU are required")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tool_version(path: str) -> str:
    completed = subprocess.run([path, "--version"], check=True, capture_output=True, text=True, timeout=30)
    return (completed.stdout or completed.stderr).strip()


def _binary_inventory(torch_library: Path | None = None) -> tuple[list[dict[str, Any]], Path]:
    library = torch_library or Path(torch.__file__).resolve().parent / "lib"
    role_patterns = {
        "torch_cuda": ("torch_cuda.dll", "libtorch_cuda.so*", "libtorch_cuda.dylib"),
        "cublas": ("cublas64_*.dll", "libcublas.so*", "libcublas.dylib"),
        "cublas_lt": ("cublasLt64_*.dll", "libcublasLt.so*", "libcublasLt.dylib"),
        "cuda_runtime": ("cudart64_*.dll", "libcudart.so*", "libcudart.dylib"),
    }
    binaries = []
    torch_cuda = None
    for role, patterns in role_patterns.items():
        matches = sorted({path.resolve() for pattern in patterns for path in library.glob(pattern) if path.is_file()})
        if not matches:
            continue
        path = matches[0]
        binaries.append({"role": role, "name": path.name, "size_bytes": path.stat().st_size, "sha256": _file_sha256(path)})
        if role == "torch_cuda":
            torch_cuda = path
    if torch_cuda is None:
        raise RuntimeError(f"No torch CUDA binary found under {library}")
    return binaries, torch_cuda


def _parse_embedded_images(listing: str) -> dict[str, Any]:
    architectures = Counter(re.findall(r"\.sm_(\d+[a-z]?)\.cubin", listing))
    return {
        "total_images": sum(architectures.values()),
        "architectures": {f"sm_{key}": architectures[key] for key in sorted(architectures)},
        "listing_sha256": hashlib.sha256(listing.encode("utf-8")).hexdigest(),
    }


def _compatible_architecture(supported: list[str], capability: tuple[int, int]) -> str:
    exact = f"sm_{capability[0]}{capability[1]}"
    if exact in supported:
        return exact
    numeric = [int(match.group(1)) for item in supported if (match := re.fullmatch(r"sm_(\d+)", item))]
    compatible = [value for value in numeric if value <= capability[0] * 10 + capability[1]]
    if not compatible:
        raise RuntimeError(f"No compatible CUDA architecture for compute capability {capability}")
    return f"sm_{max(compatible)}"


def _embedded_images(cuobjdump: str, torch_cuda: Path) -> tuple[dict[str, Any], str]:
    completed = subprocess.run(
        [cuobjdump, "--list-elf", str(torch_cuda)], check=True, capture_output=True, text=True, timeout=180
    )
    listing = completed.stdout
    return _parse_embedded_images(listing), listing


def _locate_embedded_image(output: str, image_listing: str, architecture: str, function_name: str) -> str | None:
    marker = f"Function : {function_name}"
    if marker not in output:
        return None
    architecture_images = re.findall(rf"ELF file\s+\d+:\s+(\S+\.{re.escape(architecture)}\.cubin)", image_listing)
    marker_offset = output.index(marker)
    section_index = output[:marker_offset].count("Fatbin elf code:") - 1
    return architecture_images[section_index] if 0 <= section_index < len(architecture_images) else None


def _disassemble_profiled_function(
    cuobjdump: str,
    nvdisasm: str,
    torch_cuda: Path,
    event_counts: Counter[str],
    architecture: str,
    image_listing: str,
    kernel_filter: str | None = None,
) -> dict[str, Any]:
    try:
        expression = re.compile(kernel_filter) if kernel_filter else None
    except re.error as exc:
        raise ValueError(f"Invalid --kernel regular expression: {exc}") from exc
    candidates = [
        name
        for name in sorted(event_counts, key=lambda item: (-event_counts[item], item))
        if name.startswith("_ZN2at6native") and (expression is None or expression.search(name))
    ]
    failures = []
    for name in candidates:
        completed = subprocess.run(
            [cuobjdump, "--dump-sass", "--gpu-architecture", architecture, "--function", name, str(torch_cuda)],
            capture_output=True,
            text=True,
            timeout=180,
        )
        output = completed.stdout
        marker = f"Function : {name}"
        if completed.returncode == 0 and marker in output:
            image_name = _locate_embedded_image(output, image_listing, architecture, name)
            if image_name is not None:
                with tempfile.TemporaryDirectory() as directory:
                    subprocess.run(
                        [cuobjdump, "--extract-elf", image_name, str(torch_cuda)],
                        cwd=directory,
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=180,
                    )
                    cubins = list(Path(directory).glob("*.cubin"))
                    if len(cubins) == 1:
                        cubin = cubins[0]
                        disassembled = subprocess.run(
                            [nvdisasm, str(cubin)], check=True, capture_output=True, text=True, timeout=180
                        ).stdout
                        instruction_lines = re.findall(r"/\*[0-9a-fA-F]+\*/\s+([A-Z][A-Z0-9_.]*)", disassembled)
                        symbol_present = bool(
                            re.search(rf"(?<![A-Za-z0-9_$]){re.escape(name)}(?![A-Za-z0-9_$])", disassembled)
                        )
                        if symbol_present:
                            return {
                                "profiled_kernel_name": name,
                                "observed_launch_count": event_counts[name],
                                "selection": "Highest-launch-count observed PyTorch-native symbol matching the optional filter with extractable compatible SASS.",
                                "embedded_architecture": architecture,
                                "embedded_image": image_name,
                                "embedded_image_sha256": _file_sha256(cubin),
                                "profiled_symbol_present_in_image": True,
                                "image_instruction_count": len(instruction_lines),
                                "instruction_parse": "Heuristic address/opcode-line count across the complete extracted image, not a semantic SASS parse.",
                                "image_opcode_histogram": dict(sorted(Counter(instruction_lines).items())),
                                "nvdisasm_output_sha256": hashlib.sha256(disassembled.encode("utf-8")).hexdigest(),
                                "binding": "An observed launch symbol was found in a compatible extracted image; the driver-selected image for that launch is not attested.",
                            }
        failures.append({"kernel": name, "returncode": completed.returncode, "symbol_found": marker in output})
        if len(failures) == 8:
            break
    return {"profiled_kernel_name": None, "failures": failures, "binding": "No profiled function image was extracted."}


def build_cuda_provenance_manifest(
    model: Any, inputs: dict[str, Any], kernel_filter: str | None = None, redact: bool = False
) -> dict[str, Any]:
    _require_cuda()
    cuobjdump = shutil.which("cuobjdump")
    nvdisasm = shutil.which("nvdisasm")
    if cuobjdump is None or nvdisasm is None:
        raise RuntimeError("cuobjdump and nvdisasm are required")
    from .reference_gemma import model_state_sha256

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as execution_profile:
        with torch.no_grad():
            output = model(**inputs, use_cache=False, logits_to_keep=1)
    torch.cuda.synchronize()
    cuda_events = [event for event in execution_profile.events() if event.device_type == torch.autograd.DeviceType.CUDA]
    event_counts = Counter(event.name for event in cuda_events)
    kernel_names = sorted(name for name in event_counts if not name.startswith("Memcpy"))
    binaries, torch_cuda = _binary_inventory()
    embedded, image_listing = _embedded_images(cuobjdump, torch_cuda)
    supported = torch.cuda.get_arch_list()
    capability = torch.cuda.get_device_capability()
    exact_architecture = f"sm_{capability[0]}{capability[1]}"
    compatible_architecture = _compatible_architecture(supported, capability)
    disassembly = _disassemble_profiled_function(
        cuobjdump, nvdisasm, torch_cuda, event_counts, compatible_architecture, image_listing, kernel_filter
    )
    recorded_binaries = binaries if not redact else [
        {**binary, "sha256": "redacted"} for binary in binaries
    ]
    body = {
        "scope": "Observed CUDA launch names plus binary and compatible-image fingerprints; not instruction-level semantic verification.",
        "privacy": {"redacted": redact},
        "runtime": {
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "device_name": "redacted" if redact else torch.cuda.get_device_name(),
            "device_capability": list(capability),
            "supported_binary_architectures": supported,
            "exact_architecture_advertised_by_torch": exact_architecture in supported,
            "compatible_embedded_architecture_examined": compatible_architecture,
        },
        "tools": {
            "cuobjdump": _tool_version(cuobjdump),
            "nvdisasm": _tool_version(nvdisasm),
        },
        "execution_binding": {
            "model_class": type(model).__name__,
            "model_state_sha256": model_state_sha256(model),
            "attention_implementation": getattr(model.config, "_attn_implementation", None),
            "input_ids": "redacted" if redact else inputs["input_ids"].detach().cpu().tolist(),
            "input_ids_tensor": tensor_descriptor(inputs["input_ids"]),
            "attention_mask_tensor": tensor_descriptor(inputs["attention_mask"]),
            "output_logits_tensor": tensor_descriptor(output.logits),
            "selected_token_id": int(torch.argmax(output.logits[0, -1]).item()),
        },
        "profile": {
            "cuda_event_count": len(cuda_events),
            "unique_kernel_count": len(kernel_names),
            "kernel_launch_counts": {name: event_counts[name] for name in sorted(event_counts)},
        },
        "runtime_distribution_binaries": recorded_binaries,
        "torch_cuda_embedded_images": embedded,
        "profiled_symbol_disassembly": disassembly,
        "unresolved": [
            "Driver attestation of the exact cubin selected for every launch",
            "JIT-generated and vendor-library internal kernel binary extraction",
            "Formal SASS operational semantics",
            "Instruction-by-instruction proof against IEEE-754 and bfloat16 specifications",
        ],
    }
    body["manifest_sha256"] = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    return body


def summarize_cuda_provenance(manifest: dict[str, Any]) -> dict[str, Any]:
    counts = manifest["profile"]["kernel_launch_counts"]
    families: dict[str, dict[str, int]] = {}
    for name, launches in counts.items():
        if name.startswith("Memcpy"):
            family = "memory_copy"
        elif "fmha" in name.lower():
            family = "fused_attention"
        elif "cublas" in name.lower() or "cutlass" in name.lower() or "gemm" in name.lower():
            family = "vendor_linear_algebra"
        elif name.startswith("_ZN2at6native") or name.startswith("_ZN50_GLOBAL"):
            family = "pytorch_native"
        else:
            family = "other"
        entry = families.setdefault(family, {"unique_symbols": 0, "launches": 0})
        entry["unique_symbols"] += 1
        entry["launches"] += launches
    disassembly = manifest["profiled_symbol_disassembly"]
    return {
        "scope": manifest["scope"],
        "privacy": manifest["privacy"],
        "runtime": manifest["runtime"],
        "execution_binding": manifest["execution_binding"],
        "manifest_sha256": manifest["manifest_sha256"],
        "profile": {
            "cuda_event_count": manifest["profile"]["cuda_event_count"],
            "unique_kernel_count": manifest["profile"]["unique_kernel_count"],
            "families": dict(sorted(families.items())),
        },
        "runtime_distribution_binaries": manifest["runtime_distribution_binaries"],
        "torch_cuda_embedded_images": manifest["torch_cuda_embedded_images"],
        "profiled_symbol_disassembly": {
            "profiled_kernel_name": disassembly.get("profiled_kernel_name"),
            "observed_launch_count": disassembly.get("observed_launch_count"),
            "selection": disassembly.get("selection"),
            "embedded_architecture": disassembly.get("embedded_architecture"),
            "embedded_image": disassembly.get("embedded_image"),
            "embedded_image_sha256": disassembly.get("embedded_image_sha256"),
            "profiled_symbol_present_in_image": disassembly.get("profiled_symbol_present_in_image"),
            "image_instruction_count": disassembly.get("image_instruction_count"),
            "instruction_parse": disassembly.get("instruction_parse"),
            "unique_opcodes": len(disassembly.get("image_opcode_histogram", {})),
            "nvdisasm_output_sha256": disassembly.get("nvdisasm_output_sha256"),
            "binding": disassembly.get("binding"),
        },
        "unresolved": manifest["unresolved"],
    }


def verify_cuda_provenance_manifest(manifest: dict[str, Any], verify_local_binaries: bool = False) -> dict[str, Any]:
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    hash_matches = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest() == manifest.get("manifest_sha256")
    binaries_match = None
    if verify_local_binaries:
        _require_cuda()
        current, _ = _binary_inventory()
        expected = {item["name"]: item for item in manifest.get("runtime_distribution_binaries", [])}
        binaries_match = all(expected.get(item["name"]) == item for item in current) and len(expected) == len(current)
    return {
        "valid": bool(hash_matches and binaries_match is not False),
        "manifest_hash_matches": hash_matches,
        "local_binaries_match": binaries_match,
    }
