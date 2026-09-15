from __future__ import annotations

import copy
import hashlib
import inspect
from pathlib import Path
from typing import Any

import numpy as np

from .gemma_attention_entry import _bits
from .gemma_reduction_backend import _profile_call
from .gemma_rsqrt_lookup import _runtime
from .gemma_rotary_slice import _sha, _seal, _check_hash
from .serialization import canonical_json

SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
HALF_DOMAIN = 0x7F80
ENTRIES = 2 * HALF_DOMAIN
TABLE_BYTES = 2 * ENTRIES
MODEL_SHAPE = (1, 30, 2048)
WINDOW_COUNT = 61440
WINDOW_STARTS = (0, ENTRIES - WINDOW_COUNT)
SCOPE = "Explicit empirical finite-BF16 GELU-tanh mapping for a pinned CUDA runtime; complete input encodings and declared contiguous layouts, not reconstruction of native FP32/tanh arithmetic."


def _code_sha() -> str:
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("GELU specification source changed after import")
    return SOURCE_SHA256


def finite_inputs() -> np.ndarray:
    return np.concatenate((np.arange(HALF_DOMAIN, dtype=np.uint16), np.arange(0x8000, 0x8000 + HALF_DOMAIN, dtype=np.uint16)))


def finite_index(bits: int) -> int:
    if type(bits) is not int or not 0 <= bits <= 0xFFFF or bits & 0x7F80 == 0x7F80:
        raise ValueError("GELU lookup accepts finite bfloat16 encodings only")
    return bits - 0x80 if bits & 0x8000 else bits


def _cuda_gelu(bits: np.ndarray, shape: tuple[int, ...]) -> tuple[np.ndarray, list[str]]:
    import torch
    values = torch.from_numpy(bits).view(torch.bfloat16).to("cuda").reshape(shape)
    if values.data_ptr() % 16 or not values.is_contiguous():
        raise ValueError("GELU acquisition requires aligned contiguous BF16 inputs")
    with torch.no_grad():
        output, names = _profile_call(lambda: torch.nn.functional.gelu(values, approximate="tanh"))
    if output.dtype != torch.bfloat16 or tuple(output.shape) != shape:
        raise ValueError("Native GELU output type mismatch")
    return _bits(output).reshape(-1).astype("<u2"), names


def _audit(directory: Path, values: np.ndarray) -> str:
    payload = values.astype("<u2", copy=False).tobytes()
    digest = hashlib.sha256(payload).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (digest + ".bin")
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError("Existing GELU mismatch payload is inconsistent")
    else:
        with path.open("xb") as stream:
            stream.write(payload)
    return digest


def build_gelu_plan(source_summary: dict[str, Any]) -> dict[str, Any]:
    from transformers.activations import PytorchGELUTanh
    _check_hash(source_summary, "summary_sha256")
    runtime = _runtime()
    if source_summary.get("mlp_entry_matches") is not True or canonical_json(source_summary["runtime"]) != canonical_json(runtime):
        raise ValueError("GELU specification requires passing MLP-entry evidence in this runtime")
    return _seal({"schema_version": 1, "scope": SCOPE, "source_summary_sha256": source_summary["summary_sha256"], "source_report_sha256": source_summary["source_report_sha256"],
                  "runtime": runtime, "code_sha256": _code_sha(), "entry_count": ENTRIES, "table_bytes": TABLE_BYTES,
                  "input_bits_sha256": hashlib.sha256(finite_inputs().astype("<u2").tobytes()).hexdigest(),
                  "format": "little-endian uint16 output encodings; positive finite then negative finite input encodings",
                  "operation": "torch.nn.functional.gelu", "approximate": "tanh", "activation_class": "transformers.activations.PytorchGELUTanh",
                  "activation_source_sha256": hashlib.sha256(inspect.getsource(PytorchGELUTanh.forward).encode("utf-8")).hexdigest(),
                  "repetitions": 3, "acquisition_schedule": "one warm-up and one profiled call per observation; first vector observation defines the table",
                  "layout_shape": list(MODEL_SHAPE), "layout_window_starts": list(WINDOW_STARTS), "layout_window_count": WINDOW_COUNT,
                  "layout_prediction_rule": "frozen table slice for the declared finite-input indices",
                  "finite_outputs_required": True, "nonfinite_inputs_supported": False, "native_arithmetic_reconstructed": False,
                  "unrestricted_layouts_qualified": False, "global_exactness_activation_allowed": False}, "plan_sha256")


def check_gelu_plan(plan: dict[str, Any]) -> None:
    from transformers.activations import PytorchGELUTanh
    _check_hash(plan, "plan_sha256")
    expected = {"schema_version": 1, "scope": SCOPE, "code_sha256": _code_sha(), "entry_count": ENTRIES, "table_bytes": TABLE_BYTES,
                "input_bits_sha256": hashlib.sha256(finite_inputs().astype("<u2").tobytes()).hexdigest(),
                "format": "little-endian uint16 output encodings; positive finite then negative finite input encodings",
                "operation": "torch.nn.functional.gelu", "approximate": "tanh", "activation_class": "transformers.activations.PytorchGELUTanh",
                "activation_source_sha256": hashlib.sha256(inspect.getsource(PytorchGELUTanh.forward).encode("utf-8")).hexdigest(),
                "repetitions": 3, "acquisition_schedule": "one warm-up and one profiled call per observation; first vector observation defines the table",
                "layout_shape": list(MODEL_SHAPE), "layout_window_starts": list(WINDOW_STARTS), "layout_window_count": WINDOW_COUNT,
                "layout_prediction_rule": "frozen table slice for the declared finite-input indices",
                "finite_outputs_required": True, "nonfinite_inputs_supported": False, "native_arithmetic_reconstructed": False,
                "unrestricted_layouts_qualified": False, "global_exactness_activation_allowed": False}
    if any(canonical_json(plan.get(key)) != canonical_json(value) for key, value in expected.items()) or not isinstance(plan.get("runtime"), dict):
        raise ValueError("GELU specification domain/source/profile mismatch")


def _observation(repetition: int, actual: np.ndarray, expected: np.ndarray, names: list[str], audit_dir: Path) -> dict[str, Any]:
    mismatch = int(np.count_nonzero(actual != expected))
    finite = bool(np.all(actual & 0x7F80 != 0x7F80))
    return {"repetition": repetition, "output_sha256": hashlib.sha256(actual.tobytes()).hexdigest(), "mismatch_count": mismatch,
            "finite_outputs": finite, "kernel_names": names, "mismatch_payload_sha256": _audit(audit_dir, actual) if mismatch or not finite else None}


def _manifest(plan: dict[str, Any], records: list[dict[str, Any]], table_sha: str, runtime: dict[str, Any]) -> dict[str, Any]:
    if len(records) != 3:
        raise ValueError("GELU base repetition coverage mismatch")
    kernel_match = all(item["kernel_names"] == records[0]["kernel_names"] and item["kernel_names"] and all(isinstance(name, str) and "GeluCUDAKernelImpl" in name for name in item["kernel_names"]) for item in records)
    stable = all(item["mismatch_count"] == 0 and item["finite_outputs"] is True and item["output_sha256"] == table_sha for item in records)
    matching_runtime = canonical_json(runtime) == canonical_json(plan["runtime"])
    return _seal({"schema_version": 1, "scope": SCOPE, "plan_sha256": plan["plan_sha256"], "table_sha256": table_sha, "table_bytes": TABLE_BYTES,
                  "records": records, "runtime": runtime, "kernel_names_match": kernel_match, "base_repetitions_identical": stable,
                  "runtime_matches_plan": matching_runtime, "base_table_usable": stable and kernel_match and matching_runtime,
                  "layout_validation_required": True, "native_arithmetic_reconstructed": False, "global_exactness_activation_allowed": False}, "manifest_sha256")


def acquire_gelu_table(plan: dict[str, Any], table_path: Path, audit_dir: Path) -> dict[str, Any]:
    check_gelu_plan(plan)
    if canonical_json(_runtime()) != canonical_json(plan["runtime"]):
        raise ValueError("GELU acquisition runtime differs from the plan")
    inputs, first, records = finite_inputs(), None, []
    with table_path.open("xb") as stream:
        for repetition in range(3):
            actual, names = _cuda_gelu(inputs, (ENTRIES,))
            if first is None:
                first = actual.copy()
                stream.write(first.tobytes())
            records.append(_observation(repetition, actual, first, names, audit_dir))
    _code_sha()
    return _manifest(plan, records, hashlib.sha256(first.tobytes()).hexdigest(), _runtime())


def _check_records(records: list[dict[str, Any]], expected: np.ndarray, audit_dir: Path) -> None:
    if len(records) != 3:
        raise ValueError("GELU observation repetitions incomplete")
    digest = hashlib.sha256(expected.tobytes()).hexdigest()
    for repetition, record in enumerate(records):
        if type(record["repetition"]) is not int or record["repetition"] != repetition or type(record["mismatch_count"]) is not int or not 0 <= record["mismatch_count"] <= expected.size:
            raise ValueError("Invalid GELU observation count")
        if record["mismatch_payload_sha256"] is not None:
            name = record["mismatch_payload_sha256"]
            if not isinstance(name, str) or len(name) != 64 or any(char not in "0123456789abcdef" for char in name):
                raise ValueError("Invalid GELU mismatch commitment")
            payload = (audit_dir / (name + ".bin")).read_bytes()
            if len(payload) != expected.nbytes or hashlib.sha256(payload).hexdigest() != name or record["output_sha256"] != name:
                raise ValueError("GELU audit payload mismatch")
            actual = np.frombuffer(payload, dtype="<u2")
            if record["mismatch_count"] != int(np.count_nonzero(actual != expected)) or record["finite_outputs"] is not bool(np.all(actual & 0x7F80 != 0x7F80)):
                raise ValueError("GELU mismatch accounting is inconsistent")
        elif record["output_sha256"] != digest or record["mismatch_count"] != 0 or record["finite_outputs"] is not True or not bool(np.all(expected & 0x7F80 != 0x7F80)):
            raise ValueError("Missing GELU mismatch evidence")


def load_gelu_table(plan: dict[str, Any], manifest: dict[str, Any], table_path: Path, audit_dir: Path) -> np.ndarray:
    check_gelu_plan(plan)
    _check_hash(manifest, "manifest_sha256")
    raw = table_path.read_bytes()
    if len(raw) != TABLE_BYTES or hashlib.sha256(raw).hexdigest() != manifest["table_sha256"] or manifest["plan_sha256"] != plan["plan_sha256"]:
        raise ValueError("GELU table bytes or source commitment mismatch")
    expected = _manifest(plan, manifest["records"], manifest["table_sha256"], manifest["runtime"])
    if canonical_json(expected) != canonical_json(manifest):
        raise ValueError("GELU manifest flags/scope are inconsistent")
    table = np.frombuffer(raw, dtype="<u2")
    _check_records(manifest["records"], table, audit_dir)
    return table


def _layout_report(plan: dict[str, Any], manifest: dict[str, Any], vector: list[dict[str, Any]], windows: list[dict[str, Any]], runtime: dict[str, Any]) -> dict[str, Any]:
    records = [*vector, *(item for window in windows for item in window["observations"])]
    stable_kernels = all(record["kernel_names"] == manifest["records"][0]["kernel_names"] for record in vector) and all(all(item["kernel_names"] == window["observations"][0]["kernel_names"] for item in window["observations"]) for window in windows)
    usable = stable_kernels and all(record["mismatch_count"] == 0 and record["finite_outputs"] is True and isinstance(record["kernel_names"], list) and record["kernel_names"] and all(isinstance(name, str) and "GeluCUDAKernelImpl" in name for name in record["kernel_names"]) for record in records)
    return _seal({"schema_version": 1, "scope": SCOPE, "plan_sha256": plan["plan_sha256"], "manifest_sha256": manifest["manifest_sha256"],
                  "table_sha256": manifest["table_sha256"], "vector_replay": vector, "windows": windows, "runtime": runtime,
                  "unique_finite_inputs_covered": ENTRIES, "overlapping_windows": True,
                  "declared_layouts_match": usable and canonical_json(runtime) == canonical_json(plan["runtime"]),
                  "native_arithmetic_reconstructed": False, "unrestricted_layouts_qualified": False, "global_exactness_activation_allowed": False}, "report_sha256")


def acquire_gelu_layouts(plan: dict[str, Any], manifest: dict[str, Any], table_path: Path, audit_dir: Path) -> dict[str, Any]:
    table = load_gelu_table(plan, manifest, table_path, audit_dir)
    if manifest["base_table_usable"] is not True or canonical_json(_runtime()) != canonical_json(plan["runtime"]):
        raise ValueError("GELU layout validation requires a passing base in its runtime")
    inputs, vector, windows = finite_inputs(), [], []
    for repetition in range(3):
        actual, names = _cuda_gelu(inputs, (ENTRIES,))
        vector.append(_observation(repetition, actual, table, names, audit_dir))
    for start in WINDOW_STARTS:
        expected = table[start:start + WINDOW_COUNT]
        record = {"input_index_start": start, "count": WINDOW_COUNT, "shape": list(MODEL_SHAPE), "expected_output_sha256": hashlib.sha256(expected.tobytes()).hexdigest(), "observations": []}
        for repetition in range(3):
            actual, names = _cuda_gelu(inputs[start:start + WINDOW_COUNT], MODEL_SHAPE)
            record["observations"].append(_observation(repetition, actual, expected, names, audit_dir))
        windows.append(record)
    _code_sha()
    return _layout_report(plan, manifest, vector, windows, _runtime())


def check_gelu_layouts(plan: dict[str, Any], manifest: dict[str, Any], table: np.ndarray, report: dict[str, Any], audit_dir: Path) -> None:
    _check_hash(report, "report_sha256")
    if len(report["windows"]) != len(WINDOW_STARTS):
        raise ValueError("GELU layout coverage incomplete")
    _check_records(report["vector_replay"], table, audit_dir)
    for start, record in zip(WINDOW_STARTS, report["windows"]):
        expected = table[start:start + WINDOW_COUNT]
        if type(record["input_index_start"]) is not int or record["input_index_start"] != start or type(record["count"]) is not int or record["count"] != WINDOW_COUNT or record["shape"] != list(MODEL_SHAPE) or record["expected_output_sha256"] != hashlib.sha256(expected.tobytes()).hexdigest():
            raise ValueError("GELU layout input/prediction binding mismatch")
        _check_records(record["observations"], expected, audit_dir)
    expected_report = _layout_report(plan, manifest, report["vector_replay"], report["windows"], report["runtime"])
    if canonical_json(expected_report) != canonical_json(report):
        raise ValueError("GELU layout report accounting mismatch")


class CheckedGeluLookup:
    def __init__(self, plan: dict[str, Any], manifest: dict[str, Any], table_path: Path, layout_report: dict[str, Any], audit_dir: Path):
        self._table = load_gelu_table(plan, manifest, table_path, audit_dir)
        check_gelu_layouts(plan, manifest, self._table, layout_report, audit_dir)
        if manifest["base_table_usable"] is not True or layout_report["declared_layouts_match"] is not True:
            raise ValueError("GELU finite-domain/layout evidence did not pass")
        self._runtime = copy.deepcopy(plan["runtime"])
        self._evidence = {"table_plan_sha256": plan["plan_sha256"], "table_manifest_sha256": manifest["manifest_sha256"], "table_sha256": manifest["table_sha256"], "layout_report_sha256": layout_report["report_sha256"]}
        self._activation_source_sha256 = plan["activation_source_sha256"]

    @property
    def evidence(self) -> dict[str, Any]:
        return copy.deepcopy(self._evidence)

    @property
    def activation_source_sha256(self) -> str:
        return self._activation_source_sha256

    def predict_bits(self, bits: int, runtime: dict[str, Any]) -> int:
        index = finite_index(bits)
        if canonical_json(runtime) != canonical_json(self._runtime):
            raise ValueError("GELU target runtime differs from validated evidence")
        return int(self._table[index])
