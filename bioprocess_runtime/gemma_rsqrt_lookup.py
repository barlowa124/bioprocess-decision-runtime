from __future__ import annotations

import copy
import hashlib
import inspect
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .serialization import canonical_json

MANTISSAS = 1 << 23
BASE_START = 0x3F800000
ENTRIES = 2 * MANTISSAS
CHUNK = 1 << 20
TABLE_BYTES = ENTRIES * 4
SCOPE = "Explicit empirical float32 rsqrt lookup/scaling specification for a pinned runtime; normalized table data are observed, not a reconstruction of NVIDIA arithmetic."


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _seal(body: dict[str, Any], field: str) -> dict[str, Any]:
    return {**body, field: _sha(body)}


def _check_hash(value: Any, field: str) -> None:
    if not isinstance(value, dict) or _sha({key: item for key, item in value.items() if key != field}) != value.get(field):
        raise ValueError(f"Lookup {field} commitment mismatch")


def _positive_normal(bits: int) -> int:
    if type(bits) is not int or bits & 0x80000000 or not 0x00800000 <= bits < 0x7F800000:
        raise ValueError("Lookup accepts positive normal float32 encodings only")
    return (bits >> 23) & 255


def _raw_lookup(bits: int, table: np.ndarray) -> int:
    exponent = _positive_normal(bits) - 127
    index = ((exponent & 1) << 23) | (bits & 0x7FFFFF)
    result = int(table[index]) - (exponent // 2) * MANTISSAS
    _positive_normal(result)
    return result


def _expected_chunk(table: np.ndarray, exponent_field: int, mantissa_start: int) -> np.ndarray:
    exponent = exponent_field - 127
    start = (exponent & 1) * MANTISSAS + mantissa_start
    return (table[start:start + CHUNK].astype(np.int64) - (exponent // 2) * MANTISSAS).astype("<u4")


def _code_sha() -> str:
    return _sha({fn.__name__: inspect.getsource(fn) for fn in (_positive_normal, _raw_lookup, _expected_chunk)})


def _runtime() -> dict[str, Any]:
    import subprocess
    import torch
    from .gemma_ir_interpreter import _projection_environment

    runtime = _projection_environment()
    result = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], check=True, capture_output=True, text=True, timeout=10)
    drivers = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    if len(drivers) != 1:
        raise ValueError("Could not bind one NVIDIA driver version")
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    runtime.update({"driver_version": next(iter(drivers)), "gpu_uuid_sha256": hashlib.sha256(str(properties.uuid).encode("utf-8")).hexdigest(),
                    "torch_git_version": torch.version.git_version})
    return runtime


def _cuda_values(start: int, count: int) -> tuple[np.ndarray, list[str]]:
    import torch
    from .gemma_reduction_backend import _profile_call

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for native rsqrt observation")
    inputs = np.arange(count, dtype=np.uint32) + np.uint32(start)
    value = torch.from_numpy(inputs.astype(np.int32)).to("cuda").view(torch.float32)
    if value.data_ptr() % 16 or value.stride() != (1,):
        raise ValueError("Lookup acquisition requires aligned contiguous float32 input")
    with torch.no_grad():
        output, names = _profile_call(lambda: value.rsqrt())
    if output.dtype != torch.float32 or list(output.shape) != [count]:
        raise ValueError("Native rsqrt output type mismatch")
    return output.cpu().view(torch.int32).numpy().astype("<u4"), names


def build_table_plan(rms_summary: dict[str, Any]) -> dict[str, Any]:
    _check_hash(rms_summary, "summary_sha256")
    names = {name for events in rms_summary["profiled_cuda_event_names"].values() for repetition in events for name in repetition if "rsqrt_kernel_cuda" in name}
    if len(names) != 1 or not rms_summary.get("actual_inputs_match_plan"):
        raise ValueError("Expected one observed native rsqrt kernel and bound RMS inputs")
    runtime = _runtime()
    if any(canonical_json(runtime.get(key)) != canonical_json(value) for key, value in rms_summary["runtime"].items()):
        raise ValueError("Current runtime differs from the RMS source observations")
    return _seal({"schema_version": 1, "scope": SCOPE, "source_rms_summary_sha256": rms_summary["summary_sha256"],
                  "source_rms_report_sha256": rms_summary["source_report_sha256"],
                  "normalized_input_start": BASE_START, "entry_count": ENTRIES, "table_bytes": TABLE_BYTES,
                  "chunk_size": CHUNK, "repetitions": 3, "format": "little-endian uint32 float32 output encodings",
                  "base_exponent_fields": [127, 128], "expected_kernel_names": sorted(names),
                  "runtime": runtime, "lookup_code_sha256": _code_sha(),
                  "outside_base_domain_enabled": False, "native_arithmetic_reconstructed": False,
                  "hardware_semantics_established": False, "global_exactness_activation_allowed": False}, "plan_sha256")


def _check_table_plan(plan: dict[str, Any]) -> None:
    _check_hash(plan, "plan_sha256")
    expected = {"schema_version": 1, "scope": SCOPE, "normalized_input_start": BASE_START, "entry_count": ENTRIES,
                "table_bytes": TABLE_BYTES, "chunk_size": CHUNK, "repetitions": 3,
                "format": "little-endian uint32 float32 output encodings", "base_exponent_fields": [127, 128],
                "lookup_code_sha256": _code_sha(), "outside_base_domain_enabled": False,
                "native_arithmetic_reconstructed": False, "hardware_semantics_established": False,
                "global_exactness_activation_allowed": False}
    if any(canonical_json(plan.get(key)) != canonical_json(value) for key, value in expected.items()):
        raise ValueError("Unsupported lookup table specification")
    if not isinstance(plan.get("runtime"), dict) or not isinstance(plan.get("expected_kernel_names"), list) or not plan["expected_kernel_names"]:
        raise ValueError("Missing lookup runtime or kernel binding")


def _audit_payload(directory: Path, values: np.ndarray) -> str:
    payload = values.astype("<u4", copy=False).tobytes()
    digest = hashlib.sha256(payload).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (digest + ".bin")
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Existing mismatch payload is inconsistent")
    else:
        with path.open("xb") as stream:
            stream.write(payload)
    return digest


def acquire_table(plan: dict[str, Any], table_path: Path, audit_dir: Path) -> dict[str, Any]:
    _check_table_plan(plan)
    runtime = _runtime()
    if canonical_json(runtime) != canonical_json(plan["runtime"]):
        raise ValueError("Lookup runtime differs from the frozen table plan")
    records, whole = [], hashlib.sha256()
    with table_path.open("xb") as stream:
        for start in range(0, ENTRIES, CHUNK):
            first = None
            observations = []
            for repetition in range(3):
                values, kernels = _cuda_values(BASE_START + start, CHUNK)
                payload = values.tobytes()
                digest = hashlib.sha256(payload).hexdigest()
                if first is None:
                    first = values
                    stream.write(payload)
                    whole.update(payload)
                mismatch_count = int(np.count_nonzero(first != values))
                audit = _audit_payload(audit_dir, values) if mismatch_count else None
                observations.append({"repetition": repetition, "output_sha256": digest, "kernel_names": kernels,
                                     "mismatch_count_against_first": mismatch_count, "mismatch_payload_sha256": audit})
            exponents = (first >> 23) & 255
            records.append({"offset": start, "count": CHUNK, "observations": observations,
                            "output_range_valid": bool(np.all((first & 0x80000000) == 0) and np.all((exponents >= 125) & (exponents <= 127)))})
    stable = all(observation["mismatch_count_against_first"] == 0 for record in records for observation in record["observations"])
    kernels_match = all(observation["kernel_names"] == plan["expected_kernel_names"] for record in records for observation in record["observations"])
    return _seal({"schema_version": 1, "scope": SCOPE, "plan_sha256": plan["plan_sha256"], "table_sha256": whole.hexdigest(),
                  "table_bytes": TABLE_BYTES, "records": records, "runtime": runtime,
                  "base_domain_stable": stable, "source_kernel_names_match": kernels_match,
                  "base_domain_usable": stable and kernels_match and all(record["output_range_valid"] for record in records),
                  "native_arithmetic_reconstructed": False, "global_exactness_activation_allowed": False}, "manifest_sha256")


def load_table(plan: dict[str, Any], manifest: dict[str, Any], table_path: Path) -> np.ndarray:
    _check_table_plan(plan)
    _check_hash(manifest, "manifest_sha256")
    raw = table_path.read_bytes()
    if manifest["plan_sha256"] != plan["plan_sha256"] or len(raw) != TABLE_BYTES or hashlib.sha256(raw).hexdigest() != manifest["table_sha256"]:
        raise ValueError("Lookup table bytes or source plan mismatch")
    if canonical_json(manifest["runtime"]) != canonical_json(plan["runtime"]) or any(manifest.get(key) is not True for key in ("base_domain_usable", "base_domain_stable", "source_kernel_names_match")) or manifest.get("table_bytes") != TABLE_BYTES or manifest.get("scope") != SCOPE or any(manifest.get(key) is not False for key in ("native_arithmetic_reconstructed", "global_exactness_activation_allowed")):
        raise ValueError("Lookup base domain is not usable in the declared runtime")
    table = np.frombuffer(raw, dtype="<u4")
    if len(manifest["records"]) != ENTRIES // CHUNK:
        raise ValueError("Lookup base coverage is incomplete")
    for index, record in enumerate(manifest["records"]):
        if record["offset"] != index * CHUNK or record["count"] != CHUNK or record["output_range_valid"] is not True or len(record["observations"]) != 3:
            raise ValueError("Lookup base coverage mismatch")
        values = table[index * CHUNK:(index + 1) * CHUNK]
        exponents = (values >> 23) & 255
        if not np.all((values & 0x80000000) == 0) or not np.all((exponents >= 125) & (exponents <= 127)):
            raise ValueError("Lookup contains unsupported output encodings")
        digest = hashlib.sha256(values.tobytes()).hexdigest()
        for repetition, observation in enumerate(record["observations"]):
            if canonical_json(observation) != canonical_json({"repetition": repetition, "output_sha256": digest, "kernel_names": plan["expected_kernel_names"], "mismatch_count_against_first": 0, "mismatch_payload_sha256": None}):
                raise ValueError("Lookup base repetitions or kernel evidence disagree")
    return table


def build_domain_plan(table_plan: dict[str, Any], manifest: dict[str, Any], table_path: Path) -> dict[str, Any]:
    table = load_table(table_plan, manifest, table_path)
    chunks = []
    for exponent in range(1, 255):
        for start in range(0, MANTISSAS, CHUNK):
            expected = _expected_chunk(table, exponent, start)
            chunks.append({"exponent_field": exponent, "mantissa_start": start, "input_start_bits": (exponent << 23) | start,
                           "count": CHUNK, "predicted_sha256": hashlib.sha256(expected.tobytes()).hexdigest()})
    return _seal({"schema_version": 1, "scope": SCOPE, "table_plan_sha256": table_plan["plan_sha256"],
                  "table_manifest_sha256": manifest["manifest_sha256"], "table_sha256": manifest["table_sha256"],
                  "lookup_code_sha256": _code_sha(), "mapping": "parity=(E-127)&1; index=parity*2^23+fraction; output_bits=table[index]-floor((E-127)/2)*2^23",
                  "input_domain": "all positive finite normal float32 encodings, exponent fields 1 through 254",
                  "tested_value_count": 254 * MANTISSAS, "base_domain_value_count": ENTRIES,
                  "nonbase_value_count": 252 * MANTISSAS, "chunk_size": CHUNK, "launch_shape": [CHUNK], "repetitions": 1,
                  "chunks": chunks, "runtime": copy.deepcopy(table_plan["runtime"]), "expected_kernel_names": table_plan["expected_kernel_names"],
                  "predictions_sha256": _sha(chunks), "native_arithmetic_reconstructed": False,
                  "hardware_semantics_established": False, "global_exactness_activation_allowed": False}, "plan_sha256")


def _domain_report(plan: dict[str, Any], records: list[dict[str, Any]], runtime: dict[str, Any]) -> dict[str, Any]:
    if len(records) != len(plan["chunks"]):
        raise ValueError("Incomplete rsqrt domain traversal")
    coverage = []
    for exponent in range(1, 255):
        selected = records[(exponent - 1) * (MANTISSAS // CHUNK):exponent * (MANTISSAS // CHUNK)]
        coverage.append({"exponent_field": exponent, "tested_values": sum(record["count"] for record in selected),
                         "mismatch_count": sum(record["mismatch_count"] for record in selected)})
    scope_match = canonical_json(runtime) == canonical_json(plan["runtime"]) and all(record["kernel_names"] == plan["expected_kernel_names"] for record in records)
    covered = [item["exponent_field"] for item in coverage if item["tested_values"] == MANTISSAS and item["mismatch_count"] == 0] if scope_match else []
    return _seal({"schema_version": 1, "scope": SCOPE, "plan_sha256": plan["plan_sha256"], "table_sha256": plan["table_sha256"],
                  "records": records, "runtime": runtime, "exponent_coverage": coverage, "covered_exponents": covered,
                  "mismatch_count": sum(item["mismatch_count"] for item in coverage), "tested_value_count": sum(item["tested_values"] for item in coverage),
                  "source_kernel_environment_match": scope_match, "all_positive_normal_values_match": covered == list(range(1, 255)),
                  "validated_launch_shape": plan["launch_shape"], "native_arithmetic_reconstructed": False,
                  "hardware_semantics_established": False, "global_exactness_activation_allowed": False}, "report_sha256")


def acquire_domain(table_plan: dict[str, Any], manifest: dict[str, Any], table_path: Path, plan: dict[str, Any], audit_dir: Path, persist: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    if canonical_json(plan) != canonical_json(build_domain_plan(table_plan, manifest, table_path)):
        raise ValueError("Frozen rsqrt domain plan mismatch")
    table = load_table(table_plan, manifest, table_path)
    runtime = _runtime()
    if canonical_json(runtime) != canonical_json(plan["runtime"]):
        raise ValueError("Rsqrt domain runtime mismatch")
    records = []
    for index, chunk in enumerate(plan["chunks"]):
        expected = _expected_chunk(table, chunk["exponent_field"], chunk["mantissa_start"])
        observed, kernels = _cuda_values(chunk["input_start_bits"], chunk["count"])
        mismatch_count = int(np.count_nonzero(expected != observed))
        audit = _audit_payload(audit_dir, observed) if mismatch_count else None
        records.append({**chunk, "observed_sha256": hashlib.sha256(observed.tobytes()).hexdigest(), "kernel_names": kernels,
                        "mismatch_count": mismatch_count, "mismatch_payload_sha256": audit})
        if persist is not None and (mismatch_count or (index + 1) % (MANTISSAS // CHUNK) == 0 or index + 1 == len(plan["chunks"])):
            persist({"plan_sha256": plan["plan_sha256"], "completed_chunks": len(records), "records": records})
    return _domain_report(plan, records, runtime)


def verify_domain(table_plan: dict[str, Any], manifest: dict[str, Any], table_path: Path, plan: dict[str, Any], report: dict[str, Any], audit_dir: Path, reexecute: bool = False) -> dict[str, Any]:
    try:
        if canonical_json(plan) != canonical_json(build_domain_plan(table_plan, manifest, table_path)):
            raise ValueError("Rsqrt domain prediction plan differs from the specification")
        _check_hash(report, "report_sha256")
        table = load_table(table_plan, manifest, table_path)
        if len(report["records"]) != len(plan["chunks"]):
            raise ValueError("Missing rsqrt validation chunks")
        for chunk, record in zip(plan["chunks"], report["records"]):
            if canonical_json({key: record.get(key) for key in chunk}) != canonical_json(chunk) or type(record.get("mismatch_count")) is not int or not 0 <= record["mismatch_count"] <= CHUNK:
                raise ValueError("Rsqrt validation record order or count mismatch")
            if not isinstance(record.get("kernel_names"), list) or not record["kernel_names"] or any(not isinstance(name, str) or not name for name in record["kernel_names"]):
                raise ValueError("Missing rsqrt kernel observations")
            if record["mismatch_count"]:
                digest = record.get("mismatch_payload_sha256")
                if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                    raise ValueError("Malformed rsqrt mismatch payload reference")
                raw = (audit_dir / (digest + ".bin")).read_bytes()
                if len(raw) != CHUNK * 4 or hashlib.sha256(raw).hexdigest() != digest or record["observed_sha256"] != digest:
                    raise ValueError("Missing or damaged rsqrt mismatch payload")
                expected = _expected_chunk(table, chunk["exponent_field"], chunk["mantissa_start"])
                if int(np.count_nonzero(np.frombuffer(raw, dtype="<u4") != expected)) != record["mismatch_count"]:
                    raise ValueError("Rsqrt mismatch aggregate is incorrect")
            elif record["observed_sha256"] != chunk["predicted_sha256"] or record["mismatch_payload_sha256"] is not None:
                raise ValueError("Rsqrt matching chunk has inconsistent commitments")
        expected_report = _domain_report(plan, report["records"], report["runtime"])
        integrity = canonical_json(expected_report) == canonical_json(report)
        replay = canonical_json(acquire_domain(table_plan, manifest, table_path, plan, audit_dir)) == canonical_json(report) if reexecute and integrity else None
        return {"valid": integrity and (not reexecute or replay is True), "mode": "cuda_replay" if reexecute else "integrity_only",
                "reexecution_exact": replay, "tested_value_count": expected_report["tested_value_count"], "mismatch_count": expected_report["mismatch_count"],
                "all_positive_normal_values_match": expected_report["all_positive_normal_values_match"], "covered_exponents": expected_report["covered_exponents"],
                "global_exactness_activation_allowed": False}
    except (KeyError, TypeError, ValueError, RuntimeError, OSError, AttributeError) as error:
        return {"valid": False, "reason": str(error)}


class CheckedRsqrtLookup:
    def __init__(self, table_plan: dict[str, Any], manifest: dict[str, Any], table_path: Path, domain_plan: dict[str, Any] | None = None, domain_report: dict[str, Any] | None = None, audit_dir: Path | None = None):
        self._table = load_table(table_plan, manifest, table_path)
        self._runtime = copy.deepcopy(table_plan["runtime"])
        self._covered = frozenset((127, 128))
        self._table_sha256 = manifest["table_sha256"]
        self._evidence = {"table_plan_sha256": table_plan["plan_sha256"], "table_manifest_sha256": manifest["manifest_sha256"],
                         "table_sha256": manifest["table_sha256"], "domain_plan_sha256": domain_plan["plan_sha256"] if domain_plan else None,
                         "domain_report_sha256": domain_report["report_sha256"] if domain_report else None}
        self._kernel_names = tuple(table_plan["expected_kernel_names"])
        if (domain_plan is None) != (domain_report is None):
            raise ValueError("Domain plan and report must be supplied together")
        if domain_plan is not None:
            checked = verify_domain(table_plan, manifest, table_path, domain_plan, domain_report, audit_dir or table_path.parent)
            if not checked["valid"]:
                raise ValueError("Lookup domain evidence is invalid")
            self._covered = frozenset(checked["covered_exponents"])

    @property
    def table_sha256(self) -> str:
        return self._table_sha256

    @property
    def evidence(self) -> dict[str, Any]:
        return copy.deepcopy(self._evidence)

    @property
    def kernel_names(self) -> tuple[str, ...]:
        return self._kernel_names

    def predict_bits(self, bits: int, target_runtime: dict[str, Any]) -> int:
        exponent = _positive_normal(bits)
        if exponent not in self._covered:
            raise ValueError("Rsqrt exponent has not passed complete domain validation")
        if canonical_json(target_runtime) != canonical_json(self._runtime):
            raise ValueError("Rsqrt lookup target runtime differs from its evidence")
        return _raw_lookup(bits, self._table)


def _rms_lookup_row(inputs: list[int], weights: list[int], epsilon: float, lookup: CheckedRsqrtLookup, runtime: dict[str, Any]) -> dict[str, Any]:
    from fractions import Fraction
    from .reference_gemma import _rms_bfloat_to_float, _rms_f32_add, _rms_f32_mul, _rms_f32_round, rms_sum_bits
    from .gemma_reduction_semantics import decode_finite_float32
    from .gemma_float_semantics import encode_bfloat16_rne

    if len(inputs) != len(weights) or len(inputs) not in (256, 640):
        raise ValueError("Unsupported lookup RMS width")
    values = [_rms_bfloat_to_float(bits) for bits in inputs]
    squares = [_rms_f32_mul(bits, bits) for bits in values]
    mean = _rms_f32_mul(rms_sum_bits(squares, "source_vec4_warp32"), _rms_f32_round(Fraction(1, len(inputs))))
    denominator = _rms_f32_add(mean, _rms_f32_round(Fraction.from_float(float(epsilon))))
    inverse = lookup.predict_bits(denominator, runtime)
    outputs = []
    for value, weight in zip(values, weights):
        result = _rms_f32_mul(_rms_f32_mul(value, inverse), _rms_f32_add(0x3F800000, _rms_bfloat_to_float(weight)))
        exact, negative_zero = decode_finite_float32(result)
        outputs.append(encode_bfloat16_rne(exact, negative_zero))
    return {"mean_bits": mean, "denominator_bits": denominator, "rsqrt_bits": inverse, "output_bits": outputs}


def build_rms_lookup_plan(program: dict[str, Any], source_plan: dict[str, Any], source_bundle: dict[str, Any], source_report: dict[str, Any], lookup: CheckedRsqrtLookup) -> tuple[dict[str, Any], dict[str, Any]]:
    from .reference_gemma import verify_rms_slice, _rms_code_commitment

    if not verify_rms_slice(program, source_plan, source_bundle, source_report)["valid"] or source_report["actual_inputs_match_plan"] is not True or source_report["source_template_geometry_matches"] is not True:
        raise ValueError("Lookup RMS regression requires intact actual-tensor evidence")
    runtime = _runtime()
    if any(canonical_json(runtime.get(key)) != canonical_json(value) for key, value in source_report["runtime"].items()):
        raise ValueError("Lookup RMS runtime differs from the source observations")
    predictions = {}
    for role, record in source_plan["roles"].items():
        data = source_bundle["roles"][role]
        rows = [_rms_lookup_row(row, data["weight_bits"], record["epsilon"], lookup, runtime) for row in data["input_bits"]]
        predictions[role] = {key: [row[key] for row in rows] for key in ("mean_bits", "denominator_bits", "rsqrt_bits", "output_bits")}
    bundle = {"predictions": predictions}
    body = {"schema_version": 1, "scope": "Explicit-lookup RMS regression against preserved actual tensors, with prospective native rsqrt replay at the original small layouts; not native arithmetic reconstruction or full-layer qualification.",
            "program_sha256": program["program_sha256"], "source_rms_plan_sha256": source_plan["plan_sha256"],
            "source_rms_report_sha256": source_report["report_sha256"], "lookup_evidence": copy.deepcopy(lookup.evidence),
            "lookup_code_sha256": _code_sha(), "rms_code_sha256": _rms_code_commitment(),
            "lookup_rms_code_sha256": hashlib.sha256(inspect.getsource(_rms_lookup_row).encode("utf-8")).hexdigest(),
            "runtime": runtime, "expected_kernel_names": list(lookup.kernel_names), "bundle_sha256": _sha(bundle),
            "roles": {role: {"input_shape": record["input"]["shape"], "input_sha256": record["input"]["sha256"], "weight_sha256": record["weight"]["sha256"], "prediction_sha256": _sha(predictions[role])} for role, record in source_plan["roles"].items()},
            "source_data_reused": True, "native_arithmetic_reconstructed": False, "full_first_layer_qualified": False,
            "hardware_semantics_established": False, "global_exactness_activation_allowed": False}
    return _seal(body, "plan_sha256"), bundle


def _cuda_layout(bits: list[int], shape: list[int]) -> tuple[np.ndarray, list[str]]:
    import torch
    from .gemma_reduction_backend import _profile_call

    values = torch.tensor(bits, dtype=torch.int32, device="cuda").view(torch.float32).reshape(shape)
    if values.data_ptr() % 16 or not values.is_contiguous():
        raise ValueError("RMS lookup replay requires aligned contiguous rsqrt arguments")
    with torch.no_grad():
        output, names = _profile_call(lambda: values.rsqrt())
    if output.dtype != torch.float32 or list(output.shape) != shape:
        raise ValueError("Native RMS-layout rsqrt output type mismatch")
    return output.cpu().contiguous().view(torch.int32).numpy().reshape(-1).astype("<u4"), names


def rms_lookup_comparison(plan: dict[str, Any], bundle: dict[str, Any], source_report: dict[str, Any], native_records: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    summaries = {}
    for role, predicted in bundle["predictions"].items():
        source = source_report["observations"][role]["repetitions"]
        stage_counts = {key: [sum(a != b for a, b in zip(predicted[key], record[key])) for record in source] for key in ("mean_bits", "denominator_bits", "rsqrt_bits")}
        output_counts = [sum(a != b for predicted_row, observed_row in zip(predicted["output_bits"], record["output_bits"]) for a, b in zip(predicted_row, observed_row)) for record in source]
        summaries[role] = {"stored_stage_mismatch_counts": stage_counts, "stored_output_mismatch_counts": output_counts,
                           "native_root_mismatch_counts": [record["mismatch_count"] for record in native_records[role]]}
    stored_match = all(not any(item["stored_output_mismatch_counts"]) and all(not any(counts) for counts in item["stored_stage_mismatch_counts"].values()) for item in summaries.values())
    native_match = all(not any(item["native_root_mismatch_counts"]) for item in summaries.values())
    scope_match = canonical_json(runtime) == canonical_json(plan["runtime"]) and all(record["kernel_names"] == plan["expected_kernel_names"] for records in native_records.values() for record in records)
    return _seal({"schema_version": 1, "scope": plan["scope"], "plan_sha256": plan["plan_sha256"], "bundle_sha256": plan["bundle_sha256"],
                  "source_rms_report_sha256": source_report["report_sha256"], "lookup_evidence": plan["lookup_evidence"],
                  "runtime": runtime, "role_summaries": summaries, "native_root_replays": native_records,
                  "all_stored_rms_values_and_stages_match": stored_match, "all_native_root_layout_replays_match": native_match,
                  "source_kernel_environment_match": scope_match, "lookup_rms_check_passes": stored_match and native_match and scope_match,
                  "source_data_reused": True, "native_arithmetic_reconstructed": False, "full_first_layer_qualified": False,
                  "hardware_semantics_established": False, "global_exactness_activation_allowed": False}, "report_sha256")


def acquire_rms_lookup(program: dict[str, Any], source_plan: dict[str, Any], source_bundle: dict[str, Any], source_report: dict[str, Any], lookup: CheckedRsqrtLookup, plan: dict[str, Any], bundle: dict[str, Any], audit_dir: Path) -> dict[str, Any]:
    expected_plan, expected_bundle = build_rms_lookup_plan(program, source_plan, source_bundle, source_report, lookup)
    if canonical_json(plan) != canonical_json(expected_plan) or canonical_json(bundle) != canonical_json(expected_bundle):
        raise ValueError("Lookup RMS prediction plan or bundle mismatch")
    records = {}
    for role, predicted in bundle["predictions"].items():
        shape = plan["roles"][role]["input_shape"][:-1] + [1]
        expected = np.asarray(predicted["rsqrt_bits"], dtype="<u4")
        repetitions = []
        for repetition in range(3):
            observed, names = _cuda_layout(predicted["denominator_bits"], shape)
            mismatch = int(np.count_nonzero(expected != observed))
            repetitions.append({"repetition": repetition, "shape": shape, "value_count": len(expected),
                                "input_sha256": hashlib.sha256(np.asarray(predicted["denominator_bits"], dtype="<u4").tobytes()).hexdigest(),
                                "predicted_sha256": hashlib.sha256(expected.tobytes()).hexdigest(), "observed_sha256": hashlib.sha256(observed.tobytes()).hexdigest(),
                                "kernel_names": names, "mismatch_count": mismatch,
                                "mismatch_payload_sha256": _audit_payload(audit_dir, observed) if mismatch else None})
        records[role] = repetitions
    return rms_lookup_comparison(plan, bundle, source_report, records, _runtime())


def verify_rms_lookup(program: dict[str, Any], source_plan: dict[str, Any], source_bundle: dict[str, Any], source_report: dict[str, Any], lookup: CheckedRsqrtLookup, plan: dict[str, Any], bundle: dict[str, Any], report: dict[str, Any], audit_dir: Path, reexecute: bool = False) -> dict[str, Any]:
    try:
        expected_plan, expected_bundle = build_rms_lookup_plan(program, source_plan, source_bundle, source_report, lookup)
        if canonical_json(plan) != canonical_json(expected_plan) or canonical_json(bundle) != canonical_json(expected_bundle):
            raise ValueError("Lookup RMS plan or bundle mismatch")
        _check_hash(report, "report_sha256")
        if set(report["native_root_replays"]) != set(plan["roles"]):
            raise ValueError("Missing native RMS root layouts")
        for role, repetitions in report["native_root_replays"].items():
            if len(repetitions) != 3:
                raise ValueError("Missing RMS root repetitions")
            predicted = np.asarray(bundle["predictions"][role]["rsqrt_bits"], dtype="<u4")
            input_hash = hashlib.sha256(np.asarray(bundle["predictions"][role]["denominator_bits"], dtype="<u4").tobytes()).hexdigest()
            for repetition, record in enumerate(repetitions):
                if record["repetition"] != repetition or record["shape"] != plan["roles"][role]["input_shape"][:-1] + [1] or record["value_count"] != len(predicted) or record["input_sha256"] != input_hash or record["predicted_sha256"] != hashlib.sha256(predicted.tobytes()).hexdigest():
                    raise ValueError("RMS root replay identity mismatch")
                count = record["mismatch_count"]
                if type(count) is not int or not 0 <= count <= len(predicted):
                    raise ValueError("Invalid RMS root mismatch count")
                if count:
                    digest = record["mismatch_payload_sha256"]
                    if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                        raise ValueError("Invalid RMS mismatch payload reference")
                    raw = (audit_dir / (digest + ".bin")).read_bytes()
                    if len(raw) != len(predicted) * 4 or hashlib.sha256(raw).hexdigest() != digest or digest != record["observed_sha256"] or int(np.count_nonzero(predicted != np.frombuffer(raw, dtype="<u4"))) != count:
                        raise ValueError("RMS root mismatch payload is inconsistent")
                elif record["observed_sha256"] != record["predicted_sha256"] or record["mismatch_payload_sha256"] is not None:
                    raise ValueError("RMS root matching observation is inconsistent")
        expected = rms_lookup_comparison(plan, bundle, source_report, report["native_root_replays"], report["runtime"])
        integrity = canonical_json(expected) == canonical_json(report)
        replay = canonical_json(acquire_rms_lookup(program, source_plan, source_bundle, source_report, lookup, plan, bundle, audit_dir)) == canonical_json(report) if reexecute and integrity else None
        return {"valid": integrity and (not reexecute or replay is True), "mode": "native_rsqrt_replay_and_stored_rms_regression" if reexecute else "integrity_only",
                "reexecution_exact": replay, "lookup_rms_check_passes": expected["lookup_rms_check_passes"], "role_summaries": expected["role_summaries"],
                "global_exactness_activation_allowed": False}
    except (KeyError, TypeError, ValueError, RuntimeError, OSError, AttributeError) as error:
        return {"valid": False, "reason": str(error)}
