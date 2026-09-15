from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .gemma_rsqrt_lookup import _runtime, _audit_payload
from .gemma_rotary_slice import _sha, _seal, _check_hash
from .serialization import canonical_json

SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
START = 0x30800000
STOP = 0x43000000
FINITE_STOP = 0x7F800000
CHUNK = 1 << 20
TABLE_BYTES = (STOP - START) * 4
SCOPE = "Explicit empirical CUDA exponential mapping for all negative finite float32 encodings, positive zero and negative infinity; fixed runtime and aligned contiguous launch schedule; not reconstructed native exponential arithmetic."


def _code_sha() -> str:
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Exponential specification source changed after process import")
    return SOURCE_SHA256


def _region(magnitude: int) -> str:
    return "one" if magnitude < START else "table" if magnitude < STOP else "zero"


def _constant(region: str) -> np.ndarray:
    return np.full(CHUNK, 0x3F800000 if region == "one" else 0, dtype="<u4")


def _cuda_exp(bits: np.ndarray) -> tuple[np.ndarray, list[str]]:
    import torch
    from .gemma_reduction_backend import _profile_call

    inputs = torch.from_numpy(bits.astype(np.uint32, copy=False).view(np.int32)).to("cuda").view(torch.float32)
    if inputs.data_ptr() % 16 or not inputs.is_contiguous():
        raise ValueError("Exponential acquisition requires aligned contiguous inputs")
    with torch.no_grad():
        output, names = _profile_call(lambda: inputs.exp())
    if output.dtype != torch.float32 or output.shape != inputs.shape:
        raise ValueError("Native exponential output type mismatch")
    return output.cpu().view(torch.int32).numpy().astype("<u4"), names


def build_exp_plan(source_summary: dict[str, Any]) -> dict[str, Any]:
    _check_hash(source_summary, "summary_sha256")
    runtime = _runtime()
    if source_summary.get("candidate_passes") is not False or source_summary.get("fused_internal_stages_observed") is not False or canonical_json(source_summary["runtime"]) != canonical_json(runtime):
        raise ValueError("Expected preserved failed softmax evidence in the same runtime")
    body = {"schema_version": 1, "scope": SCOPE, "source_softmax_summary_sha256": source_summary["summary_sha256"],
            "source_softmax_report_sha256": source_summary["source_report_sha256"], "runtime": runtime, "code_sha256": _code_sha(),
            "table_magnitude_start": START, "table_magnitude_stop_exclusive": STOP, "finite_magnitude_stop_exclusive": FINITE_STOP,
            "table_bytes": TABLE_BYTES, "chunk_size": CHUNK, "chunk_count": FINITE_STOP // CHUNK,
            "negative_finite_encoding_count": FINITE_STOP, "special_input_bits": [0, 0xFF800000], "special_output_bits": [0x3F800000, 0],
            "region_chunk_counts": {"one": START // CHUNK, "table": (STOP - START) // CHUNK, "zero": (FINITE_STOP - STOP) // CHUNK},
            "constant_chunk_hashes": {region: hashlib.sha256(_constant(region).tobytes()).hexdigest() for region in ("one", "zero")},
            "repetitions": 3, "format": "little-endian uint32 output encodings, ascending negative-input magnitude",
            "kernel_policy": "exp_kernel_cuda; identical kernel-name lists across full-sized chunks",
            "acquisition_schedule": "one warm-up and one profiled call per stored observation; first table observation defines table entries",
            "constant_regions_are_hypotheses_until_exhaustive_validation": True,
            "native_arithmetic_reconstructed": False, "unrestricted_layouts_qualified": False,
            "hardware_semantics_established": False, "global_exactness_activation_allowed": False}
    return _seal(body, "plan_sha256")


def check_exp_plan(plan: dict[str, Any]) -> None:
    _check_hash(plan, "plan_sha256")
    expected = {"schema_version": 1, "scope": SCOPE, "code_sha256": _code_sha(), "table_magnitude_start": START,
                "table_magnitude_stop_exclusive": STOP, "finite_magnitude_stop_exclusive": FINITE_STOP,
                "table_bytes": TABLE_BYTES, "chunk_size": CHUNK, "chunk_count": FINITE_STOP // CHUNK,
                "negative_finite_encoding_count": FINITE_STOP, "special_input_bits": [0, 0xFF800000], "special_output_bits": [0x3F800000, 0],
                "region_chunk_counts": {"one": START // CHUNK, "table": (STOP - START) // CHUNK, "zero": (FINITE_STOP - STOP) // CHUNK},
                "constant_chunk_hashes": {region: hashlib.sha256(_constant(region).tobytes()).hexdigest() for region in ("one", "zero")},
                "repetitions": 3, "format": "little-endian uint32 output encodings, ascending negative-input magnitude",
                "kernel_policy": "exp_kernel_cuda; identical kernel-name lists across full-sized chunks",
                "acquisition_schedule": "one warm-up and one profiled call per stored observation; first table observation defines table entries",
                "constant_regions_are_hypotheses_until_exhaustive_validation": True, "native_arithmetic_reconstructed": False,
                "unrestricted_layouts_qualified": False, "hardware_semantics_established": False, "global_exactness_activation_allowed": False}
    if any(canonical_json(plan.get(key)) != canonical_json(value) for key, value in expected.items()) or not isinstance(plan.get("runtime"), dict):
        raise ValueError("Exponential specification domain/profile mismatch")


def _manifest(plan: dict[str, Any], records: list[dict[str, Any]], special: list[dict[str, Any]], table_sha: str, runtime: dict[str, Any]) -> dict[str, Any]:
    if len(records) != FINITE_STOP // CHUNK or len(special) != 3:
        raise ValueError("Exponential domain coverage is incomplete")
    kernel_reference = records[0]["observations"][0]["kernel_names"]
    observations = [observation for record in records for observation in record["observations"]]
    family = all(isinstance(item["kernel_names"], list) and item["kernel_names"] and all(isinstance(name, str) and "exp_kernel_cuda" in name for name in item["kernel_names"]) for item in [*observations, *special])
    kernels_stable = all(item["kernel_names"] == kernel_reference for item in observations)
    all_match = all(item["mismatch_count"] == 0 and item["output_range_valid"] is True for item in [*observations, *special])
    runtime_matches = canonical_json(runtime) == canonical_json(plan["runtime"])
    body = {"schema_version": 1, "scope": SCOPE, "plan_sha256": plan["plan_sha256"], "table_sha256": table_sha,
            "table_bytes": TABLE_BYTES, "records": records, "special_observations": special, "runtime": runtime,
            "negative_finite_encoding_count": FINITE_STOP, "covered_input_encoding_count": FINITE_STOP + 2,
            "all_values_match_declared_mapping": all_match, "kernel_family_matches": family, "full_chunk_kernels_stable": kernels_stable,
            "runtime_matches_plan": runtime_matches, "table_usable": all_match and family and kernels_stable and runtime_matches,
            "native_arithmetic_reconstructed": False, "unrestricted_layouts_qualified": False,
            "hardware_semantics_established": False, "global_exactness_activation_allowed": False}
    return _seal(body, "manifest_sha256")


def acquire_exp_table(plan: dict[str, Any], table_path: Path, audit_dir: Path, checkpoint: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    check_exp_plan(plan)
    if canonical_json(_runtime()) != canonical_json(plan["runtime"]):
        raise ValueError("Native exponential runtime differs from the plan")
    constants = {region: _constant(region) for region in ("one", "zero")}
    records, digest = [], hashlib.sha256()
    ramp = np.arange(CHUNK, dtype=np.uint32)
    with table_path.open("xb") as stream:
        for start in range(0, FINITE_STOP, CHUNK):
            region = _region(start)
            expected = constants.get(region)
            observations = []
            for repetition in range(3):
                values, names = _cuda_exp((ramp + np.uint32(start)) | np.uint32(0x80000000))
                payload = values.tobytes()
                if expected is None:
                    expected = values.copy()
                    stream.write(payload)
                    digest.update(payload)
                mismatch = int(np.count_nonzero(values != expected))
                valid_range = bool(np.all(values <= 0x3F800000))
                observations.append({"repetition": repetition, "output_sha256": hashlib.sha256(payload).hexdigest(), "kernel_names": names,
                                     "mismatch_count": mismatch, "output_range_valid": valid_range,
                                     "mismatch_payload_sha256": _audit_payload(audit_dir, values) if mismatch or not valid_range else None})
            records.append({"magnitude_start": start, "count": CHUNK, "region": region,
                            "expected_sha256": hashlib.sha256(expected.tobytes()).hexdigest(), "observations": observations})
            if len(records) % 16 == 0:
                stream.flush()
                checkpoint({"plan_sha256": plan["plan_sha256"], "completed_chunks": len(records), "records": records, "complete": False})
    special, expected_special = [], np.asarray(plan["special_output_bits"], dtype="<u4")
    for repetition in range(3):
        values, names = _cuda_exp(np.asarray(plan["special_input_bits"], dtype=np.uint32))
        mismatch, valid_range = int(np.count_nonzero(values != expected_special)), bool(np.all(values <= 0x3F800000))
        special.append({"repetition": repetition, "output_sha256": hashlib.sha256(values.tobytes()).hexdigest(), "kernel_names": names,
                        "mismatch_count": mismatch, "output_range_valid": valid_range,
                        "mismatch_payload_sha256": _audit_payload(audit_dir, values) if mismatch or not valid_range else None})
    _code_sha()
    manifest = _manifest(plan, records, special, digest.hexdigest(), _runtime())
    checkpoint({"plan_sha256": plan["plan_sha256"], "completed_chunks": len(records), "manifest_sha256": manifest["manifest_sha256"], "complete": True})
    return manifest


def load_exp_table(plan: dict[str, Any], manifest: dict[str, Any], table_path: Path, audit_dir: Path) -> np.ndarray:
    check_exp_plan(plan)
    _check_hash(manifest, "manifest_sha256")
    raw = table_path.read_bytes()
    if len(raw) != TABLE_BYTES or hashlib.sha256(raw).hexdigest() != manifest["table_sha256"] or manifest["plan_sha256"] != plan["plan_sha256"]:
        raise ValueError("Exponential table bytes/source commitment mismatch")
    table = np.frombuffer(raw, dtype="<u4")
    expected_manifest = _manifest(plan, manifest["records"], manifest["special_observations"], manifest["table_sha256"], manifest["runtime"])
    if canonical_json(manifest) != canonical_json(expected_manifest):
        raise ValueError("Exponential manifest flags or scope mismatch")
    constants = {region: _constant(region) for region in ("one", "zero")}

    def verify_observations(observations: list[dict[str, Any]], expected: np.ndarray) -> None:
        if len(observations) != 3:
            raise ValueError("Exponential repetition coverage mismatch")
        expected_sha = hashlib.sha256(expected.tobytes()).hexdigest()
        for repetition, item in enumerate(observations):
            if type(item["repetition"]) is not int or item["repetition"] != repetition or type(item["mismatch_count"]) is not int or not 0 <= item["mismatch_count"] <= expected.size:
                raise ValueError("Invalid exponential observation counter")
            if item["mismatch_payload_sha256"] is not None:
                name = item["mismatch_payload_sha256"]
                if not isinstance(name, str) or len(name) != 64 or any(character not in "0123456789abcdef" for character in name):
                    raise ValueError("Malformed exponential audit commitment")
                payload = (audit_dir / (name + ".bin")).read_bytes()
                if hashlib.sha256(payload).hexdigest() != name or len(payload) != expected.nbytes or item["output_sha256"] != name:
                    raise ValueError("Exponential mismatch payload changed")
                actual = np.frombuffer(payload, dtype="<u4")
                if item["mismatch_count"] != int(np.count_nonzero(actual != expected)) or item["output_range_valid"] is not bool(np.all(actual <= 0x3F800000)):
                    raise ValueError("Exponential mismatch statistics disagree with the payload")
            elif item["output_sha256"] != expected_sha or item["mismatch_count"] != 0 or item["output_range_valid"] is not True:
                raise ValueError("Missing or inconsistent exponential observation payload")

    for index, record in enumerate(manifest["records"]):
        start = index * CHUNK
        region = _region(start)
        expected = table[start - START:start - START + CHUNK] if region == "table" else constants[region]
        if type(record["magnitude_start"]) is not int or record["magnitude_start"] != start or type(record["count"]) is not int or record["count"] != CHUNK or record["region"] != region or record["expected_sha256"] != hashlib.sha256(expected.tobytes()).hexdigest():
            raise ValueError("Exponential domain coordinates or expected values changed")
        if region == "table" and not np.all(expected <= 0x3F800000):
            raise ValueError("Exponential table contains out-of-range outputs")
        verify_observations(record["observations"], expected)
    verify_observations(manifest["special_observations"], np.asarray(plan["special_output_bits"], dtype="<u4"))
    return table


class CheckedExpLookup:
    def __init__(self, plan: dict[str, Any], manifest: dict[str, Any], table_path: Path, audit_dir: Path):
        self._table = load_exp_table(plan, manifest, table_path, audit_dir)
        if manifest["table_usable"] is not True:
            raise ValueError("Exponential mapping did not pass complete domain validation")
        self._runtime = copy.deepcopy(plan["runtime"])
        self._evidence = {"plan_sha256": plan["plan_sha256"], "manifest_sha256": manifest["manifest_sha256"], "table_sha256": manifest["table_sha256"]}

    @property
    def evidence(self) -> dict[str, Any]:
        return copy.deepcopy(self._evidence)

    def predict_bits(self, bits: int, runtime: dict[str, Any]) -> int:
        if canonical_json(runtime) != canonical_json(self._runtime):
            raise ValueError("Exponential target runtime differs from validated evidence")
        if type(bits) is not int or not 0 <= bits <= 0xFFFFFFFF or (bits != 0 and (not bits & 0x80000000 or (bits & 0x7FFFFFFF) > FINITE_STOP)):
            raise ValueError("Exponential specification accepts nonpositive non-NaN float32 encodings only")
        magnitude = bits & 0x7FFFFFFF
        if magnitude < START:
            return 0x3F800000
        if magnitude >= STOP:
            return 0
        return int(self._table[magnitude - START])


def replay_exp_table(plan: dict[str, Any], manifest: dict[str, Any], table_path: Path, audit_dir: Path) -> dict[str, Any]:
    table = load_exp_table(plan, manifest, table_path, audit_dir)
    if manifest["table_usable"] is not True or canonical_json(_runtime()) != canonical_json(plan["runtime"]):
        raise ValueError("Exponential replay requires passing evidence in its declared runtime")
    records, constants = [], {region: _constant(region) for region in ("one", "zero")}
    ramp = np.arange(CHUNK, dtype=np.uint32)
    for index, start in enumerate(range(0, FINITE_STOP, CHUNK)):
        region = _region(start)
        expected = table[start - START:start - START + CHUNK] if region == "table" else constants[region]
        values, names = _cuda_exp((ramp + np.uint32(start)) | np.uint32(0x80000000))
        mismatch = int(np.count_nonzero(values != expected))
        records.append({"magnitude_start": start, "count": CHUNK, "output_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
                        "mismatch_count": mismatch, "kernel_names_match": names == manifest["records"][index]["observations"][0]["kernel_names"],
                        "mismatch_payload_sha256": _audit_payload(audit_dir, values) if mismatch else None})
    special, names = _cuda_exp(np.asarray(plan["special_input_bits"], dtype=np.uint32))
    special_match = special.tolist() == plan["special_output_bits"]
    runtime = _runtime()
    body = {"schema_version": 1, "scope": SCOPE, "plan_sha256": plan["plan_sha256"], "manifest_sha256": manifest["manifest_sha256"], "table_sha256": manifest["table_sha256"],
            "records": records, "covered_input_encoding_count": FINITE_STOP + 2, "special_match": special_match,
            "special_output_bits": special.tolist(), "special_kernel_names": names,
            "runtime": runtime, "reexecution_exact": special_match and names == manifest["special_observations"][0]["kernel_names"] and all(item["mismatch_count"] == 0 and item["kernel_names_match"] for item in records) and canonical_json(runtime) == canonical_json(plan["runtime"]),
            "native_arithmetic_reconstructed": False, "global_exactness_activation_allowed": False}
    _code_sha()
    return _seal(body, "replay_sha256")
