from __future__ import annotations

import copy
import hashlib
from typing import Any

from .attention_parameters import verify_attention_parameter_certificate
from .formal_proofs import _require_z3
from .launch_arguments import verify_launch_argument_artifact
from .serialization import canonical_json

try:
    import z3
except ModuleNotFoundError:
    z3 = None


def _prove_tensor_storage_bound(name: str, tensor: dict[str, Any]) -> dict[str, Any]:
    _require_z3()
    indices = [z3.Int(f"{name}_index_{axis}") for axis in range(len(tensor["shape"]))]
    domain = z3.And(
        *[
            z3.And(index >= 0, index < dimension)
            for index, dimension in zip(indices, tensor["shape"])
        ]
    )
    element_offset = sum(index * stride for index, stride in zip(indices, tensor["stride"]))
    byte_start = tensor["data_pointer_offset_bytes"] + element_offset * tensor["element_size_bytes"]
    byte_end = byte_start + tensor["element_size_bytes"]
    solver = z3.Solver()
    solver.add(domain, z3.Or(byte_start < 0, byte_end > tensor["storage_nbytes"]))
    result = solver.check()
    return {
        "name": name,
        "shape": tensor["shape"],
        "stride": tensor["stride"],
        "element_size_bytes": tensor["element_size_bytes"],
        "storage_nbytes": tensor["storage_nbytes"],
        "data_pointer_offset_bytes": tensor["data_pointer_offset_bytes"],
        "logical_tensor_sha256": tensor["sha256"],
        "storage_base_pointer_sha256": tensor["storage_base_pointer_sha256"],
        "data_pointer_sha256": tensor["data_pointer_sha256"],
        "element_offset_formula": " + ".join(
            f"index_{axis}*{stride}" for axis, stride in enumerate(tensor["stride"])
        ),
        "byte_range_formula": f"data_pointer_offset + element_offset*{tensor['element_size_bytes']} through exclusive end +{tensor['element_size_bytes']}",
        "domain": [f"0 <= index_{axis} < {dimension}" for axis, dimension in enumerate(tensor["shape"])],
        "solver_result": str(result),
        "proved": result == z3.unsat,
        "counterexample": str(solver.model()) if result == z3.sat else None,
    }


def build_attention_logical_bounds_certificate(
    artifact: dict[str, Any], attention_certificate: dict[str, Any]
) -> dict[str, Any]:
    artifact_valid = verify_launch_argument_artifact(artifact)["valid"]
    attention_valid = verify_attention_parameter_certificate(attention_certificate)["valid"]
    dispatch = artifact["attention_dispatch_report"]["operations"][0]
    if dispatch["input_names"][:3] != ["query", "key", "value"]:
        raise ValueError("Dispatcher inputs are not schema-named Q/K/V")
    tensors = {
        "query_ptr": dispatch["inputs"][0],
        "key_ptr": dispatch["inputs"][1],
        "value_ptr": dispatch["inputs"][2],
        "output_ptr": dispatch["outputs"][0],
    }
    scalars = attention_certificate["decoded_scalars"]
    expected_strides = {
        "query_ptr": [scalars["q_strideB"], scalars["q_strideH"], scalars["q_strideM"], 1],
        "key_ptr": [scalars["k_strideB"], scalars["k_strideH"], scalars["k_strideM"], 1],
        "value_ptr": [scalars["v_strideB"], scalars["v_strideH"], scalars["v_strideM"], 1],
        "output_ptr": [
            scalars["num_queries"] * scalars["o_strideM"],
            scalars["head_dim_value"],
            scalars["o_strideM"],
            1,
        ],
    }
    proofs = [_prove_tensor_storage_bound(name, tensor) for name, tensor in tensors.items()]
    checks = {
        "launch_artifact_valid": artifact_valid,
        "attention_parameter_certificate_valid": attention_valid,
        "source_named_pointer_bindings_present": attention_certificate[
            "qkv_pointer_and_logical_commitments_bound"
        ]
        and attention_certificate["source_named_dispatch_output_pointer_bound"],
        "decoded_strides_match_retained_tensors": all(
            tensors[name]["stride"] == expected for name, expected in expected_strides.items()
        ),
        "all_tensor_bounds_proved": all(proof["proved"] for proof in proofs),
    }
    body = {
        "scope": "Machine-checked logical index-to-storage bounds for source-named Q/K/V/output tensors using retained shapes, strides, element sizes, and storage commitments; not SASS effective-address or hardware memory-safety proof.",
        "launch_artifact_sha256": artifact["artifact_sha256"],
        "attention_parameter_certificate_sha256": attention_certificate["certificate_sha256"],
        "solver": {"name": "Z3", "version": z3.get_version_string()},
        "proofs": proofs,
        "proved": sum(proof["proved"] for proof in proofs),
        "total": len(proofs),
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "logical_index_storage_bounds_established": True,
        "sass_effective_address_formula_bound": False,
        "sass_effective_address_bounds_established": False,
        "kernel_memory_safety_established": False,
        "hardware_memory_access_conformance_established": False,
    }
    body["certificate_sha256"] = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    return body


def redact_attention_logical_bounds_certificate(certificate: dict[str, Any]) -> dict[str, Any]:
    redacted = copy.deepcopy(certificate)
    redacted["privacy"] = {"redacted": True}
    for proof in redacted["proofs"]:
        proof["logical_tensor_sha256"] = "redacted"
        proof["storage_base_pointer_sha256"] = "redacted"
        proof["data_pointer_sha256"] = "redacted"
    body = {key: value for key, value in redacted.items() if key != "certificate_sha256"}
    redacted["certificate_sha256"] = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    return redacted


def verify_attention_logical_bounds_certificate(certificate: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in certificate.items() if key != "certificate_sha256"}
    hash_valid = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest() == certificate.get(
        "certificate_sha256"
    )
    recomputed = [
        _prove_tensor_storage_bound(
            proof["name"],
            {
                "shape": proof["shape"],
                "stride": proof["stride"],
                "element_size_bytes": proof["element_size_bytes"],
                "storage_nbytes": proof["storage_nbytes"],
                "data_pointer_offset_bytes": proof["data_pointer_offset_bytes"],
                "sha256": proof["logical_tensor_sha256"],
                "storage_base_pointer_sha256": proof["storage_base_pointer_sha256"],
                "data_pointer_sha256": proof["data_pointer_sha256"],
            },
        )
        for proof in certificate.get("proofs", [])
    ]
    proofs_match = recomputed == certificate.get("proofs")
    checks_consistent = certificate.get("all_checks_pass") == all(certificate.get("checks", {}).values())
    boundaries_preserved = (
        certificate.get("logical_index_storage_bounds_established") is True
        and certificate.get("sass_effective_address_formula_bound") is False
        and certificate.get("sass_effective_address_bounds_established") is False
        and certificate.get("kernel_memory_safety_established") is False
        and certificate.get("hardware_memory_access_conformance_established") is False
    )
    return {
        "valid": bool(
            hash_valid
            and proofs_match
            and checks_consistent
            and boundaries_preserved
            and certificate.get("all_checks_pass")
            and certificate.get("proved") == certificate.get("total") == len(recomputed)
        ),
        "certificate_hash_valid": hash_valid,
        "proofs_reexecuted_and_match": proofs_match,
        "checks_consistent": checks_consistent,
        "boundaries_preserved": boundaries_preserved,
    }
