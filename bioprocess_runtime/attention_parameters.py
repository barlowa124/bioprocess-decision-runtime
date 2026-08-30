from __future__ import annotations

import hashlib
from typing import Any

from .kernel_signatures import ATTENTION_SOURCE_BLOB, PYTORCH_COMMIT, verify_kernel_signature_certificate
from .launch_arguments import verify_launch_argument_artifact
from .serialization import canonical_json


def _zero_pointer_sha256() -> str:
    return hashlib.sha256(b"0").hexdigest()


def build_attention_parameter_certificate(
    artifact: dict[str, Any], signature_certificate: dict[str, Any]
) -> dict[str, Any]:
    artifact_verification = verify_launch_argument_artifact(artifact)
    signature_verification = verify_kernel_signature_certificate(signature_certificate)
    launches = [
        launch
        for launch in artifact["launch_argument_report"]["launches"]
        if launch["qualified_module_stack"]
        and launch["qualified_module_stack"][-1] == "model.layers.0.self_attn"
    ]
    if len(launches) != 1:
        raise ValueError(f"Expected one layer-0 attention launch; found {len(launches)}")
    launch = launches[0]
    if len(launch["parameters"]) != 1 or "typed_decoding" not in launch["parameters"][0]:
        raise ValueError("Attention launch does not contain one typed parameter decoding")
    decoding = launch["parameters"][0]["typed_decoding"]
    dispatch_report = artifact.get("attention_dispatch_report")
    if dispatch_report is None or len(dispatch_report.get("operations", [])) != 1:
        raise ValueError("Expected one retained scaled-dot-product dispatcher operation")
    dispatch = dispatch_report["operations"][0]
    if len(dispatch["inputs"]) < 3 or not dispatch["outputs"]:
        raise ValueError("Dispatcher record does not contain Q/K/V inputs and an output")
    scalars = decoding["scalars"]
    pointers = decoding["pointer_field_sha256"]
    expected = artifact["attention_expectations"]
    sequence = expected["sequence_length"]
    heads = expected["num_attention_heads"]
    dimension = expected["head_dim"]
    null_hash = _zero_pointer_sha256()
    checks = {
        "launch_artifact_valid": artifact_verification["valid"],
        "signature_certificate_valid": signature_verification["valid"],
        "source_commit_matches": decoding["source_commit"] == PYTORCH_COMMIT,
        "source_blob_matches": decoding["source_blob_sha1"] == ATTENTION_SOURCE_BLOB,
        "parameter_size_matches": decoding["size_bytes"] == 264,
        "head_dimensions_match": scalars["head_dim"] == dimension and scalars["head_dim_value"] == dimension,
        "sequence_dimensions_match": scalars["num_queries"] == sequence and scalars["num_keys"] == sequence,
        "batch_and_heads_match": scalars["num_batches"] == expected["batch_size"] and scalars["num_heads"] == heads,
        "scale_matches": scalars["scale"] == expected["scaling"],
        "query_strides_match": scalars["q_strideM"] == dimension
        and scalars["q_strideH"] == sequence * dimension
        and scalars["q_strideB"] == heads * sequence * dimension,
        "key_strides_match": scalars["k_strideM"] == dimension
        and scalars["k_strideH"] == sequence * dimension
        and scalars["k_strideB"] == heads * sequence * dimension,
        "value_strides_match": scalars["v_strideM"] == dimension
        and scalars["v_strideH"] == sequence * dimension
        and scalars["v_strideB"] == heads * sequence * dimension,
        "output_stride_matches": scalars["o_strideM"] == heads * dimension,
        "causal_mask_matches": scalars["custom_mask_type"] == 1 and scalars["causal_diagonal_offset"] == 0,
        "short_sequence_window_field_zero_consistent_with_config": expected["is_sliding"] is True
        and scalars["window_size"] == 0
        and sequence <= expected["sliding_window"],
        "dropout_disabled": scalars["use_dropout"] is False and scalars["dropout_prob"] == 0.0,
        "fixed_length_pointers_null": all(
            pointers[name] == null_hash
            for name in ("attn_bias_ptr", "seqstart_q_ptr", "seqstart_k_ptr", "seqlen_k_ptr")
        ),
        "qkv_and_output_pointers_present": all(
            pointers[name] != null_hash
            for name in ("query_ptr", "key_ptr", "value_ptr", "output_ptr", "output_accum_ptr")
        ),
        "dispatch_operation_matches": dispatch["operation"] == "aten._scaled_dot_product_efficient_attention.default",
        "dispatch_qkv_argument_names_match": dispatch["input_names"][:3] == ["query", "key", "value"],
        "qkv_pointer_fields_match_dispatch_inputs": pointers["query_ptr"] == dispatch["inputs"][0]["data_pointer_sha256"]
        and pointers["key_ptr"] == dispatch["inputs"][1]["data_pointer_sha256"]
        and pointers["value_ptr"] == dispatch["inputs"][2]["data_pointer_sha256"],
        "output_pointer_field_matches_dispatch_output": pointers["output_ptr"]
        == dispatch["outputs"][0]["data_pointer_sha256"],
        "qkv_shapes_match": all(
            tensor["shape"] == [expected["batch_size"], heads, sequence, dimension]
            for tensor in dispatch["inputs"][:3]
        ),
        "qkv_strides_match_dispatch": dispatch["inputs"][0]["stride"]
        == [heads * sequence * dimension, sequence * dimension, dimension, 1]
        and dispatch["inputs"][1]["stride"]
        == [heads * sequence * dimension, sequence * dimension, dimension, 1]
        and dispatch["inputs"][2]["stride"]
        == [heads * sequence * dimension, sequence * dimension, dimension, 1],
    }
    later_module_output_equal = any(
        match["parameter_index"] == 0
        and match["parameter_byte_offset"] == 0
        and any(tensor["role"] == "output" for tensor in match["matches"])
        for match in launch.get("parameter_pointer_matches", [])
    )
    body = {
        "scope": "Decoded from a verified-source, source-reconstructed ctypes layout and bound to schema-named dispatcher tensors; not a compiled, hardware-verified, or fully typed kernel signature, and not a read/write or memory-access proof.",
        "launch_artifact_sha256": artifact["artifact_sha256"],
        "signature_certificate_sha256": signature_certificate["certificate_sha256"],
        "qualified_module_stack": launch["qualified_module_stack"],
        "grid": launch["grid"],
        "block": launch["block"],
        "shared_memory_bytes": launch["shared_memory_bytes"],
        "expectations": expected,
        "decoded_scalars": scalars,
        "pointer_field_sha256": pointers,
        "attention_dispatch_operation": {
            "operation": dispatch["operation"],
            "schema": dispatch["schema"],
            "input_names": dispatch["input_names"],
            "operation_sha256": dispatch["operation_sha256"],
            "input_tensor_sha256": [tensor["sha256"] for tensor in dispatch["inputs"][:3]],
            "output_tensor_sha256": dispatch["outputs"][0]["sha256"],
        },
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "query_pointer_temporal_observation": {
            "source_field": "query_ptr",
            "parameter_byte_offset": 0,
            "later_module_output_address_equal": later_module_output_equal,
            "typed_output_binding": False,
            "resolved_as_dispatch_query_input": True,
            "interpretation": "Retaining dispatcher tensors prevents allocator reuse from being mistaken for field identity.",
        },
        "source_layout_reconstructed": True,
        "compiled_layout_verified": False,
        "typed_signature_established": False,
        "qkv_pointer_and_logical_commitments_bound": True,
        "source_named_dispatch_output_pointer_bound": True,
        "kernel_read_write_semantics_established": False,
        "memory_access_semantics_established": False,
    }
    body["certificate_sha256"] = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    return body


def verify_attention_parameter_certificate(certificate: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in certificate.items() if key != "certificate_sha256"}
    hash_valid = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest() == certificate.get(
        "certificate_sha256"
    )
    checks_consistent = certificate.get("all_checks_pass") == all(certificate.get("checks", {}).values())
    boundaries_preserved = (
        certificate.get("source_layout_reconstructed") is True
        and certificate.get("compiled_layout_verified") is False
        and certificate.get("typed_signature_established") is False
        and certificate.get("qkv_pointer_and_logical_commitments_bound") is True
        and certificate.get("source_named_dispatch_output_pointer_bound") is True
        and certificate.get("kernel_read_write_semantics_established") is False
        and certificate.get("memory_access_semantics_established") is False
        and certificate.get("query_pointer_temporal_observation", {}).get("typed_output_binding") is False
        and certificate.get("query_pointer_temporal_observation", {}).get("resolved_as_dispatch_query_input") is True
    )
    return {
        "valid": bool(hash_valid and checks_consistent and boundaries_preserved and certificate.get("all_checks_pass")),
        "certificate_hash_valid": hash_valid,
        "checks_consistent": checks_consistent,
        "boundaries_preserved": boundaries_preserved,
    }
