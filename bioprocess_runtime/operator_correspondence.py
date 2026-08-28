from __future__ import annotations

import hashlib
from typing import Any

from .serialization import canonical_json


STAGE_PATTERNS = {
    "embedding_lookup": ("indexSelect",),
    "rms_normalization": ("MeanOps", "PowKernel", "rsqrt_kernel", "BinaryFunctorIfff"),
    "rotary_embedding": ("cos_kernel", "sin_kernel", "CatArrayBatchedCopy"),
    "linear_projections": ("gemm",),
    "fused_sdpa_attention": ("fmha_cutlass",),
    "gated_gelu_mlp": ("GeluCUDAKernelImpl", "BinaryFunctorIN3c108BFloat16"),
    "residual_addition": ("CUDAFunctor_addIN3c108BFloat16",),
    "mask_and_index_construction": ("arange_cuda_out", "and_kernel_cuda"),
}


def build_operator_correspondence(suite: dict[str, Any]) -> dict[str, Any]:
    entries = suite["entries"]
    stages = []
    for stage, patterns in STAGE_PATTERNS.items():
        matches = {
            pattern: [
                {
                    "index": entry["index"],
                    "kernel_name": entry["kernel_name"],
                    "module_id": entry["module_id"],
                    "sass_canonical_sha256": entry["sass_canonical_sha256"],
                }
                for entry in entries
                if pattern.lower() in entry["kernel_name"].lower()
            ]
            for pattern in patterns
        }
        stages.append(
            {
                "stage": stage,
                "required_symbol_patterns": list(patterns),
                "matched_symbols": matches,
                "all_patterns_observed_and_attested": all(matches[pattern] for pattern in patterns),
                "semantic_equivalence_established": False,
            }
        )
    body = {
        "scope": "Architecture-to-kernel-symbol correspondence for one attested Gemma forward; symbol occurrence is not proof that every invocation implements the named mathematical stage.",
        "source_suite_sha256": suite["suite_sha256"],
        "all_distinct_kernel_symbols_attested": suite["complete"],
        "stages": stages,
        "all_stage_patterns_observed_and_attested": all(stage["all_patterns_observed_and_attested"] for stage in stages),
        "full_operator_semantic_equivalence_established": False,
        "remaining_obligations": [
            "Correlate each repeated kernel invocation to a specific model layer and operator call.",
            "Prove tensor-address, shape, stride, dtype, and scalar-parameter bindings for each launch.",
            "Complete semantics for every instruction and modifier in each matched function.",
            "Prove composed instruction semantics equal the independent Gemma operator equations.",
            "Independently validate NVIDIA hardware execution against the proposed SASS semantics.",
        ],
    }
    body["correspondence_sha256"] = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    return body


def verify_operator_correspondence(correspondence: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in correspondence.items() if key != "correspondence_sha256"}
    hash_valid = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest() == correspondence.get(
        "correspondence_sha256"
    )
    stage_claims = all(
        stage["all_patterns_observed_and_attested"]
        == all(stage["matched_symbols"][pattern] for pattern in stage["required_symbol_patterns"])
        and stage["semantic_equivalence_established"] is False
        for stage in correspondence.get("stages", [])
    )
    aggregate_claim = correspondence.get("all_stage_patterns_observed_and_attested") == all(
        stage["all_patterns_observed_and_attested"] for stage in correspondence.get("stages", [])
    )
    semantic_boundary = correspondence.get("full_operator_semantic_equivalence_established") is False
    return {
        "valid": bool(hash_valid and stage_claims and aggregate_claim and semantic_boundary),
        "correspondence_hash_valid": hash_valid,
        "stage_claims_consistent": stage_claims,
        "aggregate_claim_consistent": aggregate_claim,
        "semantic_boundary_preserved": semantic_boundary,
    }
