"""Explicit execution-profile declaration for the fixed Gemma 3 270M path.

The pinned numerical modules keep their literal specializations (sequence
length 30, positions 0..29, 30-to-32 padding, sliding window 512). This
module declares that boundary once, binds it to the frozen evidence
artifacts, and checks that both the artifacts and the source files still
express the declared profile. It is a declaration and conformance layer,
not a numerical engine: it performs no model inference, no numerical
replay, and no proof. A profile that has produced frozen evidence is never
edited in place; different sequence lengths, positions, decoding modes, or
backends require a new profile ID and new evidence.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent

PROFILE_ID = "gemma3_270m_seq30_v1"

SEQUENCE_LENGTH = 30
POSITION_IDS = list(range(SEQUENCE_LENGTH))
SLIDING_WINDOW = 512
SOFTMAX_LANES = 32
ROTARY_FREQUENCIES = 128

# Pinned checkpoint identity, from results/gemma3_270m_copy_drift_v1.json
# environment.model_file_sha256 (that artifact's own hash is bound below).
MODEL_FILE_SHA256 = {
    ".gitattributes": "34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930",
    "added_tokens.json": "50b2f405ba56a26d4913fd772089992252d7f942123cc0a034d96424221ba946",
    "chat_template.jinja": "af95fbef33b76a50e5f463ff9766f85ce84f5849d915c4bd6c1619d852ac3231",
    "config.json": "2706d3533059c6e1086badab27cc234e8ca2228975c3f73eeaef7f57cb5ec1db",
    "generation_config.json": "be9e552870ff18a6c6beb0f6811030509c040d641e6972ce53c2fb540bbb4ba0",
    "model.safetensors": "700b710a9a99c295ed546647aa81cacf9f81f4c573ea2be613a0e2517a44afab",
    "README.md": "645fadbf83bfca50136a9b10265266b3db3eec671b03c24c5f28594ff4af9a5d",
    "special_tokens_map.json": "45a857d8a2495d0be30a5d2d6de03278195eb028b6e0b8efc248bfa697d65f05",
    "tokenizer.json": "4667f2089529e8e7657cfb6d1c19910ae71ff5f28aa7ab2ff2763330affad795",
    "tokenizer.model": "1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c",
    "tokenizer_config.json": "be2182df1ad0ea735d336418896eb06e5cc2f59136e7aaeac5de2ec2197742ac",
}

# Literal specializations that must remain present in the pinned numerical
# sources for this profile to describe the code. If a source is later
# generalized, the corresponding check fails and a new profile version is
# required instead of silently widening the declared boundary.
SOURCE_SPECIALIZATIONS = [
    {"file": "gemma_independent.py", "contains": ["sequence != 30", "window != 512"],
     "meaning": "causal-mask gate admits only S=30 with a full or sliding-512 window"},
    {"file": "gemma_independent.py", "contains": ["list(range(30))", "[30, 640]"],
     "meaning": "position IDs, rotary packet positions and embedding snapshots pinned to 0..29"},
    {"file": "gemma_first_layer.py", "contains": ["[1, 30]", "[1, 4, 30, 30]", "[1, 30, 640]"],
     "meaning": "token IDs, softmax and hidden-state shapes fixed at S=30"},
    {"file": "gemma_softmax_lookup.py", "contains": ["len(bits) != 30"],
     "meaning": "softmax row lookup requires exactly 30 values"},
    {"file": "gemma_softmax_slice.py", "contains": ['"elements": 30', '"lanes": 32', "[1, 4, 30, 30]"],
     "meaning": "30 values padded into the 32-lane reduction tree"},
    {"file": "gemma_rotary_slice.py", "contains": ["positions != list(range(30))", "len(frequency_bits) != 128"],
     "meaning": "rotary evidence covers positions 0..29 and 128 frequencies only"},
    {"file": "gemma_attention_scores.py", "contains": ["(1, 4, 30, 30)", "(1, 1, 30, 256)"],
     "meaning": "score, mask and K/V shapes fixed at S=30"},
    {"file": "gemma_attention_output.py", "contains": ['"aggregation_reduction_length": 30', '"aggregation_candidate_length": 32'],
     "meaning": "attention-value aggregation pads 30 positions to 32 candidates"},
    {"file": "gemma_second_layer.py", "contains": ["[1, 30, 640]", "[1, 4, 30, 30]"],
     "meaning": "second-layer state and softmax shapes fixed at S=30"},
    {"file": "gemma_two_layers.py", "contains": ["len(ids[0]) != 30"],
     "meaning": "two-layer connected prediction rejects non-30 inputs"},
    {"file": "gemma_third_layer_scores.py", "contains": ["[1, 4, 30, 256]", "[list(range(30))]"],
     "meaning": "third-layer Q/K/V shapes and position IDs fixed at S=30"},
    {"file": "gemma_first_layer_holdout.py", "contains": ["selected[:30]", "[1, 30]"],
     "meaning": "first-layer holdout cases are 30 raw token IDs"},
    {"file": "gemma_two_layers_holdout.py", "contains": ["selected[:30]", "[1, 30]"],
     "meaning": "two-layer holdout cases are 30 raw token IDs"},
]

SUPPORTED_OPCODES = [
    "ADD", "ARANGE", "ARGMAX", "CAUSAL_MASK", "EMBEDDING", "GELU_TANH",
    "LINEAR", "MATMUL_AV", "MATMUL_QK", "MUL", "REPEAT_KV",
    "RESHAPE_TRANSPOSE_HEADS", "RMS_NORM", "ROTARY_APPLY_PAIR",
    "ROTARY_TABLE", "SCALE", "SLICE_LAST_TOKEN", "SOFTMAX",
    "TRANSPOSE_RESHAPE_HEADS",
]

PROFILE = {
    "profile_id": PROFILE_ID,
    "status": "frozen_regression_profile_not_a_general_interface",
    "model": {
        "checkpoint_reference": "unsloth/gemma-3-270m-it bound by file SHA-256, not by name",
        "class": "Gemma3ForCausalLM",
        "layers": 18,
        "hidden_size": 640,
        "attention_heads": 4,
        "head_dimension": 256,
        "mlp_intermediate": 2048,
        "vocabulary_size": 262144,
        "dtype": "torch.bfloat16",
        "attention_implementation": "eager",
        "layer_types_recorded": {"sliding_attention_layers": [i for i in range(18) if i % 6 != 5],
                                  "full_attention_layers": [5, 11, 17]},
        "model_file_sha256": MODEL_FILE_SHA256,
    },
    "input_domain": {
        "batch_size": 1,
        "sequence_length": SEQUENCE_LENGTH,
        "position_ids": POSITION_IDS,
        "mask": {
            "kind": "causal_lower_triangular",
            "allowed_windows": [None, SLIDING_WINDOW],
            "masked_fill_bfloat16_bits": "0xFF7F",
        },
    },
    "decoding_paths": {
        "independent_engine": {
            "strategy": "greedy_argmax",
            "logit_domain": "finite_bfloat16_only",
            "cache": "none_full_context_recompute",
            "use_cache": False,
            "tie_break": "first index via numpy argmax over sign-magnitude-ordered BF16 bits",
        },
        "native_acquisition_copy_drift_v1": {
            "strategy": "greedy (harness do_sample=False overrides generation_config do_sample=True)",
            "use_cache": True,
            "cache_implementation": "hybrid",
            "sliding_window": SLIDING_WINDOW,
            "sliding_window_pattern": 6,
            "max_new_tokens": 400,
            "output_scores": True,
            "inert_under_greedy": ["top_k", "top_p", "temperature"],
        },
        "decode_paths_established_equivalent": False,
    },
    "shape_specializations": SOURCE_SPECIALIZATIONS,
    "reduction_layout": {
        "softmax": {"row_values": SEQUENCE_LENGTH, "tree_lanes": SOFTMAX_LANES,
                     "padding_entries": SOFTMAX_LANES - SEQUENCE_LENGTH,
                     "padding_semantics": "negative infinity then positive-zero exponential"},
        "attention_value_matmul": {"positions": SEQUENCE_LENGTH, "padded_to": SOFTMAX_LANES},
        "rotary": {"positions": POSITION_IDS, "frequencies": ROTARY_FREQUENCIES,
                    "provider_kind": "fixed_position_empirical_global_rotary_v1"},
    },
    "arithmetic_coverage": {
        "supported_opcodes": SUPPORTED_OPCODES,
        "linear_profiles": {
            "serial": "operand_alignment_v1",
            "key_value": "dense_split_profile",
            "output": "k128:bfloat16_rne:sequential_float32_rne",
            "down": "supported_candidate",
            "vocabulary": {"mode": "gemv", "profile": "gemv_k640_stride8_fp32_rne_halves_bf16_v1",
                            "requires_explicit_opt_in": True},
        },
        "primitive_evidence": [
            "empirical per-element rsqrt/exp/GELU lookup specifications",
            "fixed-position empirical global rotary table (positions 0..29)",
            "checked fixed-position local rotary constants",
            "softmax row lookup over 30-value rows",
        ],
    },
    "bound_artifacts": {
        "baseline_token_ids": "results/gemma3_270m_independent_baseline_tokens.json",
        "operational_semantics_summary": "results/gemma3_270m_operational_semantics_summary.json",
        "independent_holdout_declaration": "results/gemma3_270m_independent_holdout_declaration.json",
        "copy_drift_probe": "results/gemma3_270m_copy_drift_v1.json",
    },
    "non_claims": [
        "Arbitrary sequence lengths, batch sizes, or position ranges",
        "Equivalence between the cached hybrid acquisition path and the full-recompute independent path",
        "Hardware or kernel-level arithmetic semantics",
        "Unrestricted-input or full-model qualification",
        "Causal explanation of any recorded output",
    ],
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def profile() -> dict[str, Any]:
    return copy.deepcopy(PROFILE)


def validate_against_artifacts(results_dir: Path) -> list[dict[str, Any]]:
    """Check that the frozen evidence still records this profile's boundary."""
    checks = []

    def record(name: str, expected: Any, observed: Any) -> None:
        checks.append({"check": name, "expected": expected, "observed": observed,
                       "passed": expected == observed})

    declaration = json.loads((results_dir / "gemma3_270m_independent_holdout_declaration.json").read_text(encoding="utf-8"))
    record("holdout_declaration.batch_size", 1, declaration.get("batch_size"))
    record("holdout_declaration.sequence_length", SEQUENCE_LENGTH, declaration.get("sequence_length"))
    record("holdout_declaration.position_ids", POSITION_IDS, declaration.get("position_ids"))
    record("holdout_declaration.instruction_count", 533, declaration.get("instruction_count"))

    baseline = json.loads((results_dir / "gemma3_270m_independent_baseline_tokens.json").read_text(encoding="utf-8"))
    record("baseline.sequence_count", 1, len(baseline))
    record("baseline.token_count", SEQUENCE_LENGTH, len(baseline[0]) if baseline else None)

    summary = json.loads((results_dir / "gemma3_270m_operational_semantics_summary.json").read_text(encoding="utf-8"))
    record("operational_summary.token_count", SEQUENCE_LENGTH, summary.get("input", {}).get("token_count"))

    probe = json.loads((results_dir / "gemma3_270m_copy_drift_v1.json").read_text(encoding="utf-8"))
    environment = probe.get("environment", {})
    record("copy_drift.decode", "greedy", environment.get("decode"))
    record("copy_drift.dtype", "bfloat16", environment.get("dtype"))
    record("copy_drift.attn_implementation", "eager", environment.get("attn_implementation"))
    record("copy_drift.max_new_tokens", 400, environment.get("max_new_tokens"))
    record("copy_drift.model_file_sha256", PROFILE["model"]["model_file_sha256"],
           environment.get("model_file_sha256"))
    return checks


def check_source_specializations(package_dir: Path) -> list[dict[str, Any]]:
    """Check that the pinned source files still contain the declared literals."""
    checks = []
    for site in SOURCE_SPECIALIZATIONS:
        path = package_dir / site["file"]
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        missing = [literal for literal in site["contains"] if literal not in text]
        checks.append({"file": site["file"], "meaning": site["meaning"],
                       "literals": site["contains"], "missing": missing,
                       "passed": not missing and path.exists()})
    return checks


def build_report(results_dir: Path = ROOT / "results",
                 package_dir: Path = Path(__file__).resolve().parent) -> dict[str, Any]:
    artifact_checks = validate_against_artifacts(results_dir)
    source_checks = check_source_specializations(package_dir)
    body = {
        "schema_version": 1,
        "kind": "explicit_execution_profile_declaration",
        "scope": "Declaration and conformance checking of the frozen S=30 execution "
                 "profile. No model inference, numerical replay, proof, or causal claim.",
        "profile": PROFILE,
        "declaration_sha256": _sha(json.dumps(PROFILE, sort_keys=True, separators=(",", ":"),
                                              allow_nan=False).encode("utf-8")),
        "module_sha256": _sha(Path(__file__).read_bytes()),
        "artifact_checks": artifact_checks,
        "source_checks": source_checks,
        "artifacts_conform": all(check["passed"] for check in artifact_checks),
        "sources_conform": all(check["passed"] for check in source_checks),
        "model_inference_performed": False,
    }
    body["conformant"] = body["artifacts_conform"] and body["sources_conform"]
    body["report_sha256"] = _sha(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                          allow_nan=False).encode("utf-8"))
    return body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = build_report(args.results_dir)
        text = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(text)
        else:
            print(text, end="")
        if not report["conformant"]:
            parser.exit(1)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        parser.exit(1, f"Execution-profile check rejected: {error}\n")


if __name__ == "__main__":
    main()
