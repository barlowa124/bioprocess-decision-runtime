"""Offline recovery of copy/drift prompt token IDs and effective decode settings.

The saved copy/drift probe recorded prompt text, generated token IDs and
top-five scores, but not the prompt token IDs or the explicit cache settings
used during acquisition. This module recovers both from hash-pinned sources
without running the model:

- prompt token IDs come from the byte-exact tokenizer assets and chat
  template (verified against the SHA-256 set recorded at acquisition time),
  rendered for a single user message with ``add_generation_prompt=True``
  exactly as ``artifacts/gemma_copy_drift_probe_v1.py`` invoked it;
- effective decode settings come from the pinned ``generation_config.json``
  and ``config.json`` plus the harness's explicit ``generate`` arguments.

The pipeline is conformance-checked against the frozen 30-token baseline:
retokenizing the recorded baseline prompt must reproduce the saved token IDs
exactly. This is evidence recovery, not fresh inference, numerical replay,
or a causal claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent

PINNED_ASSET_SHA256 = {
    "tokenizer.json": "4667f2089529e8e7657cfb6d1c19910ae71ff5f28aa7ab2ff2763330affad795",
    "tokenizer_config.json": "be2182df1ad0ea735d336418896eb06e5cc2f59136e7aaeac5de2ec2197742ac",
    "special_tokens_map.json": "45a857d8a2495d0be30a5d2d6de03278195eb028b6e0b8efc248bfa697d65f05",
    "added_tokens.json": "50b2f405ba56a26d4913fd772089992252d7f942123cc0a034d96424221ba946",
    "chat_template.jinja": "af95fbef33b76a50e5f463ff9766f85ce84f5849d915c4bd6c1619d852ac3231",
    "generation_config.json": "be9e552870ff18a6c6beb0f6811030509c040d641e6972ce53c2fb540bbb4ba0",
    "config.json": "2706d3533059c6e1086badab27cc234e8ca2228975c3f73eeaef7f57cb5ec1db",
}

BASELINE_PROMPT = ("The oxygen reading is 30 percent and declining. "
                   "Does this require review? Answer Yes or No.")

HARNESS_FILE = "artifacts/gemma_copy_drift_probe_v1.py"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_tokenizer(model_dir: Path):
    try:
        from tokenizers import Tokenizer
    except ImportError as error:
        raise RuntimeError("tokenizers==0.21.4 is required for offline retokenization") from error
    return Tokenizer.from_file(str(model_dir / "tokenizer.json"))


def verify_tokenizer_assets(model_dir: Path,
                            pinned: dict[str, str] | None = None) -> dict[str, Any]:
    """Verify the tokenizer-side assets byte-for-byte against the pinned hashes."""
    pinned = PINNED_ASSET_SHA256 if pinned is None else pinned
    report = {}
    for name, pinned_hash in pinned.items():
        path = model_dir / name
        if not path.exists():
            report[name] = {"present": False, "sha256": None, "matches_pinned": False}
            continue
        digest = _sha(path.read_bytes())
        report[name] = {"present": True, "sha256": digest,
                        "matches_pinned": digest == pinned_hash}
    missing = [name for name, entry in report.items() if not entry["present"]]
    mismatched = [name for name, entry in report.items()
                  if entry["present"] and not entry["matches_pinned"]]
    if missing or mismatched:
        raise ValueError(f"Tokenizer assets do not match the pinned acquisition hashes; "
                         f"missing={missing} mismatched={mismatched}")
    return report


def render_chat_prompt(template_source: str, special_tokens: dict[str, str],
                       user_content: str) -> str:
    """Render one user message exactly as transformers apply_chat_template does.

    Mirrors the harness call: a single user message, no system message,
    add_generation_prompt=True. transformers renders the pinned jinja
    template inside an ImmutableSandboxedEnvironment with trim_blocks and
    lstrip_blocks enabled and a raise_exception global, then tokenizes the
    rendered text with add_special_tokens=False (special tokens are literal
    text inside the template).
    """
    try:
        from jinja2 import StrictUndefined, TemplateError
        from jinja2.sandbox import ImmutableSandboxedEnvironment
    except ImportError as error:
        raise RuntimeError("jinja2>=3.1 is required for offline template rendering") from error

    def raise_exception(message: str):
        raise TemplateError(message)

    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                undefined=StrictUndefined)
    environment.globals["raise_exception"] = raise_exception
    template = environment.from_string(template_source)
    return template.render(messages=[{"role": "user", "content": user_content}],
                           add_generation_prompt=True, **special_tokens)


def _special_tokens(tokenizer_config: dict[str, Any]) -> dict[str, str]:
    return {name: value for name, value in tokenizer_config.items()
            if name.endswith("_token") and isinstance(value, str)}


def tokenize_prompt(tokenizer: Any, rendered: str) -> list[int]:
    encoding = tokenizer.encode(rendered, add_special_tokens=False)
    return list(encoding.ids)


def decode_settings(generation_config: dict[str, Any], config: dict[str, Any],
                    harness_sha256: str) -> dict[str, Any]:
    """Effective settings of the recorded acquisition decode.

    The harness called ``model.generate(inputs, max_new_tokens=400,
    do_sample=False, return_dict_in_generate=True, output_scores=True)``
    with no explicit ``use_cache`` or ``cache_implementation``, so the
    generation config's defaults applied: hybrid cache (sliding-window
    layers 512 except every 6th layer full attention), greedy argmax.
    """
    layer_types = config.get("layer_types") or []
    full = [index for index, kind in enumerate(layer_types) if kind == "full_attention"]
    sliding = [index for index, kind in enumerate(layer_types) if kind == "sliding_attention"]
    return {
        "acquisition_path": "transformers model.generate",
        "harness": {
            "file": HARNESS_FILE,
            "sha256": harness_sha256,
            "generate_arguments": {"max_new_tokens": 400, "do_sample": False,
                                    "return_dict_in_generate": True, "output_scores": True},
            "explicit_use_cache_argument": None,
        },
        "effective": {
            "decoding": "greedy_argmax",
            "use_cache": config.get("use_cache"),
            "cache_implementation": generation_config.get("cache_implementation"),
            "sliding_window": config.get("sliding_window"),
            "sliding_window_pattern": config.get("_sliding_window_pattern"),
            "full_attention_layers": full,
            "sliding_attention_layers": sliding,
            "eos_token_id": generation_config.get("eos_token_id"),
            "bos_token_id": generation_config.get("bos_token_id"),
            "pad_token_id": generation_config.get("pad_token_id"),
            "inert_under_greedy": {name: generation_config.get(name)
                                    for name in ("do_sample", "top_k", "top_p")},
        },
        "overridden_by_harness": {"do_sample": False},
        "independent_engine_contrast": {"use_cache": False,
                                       "cache_implementation": "none_full_context_recompute"},
        "paths_established_equivalent": False,
    }


def build_report(model_dir: Path, probe_path: Path, protocol_path: Path,
                 heldout_path: Path, baseline_tokens_path: Path,
                 pinned_assets: dict[str, str] | None = None) -> dict[str, Any]:
    raw = {"probe": probe_path.read_bytes(), "protocol": protocol_path.read_bytes(),
           "heldout": heldout_path.read_bytes(), "baseline_tokens": baseline_tokens_path.read_bytes()}
    probe = json.loads(raw["probe"])
    for name in ("protocol", "heldout"):
        if probe[name + "_sha256"] != _sha(raw[name]):
            raise ValueError(f"Saved {name} hash does not match the supplied source")
    sealed = json.loads(raw["heldout"])
    observed = [case for case in probe["cases"] if case["set"] == "heldout"]
    if len(sealed) != len(observed) or any(
            any(case[key] != entry[key] for key in ("set", "level", "target", "prompt"))
            for entry, case in zip(sealed, observed)):
        raise ValueError("Saved held-out prompts/order differ from the sealed inputs")

    assets = verify_tokenizer_assets(model_dir, pinned_assets)
    tokenizer_config = json.loads((model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    generation_config = json.loads((model_dir / "generation_config.json").read_text(encoding="utf-8"))
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    template_source = (model_dir / "chat_template.jinja").read_text(encoding="utf-8")
    specials = _special_tokens(tokenizer_config)
    tokenizer = _load_tokenizer(model_dir)

    baseline_ids = json.loads(raw["baseline_tokens"])
    rendered_baseline = render_chat_prompt(template_source, specials, BASELINE_PROMPT)
    recovered_baseline = tokenize_prompt(tokenizer, rendered_baseline)
    baseline_conformance = (len(baseline_ids) == 1 and recovered_baseline == baseline_ids[0])

    harness_path = ROOT / HARNESS_FILE
    harness_sha = _sha(harness_path.read_bytes()) if harness_path.exists() else None

    cases = []
    for index, case in enumerate(probe["cases"]):
        rendered = render_chat_prompt(template_source, specials, case["prompt"])
        ids = tokenize_prompt(tokenizer, rendered)
        if not ids or ids[0] != generation_config.get("bos_token_id"):
            raise ValueError(f"Case {index} does not begin with the pinned BOS token")
        digit_steps = [
            {"step": record["step"],
             "context_token_count_at_decision": len(ids) + record["step"],
             "decision_score_selects_token_index": record["step"]}
            for record in case["generation"].get("digit_token_top5", [])
        ]
        cases.append({
            "source_case_index": index,
            "set": case["set"],
            "level": case["level"],
            "target": case["target"],
            "recorded_score_class": case["score_class"],
            "prompt_utf8_sha256": _sha(case["prompt"].encode("utf-8")),
            "rendered_chat_text": rendered,
            "rendered_utf8_sha256": _sha(rendered.encode("utf-8")),
            "prompt_token_ids": ids,
            "prompt_token_count": len(ids),
            "digit_decision_contexts": digit_steps,
        })

    body = {
        "schema_version": 1,
        "kind": "offline_acquisition_binding_recovery",
        "scope": "Recovery of exact prompt token IDs and effective decode/cache "
                 "settings for the saved copy/drift probe from hash-pinned assets. "
                 "No model inference, numerical replay, or causal claim.",
        "sources": {
            "probe": {"filename": probe_path.name, "sha256": _sha(raw["probe"])},
            "protocol": {"filename": protocol_path.name, "sha256": _sha(raw["protocol"])},
            "heldout": {"filename": heldout_path.name, "sha256": _sha(raw["heldout"])},
            "baseline_tokens": {"filename": baseline_tokens_path.name,
                                 "sha256": _sha(raw["baseline_tokens"])},
        },
        "module_sha256": _sha(Path(__file__).read_bytes()),
        "tokenizer_assets": assets,
        "tokenizer_pipeline": {
            "tokenizer_library": "tokenizers (Rust); transformers GemmaTokenizerFast equivalent",
            "pinned_package_versions": {"transformers": "4.53.3", "tokenizers": "0.21.4"},
            "template_rendering": "ImmutableSandboxedEnvironment, trim_blocks/lstrip_blocks, "
                                  "single user message, add_generation_prompt=True",
            "encode": "add_special_tokens=False (special tokens are template literals)",
        },
        "decode_settings": decode_settings(generation_config, config, harness_sha),
        "baseline_conformance": {
            "baseline_prompt": BASELINE_PROMPT,
            "rendered_chat_text": rendered_baseline,
            "recovered_token_ids": recovered_baseline,
            "frozen_token_ids": baseline_ids[0] if baseline_ids else None,
            "token_ids_reproduced": baseline_conformance,
        },
        "model_inference_performed": False,
        "case_count": len(cases),
        "cases": cases,
        "limitations": [
            "Token IDs are recovered deterministically from byte-exact assets; this does not re-execute or re-attest the original forwards.",
            "Recorded top-five scores came from the cached hybrid decode path; the independent engine's full-recompute path is not established equivalent.",
            "Effective settings are recovered from pinned config bytes and harness source, not from a new runtime observation.",
            "Prompt recovery covers the saved copy/drift cases only; it does not widen the frozen S=30 execution profile.",
        ],
    }
    if not baseline_conformance:
        raise ValueError("Retokenization does not reproduce the frozen baseline token IDs")
    body["report_sha256"] = _sha(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                          allow_nan=False).encode("utf-8"))
    return body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Directory containing the pinned tokenizer/config assets")
    parser.add_argument("--probe", type=Path, default=ROOT / "results/gemma3_270m_copy_drift_v1.json")
    parser.add_argument("--protocol", type=Path, default=ROOT / "results/gemma3_270m_copy_drift_protocol_v1.md")
    parser.add_argument("--heldout", type=Path, default=ROOT / "results/gemma3_270m_copy_drift_heldout_v1.json")
    parser.add_argument("--baseline-tokens", type=Path,
                        default=ROOT / "results/gemma3_270m_independent_baseline_tokens.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = build_report(args.model_dir, args.probe, args.protocol,
                              args.heldout, args.baseline_tokens)
        text = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(text)
        else:
            print(text, end="")
    except (OSError, ValueError, KeyError, TypeError, IndexError, RuntimeError) as error:
        parser.exit(1, f"Acquisition-binding recovery rejected: {error}\n")


if __name__ == "__main__":
    main()
