"""Attention ablation at wrong-digit decisions — CPU causal-structure test.

At the recorded wrong-digit decision step, zeroes the post-softmax attention
weights of selected (layer, head) pairs toward the emitted value's token
span in a manual full-recompute decode, and records whether the argmax
flips to the correct digit. Control condition ablates toward the asked
value's span instead (the wrong digit should persist).

This intervenes on the decode path only (no weight edits); CPU backend;
bounded causal-structure evidence about a behavioral mechanism, not a proof.

Usage: python artifacts/gemma_copy_drift_attention_ablation_v1.py <model_dir>
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROBE_JSON = ROOT / "results/gemma3_270m_copy_drift_v1.json"
COUNTER_JSON = ROOT / "results/gemma3_270m_copy_drift_counterfactual_v1.json"
CAPTURE_JSON = ROOT / "results/gemma3_270m_copy_drift_attention_capture_v1.json"
OUT_JSON = ROOT / "results/gemma3_270m_copy_drift_attention_ablation_v1.json"

PINNED_MODEL_SHA256 = "700b710a9a99c295ed546647aa81cacf9f81f4c573ea2be613a0e2517a44afab"
DECISION_STEP = {"c12": 4, "c16": 3}
NEW_TOKENS = 6

CASES = {
    "c12": {"prompt_key": "c12_baseline", "level": "L3",
            "owner_value": "0.647", "asked_value": "0.643",
            "asked_entity": "cox/clinical_expression",
            "expected_emitted_digit": "7", "expected_asked_digit": "3"},
    "c16": {"prompt_key": "c16_baseline", "level": "L3",
            "owner_value": "0.647", "asked_value": "0.632",
            "asked_entity": "rsf/clinical_expression",
            "expected_emitted_digit": "4", "expected_asked_digit": "3"},
}

CONDITIONS = [
    {"name": "none", "layers": [], "heads": None, "target": None},
    {"name": "L5_all_emit", "layers": [5], "heads": None, "target": "emit"},
    {"name": "L5H1_emit", "layers": [5], "heads": [1], "target": "emit"},
    {"name": "L11_all_emit", "layers": [11], "heads": None, "target": "emit"},
    {"name": "L5_L11_all_emit", "layers": [5, 11], "heads": None, "target": "emit"},
    {"name": "all_layers_emit", "layers": list(range(18)), "heads": None, "target": "emit"},
    {"name": "L5H2_ask_control", "layers": [5], "heads": [2], "target": "ask"},
    {"name": "L5_L11_all_ask_control", "layers": [5, 11], "heads": None, "target": "ask"},
]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _value_span_tokens(pieces: list[str], prompt: str, row_entity: str,
                       value: str, prefix_chars: int) -> list[int]:
    """Token indices covering the value string inside the row for row_entity."""
    m = re.search(r"- " + re.escape(row_entity) + r": [^\n]*?(" + re.escape(value) + r")", prompt)
    if m is None:
        raise ValueError(f"value {value} not located in row for {row_entity}")
    lo, hi = m.start(1), m.end(1)
    out, acc = [], 0
    for i, piece in enumerate(pieces):
        plo, phi = acc - prefix_chars, acc + len(piece) - prefix_chars
        if plo < hi and phi > lo:
            out.append(i)
        acc += len(piece)
    return out


def main() -> None:
    model_dir = Path(sys.argv[1])
    if OUT_JSON.exists():
        raise SystemExit(f"refusing to overwrite {OUT_JSON}")
    weights_sha = _sha((model_dir / "model.safetensors").read_bytes())
    if weights_sha != PINNED_MODEL_SHA256:
        raise SystemExit("checkpoint hash does not match the pinned acquisition checkpoint")

    import torch
    import torch.nn.functional as F
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.gemma3 import modeling_gemma3 as g3

    counter = {c["case_id"]: c for c in
               json.loads(COUNTER_JSON.read_text(encoding="utf-8"))["cases"]}
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=torch.bfloat16,
        attn_implementation="eager").to("cpu")
    model.eval()

    # intervention state consulted by the patched attention path
    state = {"active": False, "layers": set(), "heads": None, "positions": None}
    repeat_kv = g3.repeat_kv

    def patched_eager(module, query, key, value, attention_mask,
                      dropout=0.0, scaling=None, **kwargs):
        # replica of transformers 4.53.3 eager_attention_forward plus ablation
        if scaling is None:
            scaling = module.head_dim ** -0.5
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)
        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
        if state["active"] and module.layer_idx in state["layers"]:
            heads = state["heads"] if state["heads"] is not None \
                else range(attn_weights.shape[1])
            for h in heads:
                attn_weights[0, h, -1, state["positions"]] = 0.0
        attn_output = torch.matmul(attn_weights, value_states)
        return attn_output.transpose(1, 2).contiguous(), attn_weights

    original = g3.eager_attention_forward
    g3.eager_attention_forward = patched_eager
    prefix = "<bos><start_of_turn>user\n"
    results = []
    try:
        for case_name, spec in CASES.items():
            prompt = counter[spec["prompt_key"]]["prompt"]
            inputs = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True, return_tensors="pt")
            pieces = [tokenizer.decode([t]) for t in inputs[0].tolist()]
            row_entities = re.findall(r"- (\S+):", prompt)
            emit_pos = _value_span_tokens(pieces, prompt, row_entities[0],
                                          spec["owner_value"], len(prefix))
            asked_pos = _value_span_tokens(pieces, prompt, spec["asked_entity"],
                                           spec["asked_value"], len(prefix))
            step = DECISION_STEP[case_name]

            for cond in CONDITIONS:
                state.update(active=False, layers=set(cond["layers"]),
                             heads=cond["heads"],
                             positions=emit_pos if cond["target"] == "emit" else asked_pos)
                context = inputs.clone()
                gen_ids, steps_top5 = [], []
                with torch.no_grad():
                    for s in range(NEW_TOKENS):
                        state["active"] = bool(cond["layers"]) and s == step
                        logits = model(input_ids=context, use_cache=False).logits[0, -1]
                        top = torch.topk(logits, 5)
                        steps_top5.append(
                            [{"token": tokenizer.decode([t.item()]),
                              "logit": round(float(v.item()), 4)}
                             for t, v in zip(top.indices, top.values)])
                        nxt = int(torch.argmax(logits))
                        gen_ids.append(nxt)
                        context = torch.cat(
                            [context, torch.tensor([[nxt]], dtype=context.dtype)], dim=1)
                state["active"] = False
                results.append({
                    "case": case_name, "condition": cond["name"],
                    "ablated_layers": cond["layers"], "ablated_heads": cond["heads"],
                    "ablated_span_label": cond["target"],
                    "ablated_token_positions": state["positions"],
                    "decision_step": step,
                    "generated_token_ids": gen_ids,
                    "generated_text": tokenizer.decode(gen_ids),
                    "top5_at_decision": steps_top5[step],
                    "selected_token_id_at_decision": gen_ids[step],
                    "expected_emitted_digit": spec["expected_emitted_digit"],
                    "expected_asked_digit": spec["expected_asked_digit"],
                })
                print(f"{case_name} {cond['name']}: {tokenizer.decode(gen_ids)!r} "
                      f"top1={steps_top5[step][0]}", flush=True)
    finally:
        g3.eager_attention_forward = original

    body = {
        "schema_version": 1,
        "kind": "cpu_attention_ablation_at_decision",
        "scope": "Post-softmax attention ablation toward emitted/asked value token "
                 "spans at the recorded wrong-digit decision step, manual "
                 "full-recompute greedy decode on CPU. Bounded causal-structure "
                 "evidence: it tests whether attention to the source span is "
                 "necessary for the wrong selection under this backend/path.",
        "sources": {
            "probe": {"filename": PROBE_JSON.name, "sha256": _sha(PROBE_JSON.read_bytes())},
            "counterfactual": {"filename": COUNTER_JSON.name,
                                "sha256": _sha(COUNTER_JSON.read_bytes())},
            "attention_capture": {"filename": CAPTURE_JSON.name,
                                   "sha256": _sha(CAPTURE_JSON.read_bytes())},
            "harness": {"filename": Path(__file__).name,
                        "sha256": _sha(Path(__file__).read_bytes())},
        },
        "binding": {"model_safetensors_sha256": weights_sha, "backend": "cpu",
                     "dtype": "bfloat16", "attn_implementation": "eager",
                     "torch_version": torch.__version__,
                     "transformers_version": transformers.__version__,
                     "decode_path": "manual full-context recompute, use_cache=False",
                     "intervention": "post-softmax attention weights zeroed at the "
                                     "decision step's last query row; no renormalization"},
        "results": results,
        "limitations": [
            "Ablating post-softmax mass without renormalization attenuates the head's output magnitude as well as its direction; a flip shows necessity of the attended span under this intervention semantics.",
            "Single-token-span ablation in selected heads; distributed redundancy can mask necessity.",
            "CPU full-recompute path only; cached hybrid and CUDA paths are unobserved.",
        ],
    }
    body["report_sha256"] = _sha(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                          allow_nan=False).encode("utf-8"))
    OUT_JSON.write_text(json.dumps(body, indent=2, allow_nan=False) + "\n",
                        encoding="utf-8")
    print("wrote", OUT_JSON)


if __name__ == "__main__":
    main()
