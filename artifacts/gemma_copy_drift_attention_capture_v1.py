"""Attention capture at wrong-digit decisions — CPU interpretability evidence.

For each declared case, runs greedy decode on CPU with output_attentions=True
(eager) and records, at the digit-decision step, the attention distribution
of the last-position query over all context positions, aggregated over
semantically labelled prompt regions (record rows/keys, value spans, query).

Paired baseline vs "fixed" counterfactual variants test whether the
behaviourally identified binding (first-row position in L3 lists, entity
key in L4 JSON) is visible in attention mass.

New-backend (CPU) evidence; interpretability observation, not a proof of
mechanism. Usage: python artifacts/gemma_copy_drift_attention_capture_v1.py <model_dir>
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
OUT_JSON = ROOT / "results/gemma3_270m_copy_drift_attention_capture_v1.json"

PINNED_MODEL_SHA256 = "700b710a9a99c295ed546647aa81cacf9f81f4c573ea2be613a0e2517a44afab"
NEW_TOKENS = 8
DEC_RE = re.compile(r"0\.\d+")

# (label, prompt source, decision step or "last_digit", emitted-owner/asked facts)
CASES = [
    {"name": "c12_baseline_row1_binding", "prompt_from": ("counterfactual", "c12_baseline"),
     "decision_step": 4, "level": "L3",
     "asked_entity": "cox/clinical_expression", "asked_metric": "harrell_c",
     "owner_entity": "cox/clinical", "owner_metric": "harrell_c",
     "expected_emitted": "0.647", "expected_asked": "0.643"},
    {"name": "c12_row_swap_fixed", "prompt_from": ("counterfactual", "c12_row_swap"),
     "decision_step": 4, "level": "L3",
     "asked_entity": "cox/clinical_expression", "asked_metric": "harrell_c",
     "owner_entity": "cox/clinical", "owner_metric": "harrell_c",
     "expected_emitted": "0.643", "expected_asked": "0.643",
     "note": "asked entity moved to row 1; CPU emitted correctly"},
    {"name": "c16_baseline_row1_binding", "prompt_from": ("counterfactual", "c16_baseline"),
     "decision_step": 3, "level": "L3",
     "asked_entity": "rsf/clinical_expression", "asked_metric": "harrell_c",
     "owner_entity": "cox/clinical", "owner_metric": "harrell_c",
     "expected_emitted": "0.647", "expected_asked": "0.632",
     "note": "CPU emits row-1 value 0.647 (CUDA record emitted row-3 0.643)"},
    {"name": "c20_baseline_row1_binding", "prompt_from": ("counterfactual", "c20_baseline"),
     "decision_step": 3, "level": "L3",
     "asked_entity": "rsf/clinical_expression", "asked_metric": "auc_36m",
     "owner_entity": "cox/clinical", "owner_metric": "auc_36m",
     "expected_emitted": "0.694", "expected_asked": "0.609",
     "note": "CPU emits row-1 value 0.694 (CUDA record emitted row-2 0.695)"},
    {"name": "c17_baseline_key_binding", "prompt_from": ("counterfactual", "c17_baseline"),
     "decision_step": 3, "level": "L4",
     "asked_entity": "rsf/clinical_expression", "asked_metric": "harrell_c",
     "owner_entity": "cox/clinical_expression", "owner_metric": "harrell_c",
     "expected_emitted": "0.643", "expected_asked": "0.632"},
    {"name": "c17_entity_rename_fixed", "prompt_from": ("counterfactual", "c17_entity_rename"),
     "decision_step": 3, "level": "L4",
     "asked_entity": "rsf/clinical_expression", "asked_metric": "harrell_c",
     "owner_entity": "zzz/other", "owner_metric": "harrell_c",
     "expected_emitted": "0.632", "expected_asked": "0.632",
     "note": "owner renamed to zzz/other; CPU emitted correctly"},
    {"name": "c21_baseline_metric_value", "prompt_from": ("counterfactual", "c21_baseline"),
     "decision_step": 2, "level": "L4",
     "asked_entity": "rsf/clinical_expression", "asked_metric": "auc_36m",
     "owner_entity": "rsf/clinical_expression", "owner_metric": "integrated_brier_6_36m",
     "expected_emitted": "0.153", "expected_asked": "0.609"},
    {"name": "c3_baseline_cpu_correct", "prompt_from": ("counterfactual", "c3_baseline"),
     "decision_step": 4, "level": "L4",
     "asked_entity": "cox/clinical", "asked_metric": "harrell_c",
     "owner_entity": "cox/clinical", "owner_metric": "uno_c",
     "expected_emitted": "0.647", "expected_asked": "0.647",
     "note": "CPU answers correctly; CUDA truncated to 0.64"},
    {"name": "c13_l4_copy_control", "prompt_from": ("probe", 13),
     "decision_step": 4, "level": "L4",
     "asked_entity": "cox/clinical_expression", "asked_metric": "harrell_c",
     "owner_entity": "cox/clinical_expression", "owner_metric": "harrell_c",
     "expected_emitted": "0.643", "expected_asked": "0.643",
     "note": "heldout copy control: same metric as c17 but record 3 of 4, correct"},
    {"name": "c2_l3_row1_copy", "prompt_from": ("probe", 2),
     "decision_step": 4, "level": "L3",
     "asked_entity": "cox/clinical", "asked_metric": "harrell_c",
     "owner_entity": "cox/clinical", "owner_metric": "harrell_c",
     "expected_emitted": "0.647", "expected_asked": "0.647",
     "note": "dev L3 copy: asked row 1 itself — degenerate under row-1 binding"},
    {"name": "c24_l3_shared_value", "prompt_from": ("probe", 24),
     "decision_step": 4, "level": "L3",
     "asked_entity": "rsf/clinical_expression", "asked_metric": "integrated_brier_6_36m",
     "owner_entity": "rsf/clinical_expression", "owner_metric": "integrated_brier_6_36m",
     "expected_emitted": "0.153", "expected_asked": "0.153",
     "note": "heldout copy: asked value identical in all four rows — degenerate control"},
]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _spans_l3(prompt: str) -> list[dict]:
    spans = []
    for m in re.finditer(r"- (\S+): (.+?) (0\.\d+)\n", prompt + "\n"):
        entity, _, value = m.group(1), m.group(2), m.group(3)
        spans.append({"label": f"record:{entity}", "start": m.start(), "end": m.end() - 1})
        spans.append({"label": f"entity:{entity}", "start": m.start(1), "end": m.end(1)})
        spans.append({"label": f"value:{entity}", "start": m.start(3), "end": m.end(3)})
    q = prompt.index("State the ")
    spans.append({"label": "query", "start": q, "end": len(prompt)})
    qe = prompt.index("for ", q) + 4
    spans.append({"label": "query_entity", "start": qe,
                  "end": prompt.index(".", qe)})
    return spans


def _spans_l4(prompt: str) -> list[dict]:
    body = prompt.split("Recorded results (JSON): ", 1)[1]
    table, tail = body.split("\nState the ", 1)
    off = prompt.index(table)
    spans = [{"label": "query", "start": prompt.index("State the "), "end": len(prompt)}]
    for em in re.finditer(r'"([^"]+)": \{', table):
        entity = em.group(1)
        end = table.index("}", em.end()) + 1
        spans.append({"label": f"record:{entity}", "start": off + em.start(), "end": off + end})
        spans.append({"label": f"key:{entity}", "start": off + em.start(1), "end": off + em.end(1)})
        inner = table[em.end():end]
        for mm in re.finditer(r'"([^"]+)": (0\.\d+|\d+)', inner):
            spans.append({"label": f"key:{entity}:{mm.group(1)}",
                          "start": off + em.end() + mm.start(1),
                          "end": off + em.end() + mm.end(1)})
            spans.append({"label": f"value:{entity}:{mm.group(1)}",
                          "start": off + em.end() + mm.start(2),
                          "end": off + em.end() + mm.end(2)})
    q = prompt.index("State the ")
    qe = prompt.index("for ", q) + 4
    spans.append({"label": "query_entity", "start": qe, "end": prompt.index(".", qe)})
    return spans


def _token_spans(pieces: list[str], prefix_chars: int,
                 spans: list[dict]) -> dict[int, list[str]]:
    """Map char-offset spans (in prompt) to token indices via decoded pieces."""
    labels: dict[int, list[str]] = {}
    acc = 0
    for i, piece in enumerate(pieces):
        lo, hi = acc - prefix_chars, acc + len(piece) - prefix_chars
        for span in spans:
            if lo < span["end"] and hi > span["start"]:
                labels.setdefault(i, []).append(span["label"])
        acc += len(piece)
    return labels


def main() -> None:
    model_dir = Path(sys.argv[1])
    if OUT_JSON.exists():
        raise SystemExit(f"refusing to overwrite {OUT_JSON}")
    weights_sha = _sha((model_dir / "model.safetensors").read_bytes())
    if weights_sha != PINNED_MODEL_SHA256:
        raise SystemExit("checkpoint hash does not match the pinned acquisition checkpoint")

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    probe = json.loads(PROBE_JSON.read_text(encoding="utf-8"))
    counter = {c["case_id"]: c for c in
               json.loads(COUNTER_JSON.read_text(encoding="utf-8"))["cases"]}
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=torch.bfloat16,
        attn_implementation="eager").to("cpu")
    model.eval()

    prefix = "<bos><start_of_turn>user\n"
    captured = []
    for spec in CASES:
        source, key = spec["prompt_from"]
        prompt = (counter[key]["prompt"] if source == "counterfactual"
                  else probe["cases"][key]["prompt"])
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, return_tensors="pt")
        pieces = [tokenizer.decode([t]) for t in inputs[0].tolist()]
        spans = _spans_l4(prompt) if spec["level"] == "L4" else _spans_l3(prompt)
        token_labels = _token_spans(pieces, len(prefix), spans)

        with torch.no_grad():
            out = model.generate(inputs, max_new_tokens=NEW_TOKENS,
                                 do_sample=False, return_dict_in_generate=True,
                                 output_scores=True, output_attentions=True)
        ids = out.sequences[0, inputs.shape[1]:].tolist()
        text = tokenizer.decode(ids).split("<end_of_turn>")[0].rstrip()
        step = spec["decision_step"]
        all_pieces = pieces + [tokenizer.decode([t]) for t in ids[:step]]

        def region_mass(layer: int, head: int):
            attn = out.attentions[step][layer][0, head, -1]  # last query row [kv]
            masses: dict[str, float] = {}
            for pos, weight in enumerate(attn.tolist()):
                for label in token_labels.get(pos, ()):
                    masses[label] = masses.get(label, 0.0) + weight
            return masses, attn

        emitted_label = f"value:{spec['owner_entity']}:{spec['owner_metric']}" \
            if spec["level"] == "L4" else f"value:{spec['owner_entity']}"
        asked_label = f"value:{spec['asked_entity']}:{spec['asked_metric']}" \
            if spec["level"] == "L4" else f"value:{spec['asked_entity']}"
        first_record = "record:" + next(
            s["label"].split(":", 1)[1] for s in spans
            if s["label"].startswith("record:"))

        layers = []
        for layer in range(model.config.num_hidden_layers):
            heads = []
            for head in range(model.config.num_attention_heads):
                masses, attn = region_mass(layer, head)
                top = torch.topk(attn, 6)
                heads.append({
                    "head": head,
                    "emitted_value_mass": round(masses.get(emitted_label, 0.0), 5),
                    "asked_value_mass": round(masses.get(asked_label, 0.0), 5),
                    "emitted_record_mass": round(masses.get(f"record:{spec['owner_entity']}", 0.0), 5),
                    "asked_record_mass": round(masses.get(f"record:{spec['asked_entity']}", 0.0), 5),
                    "first_record_mass": round(masses.get(first_record, 0.0), 5),
                    "query_mass": round(masses.get("query", 0.0), 5),
                    "top_positions": [{"pos": int(p.item()), "piece": all_pieces[p.item()],
                                       "mass": round(float(w.item()), 5)}
                                      for p, w in zip(top.indices, top.values)],
                })
            layers.append({"layer": layer, "heads": heads})

        # per-layer head-summed masses for readability
        layer_summary = []
        for layer in layers:
            agg = {}
            for key in ("emitted_value_mass", "asked_value_mass", "emitted_record_mass",
                        "asked_record_mass", "first_record_mass", "query_mass"):
                agg[key] = round(sum(h[key] for h in layer["heads"]), 5)
            agg["layer"] = layer["layer"]
            layer_summary.append(agg)

        captured.append({
            "name": spec["name"], "note": spec.get("note"),
            "level": spec["level"], "decision_step": step,
            "prompt_token_count": len(pieces),
            "context_token_count_at_decision": len(pieces) + step,
            "generated_text": text, "generated_token_ids": ids,
            "emitted_first_decimal": (DEC_RE.search(text) or [None])[0] if DEC_RE.search(text) else None,
            "labels": {"emitted_value_span": emitted_label, "asked_value_span": asked_label,
                        "first_record_span": first_record},
            "token_label_index": {str(pos): labels for pos, labels in sorted(token_labels.items())
                                   if any(l.startswith(("value:", "key:", "entity:", "query")) for l in labels)},
            "layer_summary": layer_summary,
            "layers": layers,
        })
        em = captured[-1]["emitted_first_decimal"]
        print(f"{spec['name']}: step={step} emitted={em} text={text!r}", flush=True)

    body = {
        "schema_version": 1,
        "kind": "cpu_attention_capture_at_decision",
        "scope": "Attention-mass observation at wrong-digit decision steps on CPU "
                 "backend, aggregated over semantically labelled prompt regions. "
                 "Interpretability evidence about a behavioral mechanism; not a "
                 "proof, not fresh confirmatory holdout, not CUDA-path evidence.",
        "sources": {
            "probe": {"filename": PROBE_JSON.name, "sha256": _sha(PROBE_JSON.read_bytes())},
            "counterfactual": {"filename": COUNTER_JSON.name,
                                "sha256": _sha(COUNTER_JSON.read_bytes())},
            "harness": {"filename": Path(__file__).name,
                        "sha256": _sha(Path(__file__).read_bytes())},
        },
        "binding": {"model_safetensors_sha256": weights_sha, "backend": "cpu",
                     "dtype": "bfloat16", "attn_implementation": "eager",
                     "torch_version": torch.__version__,
                     "transformers_version": transformers.__version__,
                     "decode_path": "hybrid (generation_config default)",
                     "num_layers": model.config.num_hidden_layers,
                     "num_attention_heads": model.config.num_attention_heads},
        "cases": captured,
        "limitations": [
            "Attention mass is a correlational observation at the decision step, not a causal attribution; activation interventions remain future work.",
            "Sliding-window layers see the full context here (<512 tokens), so mass distributions are not truncated evidence.",
            "Region labels derive from prompt-string spans; a token overlapping two spans is counted under both.",
            "CPU backend only; CUDA-path attention is unobserved.",
        ],
    }
    body["report_sha256"] = _sha(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                          allow_nan=False).encode("utf-8"))
    OUT_JSON.write_text(json.dumps(body, indent=2, allow_nan=False) + "\n",
                        encoding="utf-8")
    print("wrote", OUT_JSON)


if __name__ == "__main__":
    main()
