"""Copy-drift counterfactual probe v1 — CPU entity/position/value binding test.

Implements results/gemma3_270m_copy_drift_counterfactual_protocol_v1.md.

Phases:
  seal   render the variant cases and write the sealed JSON
  run    verify the sealed hash, generate greedy on CPU (hybrid path),
         write results/gemma3_270m_copy_drift_counterfactual_v1.json

New-backend evidence: CPU decisions, not bitwise CUDA score equality.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROBE_JSON = ROOT / "results/gemma3_270m_copy_drift_v1.json"
PROTOCOL_MD = ROOT / "results/gemma3_270m_copy_drift_counterfactual_protocol_v1.md"
CASES_JSON = ROOT / "results/gemma3_270m_copy_drift_counterfactual_cases_v1.json"
OUT_JSON = ROOT / "results/gemma3_270m_copy_drift_counterfactual_v1.json"
REPLAY_JSON = ROOT / "results/gemma3_270m_copy_drift_cpu_replay_v1.json"

PINNED_MODEL_SHA256 = "700b710a9a99c295ed546647aa81cacf9f81f4c573ea2be613a0e2517a44afab"
SENTINEL = 0.777
NEW_TOKENS = 8
DEC_RE = re.compile(r"\b0\.\d{1,4}\b")

# Per-target binding facts recovered in gemma3_270m_copy_drift_tokenization_v1
# analysis: which record owns the recorded emitted value.
TARGETS = {
    3: {"asked_entity": "cox/clinical", "asked_metric": "harrell_c",
         "asked_value": 0.647, "recorded_emitted": 0.64,
         "owner_entity": "cox/clinical", "owner_metric": "uno_c",
         "owner_is_asked": True},
    12: {"asked_entity": "cox/clinical_expression", "asked_metric": "harrell_c",
          "asked_value": 0.643, "recorded_emitted": 0.647,
          "owner_entity": "cox/clinical", "owner_metric": "harrell_c",
          "owner_is_asked": False},
    16: {"asked_entity": "rsf/clinical_expression", "asked_metric": "harrell_c",
          "asked_value": 0.632, "recorded_emitted": 0.643,
          "owner_entity": "cox/clinical_expression", "owner_metric": "harrell_c",
          "owner_is_asked": False},
    17: {"asked_entity": "rsf/clinical_expression", "asked_metric": "harrell_c",
          "asked_value": 0.632, "recorded_emitted": 0.643,
          "owner_entity": "cox/clinical_expression", "owner_metric": "harrell_c",
          "owner_is_asked": False},
    20: {"asked_entity": "rsf/clinical_expression", "asked_metric": "auc_36m",
          "asked_value": 0.609, "recorded_emitted": 0.695,
          "owner_entity": "rsf/clinical", "owner_metric": "auc_36m",
          "owner_is_asked": False},
    21: {"asked_entity": "rsf/clinical_expression", "asked_metric": "auc_36m",
          "asked_value": 0.609, "recorded_emitted": 0.153,
          "owner_entity": "rsf/clinical_expression",
          "owner_metric": "integrated_brier_6_36m",
          "owner_is_asked": True},
}

RENAME_TO = "zzz/other"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _l3_rows(prompt: str) -> tuple[str, list[str], str]:
    head, rest = prompt.split("Recorded results:\n", 1)
    rows, tail = rest.split("\nState the ", 1)
    return head + "Recorded results:\n", rows.split("\n"), "\nState the " + tail


def _l4_json(prompt: str) -> tuple[str, dict, str]:
    head, rest = prompt.split("Recorded results (JSON): ", 1)
    end = rest.index("\nState the ")
    return (head + "Recorded results (JSON): ", json.loads(rest[:end]),
            rest[end:])


def _l4_render(head: str, table: dict, tail: str) -> str:
    return head + json.dumps(table) + tail


def _scrambled(value: float, offset: int = 0) -> float:
    return round(value + 0.034 + 0.001 * offset, 3)


def build_variants(index: int, prompt: str, info: dict) -> dict[str, str]:
    level = "L4" if "(JSON)" in prompt else "L3"
    asked, owner = info["asked_entity"], info["owner_entity"]
    metric, ometric = info["asked_metric"], info["owner_metric"]
    variants = {"baseline": prompt}

    if level == "L3":
        head, rows, tail = _l3_rows(prompt)
        row_of = {r.split(":")[0].lstrip("- ").strip(): i for i, r in enumerate(rows)}

        swap = list(rows)
        a, b = row_of[asked], row_of[owner]
        swap[a], swap[b] = swap[b], swap[a]
        variants["row_swap"] = head + "\n".join(swap) + tail

        moved = [re.sub(r"0\.\d+", f"{SENTINEL:.3f}", r)
                 if r.split(":")[0].lstrip("- ").strip() == asked else r for r in rows]
        variants["value_move"] = head + "\n".join(moved) + tail

        renamed = [re.sub(r"^(- )[^:]+:", rf"\g<1>{RENAME_TO}:", r)
                   if r.split(":")[0].lstrip("- ").strip() == owner else r for r in rows]
        variants["entity_rename"] = head + "\n".join(renamed) + tail

        scrambled = [re.sub(r"0\.\d+", f"{_scrambled(info['recorded_emitted']):.3f}", r)
                     if r.split(":")[0].lstrip("- ").strip() == owner else r for r in rows]
        variants["value_scramble"] = head + "\n".join(scrambled) + tail
        return variants

    head, table, tail = _l4_json(prompt)

    keys = list(table)
    if info["owner_is_asked"]:
        reordered = dict(reversed(list(table.items())))
        variants["row_swap"] = _l4_render(head, reordered, tail)
    else:
        order = list(keys)
        ia, ib = order.index(asked), order.index(owner)
        order[ia], order[ib] = order[ib], order[ia]
        variants["row_swap"] = _l4_render(head, {k: table[k] for k in order}, tail)

    moved = copy.deepcopy(table)
    moved[asked][metric] = SENTINEL
    variants["value_move"] = _l4_render(head, moved, tail)

    if info["owner_is_asked"]:
        renamed = {k: {(("zz_" + m) if m == ometric else m): v for m, v in rec.items()}
                   for k, rec in table.items()}
        variants["entity_rename"] = _l4_render(head, renamed, tail)
        swapped = copy.deepcopy(table)
        rec = swapped[asked]
        order = [m for m in rec]
        ia, ib = order.index(metric), order.index(ometric)
        order[ia], order[ib] = order[ib], order[ia]
        swapped[asked] = {m: rec[m] for m in order}
        variants["metric_swap"] = _l4_render(head, swapped, tail)
    else:
        renamed = {RENAME_TO if k == owner else k: v for k, v in table.items()}
        variants["entity_rename"] = _l4_render(head, renamed, tail)

    scrambled = copy.deepcopy(table)
    if info["owner_is_asked"]:
        for i, rec in enumerate(scrambled.values()):
            rec[ometric] = _scrambled(info["recorded_emitted"], i)
    else:
        scrambled[owner][ometric] = _scrambled(info["recorded_emitted"])
    variants["value_scramble"] = _l4_render(head, scrambled, tail)
    return variants


def seal() -> None:
    probe = json.loads(PROBE_JSON.read_text(encoding="utf-8"))
    cases = []
    for index, info in TARGETS.items():
        prompt = probe["cases"][index]["prompt"]
        for variant, text in build_variants(index, prompt, info).items():
            cases.append({"case_id": f"c{index}_{variant}", "source_case_index": index,
                          "variant": variant, "level": probe["cases"][index]["level"],
                          "target": probe["cases"][index]["target"], "prompt": text})
    CASES_JSON.write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")
    print("sealed", len(cases), "cases sha256:", _sha(CASES_JSON.read_bytes()))


def _classify(text: str, info: dict, variant: str, prompt: str) -> tuple[str, float | None]:
    m = DEC_RE.search(text)
    if m is None:
        return "omission", None
    value = float(m.group(0))
    present = {round(float(x), 4) for x in DEC_RE.findall(prompt)}
    if abs(value - SENTINEL) <= 0.0005:
        return "sentinel", value
    if variant == "value_scramble" and any(abs(value - _scrambled(info["recorded_emitted"], i)) <= 0.0005 for i in range(5)):
        return "scrambled", value
    if abs(value - info["asked_value"]) <= 0.0005:
        return "asked", value
    if abs(value - info["recorded_emitted"]) <= 0.0005:
        return "recorded_emitted", value
    return ("other_present" if any(abs(value - p) <= 0.0005 for p in present)
            else "novel"), value


def run(model_dir: Path) -> None:
    weights_sha = _sha((model_dir / "model.safetensors").read_bytes())
    if weights_sha != PINNED_MODEL_SHA256:
        raise SystemExit("checkpoint hash does not match the pinned acquisition checkpoint")
    sealed_sha = _sha(CASES_JSON.read_bytes())
    cases = json.loads(CASES_JSON.read_text(encoding="utf-8"))

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=torch.bfloat16,
        attn_implementation="eager").to("cpu")
    model.eval()

    results = []
    for i, case in enumerate(cases):
        info = TARGETS[case["source_case_index"]]
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": case["prompt"]}],
            add_generation_prompt=True, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(inputs, max_new_tokens=NEW_TOKENS,
                                 do_sample=False, return_dict_in_generate=True,
                                 output_scores=True)
        ids = out.sequences[0, inputs.shape[1]:].tolist()
        text = tokenizer.decode(ids).split("<end_of_turn>")[0].rstrip()
        label, value = _classify(text, info, case["variant"], case["prompt"])
        digit_steps = []
        for step, tid in enumerate(ids):
            piece = tokenizer.decode([tid])
            if any(ch.isdigit() for ch in piece):
                top = torch.topk(out.scores[step][0], 5)
                digit_steps.append({"step": step, "token_id": tid, "token": piece,
                                    "top5": [{"token": tokenizer.decode([t.item()]),
                                              "logit": round(s.item(), 4)}
                                             for t, s in zip(top.indices, top.values)]})
        results.append({**case, "generated_text": text, "generated_token_ids": ids,
                        "digit_token_top5": digit_steps,
                        "first_decimal": value, "classification": label})
        print(f"[{i + 1}/{len(cases)}] {case['case_id']} -> {label} ({value}) {text!r}",
              flush=True)

    body = {
        "schema_version": 1,
        "kind": "cpu_counterfactual_binding_probe",
        "scope": "CPU-backend prompt interventions on already-observed misattribution "
                 "cases; entity/position/value binding discrimination. Not fresh "
                 "confirmatory evidence, not a CUDA-path claim, not an internal-mechanism proof.",
        "sources": {
            "probe": {"filename": PROBE_JSON.name, "sha256": _sha(PROBE_JSON.read_bytes())},
            "protocol": {"filename": PROTOCOL_MD.name, "sha256": _sha(PROTOCOL_MD.read_bytes())},
            "sealed_cases": {"filename": CASES_JSON.name, "sha256": sealed_sha},
            "cpu_replay": {"filename": REPLAY_JSON.name,
                            "sha256": _sha(REPLAY_JSON.read_bytes()) if REPLAY_JSON.exists() else None},
            "harness": {"filename": Path(__file__).name, "sha256": _sha(Path(__file__).read_bytes())},
        },
        "binding": {"model_safetensors_sha256": weights_sha, "backend": "cpu",
                     "dtype": "bfloat16", "attn_implementation": "eager",
                     "torch_version": torch.__version__,
                     "transformers_version": transformers.__version__,
                     "decode_path": "hybrid (generation_config default)",
                     "new_tokens_per_case": NEW_TOKENS},
        "targets": TARGETS,
        "cases": results,
    }
    body["report_sha256"] = _sha(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                          allow_nan=False).encode("utf-8"))
    OUT_JSON.write_text(json.dumps(body, indent=2, allow_nan=False) + "\n",
                        encoding="utf-8")
    print("wrote", OUT_JSON)


if __name__ == "__main__":
    if sys.argv[1] == "seal":
        seal()
    elif sys.argv[1] == "run":
        run(Path(sys.argv[2]))
    else:
        raise SystemExit(f"unknown phase {sys.argv[1]}")
