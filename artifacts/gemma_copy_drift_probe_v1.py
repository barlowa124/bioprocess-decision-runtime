"""Copy-or-drift probe v1 (step 2 of the writing-failure plan).

Implements results/gemma3_270m_copy_drift_protocol_v1.md exactly.

Phases:
  dev      run the 10 development cases
  seal     render the 17 held-out prompts and write the sealed JSON
  heldout  verify the sealed file hash, run the 17 held-out cases,
           write results/gemma3_270m_copy_drift_v1.{json,md}

Greedy decoding throughout; seed is irrelevant. The harness may not be
edited between `seal` and `heldout`; if it is, reseal and rerun.
"""
from __future__ import annotations

import os

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")  # triton absent on Windows

import hashlib
import json
import random
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / ".models/gemma-3-270m-it"
RESULTS_JSON = Path("C:/oncology-coscientist/results/luad/3097990b11d8/results.json")
PROTOCOL_MD = ROOT / "results/gemma3_270m_copy_drift_protocol_v1.md"
HELDOUT_JSON = ROOT / "results/gemma3_270m_copy_drift_heldout_v1.json"
OUT_JSON = ROOT / "results/gemma3_270m_copy_drift_v1.json"
OUT_MD = ROOT / "results/gemma3_270m_copy_drift_v1.md"

MAX_NEW_TOKENS = 400

MODELS = ["cox/clinical", "rsf/clinical",
          "cox/clinical_expression", "rsf/clinical_expression"]
METRICS = [("harrell_c", "Harrell C"), ("uno_c", "Uno C"),
           ("auc_12m", "AUC at 12 months"), ("auc_24m", "AUC at 24 months"),
           ("auc_36m", "AUC at 36 months"),
           ("integrated_brier_6_36m", "integrated Brier score (6-36 months)")]
LABEL = dict(METRICS)

T1 = ("cox/clinical", "harrell_c")
T2 = ("rsf/clinical_expression", "uno_c")

_DEC_RE = re.compile(r"\b0\.\d{1,4}\b")


def ground_truth() -> dict:
    r = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    return {m: {k: round(r["models"][m]["metrics"][k], 3) for k, _ in METRICS}
            for m in MODELS}


def _other_model(model: str) -> str:
    """L2 distractor: same suffix, other prefix (e.g. cox/clinical -> rsf/clinical)."""
    prefix, suffix = model.split("/")
    other_prefix = "rsf" if prefix == "cox" else "cox"
    return f"{other_prefix}/{suffix}"


def render_prompt(level: str, target: tuple[str, str] | None, gt: dict) -> str:
    if level == "L5":
        j = json.dumps(gt)
        return (f"Recorded results (JSON): {j}\n"
                "Write a markdown table with one row per model and one column "
                "per metric, using exactly the recorded values. No other text.")
    model, metric = target
    label, value = LABEL[metric], f"{gt[model][metric]:.3f}"
    if level == "L1":
        return (f"Recorded result: the {label} for {model} is {value}.\n"
                f"State the {label} for {model}. Answer with the number only.")
    if level == "L2":
        other = _other_model(model)
        v2 = f"{gt[other][metric]:.3f}"
        return (f"Recorded results:\n"
                f"- {model}: {label} {value}\n"
                f"- {other}: {label} {v2}\n"
                f"State the {label} for {model}. Answer with the number only.")
    if level == "L3":
        rows = "\n".join(f"- {m}: {label} {gt[m][metric]:.3f}" for m in MODELS)
        return (f"Recorded results:\n{rows}\n"
                f"State the {label} for {model}. Answer with the number only.")
    if level == "L4":
        j = json.dumps(gt)
        return (f"Recorded results (JSON): {j}\n"
                f"State the {label} for {model}. Answer with the number only.")
    raise ValueError(level)


def heldout_pairs() -> list[tuple[str, str]]:
    all_pairs = sorted(f"{m}|{k}" for m in MODELS for k, _ in METRICS)
    excl = {f"{T1[0]}|{T1[1]}", f"{T2[0]}|{T2[1]}"}
    drawn = random.Random(20240601).sample([p for p in all_pairs if p not in excl], 4)
    return [tuple(p.split("|")) for p in drawn]


def heldout_cases(gt: dict) -> list[dict]:
    cases = [{"level": lv, "target": list(t), "set": "heldout"}
             for t in heldout_pairs() for lv in ("L1", "L2", "L3", "L4")]
    cases.append({"level": "L5", "target": None, "set": "heldout"})
    for c in cases:
        c["prompt"] = render_prompt(c["level"], c["target"], gt)
    return cases


def dev_cases(gt: dict) -> list[dict]:
    cases = [{"level": lv, "target": list(t), "set": "dev"}
             for t in (T1, T2) for lv in ("L1", "L2", "L3", "L4")]
    cases += [{"level": "L5", "target": None, "set": "dev"} for _ in range(2)]
    for c in cases:
        c["prompt"] = render_prompt(c["level"], c["target"], gt)
    return cases


# ---------- generation ----------

_model = _tok = None


def _load():
    global _model, _tok
    if _model is not None:
        return _model, _tok
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    free = torch.cuda.mem_get_info()[0] / 1e9 if torch.cuda.is_available() else 0
    device = "cuda" if free >= 4.0 else "cpu"
    _tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    _model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), torch_dtype=torch.bfloat16,
        attn_implementation="eager").to(device)
    _model.eval()
    _model._probe_device = device
    return _model, _tok


def generate(prompt: str) -> dict:
    import torch
    model, tok = _load()
    messages = [{"role": "user", "content": prompt}]
    inputs = tok.apply_chat_template(messages, add_generation_prompt=True,
                                     return_tensors="pt").to(model._probe_device)
    t0 = time.monotonic()
    with torch.no_grad():
        out = model.generate(inputs, max_new_tokens=MAX_NEW_TOKENS,
                             do_sample=False, return_dict_in_generate=True,
                             output_scores=True)
    gen_ids = out.sequences[0, inputs.shape[1]:].tolist()
    pieces = [tok.decode([t]) for t in gen_ids]
    text = "".join(pieces)
    # strip gemma turn-end control tokens from the recorded text
    text = text.split("<end_of_turn>")[0].rstrip()
    digit_steps = []
    for i, (tid, piece) in enumerate(zip(gen_ids, pieces)):
        if any(ch.isdigit() for ch in piece):
            top = torch.topk(out.scores[i][0], 5)
            digit_steps.append({
                "step": i, "token_id": tid, "token": piece,
                "top5": [{"token": tok.decode([t.item()]),
                          "logit": round(s.item(), 4)}
                         for t, s in zip(top.indices, top.values)]})
    return {"generated_text": text, "generated_token_ids": gen_ids,
            "token_pieces": pieces,
            "digit_token_top5": digit_steps,
            "elapsed_s": round(time.monotonic() - t0, 1)}


# ---------- scoring ----------

def _decimals(text: str) -> list[float]:
    return [float(m.group(0)) for m in _DEC_RE.finditer(text)]


def score(case: dict, gen: dict, gt: dict) -> dict:
    R = [gt[m][k] for m in MODELS for k, _ in METRICS]
    nums = _decimals(gen["generated_text"])
    in_R = [n for n in nums if any(abs(n - r) <= 0.0005 for r in R)]
    drift = [n for n in nums if not any(abs(n - r) <= 0.0005 for r in R)]
    level = case["level"]
    asked = (gt[case["target"][0]][case["target"][1]]
             if case["target"] else None)
    asked_present = (asked is not None
                     and any(abs(n - asked) <= 0.0005 for n in nums))

    rec = {"score_class": None, "asked_value": asked, "extracted": nums,
           "drifting_decimals": drift, "first_departure": None}

    if drift:
        rec["score_class"] = "drift"
        rec["first_departure"] = first_departure(case, gen, asked, drift[0])
    elif level == "L5":
        present = sum(1 for r in R if any(abs(n - r) <= 0.0005 for n in nums))
        rec["n_present"] = present
        rec["score_class"] = "copy" if present == 24 else (
            "partial" if present else "omission")
    elif asked_present:
        rec["score_class"] = "copy"
    elif in_R:
        rec["score_class"] = "misattribution"
    else:
        rec["score_class"] = "omission"
    return rec


def first_departure(case: dict, gen: dict, asked: float | None,
                    drifted: float) -> dict:
    """First generated digit token where the emitted number's digit string
    departs from the asked value's digit string."""
    text = gen["generated_text"]
    pieces = gen["token_pieces"]
    drift_str = f"{drifted:g}"
    if asked is None:
        # L5: depart relative to any recorded value — find first differing digit
        asked_str = None
    else:
        asked_str = f"{asked:.3f}"
    m = re.search(r"(?<!\d)0?\." + re.escape(drift_str.split(".")[1]), text) \
        or re.search(r"(?<!\d)" + re.escape(drift_str), text)
    if m is None:
        return {"note": "drifting decimal not located in text", "drifted": drifted}
    start = m.start()
    # map char offset -> token index
    pos = 0
    tok_idx = None
    for i, p in enumerate(pieces):
        if pos <= start < pos + len(p):
            tok_idx = i
            break
        pos += len(p)
    if tok_idx is None:
        return {"note": "token span not found", "drifted": drifted}
    # walk from the number's first token; find first digit that differs
    emitted = re.match(r"0?\.\d+", text[start:]).group(0)
    if asked_str:
        k = next((i for i, (a, b) in enumerate(
            zip(emitted, asked_str)) if a != b), min(len(emitted), len(asked_str)))
        dep_char = start + k
        pos = 0
        for i, p in enumerate(pieces):
            if pos <= dep_char < pos + len(p) and any(
                    c.isdigit() for c in p):
                tok_idx = i
                break
            pos += len(p)
    rec = {"drifted": drifted, "emitted": emitted, "asked": asked,
           "token_step": tok_idx, "token": pieces[tok_idx]}
    ds = next((d for d in gen["digit_token_top5"] if d["step"] == tok_idx), None)
    if ds:
        rec["top5"] = ds["top5"]
        if asked_str:
            asked_digits = [c for c in asked_str if c.isdigit()]
            rec["correct_digit_in_top5"] = any(
                t["token"].strip().lstrip("0.").rstrip() and
                asked_digits and t["token"].strip()[-1:] == asked_digits[0]
                for t in ds["top5"])
    return rec


# ---------- phases ----------

def _run_cases(cases: list[dict], gt: dict) -> list[dict]:
    for i, c in enumerate(cases):
        gen = generate(c["prompt"])
        c["generation"] = {k: v for k, v in gen.items() if k != "token_pieces"}
        c.update(score(c, gen, gt))
        print(f"[{i + 1}/{len(cases)}] {c['set']} {c['level']} "
              f"{c.get('target')} -> {c['score_class']}")
    return cases


def _env_record() -> dict:
    import torch
    import transformers
    return {"device": _model._probe_device,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "dtype": "bfloat16", "attn_implementation": "eager",
            "decode": "greedy", "max_new_tokens": MAX_NEW_TOKENS,
            "model_file_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                                  for f in sorted(MODEL_DIR.iterdir()) if f.is_file()}}


def main() -> None:
    import sys
    phase = sys.argv[1]
    gt = ground_truth()

    if phase == "dev":
        cases = _run_cases(dev_cases(gt), gt)
        OUT_JSON.write_text(json.dumps({
            "protocol_sha256": hashlib.sha256(PROTOCOL_MD.read_bytes()).hexdigest(),
            "heldout_sha256": None,
            "environment": _env_record(),
            "cases": cases}, indent=2, default=str) + "\n", encoding="utf-8")
        print("wrote", OUT_JSON)

    elif phase == "seal":
        cases = heldout_cases(gt)
        HELDOUT_JSON.write_text(json.dumps(cases, indent=2) + "\n",
                                encoding="utf-8")
        print("sealed sha256:",
              hashlib.sha256(HELDOUT_JSON.read_bytes()).hexdigest())

    elif phase == "heldout":
        sealed = json.loads(HELDOUT_JSON.read_text(encoding="utf-8"))
        sha = hashlib.sha256(HELDOUT_JSON.read_bytes()).hexdigest()
        print("heldout sha256:", sha)
        cases = _run_cases(sealed, gt)
        dev = json.loads(OUT_JSON.read_text())["cases"] if OUT_JSON.exists() else []
        OUT_JSON.write_text(json.dumps({
            "protocol_sha256": hashlib.sha256(PROTOCOL_MD.read_bytes()).hexdigest(),
            "heldout_sha256": sha,
            "environment": _env_record(),
            "cases": dev + cases}, indent=2, default=str) + "\n",
            encoding="utf-8")
        write_md(dev + cases)
        print("wrote", OUT_JSON, "and", OUT_MD)
    else:
        raise SystemExit(f"unknown phase {phase}")


def write_md(cases: list[dict]) -> None:
    lines = ["# Copy-or-drift probe v1 — gemma-3-270m-it", "",
             "| set | level | target | class | asked | first drifting decimal |",
             "|---|---|---|---|---|---|"]
    for c in cases:
        tgt = "/".join(c["target"]) if c["target"] else "all"
        asked = f"{c['asked_value']:.3f}" if c["asked_value"] is not None else "—"
        drift = (str(c["drifting_decimals"][0]) if c["drifting_decimals"] else "—")
        lines.append(f"| {c['set']} | {c['level']} | {tgt} | "
                     f"{c['score_class']} | {asked} | {drift} |")
    lines += ["", "## Per-level summary", "",
              "| set | level | copy | drift | omission | misattribution | partial |",
              "|---|---|---|---|---|---|---|"]
    for s in ("dev", "heldout"):
        for lv in ("L1", "L2", "L3", "L4", "L5"):
            sub = [c for c in cases if c["set"] == s and c["level"] == lv]
            if not sub:
                continue
            cnt = {k: sum(1 for c in sub if c["score_class"] == k)
                   for k in ("copy", "drift", "omission", "misattribution", "partial")}
            lines.append(f"| {s} | {lv} | {cnt['copy']} | {cnt['drift']} | "
                         f"{cnt['omission']} | {cnt['misattribution']} | {cnt['partial']} |")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
