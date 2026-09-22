"""Fabrication probe v1 (step 1 of the writing-failure plan).

Question: does gemma-3-270m-it under HF eager greedy decoding emit numeric
claims that do not match results.json, given the exact report-agent prompt
recorded in oncology run results/luad/3097990b11d8/agent/318d61643606
attempt 1? The same prompt is also sent to Ollama gemma3:270m for a
quantized-vs-HF comparison at equal size.

No sampling anywhere: do_sample=False (HF) and temperature=0 (Ollama).
"""
from __future__ import annotations

import os

# transformers 4.53.3 gemma3 modeling invokes torch.compile; triton is not
# available on Windows, so run pure eager
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import hashlib
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ONCOCS_ROOT = Path("C:/oncology-coscientist")
AGENT_RUN = ONCOCS_ROOT / "results/luad/3097990b11d8/agent/318d61643606/agent_run.json"
RESULTS_JSON = ONCOCS_ROOT / "results/luad/3097990b11d8/results.json"
SPLIT_JSON = ONCOCS_ROOT / "splits/luad.json"
MODEL_DIR = Path(__file__).resolve().parent.parent / ".models/gemma-3-270m-it"
OUT_JSON = Path(__file__).resolve().parent.parent / "results/gemma3_270m_fabrication_probe_v1.json"

QUESTION = ("Given the exact report-agent prompt from oncology run "
            "results/luad/3097990b11d8/agent/318d61643606 (attempt 1), does "
            "gemma-3-270m-it under HF eager greedy decoding emit numeric "
            "claims that do not match results.json?")

MAX_NEW_TOKENS = 700


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def recorded_prompt() -> tuple[list[dict], dict]:
    rec = json.loads(AGENT_RUN.read_text(encoding="utf-8"))
    # transcript index 2 = report_agent attempt 1 (0=cohort, 1=analysis)
    messages = rec["transcript"][2]["messages"]
    return messages, rec


def parse_passages(user_text: str) -> dict:
    """Reconstruct the {tag: text} passage map embedded in the user prompt."""
    out = {}
    in_block = False
    for line in user_text.splitlines():
        if line.startswith("Retrieved public-domain passages"):
            in_block = True
            continue
        if in_block:
            m = re.match(r"\[(PDQ:[^\]]+)\]\s+\"(.*)\"\s*$", line)
            if m:
                out[m.group(1)] = m.group(2)
            elif line.startswith("Optionally append"):
                break
    return out


def verify(draft: str, results: dict, focus_models: list,
           passages: dict) -> dict:
    sys.path.insert(0, str(ONCOCS_ROOT))
    from oncocs.agents.verifier import flatten_results, verify_draft
    split = json.loads(SPLIT_JSON.read_text(encoding="utf-8"))
    flat = flatten_results(results)
    flat.update({f"split.{k}": float(split[k]) for k in ("n_train", "n_test")})
    return verify_draft(draft, flat, results["checks"],
                      models=results.get("models"),
                      focus_models=focus_models, passages=passages)


def classify(findings: dict, draft: str) -> str:
    """A = numeric claim mismatch; B = verifier passed; C = inconclusive
    (failure is non-numeric: sections, phrasing, citations, abstention)."""
    if findings["passed"]:
        return "B"
    numeric = (findings["unverified_numbers"] + findings["unscoped_model_numbers"]
               + findings["misattributed"])
    return "A" if numeric else "C"


def run_hf(messages: list[dict]) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    free = 0
    if torch.cuda.is_available():
        free = torch.cuda.mem_get_info()[0] / 1e9
    device = "cuda" if free >= 4.0 else "cpu"

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), torch_dtype=torch.bfloat16,
        attn_implementation="eager").to(device)
    model.eval()

    inputs = tok.apply_chat_template(messages, add_generation_prompt=True,
                                     return_tensors="pt").to(device)
    t0 = time.monotonic()
    with torch.no_grad():
        out = model.generate(inputs, max_new_tokens=MAX_NEW_TOKENS,
                             do_sample=False, return_dict_in_generate=True,
                             output_scores=True)
    elapsed = round(time.monotonic() - t0, 1)
    gen_ids = out.sequences[0, inputs.shape[1]:].tolist()
    text = tok.decode(gen_ids, skip_special_tokens=True)

    # top-5 candidate strings/logits for every generated token containing a digit
    digit_steps = []
    for i, tid in enumerate(gen_ids):
        piece = tok.decode([tid])
        if any(ch.isdigit() for ch in piece):
            scores = out.scores[i][0]
            top = torch.topk(scores, 5)
            digit_steps.append({
                "step": i,
                "token_id": tid,
                "token": piece,
                "top5": [{"token": tok.decode([t.item()]),
                          "logit": round(s.item(), 4)}
                         for t, s in zip(top.indices, top.values)],
            })

    return {"device": device, "elapsed_s": elapsed,
            "prompt_tokens": int(inputs.shape[1]),
            "generated_token_ids": gen_ids,
            "generated_text": text,
            "digit_token_top5": digit_steps,
            "model_file_sha256": {f.name: sha256_file(f) for f in
                                  sorted(MODEL_DIR.iterdir()) if f.is_file()},
            "torch_version": torch.__version__,
            "transformers_version": __import__("transformers").__version__,
            "decode": "greedy (do_sample=False; seed not applicable)",
            "attn_implementation": "eager", "dtype": "bfloat16",
            "max_new_tokens": MAX_NEW_TOKENS}


def run_ollama(messages: list[dict]) -> dict:
    body = {"model": "gemma3:270m", "messages": messages, "stream": False,
            "options": {"temperature": 0, "num_predict": MAX_NEW_TOKENS}}
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=900) as resp:
        data = json.loads(resp.read())
    return {"elapsed_s": round(time.monotonic() - t0, 1),
            "model": "gemma3:270m (Ollama, quantized)",
            "decode": "temperature=0, num_predict=700",
            "generated_text": data["message"]["content"],
            "eval_count": data.get("eval_count")}


def main() -> None:
    messages, rec = recorded_prompt()
    results = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    user_text = next(m["content"] for m in messages if m["role"] == "user")
    passages = parse_passages(user_text)
    focus = rec["analysis_plan"].get("focus_models") or list(results["models"])

    output = {
        "question": QUESTION,
        "source_agent_run": str(AGENT_RUN),
        "prompt": {
            "messages": messages,
            "message_sha256": {m["role"]: hashlib.sha256(
                m["content"].encode()).hexdigest() for m in messages},
            "transcript_index": 2,
            "prompt_sha256_recorded": rec["transcript"][2]["prompt_sha256"],
        },
        "backends": {},
    }

    hf = run_hf(messages)
    hf["verifier_findings"] = verify(hf["generated_text"], results, focus, passages)
    hf["class"] = classify(hf["verifier_findings"], hf["generated_text"])
    output["backends"]["hf_eager_270m"] = hf

    try:
        ol = run_ollama(messages)
    except Exception as exc:
        ol = {"error": str(exc)}
    else:
        ol["verifier_findings"] = verify(ol["generated_text"], results, focus, passages)
        ol["class"] = classify(ol["verifier_findings"], ol["generated_text"])
    output["backends"]["ollama_270m"] = ol

    OUT_JSON.write_text(json.dumps(output, indent=2, default=str) + "\n",
                        encoding="utf-8")
    print("hf_eager_270m:", hf["class"], "| ollama_270m:", ol.get("class", ol.get("error")))
    print("wrote", OUT_JSON)


if __name__ == "__main__":
    main()
