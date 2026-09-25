"""CPU replay of saved copy/drift wrong-digit decisions (new-backend experiment).

Replays the six misattribution targets recorded in
results/gemma3_270m_copy_drift_v1.json on a CPU backend with the byte-exact
pinned checkpoint and tokenizer assets. Two decode paths per case:

  hybrid    model.generate with the pinned generation_config defaults
            (cache_implementation="hybrid") — matches the acquisition path
            semantics on this backend
  recompute manual greedy decode, model(..., use_cache=False), full-context
            recomputation per step — matches the independent engine's decode
            semantics on this backend

This is a new-backend comparison, not a reproduction of the frozen CUDA
evidence: CPU scores are not bitwise comparable to the recorded CUDA scores,
and only token selections and top-five structure are compared. The frozen
30-token profile is unchanged by this experiment.

Usage: python artifacts/gemma_copy_drift_cpu_replay_v1.py <model_dir> <output.json>
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROBE_JSON = ROOT / "results/gemma3_270m_copy_drift_v1.json"
TOKENIZATION_JSON = ROOT / "results/gemma3_270m_copy_drift_tokenization_v1.json"
TARGET_INDICES = [3, 12, 16, 17, 20, 21]
DECISION_STEPS = {3: 4, 12: 4, 16: 3, 17: 3, 20: 3, 21: 2}
NEW_TOKENS = 8

PINNED_MODEL_SHA256 = "700b710a9a99c295ed546647aa81cacf9f81f4c573ea2be613a0e2517a44afab"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    model_dir = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    if output_path.exists():
        raise SystemExit(f"refusing to overwrite {output_path}")
    weights = model_dir / "model.safetensors"
    weights_sha = _sha(weights.read_bytes())
    if weights_sha != PINNED_MODEL_SHA256:
        raise SystemExit(f"checkpoint sha256 {weights_sha} does not match the pinned acquisition checkpoint")

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    probe = json.loads(PROBE_JSON.read_text(encoding="utf-8"))
    tokenization = json.loads(TOKENIZATION_JSON.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=torch.bfloat16,
        attn_implementation="eager").to("cpu")
    model.eval()

    cases = []
    for index in TARGET_INDICES:
        case = probe["cases"][index]
        recovered = tokenization["cases"][index]
        messages = [{"role": "user", "content": case["prompt"]}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt")
        prompt_ids = inputs[0].tolist()
        conformance = prompt_ids == recovered["prompt_token_ids"]

        t0 = time.monotonic()
        with torch.no_grad():
            out = model.generate(inputs, max_new_tokens=NEW_TOKENS,
                                 do_sample=False, return_dict_in_generate=True,
                                 output_scores=True)
        hybrid_ids = out.sequences[0, inputs.shape[1]:].tolist()
        hybrid_scores = out.scores
        hybrid_elapsed = time.monotonic() - t0

        t0 = time.monotonic()
        context = inputs.clone()
        recompute_ids = []
        recompute_logits = []
        with torch.no_grad():
            for _ in range(NEW_TOKENS):
                logits = model(input_ids=context, use_cache=False).logits[0, -1]
                recompute_logits.append(logits.clone())
                nxt = int(torch.argmax(logits))
                recompute_ids.append(nxt)
                context = torch.cat(
                    [context, torch.tensor([[nxt]], dtype=context.dtype)], dim=1)
        recompute_elapsed = time.monotonic() - t0

        step = DECISION_STEPS[index]
        def top5(scores_or_logits, i):
            source = scores_or_logits[i][0] if isinstance(scores_or_logits, tuple) else scores_or_logits[i]
            top = torch.topk(source, 5)
            return [{"token": tokenizer.decode([t.item()]),
                     "token_id": int(t.item()),
                     "logit": round(float(s.item()), 4)}
                    for t, s in zip(top.indices, top.values)]

        recorded = next((r for r in case["generation"]["digit_token_top5"]
                         if r["step"] == step), None)
        cases.append({
            "source_case_index": index,
            "level": case["level"],
            "target": case["target"],
            "recorded_score_class": case["score_class"],
            "decision_step_zero_based": step,
            "prompt_token_ids_match_recovered": conformance,
            "prompt_token_count": len(prompt_ids),
            "context_token_count_at_decision": len(prompt_ids) + step,
            "recorded_cuda": {
                "generated_token_ids_prefix": case["generation"]["generated_token_ids"][:NEW_TOKENS],
                "token_at_decision": case["generation"]["generated_token_ids"][step],
                "top5_at_decision": recorded["top5"] if recorded else None,
            },
            "cpu_hybrid": {
                "generated_token_ids": hybrid_ids,
                "generated_text": tokenizer.decode(hybrid_ids),
                "token_at_decision": hybrid_ids[step] if step < len(hybrid_ids) else None,
                "top5_at_decision": top5(hybrid_scores, step),
                "all_steps_top5": [top5(hybrid_scores, i) for i in range(len(hybrid_ids))],
                "elapsed_s": round(hybrid_elapsed, 1),
            },
            "cpu_recompute_nocache": {
                "generated_token_ids": recompute_ids,
                "generated_text": tokenizer.decode(recompute_ids),
                "token_at_decision": recompute_ids[step] if step < len(recompute_ids) else None,
                "top5_at_decision": top5(recompute_logits, step),
                "all_steps_top5": [top5(recompute_logits, i) for i in range(len(recompute_ids))],
                "elapsed_s": round(recompute_elapsed, 1),
            },
        })
        print(f"case {index}: recorded={cases[-1]['recorded_cuda']['token_at_decision']} "
              f"hybrid={cases[-1]['cpu_hybrid']['token_at_decision']} "
              f"recompute={cases[-1]['cpu_recompute_nocache']['token_at_decision']} "
              f"ids_match={conformance}", flush=True)

    body = {
        "schema_version": 1,
        "kind": "new_backend_decision_replay",
        "scope": "CPU replay of already-observed wrong-digit decisions under two decode "
                 "paths. New backend evidence: decisions and top-five structure only; "
                 "scores are not bitwise comparable to the recorded CUDA run. Not a "
                 "confirmatory holdout, numerical replay of frozen evidence, or causal claim.",
        "sources": {
            "probe": {"filename": PROBE_JSON.name, "sha256": _sha(PROBE_JSON.read_bytes())},
            "tokenization": {"filename": TOKENIZATION_JSON.name,
                              "sha256": _sha(TOKENIZATION_JSON.read_bytes())},
            "harness": {"filename": Path(__file__).name,
                        "sha256": _sha(Path(__file__).read_bytes())},
        },
        "binding": {
            "model_safetensors_sha256": weights_sha,
            "backend": "cpu",
            "dtype": "bfloat16",
            "attn_implementation": "eager",
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "decode_paths": {
                "cpu_hybrid": "generate defaults incl. generation_config cache_implementation=hybrid",
                "cpu_recompute_nocache": "manual argmax loop, use_cache=False, full-context recompute",
            },
            "new_tokens_per_case": NEW_TOKENS,
        },
        "recorded_acquisition_backend": {"device": probe["environment"]["device"],
                                          "cache_implementation": "hybrid"},
        "cases": cases,
        "limitations": [
            "CPU arithmetic is not bitwise equal to the recorded CUDA arithmetic; selection agreement is evidence about the decision, not about score equality.",
            "Hybrid-vs-recompute agreement on CPU does not establish the same agreement on CUDA.",
            "These cases were already inspected; this replay cannot serve as fresh confirmatory evidence.",
        ],
    }
    body["report_sha256"] = _sha(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                          allow_nan=False).encode("utf-8"))
    output_path.write_text(json.dumps(body, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    print("wrote", output_path)


if __name__ == "__main__":
    main()
