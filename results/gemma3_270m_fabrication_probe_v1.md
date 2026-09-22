# Fabrication probe v1 — gemma-3-270m-it

**Question:** Given the exact report-agent prompt from oncology run
`results/luad/3097990b11d8/agent/318d61643606` (attempt 1), does
gemma-3-270m-it under HF eager greedy decoding emit numeric claims that do
not match results.json?

**Outcome classes:** A = numeric claims mismatch (fabrication reproduced);
B = all numeric claims match (not reproduced); C = no verifiable numeric
claims or incoherent (inconclusive — model too small for the task).

## Result

| Backend | Class | Verifier outcome |
|---|---|---|
| `hf_eager_270m` (local checkpoint, bf16, eager, greedy, CUDA) | **C** | all 12 emitted numeric claims verified; failed only on `forbidden: C-index described as calibration` — the token sequence was the model echoing the analysis-plan JSON verbatim, which contains that phrase |
| `ollama_270m` (Ollama gemma3:270m, temp 0, num_predict 700) | **C** | all emitted numeric claims verified; same echo failure plus echo of the passage/instruction block |

## What happened

Both backends produced a degenerate report: the four required section
headers were emitted empty, followed by a correct cohort summary (501, 497,
350, 151, 61/4 dropped, missingness fractions 0.02 and 0.004 — all
verified), then a verbatim echo of the analysis-plan JSON (and, for Ollama,
the retrieved-passage block and its own citation instruction). Neither
draft reached the `## Models and metrics` content where the fabrication
failures at 4b/12b/27b occurred. The prompt region responsible for the
fabrication observed at larger sizes was therefore never exercised — the
question is inconclusive at 270M, consistent with class C.

No numeric claim in either draft mismatched results.json.

## Run record

- HF: device `cuda` (14.8 GB free at start), 349 generated tokens, 23.6 s.
- Ollama: 700 tokens (hit `num_predict` cap while echoing the prompt), 11.5 s.
- Prompt verbatim (system + user) with sha256, full token ids, and per-digit-token
  top-5 candidate strings/logits: `results/gemma3_270m_fabrication_probe_v1.json`.
- Checkpoint: `.models/gemma-3-270m-it`, file sha256s recorded in the JSON.
- Runtime: torch 2.7.1+cu128, transformers 4.53.3, bf16, eager attention,
  `TORCH_COMPILE_DISABLE=1` (triton unavailable on Windows).
