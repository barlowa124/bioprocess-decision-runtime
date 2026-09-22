# Copy-or-drift protocol v1 — gemma-3-270m-it (predeclared)

Step 2 of the writing-failure plan. Probe v1 (`gemma3_270m_fabrication_probe_v1`)
was class C: the 270M model never reached the metric-reporting part of the
full report task. This protocol shrinks the task to what the model can attempt
and asks one question at each complexity level.

**Question.** When gemma-3-270m-it is given recorded metric values in context
and asked to restate one (or all) of them, does it copy the value, drift to a
value not in the record, or omit it — and at which context-complexity level
does drift first appear?

## Fixed conditions

- Model: `.models/gemma-3-270m-it` (file sha256s as in probe v1).
- Runtime: torch 2.7.1+cu128, transformers 4.53.3, bf16, eager attention,
  `TORCH_COMPILE_DISABLE=1`, greedy (`do_sample=False`), `max_new_tokens=400`.
- Chat template applied to a single user message; no system message.
- Ground truth: `C:\oncology-coscientist\results\luad\3097990b11d8\results.json`,
  `models.{model}.metrics.{metric}`, 4 models x 6 metrics = 24 values,
  presented and compared **rounded to 3 decimals**.

Models: `cox/clinical`, `rsf/clinical`, `cox/clinical_expression`,
`rsf/clinical_expression`.
Metrics and labels: `harrell_c` "Harrell C"; `uno_c` "Uno C"; `auc_12m` "AUC at
12 months"; `auc_24m` "AUC at 24 months"; `auc_36m` "AUC at 36 months";
`integrated_brier_6_36m` "integrated Brier score (6-36 months)".
(`auc_mean` is excluded; it is not shown in reports.)

## Levels (user-message templates)

- **L1** single value:
  `Recorded result: the {label} for {model} is {value}.\nState the {label} for {model}. Answer with the number only.`
- **L2** one distractor (the other model with the same first word, e.g. cox vs rsf, same metric):
  `Recorded results:\n- {model}: {label} {value}\n- {other}: {label} {value2}\nState the {label} for {model}. Answer with the number only.`
- **L3** four models, one metric:
  `Recorded results:\n- {m1}: {label} {v1}\n- {m2}: ...\n- {m3}: ...\n- {m4}: ...\nState the {label} for {model}. Answer with the number only.`
  (models listed in the fixed order above)
- **L4** full table, one value:
  `Recorded results (JSON): {json}\nState the {label} for {model}. Answer with the number only.`
  where `{json}` is `{"cox/clinical": {"harrell_c": 0.647, ...}, ...}` with the
  six metrics per model in the fixed metric order, 3 decimals.
- **L5** full table, restate all:
  `Recorded results (JSON): {json}\nWrite a markdown table with one row per model and one column per metric, using exactly the recorded values. No other text.`

## Scoring (mechanical; no judgment)

Extract every decimal token matching `\b0\.\d{1,4}\b` from the generated text
(integers are ignored — they occur in metric labels). Let R be the set of the
24 recorded values at 3 dp.

- **copy**: every extracted decimal is within 0.0005 of some value in R, and
  (L1–L4) the asked value is present; (L5) all 24 values are present.
- **drift**: at least one extracted decimal is not within 0.0005 of any value
  in R. Record every drifting decimal, the asked value, and — from the
  generation record — the first generated digit token that departs from the
  asked value's digit string, with top-5 candidate strings and logits at that
  step and whether the correct digit is among them.
- **omission**: no drift, but the asked value (L1–L4) or at least one of the
  24 values (L5) is absent.
- **partial** (L5 only): no drift, some values absent — report count present.

A case that matches a recorded value *other than the asked one* (e.g. states
the rsf value when asked for cox) is **misattribution**: record separately; it
is not drift by this definition because the number exists in R.

## Development cases (10) — may be used to debug the harness

Targets: T1 = (`cox/clinical`, `harrell_c`), T2 = (`rsf/clinical_expression`, `uno_c`).
Cases: T1 x L1–L4, T2 x L1–L4, L5 x 2 (identical prompt; run twice to confirm
greedy determinism — outputs must be byte-identical).

## Held-out cases (17) — sealed before any generation

From the 22 (model, metric) pairs other than T1 and T2, draw 4 pairs with
`random.Random(20240601).sample(sorted(pairs), 4)`, where each pair is the
string `f"{model}|{metric}"`. Held-out = those 4 x L1–L4 (16 cases) + L5 (1).
Write all 17 rendered prompts to
`results/gemma3_270m_copy_drift_heldout_v1.json` and record its sha256 in the
run output **before** the first held-out generation. The harness may not be
edited after the held-out file is sealed; if it must be, the held-out file is
regenerated and the run restarts.

## Outputs

- `artifacts/gemma_copy_drift_probe_v1.py` (versioned with `git add -f`).
- `results/gemma3_270m_copy_drift_v1.json`: protocol sha256, held-out sha256,
  per-case prompt, generated text, token ids, per-digit-token top-5, score
  class, drifting decimals, first-departure record.
- `results/gemma3_270m_copy_drift_v1.md`: one table row per case (set, level,
  target, class, asked value, first drifting decimal if any), then a per-level
  summary (copy/drift/omission/misattribution counts, dev vs held-out).

## Interpretation rules (declared now)

- Drift at any held-out level => the deconstructed checkpoint exhibits a
  small-scale analogue of the writing failure at that level; the recorded
  first-departure forward pass is the target for step 3.
- Copy at all held-out levels => 270M copies faithfully within this task
  family; fabrication as seen at 4b+ is not reproduced by shrinking the task.
  Report this plainly; do not add levels post hoc to hunt for drift.
- Omission-dominated => task still too hard; report as such.
