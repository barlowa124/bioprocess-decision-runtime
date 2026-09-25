# Copy-drift counterfactual protocol v1 — gemma-3-270m-it (predeclared)

Step 3a of the writing-failure plan, run on a **CPU backend** (the recorded
acquisition backend is unavailable). New-backend evidence only: decisions,
not bitwise score equality with the CUDA record. The CPU replay
(`gemma3_270m_copy_drift_cpu_replay_v1.json`) reproduced all five recorded
digit substitutions under both hybrid-cache and full-recompute decode, so
this probe uses the hybrid path only.

**Question.** In the six saved misattribution cases, does the emitted value
track (a) the emitted owner's *entity name*, (b) the owner's *position* in
the record listing, or (c) the *value string itself*?

## Fixed conditions

- Checkpoint: `model.safetensors` sha256
  `700b710a9a99c295ed546647aa81cacf9f81f4c573ea2be613a0e2517a44afab` (pinned).
- Tokenizer/config assets: the seven files pinned by
  `gemma3_270m_copy_drift_v1.json` environment, hash-verified at load.
- Backend: CPU, torch 2.7.1, transformers 4.53.3, bf16, eager attention,
  greedy (`do_sample=False`), `max_new_tokens=8`, `output_scores=True`.
- Chat template: single user message, `add_generation_prompt=True`.
- Ground truth and case prompts: the saved copy/drift records; variants are
  deterministic string/JSON transformations of the saved prompt text.

## Cases

Six target cases (source indices 3, 12, 16, 17, 20, 21 of
`gemma3_270m_copy_drift_v1.json`). Each gets these variants:

- `baseline`: the saved prompt unchanged (CPU determinism check).
- `row_swap`: exchange the positions of the asked record and the emitted
  owner's record in the listing/JSON. For cases 3 and 21 (owner == asked
  record), the record order is reversed instead (asked first<->last).
- `value_move`: replace the asked value with sentinel `0.777` (present
  nowhere else); the emitted owner's value stays. Emission of the sentinel
  shows entity-following; emission of the old emitted value shows
  owner/value binding.
- `entity_rename`: rename the emitted owner's entity to `zzz/other`
  (listing only; the question still names the asked entity). Correct
  emission indicates name-confusion binding. For case 21 the wrong source
  is a metric, not an entity, so the emitted metric key is renamed
  (`integrated_brier_6_36m` -> `zz_brier`, case 3 `uno_c` -> `zz_uno`) in
  all records (`metric_rename`). Cases 3 and 21 additionally get
  `metric_swap`: the asked metric key and the emitted metric key exchange
  positions inside the asked record's JSON object (values follow keys).
- `value_scramble`: rewrite the emitted value's digits at its source
  (`+0.034`, 3-decimal format kept). Tracking the scrambled value shows
  position/owner copying rather than a fixed value attractor. For case 21
  every record's brier value is scrambled independently.

Sealed cases file: `results/gemma3_270m_copy_drift_counterfactual_cases_v1.json`,
sha256 recorded in the run output before the first generation.

## Scoring (mechanical)

Extract the first decimal matching `\b0\.\d{1,4}\b` from generated text
(`omission` if none). Classify vs: `asked` (correct), `recorded_emitted`
(the historical wrong value), `sentinel` (0.777), `scrambled` (the new
source value), `other_present` (any other value in the rendered prompt),
`novel` (drift — not in prompt). Record generated ids and per-digit top-5.

## Interpretation rules (declared now)

- `value_move` -> `recorded_emitted` : owner-or-value bound (entity ignored).
- `value_move` -> `sentinel` : correct entity retrieval restored by a
  non-attractor value — argues the failure is value-driven, not entity-blind.
- `row_swap` tracks the *entity* : entity binding; tracks the *position* :
  positional binding.
- `entity_rename` -> `asked` : sibling name confusability is the mechanism.
- `value_scramble` -> `scrambled` : copy tracks the source location, not the
  literal value.
- Any `novel` emission: the behaviour is not pure copying — report plainly.
- These are CPU-backend results; they constrain but do not close the CUDA
  mechanism question.
