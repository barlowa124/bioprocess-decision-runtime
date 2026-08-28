# Project Guidance

- Keep all process data and scenarios synthetic and public-safe.
- Do not claim GMP compliance, regulatory acceptance, process validation, or patient-safety assurance.
- Treat policy values as illustrative unless supplied and approved through a documented scientific process.
- Preserve advisory-only behavior; do not add equipment actuation without an explicit project-scope decision.
- Treat activation directions as experimental hypotheses requiring held-out and causal tests, not semantic proof.
- Describe operational traces as observed execution provenance. Describe reference certificates as fixed-input machine-checked equivalence, and bounded-domain certificates as exhaustive only over their declared canonical states; neither is a universal proof or independent primitive-kernel verification.
- Never allow an LLM to create or modify approved policy rules, coefficients, limits, or execution authority.
- Keep model checkpoints and full generated artifacts out of Git.
- Install `.[gemma]` before full operational-semantics verification; those tests skip when optional Torch/Transformers dependencies are absent.
- Run `python -m unittest discover -s tests -v` after code changes.
- Run `python -m bioprocess_runtime suite` to verify built-in scenarios.
- Run `python -m compileall -q bioprocess_runtime tests` for syntax verification.
