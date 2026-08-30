# Project Guidance

- Keep all process data and scenarios synthetic and public-safe.
- Do not claim GMP compliance, regulatory acceptance, process validation, or patient-safety assurance.
- Treat policy values as illustrative unless supplied and approved through a documented scientific process.
- Preserve advisory-only behavior; do not add equipment actuation without an explicit project-scope decision.
- Treat activation directions as experimental hypotheses requiring held-out and causal tests, not semantic proof.
- Describe operational traces as observed execution provenance. Describe reference certificates as fixed-input machine-checked equivalence, and bounded-domain certificates as exhaustive only over their declared canonical states; neither is a universal proof or independent primitive-kernel verification.
- Limit universal-proof claims to the exact SMT formulas, widths, domains, and explicit premises recorded in each obligation. Describe CUDA manifests as launch and binary provenance and SASS equations as proposed subset semantics, not hardware or instruction-level verification. Distinct-kernel suite coverage means one attested invocation per recorded symbol, not every invocation. Module NVTX bindings identify enclosing framework ranges and boundary tensors. CUPTI launch-parameter matches establish equal pointer-sized values only; do not call them typed signatures, access-direction proofs, or complete CUDA argument semantics.
- Never allow an LLM to create or modify approved policy rules, coefficients, limits, or execution authority.
- Keep model checkpoints and full generated artifacts out of Git.
- Install `.[gemma]` before full operational-semantics verification and `.[proof]` before SMT verification; optional tests skip when their dependencies are absent.
- Run `python -m unittest discover -s tests -v` after code changes.
- Run `python -m bioprocess_runtime suite` to verify built-in scenarios.
- Run `python -m compileall -q bioprocess_runtime tests` for syntax verification.
