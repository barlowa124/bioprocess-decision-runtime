# Project Guidance

- Keep all process data and scenarios synthetic and public-safe.
- Do not claim GMP compliance, regulatory acceptance, process validation, or patient-safety assurance.
- Treat policy values as illustrative unless supplied and approved through a documented scientific process.
- Preserve advisory-only behavior; do not add equipment actuation without an explicit project-scope decision.
- Run `python -m unittest discover -s tests -v` after code changes.
- Run `python -m bioprocess_runtime suite` to verify built-in scenarios.
- Run `python -m compileall -q bioprocess_runtime tests` for syntax verification.
