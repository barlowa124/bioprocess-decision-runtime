# Provenance convention

Shared across this portfolio. Every computed artifact carries a manifest
binding it to the code, config, and inputs that produced it. The manifest
is part of the result — written next to outputs, embedded in API
responses — not a log line.

## Manifest schema

```json
{
  "schema": "provenance/v1",
  "created_utc": "ISO-8601 Z",
  "git": {"sha": "<40-hex>", "dirty": false},
  "package_versions": {"<dist>": "<version>"},
  "tool": "<engine or pipeline name>",
  "config_sha256": "<64-hex or null>",
  "input_sha256": {"<path>": "<64-hex>"},
  "output_sha256": {"<path>": "<64-hex>"}
}
```

## Rules

- Hashes are SHA-256 hex digests of file bytes. Directory inputs hash as
  the sha256 of the sorted `relpath:filehash` manifest of their files.
- `git.dirty` is recorded, never hidden; dirty runs are still provenance.
- `config_sha256` covers the parsed config object (canonical JSON with
  sorted keys), not the file bytes, so formatting changes do not churn it.
- Missing inputs are omitted from `input_sha256`, not zeroed.
- `output_sha256` may be added after writing outputs; the manifest is
  rewritten once, then treated as immutable.
- Verification recomputes the hashes and compares — it never trusts
  stored values.

## Implementations

Reference helper `provenance.py` is vendored per-repo (same bytes in
each adopting repo) and emits schema `provenance/v1`.
oncology-coscientist `oncocs/evidence.py` and dockops
`src/dockops/provenance.py` predate the convention and implement the
same binding discipline (git state, input/config/output hashes,
verification by recomputation) in their own formats; they are not
retrofitted with the v1 tag since their formats are hash-pinned.
