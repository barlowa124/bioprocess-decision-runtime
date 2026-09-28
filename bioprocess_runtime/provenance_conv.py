"""Run provenance helper implementing docs/PROVENANCE.md (schema
provenance/v1). Vendored identically into each adopting repo; the
native implementations in oncology-coscientist (oncocs/evidence.py) and
dockops (src/dockops/provenance.py) conform to the same schema.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

SCHEMA = "provenance/v1"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_dir(path: str | Path) -> str:
    """Hash a directory as sha256 of its sorted relpath:filehash lines."""
    root = Path(path)
    entries = sorted(
        f"{p.relative_to(root)}:{sha256_file(p)}"
        for p in root.rglob("*") if p.is_file()
    )
    return hashlib.sha256("\n".join(entries).encode()).hexdigest()


def sha256_obj(obj) -> str:
    """Canonical JSON hash (sorted keys) — stable across file formatting."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _git() -> dict:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True,
            text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True,
            text=True, check=True).stdout.strip())
        return {"sha": sha, "dirty": dirty}
    except Exception:
        return {"sha": None, "dirty": None}


def _versions(packages) -> dict:
    return {p: v for p in packages if (v := _try_version(p))}


def _try_version(p):
    try:
        return version(p)
    except Exception:
        return None


def manifest(tool: str, inputs=None, config=None, config_path=None,
             packages=()) -> dict:
    """Build a provenance/v1 manifest. `inputs` are paths; `config` is
    hashed canonically; `config_path` hashes the file bytes if given."""
    m = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git(),
        "package_versions": _versions(packages),
        "tool": tool,
        "config_sha256": None,
        "input_sha256": {},
        "output_sha256": {},
    }
    if config is not None:
        m["config_sha256"] = sha256_obj(config)
    elif config_path is not None and Path(config_path).exists():
        m["config_sha256"] = sha256_file(config_path)
    for p in inputs or []:
        pp = Path(p)
        if pp.is_dir():
            m["input_sha256"][str(p)] = sha256_dir(pp)
        elif pp.exists():
            m["input_sha256"][str(p)] = sha256_file(pp)
    return m


def seal(m: dict, outputs) -> dict:
    """Add output hashes after writing results, then treat as immutable."""
    for p in outputs or []:
        if Path(p).exists():
            m["output_sha256"][str(p)] = sha256_file(p)
    return m


def verify(m: dict) -> list[str]:
    """Recompute every recorded hash; return a list of mismatches."""
    bad = []
    for section in ("input_sha256", "output_sha256"):
        for path, want in m.get(section, {}).items():
            p = Path(path)
            if not p.exists():
                bad.append(f"{section}:{path}: missing")
                continue
            got = sha256_dir(p) if p.is_dir() else sha256_file(p)
            if got != want:
                bad.append(f"{section}:{path}: sha256 mismatch")
    return bad


def write(m: dict, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.write_text(json.dumps(m, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    return out
