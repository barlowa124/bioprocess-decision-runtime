#!/usr/bin/env python3
"""Pre-commit secret scanner.

Scans staged additions (`--staged`, the mode the pre-commit hook uses)
or arbitrary paths (`--paths FILE...`) for credential-shaped strings.

Format families covered: Hugging Face, GitHub (all prefixes + PAT),
AWS access keys, OpenAI/Anthropic-style `sk-` keys, Slack tokens,
Google API keys, private-key blocks. Generic "password = ..." prose is
deliberately not scanned — it produces noise on vendored code; the
exposure this catches is tokens, which have fixed formats.

Deliberate allowlist: *_REDACTED markers, documented example
credentials, obvious placeholders. Additions only in staged mode —
existing history is a different problem (see git filter-repo).
"""

from __future__ import annotations

import re
import subprocess
import sys

PATTERNS = [
    (re.compile(r"hf_[A-Za-z0-9]{20,}"), "Hugging Face token"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{22,}"), "GitHub PAT"),
    (re.compile(r"gh[posur]_[A-Za-z0-9]{20,}"), "GitHub token"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AWS access key"),
    (re.compile(r"ASIA[0-9A-Z]{16}"), "AWS temporary key"),
    (re.compile(r"sk-[A-Za-z0-9_-]{20,}"), "OpenAI-style key"),
    (re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"), "Slack token"),
    (re.compile(r"AIza[0-9A-Za-z_-]{35}"), "Google API key"),
    (re.compile(r"pypi-[A-Za-z0-9_-]{16,}"), "PyPI token"),
    (re.compile(r"glpat-[A-Za-z0-9_-]{20,}"), "GitLab PAT"),
    (re.compile(r"npm_[A-Za-z0-9]{36}"), "npm token"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY"), "private key"),
]

ALLOWLIST = re.compile(
    r"(REDACTED|EXAMPLE|example|placeholder|PLACEHOLDER|"
    r"dummy|DUMMY|<[a-z_-]+>|\*{3,})")

_PREFIX = re.compile(r"^(hf_|github_pat_|gh[posur]_|AKIA|ASIA|sk-|"
                     r"xox[baprs]-|AIza)")


def _placeholder(match_text: str) -> bool:
    """True when the matched token body is an obvious placeholder
    (all x/* /0 chars) rather than a credential-shaped string."""
    body = _PREFIX.sub("", match_text)
    return bool(re.fullmatch(r"[xX*0]+", body))


def scan_text(text: str, source: str) -> list[tuple[str, int, str, str]]:
    hits = []
    for i, line in enumerate(text.splitlines(), 1):
        for pat, name in PATTERNS:
            for m in pat.finditer(line):
                token = m.group(0)
                if ALLOWLIST.search(token) or _placeholder(token):
                    continue
                hits.append((source, i, name, token[:12] + "..."))
    return hits


def staged_diff_lines() -> list[tuple[str, str]]:
    """Return (file, added-line) pairs from the staged diff."""
    out = subprocess.run(
        ["git", "diff", "--cached", "-U0", "--diff-filter=ACM",
         "--no-color"],
        capture_output=True, text=True, check=True).stdout
    pairs = []
    current = None
    for line in out.splitlines():
        if line.startswith("+++ b/"):
            current = line[6:]
        elif line.startswith("+") and not line.startswith("+++") \
                and current:
            pairs.append((current, line[1:]))
    return pairs


def main(argv: list[str]) -> int:
    hits = []
    if "--paths" in argv:
        idx = argv.index("--paths")
        for path in argv[idx + 1:]:
            try:
                with open(path, errors="replace") as f:
                    hits += scan_text(f.read(), path)
            except (OSError, UnicodeDecodeError):
                continue
    else:
        # staged mode: scan added lines only
        per_file: dict[str, str] = {}
        for fname, added in staged_diff_lines():
            per_file.setdefault(fname, "")
            per_file[fname] += added + "\n"
        for fname, text in per_file.items():
            hits += scan_text(text, fname)

    if hits:
        print("secret-like strings in staged changes:", file=sys.stderr)
        for src, ln, name, frag in hits:
            print(f"  {src}:{ln}: {name} ({frag})", file=sys.stderr)
        print("unstage or redact; if already pushed, revoke first.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
