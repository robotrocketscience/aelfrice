"""Audit a GitHub issue body for surface that already exists on `origin/main`.

Used by `.github/workflows/issue-creation-audit.yml` to flag duplicate-of-shipped
issues at creation time (closes the gap demonstrated by #521 — VocabBridge
substrate filed for work that had already shipped under #433).

Reads the issue body (stdin or `--body-file`), extracts candidate symbols
via four regex patterns, and checks each against `origin/main` with `git
ls-tree` (paths) or `git grep` (symbols). Writes a comment-ready audit
message to stdout when ≥1 hit found; emits nothing on a clean body.

The caller is responsible for fetching `origin/main` before invocation.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path

COMMENT_MARKER_PREFIX = "<!-- aelf-audit-v1"

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("module_path", re.compile(r"\b(src/aelfrice/[a-z_][a-z0-9_]*\.py)\b")),
    ("class_name", re.compile(r"\bclass\s+([A-Z][A-Za-z0-9_]+)\b")),
    ("file_path", re.compile(r"\b((?:src|tests|docs)/[A-Za-z0-9_./-]+\.[a-z]+)\b")),
    ("api_symbol", re.compile(r"`([A-Z][A-Za-z0-9_]*\.[a-z_][a-z0-9_]*)\(")),
]

# Sections to strip before extraction. These tend to *reference* existing
# surface (parent issues, prior PRs, ratification trail) rather than propose
# *new* surface, so extracting from them produces false positives.
STRIP_SECTION_HEADERS = {"refs", "references", "out of scope", "related"}

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def _strip_sections(body: str) -> str:
    """Remove any top-level (#-#####) section whose heading matches STRIP_SECTION_HEADERS."""
    out: list[str] = []
    pos = 0
    matches = list(_HEADER_RE.finditer(body))
    for i, m in enumerate(matches):
        title = m.group(2).strip().lower().rstrip(":")
        section_start = m.start()
        section_end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        if pos < section_start:
            out.append(body[pos:section_start])
        if title in STRIP_SECTION_HEADERS:
            pos = section_end
            continue
        out.append(body[section_start:section_end])
        pos = section_end
    if pos < len(body):
        out.append(body[pos:])
    return "".join(out)


def extract(body: str) -> list[tuple[str, str]]:
    body = _strip_sections(body)
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for kind, pat in PATTERNS:
        for m in pat.finditer(body):
            key = (kind, m.group(1))
            if key not in seen:
                seen.add(key)
                out.append(key)
    return out


def _git(args: list[str]) -> tuple[int, str]:
    r = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    return r.returncode, r.stdout


def check_path(path: str, ref: str) -> str | None:
    rc, out = _git(["ls-tree", "-r", "--name-only", ref, "--", path])
    if rc != 0 or out.strip() != path:
        return None
    rc, log = _git(["log", ref, "-n", "1", "--format=%h %s", "--", path])
    return f"`{path}` (shipped {log.strip()})" if rc == 0 and log.strip() else f"`{path}`"


def check_class(name: str, ref: str) -> str | None:
    rc, out = _git(["grep", "-l", f"^class {name}\\b", ref, "--", "src/"])
    if rc != 0 or not out.strip():
        return None
    paths = sorted({line.split(":", 1)[1] for line in out.strip().splitlines() if ":" in line})
    return f"`class {name}` (defined in: {', '.join(paths)})"


def check_symbol(symbol: str, ref: str) -> str | None:
    rc, out = _git(["grep", "-l", "-F", symbol, ref, "--", "src/"])
    if rc != 0 or not out.strip():
        return None
    paths = sorted({line.split(":", 1)[1] for line in out.strip().splitlines() if ":" in line})
    return f"`{symbol}` (referenced in: {', '.join(paths)})"


def find_hits(candidates: list[tuple[str, str]], ref: str) -> list[str]:
    hits: list[str] = []
    for kind, value in candidates:
        if kind in ("module_path", "file_path"):
            hit = check_path(value, ref)
        elif kind == "class_name":
            hit = check_class(value, ref)
        elif kind == "api_symbol":
            hit = check_symbol(value, ref)
        else:
            hit = None
        if hit and hit not in hits:
            hits.append(hit)
    return hits


def hit_hash(hits: list[str]) -> str:
    h = hashlib.sha256("\n".join(sorted(hits)).encode("utf-8")).hexdigest()
    return h[:12]


def render(hits: list[str]) -> str:
    bullets = "\n".join(f"  - {h}" for h in hits)
    return (
        f"{COMMENT_MARKER_PREFIX} hits:{hit_hash(hits)} -->\n"
        "🔍 **Pre-implementation audit**\n\n"
        "This issue body proposes surface that already exists on `main`:\n\n"
        f"{bullets}\n\n"
        "If this is intentional follow-up work (rename / refactor / bench gate), "
        "reply `audit-ack` to dismiss. Otherwise consider closing as superseded.\n"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--body-file", type=Path)
    p.add_argument("--main-ref", default="origin/main")
    args = p.parse_args(argv)
    body = args.body_file.read_text() if args.body_file else sys.stdin.read()
    hits = find_hits(extract(body), args.main_ref)
    if hits:
        sys.stdout.write(render(hits))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
