"""Cross-file anchor links resolve to a real heading (#1511 AC3/AC4).

`scripts/check_doc_preservation.py` validates anchors *inside* one document.
The cross-file case was the gap, and it is where the rot happens: a heading
gets reworded in `LIMITATIONS.md`, and six links in `docs/design/` that point
at its old slug break silently. Nothing fails, nothing is logged, and the
links keep rendering as links — they just land at the top of the page.

Nine such links were broken when this test was written, the oldest of them
pointing at sections that were restructured versions ago.

## Why this is a test and not another CI script

The `link-check` workflow does not catch this class. It resolves URLs over the
network, so it is subject to throttling — issue #1512 was a report of 733
"broken" links of which zero were broken. A local, offline, deterministic check
has no such failure mode and gates on every pull request through the existing
suite.

## The slug rule

GitHub lowercases the heading, drops every character that is not a letter,
digit, space, hyphen or underscore, then replaces each space with a hyphen.
**Runs of whitespace are not collapsed**, so `## A  B` becomes `a--b`. Getting
this wrong is not academic: a collapsing slugger reports 14 broken links here
where 9 are broken, and the 5 phantoms send a reader chasing anchors that
already resolve.

Repeated headings take GitHub's `-1`, `-2` disambiguation suffixes.
"""
from __future__ import annotations

import re
import subprocess
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# [text](relative/path.md#fragment)
_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+?)#([^)\s]+)\)")
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def _slug(heading: str) -> str:
    """GitHub's heading-to-anchor rule. Whitespace runs are NOT collapsed."""
    s = heading.strip().lower()
    s = re.sub(r"[^0-9a-z \-_]", "", s)
    return s.replace(" ", "-")


def _anchors(path: Path) -> set[str]:
    """Every anchor `path` defines, with GitHub's duplicate suffixes."""
    out: set[str] = set()
    seen: Counter[str] = Counter()
    in_fence = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:  # a `# comment` in a shell block is not a heading
            continue
        m = _HEADING.match(line)
        if not m:
            continue
        base = _slug(m.group(2))
        n = seen[base]
        seen[base] += 1
        out.add(base if n == 0 else f"{base}-{n}")
    return out


def _tracked_markdown() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "*.md"],
        cwd=REPO, capture_output=True, text=True, check=True, timeout=30,
    ).stdout.split()
    return [REPO / p for p in out]


def _broken() -> tuple[list[str], int]:
    """Return (failures, number of cross-file anchored links examined)."""
    failures: list[str] = []
    examined = 0
    cache: dict[Path, set[str]] = {}
    for src in _tracked_markdown():
        try:
            text = src.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for m in _LINK.finditer(text):
            rel, frag = m.group(1), m.group(2)
            if rel.startswith(("http://", "https://", "mailto:")):
                continue
            if not rel.endswith(".md"):
                continue
            target = (src.parent / rel).resolve()
            examined += 1
            rel_src = src.relative_to(REPO)
            if not target.exists():
                failures.append(f"{rel_src} -> {rel}#{frag} (no such file)")
                continue
            if target not in cache:
                cache[target] = _anchors(target)
            if frag.lower() not in cache[target]:
                failures.append(f"{rel_src} -> {rel}#{frag} (no such heading)")
    return failures, examined


@pytest.mark.timeout(60)
def test_the_scan_is_not_vacuous() -> None:
    """A check that examines nothing passes for the wrong reason.

    Follows the #1491 precedent at `scripts/check_conflict_markers.py`, which
    asserts on the count it scanned precisely because an empty `git ls-files`
    would otherwise read as success.
    """
    _, examined = _broken()
    assert examined > 20, (
        f"only {examined} cross-file anchored links examined — the scan is "
        "not reaching the documents it is supposed to check"
    )


@pytest.mark.timeout(60)
def test_every_cross_file_anchor_resolves() -> None:
    failures, _ = _broken()
    assert not failures, "broken cross-file anchors:\n  " + "\n  ".join(failures)
