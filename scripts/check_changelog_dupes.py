#!/usr/bin/env python3
"""Fail when one CHANGELOG entry restates another (#1211).

The house rule for CHANGELOG merge conflicts is to resolve insert-only:
keep both sides, never reorder, because reordering rewrites lines that
are already merged and they reappear as `+` lines that trip the
added-lines-only discretion grep.

That rule is right for *added* entries and wrong for *amended* ones.
When a branch edits an entry already on main, "keep both sides" leaves
the original and the edit as two bullets instead of one replaced
bullet. Nothing else caught it: `release-docs-check` verifies that
[Unreleased] is drained and that the version section and compare
footnote exist, not that the entries are distinct.

## Why a shared prefix, and not the obvious alternatives

*Not the issue number.* Several fixes legitimately land under one
umbrella: #1160 appears four times in v4 and #1161 three, each a
different fix with different text.

*Not the bold title.* Precise for prose titles, but v1.1.0 has two
entries both titled ``docs/promotion_path.md`` — a bare path reused as
a heading for two genuinely different notes.

*Not full containment either.* An amendment often rewrites rather than
appends: the #1168 pair diverges at character 496, where the later
revision corrects the numbers. Requiring the shorter to be a complete
prefix of the longer missed it.

A long shared *opening* is what actually distinguishes the case. These
bullets begin with a bold title and an issue link, so two entries
agreeing for 200 characters are the same entry twice. Validated across
every committed changelog (v0–v4 and the index): the two known
duplicates match at 496 and 2763 characters, and no unrelated pair
anywhere reaches 120.

Usage: check_changelog_dupes.py CHANGELOG/v4.md [...]
"""
from __future__ import annotations

import sys
from pathlib import Path

# Characters two entries must agree on before they are called the same
# entry. Between the widest incidental overlap observed (<120) and the
# narrowest real duplicate (496), so neither bound is tight.
PREFIX_THRESHOLD = 200


def _common_prefix_len(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def find_duplicates(text: str) -> list[tuple[str, int, str]]:
    """Return (section, shared_prefix_len, entry) per restated pair."""
    section = "(preamble)"
    buckets: dict[str, list[str]] = {}
    for line in text.split("\n"):
        if line.startswith("## "):
            section = line[3:].strip()
        elif line.startswith("- "):
            buckets.setdefault(section, []).append(line)

    found: list[tuple[str, int, str]] = []
    for name, entries in buckets.items():
        for i, first in enumerate(entries):
            for second in entries[i + 1:]:
                shared = _common_prefix_len(first, second)
                if shared >= PREFIX_THRESHOLD:
                    # Report the shorter: it is the superseded revision.
                    older = min(first, second, key=len)
                    found.append((name, shared, older))
    return found


def main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv[1:]]
    if not paths:
        print("usage: check_changelog_dupes.py <changelog.md> ...",
              file=sys.stderr)
        return 2

    failed = False
    for path in paths:
        if not path.is_file():
            print(f"::error file={path}::no such file", file=sys.stderr)
            failed = True
            continue
        for section, shared, older in find_duplicates(
            path.read_text(encoding="utf-8")
        ):
            failed = True
            print(
                f"::error file={path}::Two entries under '## {section}' "
                f"agree for {shared} characters — an amended entry kept "
                f"beside the revision it replaced. Keep the longer, drop "
                f"the shorter. Starts: {older[:120]}",
                file=sys.stderr,
            )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
