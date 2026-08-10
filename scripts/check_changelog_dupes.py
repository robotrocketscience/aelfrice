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

## Directories, and why comparison is global (#1475)

An argument may be a directory — `CHANGELOG/unreleased/`, one file per
entry. Every `*.md` in it except `README.md` is an entry file.

Those files are checked *together with* the `[Unreleased]` block of the
changelog files given in the same invocation, not one file at a time.
Two reasons, both failure modes this check exists to catch:

- Two PRs can add near-identical entries as two separate files. Per
  file there is one entry and never a duplicate, so a per-file check
  would report a pass having examined nothing that could fail — the
  #1160 defect class.
- During the transition an entry can exist both in the `[Unreleased]`
  block and as a file. Collation emits both, so the duplicate reaches
  the released section.

Entries from an entry file are therefore bucketed under `[Unreleased]`,
which is where collation will put them.

Usage: check_changelog_dupes.py CHANGELOG/v4.md CHANGELOG/unreleased [...]
"""
from __future__ import annotations

import sys
from collections.abc import Iterable
from pathlib import Path

# Characters two entries must agree on before they are called the same
# entry. Between the widest incidental overlap observed (<120) and the
# narrowest real duplicate (496), so neither bound is tight.
PREFIX_THRESHOLD = 200

# Section an entry file's bullet belongs to: the block it will be
# collated into.
UNRELEASED_SECTION = "[Unreleased]"

# The directory note, not an entry (it also keeps the directory
# tracked — git stores no empty trees).
README_NAME = "README.md"


def _common_prefix_len(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _collect(
    text: str,
    source: str,
    buckets: dict[str, list[tuple[str, str]]],
    section: str = "(preamble)",
) -> None:
    """Bucket `text`'s bullets by section, tagging each with `source`."""
    for line in text.split("\n"):
        if line.startswith("## "):
            section = line[3:].strip()
        elif line.startswith("- "):
            buckets.setdefault(section, []).append((line, source))


def find_duplicates_across(
    sources: Iterable[tuple[str, str, str]],
) -> list[tuple[str, int, str, str, str]]:
    """Compare every source against every other in one pass.

    `sources` is `(source_label, text, default_section)`. Returns
    `(section, shared_prefix_len, older_entry, older_source,
    other_source)` per restated pair.
    """
    buckets: dict[str, list[tuple[str, str]]] = {}
    for label, text, section in sources:
        _collect(text, label, buckets, section)

    found: list[tuple[str, int, str, str, str]] = []
    for name, entries in buckets.items():
        for i, (first, first_src) in enumerate(entries):
            for second, second_src in entries[i + 1:]:
                shared = _common_prefix_len(first, second)
                if shared >= PREFIX_THRESHOLD:
                    # Report the shorter: it is the superseded revision.
                    if len(second) < len(first):
                        older, older_src, other_src = (
                            second, second_src, first_src
                        )
                    else:
                        older, older_src, other_src = (
                            first, first_src, second_src
                        )
                    found.append(
                        (name, shared, older, older_src, other_src)
                    )
    return found


def find_duplicates(text: str) -> list[tuple[str, int, str]]:
    """Return (section, shared_prefix_len, entry) per restated pair."""
    return [
        (section, shared, older)
        for section, shared, older, _, _ in find_duplicates_across(
            [("(text)", text, "(preamble)")]
        )
    ]


def entry_files(directory: Path) -> list[Path]:
    """Entry files in `CHANGELOG/unreleased/`, in a stable order."""
    return sorted(
        (p for p in directory.glob("*.md") if p.name != README_NAME),
        key=lambda p: p.name,
    )


def main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv[1:]]
    if not paths:
        print("usage: check_changelog_dupes.py <changelog.md|dir> ...",
              file=sys.stderr)
        return 2

    failed = False
    sources: list[tuple[str, str, str]] = []
    for path in paths:
        if path.is_dir():
            # An empty directory is the expected steady state right
            # after a release, so it is not an error — but the caller
            # naming a directory that does not exist is.
            sources.extend(
                (str(f), f.read_text(encoding="utf-8"), UNRELEASED_SECTION)
                for f in entry_files(path)
            )
        elif path.is_file():
            sources.append(
                (str(path), path.read_text(encoding="utf-8"), "(preamble)")
            )
        else:
            print(f"::error file={path}::no such file or directory",
                  file=sys.stderr)
            failed = True

    for section, shared, older, older_src, other_src in (
        find_duplicates_across(sources)
    ):
        failed = True
        print(
            f"::error file={older_src}::Two entries under "
            f"'## {section}' agree for {shared} characters — an amended "
            f"entry kept beside the revision it replaced (the other is "
            f"in {other_src}). Keep the longer, drop the shorter. "
            f"Starts: {older[:120]}",
            file=sys.stderr,
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
