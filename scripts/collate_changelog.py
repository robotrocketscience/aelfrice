#!/usr/bin/env python3
"""Collate `CHANGELOG/unreleased/*.md` into a dated section (#1475).

Thirteen of fourteen open PRs on the 2026-08-10 board inserted into
`CHANGELOG/v4.md` inside lines 8-16, so every merge forced a hand
resolution on every remaining PR. Entries are single lines of
2,000-4,500 characters: git offers two 4 KB lines with no intra-line
granularity, and an entry dropped during that resolution is invisible
in the diff. One file per entry replaces that with add/add on distinct
paths, which never conflicts.

The cost of the change is honest and worth stating: it moves the
failure from merge time (loud — a conflict you cannot commit through)
to release time (quiet — an entry that silently never appears). The two
ways collation can lose an entry are pinned by
`tests/test_collate_changelog.py`: exactly one entry per input file
reaches the output, and the directory is empty afterwards.

## Entry-file format

    CHANGELOG/unreleased/<issue>-<slug>.md

    ### Fixed

    - **Title ([#1475](https://github.com/.../issues/1475)).** Body...

Exactly one `### <Category>` heading and exactly one top-level `- `
bullet per file — anything else is an error, not a guess. Indented
continuation paragraphs under the bullet are preserved verbatim.
Category must be one already in use in the changelog — the
Keep-a-Changelog six plus the house additions listed in
`CATEGORIES`.

The directory is flat and holds nothing else. Any other path under it
— a different suffix, an extensionless file, an uppercase `.MD`, a
subdirectory — is an error naming the path, never a skip: a file
collation would not collect is one the release would omit without a
word. `release-docs-check`'s leftover scan is deliberately broader
than the acceptance rule here, so the same path is loud there too.

## Ordering rule (deterministic, filesystem-independent)

1. Categories in `CATEGORIES` order — the Keep-a-Changelog six
   first, then the house additions. Empty categories are omitted.
2. Within a category, entries already in the `[Unreleased]` block come
   first, in the order they appear in the changelog file. This is what
   makes the transition non-breaking: an in-flight PR that edits the
   block instead of adding a file still releases correctly, and never
   has to rebase onto the new convention to merge.
3. Then the entry files, sorted by file *name* with `sorted()` — a
   plain byte-wise comparison of the name, never `glob`/`listdir`
   order, never mtime, never full path (which would make the result
   depend on where the repo is checked out).

Usage:

    collate_changelog.py --version 4.3.0 --date 2026-08-10 [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Keep a Changelog 1.1.0 §types-of-changes in its published order,
# then the house categories already in use across CHANGELOG/v0-v4.md
# and the index (counted there: Documentation 7, Performance 3,
# Internal 2, Reverted / Notes / Dependencies / CI / Build 1 each).
# Enumerated rather than accept-anything because the order has to be
# total: an unrecognised heading has nowhere deterministic to go. It is
# rejected on the PR that introduces it — `tests/test_collate_changelog
# .py::test_the_committed_unreleased_directory_parses` parses every
# committed entry file on every run — and not, quietly, at release.
CATEGORIES: tuple[str, ...] = (
    "Added",
    "Changed",
    "Deprecated",
    "Removed",
    "Fixed",
    "Security",
    "Performance",
    "Documentation",
    "Build",
    "CI",
    "Dependencies",
    "Internal",
    "Reverted",
    "Notes",
)

UNRELEASED_HEADER = "## [Unreleased]"

# Not an entry: the directory's own note, which also keeps the
# otherwise-empty directory tracked in git (git stores no empty trees,
# so without it the first PR to use the convention would have to create
# the directory as well — and two PRs doing that both add the same
# `.gitkeep`, which is exactly the add/add collision being removed).
README_NAME = "README.md"


class CollationError(Exception):
    """An entry file or `[Unreleased]` block that cannot be collated."""


def scan_entry_dir(directory: Path) -> tuple[list[Path], list[Path]]:
    """Partition `directory` into `(entry files, everything else)`.

    Walks the whole tree with `iterdir` instead of globbing `*.md` one
    level deep. A glob is a filter, and a filter is silent about what
    it drops: `notes.txt`, `1475-slug.markdown`, an extensionless file,
    an uppercase `.MD` on a case-sensitive filesystem, and anything
    inside a subdirectory were invisible to collation, to the release
    gate's leftover scan and to the duplicate check at once — never
    collated, never reported as stranded, never compared. The entry
    such a file holds ships as nothing at all, which is precisely the
    quiet release-time failure this convention has to buy off.

    **A subdirectory is an error, not a place to collate from.** The
    convention is one flat directory of `<issue>-<slug>.md`: collation
    order is by file *name*, which two same-named files in different
    subdirectories cannot supply unambiguously, and a grouping
    directory would be a second convention nobody wrote down. The walk
    recurses anyway so the hidden entry is named, not just the
    directory hiding it.

    A missing directory is an error too, not an empty one. `Path.glob`
    on a path that does not exist returns `[]` without complaint, so a
    typo'd `--unreleased` or a cut run from the wrong working directory
    would collate nothing, exit 0 and report the release drained.
    `check_changelog_dupes.py` already refuses a directory argument
    that does not exist; this matches it.

    Both lists are ordered deterministically — entries by name (the
    collation order), the rest by path relative to `directory` — so no
    diagnostic depends on filesystem iteration order either.
    """
    if not directory.is_dir():
        raise CollationError(
            f"{directory}: no such directory. Nothing was collated — "
            f"an empty result here means a wrong --unreleased path, "
            f"not a release with no entries."
        )
    entries: list[Path] = []
    stray: list[Path] = []
    pending = [directory]
    while pending:
        for path in pending.pop().iterdir():
            relative = path.relative_to(directory)
            if path.is_dir() and not path.is_symlink():
                stray.append(path)
                pending.append(path)
            elif relative.as_posix() == README_NAME:
                continue
            elif len(relative.parts) == 1 and path.suffix == ".md":
                entries.append(path)
            else:
                stray.append(path)
    return (
        sorted(entries, key=lambda p: p.name),
        sorted(stray, key=lambda p: p.relative_to(directory).as_posix()),
    )


def entry_files(directory: Path) -> list[Path]:
    """Entry files in collation order — sorted by name, not by glob.

    Refuses rather than skips: anything `scan_entry_dir` could not
    classify as an entry file is an error naming the path, because the
    alternative is a release that quietly omits it.
    """
    entries, stray = scan_entry_dir(directory)
    if stray:
        raise CollationError(
            f"{directory}: not entry files, and never collated — "
            + ", ".join(str(path) for path in stray)
            + f". An entry is a top-level '<issue>-<slug>.md' (plus the "
            f"'{README_NAME}' note); a subdirectory, another suffix or "
            f"an uppercase '.MD' would ship as nothing at all. Rename "
            f"or move it."
        )
    return entries


def parse_entry_file(text: str, source: str) -> tuple[str, str]:
    """Return `(category, entry)` for one entry file.

    `entry` is the bullet block verbatim, trailing whitespace stripped,
    so collation is a re-arrangement and never a re-wrap.
    """
    lines = text.split("\n")
    categories = [
        line[4:].strip() for line in lines if line.startswith("### ")
    ]
    if len(categories) != 1:
        raise CollationError(
            f"{source}: expected exactly one '### <Category>' heading, "
            f"found {len(categories)}"
        )
    category = categories[0]
    if category not in CATEGORIES:
        raise CollationError(
            f"{source}: unknown category '{category}' — expected one of "
            + ", ".join(CATEGORIES)
        )

    starts = [i for i, line in enumerate(lines) if line.startswith("- ")]
    if len(starts) != 1:
        raise CollationError(
            f"{source}: expected exactly one top-level '- ' entry, found "
            f"{len(starts)}. One file per entry is the whole point: two "
            f"entries in one file re-open the collision this replaces."
        )
    entry = "\n".join(lines[starts[0]:]).rstrip()
    return category, entry


def parse_unreleased(text: str) -> tuple[dict[str, list[str]], int, int]:
    """Split the `[Unreleased]` block into per-category entry blocks.

    Returns `(entries, start, end)` where `start`/`end` are line indices
    bounding the block (`end` exclusive, at the next `## [` header or
    end of file), so the caller can splice without touching the rest of
    the file.
    """
    lines = text.split("\n")
    try:
        start = next(
            i for i, line in enumerate(lines)
            if line.startswith(UNRELEASED_HEADER)
        )
    except StopIteration:
        raise CollationError(
            f"no '{UNRELEASED_HEADER}' header in the changelog"
        ) from None
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("## ["):
            end = i
            break

    entries: dict[str, list[str]] = {}
    category: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            assert category is not None
            entries.setdefault(category, []).append(
                "\n".join(buffer).rstrip()
            )
            buffer.clear()

    for line in lines[start + 1:end]:
        if line.startswith("### "):
            flush()
            category = line[4:].strip()
            if category not in CATEGORIES:
                raise CollationError(
                    f"[Unreleased] has unknown category '{category}'"
                )
        elif line.startswith("- "):
            flush()
            if category is None:
                raise CollationError(
                    "[Unreleased] has an entry before any '### <Category>'"
                )
            buffer.append(line)
        elif buffer:
            buffer.append(line)
    flush()
    return entries, start, end


def render_section(
    version: str, date: str, entries: dict[str, list[str]]
) -> str:
    """Render the dated version section. No trailing newline."""
    out = [f"## [{version}] - {date}"]
    for category in CATEGORIES:
        block = entries.get(category)
        if not block:
            continue
        out.append("")
        out.append(f"### {category}")
        out.append("")
        out.extend(block)
    return "\n".join(out)


def collate(
    changelog_text: str,
    files: list[tuple[str, str]],
    version: str,
    date: str,
) -> str:
    """Return the changelog with `[Unreleased]` drained into `version`.

    `files` is `(name, text)` per entry file, already in collation
    order. The `[Unreleased]` header survives, emptied, so the next
    cycle has somewhere to land.
    """
    block, start, end = parse_unreleased(changelog_text)
    merged = {category: list(block.get(category, ())) for category in
              CATEGORIES}
    for name, text in files:
        category, entry = parse_entry_file(text, name)
        merged[category].append(entry)

    lines = changelog_text.split("\n")
    section = render_section(version, date, merged)
    replacement = [UNRELEASED_HEADER, "", *section.split("\n"), ""]
    return "\n".join(lines[:start] + replacement + lines[end:])


def cut(
    changelog: Path, directory: Path, version: str, date: str,
    dry_run: bool = False,
) -> str:
    """Collate and, unless `dry_run`, write the file and empty the dir."""
    paths = entry_files(directory)
    files = [(p.name, p.read_text(encoding="utf-8")) for p in paths]
    text = collate(
        changelog.read_text(encoding="utf-8"), files, version, date
    )
    if not dry_run:
        changelog.write_text(text, encoding="utf-8")
        # Unconditional: an entry file left behind is re-collated into
        # the *next* release as a duplicate of one already shipped.
        for path in paths:
            path.unlink()
    return text


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--version", required=True, help="e.g. 4.3.0")
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--changelog", type=Path, default=None,
        help="default: CHANGELOG/v<major>.md",
    )
    parser.add_argument(
        "--unreleased", type=Path, default=Path("CHANGELOG/unreleased"),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the collated changelog; touch nothing",
    )
    args = parser.parse_args(argv[1:])

    changelog = args.changelog or Path(
        f"CHANGELOG/v{args.version.split('.')[0]}.md"
    )
    try:
        text = cut(
            changelog, args.unreleased, args.version, args.date,
            dry_run=args.dry_run,
        )
    except (CollationError, OSError) as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        sys.stdout.write(text)
    else:
        print(
            f"{changelog}: [Unreleased] drained into [{args.version}]. "
            f"Still to do by hand: the compare-link footnote "
            f"'[{args.version}]: https://github.com/...' at the bottom "
            f"of the file (release-docs-check enforces it).",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
