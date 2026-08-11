#!/usr/bin/env python3
"""Fail when merge-conflict residue is committed to a tracked file (#1491).

`CHANGELOG/v4.md` reached `main` carrying three marker lines. The suite
was green throughout: nothing in it opens the changelog to read its
shape, and the file-specific checks that do exist ask other questions —
`release-docs-check` asks whether `[Unreleased]` was drained,
`check_changelog_dupes.py` asks whether two entries restate each other.

The guard is deliberately repo-wide rather than changelog-shaped. The
same accident landed markers in `docs/design/write-log-as-truth.md`
during the #1362/#1378 rebase, and a guard scoped to the file that
happened to be hit last is a guard that catches the previous incident.

## Why two of the four markers are conditional

The open and close markers are unambiguous: seven `<` or seven `>`
followed by a space starts no legitimate line in this repo. The other
two are not. A line of exactly seven `=` is a Setext heading underline
in Markdown and a section rule in reStructuredText, and both appear in
`docs/`; seven `|` is a Markdown table row of six empty cells. So a
separator or a diff3 ancestor marker counts only in a file that already
carries an open or close marker.

That costs no real detection. Git writes all of a conflict's markers or
none — an ancestor marker never appears without the open and close ones
around it — so the only thing the condition gives up is a file someone
hand-cleaned by deleting exactly the two unambiguous lines.

The marker literals are built from repeated characters rather than
written out, so this file and its test do not trip the check they
implement. An exclusion list would have been the alternative, and an
exclusion list is a hole: the next file added to it is the next file
that can carry markers to `main` unseen.

Usage: check_conflict_markers.py [path ...]   (default: all tracked files)
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

OPEN_MARKER = "<" * 7 + " "
CLOSE_MARKER = ">" * 7 + " "
# diff3 conflict style emits this third marker for the common ancestor,
# bare from `git merge-file` and labelled from a merge or a
# `--conflict=diff3` checkout (`||||||| merged common ancestors`).
BASE_MARKER = "|" * 7
SEPARATOR = "=" * 7


def _is_base_marker(line: str) -> bool:
    """True for a diff3 ancestor marker, bare or labelled."""
    stripped = line.rstrip()
    return stripped == BASE_MARKER or stripped.startswith(BASE_MARKER + " ")


def find_markers(text: str) -> list[tuple[int, str]]:
    """Return `(line number, line)` for every conflict marker in `text`.

    Line numbers are 1-based.

    The open and close markers are the only unambiguous ones. Both the
    separator and the diff3 ancestor marker are reported only in a file
    that already carries one of those, because each has a legitimate
    reading on its own: seven `=` is a Setext heading underline and a
    reStructuredText section rule, and seven `|` is a Markdown table row
    of six empty cells. Neither ever occurs as merge residue without an
    open or close marker beside it — git writes all three or none — so
    the condition costs no real detection and removes both false
    positives.
    """
    lines = text.split("\n")
    unambiguous = [
        (n, line)
        for n, line in enumerate(lines, start=1)
        if line.startswith((OPEN_MARKER, CLOSE_MARKER))
    ]
    if not unambiguous:
        return []
    conditional = [
        (n, line)
        for n, line in enumerate(lines, start=1)
        if line.rstrip() == SEPARATOR or _is_base_marker(line)
    ]
    return sorted(unambiguous + conditional)


def tracked_files(repo: Path) -> list[Path]:
    """Every file git tracks, as absolute paths."""
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo, capture_output=True, text=True, check=True, timeout=60,
    ).stdout
    return [repo / name for name in out.split("\0") if name]


class UnreadableFile(Exception):
    """A tracked path that exists but could not be read at all.

    Raised rather than skipped. A skip is indistinguishable from a clean
    file in the summary, so an unreadable path would quietly shrink the
    scan's coverage while it went on reporting success.
    """


def _content_of(path: Path) -> str | None:
    """The text git stores for `path`, or `None` if it is binary.

    A symlink's content is its target string, so it is read with
    `readlink` rather than followed: following it would scan whatever
    the link points at — possibly outside the repository and not tracked
    at all — and a broken link would raise instead of being read.
    """
    try:
        if path.is_symlink():
            return os.readlink(path)
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None
    except OSError as exc:
        raise UnreadableFile(f"{path}: {exc}") from exc


def scan(paths: list[Path]) -> tuple[int, list[tuple[Path, int, str]]]:
    """Scan `paths`, returning `(files read, findings)`.

    Files that are not valid UTF-8 are binary as far as this check is
    concerned and are skipped without being counted, so a repository of
    nothing but images cannot report a vacuous pass. Anything else that
    cannot be read raises `UnreadableFile`.
    """
    read = 0
    findings: list[tuple[Path, int, str]] = []
    for path in paths:
        text = _content_of(path)
        if text is None:
            continue
        read += 1
        findings.extend((path, n, line) for n, line in find_markers(text))
    return read, findings


def main(argv: list[str]) -> int:
    repo = Path(__file__).resolve().parents[1]
    paths = [Path(a) for a in argv[1:]] or tracked_files(repo)

    try:
        read, findings = scan(paths)
    except UnreadableFile as exc:
        print(f"::error::unreadable tracked file: {exc}", file=sys.stderr)
        return 2
    if read == 0:
        print("::error::no readable files scanned", file=sys.stderr)
        return 2

    for path, lineno, line in findings:
        try:
            shown = path.relative_to(repo)
        except ValueError:
            shown = path
        print(
            f"::error file={shown},line={lineno}::merge-conflict marker "
            f"committed: {line[:80]}",
            file=sys.stderr,
        )
    if findings:
        print(
            f"{len(findings)} conflict marker line(s) in "
            f"{len({f[0] for f in findings})} file(s) of {read} scanned",
            file=sys.stderr,
        )
        return 1
    print(f"no conflict markers in {read} tracked text files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
