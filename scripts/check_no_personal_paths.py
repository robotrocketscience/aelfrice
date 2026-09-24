#!/usr/bin/env python3
"""Refuse absolute home-directory paths and tracked host dotdirs.

Usage:
    uv run python scripts/check_no_personal_paths.py            # scan tracked files
    uv run python scripts/check_no_personal_paths.py --range A..B # scan a diff's added lines
    uv run python scripts/check_no_personal_paths.py --dry-run  # report, always exit 0
    uv run python scripts/check_no_personal_paths.py --self-test # run the pattern fixtures

Exits non-zero on any finding, so it can gate a push or a PR.

Why this file exists (#1617). PR #1610 added `.gemini/settings.json`
carrying a developer's absolute venv path eleven times, to a PUBLIC
repo. All three pre-merge scan jobs — `secrets-scan`, `pattern-scan`,
`history-scan` — reported PASS on it. `secrets-scan` looks for
credentials and an absolute path is not one; the other two match against
patterns held in the `SCAN_RULES_B64` repository secret, which by
observation carries no home-directory rule.

Two design choices follow from that, and both are deliberate:

1. **The rule lives here, in the tree, not in a secret.** A gate whose
   patterns nobody can read is a gate nobody can audit or test. This one
   ships with fixtures (`--self-test`, and `tests/test_no_personal_paths.py`)
   so the pattern's behaviour is reviewable in a diff.
2. **It checks content AND tracking.** The path text is the disclosure,
   but the host dotdir is the vector: nothing in `.gitignore` stopped
   `.gemini/` being added in the first place.

The repo tracks no `.claude/`, `.gemini/` or `.codex/` directory and
never should; those are per-machine host configuration whose whole
content is local absolute paths.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Placeholder user segments that name no real person. Keep this list
# tight: every entry is a hole, so add one only for a form the repo
# actually uses.
PLACEHOLDER_USERS: tuple[str, ...] = (
    "runner",  # GitHub Actions runner home
    "synthetic",  # benchmarks/context-rebuilder fixture marker
    "ci",  # Windows CI fixture paths in tests/
    "user",
    "username",
    "you",
    "youruser",
    "your-user",
    "me",
    "someone",
    "example",
    "test",
    "tester",
    "developer",
    "dev",
    "alice",
    "bob",
    "USER",
    "USERNAME",
    "HOME",
)

_ALLOW = "|".join(re.escape(u) for u in PLACEHOLDER_USERS)

# A home root followed by a segment that is not a known placeholder and
# not a shell/template variable.
HOME_PATH_RE = re.compile(
    # Not mid-path: "relative/home/path" is not a home root.
    r"(?<![A-Za-z0-9_.\-])"
    # POSIX, or Windows with one or two backslashes (two = source-escaped).
    r"(?:/(?:home|Users)/|[A-Za-z]:\\{1,2}Users\\{1,2})"
    r"(?!(?:" + _ALLOW + r")(?:[/\\]|$))"
    r"(?![$<%*~{]|\.{1,2}(?:[/\\]|$))"
    r"[A-Za-z0-9][A-Za-z0-9._-]{1,31}(?:[/\\]|$)"
)

# Host configuration directories that must never be tracked.
FORBIDDEN_TRACKED_DIRS: tuple[str, ...] = (".claude/", ".gemini/", ".codex/")

# This file necessarily contains the pattern and its fixtures.
SELF = "scripts/check_no_personal_paths.py"
EXEMPT_FILES: frozenset[str] = frozenset({SELF, "tests/test_no_personal_paths.py"})

_BINARY_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".db", ".sqlite",
     ".sqlite3", ".woff", ".woff2", ".zip", ".gz", ".whl", ".so", ".dylib"}
)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout


def tracked_files() -> list[str]:
    return [f for f in _git("ls-files").splitlines() if f]


def forbidden_tracked() -> list[str]:
    """Return tracked paths inside a host configuration directory."""
    return [
        f
        for f in tracked_files()
        if any(f == d.rstrip("/") or f.startswith(d) for d in FORBIDDEN_TRACKED_DIRS)
    ]


def scan_text(path: str, text: str) -> list[tuple[str, int, str, str]]:
    out: list[tuple[str, int, str, str]] = []
    for n, line in enumerate(text.splitlines(), 1):
        m = HOME_PATH_RE.search(line)
        if m:
            out.append((path, n, m.group(0), line.strip()[:160]))
    return out


def scan_tracked() -> list[tuple[str, int, str, str]]:
    findings: list[tuple[str, int, str, str]] = []
    for f in tracked_files():
        if f in EXEMPT_FILES or Path(f).suffix.lower() in _BINARY_SUFFIXES:
            continue
        try:
            text = Path(f).read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError, OSError):
            continue
        findings.extend(scan_text(f, text))
    return findings


def scan_diff(rev_range: str) -> list[tuple[str, int, str, str]]:
    """Scan only lines a diff ADDS, over an arbitrary git range.

    A hit on a removed line is content being deleted, which shrinks the
    exposure rather than growing it, so only `+` lines count.

    `rev_range` is passed to `git diff` verbatim, so the caller chooses
    the semantics: `main...HEAD` for a PR, `<base>..<sha>` for the
    pre-push hook, or `<empty-tree>..<sha>` when a branch shares no
    ancestor with main and the whole content needs scanning.
    """
    diff = _git("diff", "--unified=0", rev_range)
    findings: list[tuple[str, int, str, str]] = []
    current = "?"
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            current = line[6:]
            continue
        if not line.startswith("+") or line.startswith("+++"):
            continue
        if current in EXEMPT_FILES:
            continue
        m = HOME_PATH_RE.search(line[1:])
        if m:
            findings.append((current, 0, m.group(0), line[1:].strip()[:160]))
    return findings


_POSITIVE = (
    "/home/jdoe/projects/aelfrice/.venv/bin/aelf-hook",  # the #1610 shape
    "/Users/someone1/x",
    "C:\\Users\\Jonathan\\AppData",
    "C:\\\\Users\\\\Jonathan\\\\AppData",
    'File "/home/jdoe/x.py", line 4',
)
_NEGATIVE = (
    "/home/runner/work/aelfrice/aelfrice",
    "/Users/synthetic/aelfrice",
    "C:\\Users\\ci\\aelf.exe",
    "/home/user/project",
    "/Users/you/project",
    "/home/$USER/x",
    "/Users/<user>/x",
    "~/.aelfrice/memory.db",
    "$HOME/.config",
    "relative/home/path/x",
    "/home/",
    "/usr/local/bin/aelf",
)


def self_test() -> int:
    bad = 0
    for s in _POSITIVE:
        if not HOME_PATH_RE.search(s):
            print(f"SELF-TEST MISS (should match): {s}")
            bad += 1
    for s in _NEGATIVE:
        m = HOME_PATH_RE.search(s)
        if m:
            print(f"SELF-TEST FIRE (should not match): {s} -> {m.group(0)!r}")
            bad += 1
    print(
        f"self-test: {len(_POSITIVE)} positives, {len(_NEGATIVE)} negatives, "
        f"{bad} failures"
    )
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--range",
        metavar="REV_RANGE",
        dest="rev_range",
        help="scan only lines ADDED by this git range (e.g. main...HEAD)",
    )
    ap.add_argument("--dry-run", action="store_true", help="report but exit 0")
    ap.add_argument("--self-test", action="store_true", help="run pattern fixtures")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    findings = scan_diff(args.rev_range) if args.rev_range else scan_tracked()
    tracked_dotdirs = forbidden_tracked()

    for path, line, hit, ctx in findings:
        where = f"{path}:{line}" if line else path
        print(f"::error::personal path {hit!r} at {where}")
        print(f"    {ctx}")
    for f in tracked_dotdirs:
        print(f"::error::host configuration directory must not be tracked: {f}")

    n = len(findings) + len(tracked_dotdirs)
    if n:
        print(
            f"\n{n} finding(s). These publish a machine-local identity to a "
            f"public repo.\nReplace an absolute path with `~/` or a "
            f"placeholder such as /home/user, and never track "
            f"{', '.join(FORBIDDEN_TRACKED_DIRS)}."
        )
    else:
        scope = (
            f"added lines in {args.rev_range}"
            if args.rev_range
            else f"{len(tracked_files())} tracked files"
        )
        print(f"clean: no personal paths, no tracked host dotdirs ({scope}).")

    return 0 if args.dry_run else (1 if n else 0)


if __name__ == "__main__":
    sys.exit(main())
