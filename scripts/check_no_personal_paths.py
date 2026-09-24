#!/usr/bin/env python3
"""Refuse absolute home-directory paths and tracked host dotdirs.

Usage:
    uv run python scripts/check_no_personal_paths.py             # tracked files
    uv run python scripts/check_no_personal_paths.py --range A..B  # added lines
    uv run python scripts/check_no_personal_paths.py --dry-run   # report, exit 0
    uv run python scripts/check_no_personal_paths.py --self-test # pattern fixtures

Exits non-zero on any finding, so it can gate a push or a PR.

Why this file exists (#1617). PR #1610 added `.gemini/settings.json`
carrying a developer's absolute venv path twelve times, to a PUBLIC
repo. All three pre-merge scan jobs — `secrets-scan`, `pattern-scan`,
`history-scan` — reported PASS on it. `secrets-scan` looks for
credentials and an absolute path is not one; the other two match against
patterns held in the `SCAN_RULES_B64` repository secret, which by
observation carries no home-directory rule.

Two design choices follow, and both are deliberate:

1. **The rule lives here, in the tree, not in a secret.** A gate whose
   patterns nobody can read is a gate nobody can audit, test, or
   regression-guard. This one ships fixtures (`--self-test`, and
   `tests/test_no_personal_paths.py`) so the pattern's behaviour is
   reviewable in the diff that changes it.
2. **It checks content AND tracking.** The path text is the disclosure,
   but the host dotdir is the vector: nothing in `.gitignore` stopped
   `.gemini/` being added in the first place.

Deliberately NOT a general personal-data scanner. It catches one high-frequency,
high-confidence shape, and it carries two escape hatches so that being
wrong costs an annotation rather than an override habit.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Placeholder user segments that name no real person.
#
# Every entry is a hole, so each is justified by a form the tree actually
# uses or a convention a reader would recognise. Read the third group
# before adding to it: those names are also real account names, and they
# are here because refusing them was measured against this tree and cost
# more than it bought.
PLACEHOLDER_USERS: tuple[str, ...] = (
    "runner",  # GitHub Actions runner home
    "synthetic",  # benchmarks/context-rebuilder fixture marker
    "ci",  # Windows fixture paths in tests/test_codex_skills.py
    # Documentation placeholders: these read as an instruction to the
    # reader, never as a real account.
    "user",
    "username",
    "you",
    "youruser",
    "your-user",
    "yourname",
    "USER",
    "USERNAME",
    "HOME",
    # Container and CI service accounts: conventional, not personal.
    "node",
    "ubuntu",
    "ec2-user",
    "pi",
    "appuser",
    "vagrant",
    "circleci",
    "jenkins",
    "linuxbrew",
    # Well-known shared Windows/macOS profiles.
    "Public",
    "Shared",
    "Default",
    "Administrator",
    # Synthetic names this tree already uses in fixtures and docstrings.
    # These are a KNOWN, ACCEPTED HOLE: `dev` and `alice` are also real
    # account names, so a genuine `/home/dev/...` disclosure would pass.
    # The trade was measured, not assumed — refusing them flagged 61
    # legitimate lines across 20 files, and a gate that noisy gets
    # switched off, which costs more than the hole. The diff-mode scan
    # plus review is the defence for this class.
    "first",
    "First",
    "last",
    "Last",
    "Middle",
    "Ana",
    "alice",
    "bob",
    "dev",
    "me",
    "someone",
    "example",
    "test",
    "tester",
    "developer",
    "myuser",
)

_ALLOW = "|".join(re.escape(u) for u in PLACEHOLDER_USERS)

# A home root followed by a segment that is not a known placeholder and
# not a shell or template variable.
#
# The trailing context is a BOUNDARY, `(?![A-Za-z0-9._-])`, not a
# required separator. Requiring `[/\\]` after the username missed a bare
# `"/Users/jdoe"` in a quoted config value — the same file class as
# #1610, so that miss mattered.
#
# The user segment needs two characters or more: single-character
# segments (`/home/u`, `/Users/x`) are this tree's fixture convention,
# and requiring one character flagged 61 legitimate lines.
HOME_PATH_RE = re.compile(
    # Not mid-path: "relative/home/path" is not a home root.
    r"(?<![A-Za-z0-9_.\-])"
    # POSIX (optionally JSON-escaped `\/`), or Windows with one or two
    # backslashes (two = source- or JSON-escaped).
    r"(?:\\?/(?:home|Users)\\?/|[A-Za-z]:\\{1,2}Users\\{1,2})"
    r"(?!(?:" + _ALLOW + r")(?![A-Za-z0-9_-]))"
    r"(?![$<%*~{]|\.{1,2}(?![A-Za-z0-9._-]))"
    r"[A-Za-z0-9][A-Za-z0-9._-]{1,31}"
    r"(?![A-Za-z0-9._-])"
)

# Host configuration directories that must never be tracked, at ANY
# depth. An earlier `startswith(".claude/")` caught only the repo root
# while `.gitignore` covers every level — a backstop must not be weaker
# than the thing it backs up.
FORBIDDEN_TRACKED_DIRS: tuple[str, ...] = (".claude", ".gemini", ".codex")

# This file and its test necessarily contain the shapes they match on.
SELF = "scripts/check_no_personal_paths.py"
EXEMPT_FILES: frozenset[str] = frozenset({SELF, "tests/test_no_personal_paths.py"})

# Escape hatches, because a gate with no override gets disabled wholesale
# and an override habit is how a real block gets waved through.
#   - a glob in `.github-pii-exempt` skips a whole file
#   - this marker on the line itself skips that line
# Both are visible in review, which a silent bypass would not be.
INLINE_ALLOW = "pii-allow"
EXEMPT_GLOB_FILE = ".github-pii-exempt"

_BINARY_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".woff", ".woff2",
     ".zip", ".gz", ".whl", ".so", ".dylib", ".ttf", ".otf", ".mp4"}
)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout


class UnreadableRange(RuntimeError):
    """The diff could not be read (usually a shallow checkout)."""


def tracked_files() -> list[str]:
    return [f for f in _git("ls-files").splitlines() if f]


def exempt_globs() -> list[str]:
    p = Path(EXEMPT_GLOB_FILE)
    if not p.is_file():
        return []
    return [
        ln.strip()
        for ln in p.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]


def _is_exempt(path: str, globs: list[str]) -> bool:
    if path in EXEMPT_FILES:
        return True
    return any(Path(path).match(g) for g in globs)


def forbidden_tracked() -> list[str]:
    """Tracked paths inside a host configuration directory, at any depth."""
    out: list[str] = []
    for f in tracked_files():
        if any(seg.lower() in FORBIDDEN_TRACKED_DIRS for seg in Path(f).parts):
            out.append(f)
    return out


def scan_text(path: str, text: str) -> list[tuple[str, int, str, str]]:
    out: list[tuple[str, int, str, str]] = []
    for n, line in enumerate(text.splitlines(), 1):
        if INLINE_ALLOW in line:
            continue
        m = HOME_PATH_RE.search(line)
        if m:
            out.append((path, n, m.group(0), line.strip()[:160]))
    return out


def _read_blobs(paths: list[str]) -> dict[str, str]:
    """Return `{path: text}` for tracked paths, read from the INDEX.

    Reads blobs rather than the working tree: a tracked symlink would
    otherwise be followed off the repository, so the check's result would
    depend on the host filesystem instead of on the commit — a
    determinism violation (#605), and one that made an early draft report
    findings out of a developer's shell profile.

    ONE `git cat-file --batch` for the whole tree, not one `git show` per
    file. The per-file form measured 74.6 ms/file, 85.9s over 1,151
    files, which blew the test's 60s budget; batching is 1 fork.

    Non-UTF-8 content is decoded latin-1 rather than skipped, so an
    encoding cannot hide a path.
    """
    if not paths:
        return {}
    # `<mode> <sha> <stage>\t<path>` — mode 120000 is a symlink, whose
    # blob is its target string. Scanning that is right: a symlink
    # pointing into a home directory publishes the path just as a file
    # containing it would.
    listing = _git("ls-files", "-s", "-z").split("\0")
    sha_for: dict[str, str] = {}
    for entry in listing:
        if not entry or "\t" not in entry:
            continue
        meta, _, path = entry.partition("\t")
        parts = meta.split()
        if len(parts) >= 2:
            sha_for[path] = parts[1]

    wanted = [(p, sha_for[p]) for p in paths if p in sha_for]
    if not wanted:
        return {}
    proc = subprocess.run(
        ["git", "cat-file", "--batch"],
        input="\n".join(sha for _, sha in wanted).encode() + b"\n",
        capture_output=True,
        check=True,
    )
    out = proc.stdout
    result: dict[str, str] = {}
    pos = 0
    for path, _sha in wanted:
        nl = out.find(b"\n", pos)
        if nl == -1:
            break
        header = out[pos:nl].split()
        if len(header) < 3:  # "<sha> missing"
            pos = nl + 1
            continue
        try:
            size = int(header[2])
        except ValueError:
            pos = nl + 1
            continue
        raw = out[nl + 1 : nl + 1 + size]
        pos = nl + 1 + size + 1  # trailing newline
        try:
            result[path] = raw.decode("utf-8")
        except UnicodeDecodeError:
            result[path] = raw.decode("latin-1", errors="replace")
    return result


def scan_tracked() -> list[tuple[str, int, str, str]]:
    globs = exempt_globs()
    candidates = [
        f
        for f in tracked_files()
        if not _is_exempt(f, globs) and Path(f).suffix.lower() not in _BINARY_SUFFIXES
    ]
    blobs = _read_blobs(candidates)
    findings: list[tuple[str, int, str, str]] = []
    for f in candidates:
        text = blobs.get(f)
        if text is None:
            continue
        findings.extend(scan_text(f, text))
    return findings


def scan_diff(rev_range: str) -> list[tuple[str, int, str, str]]:
    """Scan only lines a diff ADDS, over an arbitrary git range.

    A hit on a removed line is content being deleted, which shrinks the
    exposure rather than growing it, so only `+` lines count.
    """
    try:
        diff = _git("diff", "--unified=0", rev_range)
    except subprocess.CalledProcessError as exc:
        raise UnreadableRange(
            f"could not read {rev_range!r} (shallow checkout? fetch the "
            f"base ref first): {(exc.stderr or '').strip()[:200]}"
        ) from exc
    globs = exempt_globs()
    findings: list[tuple[str, int, str, str]] = []
    current = "?"
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            current = line[6:]
            continue
        if not line.startswith("+") or line.startswith("+++"):
            continue
        if _is_exempt(current, globs):
            continue
        body = line[1:]
        if INLINE_ALLOW in body:
            continue
        m = HOME_PATH_RE.search(body)
        if m:
            findings.append((current, 0, m.group(0), body.strip()[:160]))
    return findings


_POSITIVE = (
    "/home/jdoe/projects/aelfrice/.venv/bin/aelf-hook",  # the #1610 shape
    '"venvPath": "/Users/jdoe"',  # bare root, no trailing separator
    '"path":"\\/Users\\/jdoe\\/proj"',  # JSON-escaped slashes
    "C:\\Users\\Jdoe\\AppData",
    "C:\\\\Users\\\\Jdoe\\\\AppData",
    'File "/home/mmueller/x.py", line 4',
)
_NEGATIVE = (
    "/home/u/proj",  # single character: a fixture convention here
    # KNOWN HOLE, recorded rather than hidden: `dev` is allowlisted
    # because this tree uses it in Windows fixtures, so a real
    # account of that name would pass. See PLACEHOLDER_USERS.
    "/home/dev/projects/x",
    "/Users/x/y",
    "/home/runner/work/aelfrice/aelfrice",
    "/Users/synthetic/aelfrice",
    "C:\\Users\\ci\\aelf.exe",
    "/home/user/project",
    "/Users/you/project",
    "/Users/yourname/projects/aelfrice",
    "/home/node/app",
    "/home/ubuntu/deploy",
    "C:\\Users\\Public\\x",
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

    try:
        findings = scan_diff(args.rev_range) if args.rev_range else scan_tracked()
    except UnreadableRange as exc:
        # Fail closed: reporting "clean" for a diff we could not read
        # would make the gate decorative exactly where it matters.
        print(f"::error::{exc}")
        return 0 if args.dry_run else 1

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
            f"{', '.join(FORBIDDEN_TRACKED_DIRS)}.\n"
            f"If a hit is genuinely fine, put `{INLINE_ALLOW}` on that line "
            f"or a glob in {EXEMPT_GLOB_FILE} — both are visible in review."
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
