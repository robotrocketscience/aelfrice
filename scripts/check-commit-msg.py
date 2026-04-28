#!/usr/bin/env python3
"""Validate a commit message against the conventional-commit prefix list.

Usage (commit-msg hook):
    python scripts/check-commit-msg.py <commit-msg-file>

Usage (CI — validate a single subject line from stdin or argument):
    python scripts/check-commit-msg.py --subject "feat: add thing"

Exit codes:
    0  valid
    1  invalid prefix (message written to stderr)
    2  usage error
"""
from __future__ import annotations

import re
import sys

# Allowed conventional-commit type tokens.
# Keep in sync with CLAUDE.md and CONTRIBUTING.md.
ALLOWED_PREFIXES = (
    "feat",
    "fix",
    "perf",
    "refactor",
    "test",
    "docs",
    "build",
    "ci",
    "style",
    "revert",
    "exp",
    "chore",
    "release",
    "gate",
    "audit",
)

# Matches:  <type>[(scope)][!]: <subject>
# Examples: feat: add thing
#           feat(scope): add thing
#           feat!: breaking
#           feat(scope)!: breaking
_TYPE_PART = "|".join(re.escape(p) for p in ALLOWED_PREFIXES)
VALID_PREFIX_RE = re.compile(
    rf"^(?:{_TYPE_PART})"  # required type token
    r"(?:\([^)]+\))?"      # optional (scope)
    r"!?"                  # optional ! for breaking change
    r": ",                 # colon + space
)

# Subjects that git generates automatically and must not be rejected.
AUTO_SUBJECT_PREFIXES = ("Merge ", "Revert ")


def is_exempt(subject: str) -> bool:
    """Return True for subjects that should bypass prefix enforcement."""
    return any(subject.startswith(p) for p in AUTO_SUBJECT_PREFIXES)


def validate_subject(subject: str) -> bool:
    """Return True if *subject* satisfies the conventional-commit contract."""
    subject = subject.strip()
    if not subject:
        return False
    if is_exempt(subject):
        return True
    return bool(VALID_PREFIX_RE.match(subject))


def _first_line(text: str) -> str:
    return text.split("\n", 1)[0].strip()


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv

    if not args:
        print(
            "usage: check-commit-msg.py <commit-msg-file>\n"
            "       check-commit-msg.py --subject <subject>",
            file=sys.stderr,
        )
        return 2

    if args[0] == "--subject":
        if len(args) < 2:
            print("error: --subject requires an argument", file=sys.stderr)
            return 2
        subject = args[1]
    else:
        # Treat first argument as a file path (commit-msg hook convention).
        try:
            subject = _first_line(open(args[0]).read())
        except OSError as exc:
            print(f"error: cannot read {args[0]!r}: {exc}", file=sys.stderr)
            return 2

    if validate_subject(subject):
        return 0

    allowed = ", ".join(f"{p}:" for p in ALLOWED_PREFIXES)
    print(
        f"error: commit subject does not start with a valid conventional-commit prefix.\n"
        f"  subject : {subject!r}\n"
        f"  allowed : {allowed}\n"
        f"  pattern : <type>[(scope)][!]: <description>\n"
        f"  example : feat(cli): add --verbose flag\n"
        f"\n"
        f"Merge and Revert auto-subjects are exempt.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
