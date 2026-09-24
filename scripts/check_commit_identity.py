#!/usr/bin/env python3
"""Refuse commits whose author or committer email is not a noreply form.

Usage:
    uv run python scripts/check_commit_identity.py --range A..B
    uv run python scripts/check_commit_identity.py --range A..B --dry-run
    uv run python scripts/check_commit_identity.py --self-test

Exits non-zero on any commit carrying a routable email address, so it
can gate a push or a PR.

Why this file exists (#1617). Two commits reached `main` carrying a
contributor's real name and corporate email as the git author, while
the committer was the usual noreply identity — a machine whose
`user.email` was never set to the privacy form. Across the same
367-commit window, 364 commits used
`<id>+<login>@users.noreply.github.com` and one was Dependabot, so
those two are an anomaly rather than a convention.

Commit metadata is the worst place to put an address. It is copied into
every clone and every fork, served by the unauthenticated REST API, and
pinned by `refs/pull/<n>/head` — so unlike file content it cannot be
taken back by editing a file, and a history rewrite does not reach the
pull refs or the fork network either. The only real control is refusing
it before it is pushed, which is what this does.

GitHub's noreply forms are the accepted way to author publicly without
publishing a mailbox: `<login>@users.noreply.github.com`,
`<id>+<login>@users.noreply.github.com`, and `noreply@github.com` for
GitHub's own web-flow committer. Enable "Keep my email address private"
in GitHub account settings to get one by default.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

# The accepted no-reply shapes. Anchored: a routable address that merely
# ends in a lookalike domain must not pass.
# The local part must allow `+` (the `<id>+<login>` form GitHub issues)
# and `[` `]` (bot accounts such as `dependabot[bot]`). Getting this
# wrong fails closed and blocks every push, which is why both shapes are
# in the fixtures below.
NOREPLY_RE = re.compile(
    r"^(?:"
    r"[A-Za-z0-9][A-Za-z0-9._+\[\]-]*@users\.noreply\.github\.com"
    r"|noreply@github\.com"
    r")$"
)

# The disclosure that motivated this check was a real name AND a routable
# address. GitHub's no-reply forms anonymise the address and do nothing
# about `%an`, so an email-only gate would refuse half of what it was
# written for. A generic "is this a real name" test does not exist, so
# this repo states who it publishes as: every commit not from this set is
# refused, and the set is visible in the diff that changes it.
#
# 1,107 of 1,109 commits already use one of these. Add a contributor here
# deliberately, with the same care as widening any other allowlist.
ALLOWED_AUTHOR_NAMES: frozenset[str] = frozenset(
    {
        "rrs",
        "robotrocketscience",
        "dependabot[bot]",
        "github-actions[bot]",
        "GitHub",
        # Vendor automation identities. Not personal data — no human is
        # named — which is the distinction this gate actually cares
        # about. Already present on `main`, so refusing them would fail
        # the check on history rather than on new work.
        "Gemini CLI",
    }
)

# Addresses that are routable but name no person. The email rule exists
# to stop a personal mailbox reaching a public commit; a vendor's
# automation address discloses nobody, and these are already on `main`.
# An explicit short list rather than a domain wildcard, so widening it
# is a visible act in the diff.
ALLOWED_TOOL_EMAILS: frozenset[str] = frozenset(
    {
        "gemini-cli@google.com",
    }
)

# An address in the name field is a leak wearing the wrong hat, and no
# legitimate display name contains one.
_EMAIL_IN_NAME_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def email_is_allowed(email: str) -> bool:
    """True for a no-reply form, or a listed vendor automation address."""
    return bool(NOREPLY_RE.match(email)) or email in ALLOWED_TOOL_EMAILS


def name_is_allowed(name: str) -> bool:
    """True when `name` is one this repository publishes under."""
    if _EMAIL_IN_NAME_RE.search(name):
        return False
    return name.strip() in ALLOWED_AUTHOR_NAMES


_ACCEPTED = (
    "276464689+robotrocketscience@users.noreply.github.com",
    "49699333+dependabot[bot]@users.noreply.github.com",
    "octocat@users.noreply.github.com",
    "noreply@github.com",
)
_REFUSED = (
    "jane.doe@example.org",
    "someone@gmail.com",
    "dev@corp.io",
    "attacker@users.noreply.github.com.evil.test",
    "prefix noreply@github.com",
    "",
)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout


class UnreadableRange(RuntimeError):
    """The revision walk could not read the range that was asked for.

    Usually a shallow checkout: the CI checkout action defaults to depth
    1, so `HEAD~1..HEAD` resolves to nothing there even though it is fine
    in a full clone. Raised rather than swallowed, because a gate that
    silently reports "clean" when it could not read the history is worse
    than one that fails.
    """


def offenders(rev_range: str) -> list[tuple[str, str, str, str, str]]:
    """Return `(sha, subject, author_name, author_email, which)` per offence."""
    fmt = "%H%x1f%s%x1f%an%x1f%ae%x1f%cn%x1f%ce%x1e"
    try:
        out = _git("log", f"--format={fmt}", rev_range)
    except subprocess.CalledProcessError as exc:
        raise UnreadableRange(
            f"could not walk {rev_range!r} (shallow checkout? "
            f"fetch the base ref first): {(exc.stderr or '').strip()[:200]}"
        ) from exc
    found: list[tuple[str, str, str, str, str]] = []
    for record in out.split("\x1e"):
        record = record.strip("\n")
        if not record:
            continue
        parts = record.split("\x1f")
        if len(parts) != 6:
            continue
        sha, subject, an, ae, cn, ce = parts
        if not email_is_allowed(ae):
            found.append((sha, subject, an, ae, "author email"))
        if not email_is_allowed(ce):
            found.append((sha, subject, cn, ce, "committer email"))
        if not name_is_allowed(an):
            found.append((sha, subject, an, ae, "author name"))
        if not name_is_allowed(cn):
            found.append((sha, subject, cn, ce, "committer name"))
    return found


_ACCEPTED_NAMES = ("rrs", "dependabot[bot]", "GitHub")
_REFUSED_NAMES = (
    "Jane Doe",
    "J. Random Hacker",
    "jane@example.org",
    "Jane <jane@example.org>",
    "",
)


def self_test() -> int:
    bad = 0
    for e in _ACCEPTED:
        if not NOREPLY_RE.match(e):
            print(f"SELF-TEST rejected an accepted form: {e!r}")
            bad += 1
    for e in _REFUSED:
        if NOREPLY_RE.match(e):
            print(f"SELF-TEST accepted a routable address: {e!r}")
            bad += 1
    for n in _ACCEPTED_NAMES:
        if not name_is_allowed(n):
            print(f"SELF-TEST rejected a published name: {n!r}")
            bad += 1
    for n in _REFUSED_NAMES:
        if name_is_allowed(n):
            print(f"SELF-TEST accepted an unpublished name: {n!r}")
            bad += 1
    print(
        f"self-test: {len(_ACCEPTED)} accepted addresses, {len(_REFUSED)} "
        f"refused, {len(_ACCEPTED_NAMES)} accepted names, "
        f"{len(_REFUSED_NAMES)} refused names, {bad} failures"
    )
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--range", dest="rev_range", metavar="REV_RANGE")
    ap.add_argument("--dry-run", action="store_true", help="report but exit 0")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.rev_range:
        ap.error("--range is required unless --self-test is given")

    try:
        found = offenders(args.rev_range)
    except UnreadableRange as exc:
        # Fail closed. Reporting "clean" for a range we could not read
        # would make the gate decorative exactly where it matters.
        print(f"::error::{exc}")
        return 0 if args.dry_run else 1
    for sha, subject, name, email, which in found:
        print(f"::error::{which} is not a published identity: {sha[:12]} {subject}")
        print(f"    {name} <{email}>")
    if found:
        print(
            f"\n{len(found)} offence(s). A commit's email is copied into every "
            "clone and fork, served by the public API, and pinned by the pull "
            "ref — it cannot be taken back by editing a file.\n"
            "Fix before pushing:\n"
            "  git config user.email '<id>+<login>@users.noreply.github.com'\n"
            "  git rebase -i --exec 'git commit --amend --no-edit --reset-author -S' "
            "<base>\n"
            "and turn on 'Keep my email address private' in GitHub settings so "
            "the next machine starts correct."
        )
    else:
        print(
            f"clean: every author and committer in {args.rev_range} is a "
            "published identity with a noreply address."
        )

    return 0 if args.dry_run else (1 if found else 0)


if __name__ == "__main__":
    sys.exit(main())
