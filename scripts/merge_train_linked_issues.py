#!/usr/bin/env python3
"""Parse the `Closes #N` trailers out of a pull-request body (#1541).

Usage:
    python3 scripts/merge_train_linked_issues.py --body-file /tmp/pr_body.txt
    gh pr view N --json body --jq '.body // ""' \
        | python3 scripts/merge_train_linked_issues.py

Prints one issue number per line, deduplicated, in ascending numeric order.
Exits non-zero only when the input cannot be read; a body with no linked
issue is not an error, it prints nothing and exits 0.

## Why this is a script and not a shell pipeline

`.github/workflows/merge-train.yml` step 7/7 closes the issues a merged PR
links, because the fast-forward push this repo's signed-commit branch
protection requires does not trigger GitHub's own auto-close. It used to do
that inline:

    pr_body=$(gh pr view ... --jq '.body // ""' | head -c 8192)
    linked_issues=$(printf '%s' "${pr_body}" | grep -ioE '...')

`head -c 8192` ran **before** the grep, so a keyword written past the 8,192nd
byte was cut away and the issue silently stayed open while the merge reported
success. Measured over five consecutive merges: the keyword sat at byte 463,
9,148, 8,992, 0 and 0; both bodies over the cut lost their close, both under
it worked. The failure is invisible at the merge — the train prints "no linked
issues parsed from PR body" and exits green.

Nothing bounds the input here any more, and that is deliberate. The cap was
the defect: `grep` streams, and the quantity worth bounding was never the body
but the loop over what came out of it, which is bounded by how many distinct
issues a human wrote in one description. Re-adding an input cap would
re-introduce exactly this bug at a larger offset. A surprising count warns on
stderr rather than truncating, so the train never again drops work quietly.

## Two fidelity gaps this deliberately does NOT close

Both change which issues get closed, so they are behaviour changes rather than
bug fixes, and they are left for a decision rather than taken here:

1. **The keyword set is narrower than GitHub's.** GitHub closes on `close`,
   `closes`, `closed`, `fix`, `fixes`, `fixed`, `resolve`, `resolves` and
   `resolved`. This matches only `closes`, `fixes` and `resolves`, as the
   shell it replaces did, so `Fixed #N` in a body is not closed here though
   GitHub would have closed it on a merge commit.
2. **Fenced code blocks are not excluded.** GitHub does not act on a keyword
   inside a code fence; this does. A body demonstrating the syntax in an
   example would close the issue it names.

Direction matters when judging these: (1) misses a close, (2) makes a wrong
one. They are listed together because both are the same question — how
faithfully should the fast-forward path emulate the merge path — and that is
the question, not the parsing.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# `\b` before the keyword so `precloses #4` does not match; `\s+` after it
# because GitHub accepts any run of whitespace, a newline included. The
# keyword set is deliberately the one the shell used -- see the module
# docstring, gap 1.
LINK_RE = re.compile(r"\b(?:closes|fixes|resolves)\s+#(\d+)", re.IGNORECASE)

# Above this many distinct issues in one body, say so on stderr. Not a cap:
# every issue found is still printed. A body naming this many is more likely
# a template or a paste than a real set of links, and the merge-train log is
# the only place a human would see that.
NOISY_COUNT = 20


def linked_issues(body: str) -> list[int]:
    """Every issue number linked by a closing keyword, sorted and deduped.

    Cross-repository links (`Closes owner/repo#12`) do not match, because the
    keyword is not followed by whitespace-then-`#`. That is correct for this
    caller: the train closes issues in its own repository only.
    """
    return sorted({int(n) for n in LINK_RE.findall(body)})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--body-file",
        type=Path,
        help="read the PR body from this file instead of stdin",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "report what would be closed, one `would close #N` line per issue, "
            "instead of the bare numbers the workflow consumes"
        ),
    )
    args = ap.parse_args(argv)

    if args.body_file is not None:
        try:
            body = args.body_file.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            print(f"cannot read {args.body_file}: {exc}", file=sys.stderr)
            return 1
    else:
        body = sys.stdin.read()

    found = linked_issues(body)

    if len(found) > NOISY_COUNT:
        print(
            f"note: {len(found)} linked issues parsed from a "
            f"{len(body)}-character body; all are listed, none dropped.",
            file=sys.stderr,
        )

    for n in found:
        print(f"would close #{n}" if args.dry_run else n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
