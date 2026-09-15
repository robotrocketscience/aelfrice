#!/usr/bin/env python3
r"""Parse the closing keywords out of a pull-request body (#1541, #1549).

Usage:
    python3 scripts/merge_train_linked_issues.py --body-file /tmp/pr_body.txt
    gh pr view N --json body --jq '.body // ""' \
        | python3 scripts/merge_train_linked_issues.py

Prints one issue number per line, deduplicated, in ascending numeric order, on
stdout. Every keyword the parser found and deliberately did NOT act on prints
one `warning:` line on stderr naming the text and the reason. Exits non-zero
only when the input cannot be read; a body with no linked issue is not an
error, it prints nothing and exits 0.

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
it worked. The failure is invisible at the merge -- the train prints "no linked
issues parsed from PR body" and exits green.

Nothing bounds the input here any more, and that is deliberate. The cap was
the defect: `grep` streams, and the quantity worth bounding was never the body
but the loop over what came out of it, which is bounded by how many distinct
issues a human wrote in one description. Re-adding an input cap would
re-introduce exactly this bug at a larger offset. A surprising count warns on
stderr rather than truncating, so the train never again drops work quietly.

## The decision (#1549): this step emulates GitHub

The step stands in for a merge-commit close that the fast-forward model cannot
produce, so it reads the body the way GitHub reads it rather than as a narrow
trailer format of its own. Concretely, it now:

* Matches all nine keywords GitHub acts on -- `close`, `closes`, `closed`,
  `fix`, `fixes`, `fixed`, `resolve`, `resolves`, `resolved` -- case
  insensitively, each followed by whitespace and `#N`. The list is GitHub's
  published one, under "Linking a pull request to an issue":
  https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/linking-a-pull-request-to-an-issue
* Ignores a keyword a Markdown reader would not render as prose: inside a
  fenced code block (``` or ~~~, three characters or more, an info string
  allowed), inside an indented code block (four spaces or a tab), inside an
  inline code span, inside a block quote, or inside an HTML comment. GitHub
  does not publish these exclusions; they follow from the body being Markdown,
  and each is a wrong close this parser used to make -- a PR that documents
  the syntax, one that quotes an earlier comment containing `Fixes #N`, one
  that leaves a commented-out template line.
* Reports every rejection on stderr instead of passing over it, so a body
  whose keyword was refused never prints what a body with no keyword prints
  (#1549). stdout stays bare numbers, because the workflow parses it.

### An unclosed fence or comment runs to the end of the body

CommonMark closes an unterminated fenced block and an unterminated HTML
comment at the end of the containing document, and GitHub renders both that
way, so a lone ``` opener makes everything after it inert here too. The
alternative -- treating an unclosed fence as literal text -- would let a stray
backtick line resurrect the wrong close this change removes, and the two
errors are not symmetric: refusing a close leaves an open issue a human
notices, while making one closes an issue nobody asked to close.

### Divergences that REMAIN, deliberately

1. **Commit messages are not read.** GitHub also closes from the commit
   messages of a merged push; this reads the pull-request body only. Out of
   scope on #1549 -- it changes the surface far more than the keyword set does.
2. **Cross-repository links are not followed.** GitHub closes
   `Closes owner/repo#12` in that other repository. This train closes issues
   in its own repository only, so the link is refused -- but it is now refused
   out loud, on stderr, instead of falling through the regex unremarked.
3. **The target branch is not checked.** GitHub auto-closes only for a pull
   request targeting the default branch. The train only ever fast-forwards
   `main`, so the condition holds by construction, but nothing here asserts
   it; a train taught to push elsewhere would close issues GitHub would not.
4. **The block scan is line-based, not a CommonMark parse.** The known gaps: a
   lazy continuation line of a block quote (a line with no `>` that a reader
   still folds into the quote above it) counts as prose; a fence or an indent
   nested in a list item is measured against the left margin rather than
   against the list marker; a keyword split across a line boundary
   (`Resolves\n#7`) takes the context of the line its keyword starts on; an
   inline code span must open and close on one line; and the text after a
   `-->` on the line that closes a comment is ordinary text that cannot itself
   open a block.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# GitHub's nine closing keywords. `\b` before the keyword so `precloses #4`
# does not match; `\s+` after it because GitHub accepts any run of whitespace,
# a newline included.
KEYWORDS = (
    "close",
    "closes",
    "closed",
    "fix",
    "fixes",
    "fixed",
    "resolve",
    "resolves",
    "resolved",
)

# An owner or repository name as GitHub allows it: alphanumerics, `.`, `_` and
# `-`, neither leading nor trailing with a separator.
_NAME = r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"

# Every candidate the parser considers, whether or not it acts on it. The
# optional `repo` group is what makes a cross-repository link visible: without
# it the link simply fails to match and nothing is left to report.
LINK_RE = re.compile(
    r"\b(?P<keyword>" + "|".join(sorted(KEYWORDS, key=len, reverse=True)) + r")"
    r"\s+(?P<repo>" + _NAME + r"/" + _NAME + r")?#(?P<number>\d+)",
    re.IGNORECASE,
)

# Why a refused candidate was refused. Each string lands verbatim in the
# merge-train step log and reads as the end of "ignored ... because it is".
IN_FENCE = "inside a fenced code block"
IN_INDENT = "inside an indented code block"
IN_SPAN = "inside an inline code span"
IN_QUOTE = "inside a block quote"
IN_COMMENT = "inside an HTML comment"
CROSS_REPO = "a link to another repository, which this train does not close"

# Above this many distinct issues in one body, say so on stderr. Not a cap:
# every issue found is still printed. A body naming this many is more likely
# a template or a paste than a real set of links, and the merge-train log is
# the only place a human would see that.
NOISY_COUNT = 20

_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_QUOTE_RE = re.compile(r"^ {0,3}>")
_INDENT_RE = re.compile(r"^(?: {4}|\t)")
_TICKS_RE = re.compile(r"`+")


@dataclass(frozen=True)
class Rejection:
    """A closing keyword the parser found and chose not to act on."""

    text: str
    line: int
    reason: str

    def message(self) -> str:
        return f'warning: ignored "{self.text}" on line {self.line}: {self.reason}.'


def _scan_comments(
    text: str, start_at: int, line_start: int, spans: list[tuple[int, int, str]]
) -> bool:
    """Mark the HTML comments opening in `text[start_at:]`; report if one is open."""
    i = start_at
    while True:
        opened = text.find("<!--", i)
        if opened < 0:
            return False
        closed = text.find("-->", opened + 4)
        if closed < 0:
            spans.append((line_start + opened, line_start + len(text), IN_COMMENT))
            return True
        spans.append((line_start + opened, line_start + closed + 3, IN_COMMENT))
        i = closed + 3


def _scan_code_spans(
    text: str, line_start: int, spans: list[tuple[int, int, str]]
) -> None:
    """Mark the inline code spans on one line.

    A run of N backticks opens a span that only a run of exactly N backticks
    closes. An unmatched run is literal text, not an opener, so it marks
    nothing -- otherwise one stray backtick would silence the rest of the line.
    A span that would cross a line boundary is not recognised; see the module
    docstring's divergence 4.
    """
    pos = 0
    while True:
        opener = _TICKS_RE.search(text, pos)
        if opener is None:
            return
        width = opener.end() - opener.start()
        closer = None
        search_at = opener.end()
        while True:
            candidate = _TICKS_RE.search(text, search_at)
            if candidate is None:
                break
            if candidate.end() - candidate.start() == width:
                closer = candidate
                break
            search_at = candidate.end()
        if closer is None:
            pos = opener.end()
            continue
        spans.append((line_start + opener.start(), line_start + closer.end(), IN_SPAN))
        pos = closer.end()


def inert_spans(body: str) -> list[tuple[int, int, str]]:
    """Every `(start, end, reason)` range of the body a keyword must not fire in.

    One pass over the lines, carrying the three states a line inherits from the
    one above it: an open code fence, an open HTML comment, and an open
    indented code block. Offsets are into `body`, so a match is classified by
    where it starts.
    """
    spans: list[tuple[int, int, str]] = []
    fence: tuple[str, int] | None = None
    in_comment = False
    in_indent = False
    prev_blank = True
    pos = 0

    for raw in body.splitlines(keepends=True):
        start = pos
        pos += len(raw)
        text = raw.rstrip("\n").rstrip("\r")
        end = start + len(text)

        if in_comment:
            closed = text.find("-->")
            if closed < 0:
                spans.append((start, end, IN_COMMENT))
            else:
                spans.append((start, start + closed + 3, IN_COMMENT))
                in_comment = _scan_comments(text, closed + 3, start, spans)
            prev_blank = False
            continue

        if fence is not None:
            spans.append((start, end, IN_FENCE))
            closer = _FENCE_RE.match(text)
            if (
                closer is not None
                and closer.group(1)[0] == fence[0]
                and len(closer.group(1)) >= fence[1]
                and not closer.group(2).strip()
            ):
                fence = None
            prev_blank = False
            continue

        if not text.strip():
            # A blank line ends a paragraph but not an indented code block.
            prev_blank = True
            continue

        opener = _FENCE_RE.match(text)
        if opener is not None:
            fence = (opener.group(1)[0], len(opener.group(1)))
            spans.append((start, end, IN_FENCE))
            in_indent = False
            prev_blank = False
            continue

        indented = _INDENT_RE.match(text) is not None
        if indented and (in_indent or prev_blank):
            # An indented block cannot interrupt a paragraph, so it needs a
            # blank line above it -- or to be running already.
            in_indent = True
            spans.append((start, end, IN_INDENT))
            prev_blank = False
            continue
        in_indent = False

        if _QUOTE_RE.match(text) is not None:
            spans.append((start, end, IN_QUOTE))
            prev_blank = False
            continue

        in_comment = _scan_comments(text, 0, start, spans)
        _scan_code_spans(text, start, spans)
        prev_blank = False

    return spans


def _reason_at(offset: int, spans: list[tuple[int, int, str]]) -> str | None:
    for start, end, reason in spans:
        if start <= offset < end:
            return reason
    return None


def parse(body: str) -> tuple[list[int], list[Rejection]]:
    """Split the body's closing keywords into the acted-on and the refused.

    Returns the issue numbers to close -- sorted, deduplicated -- and one
    `Rejection` per candidate the parser declined, in the order they appear.
    """
    spans = inert_spans(body)
    found: set[int] = set()
    refused: list[Rejection] = []

    for m in LINK_RE.finditer(body):
        reason = _reason_at(m.start(), spans)
        if reason is None and m.group("repo") is not None:
            reason = CROSS_REPO
        if reason is None:
            found.add(int(m.group("number")))
            continue
        refused.append(
            Rejection(
                text=" ".join(m.group(0).split()),
                line=body.count("\n", 0, m.start()) + 1,
                reason=reason,
            )
        )

    return sorted(found), refused


def linked_issues(body: str) -> list[int]:
    """Every issue number this train will close, sorted and deduplicated.

    Cross-repository links (`Closes owner/repo#12`) are not included, because
    the train closes issues in its own repository only. They are not dropped in
    silence either -- `parse` returns them as rejections and `main` prints each
    one on stderr.
    """
    return parse(body)[0]


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

    found, refused = parse(body)

    # Diagnostics go to stderr only: the workflow reads stdout into a shell
    # variable and loops over the words in it.
    for rejection in refused:
        print(rejection.message(), file=sys.stderr)

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
