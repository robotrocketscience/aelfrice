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
  insensitively, each followed by an issue reference GitHub recognises,
  `#N` or `GH-N`, with either whitespace or a colon between the two. The
  keyword list and the colon are GitHub's, published under
  "Linking a pull request to an issue", which says: "The keywords can be
  followed by colons or in uppercase. For example: `Closes: #10`,
  `CLOSES #10`, or `CLOSES: #10`."
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

### What "followed by a colon" means here

GitHub publishes the colon but not its spacing, so this pins the three forms
its three examples leave open. The colon is a suffix of the keyword rather
than a token standing on its own:

* `Closes:#10` links. The colon is itself the delimiter, so nothing has to
  follow it -- which is why `Closes#10`, carrying no delimiter at all, still
  does not link.
* `Closes : #10` does not link. The colon has to touch the keyword.
* `Closes::#10` does not link. One colon, not a run of them.

Those two refusals are silent, as every non-match is, and that is the accepted
cost of matching what GitHub publishes rather than a superset of it: a form
GitHub would not close is a form this train must not close either.

### The colon admits a prose false positive, knowingly

A colon after one of the nine words is also how English introduces a list, and
nothing in the text distinguishes the two. Merged PR #1504's body contains the
prose "All ruled prerequisites are closed: #1329, ..." -- under this rule that
links #1329, which its author did not mean as a close directive.

It is accepted, not worked around. GitHub closes #1329 from that same body on
an ordinary merge, and this step exists to reproduce the close the
fast-forward push suppressed; a train that quietly disagreed with the platform
here would be a second surprise rather than a repair. This does not contradict
the asymmetry below. "Refuse rather than close" is the tie-break for the cases
GitHub leaves undefined -- an unterminated fence, a lazy quote continuation.
Where GitHub's behaviour is defined and published, matching it wins.

### Every documented reference form is acted on or refused out loud

The keyword page writes the syntax as a keyword plus an ISSUE-NUMBER, and the
page it links for what a reference to an issue may look like -- "Autolinked
references and URLs" --
https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/autolinked-references-and-urls#issues-and-pull-requests
-- gives five rows for four distinct forms; its
`Username/Repository#N` and `Organization_name/Repository#N` rows are one form
written twice. All four are enumerated here, because a form this parser
neither links nor refuses aloud is the silent no-op #1549 exists to kill:

* `#N` (`Closes #10`) links. The keyword page's own first row.
* `GH-N` (`Closes GH-10`) links, in any case. `GH-`, `gh-`, `Gh-` and `gH-`
  all render as the same issue reference in GitHub's renderer, checked rather
  than assumed with `gh api --method POST /markdown -f mode=gfm -f
  context=OWNER/REPO -f text='GH-10 and gh-10 and Gh-10 and gH-10 and #10'`,
  whose output links each casing, and the `#10` beside them, to the same
  issue. This repository's own advisory `pr-metadata.yml` job matches `GH-`
  under `grep -iE` too, and greets such a body with "Traceability OK".
* `OWNER/REPOSITORY#N` (`Fixes octo-org/octo-repo#100`) is refused out loud;
  see divergence 2. So is the same-repository spelling of it, which this
  parser cannot tell apart from any other.
* A full issue or pull-request URL
  (`Closes https://github.com/OWNER/REPOSITORY/issues/26`) is refused out
  loud, for the same reason and under its own wording. It names a repository,
  and this parser resolves numbers in one repository only.
* Nothing else is a reference. `Closes issue #10`, `Closes 10` and a bare
  `#10` are not forms GitHub acts on, so not matching them is fidelity.

GitHub publishes the reference syntax but not the processor that closes on it,
so acting on `GH-N` is a ruling under "emulate GitHub" taken on the reference
parser's measured behaviour, not a measurement of the close itself. It is the
one place this module widens rather than refuses on an edge GitHub leaves
undocumented, and the alternative loses either way: refusing here would have
the train contradict `pr-metadata.yml`, which has already told the author the
link is fine.

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
2. **Cross-repository and URL links are not followed.** GitHub closes
   `Closes owner/repo#12` in that other repository, and reads a full issue URL
   as a reference to that issue. This train closes issues in its own
   repository only, and cannot tell its own name from another's, so both are
   refused -- but refused out loud, on stderr, instead of falling through the
   regex unremarked.
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
   open a *block*, though it is scanned for inline spans and comments like any
   other live text.

### Code spans and comments are ranked by which opens first

Both are inline constructs, and each one's opening marker is ordinary text
inside the other: `` `a <!-- b` `` is a code span containing the characters
`<!--`, and `<!-- a `b` -->` is a comment containing two backticks. So one
left-to-right pass takes whichever opens first and consumes it whole. Scanning
one kind before the other, on any part of a line, lets the loser override the
winner and produces a wrong close in one direction and a wrong refusal in the
other.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# GitHub's nine closing keywords. `\b` before the keyword so `precloses #4`
# does not match; see `_SEPARATOR` for what may follow one.
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

# What may stand between a keyword and its `#N`: a run of whitespace, a
# newline included, or a colon bound directly to the keyword. The colon is
# GitHub's, published on the page cited in the module docstring; because it is
# itself a delimiter it does not need whitespace of its own. See that
# docstring for the three spacings GitHub's examples leave open.
_SEPARATOR = r"(?::\s*|\s+)"

# A full issue or pull-request URL, which GitHub also accepts as a reference
# to an issue. Matched so that it can be refused by name rather than missed.
_ISSUE_URL = (
    r"https?://github\.com/" + _NAME + r"/" + _NAME + r"/(?:issues|pull)/"
)

# Every candidate the parser considers, whether or not it acts on it. The
# optional `repo` group and the `url` group are what make a link this train
# will not follow visible: without them the link simply fails to match and
# nothing is left to report.
#
# `GH-` carries no group because it needs none -- it is this repository's own
# issue either way, so it lands in the same place `#N` does. It matches every
# casing, which is GitHub's behaviour; see the docstring for the probe.
#
# The longest-first sort of the alternation is legibility, not correctness.
# `re` backtracks, so `close` matching first in `closes #7` fails at the
# separator and the engine retries `closes`; sorting the nine alphabetically,
# or shortest-first, matches the same text at the same offsets. Nothing may
# depend on the order.
LINK_RE = re.compile(
    r"\b(?P<keyword>" + "|".join(sorted(KEYWORDS, key=len, reverse=True)) + r")"
    + _SEPARATOR
    + r"(?:(?P<url>" + _ISSUE_URL + r")"
    + r"|(?P<repo>" + _NAME + r"/" + _NAME + r")?#"
    + r"|GH-)"
    + r"(?P<number>\d+)",
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
ISSUE_URL = "a full issue URL, which this train does not follow"

# Above this many distinct issues in one body, say so on stderr. Not a cap:
# every issue found is still printed. A body naming this many is more likely
# a template or a paste than a real set of links, and the merge-train log is
# the only place a human would see that.
NOISY_COUNT = 20

_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_QUOTE_RE = re.compile(r"^ {0,3}>")
_INDENT_RE = re.compile(r"^(?: {4}|\t)")
_TICKS_RE = re.compile(r"`+")

# CommonMark's start condition for an HTML block opened by a comment: `<!--`
# at the head of the line, under four spaces of indentation. Such a line is a
# block of its own, so unlike a paragraph it does not lazily swallow an
# indented line written under it.
_HTML_BLOCK_RE = re.compile(r"^ {0,3}<!--")


@dataclass(frozen=True)
class Rejection:
    """A closing keyword the parser found and chose not to act on."""

    text: str
    line: int
    reason: str

    def message(self) -> str:
        return f'warning: ignored "{self.text}" on line {self.line}: {self.reason}.'


def _first_code_span(text: str, start_at: int) -> tuple[int, int] | None:
    """The leftmost inline code span opening at or after `start_at`, if any.

    A run of N backticks opens a span that only a run of exactly N backticks
    closes. An unmatched run is literal text, not an opener, so the search
    steps over it and keeps looking -- otherwise one stray backtick would
    silence the rest of the line. A span that would cross a line boundary is
    not recognised; see the module docstring's divergence 4.
    """
    pos = start_at
    while True:
        opener = _TICKS_RE.search(text, pos)
        if opener is None:
            return None
        width = opener.end() - opener.start()
        search_at = opener.end()
        while True:
            candidate = _TICKS_RE.search(text, search_at)
            if candidate is None:
                break
            if candidate.end() - candidate.start() == width:
                return opener.start(), candidate.end()
            search_at = candidate.end()
        pos = opener.end()


def _scan_inline(
    text: str, start_at: int, line_start: int, spans: list[tuple[int, int, str]]
) -> bool:
    """Mark the code spans and comments in `text[start_at:]`; is one left open?

    Code spans and HTML comments are both inline constructs, so whichever
    opens first wins and the other's marker is ordinary characters inside it:
    a `<!--` written between backticks is code, and a backtick written inside
    a comment is comment. Scanning one kind first and the other second instead
    lets the loser override the winner, in whichever direction the order
    happens to run.

    Returns True when a comment is still open at the end of the line, which
    makes every following line inert until a `-->` closes it.
    """
    pos = start_at
    while True:
        comment_at = text.find("<!--", pos)
        span = _first_code_span(text, pos)
        if span is not None and (comment_at < 0 or span[0] < comment_at):
            spans.append((line_start + span[0], line_start + span[1], IN_SPAN))
            pos = span[1]
            continue
        if comment_at < 0:
            return False
        closed = text.find("-->", comment_at + 4)
        if closed < 0:
            spans.append((line_start + comment_at, line_start + len(text), IN_COMMENT))
            return True
        spans.append((line_start + comment_at, line_start + closed + 3, IN_COMMENT))
        pos = closed + 3


def inert_spans(body: str) -> list[tuple[int, int, str]]:
    """Every `(start, end, reason)` range of the body a keyword must not fire in.

    One pass over the lines, carrying the four states a line inherits from the
    one above it: an open code fence, an open HTML comment, an open indented
    code block, and whether the line above was a paragraph line. Offsets are
    into `body`, so a match is classified by where it starts.

    `prev_paragraph` is the one that is easy to get wrong, and it is not the
    same as "the line above was not blank". An indented code block may not
    interrupt a paragraph, but every other predecessor lets one open: a fence,
    an HTML block, a preceding indented line, the start of the body and a
    blank line all do. Only a paragraph line blocks it, and a block quote
    counts as one because GitHub folds the indented line under it into the
    quote as a lazy continuation. See the module docstring's divergence 4.
    """
    spans: list[tuple[int, int, str]] = []
    fence: tuple[str, int] | None = None
    in_comment = False
    in_indent = False
    prev_paragraph = False
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
                # The rest of the line is live text, so it takes the same
                # inline path a line that never was in a comment takes.
                in_comment = _scan_inline(text, closed + 3, start, spans)
            # The whole line belongs to the HTML block, the text after `-->`
            # included, so it is not a paragraph line.
            prev_paragraph = False
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
            prev_paragraph = False
            continue

        if not text.strip():
            # A blank line ends a paragraph but not an indented code block.
            prev_paragraph = False
            continue

        opener = _FENCE_RE.match(text)
        if opener is not None:
            fence = (opener.group(1)[0], len(opener.group(1)))
            spans.append((start, end, IN_FENCE))
            in_indent = False
            prev_paragraph = False
            continue

        indented = _INDENT_RE.match(text) is not None
        if indented and (in_indent or not prev_paragraph):
            # An indented block cannot interrupt a paragraph, so it needs a
            # non-paragraph line above it -- or to be running already.
            in_indent = True
            spans.append((start, end, IN_INDENT))
            prev_paragraph = False
            continue
        in_indent = False

        if _QUOTE_RE.match(text) is not None:
            spans.append((start, end, IN_QUOTE))
            # A lazy continuation folds the next line into this quote's
            # paragraph, so an indent below it is prose, not code.
            prev_paragraph = True
            continue

        in_comment = _scan_inline(text, 0, start, spans)
        # A line that opens with `<!--` is an HTML block rather than a
        # paragraph, whatever follows the `-->` on it.
        prev_paragraph = _HTML_BLOCK_RE.match(text) is None

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
        if reason is None and m.group("url") is not None:
            reason = ISSUE_URL
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

    A link that names a repository -- `Closes owner/repo#12`, or a full issue
    URL -- is not included, because the train closes issues in its own
    repository only. It is not dropped in silence either: `parse` returns it as
    a rejection and `main` prints each one on stderr.
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
