#!/usr/bin/env python3
r"""Read the issues a pull-request body closes, by asking GitHub (#1541, #1549).

Usage:
    python3 scripts/merge_train_linked_issues.py \
        --repo OWNER/NAME --body-file /tmp/pr_body.txt
    gh pr view N --json body --jq '.body // ""' \
        | python3 scripts/merge_train_linked_issues.py --repo OWNER/NAME

Prints one issue number per line, deduplicated, in ascending numeric order, on
stdout. Every closing keyword the tool found and deliberately did NOT act on
prints one `warning:` line on stderr naming the text and the reason. Exit 0
means the answer on stdout is complete -- a body with no linked issue is not an
error, it prints nothing. Exit 1 means the body could not be read; exit 2 means
the answer could not be determined at all, and stdout is then empty on purpose.

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
the defect, and the quantity worth bounding was never the body but the loop
over what came out of it, which is bounded by how many distinct issues a human
wrote in one description. A 264,036-character body renders and links all three
of its issues, checked rather than assumed:

    python3 - <<'PY'
    import json
    body = "\n".join(["prose line, and more of it"] * 4000
                     + ["Closes #11.", "Fixes #22.", "Resolves #33."])
    json.dump({"mode": "gfm", "context": "robotrocketscience/aelfrice",
               "text": body}, open("big.json", "w"))
    PY
    gh api --method POST /markdown --input big.json | grep -o 'issues/[0-9]*'

## The decision (#1549): ask GitHub rather than emulate it

The step stands in for a merge-commit close that the fast-forward model cannot
produce, so it must read the body the way GitHub reads it. An earlier revision
tried to *emulate* that reading: it classified each line of the body as prose,
fence, indent, quote or comment with a line-state scanner, and only fired a
keyword found in prose. Five review rounds each fixed one reading and found
another block type -- the last of them a wrong close on this repository's own
pull-request template, where `## Linked issues` followed by an indented
`Fixed #7` linked #7 although GitHub renders it inside `<pre><code>` and closes
nothing. The surface being emulated was CommonMark's whole block grammar.

So this module does not parse Markdown. It renders the body through GitHub's
own renderer and reads which references GitHub itself anchored:

    gh api --method POST /markdown --input -
    {"mode": "gfm", "context": "OWNER/NAME", "text": "<the body>"}

Text inside a fenced code block, an indented code block, an inline code span or
an HTML comment never produces an issue-link anchor, so every
block-classification question disappears at once rather than one block type per
review round. The `context` is what makes `#N` resolve, and it is also this
module's only source of repository identity; see "Which repository" below.

### Which reference closes, and which merely appears

GitHub anchors every reference, closing or not, so the rendered document alone
does not say which ones close. This decides that by adjacency in the rendered
text, which is how GitHub's own close processor reads a body: walk the rendered
HTML, and for each issue-link anchor look at the text immediately before it in
the same block. If that text ends in one of the nine keywords -- `close`,
`closes`, `closed`, `fix`, `fixes`, `fixed`, `resolve`, `resolves`, `resolved`,
in any case -- optionally followed by a colon and any whitespace, the anchor is
a close directive. Otherwise it is a mention and is ignored.

Adjacency is per *block*: a paragraph, a heading, a table cell, a list item.
Text before a block boundary cannot arm an anchor after it, and neither can
text inside an inline code span, because a keyword written as code is not a
keyword. An anchor ends the run too, so `Closes #1 #2` links #1 alone, which is
GitHub's rule of one keyword per issue.

The keyword list and the colon are GitHub's, published under "Linking a pull
request to an issue", which says: "The keywords can be followed by colons or in
uppercase. For example: `Closes: #10`, `CLOSES #10`, or `CLOSES: #10`."
https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/linking-a-pull-request-to-an-issue

### What "followed by a colon" means here

GitHub publishes the colon but not its spacing, so this pins the three forms
its three examples leave open. The colon is a suffix of the keyword rather
than a token standing on its own:

* `Closes:#10` links. The colon is itself the delimiter.
* `Closes : #10` does not link. The colon has to touch the keyword.
* `Closes::#10` does not link. One colon, not a run of them.

All three render an anchor -- GitHub's renderer does not decide this, its close
processor does, and that is not published -- so the refusals are this module's
ruling and not a measurement. They are refusals rather than links because a
wrong close is worse in kind than a missed one: a missed close leaves an open
issue a human notices, a wrong one closes an issue nobody asked to close.

### The colon admits a prose false positive, knowingly

A colon after one of the nine words is also how English introduces a list, and
nothing in the text distinguishes the two. Merged PR #1504's body contains the
prose "All ruled prerequisites are closed: #1329, ..." -- under this rule that
links #1329, which its author did not mean as a close directive.

It is accepted, not worked around. GitHub closes #1329 from that same body on
an ordinary merge, and this step exists to reproduce the close the
fast-forward push suppressed; a train that quietly disagreed with the platform
here would be a second surprise rather than a repair.

### Which repository

`--repo OWNER/NAME` (or `GITHUB_REPOSITORY`) is required, and missing it is an
error rather than a default. It is sent as the render `context`, and every
anchor GitHub returns carries a `data-url` naming the repository it resolved
to, so this module can compare the two. That is new: the emulating revision
held no repository identity and therefore had to refuse `owner/repo#N` and full
issue URLs indiscriminately, its own spellings included. Now:

* An anchor resolving to this repository is closed, however it was spelled --
  `#N`, `GH-N`, `OWNER/NAME#N`, or a full
  `https://github.com/OWNER/NAME/issues/N`. GitHub renders all four identically
  and closes all four, so this does too.
* An anchor resolving anywhere else is refused out loud; see divergence 2.

### Failure policy: loud, and closing nothing

The renderer is a network call, so it can be unreachable, answer non-200, or
answer something this module cannot read. All three raise `RendererUnavailable`
and `main` returns 2 with an `error:` line on stderr and **nothing on stdout**.
Two alternatives were rejected:

* Falling back to a local parse is the emulation this change deletes, and it
  would make its wrong closes precisely when the renderer that would have
  caught them is unavailable -- the worst possible moment.
* Treating an unreachable renderer as "no links" is the silent no-op #1549
  exists to kill: the step would print what a body with no keyword prints.

Closing nothing is recoverable by a human reading the step log; a wrong close
is not. There is no retry: this step runs after the merge has landed, so a
transient failure costs one hand-run of `gh issue close`, and a retry loop
would add a timing dependency to a step whose whole job is to be legible. The
workflow turns exit 2 into a `::error::` annotation rather than swallowing it.

An empty or whitespace-only body is answered without calling the renderer at
all: it has no anchors by construction, and that keeps the commonest no-op
cheap.

### Nothing is silently unmatched

AC5 forbids exactly one outcome -- a keyword the tool neither acts on nor
reports. Two of the three refusal reasons come from the rendered document, but
the third cannot: when GitHub declines to render a reference at all, there is
no anchor to attach a reason to. `Fixes octo-org/octo-repo#100` is the standard
case, because GitHub only autolinks `owner/repo#N` for a repository it can
resolve; so is every keyword written inside a code fence.

So the body is also scanned at source level by `SOURCE_RE`, purely for
diagnostics. It never decides a close -- it cannot, being a flat regex with no
idea of blocks -- and any source candidate whose issue number was neither
closed nor already refused prints one `NOT_LINKED` warning. That keeps the
property without re-introducing a single line of block classification.

The source scan is not a superset of what closes, and does not claim to be:
`Closes *#7*` renders an anchor armed by an adjacent `Closes` while the source
reads `Closes *#`, which `SOURCE_RE` does not match. A source candidate missed
there costs a warning, never a close.

### Every documented reference form is acted on or refused out loud

The keyword page writes the syntax as a keyword plus an ISSUE-NUMBER, and the
page it links for what a reference to an issue may look like -- "Autolinked
references and URLs" --
https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/autolinked-references-and-urls#issues-and-pull-requests
-- gives five rows for four distinct forms; its `Username/Repository#N` and
`Organization_name/Repository#N` rows are one form written twice:

* `#N` (`Closes #10`) links.
* `GH-N` (`Closes GH-10`) links, in any case. GitHub's renderer anchors `GH-`,
  `gh-`, `Gh-` and `gH-` to the same issue as `#10`, so nothing here has to
  know the form at all -- it arrives as an anchor like any other.
* `OWNER/REPOSITORY#N` links when it names this repository and is refused out
  loud when it names another. When GitHub cannot resolve the repository it
  renders no anchor, and the source scan reports it as `NOT_LINKED`.
* A full issue or pull-request URL behaves identically, for the same reason:
  GitHub renders it as an anchor to the issue it names.
* Nothing else is a reference. `Closes issue #10`, `Closes 10` and a bare `#10`
  are not forms GitHub acts on, so not matching them is fidelity.

### Divergences that REMAIN, deliberately

1. **Commit messages are not read.** GitHub also closes from the commit
   messages of a merged push; this reads the pull-request body only. Out of
   scope on #1549 -- it changes the surface far more than the keyword set does.
2. **Cross-repository links are not followed.** GitHub closes
   `Closes owner/repo#12` in that other repository. This train closes issues in
   its own repository only, so an anchor whose `data-url` names another
   repository is refused -- out loud, on stderr, rather than passed over.
3. **The target branch is not checked.** GitHub auto-closes only for a pull
   request targeting the default branch. The train only ever fast-forwards
   `main`, so the condition holds by construction, but nothing here asserts
   it; a train taught to push elsewhere would close issues GitHub would not.
4. **A block quote is refused, and GitHub probably closes it.** An anchor
   inside `<blockquote>` renders exactly like one in a paragraph, so adjacency
   alone would close it. It is refused instead, out loud, because the body that
   quotes an earlier review comment saying `Fixes #N` is a real wrong close and
   this is the one place the asymmetry above outranks fidelity. GitHub does not
   publish whether its close processor skips quoted text, so this is a ruling,
   not a measurement -- and it is an exact test on the rendered tree, not a
   grammar to emulate: dropping the `in_quote` branch from `close_directives`
   reverses it in one line.
5. **The colon spacings above** are rulings for the same reason: GitHub renders
   an anchor for all three and publishes no processor.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

# GitHub's nine closing keywords. Both patterns below are built from this
# tuple, so it is the single place the set is written.
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

# The longest-first sort of both alternations is legibility, not correctness.
# `re` backtracks, so `close` matching first in `closes #7` fails at the
# separator and the engine retries `closes`; sorting the nine alphabetically,
# or shortest-first, matches the same text at the same offsets. Nothing may
# depend on the order.
_ALTERNATION = "|".join(sorted(KEYWORDS, key=len, reverse=True))

# What decides a close: the text immediately before an anchor, in the same
# block, ending in a keyword plus an optional colon bound to it and any
# whitespace. `(?:^|\W)` is the word boundary at the front, so `precloses #4`
# does not fire; `:?\s*$` is why `Closes : #4` and `Closes::#4` do not.
ADJACENT_RE = re.compile(
    r"(?:^|\W)(?P<keyword>" + _ALTERNATION + r"):?\s*$",
    re.IGNORECASE,
)

# An owner or repository name as GitHub allows it: alphanumerics, `.`, `_` and
# `-`, neither leading nor trailing with a separator.
_NAME = r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"

# The source-level scan. DIAGNOSTIC ONLY -- it never decides a close, it only
# names a candidate GitHub declined to render so that nothing is silent. See
# "Nothing is silently unmatched" in the module docstring.
SOURCE_RE = re.compile(
    r"\b(?P<keyword>" + _ALTERNATION + r")(?::\s*|\s+)"
    r"(?:https?://github\.com/" + _NAME + r"/" + _NAME + r"/(?:issues|pull)/"
    r"|(?:" + _NAME + r"/" + _NAME + r")?#"
    r"|GH-)"
    r"(?P<number>\d+)",
    re.IGNORECASE,
)

# Why a refused candidate was refused. Each string lands verbatim in the
# merge-train step log and reads as the end of "ignored ... because it is".
CROSS_REPO = "a link to another repository, which this train does not close"
IN_QUOTE = "inside a block quote"
NOT_LINKED = "not linked by GitHub's renderer, so a merge commit would not close it"

# Above this many distinct issues in one body, say so on stderr. Not a cap:
# every issue found is still printed. A body naming this many is more likely
# a template or a paste than a real set of links, and the merge-train log is
# the only place a human would see that.
NOISY_COUNT = 20

# The renderer is one HTTPS round trip on a step that already holds the
# merge-train's only concurrency slot, so it is bounded. A timeout raises
# `RendererUnavailable` like any other failure and closes nothing.
RENDER_TIMEOUT_SECONDS = 30

# The anchor GitHub emits for an issue reference, and the shape of the URL it
# hangs on it. `data-url` is always the `/issues/N` spelling even when `href`
# points at `/pull/N`, so it is read first.
_ANCHOR_CLASS = "issue-link"
_ISSUE_HREF_RE = re.compile(
    r"^https?://github\.com/(?P<owner>" + _NAME + r")/(?P<repo>" + _NAME + r")"
    r"/(?:issues|pull)/(?P<number>\d+)(?:[/?#].*)?$"
)

_QUOTE_TAG = "blockquote"

# Tags that do not break a run of text. Everything else does, which is the
# conservative direction: an unknown wrapper GitHub adds later separates text
# from an anchor rather than silently arming it. `code` is deliberately NOT
# here -- a keyword written as code is not a keyword.
_INLINE_TAGS = frozenset(
    {
        "a", "abbr", "b", "br", "cite", "del", "em", "font", "g-emoji", "i",
        "img", "ins", "kbd", "mark", "q", "s", "small", "span", "strong",
        "sub", "sup", "time", "tt", "u",
    }
)

# How much of the preceding text `ADJACENT_RE` can need. The longest keyword
# is nine characters; the rest is slack for the colon, whitespace and the
# non-word character in front.
_TAIL = 64


class RendererUnavailable(RuntimeError):
    """GitHub's renderer could not be reached, or answered unusably."""


@dataclass(frozen=True)
class Rejection:
    """A closing keyword the tool found and chose not to act on."""

    text: str
    reason: str
    line: int | None = None

    def message(self) -> str:
        where = f" on line {self.line}" if self.line is not None else ""
        return f'warning: ignored "{self.text}"{where}: {self.reason}.'


def render_markdown(body: str, repo: str, *, run: object = None) -> str:
    """The body as GitHub renders it, in GFM mode, in `repo`'s context.

    This is the seam. `run` exists so a test can pin the call without a
    network, and the tests also drive the real `subprocess.run` with a `gh`
    of their own on `PATH`, because a suite that only ever sees an injected
    callable proves nothing about the command that ships.

    The default is resolved here rather than written as `run=subprocess.run`
    in the signature: a default argument is evaluated once, at import, so the
    signature form would bind the real runner past any later replacement of
    it and quietly make an injected one a no-op.

    The payload goes on stdin rather than in argv: a pull-request body has no
    length limit worth relying on, and a quarter-megabyte one renders fine.
    """
    run = subprocess.run if run is None else run
    payload = json.dumps({"mode": "gfm", "context": repo, "text": body})
    argv = ["gh", "api", "--method", "POST", "/markdown", "--input", "-"]
    try:
        proc = run(  # type: ignore[operator]
            argv,
            input=payload,
            capture_output=True,
            text=True,
            check=False,
            timeout=RENDER_TIMEOUT_SECONDS,
        )
    except OSError as exc:
        raise RendererUnavailable(f"cannot run {argv[0]}: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RendererUnavailable(
            f"{argv[0]} did not answer within {RENDER_TIMEOUT_SECONDS}s"
        ) from exc

    if proc.returncode != 0:
        detail = " ".join((proc.stderr or "").split())[:400] or "no output"
        raise RendererUnavailable(
            f"{argv[0]} exited {proc.returncode} rendering the body: {detail}"
        )
    if not proc.stdout.strip():
        raise RendererUnavailable(
            f"{argv[0]} rendered a {len(body)}-character body as nothing"
        )
    return proc.stdout


class _AnchorScanner(HTMLParser):
    """Collects `(keyword, anchor_text, url, in_quote)` for every issue link.

    One left-to-right pass. `_tail` holds the text since the last block
    boundary, trimmed to the few characters `ADJACENT_RE` can need, so a
    paragraph of any length costs the same.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.anchors: list[tuple[str | None, str, str, bool]] = []
        self._tail = ""
        self._quote_depth = 0
        self._open: tuple[str, str, bool] | None = None
        self._anchor_text = ""

    # -- text ------------------------------------------------------------
    def handle_data(self, data: str) -> None:
        if self._open is not None:
            self._anchor_text += data
        else:
            self._tail = (self._tail + data)[-_TAIL:]

    # -- tags ------------------------------------------------------------
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == _QUOTE_TAG:
            self._quote_depth += 1
        if tag not in _INLINE_TAGS:
            self._tail = ""
            return
        if tag != "a":
            # Every other inline tag, `<br>` included, leaves the run alone.
            # A `<br>` adds no whitespace of its own because none is needed:
            # `ADJACENT_RE` allows zero, so a keyword that ends flush against
            # an anchor still arms it, which is the same rule that makes
            # `Closes:#10` a link.
            return
        attributes = {k: (v or "") for k, v in attrs}
        if _ANCHOR_CLASS not in attributes.get("class", "").split():
            # An ordinary link. It ends the run either way: a keyword before
            # `[see](url)` must not arm whatever anchor comes after it.
            self._tail = ""
            return
        url = attributes.get("data-url") or attributes.get("href", "")
        keyword = _keyword_before(self._tail)
        self._open = (keyword or "", url, self._quote_depth > 0)
        self._anchor_text = ""

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        self.handle_starttag(tag, attrs)
        if tag != "br":
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag == _QUOTE_TAG and self._quote_depth:
            self._quote_depth -= 1
        if tag == "a" and self._open is not None:
            keyword, url, in_quote = self._open
            self.anchors.append(
                (keyword or None, self._anchor_text, url, in_quote)
            )
            self._open = None
        # An anchor ends the run whether or not it was an issue link, so
        # `Closes #1 #2` arms only #1.
        if tag not in _INLINE_TAGS or tag == "a":
            self._tail = ""


def _keyword_before(tail: str) -> str | None:
    m = ADJACENT_RE.search(tail)
    return m.group("keyword") if m is not None else None


def close_directives(html: str, repo: str) -> tuple[list[int], list[Rejection]]:
    """Split the rendered document's close directives into acted-on and refused.

    A directive is an issue-link anchor with one of the nine keywords
    immediately before it in the same block. Everything else in the document,
    anchor or not, is a mention.
    """
    scanner = _AnchorScanner()
    try:
        scanner.feed(html)
        scanner.close()
    except Exception as exc:  # pragma: no cover - html.parser is lenient
        raise RendererUnavailable(f"cannot read the rendered body: {exc}") from exc

    found: set[int] = set()
    refused: list[Rejection] = []
    for keyword, text, url, in_quote in scanner.anchors:
        if keyword is None:
            continue
        target = _ISSUE_HREF_RE.match(url)
        if target is None:
            raise RendererUnavailable(
                f"an issue-link anchor carried an unreadable URL: {url!r}"
            )
        quoted = f"{keyword} {text}".strip()
        if in_quote:
            refused.append(Rejection(text=quoted, reason=IN_QUOTE))
            continue
        if f"{target['owner']}/{target['repo']}".lower() != repo.lower():
            refused.append(Rejection(text=quoted, reason=CROSS_REPO))
            continue
        found.add(int(target["number"]))
    return sorted(found), refused


def unrendered_candidates(
    body: str, accounted: set[int]
) -> list[Rejection]:
    """Source-level keywords whose issue number nothing in the render explains.

    Diagnostics only; see "Nothing is silently unmatched" in the module
    docstring. A number already closed or already refused is accounted for,
    so a body that both documents `Closes #7` in a fence and closes #7 in
    prose warns about neither.
    """
    seen: set[int] = set()
    out: list[Rejection] = []
    for m in SOURCE_RE.finditer(body):
        number = int(m.group("number"))
        if number in accounted or number in seen:
            continue
        seen.add(number)
        out.append(
            Rejection(
                text=" ".join(m.group(0).split()),
                reason=NOT_LINKED,
                line=body.count("\n", 0, m.start()) + 1,
            )
        )
    return out


def parse(
    body: str, repo: str, *, render: object = None
) -> tuple[list[int], list[Rejection]]:
    """The issue numbers to close, and every candidate refused, for one body.

    Raises `RendererUnavailable` when the answer cannot be determined; see
    the module docstring's failure policy. `render` is the injection seam, and
    its default is resolved here rather than in the signature for the reason
    `render_markdown` gives about its own.
    """
    render = render_markdown if render is None else render
    if not body.strip():
        return [], []
    html = render(body, repo)  # type: ignore[operator]
    found, refused = close_directives(html, repo)
    accounted = set(found)
    for rejection in refused:
        number = re.search(r"(\d+)\s*$", rejection.text)
        if number is not None:
            accounted.add(int(number.group(1)))
    refused = refused + unrendered_candidates(body, accounted)
    return found, refused


def linked_issues(body: str, repo: str, *, render: object = None) -> list[int]:
    """Every issue number this train will close, sorted and deduplicated."""
    return parse(body, repo, render=render)[0]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY", ""),
        help=(
            "OWNER/NAME this body belongs to; defaults to $GITHUB_REPOSITORY. "
            "Sent to GitHub as the render context and compared against every "
            "anchor, so it is required rather than guessed"
        ),
    )
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

    if not args.repo:
        print(
            "error: no repository: pass --repo OWNER/NAME or set "
            "GITHUB_REPOSITORY. Nothing was closed.",
            file=sys.stderr,
        )
        return 2

    if args.body_file is not None:
        try:
            body = args.body_file.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            print(f"cannot read {args.body_file}: {exc}", file=sys.stderr)
            return 1
    else:
        body = sys.stdin.read()

    try:
        found, refused = parse(body, args.repo)
    except RendererUnavailable as exc:
        # Loud, and stdout stays empty: the workflow must close nothing
        # rather than close the wrong thing or report a silent absence.
        print(f"error: {exc}", file=sys.stderr)
        print(
            "error: could not determine the linked issues; nothing was closed.",
            file=sys.stderr,
        )
        return 2

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
