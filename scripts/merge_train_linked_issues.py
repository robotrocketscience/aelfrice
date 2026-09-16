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

This module bounds nothing itself, and that is deliberate: the cap was the
defect, and the quantity worth bounding was never the body but the loop over
what came out of it, which is bounded by how many distinct issues a human wrote
in one description. A 264,096-character body renders and links all three of its
issues, checked rather than assumed:

    python3 - <<'PY'
    import json
    body = "\n".join(["prose line, and more of it"] * 9780
                     + ["Closes #11.", "Fixes #22.", "Resolves #33."])
    print(len(body))
    json.dump({"mode": "gfm", "context": "robotrocketscience/aelfrice",
               "text": body}, open("big.json", "w"))
    PY
    gh api --method POST /markdown --input big.json \
        | grep -o 'issues/[0-9]*' | sort -u

prints `264096`, then `issues/11`, `issues/22` and `issues/33`.

### The renderer has a cap of its own, and it must be said out loud

`POST /markdown` refuses a request above 400 KB. That is a reintroduced input
bound on a step whose whole defect was an input bound, so it is stated rather
than discovered:

    python3 -c 'import json; json.dump({"mode": "gfm", "context":
        "robotrocketscience/aelfrice", "text": "x" * 420000},
        open("too_big.json", "w"))'
    gh api --method POST /markdown --input too_big.json

answers HTTP 403, `{"code": "too_large"}`, with the message "This API renders
Markdown text up to 400 KB in size." It is nothing like `head -c 8192`: `gh`
exits non-zero, so it raises `RendererUnavailable` and the failure policy below
applies -- exit 2, an `error:` line, and **nothing closed**. A silent truncation
is what #1541 was about; a loud refusal that closes nothing is the safe side of
the same question, and `test_a_render_refused_for_size_closes_nothing_loudly`
pins it.

Can a pull-request body reach that cap? Not through the text itself. GitHub
caps an issue or pull-request body at 65,536 characters -- widely reported as
the 422 `Body is too long (maximum is 65536 characters)`, and **UNVERIFIED
here**, because measuring it would mean creating a pull request. At four UTF-8
bytes per character that is at most 262,144 bytes, under everything measured to
render. It can reach the cap through *this module's own encoding*, though:

    python3 -c 'import json; b = "\U0001F600" * 65536;
        print(len(json.dumps({"mode": "gfm", "context":
            "robotrocketscience/aelfrice", "text": b}).encode()),
              len(json.dumps({"mode": "gfm", "context":
            "robotrocketscience/aelfrice", "text": b},
            ensure_ascii=False).encode()))'

prints `786501 262213`. `json.dumps` escapes every non-ASCII character to
`\uXXXX` by default, three times the size for an astral character, and the
renderer measures the request it receives; a legal 65,536-character emoji body
was refused with that 403 until `render_markdown` passed `ensure_ascii=False`.
Unescaped, the payload is the body's UTF-8 size plus a fixed envelope, so no
body GitHub would accept can reach the cap. `RENDER_LIMIT_BYTES` and
`MAX_BODY_CHARACTERS` below record both numbers so a test can assert the
relation instead of a reader having to trust this paragraph.

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

A ruling is not allowed to be silent. `Closes : #10` is a keyword the tool
found and declined, so it prints a `warning:` line naming the spelling and
`DECLINED_SEPARATOR` as the reason, exactly like every other refusal -- see
"Nothing is silently unmatched". `DECLINED_RE` is what recognises it: the text
before the anchor ends in a keyword followed by nothing but colons and
whitespace, and `ADJACENT_RE` did not accept it. A form GitHub itself does not
act on is *not* a declined keyword and stays silent, because there is no close
directive in it to report: `precloses #7`, `Closes issue #8` and `Closes#6`
(which renders no anchor at all) are silent for the same reason a body with no
keyword is silent.

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

### Reading GitHub's own URL must not be able to abort the step

`close_directives` raises `RendererUnavailable` for an anchor whose `data-url`
it cannot read. That is right for an output shape GitHub does not emit today --
the module then cannot tell whose issue it is looking at -- but the failure
policy below turns it into **nothing closed for the whole body**, not one
anchor skipped. So the pattern that reads those URLs is a place where being
too strict is far worse than being too loose, and it was too strict: it spelled
a repository name as "alphanumeric at each end", which was this module's
invention. GitHub's own rule reserves no position -- "The repository name must
not exceed 100 characters, and can only contain ASCII letters, digits, and the
characters `.`, `-`, and `_`":
https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository

`.github` is such a name, and a common one. A body writing

    Closes github/.github#5

    Closes #1549

renders both as issue-link anchors, the first with
`data-url="https://github.com/github/.github/issues/5"`. The pattern refused
that URL, and the shipped CLI answered exit 2 on an empty stdout, closing
neither -- including the issue in its own repository. The ruling for an anchor
resolving elsewhere is CROSS_REPO: one `warning:` line, and the rest of the
body still closes.

The repair is not a wider name grammar but no name grammar at all on this
side. `_ISSUE_HREF_RE` reads the owner and the name as whole path segments,
which cannot fail for anything GitHub is able to put there, because a path
segment never contains a `/`. Width decides nothing on its own: these segments
are only ever compared against `--repo`. Each shape the audit raised was then
settled against the live renderer rather than argued:

* **A leading dot, in the name or in the owner.** Read, and refused unless it
  is this repository. GitHub anchors `.github/.github#5` as `github/.github`,
  leaving the owner's dot outside the anchor, so that spelling cannot reach
  `data-url` at all; reading it anyway costs nothing.
* **A name at the documented 100-character maximum.** Read. No length is
  written into the pattern, because a length bound here is one more way to
  abort.
* **Percent-encoding.** `https://github.com/github/%2Egithub/issues/5` renders
  no anchor, so none arrives. If one ever did, nothing here decodes, so a
  percent-encoded spelling of *this* repository compares unequal and is
  refused -- a missed close rather than a wrong one, which is the standing
  asymmetry.
* **A trailing slash or a query string.** GitHub normalises `data-url` to the
  bare `/issues/N` spelling, and the pattern accepts the others regardless.

The source-level scans keep a name grammar, because they read the author's
prose rather than GitHub's output and need somewhere to stop. Theirs is
GitHub's documented character set, ended by the literal that must follow it --
`#`, `/issues/`, `/pull/` or `/discussions/`. That is not only a diagnostic
question: `DISCUSSION_RE` decides a refusal, so a repository whose name the
grammar could not spell had no protection at all from the wrong close below.

### One case fold, in one place

GitHub canonicalises `data-url` to lower case and leaves the body exactly as
the author typed it, so
`Closes https://github.com/RobotRocketScience/aelfrice/discussions/1549`
reaches `close_directives` as a lower-cased anchor and `discussion_targets` as
a mixed-case source. The comparison between them is correct only if both sides
fold, and folding in two places is a drift waiting to happen: either half could
be deleted on its own, and deleting the source half re-opens exactly the wrong
close below. `_identity` is the single place it happens, and both callers go
through it.

### The render is not an oracle for which object a reference names

Reading the render answers "is this text a reference?" It does **not** answer
"which object does that reference name", because GitHub normalises several
distinct URL shapes into the same `issue-link` anchor with an `/issues/N`
`data-url`, and one of them names a different number space entirely. A
discussions URL is the wrong close this section exists to stop:

    Closes https://github.com/robotrocketscience/aelfrice/discussions/1549

renders as `<a class="issue-link" data-url=".../issues/1549">#1549</a>`.
Discussions are numbered independently of issues, this repository has
discussions enabled (`gh api repos/robotrocketscience/aelfrice --jq
'.has_discussions'` prints `true`), so acting on that anchor closes an
unrelated issue. It is refused, out loud.

The check has to be at *source* level: by the time the anchor is read, GitHub
has already rewritten `data-url`, and the anchor is byte-for-byte what a plain
`#1549` produces. `DISCUSSION_RE` therefore scans the body for a discussions
URL and `parse` passes the `(owner, repo, number)` triples it found to
`close_directives`, which refuses any anchor resolving to one of them. That is
deliberately blunt: a body that writes both `Closes #1549` and a discussions
URL for 1549 loses the close and gets a warning, which is the module's standing
asymmetry -- a missed close is an open issue a human sees.

Every other shape was audited against the live renderer rather than reasoned
about, one `POST /markdown` per row, and each is ruled on here:

* `/issues/N` -> `issue-link`, `/issues/N`. Closes N. The plain case.
* `/pull/N` -> `issue-link`, `/issues/N`. Closes N. Pull requests and issues
  share one number space, so this names the same object GitHub would close.
* `/issues/N#issuecomment-...` and `/pull/N#discussion_r...` -> `issue-link`,
  `/issues/N`, anchor text `#N (comment)`. Closes N. The fragment names a
  comment *on* N, so N is still the object referenced.
* `/discussions/N` -> `issue-link`, `/issues/N`. **Refused**, per above.
* `/pull/N/files` -> an ordinary `<a href>`, no `issue-link` class. Not a
  reference; the source scan reports it as `NOT_LINKED`.
* `/commit/SHA` -> `<a class="commit-link">`. Not an issue reference at all.
* `https://github.com/orgs/ORG/discussions/N` -> an ordinary `<a href>`. Only
  repository-level discussions are rewritten, so nothing to refuse.
* `/milestone/N`, `/projects/N`, `/releases/tag/...`, `/blob/...`, `/wiki/...`
  -> ordinary `<a href>`. Not references.

`tests/data/merge_train_github_renders.json` holds the `urlforms` record this
audit came from, so the rulings are replayed against GitHub's own bytes; rerun
`python3 scripts/record_merge_train_renders.py --dry-run` if GitHub changes.

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
# whitespace. `:?\s*$` is why `Closes : #4` and `Closes::#4` do not fire.
#
# The front is the word boundary, and it comes in two spellings because the
# run of text is kept as a bounded suffix (see `_TAIL`). `_FRONT` allows `^`,
# which is right when the run really does start there -- a block boundary, or
# an anchor. `_CLIPPED_FRONT` does not, because in a window that was cut short
# `^` is an artefact of the cut: `... the last one is prefixed #7` would put
# `fixed` at position 0 of a six-character window and close #7. Requiring a
# real non-word character makes a match on the window a match on the whole run
# as well, so clipping can lose a close but can never invent one.
_FRONT = r"(?:^|\W)"
_CLIPPED_FRONT = r"\W"

# A keyword the adjacency rule accepts, and a keyword it declines. The second
# is what keeps a ruling from being silent: `Closes : #4` ends in a keyword
# followed by nothing but separators, so it is reported rather than passed
# over. `precloses #7` matches neither, because the boundary fails.
_ACCEPTED = r"(?P<keyword>" + _ALTERNATION + r"):?\s*$"
_DECLINED = r"(?P<keyword>" + _ALTERNATION + r")[\s:]*$"

ADJACENT_RE = re.compile(_FRONT + _ACCEPTED, re.IGNORECASE)
DECLINED_RE = re.compile(_FRONT + _DECLINED, re.IGNORECASE)
_CLIPPED_ADJACENT_RE = re.compile(_CLIPPED_FRONT + _ACCEPTED, re.IGNORECASE)
_CLIPPED_DECLINED_RE = re.compile(_CLIPPED_FRONT + _DECLINED, re.IGNORECASE)

# An owner or repository name, as GitHub's own documentation writes the rule:
# "The repository name must not exceed 100 characters, and can only contain
# ASCII letters, digits, and the characters `.`, `-`, and `_`."
# https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository
#
# No position is reserved, so `.github` is a legal name, and a common one --
# `gh api repos/github/.github` resolves. An earlier spelling here required an
# alphanumeric at each end; that was this module's invention rather than
# GitHub's rule, and it is what made `Closes github/.github#5` abort the step.
# See `_ISSUE_HREF_RE` for the half of that defect that mattered.
#
# The 100-character bound is deliberately NOT written into the pattern. Both
# users of `_NAME` scan the author's prose, and a bound there can only lose a
# match; losing a `DISCUSSION_RE` match is a wrong close. Each use is bounded
# instead by the literal that must follow it -- `#`, `/issues/`, `/pull/` or
# `/discussions/` -- so the greedy class cannot run away into prose.
_NAME = r"[A-Za-z0-9._-]+"

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

# A repository discussion written as a URL. GitHub rewrites this into an
# `issue-link` anchor whose `data-url` is the `/issues/N` spelling, so by the
# time the render is read the discussion is indistinguishable from an issue of
# the same number -- and discussions are numbered separately. The check must
# therefore run against the source. See "The render is not an oracle for which
# object a reference names" in the module docstring.
DISCUSSION_RE = re.compile(
    r"https?://github\.com/(?P<owner>" + _NAME + r")/(?P<repo>" + _NAME + r")"
    r"/discussions/(?P<number>\d+)",
    re.IGNORECASE,
)

# Why a refused candidate was refused. Each string lands verbatim in the
# merge-train step log and reads as the end of "ignored ... because it is".
CROSS_REPO = "a link to another repository, which this train does not close"
IN_QUOTE = "inside a block quote"
NOT_LINKED = "not linked by GitHub's renderer, so a merge commit would not close it"
FROM_DISCUSSION = (
    "written as a discussions URL, which GitHub renders as an issue link "
    "although discussions carry their own numbers"
)
DECLINED_SEPARATOR = (
    "written with a separator this train does not accept between the keyword "
    "and the reference"
)

# Above this many distinct issues in one body, say so on stderr. Not a cap:
# every issue found is still printed. A body naming this many is more likely
# a template or a paste than a real set of links, and the merge-train log is
# the only place a human would see that.
NOISY_COUNT = 20

# The renderer is one HTTPS round trip on a step that already holds the
# merge-train's only concurrency slot, so it is bounded. A timeout raises
# `RendererUnavailable` like any other failure and closes nothing.
RENDER_TIMEOUT_SECONDS = 30

# GitHub's published cap on a `POST /markdown` request: above this it answers
# HTTP 403 `too_large`. Measured, not assumed -- the module docstring carries
# the command. Recorded here so a test can assert that the largest body GitHub
# would accept still fits, rather than a reader having to trust a paragraph.
RENDER_LIMIT_BYTES = 400 * 1024

# The largest body GitHub accepts on an issue or pull request, in characters.
# UNVERIFIED here: measuring it would mean creating a pull request. It is the
# widely reported 422 `Body is too long (maximum is 65536 characters)`, and it
# is used only as an upper bound -- at four UTF-8 bytes per character that is
# 262,144 bytes, comfortably inside `RENDER_LIMIT_BYTES` once `render_markdown`
# stops escaping non-ASCII.
MAX_BODY_CHARACTERS = 65_536

# The anchor GitHub emits for an issue reference, and the shape of the URL it
# hangs on it. `data-url` is always the `/issues/N` spelling even when `href`
# points at `/pull/N`, so it is read first.
#
# The owner and the name are read as whole path segments here rather than
# through `_NAME`, because on this side a name grammar is a way to abort the
# whole step. This URL is GitHub's own output, and `close_directives` raises
# `RendererUnavailable` on a `data-url` it cannot read -- so a grammar narrower
# than GitHub's closes NOTHING for the entire body, this repository's own
# issues included. `Closes github/.github#5` beside `Closes #1549` did exactly
# that: GitHub anchors it with `data-url=".../github/.github/issues/5"`, the old
# pattern forbade the leading dot, and the step exited 2 having closed neither.
#
# A path segment can never contain `/`, so `[^/?#]+` reads every owner and name
# GitHub is able to put here, at any length and whatever it encodes. Being
# wider than the legal set costs nothing: these segments are only ever compared
# against `--repo`, so anything that is not this repository is refused out loud
# as CROSS_REPO rather than closed. What still raises is a `data-url` that is
# not a github.com issue or pull URL at all, which is the output-shape change
# the raise is actually for.
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

# How much of the preceding text the adjacency rules see. This is a
# performance knob and nothing else: because a clipped window refuses `^` as a
# boundary (see `_CLIPPED_FRONT`), every match on the window is also a match on
# the whole run, so no value of `_TAIL` can turn prose into a close. Shrinking
# it can only lose a close -- a keyword separated from its anchor by more than
# `_TAIL` characters of whitespace -- which is the safe direction and the
# reason the constant is allowed to exist at all. It must stay at least as long
# as the longest keyword plus its colon, its whitespace and the boundary
# character in front; `test_the_adjacency_window_cannot_be_trimmed_silently`
# pins both the bound and the shipped value.
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

    `ensure_ascii=False` is load-bearing, not tidiness. The renderer measures
    the request it receives and refuses one over `RENDER_LIMIT_BYTES`, and
    `json.dumps` escapes an astral character to twelve ASCII bytes by default,
    so the largest body GitHub accepts becomes a 786,501-byte request and is
    refused -- see the module docstring. Unescaped it is 262,213 bytes and
    renders. That makes the encoding of stdin load-bearing too, so it is named
    rather than inherited from the runner's locale.
    """
    run = subprocess.run if run is None else run
    payload = json.dumps(
        {"mode": "gfm", "context": repo, "text": body}, ensure_ascii=False
    )
    argv = ["gh", "api", "--method", "POST", "/markdown", "--input", "-"]
    try:
        proc = run(  # type: ignore[operator]
            argv,
            input=payload,
            capture_output=True,
            text=True,
            encoding="utf-8",
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


@dataclass(frozen=True)
class _Anchor:
    """One issue-link anchor, and what the text in front of it decided."""

    keyword: str | None
    declined: str | None
    text: str
    url: str
    in_quote: bool


class _AnchorScanner(HTMLParser):
    """Collects an `_Anchor` for every issue link in the rendered document.

    One left-to-right pass. `_tail` holds the text since the last block
    boundary, trimmed to the few characters the adjacency rules can need, so a
    paragraph of any length costs the same; `_clipped` records that the trim
    actually cut something, which is what stops the cut from being readable as
    the start of a block. See `_TAIL`.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.anchors: list[_Anchor] = []
        self._tail = ""
        self._clipped = False
        self._quote_depth = 0
        self._open: tuple[str | None, str | None, str, bool] | None = None
        self._anchor_text = ""

    # -- text ------------------------------------------------------------
    def _end_run(self) -> None:
        self._tail = ""
        self._clipped = False

    def handle_data(self, data: str) -> None:
        if self._open is not None:
            self._anchor_text += data
            return
        joined = self._tail + data
        if len(joined) > _TAIL:
            self._clipped = True
            joined = joined[-_TAIL:]
        self._tail = joined

    # -- tags ------------------------------------------------------------
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == _QUOTE_TAG:
            self._quote_depth += 1
        if tag not in _INLINE_TAGS:
            self._end_run()
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
            # An ordinary link. `handle_endtag` ends the run at its `</a>`,
            # so this reset is for the anchor that has none: html.parser does
            # not synthesise a close tag, and without it `Closes <a href=...>`
            # left unterminated would arm the issue-link anchor that follows.
            self._end_run()
            return
        url = attributes.get("data-url") or attributes.get("href", "")
        keyword, declined = _keyword_before(self._tail, clipped=self._clipped)
        self._open = (keyword, declined, url, self._quote_depth > 0)
        self._anchor_text = ""

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        # The self-closing spelling, `<br/>`, is the only input that reaches
        # here; GitHub emits `<br>`. Both halves run for it exactly as they
        # would for a start tag followed by an end tag.
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag == _QUOTE_TAG and self._quote_depth:
            self._quote_depth -= 1
        if tag == "a" and self._open is not None:
            keyword, declined, url, in_quote = self._open
            self.anchors.append(
                _Anchor(
                    keyword=keyword,
                    declined=declined,
                    text=self._anchor_text,
                    url=url,
                    in_quote=in_quote,
                )
            )
            self._open = None
        # An anchor ends the run whether or not it was an issue link, so
        # `Closes #1 #2` arms only #1.
        if tag not in _INLINE_TAGS or tag == "a":
            self._end_run()


def _keyword_before(tail: str, *, clipped: bool) -> tuple[str | None, str | None]:
    """The keyword the run of text ends in, and the spelling that was declined.

    Exactly one of the two is ever set. `clipped` says the run was cut to
    `_TAIL`, which forbids reading the cut as the start of a block.
    """
    accepted = (_CLIPPED_ADJACENT_RE if clipped else ADJACENT_RE).search(tail)
    if accepted is not None:
        return accepted.group("keyword"), None
    declined = (_CLIPPED_DECLINED_RE if clipped else DECLINED_RE).search(tail)
    if declined is not None:
        return None, tail[declined.start("keyword") :]
    return None, None


def _identity(owner: str, repo: str, number: str | int) -> tuple[str, str, int]:
    """One `(owner, repo, number)` triple, folded the single way both sides use.

    The discussions refusal compares a triple read from the *source* body
    against one read from GitHub's *rendered* anchor, and the two sides do not
    agree on case: GitHub canonicalises `data-url` to lower case while the body
    keeps whatever the author typed, so
    `Closes https://github.com/RobotRocketScience/aelfrice/discussions/1549`
    matches the anchor only once both are folded. Folding in two places is a
    drift waiting to happen -- half of it can be deleted with the suite still
    green -- so it is done here, once, and both callers go through it.
    """
    return owner.lower(), repo.lower(), int(number)


def close_directives(
    html: str,
    repo: str,
    *,
    discussion_sources: frozenset[tuple[str, str, int]] = frozenset(),
) -> tuple[list[int], list[Rejection]]:
    """Split the rendered document's close directives into acted-on and refused.

    A directive is an issue-link anchor with one of the nine keywords
    immediately before it in the same block. Everything else in the document,
    anchor or not, is a mention.

    `discussion_sources` holds the `(owner, repo, number)` triples the *source*
    body spelled as a discussions URL. It cannot be recovered from `html`:
    GitHub has already rewritten such a reference into an anchor identical to
    a plain `#N`. `parse` supplies it; a caller that does not gets no
    protection from it, which is why the wiring is pinned by its own test.
    """
    scanner = _AnchorScanner()
    try:
        scanner.feed(html)
        scanner.close()
    except Exception as exc:  # pragma: no cover - html.parser is lenient
        raise RendererUnavailable(f"cannot read the rendered body: {exc}") from exc

    found: set[int] = set()
    refused: list[Rejection] = []
    for anchor in scanner.anchors:
        if anchor.keyword is None:
            if anchor.declined is not None:
                # A ruling of this module's own, and rulings are not silent.
                refused.append(
                    Rejection(
                        text=" ".join(f"{anchor.declined}{anchor.text}".split()),
                        reason=DECLINED_SEPARATOR,
                    )
                )
            continue
        target = _ISSUE_HREF_RE.match(anchor.url)
        if target is None:
            raise RendererUnavailable(
                f"an issue-link anchor carried an unreadable URL: {anchor.url!r}"
            )
        quoted = f"{anchor.keyword} {anchor.text}".strip()
        named = _identity(target["owner"], target["repo"], target["number"])
        if named in discussion_sources:
            # First, because it is the only refusal where the anchor lies
            # about which object the reference named.
            refused.append(Rejection(text=quoted, reason=FROM_DISCUSSION))
            continue
        if anchor.in_quote:
            refused.append(Rejection(text=quoted, reason=IN_QUOTE))
            continue
        if f"{named[0]}/{named[1]}" != repo.lower():
            refused.append(Rejection(text=quoted, reason=CROSS_REPO))
            continue
        found.add(named[2])
    return sorted(found), refused


def discussion_targets(body: str) -> frozenset[tuple[str, str, int]]:
    """Every `(owner, repo, number)` the body spells as a discussions URL.

    Folded through `_identity`, because these triples are compared against ones
    built from GitHub's lower-cased `data-url`.
    """
    return frozenset(
        _identity(m["owner"], m["repo"], m["number"])
        for m in DISCUSSION_RE.finditer(body)
    )


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
    found, refused = close_directives(
        html, repo, discussion_sources=discussion_targets(body)
    )
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
