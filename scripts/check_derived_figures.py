#!/usr/bin/env python3
"""#1469 — a published figure must name the script that re-derives it.

Six of eight PRs reviewed in the 2026-08-10 board sweep shipped at least one
stale or false published figure. Every one was caught by a human re-deriving
it; none by CI. This is the gate that changes that.

## The marker

A published figure carries a machine-readable marker naming its producer, the
key that producer emits it under, and the value that was published -- here, the
session-end lock prompt's item cap of 20:
    <!-- derived: benchmarks/published_constants.py#stop_prompt_max_items = 20 -->

In a Python file the identical marker is written inside a comment, where it is
held to the comment run it sits in and not to the statement below it:
    # A cap of 20 leaves the median session whole
    # <!-- derived: benchmarks/published_constants.py#stop_prompt_max_items = 20 -->
    STOP_PROMPT_MAX_ITEMS: Final[int] = 20

A marker's value must also *be* a figure in the text around it. Producer checks
bind the producer to the marker and self-consistency binds markers to each
other; neither reads the number a reader sees, so before this rule an author
could publish one number in prose and a different one in its marker and stay
green. The direction is marker -> prose only: an unmarked figure is still
grandfathered.

The syntax is the same everywhere so one scanner reads both surfaces. Six of
the seven #1469 instances shipped in source as well as in the CHANGELOG, and
#1445's number shipped in three files, so a CHANGELOG-only scanner would have
guarded a third of the corpus.

## Two classes, and the reader can tell them apart

**Store-free** — no `corpus=` attribute. CI re-runs the producer and hard-fails
on any difference between the emitted value and the published one. This is the
real gate.

**Store-backed** — carries `corpus=<label>@<YYYY-MM-DD>` and
`producer-sha=<12 hex>`. `benchmarks/stop_prompt_block_bounds.py`,
`benchmarks/scan_admission_funnel.py` and `benchmarks/sidecar_rebuild_rate.py`
read a real belief store; a public runner has none and the lab corpus must not
go there (#1456). Re-running them in CI is impossible, so the marker instead
records *which corpus, on what date* produced the figure, and CI checks the two
things it still can see:

  * **self-consistency** (hard) — the same `producer#key` published in several
    files must carry the same value. #1449 shipped 44,683 in one file and
    44,687 in four others, in one PR; that is exactly this check.
  * **code staleness** (advisory) — has the producer's source changed since the
    figure was stamped? #1445 is the clean example: correct when measured, and
    one commit later the code moved underneath it. Advisory because a producer
    edit does not prove the figure moved, and because the re-measure needs a
    store nobody in public CI has.

The class is readable off the marker itself: `corpus=` present means "this one
cannot be re-run here, and here is what it was measured against".

## Grandfathering, and the one claim that is not grandfathered

A figure with no marker is allowed. The annotation backlog is large and a gate
that fails on it would be turned off within a day. `--list-unmarked` enumerates
what is still unannotated so the backlog is countable rather than notional.

Inline code is masked before a figure is extracted, with one exception: a span
whose whole content is a number is still a figure. Without that exception the
hard rule below was satisfiable by typing two backticks, and two published
shares in `CHANGELOG/v4.md` were invisible to `--list-unmarked`. `--mask-delta`
prices the alternatives against the tree it is run on. What stays invisible,
and is documented rather than closed, is a figure inside a span that also
carries words.

The exception is the overclaim sentence, whose shape is
``<script> re-derives **every** figure here``. #1445 and #1447 both shipped it
over scripts that emitted about a
third of the entry's numbers. That sentence is only permitted in an entry where
every figure carries a marker, and that check is **hard**. Making a strong claim
is allowed; making it for free is not.

## Which text is code

Fenced code is excluded before anything else reads a document, so a figure
quoted in a shell transcript is not mistaken for a published claim. The
exclusion is a line-by-line block scanner (`code_block_spans`) following
CommonMark's fenced-code rules: an opener of three or more backticks or tildes
indented at most three spaces, a backtick opener's info string carrying no
backtick, and a closer of the same character, at least as long, with nothing
after it but whitespace.

It replaced a rule that had no notion of a block at all. The inline-code regex
paired backticks across the whole document, so a three-backtick fence line --
an odd run -- left a dangling opener that blanked every line down to the next
backtick anywhere below it. `docs/user/PRIVACY.md` is the witness: six
delimiter lines, and every marker placed on that page parsed as nothing, which
is the count a page with no markers in it also reports. A marker that never
parses is indistinguishable from a figure nobody annotated, so the gate
reported success over an unguarded figure -- the #1160 defect class, inside the
gate built to stop it.

Two deliberate divergences, both measured on this tree rather than argued:

* **An unterminated opener masks nothing.** CommonMark runs such a block to the
  end of its container; here the container would be the document, and blanking
  to the end of the document is the exact shape of the defect being fixed. It
  costs nothing to diverge *today*: no file in the scanned corpus carries an
  unterminated opener, so the two readings report the same figures and the same
  markers on this tree. `unterminated_fence` is what keeps that true -- a file
  that grows one raises an advisory naming the line, because the divergence is
  only free while nothing exercises it, and a silent divergence is how a gate
  goes quiet. The costs either way are not symmetric: masking prose
  makes a marker vanish while the gate prints success, whereas leaving an
  unclosed block's body visible makes a marker inside it parse, and the binding
  and producer checks then run on it, loudly. A second property falls out of it
  -- masking a fragment can never blank more than masking the whole document,
  so a caller holding half a block is safe.
* **Indented code blocks are not masked.** Four-space indentation is the body
  of every Python function in the corpus, and `.py` files are scanned, so
  masking them would blank most of `src/`. The previous rule did not mask them
  either, and nothing regresses.

Not masked, and not previously masked either: HTML blocks, link reference
definitions, and fences nested in a blockquote or list item, which need
container parsing this scanner does not do. A fence indented into a list item
is therefore read at its literal column.

### The inline pass: run length, and one paragraph

Blocks are masked first and the inline rule then runs on what is left, region
by region. Inside a region the rule is CommonMark's own, and the property it
turns on is **backtick run length**: a code span opens on a maximal run of
backticks and closes on the next run of *exactly* that length. A run with no
equal-length partner is literal text, not a dangling opener.

That is the half of the defect a block scanner alone does not reach. Run
length is what makes a three-backtick delimiter unpairable with the single
backtick that follows it, and an odd run needs no fence to do damage:
`docs/user/CONFIG.md` publishes a table row naming fences in code font, three
backticks in the middle of a sentence, and under the old rule the two spare
backticks ate forward from there. A fence scanner would not have looked at
that line at all, because it is not a delimiter line.

The second bound is the paragraph. Markdown parses inline content inside one
block, so a backtick in one paragraph cannot pair with one in the next.
Without it a single mistyped backtick pairs with the next backtick anywhere
below it -- and masking blocks first makes that *worse*, not better, because
the stray no longer has a fence delimiter to pair against and reaches forward
past a marker instead.

Together they give an absolute property rather than a comparison, and the
property is the acceptance bar: **a marker outside every fenced block, in a
paragraph carrying no backtick, is always parsed, at its true line.**
`tests/test_code_span_scanner_1556.py` fuzzes it over a seeded corpus of
generated fence arrangements.

One residual, stated rather than closed, and bounded by that property.
Bounding a span shifts where one can start: when a run can no longer reach
across a blank line, a later run in the next paragraph becomes an opener and
blanks a region the unbounded rule left alone. That can cost a marker the
unbounded rule kept -- `test_the_paragraph_bound_costs_a_marker_only_where_backticks_surround_it`
is the witness -- and every such marker sits in a paragraph carrying
backticks, so it was never covered by the bar above.

## Usage

    python3 scripts/check_derived_figures.py                # text checks only
    uv run python scripts/check_derived_figures.py --mode all
    python3 scripts/check_derived_figures.py --list-unmarked CHANGELOG/v4.md
    python3 scripts/check_derived_figures.py --mask-delta

`--mode text` is stdlib-only and needs no installed package, so it runs in the
`release-docs-check` job beside the other every-PR document gates. `--mode
producers` executes the store-free producers and therefore needs the package;
it runs in its own `ci.yml` job. `--mode all` is the local form.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import cast

REPO_ROOT = Path(__file__).resolve().parent.parent

# Directories scanned for markers and for overclaim sentences.
DEFAULT_ROOTS: tuple[str, ...] = (
    "CHANGELOG",
    "src",
    "benchmarks",
    "docs",
    "scripts",
    "tests",
    "README.md",
)

SCANNED_SUFFIXES: frozenset[str] = frozenset({".md", ".py"})

# Directories never scanned: generated output, fixtures and vendored trees.
SKIP_PARTS: frozenset[str] = frozenset({
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    "results", "fixtures", "oracle_fixtures", "corpus",
})

# The marker. `producer` is a repo-relative path, `key` the name the producer
# emits the figure under, `value` the published number as written.
MARKER_RE = re.compile(
    r"<!--\s*derived:\s*"
    r"(?P<producer>[^\s#]+)#(?P<key>[A-Za-z0-9_.\-]+)"
    r"\s*=\s*(?P<value>[^\s]+)"
    r"(?P<attrs>(?:\s+[a-z][a-z-]*=[^\s]+)*)"
    r"\s*-->"
)

ATTR_RE = re.compile(r"([a-z][a-z-]*)=([^\s]+)")

# `corpus=` is the class discriminator; `producer-sha=` is the staleness stamp.
CORPUS_ATTR = "corpus"
SHA_ATTR = "producer-sha"
KNOWN_ATTRS: frozenset[str] = frozenset({CORPUS_ATTR, SHA_ATTR})

# A corpus identity is a label and the date it was measured. Both halves are
# load-bearing: the label says which store, the date says which snapshot of it.
# #1449's 44,683-vs-44,687 split is two real snapshots five days apart, not an
# arithmetic error, and only the date says so.
CORPUS_RE = re.compile(r"^[A-Za-z0-9_.:/+-]+@\d{4}-\d{2}-\d{2}$")
SHA_RE = re.compile(r"^[0-9a-f]{12}$")

# The overclaim. Deliberately narrow — it fires on the sentence shape that
# actually shipped, not on every mention of re-derivation.
# `[*_]{0,2}` around each word is not decoration: #1445 shipped the sentence as
# ``re-derives **every** figure here``, and a pattern that could not see through
# the emphasis would have missed the exact instance this rule exists for.
_EMPH = r"[*_]{0,2}"
OVERCLAIM_RES: tuple[re.Pattern[str], ...] = (
    re.compile(
        rf"re-?derives?\s+{_EMPH}(?:every|all|each){_EMPH}\s+"
        rf"{_EMPH}(?:figure|number|value)",
        re.I,
    ),
    re.compile(
        rf"{_EMPH}(?:every|all|each){_EMPH}\s+{_EMPH}(?:figure|number|value)s?{_EMPH}\s+"
        r"(?:here|in this entry|in this section)[^.]{0,60}re-?derived",
        re.I,
    ),
)

# Inline code is a citation, not a claim. The rule has to be discussable in
# prose -- this file's own docstring quotes the sentence, and so does the
# CHANGELOG entry that introduces the rule -- and a checker that fires on every
# mention of itself is a checker with an exemption list, which is worse. Quoting
# the sentence inside backticks is the escape, and it is the same convention the
# repo already uses for naming code in prose.
# A citation of the sentence has to be able to contain a backtick, because the
# sentence itself names a script in code font, so the rule cannot be "one
# backtick to the next". It is CommonMark's: a span opens on a maximal run of
# backticks and closes on the next run of exactly that length, within one
# paragraph. See "### The inline pass" in the module docstring.
_NON_NEWLINE_RE = re.compile(r"[^\n]")

# A blank line. An inline span may straddle lines but never a blank one.
_PARAGRAPH_BREAK_RE = re.compile(r"\n[ \t\r]*\n")

# A maximal run of backticks. Run length is the whole point: a run of three is
# a fence delimiter, not two openers and a spare, and it can only ever pair
# with another run of three.
_BACKTICK_RUN_RE = re.compile(r"`+")


def paragraph_regions(text: str) -> list[tuple[int, int]]:
    """`(start, end)` offsets of each blank-line-delimited region of `text`.

    The regions tile the text and never overlap, so a span found in one cannot
    reach into another. The blank line itself is the boundary; which side of it
    the newlines land on does not matter, because a region boundary can only
    ever fall on whitespace.
    """
    out: list[tuple[int, int]] = []
    lo = 0
    for match in _PARAGRAPH_BREAK_RE.finditer(text):
        out.append((lo, match.start() + 1))
        lo = match.end()
    out.append((lo, len(text)))
    return out


def code_span_spans(text: str) -> list[tuple[int, int]]:
    """`(start, end)` offsets of every inline code span in `text`.

    CommonMark's rule, with the paragraph bound Markdown's block structure
    already implies. Walk the backtick runs of one paragraph left to right; the
    first run that has a later run of the same length opens a span that closes
    on it, and scanning resumes after the closer. A run with no equal-length
    partner is ordinary text and the walk steps past it -- which is the whole
    repair: under the old rule a run of three left a spare opener behind, and
    everything down to the next backtick in the file disappeared.
    """
    spans: list[tuple[int, int]] = []
    for lo, hi in paragraph_regions(text):
        runs = [
            (m.start(), m.end() - m.start())
            for m in _BACKTICK_RUN_RE.finditer(text, lo, hi)
        ]
        i = 0
        while i < len(runs):
            start, length = runs[i]
            j = i + 1
            while j < len(runs) and runs[j][1] != length:
                j += 1
            if j == len(runs):
                i += 1
                continue
            spans.append((start, runs[j][0] + length))
            i = j + 1
    return spans


def sub_code_spans(text: str, repl: Callable[[str], str]) -> str:
    """`text` with `repl` applied to each inline code span, in order.

    The replacement for `re.sub` over a span pattern. A regex cannot express
    "a run of n backticks closed by a run of n backticks" -- a backreference
    matches the same *text*, not the same length under a maximality rule -- so
    the spans are found first and spliced here.
    """
    spans = code_span_spans(text)
    if not spans:
        return text
    out: list[str] = []
    prev = 0
    for start, end in spans:
        out.append(text[prev:start])
        out.append(repl(text[start:end]))
        prev = end
    out.append(text[prev:])
    return "".join(out)


def split_lines(text: str) -> list[str]:
    """`text` split into the lines this file numbers by.

    A newline, and nothing else. `str.splitlines()` also breaks on VT, FF, FS,
    GS, RS, NEL, LS, PS and a lone CR, none of which an editor, a diff, or a
    GitHub annotation counts as a line, and none of which `parse_markers`
    counts either -- every line number here is a `count("\\n", 0, offset) + 1`.
    Mixing the two splitters is not cosmetic: `scannable_entries` indexes the
    masked line list with numbers taken from the raw one, masking replaces a
    block's body with spaces, and one VT inside a fenced block therefore made
    the masked list one line shorter and slid every entry after it. The gate
    then reported a hard binding failure, on a correct page, at the wrong line.

    The trailing empty element `split` leaves on a newline-terminated file is
    dropped, so this matches `splitlines()` on the text that has neither.
    """
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _uncited(text: str) -> str:
    """`text` with inline-code spans blanked, for claim detection."""
    return sub_code_spans(text, lambda span: " ")


def _uncited_inplace(text: str) -> str:
    """`_uncited`, but same length, so offsets still give the right line.

    Marker parsing uses this rather than `_uncited`: a marker quoted inside
    backticks is a citation of the syntax, not a published figure, and the same
    convention already governs the overclaim sentence. Before this, the
    CHANGELOG entry that *introduces* the marker was itself parsed as carrying
    one, so documenting the format published a figure.
    """
    # Newlines are kept, not blanked with the rest. A code span can straddle
    # lines, and eating its newlines shifts every reported line number after it
    # -- `src/aelfrice/hook.py` markers came back two lines early.
    return sub_code_spans(text, lambda span: _NON_NEWLINE_RE.sub(" ", span))


# --------------------------------------------------------------------------
# Fenced code. See "## Which text is code" in the module docstring.
# --------------------------------------------------------------------------

# One candidate delimiter line: optional indent, a run of three or more
# backticks or tildes, then the rest of the line. Whether the line is an
# opener, a closer or ordinary text is decided by `code_block_spans`, not here
# -- that decision needs the scanner's state, and no regex has it.
_FENCE_LINE_RE = re.compile(r"^(?P<indent>[ \t]*)(?P<fence>`{3,}|~{3,})(?P<info>.*)$")

# CommonMark: a code fence may be indented by at most three spaces. The fourth
# space starts an indented code block instead, which is a different construct
# and one this scanner deliberately does not mask -- see the docstring.
MAX_FENCE_INDENT = 3


def _scan_fences(text: str) -> tuple[list[tuple[int, int]], tuple[int, str] | None]:
    """Every terminated fenced block, and the opener left dangling, if any.

    One state machine answering both questions, because two would drift: the
    advisory exists to say "this document exercises the divergence", and it can
    only say that if it is reading the same fences the masker read.
    """
    spans: list[tuple[int, int]] = []
    pos = 0
    open_char = ""
    open_len = 0
    open_start = 0
    open_line = 0
    line_no = 0
    for line in text.split("\n"):
        line_no += 1
        end = pos + len(line)
        match = _FENCE_LINE_RE.match(line)
        if match is not None:
            fence = match.group("fence")
            info = match.group("info")
            # Tabs are worth four columns, so a tab-indented delimiter is not
            # smuggled under the three-space limit.
            indented = len(match.group("indent").expandtabs(4)) > MAX_FENCE_INDENT
            if open_char:
                # A closer is the same character, at least as long, and carries
                # no info string. ``` ```python ``` inside an open block is body
                # text, not a closer. `info.strip()` rather than `info` is what
                # makes a CRLF document work: the carriage return is trailing
                # whitespace on the delimiter line, which CommonMark allows,
                # and reading it as an info string leaves every block in the
                # document unterminated.
                if (
                    not indented
                    and fence[0] == open_char
                    and len(fence) >= open_len
                    and not info.strip()
                ):
                    spans.append((open_start, end))
                    open_char, open_len = "", 0
            elif not indented and not (fence[0] == "`" and "`" in info):
                # A backtick opener's info string may not contain a backtick.
                # That rule alone disqualifies the one live delimiter-looking
                # line this repo has in prose,
                # `tests/test_noise_harness_and_fences_1371.py:198`.
                open_char, open_len, open_start = fence[0], len(fence), pos
                open_line = line_no
        pos = end + 1
    dangling = (open_line, open_char * open_len) if open_char else None
    return spans, dangling


def code_block_spans(text: str) -> list[tuple[int, int]]:
    """`(start, end)` character offsets of every *terminated* fenced block.

    A block runs from the first character of its opening delimiter line to the
    last character of its closing delimiter line, newline excluded. Offsets, so
    the caller can blank the span in place and keep every line number.

    The rules are CommonMark's, with one divergence stated in the module
    docstring: an unterminated opener yields no span at all.
    """
    return _scan_fences(text)[0]


def unterminated_fence(text: str) -> tuple[int, str] | None:
    """`(line, delimiter)` of an opening fence never closed, or None.

    The divergence from CommonMark, made visible. Nothing here masks such a
    block, so its body is read as prose: a figure inside it becomes a published
    claim and a marker inside it becomes a real marker. That is the safe
    direction -- it fails loudly rather than quietly -- but it is still a
    reading no author asked for, so the gate says so.
    """
    return _scan_fences(text)[1]


def _blank_blocks(text: str, outside: Callable[[str], str]) -> str:
    """Blank every fenced block in `text`; run `outside` on what is left.

    `outside` is applied per region rather than to the joined result because an
    inline code span may not straddle a fenced block. Running the span rule
    over a document whose blocks are already blank would let it.

    Length-preserving, which is the contract the rest of this file depends on:
    every reported line number is a `count("\\n", 0, offset)`, so a masker that
    changed any offset would move diagnostics off the line they describe.
    `outside` must preserve length too.
    """
    spans = code_block_spans(text)
    if not spans:
        return outside(text)
    out: list[str] = []
    prev = 0
    for start, end in spans:
        out.append(outside(text[prev:start]))
        out.append(_NON_NEWLINE_RE.sub(" ", text[start:end]))
        prev = end
    out.append(outside(text[prev:]))
    return "".join(out)


def mask_code_blocks(text: str) -> str:
    """`text` with fenced code blanked and nothing else touched."""
    return _blank_blocks(text, lambda region: region)


def mask_document(text: str) -> str:
    """`text` with fenced code blanked and inline code blanked around it.

    The form every marker scan reads. `read_scannable` is the same thing off
    disk.
    """
    return _blank_blocks(text, _uncited_inplace)


def scannable_entries(path: Path) -> list[tuple[int, str]]:
    """`split_entries` over `path`, with fenced code blanked in each entry.

    Blocks are masked over the whole document and the entries are then sliced
    out of the result; the boundaries themselves are still read off the raw
    file. Splitting the masked text instead would let masking redraw the
    entries -- a blanked block is a run of whitespace-only lines, and
    `split_entries` ends an entry at a blank line. Masking is
    length-preserving and both line lists come from `split_lines`, so the two
    correspond exactly -- see that function for the splitter that made them
    disagree.

    The inline-code rule is deliberately *not* applied here: it stays with
    `extract_figures`, per entry, where #1469 put it. Only the block decision
    needs the whole document.
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    masked_lines = split_lines(mask_code_blocks(raw))
    out: list[tuple[int, str]] = []
    for start, body in split_entries(path, raw):
        stop = start + body.count("\n") + 1
        out.append((start, "\n".join(masked_lines[start - 1 : stop - 1])))
    return out


def read_scannable(path: Path) -> str:
    """`path`'s text, masked once, ready for every whole-file scan here.

    Whole-file: a block's opener and its closer can land in different entries
    once `split_entries` has run, and a scanner handed half a block has to
    guess which half it is holding. That guess is the defect. `scannable_entries`
    is the per-entry form, and it makes the block decision over the whole
    document for the same reason.
    """
    return mask_document(path.read_text(encoding="utf-8", errors="replace"))


# Figure extraction. Applied only inside an entry that carries an overclaim
# sentence, or under --list-unmarked.
#
# Masked out before extraction, because none of these is a measured figure:
# markdown links, issue refs, dotted version strings, ISO dates, section refs,
# and the markers themselves. Inline code is masked too, but conditionally --
# see `_mask_code_spans`.
_MASKS: tuple[re.Pattern[str], ...] = (
    re.compile(r"<!--.*?-->", re.S),
    re.compile(r"\[[^\]]*\]\([^)]*\)"),
    re.compile(r"https?://\S+"),
    re.compile(r"#\d+"),
    re.compile(r"\bv?\d+(?:\.\d+){2,}"),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"§\s*\d+(?:\.\d+)*"),
)

# A code span whose whole content is a number. This is the one span an author
# reads as a published figure rather than as code, and it is this repo's house
# style for a measured value.
_BARE_FIGURE_SPAN_RE = re.compile(r"\A\s*\d+(?:[,_]\d{3})*(?:\.\d+)?\s*(?:%|x|×)?\s*\Z")


def _mask_code_spans(text: str) -> str:
    """Blank inline-code spans, keeping one that is wholly a number.

    Masking every span put a published figure outside every check in this file
    -- including the hard overclaim rule, which an author could then satisfy by
    typing two backticks. It was not hypothetical: `CHANGELOG/v4.md`'s #1356
    entry publishes its two headline shares as ``93.69%`` and ``94.86%`` in
    code font, and `--list-unmarked` reported neither, which falsified this
    gate's own claim that grandfathered is not the same as invisible.

    A span that is anything else stays masked, because a false positive here
    makes the overclaim rule unsatisfiable, which is the same as deleting it:
    ``STOP_PROMPT_MAX_ITEMS = 20``, ``[:16]`` and ``busy_timeout=5000`` are
    code, not figures.

    The narrowing is measured rather than argued. `--mask-delta` re-derives, on
    whatever tree it is run against, how many figures each of the three
    candidate rules makes visible; dropping the mask entirely costs an order of
    magnitude more than this rule does.

    The residue is documented rather than closed: a figure inside a span that
    also carries words -- ``93.69% of rows`` -- is still invisible here.
    """

    def repl(span: str) -> str:
        inner = span.strip("`")
        return f" {inner} " if _BARE_FIGURE_SPAN_RE.match(inner) else " "

    return sub_code_spans(text, repl)


def _mask_every_code_span(text: str) -> str:
    """Every span masked, the bare-number exception included. `--mask-delta`."""
    return sub_code_spans(text, lambda span: " ")


def _mask_no_code_span(text: str) -> str:
    """No inline-code mask at all. Kept for `--mask-delta` only."""
    return text

# Thousands groups are matched as `,ddd` / `_ddd` rather than as a loose
# `[\d,_]*` class, which swallowed the sentence comma after a figure and
# reported `41,929,` as the unmarked value -- a diagnostic the author cannot
# grep for is a diagnostic they ignore.
FIGURE_RE = re.compile(r"(?<![\w.])\d+(?:[,_]\d{3})*(?:\.\d+)?\s*(?:%|x|×)?")


class Marker:
    """One `<!-- derived: ... -->` occurrence."""

    __slots__ = ("path", "line", "producer", "key", "value", "corpus", "sha", "errors")

    def __init__(
        self,
        path: Path,
        line: int,
        producer: str,
        key: str,
        value: str,
        corpus: str | None,
        sha: str | None,
        errors: list[str],
    ) -> None:
        self.path = path
        self.line = line
        self.producer = producer
        self.key = key
        self.value = value
        self.corpus = corpus
        self.sha = sha
        self.errors = errors

    @property
    def ident(self) -> str:
        return f"{self.producer}#{self.key}"

    @property
    def store_backed(self) -> bool:
        return self.corpus is not None


def normalise(value: object) -> str:
    """Canonical form for comparing a published rendering to an emitted value.

    `11,508`, `11508` and `11_508` are one figure; so are `299.7x` and `299.7`,
    and `8.69%` and `8.69`. Trailing zeros are dropped so `20` and `20.0` agree
    -- a producer emitting an int and prose writing a float is not a defect.
    """
    text = str(value).strip().rstrip("%xX×").replace(",", "").replace("_", "")
    try:
        num = float(text)
    except ValueError:
        return text.casefold()
    if num == int(num):
        return str(int(num))
    return repr(num)


def iter_files(roots: list[str]) -> list[Path]:
    """Every scanned file under `roots`, sorted, deterministic."""
    out: list[Path] = []
    for root in roots:
        base = REPO_ROOT / root
        if base.is_file():
            if base.suffix in SCANNED_SUFFIXES:
                out.append(base)
            continue
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix not in SCANNED_SUFFIXES:
                continue
            if SKIP_PARTS & set(path.relative_to(REPO_ROOT).parts):
                continue
            out.append(path)
    return sorted(set(out))


def parse_markers(path: Path, text: str) -> list[Marker]:
    """Every marker in `text`, with grammar errors attached rather than raised.

    A malformed marker is reported, never skipped. A marker the scanner cannot
    read is a figure nobody is guarding while the prose says otherwise, which is
    worse than no marker at all.
    """
    markers: list[Marker] = []
    scanned = _uncited_inplace(text)
    for match in MARKER_RE.finditer(scanned):
        line = scanned.count("\n", 0, match.start()) + 1
        attrs: dict[str, str] = {}
        errors: list[str] = []
        for name, val in ATTR_RE.findall(match.group("attrs") or ""):
            if name in attrs:
                errors.append(f"duplicate attribute {name!r}")
            attrs[name] = val
        for name in attrs:
            if name not in KNOWN_ATTRS:
                errors.append(
                    f"unknown attribute {name!r}; known: {sorted(KNOWN_ATTRS)}"
                )
        corpus = attrs.get(CORPUS_ATTR)
        sha = attrs.get(SHA_ATTR)
        if corpus is not None:
            if not CORPUS_RE.match(corpus):
                errors.append(
                    f"corpus={corpus!r} is not '<label>@<YYYY-MM-DD>'; the date "
                    "is what distinguishes two snapshots of one store"
                )
            if sha is None:
                errors.append(
                    "store-backed marker (corpus=) must also carry "
                    "producer-sha=<12 hex> or staleness cannot be checked"
                )
        elif sha is not None:
            errors.append(
                "producer-sha= without corpus=: a store-free figure is checked "
                "by re-running the producer, so the stamp is misleading"
            )
        if sha is not None and not SHA_RE.match(sha):
            errors.append(f"producer-sha={sha!r} is not 12 lowercase hex digits")
        markers.append(
            Marker(
                path=path,
                line=line,
                producer=match.group("producer"),
                key=match.group("key"),
                value=match.group("value"),
                corpus=corpus,
                sha=sha,
                errors=errors,
            )
        )
    return markers


# The stamp is excluded from the bytes it is a stamp over. Without this the
# hash of a producer that documents its own figures is a moving target: writing
# the new stamp into its docstring changes the bytes, which changes the hash,
# which invalidates the stamp just written. Measured, not reasoned about --
# `--restamp` ran twice on `benchmarks/spine_fan_in_baseline.py` and reported a
# different "current" hash each time. Stripping the stamp value makes
# restamping a fixed point in one pass, and costs nothing: the stamp is the one
# span of the file that can never be the reason a figure moved.
_SHA_STRIP_RE = re.compile(rb"producer-sha=[0-9a-f]{12}")


def producer_path(producer: str) -> Path | None:
    """`producer` resolved under the repo, or None if it escapes it.

    `REPO_ROOT / producer` is not containment: an absolute `producer` discards
    REPO_ROOT entirely (`Path('/repo') / '/bin/sh'` is `/bin/sh`), and `..`
    walks out. `--mode producers` execs this path under `sys.executable`, so a
    marker is committed input to a subprocess argv and has to be bounded.
    """
    if producer.startswith("/") or producer.startswith("\\"):
        return None
    path = (REPO_ROOT / producer).resolve()
    if not path.is_relative_to(REPO_ROOT.resolve()):
        return None
    return path


def producer_sha(producer: str) -> str | None:
    """First 12 hex of sha256 over the producer's bytes, or None if missing.

    Every `producer-sha=` value in the file is blanked first; see above.
    """
    path = producer_path(producer)
    if path is None or not path.is_file():
        return None
    body = _SHA_STRIP_RE.sub(b"producer-sha=", path.read_bytes())
    return hashlib.sha256(body).hexdigest()[:12]


def split_entries(path: Path, text: str) -> list[tuple[int, str]]:
    """Split a file into the units a claim -- and a marker -- is scoped to.

    A CHANGELOG entry is a top-level `- ` bullet plus its continuation lines --
    the unit a reader reads as one claim, and the unit `CHANGELOG/unreleased/`
    stores one per file. Anywhere else the unit is a blank-line-delimited
    paragraph, which is the widest scope a comment block can reasonably be held
    to.

    Two properties this has to hold, both of which the first version broke:

    * **Bullet scoping is markdown-only.** That version switched a whole file
      into bullet mode on a single `- ` line anywhere in it, so one docstring
      list turned every paragraph in the file into "not an entry" and the
      overclaim rule silently stopped running on that file. No share is
      published for how many files that was: the denominator is the scanned
      corpus, which moves on every merge.
      A `- ` inside a Python docstring is a list item, not a changelog entry.
    * **Every non-blank line lands in exactly one entry.** That version dropped
      everything above the first bullet and everything after a heading. A rule
      that cannot see a line cannot guard it and reports green while doing so,
      which is worse than not running at all. The invariant is not a claim
      here: `test_derived_figures_1469.py` asserts it over every scanned file.

    Returns `(first_line_number, block_text)` pairs, in file order.
    """
    lines = split_lines(text)
    bulleted = path.suffix == ".md"
    entries: list[tuple[int, str]] = []
    start: int | None = None
    buf: list[str] = []
    in_bullet = False

    def flush() -> None:
        nonlocal start, buf, in_bullet
        if start is not None and any(line.strip() for line in buf):
            entries.append((start, "\n".join(buf)))
        start, buf, in_bullet = None, [], False

    for idx, line in enumerate(lines, start=1):
        if bulleted and line.startswith("- "):
            flush()
            start, buf, in_bullet = idx, [line], True
            continue
        if in_bullet:
            # A bullet entry runs through blank and indented continuation
            # lines; it ends at the next top-level bullet, at a heading, or at
            # any other dedent to column zero.
            if not line.strip() or line[:1].isspace():
                buf.append(line)
                continue
            flush()
        if line.strip():
            if start is None:
                start = idx
            buf.append(line)
        else:
            flush()
    flush()
    return entries


def extract_figures(
    block: str, mask_spans: Callable[[str], str] = _mask_code_spans
) -> list[str]:
    """Numeric figures in `block`, in order, deduplicated by normalised value.

    `mask_spans` is the inline-code rule. It is a parameter so `--mask-delta`
    can price the alternatives against this tree instead of asserting a cost;
    every caller in the gate itself takes the default.
    """
    masked = mask_spans(block)
    for pattern in _MASKS:
        masked = pattern.sub(" ", masked)
    seen: set[str] = set()
    out: list[str] = []
    for match in FIGURE_RE.finditer(masked):
        raw = match.group(0).strip()
        norm = normalise(raw)
        if norm in seen:
            continue
        seen.add(norm)
        out.append(raw)
    return out


# A marker written as a Python comment annotates the comment it sits in, not
# the statement underneath it.
_PY_COMMENT_RE = re.compile(r"^\s*#")


def binding_block(path: Path, start: int, block: str, line: int) -> str:
    """The text a marker on `line` is held to, narrowed from its entry.

    In a Python file a comment marker is narrowed to the run of comment lines
    it belongs to -- the second example in this module's docstring. Without the
    narrowing, the `STOP_PROMPT_MAX_ITEMS` assignment two lines below satisfies
    the marker no matter what the sentence above it says: edit the comment to
    read 25 and the enclosing paragraph still contains a 20. The assignment is
    what the producer already re-runs; the sentence is the figure a reader
    actually sees, and it is the one this scope holds.
    """
    if path.suffix != ".py":
        return block
    lines = block.split("\n")
    idx = line - start
    if not (0 <= idx < len(lines)) or not _PY_COMMENT_RE.match(lines[idx]):
        return block
    lo = idx
    while lo > 0 and _PY_COMMENT_RE.match(lines[lo - 1]):
        lo -= 1
    hi = idx
    while hi + 1 < len(lines) and _PY_COMMENT_RE.match(lines[hi + 1]):
        hi += 1
    return "\n".join(lines[lo : hi + 1])


def rel(path: Path) -> str:
    """Repo-relative path, falling back to the absolute one.

    The fallback is not decoration: the tests drive the same functions over a
    tmp_path tree, and a `relative_to` that raised there would mean the live
    scan and the unit scan run different code.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


class Report:
    """Accumulates findings; hard ones set the exit code, advisory ones do not."""

    def __init__(self, github: bool) -> None:
        self.github = github
        self.hard: list[str] = []
        self.advisory: list[str] = []

    def fail(self, path: Path, line: int, message: str) -> None:
        self.hard.append(f"{rel(path)}:{line}: {message}")
        if self.github:
            print(f"::error file={rel(path)},line={line}::{message}")

    def warn(self, path: Path, line: int, message: str) -> None:
        self.advisory.append(f"{rel(path)}:{line}: {message}")
        if self.github:
            print(f"::warning file={rel(path)},line={line}::{message}")


def check_binding(files: list[Path], markers: list[Marker], report: Report) -> None:
    """Every marker's value must be the figure the surrounding text publishes.

    This is the link the rest of the gate does not make. Producer checks bind
    the producer to the marker; self-consistency binds markers to each other.
    Neither of them looks at the number a reader sees, so both stay green while
    the prose says something else -- which is the whole failure mode #1469 was
    filed for, reproduced through #1469's own gate.

    Two ways it bites. Editing a published figure and leaving its marker alone
    (`**3,448,428 bytes**` -> `**9,999,999 bytes**`) now fails, because 3448428
    is no longer in the block. And the repair path when a producer legitimately
    moves -- bump the six markers to 21 and leave every sentence around them
    reading 20 -- now fails at all six sites for the same reason.

    The direction is marker -> text, not text -> marker: an unmarked figure
    stays grandfathered (see `--list-unmarked`), and only the overclaim
    sentence buys out of that.
    """
    by_path: dict[Path, list[Marker]] = {}
    for marker in markers:
        by_path.setdefault(marker.path, []).append(marker)
    for path, group in sorted(by_path.items(), key=lambda kv: str(kv[0])):
        entries = scannable_entries(path)
        for marker in group:
            block = next(
                (
                    binding_block(path, start, body, marker.line)
                    for start, body in entries
                    if start <= marker.line <= start + body.count("\n")
                ),
                None,
            )
            figures = [] if block is None else extract_figures(block)
            want = normalise(marker.value)
            if want in {normalise(f) for f in figures}:
                continue
            shown = ", ".join(figures[:8]) if figures else "none"
            report.fail(
                marker.path,
                marker.line,
                f"{marker.ident} is published as {marker.value}, but no figure "
                f"reading {marker.value} appears in the text it annotates "
                f"(figures there: {shown}). A marker whose value is in no "
                "surrounding sentence guards nothing: move it beside the "
                "figure, or correct one of the two.",
            )


def check_text(files: list[Path], report: Report) -> list[Marker]:
    """Grammar, binding, self-consistency, staleness and the overclaim."""
    markers: list[Marker] = []
    for path in files:
        raw = path.read_text(encoding="utf-8", errors="replace")
        # The divergence, announced. Advisory rather than hard: an unterminated
        # opener is usually a typo in prose, and prose that fails a figure gate
        # is a gate authors route around. What it must not be is silent.
        dangling = unterminated_fence(raw)
        if dangling is not None:
            line, delimiter = dangling
            report.warn(
                path,
                line,
                f"code fence {delimiter!r} opens here and is never closed. "
                "This scanner masks nothing for an unterminated fence, so "
                "everything below it is read as prose: a figure there is a "
                "published claim and a marker there is a real marker. Close "
                f"the block with a line of at least {len(delimiter)} "
                f"{delimiter[0]!r}, or -- if the line was never meant as a "
                f"fence -- indent it by {MAX_FENCE_INDENT + 1} spaces so it "
                "cannot open one.",
            )
        text = mask_document(raw)
        if "derived:" in text:
            markers.extend(parse_markers(path, text))

    for marker in markers:
        for err in marker.errors:
            report.fail(marker.path, marker.line, f"malformed marker: {err}")
        resolved = producer_path(marker.producer)
        if resolved is None:
            report.fail(
                marker.path,
                marker.line,
                f"producer {marker.producer!r} is not inside the repository; a "
                "marker names a repo-relative path and nothing else",
            )
        elif not resolved.is_file():
            report.fail(
                marker.path,
                marker.line,
                f"producer {marker.producer!r} does not exist",
            )

    check_binding(files, markers, report)

    # Self-consistency: one producer#key, one published value, everywhere.
    by_ident: dict[str, list[Marker]] = {}
    for marker in markers:
        by_ident.setdefault(marker.ident, []).append(marker)
    for ident, group in sorted(by_ident.items()):
        values = {normalise(m.value) for m in group}
        if len(values) > 1:
            sites = ", ".join(f"{rel(m.path)}:{m.line}={m.value}" for m in group)
            for marker in group:
                report.fail(
                    marker.path,
                    marker.line,
                    f"{ident} is published with {len(values)} different values "
                    f"({sites}); one figure, one value",
                )

    # Code staleness: advisory, and only meaningful for store-backed figures.
    for marker in markers:
        if not marker.store_backed or marker.sha is None:
            continue
        current = producer_sha(marker.producer)
        if current is None or current == marker.sha:
            continue
        report.warn(
            marker.path,
            marker.line,
            f"{marker.ident} was stamped against {marker.producer} at "
            f"{marker.sha}, which is now {current}. The producer changed since "
            "this figure was measured; re-derive it against "
            f"{marker.corpus} or restamp if the change cannot move it.",
        )

    # The overclaim sentence. Hard.
    for path in files:
        text = read_scannable(path)
        if not any(p.search(_uncited(text)) for p in OVERCLAIM_RES):
            continue
        for start, block in scannable_entries(path):
            if not any(p.search(_uncited(block)) for p in OVERCLAIM_RES):
                continue
            published = {normalise(m.value) for m in parse_markers(path, block)}
            missing = [f for f in extract_figures(block) if normalise(f) not in published]
            if missing:
                shown = ", ".join(missing[:12])
                more = "" if len(missing) <= 12 else f" (+{len(missing) - 12} more)"
                report.fail(
                    path,
                    start,
                    "this entry claims every figure is re-derived, but "
                    f"{len(missing)} of them carry no marker: {shown}{more}. "
                    "Annotate them, or narrow the sentence to what the script "
                    "actually emits.",
                )
    return markers


def producer_env(cache_root: str) -> dict[str, str]:
    """The environment a producer subprocess runs under.

    `PYTHONPYCACHEPREFIX` points at a directory this run owns, so the child
    compiles every module from the source that is on disk now. Without it the
    child reads `__pycache__` beside the source, and CPython validates a cached
    entry on the source's (mtime-seconds, size) alone: an edit and a revert of
    the same size inside one mtime second leave the interpreter running the
    other version's bytecode while the working tree is clean and every hash
    matches. That is not a hypothetical here — a same-size mutation is exactly
    how the arms in `tests/test_render_cost_1526.py` are checked, and a gate
    that re-derives published figures must read the tree it is gating.

    Everything else is inherited. The `AELFRICE_` prefix is cleared by the
    producer itself (`figures()` in `benchmarks/injection_budget_bytes.py`),
    not stripped here, so a producer that stops clearing it fails this gate
    instead of being covered by it.
    """
    return {**os.environ, "PYTHONPYCACHEPREFIX": cache_root}


def check_producers(markers: list[Marker], report: Report) -> None:
    """Run every store-free producer and diff its output against the prose."""
    by_producer: dict[str, list[Marker]] = {}
    for marker in markers:
        if marker.store_backed or marker.errors:
            continue
        by_producer.setdefault(marker.producer, []).append(marker)

    with tempfile.TemporaryDirectory(prefix="derived-figures-pyc-") as pyc:
        for producer, group in sorted(by_producer.items()):
            path = producer_path(producer)
            if path is None or not path.is_file():
                continue  # already reported by check_text
            proc = subprocess.run(
                [sys.executable, str(path), "--emit-figures"],
                capture_output=True,
                text=True,
                cwd=str(REPO_ROOT),
                env=producer_env(pyc),
                timeout=300,
                check=False,
            )
            if proc.returncode != 0:
                for marker in group:
                    report.fail(
                        marker.path,
                        marker.line,
                        f"producer {producer} exited {proc.returncode} under "
                        f"--emit-figures: {proc.stderr.strip()[:400]}",
                    )
                continue
            decoded: object
            try:
                decoded = json.loads(proc.stdout)
            except json.JSONDecodeError as exc:
                for marker in group:
                    report.fail(
                        marker.path,
                        marker.line,
                        f"producer {producer} did not emit JSON on stdout ({exc})",
                    )
                continue
            if not isinstance(decoded, dict):
                for marker in group:
                    report.fail(
                        marker.path,
                        marker.line,
                        f"producer {producer} emitted {type(decoded).__name__}, "
                        "expected a JSON object of key -> value",
                    )
                continue
            emitted = cast("dict[str, object]", decoded)
            for marker in group:
                if marker.key not in emitted:
                    report.fail(
                        marker.path,
                        marker.line,
                        f"producer {producer} emits no key {marker.key!r}; it emits "
                        f"{sorted(emitted)}",
                    )
                    continue
                got = normalise(emitted[marker.key])
                want = normalise(marker.value)
                if got != want:
                    report.fail(
                        marker.path,
                        marker.line,
                        f"published {marker.ident} = {marker.value}, but "
                        f"{producer} now emits {emitted[marker.key]}. The figure is "
                        "stale: re-derive it, or fix the producer.",
                    )


def restamp(files: list[Path]) -> int:
    """Rewrite every `producer-sha=` to the producer's current hash.

    Two uses, and only two. A producer whose own docstring carries a marker
    cannot be stamped by hand — writing the hash changes the bytes the hash is
    over — so the stamp has to be applied after the edit rather than during it.
    And a producer edit that provably cannot move the figure (a docstring
    correction, a rename) leaves an advisory warning standing on every file
    quoting it, which is how a warning becomes wallpaper.

    Restamping asserts "I have re-derived this figure, or I know this edit
    cannot have moved it". It is never a way to clear a warning that has not
    been looked at, and it is deliberately not run by CI.
    """
    changed = 0
    for path in files:
        text = path.read_text(encoding="utf-8")
        if "derived:" not in text:
            continue
        out = text
        # Parsed from the masked text, rewritten into the raw text: a stamp
        # inside a fenced block is an example of the syntax, not a figure.
        for marker in parse_markers(path, mask_document(text)):
            if marker.sha is None:
                continue
            current = producer_sha(marker.producer)
            if current is None or current == marker.sha:
                continue
            out = out.replace(
                f"{SHA_ATTR}={marker.sha}", f"{SHA_ATTR}={current}"
            )
        if out != text:
            path.write_text(out, encoding="utf-8")
            changed += 1
            print(f"restamped {rel(path)}")
    print(f"{changed} file(s) restamped.")
    return 0


def unmarked_total(files: list[Path], mask_spans: Callable[[str], str]) -> int:
    """How many figures carry no marker under a given inline-code rule."""
    total = 0
    for path in files:
        for _start, block in scannable_entries(path):
            published = {normalise(m.value) for m in parse_markers(path, block)}
            total += sum(
                1
                for f in extract_figures(block, mask_spans)
                if normalise(f) not in published
            )
    return total


def mask_delta(files: list[Path]) -> int:
    """Price the three candidate inline-code rules against this tree.

    The middle row is what ships. The rule below it is what the gate did
    first, and it hid published figures; the rule above it sees every code
    identifier, which would make the hard overclaim rule unsatisfiable.

    Reporting only, and the numbers move with the corpus, which is why they
    are printed on demand rather than written into prose.
    """
    strict = unmarked_total(files, _mask_every_code_span)
    shipped = unmarked_total(files, _mask_code_spans)
    loose = unmarked_total(files, _mask_no_code_span)
    print(f"unmarked figures over {len(files)} files, by inline-code rule:")
    print(f"  mask every code span (pre-fix) : {strict}")
    print(f"  mask all but a bare number     : {shipped}  (+{shipped - strict})")
    print(f"  mask nothing                   : {loose}  (+{loose - strict})")
    return 0


def list_unmarked(files: list[Path]) -> int:
    """Enumerate figures that carry no marker. Reporting only; always exit 0."""
    total = 0
    for path in files:
        for start, block in scannable_entries(path):
            published = {normalise(m.value) for m in parse_markers(path, block)}
            missing = [f for f in extract_figures(block) if normalise(f) not in published]
            if not missing:
                continue
            total += len(missing)
            print(f"{rel(path)}:{start}: {len(missing)} unmarked: {', '.join(missing[:20])}")
    print(f"\n{total} unmarked figures across {len(files)} files.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument(
        "--mode",
        choices=("text", "producers", "all"),
        default="text",
        help="text: grammar/self-consistency/staleness/overclaim, stdlib only. "
        "producers: re-run store-free producers (needs the package). "
        "all: both.",
    )
    ap.add_argument("--list-unmarked", action="store_true")
    ap.add_argument(
        "--mask-delta",
        action="store_true",
        help="report how many figures each candidate inline-code rule makes "
        "visible on this tree. Reporting only.",
    )
    ap.add_argument(
        "--restamp",
        action="store_true",
        help="rewrite producer-sha= to each producer's current hash. Local "
        "only; asserts the figure was re-derived or cannot have moved.",
    )
    ap.add_argument(
        "--github",
        action="store_true",
        help="emit ::error/::warning workflow annotations as well as text",
    )
    ap.add_argument("paths", nargs="*", default=None)
    args = ap.parse_args(argv)

    roots = args.paths if args.paths else list(DEFAULT_ROOTS)
    files = iter_files(roots)
    if not files:
        print("no files scanned; check the paths given", file=sys.stderr)
        return 1

    if args.restamp:
        return restamp(files)
    if args.mask_delta:
        return mask_delta(files)
    if args.list_unmarked:
        return list_unmarked(files)

    report = Report(github=args.github)
    markers = check_text(files, report)
    if args.mode in ("producers", "all"):
        check_producers(markers, report)

    store_free = sum(1 for m in markers if not m.store_backed)
    print(
        f"{len(markers)} derived-figure markers across {len(files)} files "
        f"({store_free} store-free, {len(markers) - store_free} store-backed)."
    )
    for line in report.advisory:
        print(f"advisory: {line}")
    for line in report.hard:
        print(f"ERROR: {line}", file=sys.stderr)
    if report.hard:
        print(f"\n{len(report.hard)} hard failure(s).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
