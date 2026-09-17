"""The derived-figure gate decides which text is code with a scanner (#1556).

`scripts/check_derived_figures.py` excludes code before it parses a marker, so
a figure quoted in a shell transcript is not read as a published claim. It used
to do that with one regex -- ``` ``.*?``|`[^`]*` ``` -- which paired backticks
across the whole document with no notion of a block and no notion of run
length. A three-backtick line is an odd run: two backticks paired, and the
third left behind as an opener that blanked everything down to the next
backtick anywhere below it. `docs/user/PRIVACY.md` carries six delimiter lines
and parsed **zero** markers -- the count a page carrying no markers at all
reports. A marker that never parses is indistinguishable from a figure nobody
annotated, so the gate printed success over an unguarded figure.

The repair is two rules, and both are needed. A fenced-block scanner alone
leaves the odd run in *prose*: `docs/user/CONFIG.md` names fences in code font
in the middle of a table row, which no fence scanner looks at, and the spare
backticks there ate forward just the same. A run-length matcher alone leaves
the block: a marker written as an example inside a fenced block would parse as
a published figure. So `code_block_spans` masks blocks and `code_span_spans`
pairs runs by length inside one paragraph, and the tests below hold each
against the arrangement the other cannot see.

The instrument at the centre of this module is differential:
`test_no_generated_document_loses_a_marker_that_main_parsed` runs the shipped
scanner and the pre-fix one over a seeded corpus of fence and backtick
arrangements and holds the shipped one to a superset. A hand-written list of
cases is a different instrument, and the regressions this change went through
were each a case nobody thought to write.

The superset is not universal, and the exception is named rather than hidden:
where the pre-fix rule leaked a marker *out of* a real code block, this scanner
correctly hides it. `test_a_double_backtick_in_a_block_body_leaked_a_marker_on_main`
is that witness, and it is the one class of document where fewer markers parse.

Layers, and what each adds:

  * **unit** -- `code_block_spans` against CommonMark's fenced-code rules, and
    `code_span_spans` against its run-length rule. Cheapest layer that can see
    an off-by-one in a closer's length or an indent bound.
  * **integration** -- the real `check_text` over a tmp_path tree, because the
    properties that matter (a contradicted figure still fails, a marker in a
    block is still ignored, a reported line number is still the marker's line)
    are properties of the gate, not of the masker.
  * **live corpus** -- the gate over this repository, because the defect was
    found on this repository and the count is the evidence it is closed.
"""
from __future__ import annotations

import importlib.util
import os
import random
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "check_derived_figures.py"

_spec = importlib.util.spec_from_file_location("_cdf_1556", _SCRIPT)
assert _spec and _spec.loader
# Declared `Any` for the same reason `test_derived_figures_1469.py` declares it:
# pyright runs `tests/` in strict mode and an implicitly-typed module object
# turns every attribute read into an `Unknown`.
cdf: Any = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cdf)


# --- the pre-fix scanner, frozen -----------------------------------------
#
# A verbatim copy of what `main` did: no block scanner, no run length, only the
# inline-code regex, whose `` `` .*? `` `` alternative swallowed fenced blocks
# as a side effect. Frozen here rather than imported so the differential runs
# on a shallow checkout that cannot reach the baseline commit;
# `test_the_frozen_baseline_is_what_main_actually_shipped` proves the copy
# faithful whenever git can reach it.
_BASELINE_INLINE_CODE_RE = re.compile(r"``.*?``|`[^`]*`", re.S)
_BASELINE_NON_NEWLINE_RE = re.compile(r"[^\n]")

_BASELINE_REF_ENV = "AELF_DERIVED_FIGURES_BASELINE_REF"
_BASELINE_REF_DEFAULT = "f377b32ca4dd80ed750b8770625d834cee991522"


def baseline_mask(text: str) -> str:
    """`text` under the pre-fix rule, blanked in place."""
    return _BASELINE_INLINE_CODE_RE.sub(
        lambda m: _BASELINE_NON_NEWLINE_RE.sub(" ", m.group(0)), text
    )


def baseline_keys(text: str) -> set[str]:
    """Marker keys the pre-fix scanner parses out of `text`."""
    return {m.group("key") for m in cdf.MARKER_RE.finditer(baseline_mask(text))}


def branch_keys(text: str) -> set[str]:
    """Marker keys the shipped scanner parses out of `text`."""
    return {m.group("key") for m in cdf.MARKER_RE.finditer(cdf.mask_document(text))}


def _load_baseline_module(tmp_path: Path) -> Any:
    """`main`'s copy of the gate, or None when git cannot reach it.

    Written into `tmp_path` and imported from there rather than executed out of
    a string: the module derives `REPO_ROOT` from its own location, and it
    should not be able to reach the real tree.
    """
    ref = os.environ.get(_BASELINE_REF_ENV, _BASELINE_REF_DEFAULT)
    proc = subprocess.run(
        ["git", "show", f"{ref}:scripts/check_derived_figures.py"],
        capture_output=True,
        text=True,
        cwd=str(_REPO),
        timeout=60,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        return None
    source = tmp_path / "scripts" / "baseline_check_derived_figures.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(proc.stdout, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("_cdf_baseline_1556", source)
    assert spec and spec.loader
    module: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- the corpus -----------------------------------------------------------


@dataclass(frozen=True)
class Doc:
    """One generated document and what the generator put where.

    `in_code` and `outside` are the generator's own bookkeeping, filled only
    for documents built entirely out of segments whose CommonMark reading is
    not in question -- a well-formed block, or a delimiter-looking line that
    cannot open one. `ambiguous` documents carry an adversarial segment whose
    reading is the thing under test, so their marker sets are asserted against
    the pre-fix scanner rather than against an intent the generator declared.
    """

    text: str
    in_code: frozenset[str]
    outside: frozenset[str]
    ambiguous: bool
    label: str


_PRODUCER = "benchmarks/published_constants.py"
_INFO = ("", "python", "bash", "text")
_PROSE = (
    "The measured figure is {n}.",
    "A sentence with no figure in it at all.",
    "Two values, {n} and {n}, published together.",
    # An unmatched backtick: a typo, and the reason the inline pass is bounded
    # to a paragraph. Blocks are masked first and the span rule restarts after
    # each block, so a stray backtick that main consumed against a fence
    # delimiter instead reaches forward to the next backtick below it.
    "Bounded by `DEFAULT_HOOK_TOKEN_BUDGET ({n} tokens).",
    "A closed `span` and one stray ` on the same line.",
    # An odd run in prose, with no fence anywhere near it. This is the live
    # shape at `docs/user/CONFIG.md`: a table row naming fences in code font.
    # A fenced-block scanner never looks at this line, so it is the arrangement
    # that separates run-length pairing from block masking.
    "| `snapshot` | First sentence (split outside ``` fences) + `...`. |",
    # The same, unbalanced: a lone run of three with no partner run of three.
    "A row naming ``` and nothing else that length.",
)


def _marker(key: str, value: object, producer: str = _PRODUCER) -> str:
    """One marker, assembled rather than written out.

    Never spelled as a literal anywhere in this module: `tests/` is inside the
    scanned corpus, so a literal marker in a fixture is a marker the live scan
    picks up -- and it then names a producer that does not exist and publishes
    a figure no sentence carries. Two hard failures, from a test fixture.
    """
    return f"<!-- {'derived'}: {producer}#{key} = {value} -->"


class _Keys:
    """Hands out a unique marker key per plant, so sets are comparable."""

    def __init__(self) -> None:
        self.n = 0

    def next(self) -> str:
        self.n += 1
        return f"k{self.n}"


def _prose_lines(rng: random.Random, keys: _Keys) -> tuple[list[str], set[str]]:
    lines: list[str] = []
    planted: set[str] = set()
    for _ in range(rng.randint(1, 3)):
        lines.append(rng.choice(_PROSE).format(n=rng.randint(2, 999)))
    if rng.random() < 0.7:
        # Most markers get a paragraph of their own, so the corpus carries the
        # arrangement the paragraph property is about: a stray backtick in one
        # paragraph and a marker in a backtick-free one below it. A marker
        # crowded in with the prose that precedes it cannot distinguish "the
        # inline pass is bounded" from "the inline pass is off".
        if rng.random() < 0.5:
            lines.append("")
            lines.append(f"The library budget is {rng.randint(2, 999)} tokens.")
        key = keys.next()
        lines.append(_marker(key, rng.randint(2, 999)))
        planted.add(key)
    return lines, planted


def _block(rng: random.Random, keys: _Keys) -> tuple[list[str], set[str]]:
    """A well-formed fenced block: opener, body, closer of the same character."""
    # Backticks are weighted: they are what the corpus ships and what the
    # pre-fix rule could see at all, so they are where the defect lives. A
    # tilde block is invisible to the pre-fix rule, which makes it a test of
    # AC4 rather than of AC2.
    char = "`" if rng.random() < 0.7 else "~"
    open_len = rng.randint(3, 5)
    indent = " " * rng.randint(0, cdf.MAX_FENCE_INDENT)
    info = rng.choice(_INFO)
    lines = [f"{indent}{char * open_len}{info}"]
    planted: set[str] = set()
    for _ in range(rng.randint(0, 3)):
        if rng.random() < 0.5:
            key = keys.next()
            lines.append(_marker(key, rng.randint(2, 999)))
            planted.add(key)
        else:
            lines.append(f"value = {rng.randint(2, 999)}")
    close_indent = " " * rng.randint(0, cdf.MAX_FENCE_INDENT)
    lines.append(f"{close_indent}{char * rng.randint(open_len, open_len + 2)}")
    return lines, planted


def _stray_prose_delimiter(rng: random.Random) -> list[str]:
    """A delimiter-looking line that CommonMark cannot read as an opener.

    Two shapes, both live in this repository: indented past the three-space
    limit, and a backtick run whose info string carries a backtick -- which is
    `tests/test_noise_harness_and_fences_1371.py:198`, the line that made an
    early regex repair blank a file to its end.
    """
    if rng.random() < 0.5:
        return [f"{' ' * rng.randint(4, 7)}{'`' * rng.randint(3, 4)}"]
    return ["``` ```python ``` on its own line is body text in an open block"]


def _adversarial(rng: random.Random, keys: _Keys) -> tuple[list[str], set[str], str]:
    """One arrangement whose block structure is the question, always last."""
    kind = rng.choice(
        ("bare_opener", "unterminated", "marker_after_delimiter", "short_closer")
    )
    char = rng.choice(("`", "~"))
    planted: set[str] = set()

    def plant() -> str:
        key = keys.next()
        planted.add(key)
        return _marker(key, rng.randint(2, 999))

    if kind == "bare_opener":
        lines = [char * 3, "Prose under a delimiter nobody closed.", plant()]
    elif kind == "unterminated":
        lines = [f"{char * 3}python", f"value = {rng.randint(2, 999)}", plant()]
    elif kind == "marker_after_delimiter":
        lines = [
            char * 3,
            f"value = {rng.randint(2, 999)}",
            f"{char * 3} {plant()}",
        ]
    else:
        lines = [char * 4, plant(), char * 3]
    return lines, planted, kind


def corpus(seed: int, count: int) -> list[Doc]:
    """`count` documents from a seeded RNG. Deterministic for a given seed."""
    rng = random.Random(seed)
    docs: list[Doc] = []
    for i in range(count):
        keys = _Keys()
        lines: list[str] = []
        in_code: set[str] = set()
        outside: set[str] = set()
        for _ in range(rng.randint(2, 6)):
            kind = rng.random()
            if kind < 0.45:
                got, planted = _prose_lines(rng, keys)
                outside |= planted
            elif kind < 0.85:
                got, planted = _block(rng, keys)
                in_code |= planted
            else:
                got = _stray_prose_delimiter(rng)
            lines.extend(got)
            lines.append("")
        label = "plain"
        ambiguous = i % 3 == 0
        if ambiguous:
            # Adversarial segments always terminate the document. Everything
            # they can affect is then the trailing prose they carry with them,
            # which keeps the arrangement readable without the generator having
            # to re-implement the scanner to say where the block ends.
            got, planted, label = _adversarial(rng, keys)
            lines.extend(got)
        text = "\n".join(lines)
        if rng.random() < 0.5:
            text += "\n"
        if rng.random() < 0.25:
            text = text.replace("\n", "\r\n")
            label += "+crlf"
        docs.append(
            Doc(
                text=text,
                in_code=frozenset(in_code),
                outside=frozenset(outside),
                ambiguous=ambiguous,
                label=label,
            )
        )
    return docs


SEED = 1556
COUNT = 300


def test_the_seeded_corpus_is_the_corpus_this_module_was_measured_on() -> None:
    """Pins the instrument. Every count below is a count over these documents.

    Without this, a change to the generator moves the numbers the other tests
    assert and the move reads as a change in the scanner.
    """
    docs = corpus(SEED, COUNT)
    planted = sum(len(d.in_code) + len(d.outside) for d in docs)
    assert len(docs) == 300
    assert planted == 728
    assert sum(1 for d in docs if d.ambiguous) == 100
    assert sum(1 for d in docs if "crlf" in d.label) == 82
    assert corpus(SEED, COUNT)[7].text == docs[7].text, "not deterministic"


def test_the_corpus_carries_an_odd_backtick_run_in_prose_with_no_fence() -> None:
    """The arrangement a fenced-block scanner cannot see, present by count.

    `docs/user/CONFIG.md` names fences in code font inside a table row. No
    fence scanner reads that line -- it is not a delimiter line -- so without
    documents of this shape the corpus would price only half the repair.
    """
    docs = corpus(SEED, COUNT)
    unfenced_odd_run = [
        d
        for d in docs
        if cdf.code_block_spans(d.text) == []
        and any(len(m.group(0)) == 3 for m in re.finditer(r"`+", d.text))
    ]
    assert len(unfenced_odd_run) == 38, "no fence-free odd run means nothing to price"


def test_no_generated_document_loses_a_marker_that_main_parsed() -> None:
    """AC2, the regression bar. The shipped scanner parses a superset.

    The exception is enumerated rather than waived: a marker the pre-fix rule
    parsed and this one does not must be one the generator planted inside a
    well-formed fenced block, which is a marker the pre-fix rule leaked out of
    code and AC4 says to ignore. Anything else is the defect coming back.
    """
    docs = corpus(SEED, COUNT)
    unexplained: list[tuple[str, set[str]]] = []
    for doc in docs:
        lost = baseline_keys(doc.text) - branch_keys(doc.text)
        if lost - doc.in_code:
            unexplained.append((doc.label, lost - doc.in_code))
    assert unexplained == [], (
        f"{len(unexplained)} documents lose a marker main parsed: {unexplained[:4]}"
    )


def test_the_corpus_contains_the_defect_this_bar_is_set_against() -> None:
    """The distinguishing arm: a superset assertion over a corpus where the two
    scanners agree everywhere proves nothing. The gain is where main's pairing
    blanked prose, and it is counted."""
    docs = corpus(SEED, COUNT)
    gained = sum(len(branch_keys(d.text) - baseline_keys(d.text)) for d in docs)
    lost = sum(len(baseline_keys(d.text) - branch_keys(d.text)) for d in docs)
    assert gained == 227, "markers main blanked out of prose that now parse"
    assert lost == 93, "markers main leaked out of a code block that are now hidden"


def test_no_marker_planted_inside_a_fenced_block_parses() -> None:
    """AC4 -- the property the masking exists for.

    Over the unambiguous half of the corpus, where the CommonMark reading is
    not in question: a well-formed block, or a delimiter-looking line that
    cannot open one. A gate that only proved markers parse would have deleted
    this feature and called it a fix.
    """
    docs = [d for d in corpus(SEED, COUNT) if not d.ambiguous]
    assert docs, "an empty corpus would pass this vacuously"
    planted = sum(len(d.in_code) for d in docs)
    assert planted == 242, "no marker inside a block means nothing to ignore"
    for doc in docs:
        assert branch_keys(doc.text) & doc.in_code == set(), doc.text


def test_a_marker_planted_in_prose_parses_unless_a_code_span_hid_it() -> None:
    """The distinguishing arm: a masker that blanked every document would pass
    the AC4 test above and nothing else.

    The exemption is not the fence rule. A few documents in this corpus open
    an inline code span that straddles the marker, which is the citation
    convention `_uncited_inplace` has enforced since #1469 and which blanks
    them on the pre-fix scanner too. So the bar is relative: this scanner
    loses no prose marker the pre-fix one kept.
    """
    docs = [d for d in corpus(SEED, COUNT) if not d.ambiguous]
    parsed = 0
    for doc in docs:
        lost = doc.outside - branch_keys(doc.text)
        assert lost <= doc.outside - baseline_keys(doc.text), doc.text
        parsed += len(doc.outside & branch_keys(doc.text))
    assert parsed == 253, "prose markers that parse; zero would be a dead gate"


def _paragraph_bounds(text: str, offset: int) -> tuple[int, int]:
    """The blank-line-delimited paragraph holding `offset`."""
    start = 0
    for m in re.finditer(r"\n[ \t\r]*\n", text[:offset]):
        start = m.end()
    after = re.search(r"\n[ \t\r]*\n", text[offset:])
    end = len(text) if after is None else offset + after.start() + 1
    return start, end


def unblankable(text: str) -> set[tuple[int, str]]:
    """`(line, key)` of every marker the masker is obliged to leave alone.

    The obligation: the marker sits outside every fenced block, and its own
    paragraph carries no backtick. No reading of Markdown makes such a marker
    code, so no masker may blank it -- and because the bound is absolute rather
    than a comparison with an older rule, it survives the older rule being
    wrong. `docs/user/PRIVACY.md` is one of these and parsed zero markers.
    """
    blocks = cdf.code_block_spans(text)
    out: set[tuple[int, str]] = set()
    for m in cdf.MARKER_RE.finditer(text):
        at = m.start()
        if any(s <= at < e for s, e in blocks):
            continue
        start, end = _paragraph_bounds(text, at)
        if "`" in text[start:end]:
            continue
        out.add((text.count("\n", 0, at) + 1, m.group("key")))
    return out


def parsed_at(text: str) -> set[tuple[int, str]]:
    """`(line, key)` of every marker the shipped scanner parses."""
    masked = cdf.mask_document(text)
    return {
        (masked.count("\n", 0, m.start()) + 1, m.group("key"))
        for m in cdf.MARKER_RE.finditer(masked)
    }


def test_no_marker_in_a_backtick_free_paragraph_outside_a_block_is_blanked() -> None:
    """The absolute bar, stated as a property of this scanner.

    A differential against the pre-fix rule prices the change but cannot
    outlive it: once the old rule is gone, "no worse than the rule that was
    wrong" says nothing. So the bar here is absolute -- a marker outside every
    block, in a paragraph carrying no backtick, parses, at its own line -- and
    `test_no_generated_document_loses_a_marker_that_main_parsed` keeps the
    differential arm beside it.
    """
    docs = corpus(SEED, COUNT)
    obliged = sum(len(unblankable(d.text)) for d in docs)
    assert obliged == 268, "no obligations would pass this vacuously"
    violations = [
        (d.label, sorted(unblankable(d.text) - parsed_at(d.text)))
        for d in docs
        if unblankable(d.text) - parsed_at(d.text)
    ]
    assert violations == [], (
        f"{len(violations)} documents blank a marker no reading calls code: "
        f"{violations[:4]}"
    )


def baseline_parsed_at(text: str) -> set[tuple[int, str]]:
    """`(line, key)` of every marker the pre-fix scanner parsed."""
    masked = baseline_mask(text)
    return {
        (masked.count("\n", 0, m.start()) + 1, m.group("key"))
        for m in cdf.MARKER_RE.finditer(masked)
    }


def test_the_pre_fix_rule_violated_that_bar_on_this_same_corpus() -> None:
    """The distinguishing arm for the bar above: a property no rule ever broke
    is a property that proves nothing about the repair."""
    docs = corpus(SEED, COUNT)
    violated = [d for d in docs if unblankable(d.text) - baseline_parsed_at(d.text)]
    assert len(violated) == 93, "the pre-fix rule's violations of the same bar"


def test_a_stray_backtick_above_a_fence_used_to_delete_the_marker_below_it() -> None:
    """The reduced witness for the property above, and the shape that shipped.

    One mistyped backtick near a fenced block. Masking the block restarts the
    inline pass after it, so the stray reaches forward to the next backtick in
    the region -- past a marker sitting in a paragraph of its own -- and the
    gate then reports the page as carrying no marker at all, which is the count
    a page with nothing to guard reports.
    """
    marker = _marker("budget", "2,400", "benchmarks/p.py")
    text = (
        "## Token budgets\n\n"
        "```python\nDEFAULT_HOOK_TOKEN_BUDGET = 1500\n```\n\n"
        "Every turn is bounded by `DEFAULT_HOOK_TOKEN_BUDGET (1,500 tokens).\n\n"
        f"The library budget is 2,400 tokens.\n{marker}\n\n"
        "Trailing prose naming `something` in code font.\n"
    )
    assert unblankable(text) == {(10, "budget")}
    assert parsed_at(text) == {(10, "budget")}


# --- unit: run-length pairing --------------------------------------------


def spans_of(text: str) -> list[str]:
    """The substrings `code_span_spans` calls inline code, in order."""
    return [text[a:b] for a, b in cdf.code_span_spans(text)]


def test_a_run_pairs_only_with_a_run_of_the_same_length() -> None:
    """CommonMark's rule, and the half of #1556 no fence scanner reaches.

    A run of three in prose has no partner of three here, so it is literal
    text. The pre-fix rule read it as a pair plus a spare opener and blanked
    from the spare to the next backtick in the file.
    """
    text = "a ``` b ` c ` d"
    assert spans_of(text) == ["` c `"], "the three-run is literal; the ones pair"
    assert "b" in cdf.mask_document(text), "the text after the three-run survives"
    assert "b" not in baseline_mask(text), "the rule this replaces ate it"


def test_two_runs_of_three_pair_with_each_other() -> None:
    """The distinguishing arm: a rule that never paired a long run would pass
    the test above and would also stop masking legitimate code."""
    assert spans_of("a ``` b ``` c") == ["``` b ```"]


def test_a_longer_run_between_two_shorter_ones_does_not_close_them() -> None:
    """Run length, not "the next backtick". The five-run is a different span
    delimiter, and stepping over it is what keeps the outer pair intact."""
    assert spans_of("`a ````` b`") == ["`a ````` b`"]


def test_a_span_may_contain_a_shorter_run() -> None:
    """The reason the pre-fix rule tried double backticks first: a citation of
    this repo's own syntax has to be able to contain a backtick."""
    assert spans_of("``a `b` c``") == ["``a `b` c``"]


def test_scanning_resumes_after_a_closer_not_inside_it() -> None:
    """Two spans, not one span and a swallowed middle."""
    assert spans_of("`a` and `b`") == ["`a`", "`b`"]


def test_an_unpaired_run_is_literal_and_the_walk_steps_past_it() -> None:
    """The dangling-opener repair, stated directly: an unpaired run must not
    consume the text after it, and must not stop the runs after it pairing."""
    assert spans_of("`` a ` b ` c") == ["` b `"]


def test_a_span_may_straddle_a_line_but_not_a_blank_line() -> None:
    """#1469 needs a span to straddle lines: `src/aelfrice/hook.py` carries
    wrapped citations, and a rule held to one line reported their markers two
    lines early. So the bound is the paragraph, not the line."""
    assert spans_of("a `wraps\nonto the next line` b") == ["`wraps\nonto the next line`"]
    assert spans_of("a `never closes\n\nb ` c") == []


def test_the_paragraph_bound_holds_across_a_crlf_blank_line() -> None:
    """A CRLF document's blank line is `\\n\\r\\n`. Reading the carriage return
    as content makes it a non-blank line, and the bound stops existing on every
    file a Windows editor touched."""
    assert spans_of("a `never closes\r\n\r\nb ` c") == []


def _keys_under(text: str, bounded: bool) -> set[str]:
    """Marker keys `mask_document` leaves visible, with the bound on or off.

    The off arm treats the whole document as one paragraph, which is the only
    difference between the two rules; run-length pairing stays on in both, so
    what this measures is the bound and nothing else.
    """
    saved = cdf.paragraph_regions
    if not bounded:
        cdf.paragraph_regions = lambda t: [(0, len(t))]
    try:
        masked = cdf.mask_document(text)
    finally:
        cdf.paragraph_regions = saved
    return {m.group("key") for m in cdf.MARKER_RE.finditer(masked)}


def test_the_paragraph_bound_costs_a_marker_only_where_backticks_surround_it() -> None:
    """The residual the bound buys its gain with, pinned rather than described.

    Bounding a span shifts where one can start. When a run can no longer reach
    across a blank line, a run in the next paragraph becomes an opener and
    blanks a region the unbounded rule left alone.

    The cost is bounded by the property, and that is the whole claim: the
    marker it loses sits in a paragraph carrying backticks, so it was never
    unblankable. A marker in a backtick-free paragraph cannot be reached by
    this shift, which is what makes the residual a class and not a hole.
    """
    marker = _marker("shifted", 42, "benchmarks/p.py")
    text = f"An opener `here\n\nand a closer` then {marker} then `a span`.\n"
    assert _keys_under(text, bounded=False) == {"shifted"}
    assert _keys_under(text, bounded=True) == set()
    assert unblankable(text) == set(), "the bar never covered this marker"


def test_the_paragraph_bound_gains_far_more_than_it_shifts_away() -> None:
    """The distinguishing arm: a bound that only ever lost markers would pass
    the test above. Over the pinned corpus it loses none and gains many."""
    docs = corpus(SEED, COUNT)
    gained = lost = 0
    for doc in docs:
        old = _keys_under(doc.text, bounded=False)
        new = _keys_under(doc.text, bounded=True)
        gained += len(new - old)
        lost += len(old - new)
    assert gained == 81, "markers the unbounded rule blanked out of prose"
    assert lost == 0, "the shift does not reach this corpus"


def test_a_double_backtick_in_a_block_body_leaked_a_marker_on_main() -> None:
    """The one class of text that parses fewer markers than main parsed.

    main's pairing ended its span on the first `` `` `` inside the body, which
    left the rest of the block -- marker included -- unmasked. That marker is
    inside a fenced code block, so AC4 says to ignore it and AC2's superset
    cannot also hold. Naming the case is the point: it is the whole exception.
    """
    text = f"```\n`` {_marker('cap', 20, 'benchmarks/p.py')}\n```\n"
    assert baseline_keys(text) == {"cap"}
    assert branch_keys(text) == set()


@pytest.mark.timeout(60)
def test_the_frozen_baseline_is_what_main_actually_shipped(tmp_path: Path) -> None:
    """The frozen copy above must be the rule the baseline commit ran.

    Skips rather than degrades when git cannot reach the commit -- a shallow
    CI checkout is the normal case. Set AELF_DERIVED_FIGURES_BASELINE_REF to a
    reachable pre-#1556 revision to run it there.
    """
    baseline = _load_baseline_module(tmp_path)
    if baseline is None:
        pytest.skip(
            "baseline revision unreachable; set "
            f"{_BASELINE_REF_ENV} to a reachable pre-#1556 revision"
        )
    assert not hasattr(baseline, "code_block_spans"), (
        f"{_BASELINE_REF_ENV} names a revision that already carries the #1556 "
        "scanner, so the differential would compare the fix against itself"
    )
    inline = baseline._INLINE_CODE_RE  # noqa: SLF001
    assert inline.pattern == _BASELINE_INLINE_CODE_RE.pattern
    assert inline.flags == _BASELINE_INLINE_CODE_RE.flags
    assert baseline.MARKER_RE.pattern == cdf.MARKER_RE.pattern, (
        "marker syntax moved; the frozen baseline no longer parses what main did"
    )


# --- unit: the block scanner ---------------------------------------------


def masked(text: str) -> str:
    """`text` with fenced blocks blanked and nothing else touched."""
    return cdf.mask_code_blocks(text)


def in_code(text: str, needle: str) -> bool:
    """Is `needle` inside a span the scanner called code?"""
    assert needle in text, "the fixture does not contain the needle"
    return needle not in masked(text)


@pytest.mark.parametrize("length", [3, 4, 7])
@pytest.mark.parametrize("char", ["`", "~"])
def test_a_run_of_three_or_more_of_either_character_opens_a_block(
    char: str, length: int
) -> None:
    """CommonMark's opener. Tildes matter: the pre-fix rule saw backticks only,
    so a tilde block hid nothing and a marker inside one was read as a claim."""
    fence = char * length
    assert in_code(f"{fence}\nSECRET = 1\n{fence}\n", "SECRET")


@pytest.mark.parametrize("char", ["`", "~"])
def test_two_of_the_character_is_not_a_fence(char: str) -> None:
    """The distinguishing arm for the length bound: at two it is inline code or
    prose, and blanking a document from there is the #1556 defect."""
    assert not in_code(f"{char * 2}\nvalue = 1\n{char * 2}\n", "value")


def test_a_closer_shorter_than_its_opener_does_not_close() -> None:
    """A four-backtick block is how an author writes a block that contains a
    three-backtick block. Accepting the short run as a closer ends the block
    early and leaks the rest of it."""
    text = "````\n```\nSECRET = 1\n```\n````\nprose\n"
    assert in_code(text, "SECRET")
    assert not in_code(text, "prose")


def test_a_closer_longer_than_its_opener_closes() -> None:
    text = "```\nSECRET = 1\n`````\nprose\n"
    assert in_code(text, "SECRET")
    assert not in_code(text, "prose")


def test_a_closer_of_the_other_character_does_not_close() -> None:
    """`~~~` cannot end a backtick block, so everything up to the real closer is
    body text."""
    text = "```\n~~~\nSECRET = 1\n```\nprose\n"
    assert in_code(text, "SECRET")
    assert not in_code(text, "prose")


def test_an_info_string_is_allowed_on_the_opener() -> None:
    assert in_code("```python\nSECRET = 1\n```\n", "SECRET")


def test_an_info_string_on_a_closing_line_makes_it_body_text() -> None:
    """CommonMark: a closer carries nothing but whitespace. Accepting an info
    string there ends the block at the wrong line and leaks the remainder --
    the same reading `test_noise_harness_and_fences_1371.py` pins for the
    ingest fence stripper."""
    text = "```\nfirst = 1\n```python\nSECRET = 1\n```\nprose\n"
    assert in_code(text, "SECRET")
    assert not in_code(text, "prose")


def test_a_backtick_in_a_backtick_openers_info_string_disqualifies_it() -> None:
    """The live shape: `tests/test_noise_harness_and_fences_1371.py:198` is
    prose whose first characters are a backtick run. Reading it as an opener is
    what blanked a whole file to its end in an early regex repair.

    The trailing delimiter is what makes this distinguishing. Without it the
    mistaken block would be unterminated, and an unterminated block masks
    nothing here, so dropping the rule would look identical.
    """
    text = "``` ```python ``` on its own line is body text\nprose 12\n```\n"
    assert not in_code(text, "prose 12")


def test_a_tilde_openers_info_string_may_contain_tildes() -> None:
    """The no-backtick rule is backtick-only in CommonMark, and widening it to
    both characters would make a legitimate tilde block invisible."""
    assert in_code("~~~a~b\nSECRET = 1\n~~~\n", "SECRET")


@pytest.mark.parametrize("indent", [0, 1, 2, 3])
def test_a_fence_indented_up_to_three_spaces_opens_a_block(indent: int) -> None:
    text = f"{' ' * indent}```\nSECRET = 1\n{' ' * indent}```\n"
    assert in_code(text, "SECRET")


def test_a_fence_indented_four_spaces_does_not_open_a_block() -> None:
    """The fourth space is an indented code block, a construct this scanner
    deliberately does not mask -- four-space indentation is the body of every
    Python function in the scanned corpus."""
    assert not in_code("    ```\nvalue = 1\n    ```\n", "value")


def test_a_tab_indented_fence_does_not_open_a_block() -> None:
    """A tab is four columns. Counting it as one character would let a
    tab-indented delimiter under the three-space bound."""
    assert not in_code("\t```\nvalue = 1\n\t```\n", "value")


def test_an_unterminated_block_masks_nothing() -> None:
    """The declared divergence from CommonMark, and the regression bar.

    CommonMark runs an unterminated block to the end of its container. Doing
    that here blanks every marker below a lone delimiter line in prose, which
    is the defect this scanner replaces, relocated.
    """
    text = f"```\nvalue = 1\n{_marker('cap', 20, 'benchmarks/p.py')}\n"
    assert masked(text) == text
    assert branch_keys(text) == {"cap"}


def test_a_fence_on_the_last_line_with_no_trailing_newline_closes_its_block() -> None:
    """The closing delimiter itself is part of the block, final line or not.

    Asserting that no backtick survives, rather than only that the body is
    hidden: a span that stopped at the start of the closing line still hides
    the body, and leaves a bare delimiter behind for the inline rule to pair
    with something far away.
    """
    text = "```\nSECRET = 1\n```"
    out = masked(text)
    assert in_code(text, "SECRET")
    assert "`" not in out, "the closer on the final line is part of the block"
    assert len(out) == len(text)


def test_crlf_line_endings_do_not_stop_a_closer_closing() -> None:
    """The carriage return is trailing whitespace on the delimiter line.
    Reading it as content makes every closer in a CRLF document carry an info
    string, so every block runs unterminated."""
    text = "```\r\nSECRET = 1\r\n```\r\nprose 12\r\n"
    assert in_code(text, "SECRET")
    assert not in_code(text, "prose 12")


def test_adjacent_blocks_are_two_blocks_and_the_prose_between_them_survives() -> None:
    text = "```\nA = 1\n```\nkeep me\n```\nB = 2\n```\n"
    assert len(cdf.code_block_spans(text)) == 2
    assert not in_code(text, "keep me")


def test_a_marker_sharing_a_line_with_a_delimiter_still_parses() -> None:
    """A closing regex that swallowed the rest of the line would take the
    marker with it. Here the line is not a closer at all -- it carries content
    -- so the block is unterminated and masks nothing."""
    text = f"```\nvalue = 1\n``` {_marker('cap', 20, 'benchmarks/p.py')}\n"
    assert branch_keys(text) == {"cap"}


def test_masking_preserves_length_and_every_newline() -> None:
    """The contract every reported line number rests on. A masker that ate a
    newline moves an `::error` annotation off the line it describes."""
    text = "a\n```py\nx = 1\ny = 2\n```\nb\n"
    out = masked(text)
    assert len(out) == len(text)
    assert [i for i, c in enumerate(out) if c == "\n"] == [
        i for i, c in enumerate(text) if c == "\n"
    ]
    assert out.splitlines()[2].strip() == ""


def test_an_over_indented_closing_line_is_body_text_not_a_closer() -> None:
    """The indent bound applies to the closer as well as the opener.

    Without it a four-space-indented delimiter ends the block, and the prose
    that the real closer was still holding inside it falls out into the
    document -- with the real closer then reading as a fresh opener.
    """
    text = "```\nSECRET = 1\n    ```\nprose 12\n```\n"
    assert in_code(text, "SECRET")
    assert in_code(text, "prose 12")


# --- integration: the gate over a real tree -------------------------------


@pytest.fixture()
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway tree the scanner treats as the repo root.

    Writes only under tmp_path; `REPO_ROOT` is redirected so the producer path
    check resolves inside it.
    """
    monkeypatch.setattr(cdf, "REPO_ROOT", tmp_path)
    (tmp_path / "benchmarks").mkdir()
    (tmp_path / "benchmarks" / "p.py").write_text("# a producer\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    return tmp_path


def _check(repo: Path, name: str) -> Any:
    report = cdf.Report(github=False)
    cdf.check_text([repo / name], report)
    return report


_PAGE = (
    "# A page\n"
    "\n"
    "To confirm, run this:\n"
    "\n"
    "```bash\n"
    "grep -rE 'socket' src/\n"
    "```\n"
    "\n"
    "And this:\n"
    "\n"
    "```bash\n"
    "grep -rn 'pathlib' src/\n"
    "```\n"
    "\n"
    "- **A budget for each query.** The default is {value} tokens.\n"
    "  " + _marker("budget", "{marker}", "benchmarks/p.py") + "\n"
)


def test_a_marked_figure_below_two_code_blocks_is_checked_not_ignored(
    repo: Path,
) -> None:
    """The #1556 defect, at the gate: the marker sits below the delimiter lines
    of two code blocks, exactly where `docs/user/PRIVACY.md` carries its two."""
    page = repo / "docs" / "page.md"
    page.write_text(_PAGE.format(value="1,500", marker="1,500"), encoding="utf-8")
    markers = cdf.check_text([page], cdf.Report(github=False))
    assert [m.key for m in markers] == ["budget"]
    assert _check(repo, "docs/page.md").hard == []


def test_a_marked_figure_its_prose_contradicts_still_hard_fails(repo: Path) -> None:
    """The property the gate exists for. If a marker below a code block only
    *parsed*, and nothing then checked it, the fix would be cosmetic."""
    page = repo / "docs" / "page.md"
    page.write_text(_PAGE.format(value="1,500", marker="9,999"), encoding="utf-8")
    report = _check(repo, "docs/page.md")
    assert len(report.hard) == 1, report.hard
    assert "no figure reading 9,999 appears" in report.hard[0]


@pytest.mark.parametrize("fence", ["```", "~~~"])
def test_a_marker_inside_a_real_fenced_block_is_still_ignored(
    repo: Path, fence: str
) -> None:
    """AC4. A page that shows an example of the syntax publishes nothing, and a
    test that only proved markers parse would have deleted this.

    The tilde arm is the one that proves the block scanner is doing it: the
    pre-fix inline-code rule hid a backtick block by accident and a tilde block
    not at all, so only the tilde arm separates the scanner from its absence.
    """
    page = repo / "docs" / "page.md"
    page.write_text(
        "# A page\n"
        "\n"
        "A marker is written like this:\n"
        "\n"
        f"{fence}\n" + _marker("budget", "1,500", "benchmarks/p.py") + "\n"
        f"{fence}\n"
        "\n"
        "The default is 2,400 tokens.\n",
        encoding="utf-8",
    )
    report = cdf.Report(github=False)
    assert cdf.check_text([page], report) == []
    assert report.hard == []


def test_the_reported_line_is_the_markers_own_line_not_an_offset_one(
    repo: Path,
) -> None:
    """AC5, asserted on the line number rather than on the count.

    Blanking is length-preserving, so an offset still names the right line. The
    failure this guards was live once: eating a span's newlines reported every
    marker after it two lines early.
    """
    page = repo / "docs" / "page.md"
    page.write_text(_PAGE.format(value="1,500", marker="9,999"), encoding="utf-8")
    markers = cdf.check_text([page], cdf.Report(github=False))
    assert [m.line for m in markers] == [16]
    assert (
        page.read_text(encoding="utf-8")
        .splitlines()[15]
        .strip()
        .startswith("<!-- derived:")
    )
    assert _check(repo, "docs/page.md").hard[0].startswith("docs/page.md:16:")


# `str.splitlines()` breaks on all of these; a newline is the only one this
# gate counts as a line, and the only one an editor, a diff or a GitHub
# annotation counts either. A lone CR is absent on purpose: `Path.read_text`
# translates it to a newline before any of this code sees it, so it *is* a line
# break by the time the gate runs. Every other character here survives the
# read.
#
# Built with `chr` rather than written out. `tests/` is inside the scanned
# corpus, and a literal U+2028 in this file would make this module the only
# file in the repository whose `splitlines()` and `split_lines` disagree -- a
# fixture rigging the invariant it is here to test.
_SPLITLINES_ONLY = [
    pytest.param(chr(0x0B), id="vertical-tab"),
    pytest.param(chr(0x0C), id="form-feed"),
    pytest.param(chr(0x1C), id="file-separator"),
    pytest.param(chr(0x1D), id="group-separator"),
    pytest.param(chr(0x1E), id="record-separator"),
    pytest.param(chr(0x85), id="next-line"),
    pytest.param(chr(0x2028), id="line-separator"),
    pytest.param(chr(0x2029), id="paragraph-separator"),
]


def _page_with(separator: str, marker_value: str) -> str:
    """A correct page carrying `separator` inside its fenced block.

    The marker lands on line 8 in every arm, because none of these characters
    is a line break to anything that reports a line.
    """
    return (
        "# A page\n"
        "\n"
        "```text\n"
        f"alpha{separator}beta\n"
        "```\n"
        "\n"
        "The library budget is 2,400 tokens.\n"
        + _marker("budget", marker_value, "benchmarks/p.py")
        + "\n"
    )


@pytest.mark.parametrize("separator", _SPLITLINES_ONLY)
def test_a_splitlines_only_character_in_a_block_does_not_fail_a_correct_page(
    repo: Path, separator: str
) -> None:
    """A hard failure invented by the gate, on a page with nothing wrong.

    `scannable_entries` indexes a masked line list with line numbers taken from
    the raw one. Masking replaces a block's body with spaces, so one of these
    characters inside a block made a `splitlines()` list one line shorter,
    every entry after it slid, and the marker's entry no longer held the
    sentence publishing its figure. `split_lines` is the repair: one splitter,
    the newline.
    """
    page = repo / "docs" / "page.md"
    page.write_text(_page_with(separator, "2,400"), encoding="utf-8")
    assert _check(repo, "docs/page.md").hard == []


@pytest.mark.parametrize("separator", _SPLITLINES_ONLY)
def test_a_splitlines_only_character_does_not_move_the_reported_line(
    repo: Path, separator: str
) -> None:
    """The distinguishing arm: silence is not the property, the line is.

    A masker that blanked the whole document would pass the test above and
    report nothing at all. This one contradicts the marker on purpose, so the
    gate must fail -- at line 8, the marker's own line.
    """
    page = repo / "docs" / "page.md"
    page.write_text(_page_with(separator, "9,999"), encoding="utf-8")
    markers = cdf.check_text([page], cdf.Report(github=False))
    assert [m.line for m in markers] == [8]
    hard = _check(repo, "docs/page.md").hard
    assert len(hard) == 1, hard
    assert hard[0].startswith("docs/page.md:8:"), hard[0]
    assert "no figure reading 9,999 appears" in hard[0]


def test_split_lines_counts_newlines_and_nothing_else() -> None:
    """The unit under the two integration arms above.

    Cheapest layer that can catch the splitter drifting back; the integration
    arms add that the drift reaches a reported line number.
    """
    odd = "".join(chr(c) for c in (0x0B, 0x0C, 0x85, 0x2028, 0x2029))
    assert cdf.split_lines(f"a{odd}b\nc") == [f"a{odd}b", "c"]
    assert len(f"a{odd}b\nc".splitlines()) == 7, "the splitter this replaces"
    assert cdf.split_lines("a\nb\n") == ["a", "b"]
    assert cdf.split_lines("a\nb") == ["a", "b"]
    assert cdf.split_lines("") == []


def test_both_entry_line_lists_agree_on_a_block_holding_a_splitlines_char(
    repo: Path,
) -> None:
    """The invariant `scannable_entries` rests on, asserted directly.

    It slices the masked line list with numbers from the raw one, so the two
    must be the same length. Under `splitlines()` they were not, and nothing
    said so -- the slice just returned the wrong lines.
    """
    page = repo / "docs" / "page.md"
    raw = _page_with(chr(0x0B), "2,400")
    page.write_text(raw, encoding="utf-8")
    text = page.read_text(encoding="utf-8")
    assert len(cdf.split_lines(cdf.mask_code_blocks(text))) == len(
        cdf.split_lines(text)
    )
    entries = dict(cdf.scannable_entries(page))
    assert "The library budget is 2,400 tokens." in entries[7]


def test_a_block_split_across_two_entries_is_still_one_block(repo: Path) -> None:
    """Why masking is a whole-document decision.

    A fenced block with a blank line in it lands in two `split_entries`
    entries. An entry-level scanner sees an opener with no closer in one and a
    closer with no opener in the other, and has to guess; the figures inside
    the block then read as published claims.
    """
    page = repo / "docs" / "page.md"
    page.write_text(
        "- **Entry.** The cap is 20.\n"
        "  " + _marker("cap", 20, "benchmarks/p.py") + "\n"
        "\n"
        "```\n"
        "first = 111\n"
        "\n"
        "second = 222\n"
        "```\n",
        encoding="utf-8",
    )
    figures: list[str] = []
    for _start, block in cdf.scannable_entries(page):
        figures.extend(cdf.extract_figures(block))
    assert "111" not in figures and "222" not in figures
    assert "20" in figures


def test_masking_does_not_redraw_the_entry_boundaries(repo: Path) -> None:
    """A blanked block is a run of whitespace-only lines, and `split_entries`
    ends a paragraph entry at a blank line. Splitting the masked text would cut
    this page in two between the figure and the marker that names it, and the
    binding check would then fail on a page that is correct.
    """
    page = repo / "docs" / "page.md"
    page.write_text(
        "The cap is 20.\n"
        "```\n"
        "x = 1\n"
        "```\n" + _marker("cap", 20, "benchmarks/p.py") + "\n",
        encoding="utf-8",
    )
    starts = [start for start, _ in cdf.scannable_entries(page)]
    assert starts == [1], "the page is one entry; masking must not split it"
    assert _check(repo, "docs/page.md").hard == []


# --- the unterminated-fence advisory, checked against its own advice -------


_UNTERMINATED = (
    "# A page\n"
    "\n"
    "```python\n"
    "value = 1\n"
    "\n"
    "The library budget is 2,400 tokens.\n"
    + _marker("budget", "2,400", "benchmarks/p.py")
    + "\n"
)


def _advisories(repo: Path, body: str) -> list[str]:
    page = repo / "docs" / "page.md"
    page.write_text(body, encoding="utf-8")
    report = cdf.Report(github=False)
    cdf.check_text([page], report)
    assert report.hard == [], report.hard
    return report.advisory


def test_an_unterminated_fence_raises_an_advisory_naming_its_line(
    repo: Path,
) -> None:
    """The divergence announced. Advisory, not hard: an unterminated opener is
    usually a typo in prose, and prose that fails a figure gate is a gate
    authors route around. What it must not be is silent."""
    advisory = _advisories(repo, _UNTERMINATED)
    assert len(advisory) == 1, advisory
    assert advisory[0].startswith("docs/page.md:3:"), advisory[0]
    assert "never closed" in advisory[0]


def test_the_advisory_is_silent_on_a_document_that_closes_its_fences(
    repo: Path,
) -> None:
    """The distinguishing arm: an advisory that fired on every document would
    pass the test above and would be wallpaper by the second file."""
    assert _advisories(repo, _PAGE.format(value="1,500", marker="1,500")) == []


@pytest.mark.parametrize(
    ("remedy", "fixed"),
    [
        # "Close the block with a line of at least 3 '`'."
        ("close", _UNTERMINATED.replace("value = 1\n", "value = 1\n```\n")),
        # "Or indent it by four spaces so it cannot open one."
        ("indent", _UNTERMINATED.replace("```python\n", "    ```python\n")),
    ],
)
def test_each_remedy_the_advisory_names_clears_it_and_keeps_the_marker(
    repo: Path, remedy: str, fixed: str
) -> None:
    """AC6. An advisory is only worth printing if following it works.

    Both remedies are applied verbatim as the message words them, and both are
    held to two things: the advisory goes away, and the marker below still
    parses. A remedy that silenced the warning by hiding the figure would be
    worse than the warning.
    """
    assert _advisories(repo, fixed) == [], remedy
    markers = cdf.check_text([repo / "docs" / "page.md"], cdf.Report(github=False))
    assert [m.key for m in markers] == ["budget"], remedy


def test_no_file_in_the_scanned_corpus_exercises_the_divergence() -> None:
    """The claim the module docstring makes, asserted rather than written.

    While this holds, the unterminated-opener divergence from CommonMark costs
    nothing on this tree: no document is read one way by this scanner and
    another way by a Markdown renderer.
    """
    files = cdf.iter_files(list(cdf.DEFAULT_ROOTS))
    assert len(files) > 100, "an empty scan would pass this vacuously"
    dangling = [
        (cdf.rel(p), cdf.unterminated_fence(p.read_text(encoding="utf-8", errors="replace")))
        for p in files
    ]
    assert [d for d in dangling if d[1] is not None] == []


# --- the live corpus ------------------------------------------------------


@pytest.mark.timeout(120)
def test_the_privacy_page_publishes_the_two_budgets_it_names() -> None:
    """AC3, over the file the defect was found on.

    `docs/user/PRIVACY.md` carries six delimiter lines above its budget bullet,
    and every marker on the page parsed as nothing. Asserting the two keys
    rather than a count: a count rises for any reason, including a marker added
    somewhere else entirely.
    """
    page = _REPO / "docs" / "user" / "PRIVACY.md"
    text = page.read_text(encoding="utf-8")
    assert text.count("```") == 6, "the delimiter lines the defect needed are gone"
    assert baseline_keys(text) == set(), (
        "the pre-fix scanner should still see nothing here; if it does, this "
        "page no longer reproduces the defect and the test below proves less"
    )
    report = cdf.Report(github=False)
    markers = cdf.check_text([page], report)
    assert {m.key for m in markers} == {"hook_token_budget", "retrieval_token_budget"}
    assert report.hard == [], "\n".join(report.hard)


@pytest.mark.timeout(300)
def test_the_whole_repository_still_passes_the_text_checks() -> None:
    """The gate over its own corpus: more markers than before, and still green.

    The count is a floor rather than an equality -- every merge adds markers --
    but it is above what the pre-fix scanner reached on this same tree, which
    is the claim being made.
    """
    files = cdf.iter_files(list(cdf.DEFAULT_ROOTS))
    report = cdf.Report(github=False)
    markers = cdf.check_text(files, report)
    assert len(markers) >= 199, "the two PRIVACY.md markers are part of this floor"
    by_file = {cdf.rel(m.path) for m in markers}
    assert "docs/user/PRIVACY.md" in by_file
    assert report.hard == [], "\n".join(report.hard)


@pytest.mark.timeout(300)
def test_the_live_corpus_is_where_the_pre_fix_rule_went_blind() -> None:
    """The gain, measured on the tree the defect was reported against.

    A share rather than a list: the files move on every merge. The denominator
    is every non-blank line that sits outside every fenced block -- prose, by
    the reading both rules agree on -- and the numerator is the lines the
    pre-fix rule blanked wholly, so the gate could not read a marker or a
    figure on them at all.
    """
    prose = blind = 0
    for path in cdf.iter_files(list(cdf.DEFAULT_ROOTS)):
        if path.suffix != ".md":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        blocks = cdf.code_block_spans(text)
        offset = 0
        for line in cdf.split_lines(text):
            at = offset
            offset += len(line) + 1
            if not line.strip() or any(s <= at < e for s, e in blocks):
                continue
            prose += 1
            if not baseline_mask(text)[at : at + len(line)].strip():
                blind += 1
    assert prose > 10_000, "an empty denominator would pass this vacuously"
    assert blind / prose > 0.25, (
        f"{blind}/{prose} markdown prose lines were wholly blanked by the "
        "pre-fix rule; if this share has collapsed the corpus no longer "
        "reproduces what #1556 was filed for"
    )
