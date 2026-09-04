"""Tests for the documentation fact-preservation checker (#1509).

The checker exists to make a style conversion reviewable. Its whole value is
that it fails when a rewrite loses a fact, so a test suite that only shows it
passing on good input proves nothing: a checker that returned 0 unconditionally
would pass that suite.

So the shape here is a mutation suite. `_REWRITTEN` is a realistic ASD-STE100
conversion of `_ORIGINAL` -- every sentence differs, no fact does -- and it must
pass. Each mutation then takes that same passing rewrite and removes exactly one
fact, and each must fail. A mutation that still passes is a hole in the checker.

The known limitation is pinned too, in
`test_a_swapped_pair_of_numbers_is_not_detected`. Numbers compare as a multiset,
so exchanging two figures between sentences survives the check. That test exists
so the gap is recorded rather than discovered later by someone who trusted the
checker further than it goes.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from check_doc_preservation import compare, fences, inline_code  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_doc_preservation.py"

_ORIGINAL = """---
title: Retrieval
slug: retrieval
---

# Retrieval

Under the hood, the retriever just works: it grabs the 12 highest-scoring
beliefs and, while doing so, quietly drops anything below the floor -- see
[the ranking notes](https://example.invalid/ranking) for the gory details.

## How it scores

Scoring runs BM25 over 3 fields, which is roughly 4.5x faster than the old
path.

| field | weight |
| --- | --- |
| body | 1.0 |
| anchor | 2.0 |

Run it like so:

```bash
aelf retrieve --limit 12
```

The `--limit` flag caps the result set. Jump to [scoring](#how-it-scores).
"""

# A realistic conversion: every sentence is rewritten, the heading is reworded
# to remove the colloquial "How it scores", and the in-document link that
# pointed at it is updated in the same edit. No fact moves.
_REWRITTEN = """---
title: Retrieval
slug: retrieval
---

# Retrieval

The retriever selects the 12 beliefs with the highest scores. The retriever
then removes each belief with a score below the floor. For the details of the
ranking, refer to [the ranking notes](https://example.invalid/ranking).

## The scoring method

The retriever scores each belief with BM25 over 3 fields. This method is
approximately 4.5x faster than the previous method.

| field | weight |
| --- | --- |
| body | 1.0 |
| anchor | 2.0 |

To run the retriever, use this command:

```bash
aelf retrieve --limit 12
```

The `--limit` flag sets the maximum size of the result set. Refer to
[the scoring method](#the-scoring-method).
"""


def _prose_lines(text: str) -> set[str]:
    """Body lines only: no front matter, headings, tables or code.

    Those parts are identical between the two fixtures by design, so a naive
    comparison finds them "shared" and the guard never fires.
    """
    body = text.split("---\n", 2)[-1]
    out = set()
    in_code = False
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code or not stripped:
            continue
        if stripped.startswith(("#", "|")):
            continue
        out.add(stripped)
    return out


def _losses(before: str, after: str) -> list[str]:
    losses, _ = compare(before, after)
    return losses


def test_a_faithful_ste_conversion_passes() -> None:
    """The anti-vacuity case.

    Without this, every mutation test below would also be satisfied by a
    checker that rejected all input.
    """
    assert _losses(_ORIGINAL, _REWRITTEN) == []


def test_the_conversion_under_test_really_did_rewrite_the_prose() -> None:
    """Guard the guard: if `_REWRITTEN` drifted toward `_ORIGINAL`, the suite
    would keep passing while testing nothing interesting."""
    shared = _prose_lines(_ORIGINAL) & _prose_lines(_REWRITTEN)
    assert not shared, (
        f"the fixture rewrite shares prose with the original, so it is no "
        f"longer a realistic conversion: {sorted(shared)}"
    )
    # And the conversion must actually have done the STE job it claims to.
    for idiom in ("Under the hood", "just works", "gory details", "like so"):
        assert idiom in _ORIGINAL
        assert idiom not in _REWRITTEN


@pytest.mark.parametrize(
    ("name", "old", "new", "expected"),
    [
        ("a dropped figure", "4.5x faster", "much faster", "number"),
        ("a rounded figure", "4.5x faster", "5x faster", "number"),
        ("a dropped citation", "https://example.invalid/ranking", "#", "url"),
        ("a reworded flag name", "`--limit` flag sets", "`--max` flag sets", "inline code"),
        ("an edited code block", "aelf retrieve --limit 12", "aelf retrieve", "code block"),
        ("a dropped table row", "| anchor | 2.0 |\n", "", "table rows"),
        ("a renamed title", "# Retrieval\n", "# Getting beliefs back\n", "title changed"),
    ],
)
def test_removing_one_fact_fails_the_check(
    name: str, old: str, new: str, expected: str
) -> None:
    mutated = _REWRITTEN.replace(old, new, 1)
    assert mutated != _REWRITTEN, f"mutation {name!r} did not apply to the fixture"
    losses = _losses(_ORIGINAL, mutated)
    assert losses, f"mutation {name!r} was not detected"
    assert any(expected in line for line in losses), (
        f"mutation {name!r} was detected, but not as a {expected!r} loss: {losses}"
    )


def test_a_dropped_section_fails_the_check() -> None:
    mutated = _REWRITTEN.replace("## The scoring method\n", "")
    assert any("section count dropped" in line for line in _losses(_ORIGINAL, mutated))


def test_a_dropped_front_matter_key_fails_the_check() -> None:
    mutated = _REWRITTEN.replace("slug: retrieval\n", "")
    assert any("front-matter key" in line for line in _losses(_ORIGINAL, mutated))


def test_rewording_a_heading_without_fixing_its_link_fails_the_check() -> None:
    """The one way a reworded heading does lose something.

    Rewording is required by the style, so it cannot be reported on its own.
    Orphaning a link that pointed at the old anchor is a real loss, and it is
    invisible in a rendered page until somebody clicks it.
    """
    mutated = _REWRITTEN.replace("## The scoring method", "## The method used for scoring")
    losses = _losses(_ORIGINAL, mutated)
    assert any("points at no heading" in line for line in losses), losses


def test_rewording_a_heading_and_its_link_together_passes() -> None:
    mutated = _REWRITTEN.replace("The scoring method", "The method used for scoring")
    mutated = mutated.replace("#the-scoring-method", "#the-method-used-for-scoring")
    assert _losses(_ORIGINAL, mutated) == []


def test_adding_a_subheading_passes() -> None:
    """Breaking a long section into named parts is a conversion move, not a
    defect. The check is one-directional on purpose."""
    mutated = _REWRITTEN.replace(
        "| field | weight |", "### The field weights\n\n| field | weight |"
    )
    assert _losses(_ORIGINAL, mutated) == []


def test_a_swapped_pair_of_numbers_is_not_detected() -> None:
    """Pins a known gap rather than leaving it to be discovered.

    Numbers compare as a multiset, so exchanging two figures between sentences
    keeps the multiset identical. Detecting this needs a reader who knows which
    number belongs to which claim. The docstring of the script says so; this
    test makes the statement fail loudly if the behaviour ever changes.
    """
    mutated = _REWRITTEN.replace("12 beliefs", "PLACEHOLDER_A")
    mutated = mutated.replace("3 fields", "12 fields")
    mutated = mutated.replace("PLACEHOLDER_A", "3 beliefs")
    assert mutated != _REWRITTEN
    assert _losses(_ORIGINAL, mutated) == [], (
        "the multiset gap closed; update the script docstring and this test"
    )


def test_a_code_span_that_wrapped_a_line_break_survives_a_reflow() -> None:
    """A conversion reflows paragraphs, so a wrapped span lands on one line.

    Matching code spans line by line pairs the opening backtick with the NEXT
    span's backtick instead, which invents junk tokens on both sides and can
    hide a real loss behind the noise.
    """
    before = "# T\n\nRun the `aelf gate\nlist` aggregator and the `aelf doctor` check.\n"
    after = "# T\n\nRun the `aelf gate list` aggregator. Then run the `aelf doctor` check.\n"
    assert _losses(before, after) == []


def test_a_renamed_identifier_still_fails_after_the_reflow_allowance() -> None:
    """The allowance above must not swallow a genuine rename."""
    before = "# T\n\nRun the `aelf gate\nlist` aggregator.\n"
    after = "# T\n\nRun the `aelf gate show` aggregator.\n"
    losses = _losses(before, after)
    assert any("inline code" in line for line in losses), losses


_INDENTED_FENCE = """# T

1. Do the first thing:

   ```json
   {"a": 1}
   ```
2. Then set `FIRST` and `SECOND`.
3. Finally read `tests/test_thing.py`.
"""


def test_a_fence_indented_inside_a_list_is_recognised() -> None:
    """An unstripped fence corrupts every inline span after it.

    The stray backtick run pairs with the next real span's opening backtick, so
    the tool reports prose fragments as lost identifiers. That is noise loud
    enough to bury a genuine loss, and it fails a rewrite that changed nothing.
    """
    spans = inline_code(_INDENTED_FENCE)
    assert set(spans) == {"FIRST", "SECOND", "tests/test_thing.py"}, spans
    # And the block itself must be seen as code, not as prose.
    assert '{"a": 1}' in "".join(fences(_INDENTED_FENCE))


def test_an_indented_fence_does_not_make_an_untouched_rewrite_fail() -> None:
    reflowed = _INDENTED_FENCE.replace(
        "2. Then set `FIRST` and `SECOND`.",
        "2. Then set `FIRST`. Then set `SECOND`.",
    )
    assert _losses(_INDENTED_FENCE, reflowed) == []


@pytest.mark.timeout(30)
def test_the_command_line_entry_point_reports_pass_and_fail(tmp_path: Path) -> None:
    original = tmp_path / "before.md"
    good = tmp_path / "good.md"
    bad = tmp_path / "bad.md"
    original.write_text(_ORIGINAL, encoding="utf-8")
    good.write_text(_REWRITTEN, encoding="utf-8")
    bad.write_text(_REWRITTEN.replace("4.5x", "5x"), encoding="utf-8")

    ok = subprocess.run(
        [sys.executable, str(SCRIPT), str(original), str(good)],
        capture_output=True, text=True, check=False, timeout=30,
    )
    assert ok.returncode == 0, ok.stdout
    assert "PASS" in ok.stdout

    fail = subprocess.run(
        [sys.executable, str(SCRIPT), str(original), str(bad)],
        capture_output=True, text=True, check=False, timeout=30,
    )
    assert fail.returncode == 1
    assert "FAIL" in fail.stdout


@pytest.mark.timeout(30)
def test_wrong_argument_count_is_a_usage_error(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path / "only-one.md")],
        capture_output=True, text=True, check=False, timeout=30,
    )
    assert result.returncode == 2


class TestBacktickRuns:
    """A literal ``` inside prose is not two delimiters and a leftover.

    The span scanner used to pair backticks one at a time, so a table cell
    containing a literal triple backtick produced a span plus a stray tick, and
    the stray swallowed the text up to the next real span. `docs/user/CONFIG.md`
    then reported a phantom loss of the token `fences) +`, and that cell could
    not be reworded at all: every rewrite of it failed a check that was wrong.

    The mutation half matters more than the happy path here. A scanner that
    returned an empty Counter would satisfy "no phantom token", so each test
    below also pins a span the scanner must still find.
    """

    def test_a_literal_triple_backtick_yields_no_phantom_token(self) -> None:
        cell = "| `snapshot` | First sentence (split outside ``` fences) + `…`. |"
        spans = inline_code(cell)
        assert "fences) +" not in spans, (
            "the leftover backtick paired with the next span's opener"
        )
        assert spans["snapshot"] == 1
        assert spans["…"] == 1

    def test_rewording_around_a_literal_triple_backtick_is_not_a_loss(self) -> None:
        before = "# T\n\n| a | split outside ``` fences) + `…`. |\n"
        after = "# T\n\n| a | split outside ``` fences, plus `…`. |\n"
        losses, _notes = compare(before, after)
        assert losses == []

    def test_a_genuinely_dropped_span_is_still_a_loss(self) -> None:
        """The fix must not buy its silence by counting nothing."""
        before = "# T\n\n| a | split outside ``` fences) + `…`. |\n"
        after = "# T\n\n| a | split outside ``` fences, plus nothing. |\n"
        losses, _notes = compare(before, after)
        assert any("…" in loss for loss in losses)

    def test_a_run_closes_only_on_a_run_of_its_own_length(self) -> None:
        """``x`` holds a literal backtick; one tick must not close it."""
        assert inline_code("a ``code with ` tick`` b")["code with ` tick"] == 1

    def test_a_span_still_survives_one_reflowed_line_break(self) -> None:
        assert inline_code("use `aelf\nlock` now")["aelf lock"] == 1

    def test_an_unpartnered_run_does_not_run_away_across_paragraphs(self) -> None:
        text = "``` alone\n\nparagraph two\n\nand `real` span\n"
        assert inline_code(text)["real"] == 1
        assert len(inline_code(text)) == 1
