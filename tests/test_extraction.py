"""Tests for the sentence-extraction utility."""
from __future__ import annotations

import pytest

from aelfrice.extraction import extract_sentences, sentence_forms


def test_extracts_simple_sentences() -> None:
    text = "The cat sat on the mat. The dog barked loudly. Birds fly south."
    out = extract_sentences(text)
    assert len(out) == 3
    assert out[0].startswith("The cat")


def test_strips_code_blocks() -> None:
    text = "Hello there friend.\n```python\nprint('hi')\n```\nThe end is near."
    out = extract_sentences(text)
    assert all("print" not in s for s in out)
    assert any("Hello there friend" in s for s in out)


def test_strips_inline_code() -> None:
    out = extract_sentences("Use the `foo()` function carefully here.")
    assert len(out) == 1
    assert "foo" not in out[0]


def test_strips_urls() -> None:
    out = extract_sentences("See https://example.com/path for the docs page.")
    assert all("example.com" not in s for s in out)


def test_strips_markdown_headers_and_emphasis() -> None:
    text = "# Section Header\nThis is **bold** important text here."
    out = extract_sentences(text)
    assert any("important text" in s for s in out)
    assert all("**" not in s for s in out)
    assert all("#" not in s for s in out)


def test_discards_short_fragments() -> None:
    text = "Long enough sentence here. Too short. Another long enough one."
    out = extract_sentences(text)
    # "Too short." is 10 chars exactly, kept; shorter would be dropped.
    # Use clearly-short fragment:
    text2 = "Long enough sentence here. ok. Another long enough one."
    out2 = extract_sentences(text2)
    assert all("ok" != s.strip(".") for s in out2)
    assert len(out) >= 2


def test_empty_text_returns_empty_list() -> None:
    assert extract_sentences("") == []


def test_strips_table_rows_and_list_markers() -> None:
    text = "Intro paragraph here for context.\n| col1 | col2 |\n- bullet item one"
    out = extract_sentences(text)
    assert all("|" not in s for s in out)
    assert any("Intro paragraph" in s for s in out)


def test_underscore_identifiers_preserved() -> None:
    """Regression: italic-stripping regex must not mangle snake_case
    identifiers, file paths, or error codes that contain underscores."""
    text = "The auth_service.py module imports from user_session.py."
    out = extract_sentences(text)
    joined = " ".join(out)
    assert "auth_service.py" in joined, f"snake_case file_path mangled: {joined!r}"
    assert "user_session.py" in joined, f"snake_case file_path mangled: {joined!r}"


def test_multiple_underscored_tokens_preserved() -> None:
    """Regression: multiple underscored tokens in one sentence must
    not be glued together by an unanchored italic regex."""
    text = "The error E_AUTH_001 is raised by auth_service.py at runtime."
    out = extract_sentences(text)
    joined = " ".join(out)
    assert "E_AUTH_001" in joined, f"error_code mangled: {joined!r}"
    assert "auth_service.py" in joined, f"file_path mangled: {joined!r}"


def test_underscore_italic_still_stripped() -> None:
    """The fix must keep stripping real Markdown italic at word
    boundaries (whitespace/punctuation outside the underscores)."""
    text = "This is _emphasized_ text in a sentence."
    out = extract_sentences(text)
    joined = " ".join(out)
    assert "_emphasized_" not in joined
    assert "emphasized" in joined


def test_underscore_bold_still_stripped() -> None:
    """The fix must keep stripping real Markdown __bold__ at word
    boundaries."""
    text = "This is __very important__ to remember in a sentence."
    out = extract_sentences(text)
    joined = " ".join(out)
    assert "__very important__" not in joined
    assert "very important" in joined


def test_mixed_italic_and_identifier() -> None:
    """A sentence containing both real italic and identifiers must
    strip the italic and preserve the identifier."""
    text = "Use _emphasis_ when documenting auth_service.py behavior in a sentence."
    out = extract_sentences(text)
    joined = " ".join(out)
    assert "_emphasis_" not in joined
    assert "emphasis" in joined
    assert "auth_service.py" in joined


# --- #1874: the store gets the whole form, and decisions read the holed one --


def _whole(text: str) -> list[str]:
    return [form.whole for form in sentence_forms(text)]


@pytest.mark.parametrize(
    "sentence",
    [
        "Use `__init__` and `**kwargs` in the widget constructor.",
        "Set `*a*` and `_b_` as the two widget flags here.",
        "Read https://example.invalid/a_b_c/*d*/__e__ for the widget docs.",
        "The note says `alpha. beta` is one flag to keep together.",
        "Open `src/widget/core.py` and example.invalid/a.b for the widget.",
        "See https://example.invalid/a. Then read the second part.",
        "Is it https://example.invalid/a? Then read the second part.",
        "Read **https://example.invalid/a** for the docs.",
    ],
)
def test_the_whole_form_keeps_each_span_exactly(sentence: str) -> None:
    assert _whole(sentence) == [sentence]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Stop right there. `zqtool` runs the rest here.", ["Stop right there.", "`zqtool` runs the rest here."]),
        ("- `zqtool` runs the rest here.", ["`zqtool` runs the rest here."]),
        ("1. `zqtool` runs the rest here.", ["`zqtool` runs the rest here."]),
        ("# `zqtool` runs the rest here.", ["`zqtool` runs the rest here."]),
        ("- `zqtool`\nruns the rest here.", ["`zqtool` runs the rest here."]),
        ("`zqtool` 1. runs the rest here.", ["`zqtool` runs the rest here."]),
        ("| `a` | b |\nUse `c` for the widget now.", ["Use `c` for the widget now."]),
        ("`a` is short.\nUse `c` for the widget now.", ["Use `c` for the widget now."]),
        ("Stop right there. https://example.invalid/a runs next here.", ["Stop right there.", "https://example.invalid/a runs next here."]),
    ],
)
def test_each_span_lands_in_its_own_sentence(text: str, expected: list[str]) -> None:
    assert _whole(text) == expected


@pytest.mark.parametrize(
    "line",
    [
        "`git push --force origin main`",
        "`first --command` `second --command`",
        "https://example.invalid/only/a/link/on/this/line",
        "- https://example.invalid/a/link/in/a/list/item",
        "⎿ `pasted --command --after --a --glyph`",
        "Run `pytest -q tests/`.",
    ],
)
def test_the_floor_reads_the_holed_form(line: str) -> None:
    assert sentence_forms(line) == []


@pytest.mark.parametrize(
    "first, second",
    [
        ("`zqone`", "`zqtwo`"),
        ("https://one.example.invalid/a", "https://two.example.invalid/b"),
        ("`zqone`", "https://two.example.invalid/b"),
        ("https://one.example.invalid/a", "`zqtwo`"),
    ],
)
@pytest.mark.parametrize(
    "prefix, kept",
    [
        ("- ", ""),
        ("* ", ""),
        ("1. ", ""),
        ("# ", ""),
        ("Stop right there. ", "Stop right there."),
        ("Stop right there! ", "Stop right there!"),
    ],
)
def test_two_spans_keep_their_separator_after_a_removed_marker(
    first: str, second: str, prefix: str, kept: str,
) -> None:
    out = _whole(f"{prefix}{first} {second} are the widget docs.")
    expected = [kept] if kept else []
    assert out == [*expected, f"{first} {second} are the widget docs."]


def test_spans_that_touched_stay_touching_after_a_removed_marker() -> None:
    assert _whole("- `zqone``zqtwo` are the widget docs.") == ["`zqone``zqtwo` are the widget docs."]


def test_spans_in_a_table_row_do_not_join_the_next_line() -> None:
    assert _whole("| `a` | `b` |\nUse `c` for the widget now.") == ["Use `c` for the widget now."]


def test_backticks_paired_across_lines_stay_removed() -> None:
    text = "A stray ` opens here\n15 +cd some-folder\nand ` closes it there at last."
    assert _whole(text) == ["A stray   closes it there at last."]


def test_a_url_ends_at_the_hole_a_span_leaves() -> None:
    text = "Read https://example.invalid/a`b first\n15 +cd some-folder\nand ` closes it there."
    assert _whole(text) == ["Read https://example.invalid/a  closes it there."]


def test_a_fence_stays_removed_next_to_inline_code() -> None:
    text = "Run `widget --check` first.\n```bash\nwidget --other-flag\n```\nThen stop here."
    assert _whole(text) == ["Run `widget --check` first.", "Then stop here."]


def test_a_long_span_stays_whole() -> None:
    """Pins the decision to set no span cap, from the #1874 measurement."""
    span = "`" + "x" * 5000 + "`"
    assert _whole(f"Paste {span} into the widget.") == [f"Paste {span} into the widget."]


def test_input_cannot_forge_a_held_span() -> None:
    out = _whole("Before \x1f and `real --span` after this.")
    assert out == ["Before   and `real --span` after this."]


@pytest.mark.parametrize(
    "text",
    [
        "Use `__init__` and `**kwargs` in the widget constructor.",
        "The note says `alpha. beta` is one flag to keep together.",
        "See https://example.invalid/a. Then read the second part.",
        "Read https://example.invalid/a_b_c for the widget docs.",
    ],
)
def test_extraction_is_stable_on_its_own_output(text: str) -> None:
    """A belief's stored text splits to itself, as `stored_sentences` needs."""
    for form in sentence_forms(text):
        assert sentence_forms(form.whole) == [form]
