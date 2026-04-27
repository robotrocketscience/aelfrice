"""Tests for the sentence-extraction utility."""
from __future__ import annotations

from aelfrice.extraction import extract_sentences


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
