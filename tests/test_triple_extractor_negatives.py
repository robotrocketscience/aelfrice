"""Negative cases: empty / non-relational / structurally similar input
that should NOT produce false-positive triples."""
from __future__ import annotations

import pytest

from aelfrice.triple_extractor import extract_triples


@pytest.mark.parametrize("text", [
    "",
    "   ",
    "\n\n",
])
def test_empty_input_returns_empty(text: str) -> None:
    assert extract_triples(text) == []


@pytest.mark.parametrize("text", [
    "no relational structure here",
    "the brown fox jumped",
    "hello world",
    "this is a sentence without a relational verb",
    "Pi equals 3.14",
])
def test_non_relational_returns_empty(text: str) -> None:
    assert extract_triples(text) == []


def test_word_inside_other_word_does_not_match() -> None:
    """`supports` inside `unsupportsable` should not trigger SUPPORTS."""
    # No real English example; force the engine with a token where the
    # template appears mid-word.
    text = "the foo unsupportsable bar"
    assert extract_triples(text) == []


def test_punctuation_terminates_noun_phrase() -> None:
    """A noun phrase should not bleed across a sentence boundary."""
    # "X. Y supports Z" — the relation should match only "Y supports Z".
    text = "alpha. the index supports queries"
    triples = extract_triples(text)
    assert len(triples) == 1
    assert triples[0].subject == "the index"
    assert triples[0].object == "queries"
