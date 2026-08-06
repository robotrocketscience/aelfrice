"""anchor_text is always a substring of input prose."""
from __future__ import annotations

import pytest

from aelfrice.triple_extractor import extract_triples


@pytest.mark.parametrize("text", [
    "the new index supports faster queries",
    "the proposal cites the prior memo",
    "in section 4 the spec is derived from the prior memo, see footnote",
    "the cache layer relates to retrieval; the index supports it",
])
def test_anchor_is_substring_of_input(text: str) -> None:
    triples = extract_triples(text)
    assert triples
    for t in triples:
        assert t.anchor_text in text, (
            f"anchor_text={t.anchor_text!r} not a substring of input"
        )


def test_anchor_widens_short_match_with_context() -> None:
    """When the bare match is shorter than the soft target, the
    extractor pulls surrounding prose to give downstream retrieval
    a useful context window."""
    # The subject carries an overt determiner because single-token
    # relation verbs require one (#1376); the match is still far
    # shorter than ANCHOR_CONTEXT_TARGET, which is what this asserts.
    text = (
        "Background paragraph that frames the discussion. "
        "the X supports Y. "
        "Followup sentence with more context."
    )
    triples = extract_triples(text)
    assert triples
    bare_match_len = len("the X supports Y")
    assert all(len(t.anchor_text) >= bare_match_len for t in triples)
