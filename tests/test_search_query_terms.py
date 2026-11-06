"""The search-tool lane builds its FTS5 query from content words only."""

from __future__ import annotations

from aelfrice.hook_search_tool import _extract_query


def _q(text: str) -> str | None:
    return _extract_query({"tool_input": {"query": text}})


def test_the_search_query_drops_stopwords_and_repeats() -> None:
    assert _q("what is the paper on the phase noise in the Paper") == (
        "paper OR phase OR noise"
    )


def test_a_query_of_only_stopwords_is_skipped() -> None:
    assert _q("what is this and how does it") is None


def test_the_token_limit_counts_distinct_terms() -> None:
    q = _q("alpha alpha alpha alpha alpha beta gamma delta epsilon zeta")
    assert q == "alpha OR beta OR gamma OR delta OR epsilon"
