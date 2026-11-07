"""The search-tool lane builds its FTS5 query from content words only."""

from __future__ import annotations

import pytest

from aelfrice.hook_search_tool import _extract_query


def _q(text: str) -> str | None:
    return _extract_query({"tool_input": {"query": text}})


def _p(pattern: str) -> str | None:
    return _extract_query({"tool_input": {"pattern": pattern}})


def test_the_search_query_drops_stopwords_and_repeats() -> None:
    assert _q("what is the paper on the phase noise in the Paper") == (
        "paper OR phase OR noise"
    )


def test_a_query_of_only_stopwords_is_skipped() -> None:
    assert _q("what is this and how does it") is None


def test_the_token_limit_counts_distinct_terms() -> None:
    q = _q("alpha alpha alpha alpha alpha beta gamma delta epsilon zeta")
    assert q == "alpha OR beta OR gamma OR delta OR epsilon"


def test_the_token_limit_counts_after_stopwords_are_gone() -> None:
    q = _q("what the alpha and the beta for gamma with delta from epsilon zeta")
    assert q == "alpha OR beta OR gamma OR delta OR epsilon"


def test_a_webfetch_prompt_drops_stopwords() -> None:
    assert _extract_query({"tool_input": {"prompt": "what does the parser use"}}) == "parser"


def test_a_stopword_in_any_case_is_dropped() -> None:
    assert _q("WHAT The parser") == "parser"


def test_a_query_term_keeps_its_case() -> None:
    assert _q("the Parser and FTS5") == "Parser OR FTS5"
    assert _p("GetUser") == "GetUser"


@pytest.mark.parametrize("pattern, want", [
    ("get", "get"),
    ("using", "using"),
    ("from|import", "from OR import"),
    ("get_user|using|from", "get_user OR using OR from"),
    ("**/use/*.py", "use"),
])
def test_a_code_pattern_keeps_its_identifiers(pattern: str, want: str) -> None:
    assert _p(pattern) == want


@pytest.mark.parametrize("pattern, want", [
    ("get|Get|get", "get"),
    ("Get|get|GET", "Get"),
])
def test_a_code_pattern_still_drops_repeats_in_any_case(pattern: str, want: str) -> None:
    assert _p(pattern) == want
