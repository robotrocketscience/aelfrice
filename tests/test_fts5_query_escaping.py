"""FTS5 query input must not trip on user-typeable special characters.

Surfaced during the v0.5.0 onboard regression: `search_beliefs("v0.5")`
raised `sqlite3.OperationalError: fts5: syntax error near "."`. The
fix is to escape each whitespace-separated token by double-quote
wrapping (with embedded `"` doubled per FTS5 syntax) before passing
the query to MATCH. Tests assert behavior through the public
`search_beliefs` contract; the escape helper itself is internal.

Atomic short tests: one property each, all use :memory: SQLite.
"""
from __future__ import annotations

from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief
from aelfrice.store import Store


def _mk(bid: str, content: str) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        demotion_pressure=0,
        created_at="2026-04-26T00:00:00Z",
        last_retrieved_at=None,
    )


# --- Special-character queries no longer raise --------------------------


def test_search_with_dot_in_query_does_not_raise() -> None:
    s = Store(":memory:")
    s.insert_belief(_mk("b1", "the project ships at v0.5 with the regex fallback"))
    hits = s.search_beliefs("v0.5")
    assert any(h.id == "b1" for h in hits)


def test_search_with_hyphen_does_not_raise() -> None:
    s = Store(":memory:")
    s.insert_belief(_mk("b1", "we use multi-hop retrieval at scale"))
    hits = s.search_beliefs("multi-hop")
    assert any(h.id == "b1" for h in hits)


def test_search_with_parens_does_not_raise() -> None:
    s = Store(":memory:")
    s.insert_belief(_mk("b1", "the function is called foo(bar) for legacy reasons"))
    hits = s.search_beliefs("foo(bar)")
    assert any(h.id == "b1" for h in hits)


def test_search_with_slash_does_not_raise() -> None:
    s = Store(":memory:")
    s.insert_belief(_mk("b1", "documented at docs/install.md as the canonical path"))
    hits = s.search_beliefs("docs/install.md")
    assert any(h.id == "b1" for h in hits)


def test_search_with_fts5_keyword_does_not_raise() -> None:
    """FTS5 reserves AND, OR, NEAR, NOT as boolean operators. The escape
    wrapper quotes them as literal words so they don't hit a syntax error."""
    s = Store(":memory:")
    s.insert_belief(_mk("b1", "the rule is AND not OR for this branch"))
    hits = s.search_beliefs("AND")
    assert isinstance(hits, list)


def test_search_with_quote_bearing_query_does_not_raise() -> None:
    s = Store(":memory:")
    s.insert_belief(
        _mk("b1", 'the docstring says "use this only when stable" verbatim')
    )
    hits = s.search_beliefs('"use')
    assert isinstance(hits, list)


def test_search_with_colon_in_query_does_not_raise() -> None:
    s = Store(":memory:")
    s.insert_belief(_mk("b1", "the source format is doc:README.md:p0 for provenance"))
    hits = s.search_beliefs("doc:README.md:p0")
    assert isinstance(hits, list)


# --- Empty / whitespace returns empty list -------------------------------


def test_empty_query_returns_empty_list() -> None:
    s = Store(":memory:")
    s.insert_belief(_mk("b1", "any content here at all"))
    assert s.search_beliefs("") == []


def test_whitespace_only_query_returns_empty_list() -> None:
    s = Store(":memory:")
    s.insert_belief(_mk("b1", "any content here at all"))
    assert s.search_beliefs("   \n\t  ") == []


# --- Existing behavior preserved ---------------------------------------


def test_single_word_query_still_finds_match() -> None:
    s = Store(":memory:")
    s.insert_belief(_mk("b1", "bananas are yellow"))
    s.insert_belief(_mk("b2", "apples are red"))
    hits = s.search_beliefs("bananas")
    assert {h.id for h in hits} == {"b1"}


def test_multi_word_query_implicit_and() -> None:
    """Two tokens AND together: only beliefs containing both match."""
    s = Store(":memory:")
    s.insert_belief(_mk("b1", "bananas are yellow"))
    s.insert_belief(_mk("b2", "yellow submarines are common"))
    s.insert_belief(_mk("b3", "bananas are tasty"))
    hits = s.search_beliefs("bananas yellow")
    ids = {h.id for h in hits}
    assert ids == {"b1"}


def test_no_match_returns_empty_list() -> None:
    s = Store(":memory:")
    s.insert_belief(_mk("b1", "the cat sat on the mat"))
    assert s.search_beliefs("xenomorph") == []


def test_limit_clamp_still_honored() -> None:
    s = Store(":memory:")
    for i in range(7):
        s.insert_belief(_mk(f"b{i}", "shared keyword bananas across all"))
    hits = s.search_beliefs("bananas", limit=3)
    assert len(hits) == 3
