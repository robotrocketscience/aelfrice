"""The FTS5 lane answers multi-word queries (#1177, #1158).

`_escape_fts5_query` joined every whitespace token with spaces, which
FTS5 reads as an implicit AND. Requiring *every* token to be present is
zero-recall on natural-language queries — the shape a UserPromptSubmit
prompt actually has. Measured over 503 distinct logged prompts against a
live 44,584-belief store, the conjunctive form returned nothing for
28.7% of user turns and 100% of harness blocks.

`search_beliefs` now ORs the rarest few tokens instead. These tests pin
the properties that fix rests on, each chosen so that reverting the
corresponding piece of the implementation fails one of them:

  1. A multi-word query where no single belief holds every term returns
     hits (the cliff itself).
  2. The trim ranks by document frequency, so the discriminating token
     survives and the ubiquitous one is dropped.
  3. Ranking resolves tokens through FTS5's own tokenizer, not a Python
     approximation of it — the underscore/diacritic cases where the two
     disagree.
  4. The expression carries original tokens, never re-emitted stems,
     because the porter stemmer is not idempotent.
  5. Escaping still holds: FTS5 operators and quotes in user input are
     data, not syntax.
  6. The conjunctive builder is still available and still conjunctive.
"""
from __future__ import annotations

import sqlite3

import pytest

from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief
from aelfrice.store import (
    _escape_fts5_query,
    _escape_fts5_query_disjunctive,
    MemoryStore,
)


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
        created_at="2026-07-30T00:00:00Z",
        last_retrieved_at=None,
        session_id="s1",
        origin="agent_inferred",
    )


@pytest.fixture()
def store(tmp_path):
    s = MemoryStore(str(tmp_path / "memory.db"))
    yield s
    s.close()


def test_multi_word_query_is_not_zero_recall(store: MemoryStore) -> None:
    """Property 1 — the cliff. No single belief contains every token of
    the query, which is what made the conjunctive lane return nothing.
    """
    store.insert_belief(_mk("b1", "the retrieval budget is configured per project"))
    store.insert_belief(_mk("b2", "injection happens in the prompt hook"))

    query = "how do I configure the retrieval budget for injection"
    assert store.search_beliefs(query, limit=10), (
        "multi-word query returned no hits — the implicit-AND cliff is back"
    )
    # And the conjunctive builder still demonstrates the defect, so this
    # test cannot pass by the corpus happening to contain every token.
    conjunctive = _escape_fts5_query(query)
    rows = store._conn.execute(  # noqa: SLF001
        "SELECT b.id FROM beliefs b JOIN beliefs_fts f ON f.id = b.id "
        "WHERE beliefs_fts MATCH ?",
        (conjunctive,),
    ).fetchall()
    assert rows == [], "fixture no longer reproduces the AND cliff"


def test_trim_ranks_by_document_frequency(store: MemoryStore) -> None:
    """Property 2 — the trim is ordered by document frequency, rarest
    first, not by position in the query.

    Four terms are planted at four separate frequencies and the query
    lists them commonest-first, so keeping the first N in query order
    (or any order-preserving trim) would select the opposite set.
    """
    # alpha: 40 docs, beta: 20, gamma: 5, delta: 1.
    for i in range(40):
        words = ["alpha"]
        if i < 20:
            words.append("beta")
        if i < 5:
            words.append("gamma")
        if i < 1:
            words.append("delta")
        store.insert_belief(_mk(f"doc{i}", " ".join(words) + f" filler{i}"))

    ranked = store._fts5_rarest_tokens(  # noqa: SLF001
        ["alpha", "beta", "gamma", "delta"], 4,
    )
    assert ranked == ["delta", "gamma", "beta", "alpha"], (
        "tokens are not ordered ascending by document frequency"
    )

    expr = store._fts5_match_expression("alpha beta gamma delta")  # noqa: SLF001
    assert expr == '"delta" OR "gamma" OR "beta"', expr
    assert "alpha" not in expr, f"commonest token survived the trim: {expr!r}"


def test_ranking_uses_fts5_tokenizer_not_a_python_approximation(
    store: MemoryStore,
) -> None:
    """Property 3. `unicode61` splits on `_` and folds diacritics; the
    Python tokenizer in `aelfrice.bm25` does neither. Ranking a query
    against the index with the wrong tokenizer leaves tokens unresolved,
    and unresolved tokens collapse to df 0 and masquerade as the rarest
    — the failure #1158 records for the IDF-clip lane.

    `ADD_TO_LIST` must therefore resolve (as add/to/list) rather than
    being treated as an unknown, and so must an accented token.
    """
    for i in range(30):
        store.insert_belief(_mk(f"pad{i}", f"add to list padding {i}"))
    store.insert_belief(_mk("acc", "the café menu"))

    assert store._fts5_rarest_tokens(["ADD_TO_LIST"], 3) == ["ADD_TO_LIST"]  # noqa: SLF001
    # Resolves via its parts, not as an opaque unknown token.
    assert store._fts5_rarest_tokens(["ADD_TO_LIST", "zzzznotindexed"], 3) == [  # noqa: SLF001
        "ADD_TO_LIST"
    ], "a token absent from the index should be dropped, not ranked rarest"
    # Diacritic folding: 'cafe' is what got indexed for 'café'.
    assert store._fts5_rarest_tokens(["café"], 3) == ["café"]  # noqa: SLF001


def test_expression_carries_original_tokens_not_stems(store: MemoryStore) -> None:
    """Property 4. The porter stemmer is not idempotent — 416 of the
    15,208 terms on a live store stem again to something else (`abus` ->
    `abu`). Emitting the stemmed term would silently stop matching those
    documents, so the expression must carry what the user typed.
    """
    store.insert_belief(_mk("b1", "abusive configuration"))
    expr = store._fts5_match_expression("abusive")  # noqa: SLF001
    assert "abusive" in expr, f"expression re-emitted a stem: {expr!r}"
    assert store.search_beliefs("abusive", limit=5), (
        "a term whose stem is not idempotent stopped matching"
    )


@pytest.mark.parametrize(
    "query",
    [
        "AND OR NEAR",
        'quote"inside',
        "dash-token slash/token dot.token",
        "(parens) [brackets] *star",
        "",
        "   ",
        "--- ... ###",
    ],
)
def test_operators_and_punctuation_stay_data(store: MemoryStore, query: str) -> None:
    """Property 5. Quoting is what keeps FTS5 syntax out of user input;
    the disjunction must not have opened a hole in it. None of these may
    raise OperationalError.
    """
    store.insert_belief(_mk("b1", "ordinary content"))
    try:
        store.search_beliefs(query, limit=5)
        store.search_beliefs_scored(query, limit=5)
    except sqlite3.OperationalError as exc:  # pragma: no cover - the failure
        pytest.fail(f"query {query!r} reached FTS5 as syntax: {exc}")


def test_blank_query_short_circuits(store: MemoryStore) -> None:
    """Empty input never reaches the engine, which would raise on an
    empty MATCH expression."""
    assert store._fts5_match_expression("") == ""  # noqa: SLF001
    assert store._fts5_match_expression("   ") == ""  # noqa: SLF001
    assert store.search_beliefs("", limit=5) == []
    assert store.search_beliefs_scored("   ", limit=5) == []


def test_conjunctive_builder_is_still_conjunctive() -> None:
    """Property 6. `_escape_fts5_query` is retained for callers that do
    want every token present; it must not have drifted to OR.
    """
    assert _escape_fts5_query("alpha beta") == '"alpha" "beta"'
    assert _escape_fts5_query("") == ""
    assert _escape_fts5_query('say "hi"') == '"say" """hi"""'


def test_disjunctive_fallback_keeps_every_token() -> None:
    """The no-corpus-statistics path used when the probe is unavailable.
    It still fixes the cliff; it just cannot trim."""
    assert _escape_fts5_query_disjunctive("alpha beta") == '"alpha" OR "beta"'
    assert _escape_fts5_query_disjunctive("") == ""


def test_search_still_works_when_the_probe_is_unavailable(
    store: MemoryStore, monkeypatch,
) -> None:
    """An SQLite built without fts5vocab degrades to the full OR rather
    than to an exception or to zero recall."""
    store.insert_belief(_mk("b1", "the retrieval budget is configured"))
    monkeypatch.setattr(
        MemoryStore, "_ensure_fts5_query_probe", lambda self: False,
    )
    query = "how do I configure the retrieval budget"
    expr = store._fts5_match_expression(query)  # noqa: SLF001
    assert expr == _escape_fts5_query_disjunctive(query)
    assert store.search_beliefs(query, limit=5)
