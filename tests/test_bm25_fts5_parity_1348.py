"""BM25F / FTS5 tokenisation parity (#1348, parent #1158 defect 2).

The BM25F index lane and the FTS5 lane are advertised as
interchangeable — `retrieval` swaps one for the other at
`posterior_weight = 0.0`, and `AELFRICE_BM25F=0` is the documented way
to debug one against the other. That only holds if they key their
vocabularies the same way, and they did not: `beliefs_fts` is declared
`porter unicode61` while `tokenize_stemmed` used `\\w+` and an unguarded
Porter stemmer.

What makes this worth a dedicated module rather than another case in
`test_bm25_index.py` is the shape of the bug that preceded it. The
comment justifying the old regex cited
`test_bm25_index.py::test_w0_equivalence_with_fts5` — a test that has
never existed — and the real test it meant,
`test_w0_topk_matches_fts5_baseline`, runs on three pure-ASCII
documents with no underscore and no diacritic. The approximation was
guarded only by inputs on which it could not fail.

So the oracle here is a **real** SQLite `fts5` table declared exactly as
`store.py` declares `beliefs_fts`, read back through
`fts5vocab(..., 'instance')` — never a second Python reimplementation of
what unicode61 is believed to do. And `test_corpus_distinguishes_the_two_
implementations` pins the corpus itself: it asserts the pre-#1348
pipeline *fails* on this fixture, so a future edit cannot quietly make
the parity test green by softening the inputs.
"""
from __future__ import annotations

import re
import sqlite3

import pytest

from aelfrice.bm25 import (
    _FTS5_TOKEN_PATTERN,
    _PORTER_STEMMER,
    BM25Index,
    _stem,
    tokenize,
    tokenize_stemmed,
)
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief
from aelfrice.store import MemoryStore

# Exactly the declaration `store.py` uses for `beliefs_fts`. Written out
# rather than imported: if that declaration changes, this module should
# start failing, not silently follow it.
FTS5_DECL = "porter unicode61"

# The fixture the acceptance criterion asks for. Every line carries at
# least one input on which `\w+` + unguarded Porter disagrees with
# unicode61, and between them they cover each divergence class:
# underscore-joined identifiers, diacritics, mixed case, and short words
# (the class the byte guard fixes, and the one a hand-picked corpus is
# most likely to omit).
PARITY_CORPUS = (
    "ADD_TO_LIST appends to the queue",
    "snake_case_name and CamelCase and SCREAMING_SNAKE",
    "café naïve résumé jalapeño",
    "max_retry_count is 5 as of v4.1.0",
    "x86_64 builds vs arm64 builds",
    "it is as good as it was, s'il vous plaît",
    "MixedCase_Identifiers Are Common",
    "__dunder__ and _leading and trailing_",
    "running as it was, abusive organizations",
)

# unicode61 folds U+00B5 MICRO SIGN to Greek mu (U+03BC) — a
# compatibility mapping that canonical decomposition does not perform,
# so `_fold_diacritics` cannot reach it. Held here as a named, asserted
# residual rather than dropped from the corpus, because a divergence
# that nothing records is how #1348 happened.
KNOWN_RESIDUAL = "the µs latency budget"


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
        created_at="2026-08-05T00:00:00Z",
        last_retrieved_at=None,
    )


def fts5_terms(text: str) -> list[str]:
    """The terms SQLite indexes for `text`, in offset order.

    Both virtual tables live in `temp` because the three-argument
    `fts5vocab('temp', <tbl>, 'instance')` form only parses there — the
    same reason `store.py::_ensure_fts5_query_probe` creates its probe
    in `temp` too.
    """
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute(
            f"CREATE VIRTUAL TABLE temp.probe USING fts5(t, "
            f"tokenize='{FTS5_DECL}')"
        )
        conn.execute(
            "CREATE VIRTUAL TABLE temp.probe_terms "
            "USING fts5vocab('temp', 'probe', 'instance')"
        )
        conn.execute("INSERT INTO temp.probe(rowid, t) VALUES (1, ?)", (text,))
        rows = conn.execute(
            "SELECT term, offset FROM temp.probe_terms ORDER BY offset"
        ).fetchall()
    finally:
        conn.close()
    return [term for term, _offset in rows]


_LEGACY_PATTERN = re.compile(r"\w+", re.UNICODE)


def _legacy_tokenize_stemmed(text: str) -> list[str]:
    """The pre-#1348 pipeline: `\\w+`, no fold, unguarded Porter.

    Spelled out here rather than reached for through the shipped code,
    so this stays a fixed historical baseline even as `bm25` changes.
    """
    return [
        _PORTER_STEMMER.stemWord(m.group(0).lower())
        for m in _LEGACY_PATTERN.finditer(text)
    ]


@pytest.mark.parametrize("text", PARITY_CORPUS)
def test_tokenize_stemmed_matches_the_fts5_term_list(text: str) -> None:
    """`tokenize_stemmed` reproduces what FTS5 actually indexes.

    Order-sensitive on purpose: comparing sorted sets would hide a
    pipeline that emits the right terms in the wrong positions, and the
    BM25F lane's term-frequency counts depend on the multiset, not the
    set.
    """
    assert tokenize_stemmed(text) == fts5_terms(text)


@pytest.mark.parametrize("text", PARITY_CORPUS)
def test_corpus_distinguishes_the_two_implementations(text: str) -> None:
    """Every fixture line must be one the old pipeline got wrong.

    This is the assertion that keeps the parity test honest. Without it
    someone can make `test_tokenize_stemmed_matches_the_fts5_term_list`
    pass by replacing the corpus with ASCII lowercase prose — which is
    precisely how the approximation survived review the first time.
    """
    assert _legacy_tokenize_stemmed(text) != fts5_terms(text)


def test_micro_sign_is_a_known_and_bounded_residual() -> None:
    """Parity is close, not exact, and the gap is named.

    SQLite folds U+00B5 to Greek mu; NFD does not. Special-casing it
    would buy 12 of 44,655 live beliefs, so it is recorded instead. If a
    later change makes this pass, the assertion should be deleted and
    the residual count in `tokenize_stemmed`'s docstring updated — it
    failing is the signal, not an excuse to loosen it.
    """
    ours = tokenize_stemmed(KNOWN_RESIDUAL)
    theirs = fts5_terms(KNOWN_RESIDUAL)
    assert ours != theirs
    assert "µ" in ours and "μ" in theirs
    # Everything apart from the micro sign still agrees.
    assert [t for t in ours if t != "µ"] == [t for t in theirs if t != "μ"]


@pytest.mark.parametrize(
    "text", ["ﬁle", "①", "x²", "ＡＢＣ", "½ cup", "Ⅸ"],
)
def test_compatibility_forms_are_not_folded(text: str) -> None:
    """The fold is NFD, and swapping it for NFKD is a regression.

    unicode61 applies essentially no compatibility mappings; NFKD
    applies all of them. Measured over 44,655 live beliefs NFKD fixes 24
    documents a bare split gets wrong and breaks 49 it gets right — it
    is net-negative against doing nothing, which is why #1348 did not
    take the normalisation form its own body proposed. These inputs are
    the ones that separate the two forms.
    """
    assert tokenize_stemmed(text) == fts5_terms(text)


def test_shipped_word_class_and_stem_guards_are_the_measured_ones() -> None:
    """Pin the values, not just the mechanism.

    The tests above would still pass if the pattern and the guard bounds
    were monkeypatched at runtime, so pin what actually ships. `3` and
    `64` are SQLite's own bounds, measured in UTF-8 bytes.
    """
    assert _FTS5_TOKEN_PATTERN.pattern == r"[^\W_]+"

    # 2 bytes: below SQLite's floor, left alone. `is` -> `i` was the
    # single largest source of divergence on the live store.
    assert _stem("is") == "is"
    assert _stem("as") == "as"
    # 3 bytes, including a multi-byte character: stemmed.
    assert _stem("μs") == "μ"
    assert _stem("running") == "run"
    # 64 bytes stemmed, 65 passed through.
    assert _stem("a" * 63 + "s") == "a" * 63
    assert _stem("a" * 64 + "s") == "a" * 64 + "s"


def test_tokenize_still_keeps_underscores_whole() -> None:
    """`tokenize()` is deliberately *not* changed by #1348.

    Its consumers — consolidation blocking shingles, dedup,
    `relationship_detector`, `query_understanding.strategy` — compare
    against their own vocabularies and owe FTS5 nothing. Splitting them
    on `_` would move dedup behaviour with no measurement behind it, so
    the divergence is intentional and pinned here.
    """
    assert tokenize("ADD_TO_LIST") == ["add_to_list"]
    assert tokenize("max_retry_count") == ["max_retry_count"]
    assert tokenize("café") == ["café"]
    assert tokenize_stemmed("ADD_TO_LIST") == ["add", "to", "list"]


def test_bm25f_lane_answers_the_queries_fts5_answers() -> None:
    """The operator-visible symptom, end to end.

    Before #1348 the default BM25F lane returned zero results for each
    of these while the FTS5 lane returned a hit — the divergence was not
    a ranking nuance, it was a blank result set.
    """
    store = MemoryStore(":memory:")
    store.insert_belief(
        _mk("b1", "the max_retry_count knob caps reconnection attempts")
    )
    store.insert_belief(_mk("b2", "the café menu changes weekly"))
    store.insert_belief(_mk("b3", "garden grows tomatoes"))

    index = BM25Index.build(store, anchor_weight=0)
    for query in ("retry", "count", "cafe", "max_retry_count", "café"):
        bm25f = {bid for bid, _score in index.score(query, top_k=10)}
        fts5 = {b.id for b in store.search_beliefs(query, limit=10)}
        assert bm25f, f"BM25F lane returned nothing for {query!r}"
        assert bm25f == fts5, f"lanes disagree on {query!r}"
