"""Tests for the `bulk=` parameter on `ingest_turn` (issue #194).

`bulk=True` was originally a research-line / batch-ingest hint that
suppressed `store.record_corroboration` for duplicate sentences. Under
#264 the entry point no longer owns dedup-or-corroborate decisions —
every raw sentence appends a row to `ingest_log` and the derivation
worker handles canonical-belief insert vs corroborate via
`insert_or_corroborate`. The bulk parameter is now a no-op kept for
API parity; both `bulk=True` and `bulk=False` produce identical final
state including the corroboration audit table.

In-memory store, deterministic, no LLM/IO.
"""
from __future__ import annotations

from aelfrice.ingest import ingest_turn
from aelfrice.store import MemoryStore


def _fresh_store() -> MemoryStore:
    return MemoryStore(":memory:")


def test_bulk_default_is_false() -> None:
    """`bulk` defaults to False — preserves prior behaviour for all
    existing callers (signature is keyword-only, additive).
    """
    store = _fresh_store()
    text = "The sky is blue. Water is wet."
    n = ingest_turn(store, text=text, source="user")
    assert n >= 1


def test_bulk_keyword_only() -> None:
    """`bulk` is keyword-only — positional pass should fail. Guards
    against a future arg being inserted before it.
    """
    store = _fresh_store()
    import pytest

    with pytest.raises(TypeError):
        # 7 positional args (bulk would be the 7th) — should be rejected
        # because bulk is keyword-only.
        ingest_turn(store, "Hello world.", "user", None, None, "", True)  # type: ignore[misc]


def test_bulk_unique_sentences_same_final_belief_state() -> None:
    """For unique-sentence input, bulk=True and bulk=False produce
    identical final belief state. (No duplicates means no corroboration
    writes either way — this is a sanity baseline.)
    """
    text = "The sky is blue. Water is wet. Fire is hot."

    store_bulk = _fresh_store()
    store_normal = _fresh_store()
    n_bulk = ingest_turn(store_bulk, text=text, source="user", bulk=True)
    n_normal = ingest_turn(store_normal, text=text, source="user", bulk=False)

    assert n_bulk == n_normal
    assert sorted(store_bulk.list_belief_ids()) == sorted(store_normal.list_belief_ids())


def test_bulk_is_now_a_noop_under_pr264() -> None:
    """Post-#264: bulk=True and bulk=False produce identical state in
    every observable way — same beliefs, same corroboration count, same
    log rows. The bulk parameter is preserved for API parity but no
    longer changes behavior. The pre-#264 fast-path that skipped
    record_corroboration on duplicate sentences is gone (the entry
    point no longer calls record_corroboration directly; the worker's
    insert_or_corroborate records a corroboration on every duplicate
    raw input).
    """
    text = "The sky is blue."
    source = "user"

    store_bulk = _fresh_store()
    ingest_turn(store_bulk, text=text, source=source, bulk=True)
    ingest_turn(store_bulk, text=text, source=source, bulk=True)
    ingest_turn(store_bulk, text=text, source=source, bulk=True)

    store_normal = _fresh_store()
    ingest_turn(store_normal, text=text, source=source, bulk=False)
    ingest_turn(store_normal, text=text, source=source, bulk=False)
    ingest_turn(store_normal, text=text, source=source, bulk=False)

    bulk_ids = sorted(store_bulk.list_belief_ids())
    normal_ids = sorted(store_normal.list_belief_ids())
    assert bulk_ids == normal_ids
    assert len(bulk_ids) == 1

    bid = bulk_ids[0]
    # Two duplicate ingests after the first produce two corroboration
    # rows in BOTH modes — the worker's insert_or_corroborate runs on
    # every duplicate raw input regardless of bulk.
    assert store_normal.count_corroborations(bid) == 2
    assert store_bulk.count_corroborations(bid) == 2
    # And the log row count is also identical (3 raw inputs each).
    assert store_normal.count_ingest_log() == 3
    assert store_bulk.count_ingest_log() == 3


def test_bulk_writes_ingest_log() -> None:
    """bulk=True must still write the v2.0 ingest_log row — that is the
    source-of-truth record for the parallel-write phase, NOT a per-turn
    side effect to skip.
    """
    store = _fresh_store()
    n = ingest_turn(store, text="The sky is blue.", source="user", bulk=True)
    assert n == 1
    assert store.count_ingest_log() == 1
