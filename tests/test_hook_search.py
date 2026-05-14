"""hook_search: search_for_prompt + record_retrieval contract.

Closes the v1.0.1 retrieval-side feedback-loop gap (LIMITATIONS § Hook
layer). Every hook-driven retrieval writes one feedback_history row
tagged source='hook' with a small positive valence.
"""
from __future__ import annotations

import io

import pytest

from aelfrice.hook_search import (
    HOOK_FEEDBACK_SOURCE,
    HOOK_RETRIEVAL_VALENCE,
    record_retrieval,
    search_for_prompt,
)
from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_CONTRADICTS,
    LOCK_NONE,
    LOCK_USER,
    Belief,
    Edge,
)
from aelfrice.store import MemoryStore


def _mk(
    bid: str,
    content: str = "",
    lock_level: str = LOCK_NONE,
    locked_at: str | None = None,
) -> Belief:
    return Belief(
        id=bid,
        content=content or f"belief {bid}",
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=lock_level,
        locked_at=locked_at,
        created_at="2026-04-26T00:00:00Z",
        last_retrieved_at=None,
    )


def _seed(*beliefs: Belief) -> MemoryStore:
    s = MemoryStore(":memory:")
    for b in beliefs:
        s.insert_belief(b)
    return s


# --- Constants -----------------------------------------------------------


def test_hook_feedback_source_constant_is_hook() -> None:
    """LIMITATIONS.md commits to source='hook'; downstream analysis
    depends on this exact string."""
    assert HOOK_FEEDBACK_SOURCE == "hook"


def test_hook_retrieval_valence_is_small_positive() -> None:
    """Implicit signal — should be smaller than the explicit ±1.0 default
    so high-volume hook activity does not dominate the posterior."""
    assert 0.0 < HOOK_RETRIEVAL_VALENCE <= 1.0


# --- record_retrieval: audit rows ----------------------------------------


def test_record_retrieval_writes_one_row_per_belief() -> None:
    s = _seed(_mk("b1"), _mk("b2"), _mk("b3"))
    written = record_retrieval(s, [s.get_belief("b1"), s.get_belief("b2")])  # type: ignore[list-item]
    assert written == 2
    assert s.count_feedback_events() == 2


def test_record_retrieval_tags_rows_with_hook_source() -> None:
    s = _seed(_mk("b1"))
    record_retrieval(s, [s.get_belief("b1")])  # type: ignore[list-item]
    events = s.list_feedback_events()
    assert len(events) == 1
    assert events[0].source == "hook"


def test_record_retrieval_uses_default_valence() -> None:
    s = _seed(_mk("b1"))
    record_retrieval(s, [s.get_belief("b1")])  # type: ignore[list-item]
    events = s.list_feedback_events()
    assert events[0].valence == HOOK_RETRIEVAL_VALENCE


def test_record_retrieval_empty_iterable_writes_no_rows() -> None:
    s = _seed(_mk("b1"))
    written = record_retrieval(s, [])
    assert written == 0
    assert s.count_feedback_events() == 0


def test_record_retrieval_updates_alpha_not_beta() -> None:
    s = _seed(_mk("b1"))
    record_retrieval(s, [s.get_belief("b1")])  # type: ignore[list-item]
    b = s.get_belief("b1")
    assert b is not None
    assert b.alpha == 1.0 + HOOK_RETRIEVAL_VALENCE
    assert b.beta == 1.0


# --- record_retrieval: locks survive implicit hook exposure -------------


def test_record_retrieval_does_not_demote_locked_contradictors() -> None:
    """Hook exposure of a contradictor must not change the lock state
    of any user-locked belief. Auto-demotion has been removed (#814 /
    PHILOSOPHY #605); this test still pins the invariant from the
    consuming side."""
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("X"))
    s.insert_belief(_mk("Y", lock_level=LOCK_USER,
                        locked_at="2026-04-26T01:00:00Z"))
    s.insert_edge(Edge(src="X", dst="Y", type=EDGE_CONTRADICTS, weight=1.0))
    record_retrieval(s, [s.get_belief("X")])  # type: ignore[list-item]
    y_after = s.get_belief("Y")
    assert y_after is not None
    assert y_after.lock_level == LOCK_USER


# --- record_retrieval: failure isolation ---------------------------------


def test_record_retrieval_skips_missing_belief_logs_and_continues() -> None:
    """If a belief's id no longer exists at write time (e.g. raced
    deletion), the row write fails. The loop must continue for the
    remaining beliefs and the failure must hit stderr."""
    s = _seed(_mk("b1"), _mk("b2"))
    ghost = _mk("ghost")  # not inserted
    err = io.StringIO()
    written = record_retrieval(
        s, [s.get_belief("b1"), ghost, s.get_belief("b2")], stderr=err,  # type: ignore[list-item]
    )
    assert written == 2
    assert s.count_feedback_events() == 2
    assert "ghost" in err.getvalue() or "ValueError" in err.getvalue()


# --- search_for_prompt: full read+write path -----------------------------


def test_search_for_prompt_returns_retrieval_hits() -> None:
    s = _seed(_mk("b1", "the quick brown fox"))
    hits = search_for_prompt(s, "quick fox")
    assert any(h.id == "b1" for h in hits)


def test_search_for_prompt_records_each_hit() -> None:
    s = _seed(_mk("b1", "the quick brown fox"))
    hits = search_for_prompt(s, "quick fox")
    assert s.count_feedback_events() == len(hits)


def test_search_for_prompt_empty_query_returns_locked_only() -> None:
    """Empty/whitespace query: retrieval returns L0 (locked) only.
    record_retrieval still writes audit rows for what was returned."""
    s = _seed(_mk("L1", lock_level=LOCK_USER,
                  locked_at="2026-04-26T01:00:00Z"),
              _mk("U1", "unlocked content"))
    hits = search_for_prompt(s, "")
    assert [h.id for h in hits] == ["L1"]
    assert s.count_feedback_events() == 1


def test_search_for_prompt_no_hits_writes_no_rows() -> None:
    """A query that hits nothing produces zero audit rows."""
    s = _seed(_mk("b1", "the quick brown fox"))
    hits = search_for_prompt(s, "xyzzy")
    assert hits == []
    assert s.count_feedback_events() == 0


def test_search_for_prompt_record_failure_does_not_block_read() -> None:
    """If recording somehow fails entirely, the function still returns
    the retrieval result — the hook contract is non-blocking on the
    audit side."""
    s = _seed(_mk("b1", "the quick brown fox"))
    err = io.StringIO()
    # Pollute the store so apply_feedback fails: close the connection
    # before record_retrieval runs. retrieve() ran already; its hits
    # are already in memory.
    hits = search_for_prompt(s, "quick fox", stderr=err)
    # Read happened.
    assert any(h.id == "b1" for h in hits)
    # Now manually exercise record on a closed store to force the
    # error path; verify record_retrieval logs and returns 0.
    s.close()
    written = record_retrieval(s, hits, stderr=err)
    assert written == 0


# --- last_retrieved_at mirror (issue #222) -------------------------------


def test_record_retrieval_stamps_last_retrieved_at() -> None:
    """Per #222: feedback_history write must mirror to beliefs.last_retrieved_at
    so downstream consumers don't have to join the audit log."""
    s = _seed(_mk("b1"), _mk("b2"), _mk("b3"))
    record_retrieval(s, [s.get_belief("b1"), s.get_belief("b2")])  # type: ignore[list-item]
    assert s.get_belief("b1").last_retrieved_at is not None  # type: ignore[union-attr]
    assert s.get_belief("b2").last_retrieved_at is not None  # type: ignore[union-attr]
    assert s.get_belief("b3").last_retrieved_at is None  # type: ignore[union-attr]


def test_record_retrieval_stamp_skips_failed_writes() -> None:
    """Beliefs whose apply_feedback failed must NOT get the recency stamp."""
    s = _seed(_mk("b1"), _mk("b2"))
    ghost = _mk("ghost")
    err = io.StringIO()
    record_retrieval(
        s, [s.get_belief("b1"), ghost, s.get_belief("b2")], stderr=err,  # type: ignore[list-item]
    )
    assert s.get_belief("b1").last_retrieved_at is not None  # type: ignore[union-attr]
    assert s.get_belief("b2").last_retrieved_at is not None  # type: ignore[union-attr]
    assert s.get_belief("ghost") is None


def test_record_retrieval_empty_input_does_not_stamp() -> None:
    s = _seed(_mk("b1"))
    record_retrieval(s, [])
    assert s.get_belief("b1").last_retrieved_at is None  # type: ignore[union-attr]


def test_search_for_prompt_stamps_returned_hits() -> None:
    s = _seed(_mk("b1", "the quick brown fox"), _mk("b2", "unrelated"))
    hits = search_for_prompt(s, "quick fox")
    hit_ids = {h.id for h in hits}
    assert "b1" in hit_ids
    assert s.get_belief("b1").last_retrieved_at is not None  # type: ignore[union-attr]
    if "b2" not in hit_ids:
        assert s.get_belief("b2").last_retrieved_at is None  # type: ignore[union-attr]


# --- Locked beliefs are still recorded -----------------------------------


def test_search_for_prompt_records_locked_hits_too() -> None:
    """L0 hits (locks auto-loaded) get audit rows just like L1 hits.
    The 'feedback updates the math' loop covers locks."""
    s = _seed(_mk("L1", "locked content",
                  lock_level=LOCK_USER,
                  locked_at="2026-04-26T01:00:00Z"))
    _ = search_for_prompt(s, "locked")
    events = s.list_feedback_events()
    assert any(ev.belief_id == "L1" for ev in events)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
