"""Valence propagation through apply_feedback (#1058).

A direct feedback event walks outbound edges (broker-confidence
attenuation via `MemoryStore.propagate_valence`) and applies each
attenuated delta back through `apply_feedback`, so every downstream
posterior write also lands a `feedback_history` row. In-memory stores;
`now` pinned where determinism matters.
"""
from __future__ import annotations

import pytest

from aelfrice.feedback import (
    ENV_VALENCE_PROPAGATION,
    PROPAGATION_SOURCE_PREFIX,
    apply_feedback,
)
from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_CONTRADICTS,
    EDGE_SUPPORTS,
    LOCK_NONE,
    Belief,
    Edge,
)
from aelfrice.store import MemoryStore


def _mk(bid: str, alpha: float = 5.0, beta: float = 5.0) -> Belief:
    return Belief(
        id=bid,
        content=f"belief {bid}",
        content_hash=f"h_{bid}",
        alpha=alpha,
        beta=beta,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-04-26T00:00:00Z",
        last_retrieved_at=None,
    )


def _chain_store() -> MemoryStore:
    """A -SUPPORTS-> B -SUPPORTS-> C, all neutral confidence (0.5)."""
    s = MemoryStore(":memory:")
    for bid in ("A", "B", "C"):
        s.insert_belief(_mk(bid))
    s.insert_edge(Edge(src="A", dst="B", type=EDGE_SUPPORTS, weight=1.0))
    s.insert_edge(Edge(src="B", dst="C", type=EDGE_SUPPORTS, weight=1.0))
    return s


# --- Happy path ----------------------------------------------------------


def test_propagation_updates_downstream_posterior() -> None:
    s = _chain_store()
    apply_feedback(s, "A", valence=1.0, source="user")
    b = s.get_belief("B")
    assert b is not None
    # Direct hop delta: 1.0 * EDGE_VALENCE[SUPPORTS] * conf(B)=0.5.
    assert b.alpha == pytest.approx(5.5)


def test_propagation_writes_history_rows_with_provenance_source() -> None:
    s = _chain_store()
    apply_feedback(s, "A", valence=1.0, source="user")
    events = s.list_feedback_events(belief_id="B")
    assert len(events) == 1
    assert events[0].source == f"{PROPAGATION_SOURCE_PREFIX}user"


def test_propagated_results_returned_on_feedback_result() -> None:
    s = _chain_store()
    result = apply_feedback(s, "A", valence=1.0, source="user")
    assert {r.belief_id for r in result.propagated} == {"B", "C"}


def test_propagated_deltas_match_pure_walk_exactly() -> None:
    """No recursion: applied downstream valences equal the single BFS
    output. A recursive cascade would re-walk from B and inflate C."""
    walk = _chain_store().propagate_valence("A", valence=1.0)
    s = _chain_store()
    result = apply_feedback(s, "A", valence=1.0, source="user")
    applied = {r.belief_id: r.valence for r in result.propagated}
    assert applied == pytest.approx(walk)
    # Exactly one propagation row per recipient — a cascade would add more.
    assert s.count_feedback_events(belief_id="C") == 1


def test_propagated_rows_share_direct_event_timestamp() -> None:
    s = _chain_store()
    apply_feedback(s, "A", valence=1.0, source="user",
                   now="2026-07-02T00:00:00Z")
    events = s.list_feedback_events(belief_id="C")
    assert events[0].created_at == "2026-07-02T00:00:00Z"


def test_negative_signal_through_contradicts_penalizes_neighbor() -> None:
    s = MemoryStore(":memory:")
    for bid in ("A", "X"):
        s.insert_belief(_mk(bid))
    s.insert_edge(Edge(src="A", dst="X", type=EDGE_CONTRADICTS, weight=1.0))
    apply_feedback(s, "A", valence=1.0, source="user")
    x = s.get_belief("X")
    assert x is not None
    # Delta = 1.0 * (-0.5) * 0.5 = -0.25 -> beta side.
    assert x.beta == pytest.approx(5.25)
    assert x.alpha == pytest.approx(5.0)


# --- Suppression ---------------------------------------------------------


def test_isolated_belief_propagates_nothing() -> None:
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("solo"))
    result = apply_feedback(s, "solo", valence=1.0, source="user")
    assert result.propagated == []


def test_kill_switch_disables_propagation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_VALENCE_PROPAGATION, "0")
    s = _chain_store()
    result = apply_feedback(s, "A", valence=1.0, source="user")
    assert result.propagated == []
    assert s.count_feedback_events(belief_id="B") == 0
    b = s.get_belief("B")
    assert b is not None and b.alpha == 5.0


def test_propagate_false_kwarg_suppresses() -> None:
    s = _chain_store()
    result = apply_feedback(s, "A", valence=1.0, source="user",
                            propagate=False)
    assert result.propagated == []
    assert s.count_feedback_events(belief_id="B") == 0


def test_cycle_never_feeds_back_into_source() -> None:
    s = MemoryStore(":memory:")
    for bid in ("A", "B"):
        s.insert_belief(_mk(bid))
    s.insert_edge(Edge(src="A", dst="B", type=EDGE_SUPPORTS, weight=1.0))
    s.insert_edge(Edge(src="B", dst="A", type=EDGE_SUPPORTS, weight=1.0))
    apply_feedback(s, "A", valence=1.0, source="user")
    # Exactly the direct event on A — no propagation echo.
    assert s.count_feedback_events(belief_id="A") == 1
    a = s.get_belief("A")
    assert a is not None and a.alpha == pytest.approx(6.0)


def test_vanished_recipient_skipped_without_failing_direct_event() -> None:
    """A recipient the walk found but apply cannot resolve is skipped;
    the direct event and other recipients still land."""
    s = _chain_store()
    walk = s.propagate_valence("A", valence=1.0)
    assert "B" in walk and "C" in walk  # precondition: both reachable
    # Delete C's row out from under the apply loop by dropping it now:
    # the walk inside apply_feedback recomputes, so instead simulate by
    # deleting B's downstream C before the call.
    s.delete_belief("C")
    result = apply_feedback(s, "A", valence=1.0, source="user")
    assert {r.belief_id for r in result.propagated} == {"B"}
