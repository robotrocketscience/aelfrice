"""Demotion-pressure increment on CONTRADICTS-edge propagation to a lock.

When apply_feedback fires on a belief X with a CONTRADICTS edge to a
user-locked belief Y, Y's demotion_pressure increments by 1. This is the
mechanism that closes the v2.0 write-only-bug for E22 — the pressure now
both writes (here) and reads (in the auto-demote path landing next).

Edge-type filter: only CONTRADICTS edges fire pressure. Lock filter: only
user-locked targets accumulate. Valence filter: only positive valence on
the source fires (positive signal on a contradictor weakens the contradicted
lock; negative signal on a contradictor weakens the contradictor itself).
"""
from __future__ import annotations

from aelfrice.feedback import apply_feedback
from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_CITES,
    EDGE_CONTRADICTS,
    EDGE_RELATES_TO,
    EDGE_SUPERSEDES,
    EDGE_SUPPORTS,
    LOCK_NONE,
    LOCK_USER,
    Belief,
    Edge,
)
from aelfrice.store import MemoryStore


def _mk(
    bid: str,
    lock_level: str = LOCK_NONE,
    locked_at: str | None = None,
    demotion_pressure: int = 0,
) -> Belief:
    return Belief(
        id=bid,
        content=f"belief {bid}",
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=lock_level,
        locked_at=locked_at,
        demotion_pressure=demotion_pressure,
        created_at="2026-04-26T00:00:00Z",
        last_retrieved_at=None,
    )


def _build(source_id: str, target_id: str, edge_type: str,
           target_lock: str = LOCK_USER) -> MemoryStore:
    s = MemoryStore(":memory:")
    s.insert_belief(_mk(source_id))
    s.insert_belief(_mk(target_id, lock_level=target_lock,
                        locked_at="2026-04-26T01:00:00Z"))
    s.insert_edge(Edge(src=source_id, dst=target_id, type=edge_type, weight=1.0))
    return s


def _pressure(store: MemoryStore, bid: str) -> int:
    got = store.get_belief(bid)
    assert got is not None
    return got.demotion_pressure


# --- Edge-type filter (CONTRADICTS only) ---------------------------------


def test_positive_feedback_on_contradictor_increments_locked_pressure() -> None:
    s = _build("X", "Y", EDGE_CONTRADICTS)
    apply_feedback(s, "X", valence=1.0, source="user")
    assert _pressure(s, "Y") == 1


def test_supports_edge_does_not_increment_pressure() -> None:
    s = _build("X", "Y", EDGE_SUPPORTS)
    apply_feedback(s, "X", valence=1.0, source="user")
    assert _pressure(s, "Y") == 0


def test_cites_edge_does_not_increment_pressure() -> None:
    s = _build("X", "Y", EDGE_CITES)
    apply_feedback(s, "X", valence=1.0, source="user")
    assert _pressure(s, "Y") == 0


def test_supersedes_edge_does_not_increment_pressure() -> None:
    s = _build("X", "Y", EDGE_SUPERSEDES)
    apply_feedback(s, "X", valence=1.0, source="user")
    assert _pressure(s, "Y") == 0


def test_relates_to_edge_does_not_increment_pressure() -> None:
    s = _build("X", "Y", EDGE_RELATES_TO)
    apply_feedback(s, "X", valence=1.0, source="user")
    assert _pressure(s, "Y") == 0


# --- Lock filter (user-locked targets only) ------------------------------


def test_unlocked_target_does_not_accumulate_pressure() -> None:
    s = _build("X", "Y", EDGE_CONTRADICTS, target_lock=LOCK_NONE)
    apply_feedback(s, "X", valence=1.0, source="user")
    assert _pressure(s, "Y") == 0


# --- Valence filter (positive only) --------------------------------------


def test_negative_valence_does_not_increment_pressure() -> None:
    s = _build("X", "Y", EDGE_CONTRADICTS)
    apply_feedback(s, "X", valence=-1.0, source="user")
    assert _pressure(s, "Y") == 0


# --- Direction (outbound from source) ------------------------------------


def test_inbound_contradicts_edge_does_not_increment_pressure() -> None:
    """Edge Y->X CONTRADICTS means Y contradicts X. Feedback on X should
    not pressure Y under the 1-hop outbound semantics."""
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("X"))
    s.insert_belief(_mk("Y", lock_level=LOCK_USER,
                        locked_at="2026-04-26T01:00:00Z"))
    s.insert_edge(Edge(src="Y", dst="X", type=EDGE_CONTRADICTS, weight=1.0))
    apply_feedback(s, "X", valence=1.0, source="user")
    assert _pressure(s, "Y") == 0


# --- Accumulation --------------------------------------------------------


def test_pressure_accumulates_across_repeated_positive_calls() -> None:
    s = _build("X", "Y", EDGE_CONTRADICTS)
    apply_feedback(s, "X", valence=1.0, source="user")
    apply_feedback(s, "X", valence=1.0, source="user")
    apply_feedback(s, "X", valence=1.0, source="user")
    assert _pressure(s, "Y") == 3


def test_pressure_starts_from_existing_value_not_zero() -> None:
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("X"))
    s.insert_belief(_mk("Y", lock_level=LOCK_USER,
                        locked_at="2026-04-26T01:00:00Z",
                        demotion_pressure=2))
    s.insert_edge(Edge(src="X", dst="Y", type=EDGE_CONTRADICTS, weight=1.0))
    apply_feedback(s, "X", valence=1.0, source="user")
    assert _pressure(s, "Y") == 3


# --- Multi-target --------------------------------------------------------


def test_two_locked_contradicted_targets_both_pressured() -> None:
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("X"))
    s.insert_belief(_mk("Y1", lock_level=LOCK_USER,
                        locked_at="2026-04-26T01:00:00Z"))
    s.insert_belief(_mk("Y2", lock_level=LOCK_USER,
                        locked_at="2026-04-26T02:00:00Z"))
    s.insert_edge(Edge(src="X", dst="Y1", type=EDGE_CONTRADICTS, weight=1.0))
    s.insert_edge(Edge(src="X", dst="Y2", type=EDGE_CONTRADICTS, weight=1.0))
    apply_feedback(s, "X", valence=1.0, source="user")
    assert _pressure(s, "Y1") == 1
    assert _pressure(s, "Y2") == 1


def test_mix_of_locked_and_unlocked_targets_only_locked_pressured() -> None:
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("X"))
    s.insert_belief(_mk("Y_locked", lock_level=LOCK_USER,
                        locked_at="2026-04-26T01:00:00Z"))
    s.insert_belief(_mk("Y_open"))
    s.insert_edge(Edge(src="X", dst="Y_locked", type=EDGE_CONTRADICTS,
                       weight=1.0))
    s.insert_edge(Edge(src="X", dst="Y_open", type=EDGE_CONTRADICTS,
                       weight=1.0))
    apply_feedback(s, "X", valence=1.0, source="user")
    assert _pressure(s, "Y_locked") == 1
    assert _pressure(s, "Y_open") == 0


# --- Result object reports pressured_locks -------------------------------


def test_result_reports_single_pressured_lock() -> None:
    s = _build("X", "Y", EDGE_CONTRADICTS)
    result = apply_feedback(s, "X", valence=1.0, source="user")
    assert result.pressured_locks == ["Y"]


def test_result_reports_no_pressured_locks_when_no_contradiction() -> None:
    s = _build("X", "Y", EDGE_SUPPORTS)
    result = apply_feedback(s, "X", valence=1.0, source="user")
    assert result.pressured_locks == []


def test_result_reports_multiple_pressured_locks_sorted() -> None:
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("X"))
    s.insert_belief(_mk("Yb", lock_level=LOCK_USER,
                        locked_at="2026-04-26T01:00:00Z"))
    s.insert_belief(_mk("Ya", lock_level=LOCK_USER,
                        locked_at="2026-04-26T02:00:00Z"))
    s.insert_edge(Edge(src="X", dst="Yb", type=EDGE_CONTRADICTS, weight=1.0))
    s.insert_edge(Edge(src="X", dst="Ya", type=EDGE_CONTRADICTS, weight=1.0))
    result = apply_feedback(s, "X", valence=1.0, source="user")
    assert result.pressured_locks == ["Ya", "Yb"]


def test_negative_valence_result_pressured_locks_empty() -> None:
    s = _build("X", "Y", EDGE_CONTRADICTS)
    result = apply_feedback(s, "X", valence=-1.0, source="user")
    assert result.pressured_locks == []


# --- propagate=False suppresses the walk (non-corrective signals) --------


def test_propagate_false_does_not_increment_pressure() -> None:
    """Hook-driven retrievals pass propagate=False so that implicit
    exposure does not pressure-walk locked beliefs on every prompt."""
    s = _build("X", "Y", EDGE_CONTRADICTS)
    apply_feedback(s, "X", valence=1.0, source="hook", propagate=False)
    assert _pressure(s, "Y") == 0


def test_propagate_false_still_updates_posterior() -> None:
    """propagate=False suppresses the walk, not the Bayesian update."""
    s = _build("X", "Y", EDGE_CONTRADICTS)
    result = apply_feedback(s, "X", valence=0.5, source="hook",
                            propagate=False)
    assert result.new_alpha == 1.5
    assert result.new_beta == 1.0


def test_propagate_false_still_writes_audit_row() -> None:
    s = _build("X", "Y", EDGE_CONTRADICTS)
    result = apply_feedback(s, "X", valence=1.0, source="hook",
                            propagate=False)
    assert result.event_id > 0
    assert s.count_feedback_events() == 1


def test_propagate_false_result_pressured_locks_empty() -> None:
    s = _build("X", "Y", EDGE_CONTRADICTS)
    result = apply_feedback(s, "X", valence=1.0, source="hook",
                            propagate=False)
    assert result.pressured_locks == []
    assert result.demoted_locks == []


def test_propagate_default_true_preserves_corrective_walk() -> None:
    """Existing callers without the kwarg get the same pressure walk."""
    s = _build("X", "Y", EDGE_CONTRADICTS)
    apply_feedback(s, "X", valence=1.0, source="user")
    assert _pressure(s, "Y") == 1
