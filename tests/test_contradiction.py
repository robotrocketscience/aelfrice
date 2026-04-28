"""Contradiction tie-breaker: precedence, resolution, audit, idempotence.

Atomic short tests, one property each. Every test uses an in-memory
store and finishes in milliseconds.
"""
from __future__ import annotations

import pytest

from aelfrice.contradiction import (
    CLASS_NAMES,
    PRECEDENCE_AGENT_INFERRED,
    PRECEDENCE_DOCUMENT_RECENT,
    PRECEDENCE_USER_CORRECTED,
    PRECEDENCE_USER_STATED,
    PRECEDENCE_USER_VALIDATED,
    SOURCE_PREFIX,
    auto_resolve_all_contradictions,
    find_unresolved_contradictions,
    precedence_class,
    precedence_class_name,
    resolve_contradiction,
)
from aelfrice.models import (
    BELIEF_CORRECTION,
    BELIEF_FACTUAL,
    EDGE_CONTRADICTS,
    EDGE_SUPERSEDES,
    LOCK_NONE,
    LOCK_USER,
    ORIGIN_AGENT_INFERRED,
    ORIGIN_USER_VALIDATED,
    Belief,
    Edge,
)
from aelfrice.store import MemoryStore


def _mk(
    bid: str,
    *,
    btype: str = BELIEF_FACTUAL,
    lock: str = LOCK_NONE,
    locked_at: str | None = None,
    created_at: str = "2026-04-26T00:00:00Z",
    origin: str = "unknown",
) -> Belief:
    return Belief(
        id=bid,
        content=f"belief {bid}",
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=btype,
        lock_level=lock,
        locked_at=locked_at,
        demotion_pressure=0,
        created_at=created_at,
        last_retrieved_at=None,
        origin=origin,
    )


def _seed(*beliefs: Belief) -> MemoryStore:
    s = MemoryStore(":memory:")
    for b in beliefs:
        s.insert_belief(b)
    return s


# --- Precedence class assignment ----------------------------------------


def test_user_locked_belief_is_user_stated() -> None:
    b = _mk("X", lock=LOCK_USER, locked_at="2026-04-26T01:00:00Z")
    assert precedence_class(b) == PRECEDENCE_USER_STATED
    assert precedence_class_name(b) == "user_stated"


def test_correction_belief_is_user_corrected() -> None:
    b = _mk("X", btype=BELIEF_CORRECTION)
    assert precedence_class(b) == PRECEDENCE_USER_CORRECTED
    assert precedence_class_name(b) == "user_corrected"


def test_factual_belief_is_document_recent() -> None:
    b = _mk("X")
    assert precedence_class(b) == PRECEDENCE_DOCUMENT_RECENT
    assert precedence_class_name(b) == "document_recent"


def test_locked_correction_is_user_stated_not_corrected() -> None:
    """Lock takes priority over type — a locked correction is still
    user-asserted ground truth."""
    b = _mk("X", btype=BELIEF_CORRECTION, lock=LOCK_USER,
            locked_at="2026-04-26T01:00:00Z")
    assert precedence_class(b) == PRECEDENCE_USER_STATED


def test_user_validated_origin_maps_to_user_validated_class() -> None:
    b = _mk("X", origin=ORIGIN_USER_VALIDATED)
    assert precedence_class(b) == PRECEDENCE_USER_VALIDATED
    assert precedence_class_name(b) == "user_validated"


def test_agent_inferred_origin_maps_to_agent_inferred_class() -> None:
    b = _mk("X", origin=ORIGIN_AGENT_INFERRED)
    assert precedence_class(b) == PRECEDENCE_AGENT_INFERRED
    assert precedence_class_name(b) == "agent_inferred"


def test_unknown_origin_falls_through_to_document_recent() -> None:
    """v1.0/v1.1 absorption — unknown is treated as document_recent."""
    b = _mk("X", origin="unknown")
    assert precedence_class(b) == PRECEDENCE_DOCUMENT_RECENT


def test_lock_short_circuits_user_validated_origin() -> None:
    """A locked belief tagged user_validated still resolves as user_stated.
    Locks always win regardless of the origin string."""
    b = _mk("X", lock=LOCK_USER, locked_at="2026-04-26T01:00:00Z",
            origin=ORIGIN_USER_VALIDATED)
    assert precedence_class(b) == PRECEDENCE_USER_STATED


# --- _pick_winner via resolve_contradiction -----------------------------


def test_user_stated_beats_user_corrected() -> None:
    a = _mk("A", lock=LOCK_USER, locked_at="2026-04-26T01:00:00Z")
    b = _mk("B", btype=BELIEF_CORRECTION)
    s = _seed(a, b)
    s.insert_edge(Edge(src="A", dst="B", type=EDGE_CONTRADICTS, weight=1.0))
    result = resolve_contradiction(s, "A", "B")
    assert result.winner_id == "A"
    assert result.loser_id == "B"
    assert result.rule_fired == "user_stated_beats_user_corrected"


def test_user_stated_beats_document_recent() -> None:
    a = _mk("A", lock=LOCK_USER, locked_at="2026-04-26T01:00:00Z")
    b = _mk("B")  # factual, default class
    s = _seed(a, b)
    s.insert_edge(Edge(src="A", dst="B", type=EDGE_CONTRADICTS, weight=1.0))
    result = resolve_contradiction(s, "A", "B")
    assert result.winner_id == "A"
    assert result.rule_fired == "user_stated_beats_document_recent"


def test_user_corrected_beats_document_recent() -> None:
    a = _mk("A", btype=BELIEF_CORRECTION)
    b = _mk("B")
    s = _seed(a, b)
    s.insert_edge(Edge(src="A", dst="B", type=EDGE_CONTRADICTS, weight=1.0))
    result = resolve_contradiction(s, "A", "B")
    assert result.winner_id == "A"
    assert result.rule_fired == "user_corrected_beats_document_recent"


def test_user_corrected_beats_user_validated() -> None:
    a = _mk("A", btype=BELIEF_CORRECTION)
    b = _mk("B", origin=ORIGIN_USER_VALIDATED)
    s = _seed(a, b)
    s.insert_edge(Edge(src="A", dst="B", type=EDGE_CONTRADICTS, weight=1.0))
    result = resolve_contradiction(s, "A", "B")
    assert result.winner_id == "A"
    assert result.rule_fired == "user_corrected_beats_user_validated"


def test_user_validated_beats_document_recent() -> None:
    a = _mk("A", origin=ORIGIN_USER_VALIDATED)
    b = _mk("B")  # origin=unknown -> document_recent
    s = _seed(a, b)
    s.insert_edge(Edge(src="A", dst="B", type=EDGE_CONTRADICTS, weight=1.0))
    result = resolve_contradiction(s, "A", "B")
    assert result.winner_id == "A"
    assert result.rule_fired == "user_validated_beats_document_recent"


def test_user_validated_beats_agent_inferred() -> None:
    a = _mk("A", origin=ORIGIN_USER_VALIDATED)
    b = _mk("B", origin=ORIGIN_AGENT_INFERRED)
    s = _seed(a, b)
    s.insert_edge(Edge(src="A", dst="B", type=EDGE_CONTRADICTS, weight=1.0))
    result = resolve_contradiction(s, "A", "B")
    assert result.winner_id == "A"
    assert result.rule_fired == "user_validated_beats_agent_inferred"


def test_document_recent_beats_agent_inferred() -> None:
    a = _mk("A")  # unknown -> document_recent
    b = _mk("B", origin=ORIGIN_AGENT_INFERRED)
    s = _seed(a, b)
    s.insert_edge(Edge(src="A", dst="B", type=EDGE_CONTRADICTS, weight=1.0))
    result = resolve_contradiction(s, "A", "B")
    assert result.winner_id == "A"
    assert result.rule_fired == "document_recent_beats_agent_inferred"


def test_argument_order_does_not_affect_outcome() -> None:
    """Calling resolve(A, B) must produce the same winner as resolve(B, A)."""
    a = _mk("A", lock=LOCK_USER, locked_at="2026-04-26T01:00:00Z")
    b = _mk("B")
    s = _seed(a, b)
    s.insert_edge(Edge(src="A", dst="B", type=EDGE_CONTRADICTS, weight=1.0))
    forward = resolve_contradiction(s, "A", "B")
    s2 = _seed(a, b)
    s2.insert_edge(Edge(src="A", dst="B", type=EDGE_CONTRADICTS, weight=1.0))
    reverse = resolve_contradiction(s2, "B", "A")
    assert forward.winner_id == reverse.winner_id
    assert forward.rule_fired == reverse.rule_fired


# --- Same-class tie-breaks ----------------------------------------------


def test_same_class_more_recent_wins() -> None:
    a = _mk("A", created_at="2026-04-26T00:00:00Z")
    b = _mk("B", created_at="2026-04-27T00:00:00Z")  # newer
    s = _seed(a, b)
    s.insert_edge(Edge(src="A", dst="B", type=EDGE_CONTRADICTS, weight=1.0))
    result = resolve_contradiction(s, "A", "B")
    assert result.winner_id == "B"
    assert "by_recency" in result.rule_fired


def test_same_class_same_timestamp_higher_id_wins() -> None:
    """Final tiebreak: alphabetical id ordering (deterministic)."""
    a = _mk("A", created_at="2026-04-26T00:00:00Z")
    b = _mk("B", created_at="2026-04-26T00:00:00Z")
    s = _seed(a, b)
    s.insert_edge(Edge(src="A", dst="B", type=EDGE_CONTRADICTS, weight=1.0))
    result = resolve_contradiction(s, "A", "B")
    assert result.winner_id == "B"
    assert "by_id" in result.rule_fired


def test_same_class_same_timestamp_id_winner_is_deterministic() -> None:
    """Re-running the same resolution must produce identical results."""
    a = _mk("A", created_at="2026-04-26T00:00:00Z")
    b = _mk("B", created_at="2026-04-26T00:00:00Z")
    s1 = _seed(a, b)
    s1.insert_edge(Edge(src="A", dst="B", type=EDGE_CONTRADICTS, weight=1.0))
    r1 = resolve_contradiction(s1, "A", "B")
    s2 = _seed(a, b)
    s2.insert_edge(Edge(src="A", dst="B", type=EDGE_CONTRADICTS, weight=1.0))
    r2 = resolve_contradiction(s2, "A", "B")
    assert r1.winner_id == r2.winner_id
    assert r1.rule_fired == r2.rule_fired


# --- SUPERSEDES edge written --------------------------------------------


def test_resolve_inserts_supersedes_edge() -> None:
    a = _mk("A", lock=LOCK_USER, locked_at="2026-04-26T01:00:00Z")
    b = _mk("B")
    s = _seed(a, b)
    s.insert_edge(Edge(src="A", dst="B", type=EDGE_CONTRADICTS, weight=1.0))
    result = resolve_contradiction(s, "A", "B")
    assert result.supersedes_created is True
    edge = s.get_edge(result.winner_id, result.loser_id, EDGE_SUPERSEDES)
    assert edge is not None


def test_resolve_idempotent_no_duplicate_supersedes() -> None:
    a = _mk("A", lock=LOCK_USER, locked_at="2026-04-26T01:00:00Z")
    b = _mk("B")
    s = _seed(a, b)
    s.insert_edge(Edge(src="A", dst="B", type=EDGE_CONTRADICTS, weight=1.0))
    first = resolve_contradiction(s, "A", "B")
    second = resolve_contradiction(s, "A", "B")
    assert first.supersedes_created is True
    assert second.supersedes_created is False
    assert first.winner_id == second.winner_id


# --- Audit row ----------------------------------------------------------


def test_resolve_writes_audit_row_on_loser() -> None:
    a = _mk("A", lock=LOCK_USER, locked_at="2026-04-26T01:00:00Z")
    b = _mk("B")
    s = _seed(a, b)
    s.insert_edge(Edge(src="A", dst="B", type=EDGE_CONTRADICTS, weight=1.0))
    resolve_contradiction(s, "A", "B")
    events = s.list_feedback_events()
    assert len(events) == 1
    ev = events[0]
    assert ev.belief_id == "B"
    assert ev.valence == 0.0
    assert ev.source.startswith(f"{SOURCE_PREFIX}:")
    assert "user_stated_beats_document_recent" in ev.source


def test_resolve_idempotent_writes_second_audit_row() -> None:
    """Re-resolving the same pair leaves a fresh audit row each time —
    so re-runs are visible in the log."""
    a = _mk("A", lock=LOCK_USER, locked_at="2026-04-26T01:00:00Z")
    b = _mk("B")
    s = _seed(a, b)
    s.insert_edge(Edge(src="A", dst="B", type=EDGE_CONTRADICTS, weight=1.0))
    resolve_contradiction(s, "A", "B")
    resolve_contradiction(s, "A", "B")
    assert s.count_feedback_events() == 2


def test_resolve_audit_row_carries_now_kwarg() -> None:
    a = _mk("A", lock=LOCK_USER, locked_at="2026-04-26T01:00:00Z")
    b = _mk("B")
    s = _seed(a, b)
    s.insert_edge(Edge(src="A", dst="B", type=EDGE_CONTRADICTS, weight=1.0))
    resolve_contradiction(s, "A", "B", now="2026-04-30T12:00:00Z")
    ev = s.list_feedback_events()[0]
    assert ev.created_at == "2026-04-30T12:00:00Z"


def test_resolve_audit_row_zero_valence_does_not_affect_replay() -> None:
    """The audit row must have valence=0.0 so a future feedback-replay
    pass doesn't treat the tie-breaker as a real Bayesian update."""
    a = _mk("A", lock=LOCK_USER, locked_at="2026-04-26T01:00:00Z")
    b = _mk("B")
    s = _seed(a, b)
    s.insert_edge(Edge(src="A", dst="B", type=EDGE_CONTRADICTS, weight=1.0))
    resolve_contradiction(s, "A", "B")
    events = s.list_feedback_events()
    assert all(ev.valence == 0.0 for ev in events
                if ev.source.startswith(f"{SOURCE_PREFIX}:"))


# --- Missing belief errors ----------------------------------------------


def test_resolve_raises_on_missing_a() -> None:
    s = _seed(_mk("B"))
    with pytest.raises(ValueError, match="ghost"):
        resolve_contradiction(s, "ghost", "B")


def test_resolve_raises_on_missing_b() -> None:
    s = _seed(_mk("A"))
    with pytest.raises(ValueError, match="ghost"):
        resolve_contradiction(s, "A", "ghost")


# --- find_unresolved_contradictions -------------------------------------


def test_find_no_contradicts_returns_empty() -> None:
    s = _seed(_mk("A"), _mk("B"))
    assert find_unresolved_contradictions(s) == []


def test_find_one_contradicts_returns_pair() -> None:
    s = _seed(_mk("A"), _mk("B"))
    s.insert_edge(Edge(src="A", dst="B", type=EDGE_CONTRADICTS, weight=1.0))
    pairs = find_unresolved_contradictions(s)
    assert pairs == [("A", "B")]


def test_find_canonicalises_direction() -> None:
    """B→A CONTRADICTS is the same logical pair as A→B."""
    s = _seed(_mk("A"), _mk("B"))
    s.insert_edge(Edge(src="B", dst="A", type=EDGE_CONTRADICTS, weight=1.0))
    pairs = find_unresolved_contradictions(s)
    assert pairs == [("A", "B")]  # canonical: lower id first


def test_find_dedupes_both_directions() -> None:
    """A→B and B→A together should still yield one pair."""
    s = _seed(_mk("A"), _mk("B"))
    s.insert_edge(Edge(src="A", dst="B", type=EDGE_CONTRADICTS, weight=1.0))
    s.insert_edge(Edge(src="B", dst="A", type=EDGE_CONTRADICTS, weight=1.0))
    pairs = find_unresolved_contradictions(s)
    assert len(pairs) == 1


def test_find_skips_already_resolved_pair() -> None:
    """A pair with an existing SUPERSEDES edge is not unresolved."""
    s = _seed(_mk("A"), _mk("B"))
    s.insert_edge(Edge(src="A", dst="B", type=EDGE_CONTRADICTS, weight=1.0))
    s.insert_edge(Edge(src="A", dst="B", type=EDGE_SUPERSEDES, weight=1.0))
    pairs = find_unresolved_contradictions(s)
    assert pairs == []


# --- auto_resolve_all_contradictions ------------------------------------


def test_auto_resolve_processes_all_pairs() -> None:
    s = _seed(
        _mk("A", lock=LOCK_USER, locked_at="2026-04-26T01:00:00Z"),
        _mk("B"),
        _mk("C", lock=LOCK_USER, locked_at="2026-04-26T02:00:00Z"),
        _mk("D"),
    )
    s.insert_edge(Edge(src="A", dst="B", type=EDGE_CONTRADICTS, weight=1.0))
    s.insert_edge(Edge(src="C", dst="D", type=EDGE_CONTRADICTS, weight=1.0))
    results = auto_resolve_all_contradictions(s)
    assert len(results) == 2
    winner_ids = {r.winner_id for r in results}
    assert winner_ids == {"A", "C"}


def test_auto_resolve_skips_already_resolved() -> None:
    s = _seed(
        _mk("A", lock=LOCK_USER, locked_at="2026-04-26T01:00:00Z"),
        _mk("B"),
    )
    s.insert_edge(Edge(src="A", dst="B", type=EDGE_CONTRADICTS, weight=1.0))
    s.insert_edge(Edge(src="A", dst="B", type=EDGE_SUPERSEDES, weight=1.0))
    results = auto_resolve_all_contradictions(s)
    assert results == []


def test_auto_resolve_skips_pair_with_missing_endpoint() -> None:
    """If a CONTRADICTS edge points at a deleted belief, skip don't
    crash."""
    s = _seed(_mk("A"))
    # Insert an edge pointing at a non-existent B.
    s.insert_edge(Edge(src="A", dst="ghost", type=EDGE_CONTRADICTS, weight=1.0))
    results = auto_resolve_all_contradictions(s)
    assert results == []


def test_auto_resolve_idempotent_second_run_no_op() -> None:
    s = _seed(
        _mk("A", lock=LOCK_USER, locked_at="2026-04-26T01:00:00Z"),
        _mk("B"),
    )
    s.insert_edge(Edge(src="A", dst="B", type=EDGE_CONTRADICTS, weight=1.0))
    first = auto_resolve_all_contradictions(s)
    second = auto_resolve_all_contradictions(s)
    assert len(first) == 1
    assert second == []  # all resolved; nothing to do


# --- Class-name constants -----------------------------------------------


def test_class_names_cover_all_three_classes() -> None:
    assert CLASS_NAMES[PRECEDENCE_USER_STATED] == "user_stated"
    assert CLASS_NAMES[PRECEDENCE_USER_CORRECTED] == "user_corrected"
    assert CLASS_NAMES[PRECEDENCE_DOCUMENT_RECENT] == "document_recent"


def test_source_prefix_is_stable() -> None:
    """Wire format. Renaming requires a migration."""
    assert SOURCE_PREFIX == "contradiction_tiebreaker"
