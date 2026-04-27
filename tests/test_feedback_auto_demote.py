"""Auto-demote: locked belief demotes when demotion_pressure >= threshold.

Closes the v2.0 demotion-write-only bug end-to-end (E22): pressure now
both writes and reads, and crossing the configurable threshold demotes
the lock to lock_level='none' with demotion_pressure reset to 0 so the
belief can be re-locked cleanly later.

DEMOTION_THRESHOLD is a module-level constant (default 5). Tests that
need a different threshold rebind it via monkeypatch.
"""
from __future__ import annotations

import pytest

from aelfrice import feedback as feedback_module
from aelfrice.feedback import apply_feedback
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


def _build(starting_pressure: int = 0) -> MemoryStore:
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("X"))
    s.insert_belief(_mk("Y", lock_level=LOCK_USER,
                        locked_at="2026-04-26T01:00:00Z",
                        demotion_pressure=starting_pressure))
    s.insert_edge(Edge(src="X", dst="Y", type=EDGE_CONTRADICTS, weight=1.0))
    return s


def _y(store: MemoryStore) -> Belief:
    got = store.get_belief("Y")
    assert got is not None
    return got


# --- Below threshold -----------------------------------------------------


def test_four_events_keeps_lock_level_user() -> None:
    s = _build()
    for _ in range(4):
        apply_feedback(s, "X", valence=1.0, source="user")
    assert _y(s).lock_level == LOCK_USER


def test_four_events_pressure_at_four() -> None:
    s = _build()
    for _ in range(4):
        apply_feedback(s, "X", valence=1.0, source="user")
    assert _y(s).demotion_pressure == 4


def test_four_events_locked_at_preserved() -> None:
    s = _build()
    for _ in range(4):
        apply_feedback(s, "X", valence=1.0, source="user")
    assert _y(s).locked_at == "2026-04-26T01:00:00Z"


# --- At threshold (default 5) --------------------------------------------


def test_five_events_demotes_lock_level_to_none() -> None:
    s = _build()
    for _ in range(5):
        apply_feedback(s, "X", valence=1.0, source="user")
    assert _y(s).lock_level == LOCK_NONE


def test_five_events_clears_locked_at() -> None:
    s = _build()
    for _ in range(5):
        apply_feedback(s, "X", valence=1.0, source="user")
    assert _y(s).locked_at is None


def test_five_events_resets_demotion_pressure_to_zero() -> None:
    s = _build()
    for _ in range(5):
        apply_feedback(s, "X", valence=1.0, source="user")
    assert _y(s).demotion_pressure == 0


def test_demoted_belief_still_exists_in_store() -> None:
    s = _build()
    for _ in range(5):
        apply_feedback(s, "X", valence=1.0, source="user")
    assert s.get_belief("Y") is not None


def test_demoted_belief_alpha_beta_unchanged_by_demotion() -> None:
    """Demote affects lock state, not posterior."""
    s = _build()
    pre = _y(s)
    pre_alpha, pre_beta = pre.alpha, pre.beta
    for _ in range(5):
        apply_feedback(s, "X", valence=1.0, source="user")
    post = _y(s)
    assert (post.alpha, post.beta) == (pre_alpha, pre_beta)


# --- Threshold from already-pressured starting state ---------------------


def test_one_event_demotes_when_starting_pressure_is_four() -> None:
    s = _build(starting_pressure=4)
    apply_feedback(s, "X", valence=1.0, source="user")
    assert _y(s).lock_level == LOCK_NONE


def test_zero_events_does_not_demote_even_at_starting_pressure_four() -> None:
    s = _build(starting_pressure=4)
    # No feedback fired -> still locked, still pressure 4.
    assert _y(s).lock_level == LOCK_USER
    assert _y(s).demotion_pressure == 4


# --- Threshold reconfiguration -------------------------------------------


def test_custom_threshold_three_demotes_at_three(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(feedback_module, "DEMOTION_THRESHOLD", 3)
    s = _build()
    apply_feedback(s, "X", valence=1.0, source="user")
    apply_feedback(s, "X", valence=1.0, source="user")
    apply_feedback(s, "X", valence=1.0, source="user")
    assert _y(s).lock_level == LOCK_NONE


def test_custom_threshold_three_does_not_demote_at_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(feedback_module, "DEMOTION_THRESHOLD", 3)
    s = _build()
    apply_feedback(s, "X", valence=1.0, source="user")
    apply_feedback(s, "X", valence=1.0, source="user")
    assert _y(s).lock_level == LOCK_USER


# --- Result reports demoted_locks ----------------------------------------


def test_below_threshold_demoted_locks_empty() -> None:
    s = _build()
    result = apply_feedback(s, "X", valence=1.0, source="user")
    assert result.demoted_locks == []


def test_at_threshold_demoted_locks_contains_target() -> None:
    s = _build(starting_pressure=4)
    result = apply_feedback(s, "X", valence=1.0, source="user")
    assert result.demoted_locks == ["Y"]


def test_at_threshold_pressured_and_demoted_both_contain_target() -> None:
    s = _build(starting_pressure=4)
    result = apply_feedback(s, "X", valence=1.0, source="user")
    assert "Y" in result.pressured_locks
    assert "Y" in result.demoted_locks


# --- Independent demotion across multiple targets ------------------------


def test_only_at_threshold_target_demotes_others_remain_locked() -> None:
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("X"))
    s.insert_belief(_mk("Y_high", lock_level=LOCK_USER,
                        locked_at="2026-04-26T01:00:00Z",
                        demotion_pressure=4))
    s.insert_belief(_mk("Y_low", lock_level=LOCK_USER,
                        locked_at="2026-04-26T02:00:00Z",
                        demotion_pressure=0))
    s.insert_edge(Edge(src="X", dst="Y_high", type=EDGE_CONTRADICTS,
                       weight=1.0))
    s.insert_edge(Edge(src="X", dst="Y_low", type=EDGE_CONTRADICTS,
                       weight=1.0))
    apply_feedback(s, "X", valence=1.0, source="user")
    assert s.get_belief("Y_high"), "Y_high must still exist"
    assert s.get_belief("Y_high").lock_level == LOCK_NONE  # type: ignore[union-attr]
    assert s.get_belief("Y_low"), "Y_low must still exist"
    assert s.get_belief("Y_low").lock_level == LOCK_USER  # type: ignore[union-attr]


def test_demoted_belief_no_longer_pressured_by_subsequent_feedback() -> None:
    """Once demoted, the lock filter excludes the belief from further
    pressure accumulation. This locks the v2.0 bug closed end-to-end."""
    s = _build(starting_pressure=4)
    apply_feedback(s, "X", valence=1.0, source="user")  # demotes
    apply_feedback(s, "X", valence=1.0, source="user")  # should be no-op for Y
    apply_feedback(s, "X", valence=1.0, source="user")  # also no-op
    y = _y(s)
    assert y.lock_level == LOCK_NONE
    assert y.demotion_pressure == 0
