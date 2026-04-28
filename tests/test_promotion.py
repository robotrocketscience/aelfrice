"""Origin-tier promotion: provenance flip, refusal, audit, reversibility.

Atomic short tests, one property each. Mirror style with
test_contradiction.py.
"""
from __future__ import annotations

import pytest

from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    LOCK_USER,
    ORIGIN_AGENT_INFERRED,
    ORIGIN_UNKNOWN,
    ORIGIN_USER_STATED,
    ORIGIN_USER_VALIDATED,
    Belief,
)
from aelfrice.promotion import (
    SOURCE_PROMOTE_USER_VALIDATED,
    SOURCE_REVERT_TO_AGENT_INFERRED,
    devalidate,
    promote,
)
from aelfrice.store import MemoryStore


def _mk(
    bid: str,
    *,
    origin: str = ORIGIN_AGENT_INFERRED,
    lock: str = LOCK_NONE,
    locked_at: str | None = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    btype: str = BELIEF_FACTUAL,
    created_at: str = "2026-04-26T00:00:00Z",
) -> Belief:
    return Belief(
        id=bid,
        content=f"belief {bid}",
        content_hash=f"h_{bid}",
        alpha=alpha,
        beta=beta,
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


# --- Provenance flip ----------------------------------------------------


def test_validate_changes_origin_to_user_validated() -> None:
    s = _seed(_mk("X", origin=ORIGIN_AGENT_INFERRED))
    result = promote(s, "X")
    assert result.prior_origin == ORIGIN_AGENT_INFERRED
    assert result.new_origin == ORIGIN_USER_VALIDATED
    after = s.get_belief("X")
    assert after is not None
    assert after.origin == ORIGIN_USER_VALIDATED


def test_validate_works_on_unknown_origin() -> None:
    """v1.0/v1.1 rows that landed as 'unknown' can still promote."""
    s = _seed(_mk("X", origin=ORIGIN_UNKNOWN))
    result = promote(s, "X")
    assert result.prior_origin == ORIGIN_UNKNOWN
    assert result.new_origin == ORIGIN_USER_VALIDATED


def test_validate_does_not_change_lock_level() -> None:
    s = _seed(_mk("X", origin=ORIGIN_AGENT_INFERRED, lock=LOCK_NONE))
    promote(s, "X")
    after = s.get_belief("X")
    assert after is not None
    assert after.lock_level == LOCK_NONE


def test_validate_does_not_change_alpha_beta() -> None:
    """Posteriors preserved — promotion is provenance, not evidence."""
    s = _seed(_mk("X", alpha=2.5, beta=1.5))
    promote(s, "X")
    after = s.get_belief("X")
    assert after is not None
    assert after.alpha == 2.5
    assert after.beta == 1.5


def test_validate_does_not_change_type() -> None:
    s = _seed(_mk("X", btype=BELIEF_FACTUAL))
    promote(s, "X")
    after = s.get_belief("X")
    assert after is not None
    assert after.type == BELIEF_FACTUAL


def test_validate_idempotent_no_double_audit_row() -> None:
    s = _seed(_mk("X", origin=ORIGIN_AGENT_INFERRED))
    first = promote(s, "X")
    second = promote(s, "X")
    assert first.audit_event_id is not None
    assert second.audit_event_id is None
    assert second.already_validated is True
    assert s.count_feedback_events() == 1


# --- Refusal cases ------------------------------------------------------


def test_validate_raises_on_missing_belief() -> None:
    s = _seed(_mk("X"))
    with pytest.raises(ValueError, match="ghost"):
        promote(s, "ghost")


def test_validate_refuses_locked_belief() -> None:
    s = _seed(_mk("X", lock=LOCK_USER, locked_at="2026-04-26T01:00:00Z",
                  origin=ORIGIN_USER_STATED))
    with pytest.raises(ValueError, match="locked"):
        promote(s, "X")


def test_validate_refuses_user_stated_origin() -> None:
    """Inconsistency case: origin=user_stated without lock. Refuse."""
    s = _seed(_mk("X", origin=ORIGIN_USER_STATED))  # no lock
    with pytest.raises(ValueError, match="user_stated"):
        promote(s, "X")


# --- Audit row ----------------------------------------------------------


def test_validate_writes_audit_row_with_promotion_prefix() -> None:
    s = _seed(_mk("X"))
    promote(s, "X")
    events = s.list_feedback_events()
    assert len(events) == 1
    ev = events[0]
    assert ev.belief_id == "X"
    assert ev.source == SOURCE_PROMOTE_USER_VALIDATED
    assert ev.source.startswith("promotion:")


def test_validate_audit_row_has_zero_valence() -> None:
    s = _seed(_mk("X"))
    promote(s, "X")
    ev = s.list_feedback_events()[0]
    assert ev.valence == 0.0


def test_validate_audit_row_carries_now_kwarg() -> None:
    s = _seed(_mk("X"))
    promote(s, "X", now="2026-04-30T12:00:00Z")
    ev = s.list_feedback_events()[0]
    assert ev.created_at == "2026-04-30T12:00:00Z"


def test_validate_audit_row_belief_id_is_subject() -> None:
    s = _seed(_mk("X"), _mk("Y"))
    promote(s, "X")
    ev = s.list_feedback_events()[0]
    assert ev.belief_id == "X"


def test_validate_custom_source_label() -> None:
    s = _seed(_mk("X"))
    promote(s, "X", source_label="promotion:mcp_validate")
    ev = s.list_feedback_events()[0]
    assert ev.source == "promotion:mcp_validate"


# --- Reversibility -------------------------------------------------------


def test_devalidate_flips_origin_back_to_agent_inferred() -> None:
    s = _seed(_mk("X", origin=ORIGIN_USER_VALIDATED))
    result = devalidate(s, "X")
    assert result.prior_origin == ORIGIN_USER_VALIDATED
    assert result.new_origin == ORIGIN_AGENT_INFERRED
    after = s.get_belief("X")
    assert after is not None
    assert after.origin == ORIGIN_AGENT_INFERRED


def test_devalidate_writes_revert_audit_row() -> None:
    s = _seed(_mk("X", origin=ORIGIN_USER_VALIDATED))
    devalidate(s, "X")
    ev = s.list_feedback_events()[0]
    assert ev.source == SOURCE_REVERT_TO_AGENT_INFERRED


def test_devalidate_refuses_non_validated_belief() -> None:
    s = _seed(_mk("X", origin=ORIGIN_AGENT_INFERRED))
    with pytest.raises(ValueError, match="not user_validated"):
        devalidate(s, "X")


def test_devalidate_refuses_locked_belief() -> None:
    s = _seed(_mk("X", lock=LOCK_USER, locked_at="2026-04-26T01:00:00Z",
                  origin=ORIGIN_USER_VALIDATED))
    with pytest.raises(ValueError, match="locked"):
        devalidate(s, "X")


def test_validate_then_devalidate_preserves_alpha_beta() -> None:
    s = _seed(_mk("X", alpha=2.5, beta=1.5))
    promote(s, "X")
    devalidate(s, "X")
    after = s.get_belief("X")
    assert after is not None
    assert after.alpha == 2.5
    assert after.beta == 1.5


def test_validate_devalidate_revalidate_round_trip() -> None:
    s = _seed(_mk("X"))
    promote(s, "X")
    devalidate(s, "X")
    promote(s, "X")
    after = s.get_belief("X")
    assert after is not None
    assert after.origin == ORIGIN_USER_VALIDATED
    # Three audit rows: promote, devalidate, promote.
    assert s.count_feedback_events() == 3
