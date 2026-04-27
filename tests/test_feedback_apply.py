"""apply_feedback: Bayesian-update + audit-row contract.

One assertion per test where practical; every test uses an in-memory
store and finishes in milliseconds.
"""
from __future__ import annotations

import pytest

from aelfrice.feedback import FeedbackResult, apply_feedback
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief
from aelfrice.scoring import posterior_mean
from aelfrice.store import Store


def _mk(bid: str = "b1", alpha: float = 1.0, beta: float = 1.0) -> Belief:
    return Belief(
        id=bid,
        content=f"belief {bid}",
        content_hash=f"h_{bid}",
        alpha=alpha,
        beta=beta,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        demotion_pressure=0,
        created_at="2026-04-26T00:00:00Z",
        last_retrieved_at=None,
    )


def _store_with(b: Belief) -> Store:
    s = Store(":memory:")
    s.insert_belief(b)
    return s


# --- Happy path ----------------------------------------------------------


def test_positive_valence_increments_alpha() -> None:
    s = _store_with(_mk(alpha=2.0, beta=3.0))
    apply_feedback(s, "b1", valence=1.0, source="user")
    got = s.get_belief("b1")
    assert got is not None
    assert got.alpha == 3.0


def test_positive_valence_does_not_change_beta() -> None:
    s = _store_with(_mk(alpha=2.0, beta=3.0))
    apply_feedback(s, "b1", valence=1.0, source="user")
    got = s.get_belief("b1")
    assert got is not None
    assert got.beta == 3.0


def test_negative_valence_increments_beta() -> None:
    s = _store_with(_mk(alpha=2.0, beta=3.0))
    apply_feedback(s, "b1", valence=-1.0, source="user")
    got = s.get_belief("b1")
    assert got is not None
    assert got.beta == 4.0


def test_negative_valence_does_not_change_alpha() -> None:
    s = _store_with(_mk(alpha=2.0, beta=3.0))
    apply_feedback(s, "b1", valence=-1.0, source="user")
    got = s.get_belief("b1")
    assert got is not None
    assert got.alpha == 2.0


def test_fractional_valence_is_honored() -> None:
    s = _store_with(_mk(alpha=2.0, beta=3.0))
    apply_feedback(s, "b1", valence=0.25, source="user")
    got = s.get_belief("b1")
    assert got is not None
    assert got.alpha == 2.25


# --- Posterior math ------------------------------------------------------


def test_positive_event_raises_posterior_mean() -> None:
    s = _store_with(_mk(alpha=1.0, beta=1.0))
    before = posterior_mean(1.0, 1.0)
    apply_feedback(s, "b1", valence=1.0, source="user")
    got = s.get_belief("b1")
    assert got is not None
    after = posterior_mean(got.alpha, got.beta)
    assert after > before


def test_negative_event_lowers_posterior_mean() -> None:
    s = _store_with(_mk(alpha=1.0, beta=1.0))
    before = posterior_mean(1.0, 1.0)
    apply_feedback(s, "b1", valence=-1.0, source="user")
    got = s.get_belief("b1")
    assert got is not None
    after = posterior_mean(got.alpha, got.beta)
    assert after < before


# --- Audit row contract --------------------------------------------------


def test_each_call_writes_exactly_one_history_row() -> None:
    s = _store_with(_mk())
    assert s.count_feedback_events("b1") == 0
    apply_feedback(s, "b1", valence=1.0, source="user")
    assert s.count_feedback_events("b1") == 1


def test_three_calls_write_three_history_rows() -> None:
    s = _store_with(_mk())
    apply_feedback(s, "b1", valence=1.0, source="user")
    apply_feedback(s, "b1", valence=-1.0, source="user")
    apply_feedback(s, "b1", valence=1.0, source="user")
    assert s.count_feedback_events("b1") == 3


def test_history_row_records_valence_verbatim() -> None:
    s = _store_with(_mk())
    apply_feedback(s, "b1", valence=0.7, source="user", now="2026-04-26T18:00:00Z")
    events = s.list_feedback_events(belief_id="b1")
    assert events[0].valence == 0.7


def test_history_row_records_source_verbatim() -> None:
    s = _store_with(_mk())
    apply_feedback(s, "b1", valence=1.0, source="tool:pytest",
                   now="2026-04-26T18:00:00Z")
    events = s.list_feedback_events(belief_id="b1")
    assert events[0].source == "tool:pytest"


def test_history_row_records_provided_timestamp() -> None:
    s = _store_with(_mk())
    apply_feedback(s, "b1", valence=1.0, source="user",
                   now="2026-04-26T18:00:00Z")
    events = s.list_feedback_events(belief_id="b1")
    assert events[0].created_at == "2026-04-26T18:00:00Z"


def test_history_row_id_returned_in_result_matches_storage() -> None:
    s = _store_with(_mk())
    result = apply_feedback(s, "b1", valence=1.0, source="user",
                            now="2026-04-26T18:00:00Z")
    events = s.list_feedback_events(belief_id="b1")
    assert events[0].id == result.event_id


# --- Result object -------------------------------------------------------


def test_result_object_is_feedback_result_typed() -> None:
    s = _store_with(_mk())
    result = apply_feedback(s, "b1", valence=1.0, source="user")
    assert isinstance(result, FeedbackResult)


def test_result_priors_match_pre_call_state() -> None:
    s = _store_with(_mk(alpha=2.0, beta=3.0))
    result = apply_feedback(s, "b1", valence=1.0, source="user")
    assert result.prior_alpha == 2.0
    assert result.prior_beta == 3.0


def test_result_new_values_match_post_call_state() -> None:
    s = _store_with(_mk(alpha=2.0, beta=3.0))
    result = apply_feedback(s, "b1", valence=1.0, source="user")
    assert result.new_alpha == 3.0
    assert result.new_beta == 3.0


# --- Errors --------------------------------------------------------------


def test_zero_valence_raises_value_error() -> None:
    s = _store_with(_mk())
    with pytest.raises(ValueError):
        apply_feedback(s, "b1", valence=0.0, source="user")


def test_zero_valence_does_not_write_history() -> None:
    s = _store_with(_mk())
    with pytest.raises(ValueError):
        apply_feedback(s, "b1", valence=0.0, source="user")
    assert s.count_feedback_events("b1") == 0


def test_zero_valence_does_not_change_posterior() -> None:
    s = _store_with(_mk(alpha=2.0, beta=3.0))
    with pytest.raises(ValueError):
        apply_feedback(s, "b1", valence=0.0, source="user")
    got = s.get_belief("b1")
    assert got is not None
    assert (got.alpha, got.beta) == (2.0, 3.0)


def test_empty_source_raises_value_error() -> None:
    s = _store_with(_mk())
    with pytest.raises(ValueError):
        apply_feedback(s, "b1", valence=1.0, source="")


def test_unknown_belief_id_raises_value_error() -> None:
    s = Store(":memory:")
    with pytest.raises(ValueError):
        apply_feedback(s, "nonexistent", valence=1.0, source="user")


def test_unknown_belief_id_does_not_write_history() -> None:
    s = Store(":memory:")
    with pytest.raises(ValueError):
        apply_feedback(s, "nonexistent", valence=1.0, source="user")
    assert s.count_feedback_events() == 0
