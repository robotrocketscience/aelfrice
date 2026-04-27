"""End-to-end regression: feedback loop closes lock-pressure-demote.

Cumulative integration scenario added at v0.4.0. Exercises store +
scoring (posterior + lock floor) + retrieval (L0/L1) + feedback
(apply_feedback + history + pressure + auto-demote) in a single
realistic flow. Each atomic test asserts one property; the same
helper builds the fixture so the scenario runs end-to-end inside
each test independently.

Marked @pytest.mark.regression so future milestones can run only
the cumulative suite via `pytest -m regression` if needed. Default
collection includes regression tests in every staging-gate run.
"""
from __future__ import annotations

import pytest

from aelfrice.feedback import DEMOTION_THRESHOLD, apply_feedback
from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_CONTRADICTS,
    LOCK_NONE,
    LOCK_USER,
    Belief,
    Edge,
)
from aelfrice.retrieval import retrieve
from aelfrice.scoring import posterior_mean
from aelfrice.store import Store

pytestmark = pytest.mark.regression


def _seed_store() -> Store:
    """Common fixture: locked belief Y challenged by contradictor X.

    X has content matching the search query "main"; Y has content
    matching "master". Edge X -> Y CONTRADICTS. Y is user-locked.
    """
    s = Store(":memory:")
    s.insert_belief(Belief(
        id="X",
        content="we always use main as the default branch name",
        content_hash="h_X",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        demotion_pressure=0,
        created_at="2026-04-26T00:00:00Z",
        last_retrieved_at=None,
    ))
    s.insert_belief(Belief(
        id="Y",
        content="the default branch is master in this project",
        content_hash="h_Y",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_USER,
        locked_at="2026-04-26T01:00:00Z",
        demotion_pressure=0,
        created_at="2026-04-26T00:00:00Z",
        last_retrieved_at=None,
    ))
    s.insert_edge(Edge(src="X", dst="Y", type=EDGE_CONTRADICTS, weight=1.0))
    return s


def _y(store: Store) -> Belief:
    got = store.get_belief("Y")
    assert got is not None
    return got


def _x(store: Store) -> Belief:
    got = store.get_belief("X")
    assert got is not None
    return got


# --- Initial state ------------------------------------------------------


def test_initial_state_lock_in_l0() -> None:
    s = _seed_store()
    hits = retrieve(s, query="master")
    locked_ids = [b.id for b in hits if b.lock_level == LOCK_USER]
    assert "Y" in locked_ids


def test_initial_state_lock_above_l1_match() -> None:
    s = _seed_store()
    hits = retrieve(s, query="branch")
    ids = [b.id for b in hits]
    # Y is locked -> always L0; X (FTS5 hit on "branch") -> L1.
    assert ids.index("Y") < ids.index("X")


def test_initial_state_demotion_pressure_zero() -> None:
    s = _seed_store()
    assert _y(s).demotion_pressure == 0


def test_initial_state_x_posterior_at_prior() -> None:
    s = _seed_store()
    x = _x(s)
    assert posterior_mean(x.alpha, x.beta) == 0.5


# --- After one positive feedback event on X ------------------------------


def test_one_event_increments_pressure_to_one() -> None:
    s = _seed_store()
    apply_feedback(s, "X", valence=1.0, source="user")
    assert _y(s).demotion_pressure == 1


def test_one_event_keeps_lock_user() -> None:
    s = _seed_store()
    apply_feedback(s, "X", valence=1.0, source="user")
    assert _y(s).lock_level == LOCK_USER


def test_one_event_writes_one_history_row() -> None:
    s = _seed_store()
    apply_feedback(s, "X", valence=1.0, source="user")
    assert s.count_feedback_events() == 1


def test_one_event_increments_x_alpha() -> None:
    s = _seed_store()
    apply_feedback(s, "X", valence=1.0, source="user")
    assert _x(s).alpha == 2.0


# --- After threshold-many events: lock demotes ---------------------------


def test_threshold_events_demote_lock_to_none() -> None:
    s = _seed_store()
    for _ in range(DEMOTION_THRESHOLD):
        apply_feedback(s, "X", valence=1.0, source="user")
    assert _y(s).lock_level == LOCK_NONE


def test_threshold_events_clear_locked_at() -> None:
    s = _seed_store()
    for _ in range(DEMOTION_THRESHOLD):
        apply_feedback(s, "X", valence=1.0, source="user")
    assert _y(s).locked_at is None


def test_threshold_events_reset_demotion_pressure_to_zero() -> None:
    s = _seed_store()
    for _ in range(DEMOTION_THRESHOLD):
        apply_feedback(s, "X", valence=1.0, source="user")
    assert _y(s).demotion_pressure == 0


def test_threshold_events_y_alpha_beta_unchanged() -> None:
    """Demotion changes lock state, never the contradicted belief's posterior."""
    s = _seed_store()
    pre_alpha, pre_beta = _y(s).alpha, _y(s).beta
    for _ in range(DEMOTION_THRESHOLD):
        apply_feedback(s, "X", valence=1.0, source="user")
    post = _y(s)
    assert (post.alpha, post.beta) == (pre_alpha, pre_beta)


def test_threshold_events_x_alpha_increments_by_threshold() -> None:
    s = _seed_store()
    for _ in range(DEMOTION_THRESHOLD):
        apply_feedback(s, "X", valence=1.0, source="user")
    assert _x(s).alpha == 1.0 + float(DEMOTION_THRESHOLD)


def test_threshold_events_history_has_threshold_rows() -> None:
    s = _seed_store()
    for _ in range(DEMOTION_THRESHOLD):
        apply_feedback(s, "X", valence=1.0, source="user")
    assert s.count_feedback_events() == DEMOTION_THRESHOLD


def test_threshold_events_history_rows_in_strict_id_order() -> None:
    """Audit row ids monotonically increase per call so chronological
    order is recoverable from id."""
    s = _seed_store()
    for _ in range(DEMOTION_THRESHOLD):
        apply_feedback(s, "X", valence=1.0, source="user")
    events = s.list_feedback_events()
    ids = [e.id for e in events]
    # list_feedback_events returns DESC; reversed -> ASC.
    assert list(reversed(ids)) == sorted(ids)


# --- Retrieval after demotion -------------------------------------------


def test_after_demote_y_no_longer_in_l0() -> None:
    s = _seed_store()
    for _ in range(DEMOTION_THRESHOLD):
        apply_feedback(s, "X", valence=1.0, source="user")
    hits = retrieve(s, query="master")
    locked_ids = [b.id for b in hits if b.lock_level == LOCK_USER]
    assert "Y" not in locked_ids


def test_after_demote_y_still_findable_via_l1() -> None:
    s = _seed_store()
    for _ in range(DEMOTION_THRESHOLD):
        apply_feedback(s, "X", valence=1.0, source="user")
    hits = retrieve(s, query="master")
    ids = [b.id for b in hits]
    assert "Y" in ids


def test_after_demote_y_lock_level_in_retrieval_output_is_none() -> None:
    s = _seed_store()
    for _ in range(DEMOTION_THRESHOLD):
        apply_feedback(s, "X", valence=1.0, source="user")
    hits = retrieve(s, query="master")
    y_in_hits = [b for b in hits if b.id == "Y"]
    assert len(y_in_hits) == 1
    assert y_in_hits[0].lock_level == LOCK_NONE


# --- Subsequent feedback after demotion is a no-op for pressure ---------


def test_subsequent_feedback_after_demote_does_not_repressure() -> None:
    s = _seed_store()
    for _ in range(DEMOTION_THRESHOLD + 3):
        apply_feedback(s, "X", valence=1.0, source="user")
    # After demote at threshold, the next 3 events still update X and write
    # history rows but do not pressure the (no-longer-locked) Y.
    assert _y(s).demotion_pressure == 0


def test_subsequent_feedback_after_demote_still_writes_history() -> None:
    s = _seed_store()
    n = DEMOTION_THRESHOLD + 3
    for _ in range(n):
        apply_feedback(s, "X", valence=1.0, source="user")
    assert s.count_feedback_events() == n
