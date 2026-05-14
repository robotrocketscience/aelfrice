"""Unit tests for the post-rank score adjusters in `aelfrice.uri_baki`.

Issue #153 is a research issue; the deliverable is the benchmark
result table. These tests pin the pure-function semantics so the
benchmark cannot drift away from the documented effects.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from aelfrice.models import (
    LOCK_NONE,
    LOCK_USER,
    ORIGIN_UNKNOWN,
    RETENTION_UNKNOWN,
    Belief,
)
from aelfrice.uri_baki import (
    DEFAULT_LOCKED_FLOOR,
    DEFAULT_RECENCY_LAMBDA,
    DEFAULT_SUPERSESSION_FACTOR,
    apply_locked_floor,
    apply_recency_decay,
    apply_supersession_demote,
)


def _b(
    *,
    bid: str = "b1",
    lock: str = LOCK_NONE,
    created_at: str = "2026-04-01T00:00:00+00:00",
) -> Belief:
    return Belief(
        id=bid,
        content="x",
        content_hash="h",
        alpha=1.0,
        beta=1.0,
        type="factual",
        lock_level=lock,
        locked_at=None,
        created_at=created_at,
        last_retrieved_at=None,
        session_id=None,
        origin=ORIGIN_UNKNOWN,
        corroboration_count=0,
        hibernation_score=None,
        activation_condition=None,
        retention_class=RETENTION_UNKNOWN,
    )


# --- locked floor ---------------------------------------------------------


def test_locked_floor_raises_below_floor() -> None:
    beliefs = [_b(bid="b1", lock=LOCK_USER), _b(bid="b2", lock=LOCK_NONE)]
    out = apply_locked_floor(beliefs, [-3.0, -3.0], floor=-1.0)
    assert out == [-1.0, -3.0]


def test_locked_floor_does_not_demote() -> None:
    beliefs = [_b(bid="b1", lock=LOCK_USER)]
    # locked belief already above floor — must not change
    out = apply_locked_floor(beliefs, [5.0], floor=-1.0)
    assert out == [5.0]


def test_locked_floor_default_is_zero() -> None:
    assert DEFAULT_LOCKED_FLOOR == 0.0


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        apply_locked_floor([_b()], [1.0, 2.0])


# --- supersession demote --------------------------------------------------


def test_supersession_demote_default_factor() -> None:
    assert DEFAULT_SUPERSESSION_FACTOR == 0.5


def test_supersession_demote_applies_factor() -> None:
    beliefs = [_b(bid="b1"), _b(bid="b2"), _b(bid="b3")]
    out = apply_supersession_demote(beliefs, [4.0, 4.0, 4.0], {"b2"})
    assert out == [4.0, 2.0, 4.0]


def test_supersession_demote_empty_set_is_noop() -> None:
    beliefs = [_b(bid="b1")]
    out = apply_supersession_demote(beliefs, [3.0], set())
    assert out == [3.0]


def test_supersession_demote_factor_one_is_noop() -> None:
    beliefs = [_b(bid="b1")]
    out = apply_supersession_demote(beliefs, [3.0], {"b1"}, factor=1.0)
    assert out == [3.0]


def test_supersession_demote_factor_zero_zeros_targets() -> None:
    beliefs = [_b(bid="b1")]
    out = apply_supersession_demote(beliefs, [3.0], {"b1"}, factor=0.0)
    assert out == [0.0]


# --- recency decay --------------------------------------------------------


def test_recency_decay_default_lambda_180_day_halflife() -> None:
    # half-life = ln(2)/lambda. With lam=1/180, half-life ≈ 124.77 days.
    # The issue spec calls out 180-day half-life as the intent; the
    # default constant is 1/180 verbatim, and the benchmark sweeps lam.
    assert DEFAULT_RECENCY_LAMBDA == pytest.approx(1.0 / 180.0)


def test_recency_decay_zero_age_is_noop() -> None:
    now = datetime(2026, 5, 1, tzinfo=timezone.utc)
    beliefs = [_b(bid="b1", created_at="2026-05-01T00:00:00+00:00")]
    out = apply_recency_decay(beliefs, [4.0], now=now)
    assert out == [4.0]


def test_recency_decay_lambda_zero_is_noop() -> None:
    now = datetime(2026, 5, 1, tzinfo=timezone.utc)
    beliefs = [_b(created_at="2020-01-01T00:00:00+00:00")]
    out = apply_recency_decay(beliefs, [4.0], now=now, lam=0.0)
    assert out == [4.0]


def test_recency_decay_one_halflife() -> None:
    now = datetime(2026, 5, 1, tzinfo=timezone.utc)
    halflife_days = math.log(2.0) * 180.0
    older = now - timedelta(days=halflife_days)
    beliefs = [_b(created_at=older.isoformat())]
    out = apply_recency_decay(beliefs, [4.0], now=now)
    assert out[0] == pytest.approx(2.0, rel=1e-6)


def test_recency_decay_future_timestamp_is_noop() -> None:
    now = datetime(2026, 5, 1, tzinfo=timezone.utc)
    future = now + timedelta(days=10)
    beliefs = [_b(created_at=future.isoformat())]
    out = apply_recency_decay(beliefs, [4.0], now=now)
    assert out == [4.0]


def test_recency_decay_unparseable_timestamp_is_noop() -> None:
    now = datetime(2026, 5, 1, tzinfo=timezone.utc)
    beliefs = [_b(created_at="not-a-date")]
    out = apply_recency_decay(beliefs, [4.0], now=now)
    assert out == [4.0]


def test_recency_decay_z_suffix_parses() -> None:
    now = datetime(2026, 5, 1, tzinfo=timezone.utc)
    beliefs = [_b(created_at="2026-04-01T00:00:00Z")]
    out = apply_recency_decay(beliefs, [1.0], now=now)
    # 30 days at lam=1/180: factor = exp(-30/180) ≈ 0.8465
    assert out[0] == pytest.approx(math.exp(-30.0 / 180.0), rel=1e-6)
