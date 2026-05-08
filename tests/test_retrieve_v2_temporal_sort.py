"""Tests for retrieve_v2 temporal_sort kwarg + half-life resolver (#473)."""
from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
from aelfrice.retrieval import (
    DEFAULT_TEMPORAL_HALF_LIFE_SECONDS,
    ENV_TEMPORAL_HALF_LIFE,
    _apply_temporal_decay,
    _belief_age_seconds,
    resolve_temporal_half_life,
    retrieve_v2,
)
from aelfrice.store import MemoryStore


# A fixed reference epoch used by all temporal-decay assertions, so
# absolute "now" drift between test invocations cannot perturb results.
NOW = datetime(2026, 5, 8, 0, 0, 0, tzinfo=timezone.utc)


def _belief(
    *,
    idx: int,
    content: str,
    age_seconds: float,
    locked: bool = False,
) -> Belief:
    created = NOW - timedelta(seconds=age_seconds)
    return Belief(
        id=f"b{idx}",
        content=content,
        content_hash=f"h{idx}",
        alpha=2.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_USER if locked else LOCK_NONE,
        locked_at=created.isoformat() if locked else None,
        demotion_pressure=0,
        created_at=created.isoformat(),
        last_retrieved_at=None,
    )


# --- _belief_age_seconds boundaries ---------------------------------------


def test_belief_age_zero_when_created_at_now() -> None:
    b = _belief(idx=1, content="x", age_seconds=0.0)
    assert _belief_age_seconds(b, NOW) == pytest.approx(0.0)


def test_belief_age_clamped_at_zero_for_future_timestamps() -> None:
    b = _belief(idx=1, content="x", age_seconds=-3600.0)
    assert _belief_age_seconds(b, NOW) == 0.0


def test_belief_age_returns_zero_for_malformed_timestamp() -> None:
    b = _belief(idx=1, content="x", age_seconds=0.0)
    b.created_at = "not-an-iso-timestamp"
    assert _belief_age_seconds(b, NOW) == 0.0


def test_belief_age_returns_zero_for_empty_timestamp() -> None:
    b = _belief(idx=1, content="x", age_seconds=0.0)
    b.created_at = ""
    assert _belief_age_seconds(b, NOW) == 0.0


def test_belief_age_treats_naive_timestamp_as_utc() -> None:
    b = _belief(idx=1, content="x", age_seconds=0.0)
    b.created_at = (NOW - timedelta(seconds=120)).replace(tzinfo=None).isoformat()
    assert _belief_age_seconds(b, NOW) == pytest.approx(120.0)


# --- _apply_temporal_decay shape + invariants -----------------------------


def test_decay_empty_input_returns_empty() -> None:
    assert _apply_temporal_decay([], 604800.0, now=NOW) == []


def test_decay_zero_half_life_returns_input_unchanged() -> None:
    bs = [_belief(idx=i, content=str(i), age_seconds=i * 3600.0) for i in range(3)]
    assert _apply_temporal_decay(bs, 0.0, now=NOW) == bs
    assert _apply_temporal_decay(bs, -1.0, now=NOW) == bs


def test_decay_pins_locked_at_head() -> None:
    locked = _belief(
        idx=0, content="locked-old", age_seconds=10 * 24 * 3600.0, locked=True
    )
    fresh = _belief(idx=1, content="fresh", age_seconds=60.0)
    older = _belief(idx=2, content="older", age_seconds=3 * 24 * 3600.0)
    out = _apply_temporal_decay([locked, fresh, older], 86400.0, now=NOW)
    assert out[0].id == "b0"


def test_decay_recent_outranks_older_at_same_position() -> None:
    """When two unlocked beliefs sit at the same upstream rank-proxy
    (different positions), recency still moves the recent ahead if the
    decay penalty on the older one is large enough.

    Position 0 (rank_score 1.0) old vs position 1 (rank_score 0.5) recent:
    at half_life 1d and ages 10d / 0d, scores are 1.0 * 2^-10 ≈ 0.001 and
    0.5 * 1.0 = 0.5. Recent must win.
    """
    very_old = _belief(idx=0, content="old", age_seconds=10 * 86400.0)
    fresh = _belief(idx=1, content="fresh", age_seconds=0.0)
    out = _apply_temporal_decay([very_old, fresh], 86400.0, now=NOW)
    assert [b.id for b in out] == ["b1", "b0"]


def test_decay_preserves_order_when_all_same_age() -> None:
    """Equal ages → same decay multiplier → rank-proxy alone decides.
    Original head-of-list belief stays at head.
    """
    bs = [
        _belief(idx=0, content="a", age_seconds=3600.0),
        _belief(idx=1, content="b", age_seconds=3600.0),
        _belief(idx=2, content="c", age_seconds=3600.0),
    ]
    out = _apply_temporal_decay(bs, 86400.0, now=NOW)
    assert [b.id for b in out] == ["b0", "b1", "b2"]


def test_decay_does_not_reorder_locked_among_themselves() -> None:
    a = _belief(idx=0, content="a", age_seconds=10 * 86400.0, locked=True)
    b = _belief(idx=1, content="b", age_seconds=86400.0, locked=True)
    c = _belief(idx=2, content="c", age_seconds=0.0)
    out = _apply_temporal_decay([a, b, c], 86400.0, now=NOW)
    assert [x.id for x in out[:2]] == ["b0", "b1"]


# --- Decay math at half-life boundaries -----------------------------------


def test_decay_factor_at_half_life_is_one_half() -> None:
    """At age == half_life, decay multiplier should be exactly 0.5.

    We isolate the math by placing the recent belief at upstream rank 1
    (proxy 0.5) and the at-half-life belief at upstream rank 0 (proxy
    1.0). After decay: 1.0 * 0.5 = 0.5 vs 0.5 * 1.0 = 0.5 — tie. Stable
    sort keeps the head-of-list belief first.
    """
    half_life = 86400.0
    at_half = _belief(idx=0, content="half-old", age_seconds=half_life)
    fresh = _belief(idx=1, content="fresh", age_seconds=0.0)
    out = _apply_temporal_decay([at_half, fresh], half_life, now=NOW)
    assert [b.id for b in out] == ["b0", "b1"]


def test_decay_factor_at_two_half_lives_is_one_quarter() -> None:
    """At age == 2 * half_life, decay multiplier is 0.25. Position-0
    score = 0.25 vs position-1 score = 0.5 * 1.0 = 0.5 → fresh wins.
    """
    half_life = 86400.0
    at_two = _belief(idx=0, content="two-old", age_seconds=2.0 * half_life)
    fresh = _belief(idx=1, content="fresh", age_seconds=0.0)
    out = _apply_temporal_decay([at_two, fresh], half_life, now=NOW)
    assert [b.id for b in out] == ["b1", "b0"]


# --- resolve_temporal_half_life precedence --------------------------------


def test_resolve_default_when_nothing_set(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(ENV_TEMPORAL_HALF_LIFE, raising=False)
    assert resolve_temporal_half_life(start=tmp_path) == DEFAULT_TEMPORAL_HALF_LIFE_SECONDS


def test_resolve_env_overrides_explicit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(ENV_TEMPORAL_HALF_LIFE, "3600")
    assert resolve_temporal_half_life(86400.0, start=tmp_path) == 3600.0


def test_resolve_explicit_overrides_toml(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(ENV_TEMPORAL_HALF_LIFE, raising=False)
    (tmp_path / ".aelfrice.toml").write_text(
        "[retrieval]\ntemporal_half_life_seconds = 3600\n"
    )
    assert resolve_temporal_half_life(7200.0, start=tmp_path) == 7200.0


def test_resolve_toml_overrides_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(ENV_TEMPORAL_HALF_LIFE, raising=False)
    (tmp_path / ".aelfrice.toml").write_text(
        "[retrieval]\ntemporal_half_life_seconds = 12345\n"
    )
    assert resolve_temporal_half_life(start=tmp_path) == 12345.0


def test_resolve_rejects_non_positive_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(ENV_TEMPORAL_HALF_LIFE, "0")
    assert resolve_temporal_half_life(start=tmp_path) == DEFAULT_TEMPORAL_HALF_LIFE_SECONDS


def test_resolve_rejects_non_numeric_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(ENV_TEMPORAL_HALF_LIFE, "not-a-number")
    assert resolve_temporal_half_life(start=tmp_path) == DEFAULT_TEMPORAL_HALF_LIFE_SECONDS


def test_resolve_falls_through_non_positive_kwarg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(ENV_TEMPORAL_HALF_LIFE, raising=False)
    assert (
        resolve_temporal_half_life(0.0, start=tmp_path)
        == DEFAULT_TEMPORAL_HALF_LIFE_SECONDS
    )
    assert (
        resolve_temporal_half_life(-1.0, start=tmp_path)
        == DEFAULT_TEMPORAL_HALF_LIFE_SECONDS
    )


# --- retrieve_v2 integration ----------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    s = MemoryStore(str(tmp_path / "rv2_temporal.db"))
    yield s
    s.close()


def _insert(store: MemoryStore, idx: int, content: str, age_days: float, locked: bool = False) -> None:
    store.insert_belief(
        _belief(idx=idx, content=content, age_seconds=age_days * 86400.0, locked=locked)
    )


def test_retrieve_v2_temporal_sort_off_is_byte_identical(
    monkeypatch: pytest.MonkeyPatch, store: MemoryStore
) -> None:
    """temporal_sort=False (the default) leaves the merged order
    untouched relative to a call that omits the kwarg entirely."""
    monkeypatch.delenv(ENV_TEMPORAL_HALF_LIFE, raising=False)
    _insert(store, 1, "alpha factual statement", age_days=10)
    _insert(store, 2, "alpha factual paragraph", age_days=1)
    _insert(store, 3, "alpha factual content", age_days=0)
    a = retrieve_v2(store, "alpha", budget=10_000)
    b = retrieve_v2(store, "alpha", budget=10_000, temporal_sort=False)
    assert [x.id for x in a.beliefs] == [x.id for x in b.beliefs]


def test_retrieve_v2_temporal_sort_pins_locked_at_head(
    monkeypatch: pytest.MonkeyPatch, store: MemoryStore
) -> None:
    monkeypatch.delenv(ENV_TEMPORAL_HALF_LIFE, raising=False)
    _insert(store, 1, "alpha factual statement old locked", age_days=30, locked=True)
    _insert(store, 2, "alpha factual paragraph fresh", age_days=0)
    result = retrieve_v2(
        store,
        "alpha",
        budget=10_000,
        include_locked=True,
        temporal_sort=True,
        temporal_half_life_seconds=86400.0,
    )
    assert result.beliefs[0].lock_level == LOCK_USER


def test_retrieve_v2_temporal_sort_explicit_half_life_kwarg(
    monkeypatch: pytest.MonkeyPatch, store: MemoryStore
) -> None:
    """Explicit kwarg flows through to the decay calculation. With a
    very short half-life, even a moderately old belief gets crushed
    relative to a fresh one regardless of upstream rank order."""
    monkeypatch.delenv(ENV_TEMPORAL_HALF_LIFE, raising=False)
    _insert(store, 1, "alpha factual statement old", age_days=10)
    _insert(store, 2, "alpha factual statement new", age_days=0)
    result = retrieve_v2(
        store,
        "alpha factual statement",
        budget=10_000,
        temporal_sort=True,
        temporal_half_life_seconds=3600.0,
    )
    ids = [b.id for b in result.beliefs]
    assert ids.index("b2") < ids.index("b1")
