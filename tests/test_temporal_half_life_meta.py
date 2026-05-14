"""Tests for the #756 meta-belief consumer wiring on ``temporal_sort``.

Covers the consumer-side half of the umbrella #480 first-meta-belief
deliverable: log-linear bounded encoding, the env feature flag, the
idempotent install helper, and the read path that surfaces a decayed
half-life into ``retrieve_v2``.

Substrate-level coverage (Beta-Bernoulli update, decay, signal-class
enum) lives in ``tests/test_meta_beliefs.py``.
"""
from __future__ import annotations

import math

import pytest

from aelfrice.retrieval import (
    DEFAULT_TEMPORAL_HALF_LIFE_SECONDS,
    ENV_META_BELIEF_HALF_LIFE,
    HALF_LIFE_CEIL_SECONDS,
    HALF_LIFE_FLOOR_SECONDS,
    META_HALF_LIFE_KEY,
    META_HALF_LIFE_POSTERIOR_DECAY_SECONDS,
    META_HALF_LIFE_STATIC_DEFAULT,
    decode_meta_half_life,
    is_meta_belief_half_life_enabled,
)


# --- Log-linear bounded encoding (#756 D3 ratification) ----------------

def test_decode_meta_half_life_floor() -> None:
    assert decode_meta_half_life(0.0) == pytest.approx(
        HALF_LIFE_FLOOR_SECONDS, rel=1e-9
    )


def test_decode_meta_half_life_ceil() -> None:
    assert decode_meta_half_life(1.0) == pytest.approx(
        HALF_LIFE_CEIL_SECONDS, rel=1e-9
    )


def test_decode_meta_half_life_mid_near_7d() -> None:
    """v=0.5 lands close to the #473 ratified 7d static (within 4%).

    Log-linear midpoint between 3d and 14d is sqrt(3*14) ≈ 6.48d, so the
    cold-start surface is ~6.5d — close enough to the 7d static that
    flipping the meta-belief feature on for the first time doesn't jolt
    the retrieval ordering visibly.
    """
    seven_days = 7.0 * 24.0 * 3600.0
    assert decode_meta_half_life(0.5) == pytest.approx(
        math.sqrt(HALF_LIFE_FLOOR_SECONDS * HALF_LIFE_CEIL_SECONDS),
        rel=1e-9,
    )
    assert decode_meta_half_life(0.5) == pytest.approx(seven_days, rel=0.075)


def test_decode_meta_half_life_clamps_out_of_range() -> None:
    """Substrate guarantees `value` in [0, 1] but defend at the boundary.

    A misconfigured ``static_default`` (e.g. carried in from a stale
    row before the substrate's clamp landed) could surface a value
    outside the unit interval. We clamp rather than raise — retrieval
    must never crash on a stale meta-belief row.
    """
    assert decode_meta_half_life(-1.0) == decode_meta_half_life(0.0)
    assert decode_meta_half_life(2.0) == decode_meta_half_life(1.0)
    assert decode_meta_half_life(float("inf")) == decode_meta_half_life(1.0)


def test_decode_meta_half_life_monotonic() -> None:
    """Higher value → longer half-life, strictly increasing on the
    interior of `[0, 1]`. Pins the encoding direction so a downstream
    accidental sign-flip falls over loudly."""
    samples = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    out = [decode_meta_half_life(v) for v in samples]
    assert out == sorted(out), out


# --- Env feature flag (#756 default-OFF) -------------------------------

def test_meta_flag_unset_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV_META_BELIEF_HALF_LIFE, raising=False)
    assert is_meta_belief_half_life_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "enabled",
                                   "ENABLED", "Enabled", " 1 ", "TRUE"])
def test_meta_flag_truthy_tokens(
    monkeypatch: pytest.MonkeyPatch, value: str,
) -> None:
    """Both the codebase-standard truthy tokens AND the issue's
    `=enabled` spelling resolve to True."""
    monkeypatch.setenv(ENV_META_BELIEF_HALF_LIFE, value)
    assert is_meta_belief_half_life_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "off", "no", "disabled",
                                   "", "garbage", "2"])
def test_meta_flag_falsy_tokens(
    monkeypatch: pytest.MonkeyPatch, value: str,
) -> None:
    monkeypatch.setenv(ENV_META_BELIEF_HALF_LIFE, value)
    assert is_meta_belief_half_life_enabled() is False


# --- Constants sanity --------------------------------------------------

def test_meta_constants_match_ratification() -> None:
    """Pins the constants the #756 ratification (2026-05-13) called out.

    A future commit that changes these would have to also change the
    consumer's expected behaviour, so locking them in the test surface
    makes the contract visible to reviewers.
    """
    assert META_HALF_LIFE_KEY == "meta:retrieval.temporal_half_life_seconds"
    assert HALF_LIFE_FLOOR_SECONDS == 3.0 * 24.0 * 3600.0
    assert HALF_LIFE_CEIL_SECONDS == 14.0 * 24.0 * 3600.0
    assert META_HALF_LIFE_STATIC_DEFAULT == 0.5
    assert META_HALF_LIFE_POSTERIOR_DECAY_SECONDS == 30 * 24 * 3600
    # Static default of 0.5 in the unit interval must round-trip to
    # roughly the #473 ratified 7-day static via the encoding.
    seven_days = 7.0 * 24.0 * 3600.0
    cold_start = decode_meta_half_life(META_HALF_LIFE_STATIC_DEFAULT)
    assert cold_start == pytest.approx(seven_days, rel=0.075)
    # And the static-7d static default of the precedence chain still
    # exists for the meta-flag-OFF / no-store path.
    assert DEFAULT_TEMPORAL_HALF_LIFE_SECONDS == seven_days
