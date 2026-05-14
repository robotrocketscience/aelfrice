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

from aelfrice.meta_beliefs import SIGNAL_LATENCY
from aelfrice.retrieval import (
    DEFAULT_TEMPORAL_HALF_LIFE_SECONDS,
    ENV_META_BELIEF_HALF_LIFE,
    ENV_TEMPORAL_HALF_LIFE,
    HALF_LIFE_CEIL_SECONDS,
    HALF_LIFE_FLOOR_SECONDS,
    META_HALF_LIFE_KEY,
    META_HALF_LIFE_POSTERIOR_DECAY_SECONDS,
    META_HALF_LIFE_STATIC_DEFAULT,
    decode_meta_half_life,
    install_temporal_half_life_meta_belief,
    is_meta_belief_half_life_enabled,
    resolve_temporal_half_life_with_meta,
)
from aelfrice.store import MemoryStore


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


# --- Install helper (idempotency) ---------------------------------------

def test_install_meta_belief_first_call_returns_true() -> None:
    store = MemoryStore(":memory:")
    assert install_temporal_half_life_meta_belief(store, now_ts=1700000000) is True
    state = store.read_meta_belief_state(META_HALF_LIFE_KEY)
    assert state is not None
    assert state.static_default == META_HALF_LIFE_STATIC_DEFAULT
    assert state.half_life_seconds == META_HALF_LIFE_POSTERIOR_DECAY_SECONDS
    assert state.signal_weights == {SIGNAL_LATENCY: 1.0}


def test_install_meta_belief_second_call_no_op() -> None:
    store = MemoryStore(":memory:")
    install_temporal_half_life_meta_belief(store, now_ts=1700000000)
    # A second install must not overwrite — even if the timestamp shifts.
    assert install_temporal_half_life_meta_belief(
        store, now_ts=1700000999,
    ) is False
    state = store.read_meta_belief_state(META_HALF_LIFE_KEY)
    assert state is not None
    # last_updated_ts stays at first install — config rows are never
    # silently mutated by an install call.
    assert state.last_updated_ts == 1700000000


# --- Meta-aware resolver (precedence + fallback) ----------------------

def test_resolver_no_store_falls_through_to_static(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``store=None`` with meta-flag ON still resolves to the 7d static
    — no crash, no surprise meta-belief lookup against a non-existent
    handle."""
    monkeypatch.delenv(ENV_TEMPORAL_HALF_LIFE, raising=False)
    monkeypatch.setenv(ENV_META_BELIEF_HALF_LIFE, "enabled")
    assert resolve_temporal_half_life_with_meta(
        None, now_ts=1700000000,
    ) == pytest.approx(DEFAULT_TEMPORAL_HALF_LIFE_SECONDS)


def test_resolver_flag_off_ignores_meta_belief(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bench-gate clause: until the flag flips on, the meta-belief is
    invisible to retrieval even when installed."""
    monkeypatch.delenv(ENV_TEMPORAL_HALF_LIFE, raising=False)
    monkeypatch.delenv(ENV_META_BELIEF_HALF_LIFE, raising=False)
    store = MemoryStore(":memory:")
    install_temporal_half_life_meta_belief(store, now_ts=1700000000)
    assert resolve_temporal_half_life_with_meta(
        store, now_ts=1700000000,
    ) == pytest.approx(DEFAULT_TEMPORAL_HALF_LIFE_SECONDS)


def test_resolver_flag_on_cold_start_returns_decoded_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flag on, meta-belief installed, no evidence yet: surface value
    is the static_default, which decodes to ~6.5d via log-linear."""
    monkeypatch.delenv(ENV_TEMPORAL_HALF_LIFE, raising=False)
    monkeypatch.setenv(ENV_META_BELIEF_HALF_LIFE, "enabled")
    store = MemoryStore(":memory:")
    install_temporal_half_life_meta_belief(store, now_ts=1700000000)
    resolved = resolve_temporal_half_life_with_meta(
        store, now_ts=1700000000,
    )
    expected = decode_meta_half_life(META_HALF_LIFE_STATIC_DEFAULT)
    assert resolved == pytest.approx(expected, rel=1e-9)


def test_resolver_explicit_kwarg_wins_over_meta_belief(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit operator override bypasses the adaptive layer."""
    monkeypatch.delenv(ENV_TEMPORAL_HALF_LIFE, raising=False)
    monkeypatch.setenv(ENV_META_BELIEF_HALF_LIFE, "enabled")
    store = MemoryStore(":memory:")
    install_temporal_half_life_meta_belief(store, now_ts=1700000000)
    five_days = 5.0 * 24.0 * 3600.0
    resolved = resolve_temporal_half_life_with_meta(
        store, now_ts=1700000000, explicit=five_days,
    )
    assert resolved == pytest.approx(five_days)


def test_resolver_env_var_wins_over_meta_belief(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_TEMPORAL_HALF_LIFE, "172800")  # 2 days
    monkeypatch.setenv(ENV_META_BELIEF_HALF_LIFE, "enabled")
    store = MemoryStore(":memory:")
    install_temporal_half_life_meta_belief(store, now_ts=1700000000)
    resolved = resolve_temporal_half_life_with_meta(
        store, now_ts=1700000000,
    )
    assert resolved == pytest.approx(2.0 * 24.0 * 3600.0)


def test_resolver_flag_on_meta_uninstalled_falls_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flag on but meta-belief not installed: fall through to static."""
    monkeypatch.delenv(ENV_TEMPORAL_HALF_LIFE, raising=False)
    monkeypatch.setenv(ENV_META_BELIEF_HALF_LIFE, "enabled")
    store = MemoryStore(":memory:")  # no install_temporal_half_life_meta_belief
    assert resolve_temporal_half_life_with_meta(
        store, now_ts=1700000000,
    ) == pytest.approx(DEFAULT_TEMPORAL_HALF_LIFE_SECONDS)


def test_resolver_latency_evidence_shifts_resolved_half_life(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end smoke through the store: feeding consistent-fast
    latency evidence pushes the surfaced value above 0.5 and the
    resolved half-life above the cold-start ~6.5d.

    Responsiveness check (corresponds to the original AC's
    "responsiveness" item, retargeted to the `latency` signal per the
    D4 deferral of `relevance` to #779).
    """
    monkeypatch.delenv(ENV_TEMPORAL_HALF_LIFE, raising=False)
    monkeypatch.setenv(ENV_META_BELIEF_HALF_LIFE, "enabled")
    store = MemoryStore(":memory:")
    base_ts = 1700000000
    install_temporal_half_life_meta_belief(store, now_ts=base_ts)
    cold_start = resolve_temporal_half_life_with_meta(store, now_ts=base_ts)

    # 100 events of strong-positive latency evidence (evidence=1.0).
    for i in range(100):
        store.update_meta_belief(
            META_HALF_LIFE_KEY, SIGNAL_LATENCY,
            evidence=1.0,
            now_ts=base_ts + i,
        )
    shifted = resolve_temporal_half_life_with_meta(
        store, now_ts=base_ts + 100,
    )
    assert shifted > cold_start, (shifted, cold_start)
    # Responsiveness: >10% shift after 100 strong-signal events.
    assert shifted / cold_start > 1.10, (shifted, cold_start)
