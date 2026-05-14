"""Tests for the #758 meta-belief consumer wiring on gamma-rerank posterior temperature.

Sub-task D of umbrella #480. Pattern reuse of the #756 temporal_half_life_seconds,
#757 bm25f_anchor_weight, #759 bfs_depth_budget, and #760 expansion_gate consumers
-- same shape, scoped to the Boltzmann temperature T used by gamma_posterior_score.

Substrate-level coverage (Beta-Bernoulli update, decay, signal-class enum) lives
in tests/test_meta_beliefs.py. The gamma-rerank surface and flag (#796) are covered
in tests/test_retrieve_gamma_flag.py and tests/test_scoring_gamma.py.

Two-axis state machine:
  - ENV_USE_GAMMA_POSTERIOR_TEMPERATURE gates whether the rerank uses T at all.
  - ENV_META_BELIEF_POSTERIOR_TEMPERATURE gates whether the sweeper delivers
    relevance evidence so T can learn.
  The axes are independent; this file covers the second axis only.
"""
from __future__ import annotations

import math

import pytest

from aelfrice.meta_beliefs import SIGNAL_RELEVANCE
from aelfrice.retrieval import (
    ENV_META_BELIEF_POSTERIOR_TEMPERATURE,
    META_POSTERIOR_TEMPERATURE_KEY,
    META_POSTERIOR_TEMPERATURE_POSTERIOR_DECAY_SECONDS,
    META_POSTERIOR_TEMPERATURE_STATIC_DEFAULT,
    POSTERIOR_TEMPERATURE_CEIL,
    POSTERIOR_TEMPERATURE_FLOOR,
    install_posterior_temperature_meta_belief,
    is_meta_belief_posterior_temperature_enabled,
    resolve_posterior_temperature_with_meta,
)
from aelfrice.store import MemoryStore


# ---------------------------------------------------------------------------
# Env feature flag (#758 default-OFF)
# ---------------------------------------------------------------------------

def test_meta_flag_unset_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_META_BELIEF_POSTERIOR_TEMPERATURE, raising=False)
    assert is_meta_belief_posterior_temperature_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "enabled"])
def test_meta_flag_truthy_tokens(
    monkeypatch: pytest.MonkeyPatch, value: str,
) -> None:
    monkeypatch.setenv(ENV_META_BELIEF_POSTERIOR_TEMPERATURE, value)
    assert is_meta_belief_posterior_temperature_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "disabled", ""])
def test_meta_flag_falsy_tokens(
    monkeypatch: pytest.MonkeyPatch, value: str,
) -> None:
    monkeypatch.setenv(ENV_META_BELIEF_POSTERIOR_TEMPERATURE, value)
    assert is_meta_belief_posterior_temperature_enabled() is False


# ---------------------------------------------------------------------------
# Constants pin (#796/#758 ratification)
# ---------------------------------------------------------------------------

def test_meta_constants_match_ratification() -> None:
    """Pin the ratified defaults so a careless edit surfaces here, not
    in a downstream gate regression."""
    assert META_POSTERIOR_TEMPERATURE_KEY == "meta:retrieval.posterior_temperature"
    assert POSTERIOR_TEMPERATURE_FLOOR == 0.5
    assert POSTERIOR_TEMPERATURE_CEIL == 2.0
    assert META_POSTERIOR_TEMPERATURE_STATIC_DEFAULT == 0.5
    assert META_POSTERIOR_TEMPERATURE_POSTERIOR_DECAY_SECONDS == 30 * 24 * 3600


# ---------------------------------------------------------------------------
# Install helper (idempotency)
# ---------------------------------------------------------------------------

def test_install_meta_belief_first_call_returns_true() -> None:
    store = MemoryStore(":memory:")
    assert install_posterior_temperature_meta_belief(
        store, now_ts=1700000000
    ) is True
    state = store.read_meta_belief_state(META_POSTERIOR_TEMPERATURE_KEY)
    assert state is not None


def test_install_meta_belief_second_call_no_op() -> None:
    """Re-install must NOT overwrite the existing row -- accumulated
    posterior mass for the relevance sub-posterior must not be lost."""
    store = MemoryStore(":memory:")
    install_posterior_temperature_meta_belief(store, now_ts=1700000000)
    assert install_posterior_temperature_meta_belief(
        store, now_ts=1700000001
    ) is False


def test_install_meta_belief_idempotent_row_count() -> None:
    """Two installs must produce exactly one meta-belief row."""
    store = MemoryStore(":memory:")
    install_posterior_temperature_meta_belief(store, now_ts=1700000000)
    install_posterior_temperature_meta_belief(store, now_ts=1700000001)
    beliefs = store.list_meta_beliefs()
    matching = [b for b in beliefs if b.key == META_POSTERIOR_TEMPERATURE_KEY]
    assert len(matching) == 1


def test_install_meta_belief_uses_relevance_signal_only() -> None:
    """Single-signal subscription: relevance only, weight 1.0, no other signals.

    Temperature changes the rerank distribution shape; the only natural
    feedback is whether the top-K beliefs surfaced were actually referenced.
    """
    store = MemoryStore(":memory:")
    install_posterior_temperature_meta_belief(store, now_ts=1700000000)
    state = store.read_meta_belief_state(META_POSTERIOR_TEMPERATURE_KEY)
    assert state is not None
    assert state.signal_weights == {SIGNAL_RELEVANCE: 1.0}


def test_install_meta_belief_half_life_and_static_default() -> None:
    """Installed row must carry the ratified 30d half-life and static_default=0.5."""
    store = MemoryStore(":memory:")
    install_posterior_temperature_meta_belief(store, now_ts=1700000000)
    state = store.read_meta_belief_state(META_POSTERIOR_TEMPERATURE_KEY)
    assert state is not None
    assert state.half_life_seconds == 30 * 24 * 3600
    assert state.static_default == 0.5


# ---------------------------------------------------------------------------
# Cold-start byte-identity: decode(0.5) == T = 1.0 exactly
# ---------------------------------------------------------------------------

def test_cold_start_decode_is_1_0() -> None:
    """With meta-belief at static_default=0.5, resolve_posterior_temperature_with_meta
    must return exactly 1.0 -- the geometric mean of FLOOR=0.5 and CEIL=2.0 in
    log space, byte-identical to the log-additive partial_bayesian_score baseline.
    """
    store = MemoryStore(":memory:")
    install_posterior_temperature_meta_belief(store, now_ts=1700000000)
    t = resolve_posterior_temperature_with_meta(store, now_ts=1700000001)
    assert t == pytest.approx(1.0, abs=1e-12), (
        f"cold-start T expected 1.0, got {t}"
    )


def test_cold_start_byte_identity_formula() -> None:
    """Cross-check the geometric-mean formula: exp((log(0.5) + log(2.0)) / 2) == 1.0."""
    geometric_mean = math.exp((math.log(POSTERIOR_TEMPERATURE_FLOOR) + math.log(POSTERIOR_TEMPERATURE_CEIL)) / 2)
    assert geometric_mean == pytest.approx(1.0, abs=1e-12)
    # And the log-linear decode at v=0.5 lands there:
    log_floor = math.log(POSTERIOR_TEMPERATURE_FLOOR)
    log_ceil = math.log(POSTERIOR_TEMPERATURE_CEIL)
    decoded = math.exp(log_floor + META_POSTERIOR_TEMPERATURE_STATIC_DEFAULT * (log_ceil - log_floor))
    assert decoded == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Evidence responsiveness
# ---------------------------------------------------------------------------

def test_strong_positive_relevance_evidence_raises_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """100 strong-positive relevance events (evidence=1.0) push the Beta-
    Bernoulli posterior mean toward 1.0, which decodes via the log-linear
    map to T toward CEIL (flatter distribution).

    Signal direction: evidence=1.0 alpha-increments the sub-posterior,
    pulling the surfaced value toward 1.0, which maps to POSTERIOR_TEMPERATURE_CEIL
    (T=2.0). T after evidence must be >= cold-start T (1.0) and bounded by CEIL.
    """
    store = MemoryStore(":memory:")
    install_posterior_temperature_meta_belief(store, now_ts=1700000000)
    cold_start_t = resolve_posterior_temperature_with_meta(store, now_ts=1700000000)
    base_ts = 1700000001
    for i in range(100):
        store.update_meta_belief(
            META_POSTERIOR_TEMPERATURE_KEY,
            SIGNAL_RELEVANCE,
            evidence=1.0,
            now_ts=base_ts + i,
        )
    after_t = resolve_posterior_temperature_with_meta(store, now_ts=base_ts + 100)
    assert after_t >= cold_start_t, (
        f"after 100 strong-positive relevance events expected T >= {cold_start_t} "
        f"(toward CEIL={POSTERIOR_TEMPERATURE_CEIL}), got T={after_t}"
    )
    assert after_t <= POSTERIOR_TEMPERATURE_CEIL, (
        f"T must stay <= CEIL={POSTERIOR_TEMPERATURE_CEIL}, got {after_t}"
    )


def test_strong_negative_relevance_evidence_lowers_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """100 strong-negative relevance events (evidence=0.0) push the Beta-
    Bernoulli posterior mean toward 0.0, which decodes to T toward FLOOR
    (sharper distribution -- the rerank discriminates more strongly).

    T after evidence must be <= cold-start T (1.0) and bounded by FLOOR.
    """
    store = MemoryStore(":memory:")
    install_posterior_temperature_meta_belief(store, now_ts=1700000000)
    cold_start_t = resolve_posterior_temperature_with_meta(store, now_ts=1700000000)
    base_ts = 1700000001
    for i in range(100):
        store.update_meta_belief(
            META_POSTERIOR_TEMPERATURE_KEY,
            SIGNAL_RELEVANCE,
            evidence=0.0,
            now_ts=base_ts + i,
        )
    after_t = resolve_posterior_temperature_with_meta(store, now_ts=base_ts + 100)
    assert after_t <= cold_start_t, (
        f"after 100 strong-negative relevance events expected T <= {cold_start_t} "
        f"(toward FLOOR={POSTERIOR_TEMPERATURE_FLOOR}), got T={after_t}"
    )
    assert after_t >= POSTERIOR_TEMPERATURE_FLOOR, (
        f"T must stay >= FLOOR={POSTERIOR_TEMPERATURE_FLOOR}, got {after_t}"
    )


# ---------------------------------------------------------------------------
# Determinism (#605): same evidence sequence + same now_ts -> identical value
# ---------------------------------------------------------------------------

def test_determinism_same_evidence_same_result() -> None:
    """Running the install and a fixed evidence sequence twice with the same
    now_ts produces identical read_meta_belief_value (#605 locked contract).
    """
    def _run() -> float | None:
        store = MemoryStore(":memory:")
        install_posterior_temperature_meta_belief(store, now_ts=1700000000)
        for i in range(20):
            store.update_meta_belief(
                META_POSTERIOR_TEMPERATURE_KEY,
                SIGNAL_RELEVANCE,
                evidence=float(i % 2),
                now_ts=1700000001 + i,
            )
        return store.read_meta_belief_value(META_POSTERIOR_TEMPERATURE_KEY, now_ts=1700000100)

    first = _run()
    second = _run()
    assert first is not None
    assert first == second, (
        f"determinism violated: first={first}, second={second}"
    )


# ---------------------------------------------------------------------------
# get_active_meta_belief_consumers integration
# ---------------------------------------------------------------------------

def test_active_consumers_excludes_key_when_flag_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the posterior-temperature env flag is unset, the key must NOT
    appear in get_active_meta_belief_consumers()."""
    from aelfrice.retrieval import (
        ENV_META_BELIEF_BM25F_ANCHOR_WEIGHT,
        ENV_META_BELIEF_HALF_LIFE,
        get_active_meta_belief_consumers,
    )
    monkeypatch.delenv(ENV_META_BELIEF_HALF_LIFE, raising=False)
    monkeypatch.delenv(ENV_META_BELIEF_BM25F_ANCHOR_WEIGHT, raising=False)
    monkeypatch.delenv(ENV_META_BELIEF_POSTERIOR_TEMPERATURE, raising=False)
    consumers = get_active_meta_belief_consumers()
    assert META_POSTERIOR_TEMPERATURE_KEY not in consumers


def test_active_consumers_includes_key_when_flag_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the posterior-temperature env flag is set truthy, the key must appear
    in get_active_meta_belief_consumers() so the sweeper delivers relevance evidence."""
    from aelfrice.retrieval import (
        ENV_META_BELIEF_BM25F_ANCHOR_WEIGHT,
        ENV_META_BELIEF_HALF_LIFE,
        get_active_meta_belief_consumers,
    )
    monkeypatch.delenv(ENV_META_BELIEF_HALF_LIFE, raising=False)
    monkeypatch.delenv(ENV_META_BELIEF_BM25F_ANCHOR_WEIGHT, raising=False)
    monkeypatch.setenv(ENV_META_BELIEF_POSTERIOR_TEMPERATURE, "enabled")
    consumers = get_active_meta_belief_consumers()
    assert META_POSTERIOR_TEMPERATURE_KEY in consumers


def test_active_consumers_sorted_with_multiple_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Consumers list is always alphabetically sorted when multiple flags are on."""
    from aelfrice.retrieval import (
        ENV_META_BELIEF_BM25F_ANCHOR_WEIGHT,
        ENV_META_BELIEF_HALF_LIFE,
        get_active_meta_belief_consumers,
    )
    monkeypatch.setenv(ENV_META_BELIEF_HALF_LIFE, "1")
    monkeypatch.setenv(ENV_META_BELIEF_BM25F_ANCHOR_WEIGHT, "1")
    monkeypatch.setenv(ENV_META_BELIEF_POSTERIOR_TEMPERATURE, "1")
    out = get_active_meta_belief_consumers()
    assert out == sorted(out), f"consumers not sorted: {out}"
    assert META_POSTERIOR_TEMPERATURE_KEY in out
