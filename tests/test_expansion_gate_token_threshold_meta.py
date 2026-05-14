"""Tests for the #760 meta-belief consumer wiring on expansion-gate token threshold.

Sub-task F of umbrella #480. Pattern reuse of the #756
``temporal_half_life_seconds``, #757 ``bm25f_anchor_weight``, and #759
``bfs_depth_budget`` consumers — same shape, scoped to the
``BROAD_PROMPT_TOKEN_THRESHOLD`` integer that ``should_run_expansion`` uses.

Substrate-level coverage (Beta-Bernoulli update, decay, signal-class
enum) lives in ``tests/test_meta_beliefs.py``.
"""
from __future__ import annotations

import math

import pytest

from aelfrice.expansion_gate import BROAD_PROMPT_TOKEN_THRESHOLD
from aelfrice.meta_beliefs import SIGNAL_RELEVANCE
from aelfrice.retrieval import (
    ENV_META_BELIEF_EXPANSION_GATE_TOKEN_THRESHOLD,
    EXPANSION_GATE_TOKEN_THRESHOLD_CEIL,
    EXPANSION_GATE_TOKEN_THRESHOLD_FLOOR,
    META_EXPANSION_GATE_TOKEN_THRESHOLD_KEY,
    META_EXPANSION_GATE_TOKEN_THRESHOLD_POSTERIOR_DECAY_SECONDS,
    META_EXPANSION_GATE_TOKEN_THRESHOLD_STATIC_DEFAULT,
    decode_expansion_gate_token_threshold,
    is_meta_belief_expansion_gate_token_threshold_enabled,
)


# --- Log-linear bounded encoding (#760 pattern reuse of #756/#757/#759 D3) ---

def test_decode_expansion_gate_token_threshold_floor() -> None:
    """v=0.0 must decode to EXPANSION_GATE_TOKEN_THRESHOLD_FLOOR (20)."""
    assert decode_expansion_gate_token_threshold(0.0) == EXPANSION_GATE_TOKEN_THRESHOLD_FLOOR
    assert decode_expansion_gate_token_threshold(0.0) == 20


def test_decode_expansion_gate_token_threshold_ceil() -> None:
    """v=1.0 must decode to EXPANSION_GATE_TOKEN_THRESHOLD_CEIL (320)."""
    assert decode_expansion_gate_token_threshold(1.0) == EXPANSION_GATE_TOKEN_THRESHOLD_CEIL
    assert decode_expansion_gate_token_threshold(1.0) == 320


def test_decode_expansion_gate_token_threshold_mid() -> None:
    """v=0.5 must decode to exactly 80.

    The geometric mean of 20 and 320 is sqrt(6400) = 80.0 exactly,
    so log-linear interpolation at v=0.5 round-trips to 80 without
    floating-point error. This is the byte-identical cold-start property:
    decode(static_default) == BROAD_PROMPT_TOKEN_THRESHOLD.
    """
    result = decode_expansion_gate_token_threshold(0.5)
    assert result == 80
    # Cross-check against the spec formula: geometric mean of floor and ceil
    expected = int(round(math.sqrt(
        EXPANSION_GATE_TOKEN_THRESHOLD_FLOOR * EXPANSION_GATE_TOKEN_THRESHOLD_CEIL,
    )))
    assert result == expected
    # Cold-start byte-identical property: decode(0.5) == BROAD_PROMPT_TOKEN_THRESHOLD
    assert result == BROAD_PROMPT_TOKEN_THRESHOLD


def test_decode_expansion_gate_token_threshold_clamps_below_zero() -> None:
    """Negative values clamp to the floor; no crash on stale rows."""
    assert decode_expansion_gate_token_threshold(-0.5) == decode_expansion_gate_token_threshold(0.0)
    assert decode_expansion_gate_token_threshold(-1.0) == EXPANSION_GATE_TOKEN_THRESHOLD_FLOOR


def test_decode_expansion_gate_token_threshold_clamps_above_one() -> None:
    """Values >1 clamp to the ceil; no crash on stale rows."""
    assert decode_expansion_gate_token_threshold(1.5) == decode_expansion_gate_token_threshold(1.0)
    assert decode_expansion_gate_token_threshold(float("inf")) == EXPANSION_GATE_TOKEN_THRESHOLD_CEIL


def test_decode_expansion_gate_token_threshold_returns_int() -> None:
    """``should_run_expansion`` compares len(tokens) > threshold.

    Decode must return ``int`` not ``float`` at every sample point.
    """
    for v in (0.0, 0.123, 0.5, 0.876, 1.0):
        result = decode_expansion_gate_token_threshold(v)
        assert isinstance(result, int), (v, type(result))


def test_decode_expansion_gate_token_threshold_result_in_bounds() -> None:
    """All decoded values must lie in [FLOOR, CEIL] = [20, 320]."""
    for k in range(101):
        result = decode_expansion_gate_token_threshold(k / 100.0)
        assert EXPANSION_GATE_TOKEN_THRESHOLD_FLOOR <= result <= EXPANSION_GATE_TOKEN_THRESHOLD_CEIL, (
            k, result
        )


def test_decode_expansion_gate_token_threshold_monotonic() -> None:
    """Higher value → larger threshold, monotonic non-decreasing.

    Rounding to int means strict monotonicity holds only at the float
    level; the int step function must never go backwards.
    """
    previous = decode_expansion_gate_token_threshold(0.0)
    for k in range(1, 101):
        current = decode_expansion_gate_token_threshold(k / 100.0)
        assert current >= previous, (k, previous, current)
        previous = current


# --- Env feature flag (#760 default-OFF) ----------------------------------

def test_meta_flag_unset_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV_META_BELIEF_EXPANSION_GATE_TOKEN_THRESHOLD, raising=False)
    assert is_meta_belief_expansion_gate_token_threshold_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "enabled"])
def test_meta_flag_truthy_tokens(
    monkeypatch: pytest.MonkeyPatch, value: str,
) -> None:
    monkeypatch.setenv(ENV_META_BELIEF_EXPANSION_GATE_TOKEN_THRESHOLD, value)
    assert is_meta_belief_expansion_gate_token_threshold_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "disabled", ""])
def test_meta_flag_falsy_tokens(
    monkeypatch: pytest.MonkeyPatch, value: str,
) -> None:
    monkeypatch.setenv(ENV_META_BELIEF_EXPANSION_GATE_TOKEN_THRESHOLD, value)
    assert is_meta_belief_expansion_gate_token_threshold_enabled() is False


# --- Constants pin (#760 ratification) ------------------------------------

def test_meta_constants_match_ratification() -> None:
    """Pin the #760 ratified defaults so a careless edit surfaces here,
    not in a downstream gate regression."""
    assert META_EXPANSION_GATE_TOKEN_THRESHOLD_KEY == (
        "meta:retrieval.expansion_gate.token_threshold"
    )
    assert EXPANSION_GATE_TOKEN_THRESHOLD_FLOOR == 20
    assert EXPANSION_GATE_TOKEN_THRESHOLD_CEIL == 320
    assert META_EXPANSION_GATE_TOKEN_THRESHOLD_STATIC_DEFAULT == 0.5
    assert META_EXPANSION_GATE_TOKEN_THRESHOLD_POSTERIOR_DECAY_SECONDS == 30 * 24 * 3600


def test_cold_start_byte_identical_to_hardcoded() -> None:
    """decode(STATIC_DEFAULT) == BROAD_PROMPT_TOKEN_THRESHOLD — the
    core invariant of the #760 spec. A cold-start install with the
    meta-belief flag on must be expansion-gate byte-identical to a
    flag-off install on the same corpus."""
    assert decode_expansion_gate_token_threshold(
        META_EXPANSION_GATE_TOKEN_THRESHOLD_STATIC_DEFAULT,
    ) == BROAD_PROMPT_TOKEN_THRESHOLD


# --- Signal class -----------------------------------------------------------

def test_signal_relevance_constant_available() -> None:
    """SIGNAL_RELEVANCE must be importable — it's the only subscribed class
    for the #760 install helper (shipped in Commit 2)."""
    assert SIGNAL_RELEVANCE == "relevance"
