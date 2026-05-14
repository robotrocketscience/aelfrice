"""Tests for the #759 meta-belief consumer wiring on BFS depth budget.

Sub-task E of umbrella #480. Pattern reuse of the #756
``temporal_half_life_seconds`` and #757 ``bm25f_anchor_weight``
consumers — same shape, scoped to the ``bfs_max_depth`` integer that
``expand_bfs`` accepts.

Substrate-level coverage (Beta-Bernoulli update, decay, signal-class
enum) lives in ``tests/test_meta_beliefs.py``.
"""
from __future__ import annotations

import math

import pytest

from aelfrice.bfs_multihop import DEFAULT_MAX_DEPTH as BFS_DEFAULT_MAX_DEPTH
from aelfrice.meta_beliefs import SIGNAL_LATENCY
from aelfrice.retrieval import (
    BFS_DEPTH_BUDGET_CEIL,
    BFS_DEPTH_BUDGET_FLOOR,
    ENV_META_BELIEF_BFS_DEPTH_BUDGET,
    META_BFS_DEPTH_BUDGET_KEY,
    META_BFS_DEPTH_BUDGET_POSTERIOR_DECAY_SECONDS,
    META_BFS_DEPTH_BUDGET_STATIC_DEFAULT,
    decode_bfs_depth_budget,
    install_bfs_depth_budget_meta_belief,
    is_meta_belief_bfs_depth_budget_enabled,
    resolve_bfs_depth_budget_with_meta,
)
from aelfrice.store import MemoryStore


# --- Log-linear bounded encoding (#759 pattern reuse of #756/#757 D3) ---

def test_decode_bfs_depth_budget_floor() -> None:
    """v=0.0 must decode to BFS_DEPTH_BUDGET_FLOOR (1)."""
    assert decode_bfs_depth_budget(0.0) == BFS_DEPTH_BUDGET_FLOOR
    assert decode_bfs_depth_budget(0.0) == 1


def test_decode_bfs_depth_budget_ceil() -> None:
    """v=1.0 must decode to BFS_DEPTH_BUDGET_CEIL (6)."""
    assert decode_bfs_depth_budget(1.0) == BFS_DEPTH_BUDGET_CEIL
    assert decode_bfs_depth_budget(1.0) == 6


def test_decode_bfs_depth_budget_mid() -> None:
    """v=0.5 decodes to 2 (round(sqrt(1*6)) = round(2.449) = 2).

    The spec rationale: cold-start with the meta-belief on gently shifts
    the effective depth one hop below BFS_DEFAULT_MAX_DEPTH (2) — which
    happens to also be 2. So at cold start, the depth is byte-identical
    to the pre-#759 default. The spec documents this as intentional.
    """
    result = decode_bfs_depth_budget(0.5)
    assert result == 2
    # Cross-check against the spec formula
    expected = int(round(math.sqrt(BFS_DEPTH_BUDGET_FLOOR * BFS_DEPTH_BUDGET_CEIL)))
    assert result == expected


def test_decode_bfs_depth_budget_clamps_below_zero() -> None:
    """Negative values clamp to the floor; no crash on stale rows."""
    assert decode_bfs_depth_budget(-1.0) == decode_bfs_depth_budget(0.0)


def test_decode_bfs_depth_budget_clamps_above_one() -> None:
    """Values >1 clamp to the ceil; no crash on stale rows."""
    assert decode_bfs_depth_budget(2.0) == decode_bfs_depth_budget(1.0)
    assert decode_bfs_depth_budget(float("inf")) == decode_bfs_depth_budget(1.0)


def test_decode_bfs_depth_budget_returns_int() -> None:
    """``expand_bfs`` accepts only ``int max_depth``.

    Decode must return ``int`` not ``float`` at every sample point.
    """
    for v in (0.0, 0.123, 0.5, 0.876, 1.0):
        result = decode_bfs_depth_budget(v)
        assert isinstance(result, int), (v, type(result))


def test_decode_bfs_depth_budget_result_in_bounds() -> None:
    """All decoded values must lie in [FLOOR, CEIL] = [1, 6]."""
    for k in range(101):
        result = decode_bfs_depth_budget(k / 100.0)
        assert BFS_DEPTH_BUDGET_FLOOR <= result <= BFS_DEPTH_BUDGET_CEIL, (
            k, result
        )


def test_decode_bfs_depth_budget_monotonic_non_decreasing() -> None:
    """Higher value → larger depth, monotonic non-decreasing.

    Rounding to int means strict monotonicity holds only at the float
    level; the int step function must never go backwards.
    """
    previous = decode_bfs_depth_budget(0.0)
    for k in range(1, 101):
        current = decode_bfs_depth_budget(k / 100.0)
        assert current >= previous, (k, previous, current)
        previous = current


def test_decode_bfs_depth_budget_rounding_at_band_boundaries() -> None:
    """Rounding-band check: step from 1→2 somewhere in (0, 0.5)."""
    # decode at 0.0 is 1, at 0.5 is 2; a transition must exist
    transition_seen = False
    prev = decode_bfs_depth_budget(0.0)
    for k in range(1, 51):
        cur = decode_bfs_depth_budget(k / 100.0)
        if cur > prev:
            transition_seen = True
        prev = cur
    assert transition_seen, "expected 1→2 transition in v in (0, 0.5]"


# --- Env feature flag (#759 default-OFF) ----------------------------------

def test_meta_flag_unset_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV_META_BELIEF_BFS_DEPTH_BUDGET, raising=False)
    assert is_meta_belief_bfs_depth_budget_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "enabled"])
def test_meta_flag_truthy_tokens(
    monkeypatch: pytest.MonkeyPatch, value: str,
) -> None:
    monkeypatch.setenv(ENV_META_BELIEF_BFS_DEPTH_BUDGET, value)
    assert is_meta_belief_bfs_depth_budget_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "disabled", ""])
def test_meta_flag_falsy_tokens(
    monkeypatch: pytest.MonkeyPatch, value: str,
) -> None:
    monkeypatch.setenv(ENV_META_BELIEF_BFS_DEPTH_BUDGET, value)
    assert is_meta_belief_bfs_depth_budget_enabled() is False


# --- Constants pin (#759 ratification) ------------------------------------

def test_meta_constants_match_ratification() -> None:
    """Pin the #759 ratified defaults so a careless edit surfaces here,
    not in a downstream latency regression."""
    assert META_BFS_DEPTH_BUDGET_KEY == "meta:retrieval.bfs_depth_budget"
    assert BFS_DEPTH_BUDGET_FLOOR == 1
    assert BFS_DEPTH_BUDGET_CEIL == 6
    assert META_BFS_DEPTH_BUDGET_STATIC_DEFAULT == 0.5
    assert META_BFS_DEPTH_BUDGET_POSTERIOR_DECAY_SECONDS == 30 * 24 * 3600


# --- Install helper (idempotency) -----------------------------------------

def test_install_meta_belief_first_call_returns_true() -> None:
    store = MemoryStore(":memory:")
    assert install_bfs_depth_budget_meta_belief(store, now_ts=1700000000) is True
    state = store.read_meta_belief_state(META_BFS_DEPTH_BUDGET_KEY)
    assert state is not None


def test_install_meta_belief_second_call_no_op() -> None:
    """Re-install must NOT overwrite the existing row. The surfaced depth
    budget would silently shift under BFS otherwise — and accumulated
    posterior mass would be lost."""
    store = MemoryStore(":memory:")
    install_bfs_depth_budget_meta_belief(store, now_ts=1700000000)
    assert install_bfs_depth_budget_meta_belief(store, now_ts=1700000001) is False


def test_install_meta_belief_uses_latency_signal() -> None:
    """MVP ships with latency only (bfs_depth deferred to #779)."""
    store = MemoryStore(":memory:")
    install_bfs_depth_budget_meta_belief(store, now_ts=1700000000)
    state = store.read_meta_belief_state(META_BFS_DEPTH_BUDGET_KEY)
    assert state is not None
    # latency is the only subscribed signal class
    assert state.signal_weights == {SIGNAL_LATENCY: 1.0}


# --- Meta-aware resolver (precedence + fallback) --------------------------

def test_resolver_no_store_falls_through_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """store=None collapses to BFS_DEFAULT_MAX_DEPTH unchanged."""
    monkeypatch.delenv(ENV_META_BELIEF_BFS_DEPTH_BUDGET, raising=False)
    assert resolve_bfs_depth_budget_with_meta(
        None, now_ts=1700000000
    ) == BFS_DEFAULT_MAX_DEPTH


def test_resolver_flag_off_ignores_installed_meta_belief(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default-OFF invariant: with the env flag unset, an installed
    meta-belief is invisible to the resolver."""
    monkeypatch.delenv(ENV_META_BELIEF_BFS_DEPTH_BUDGET, raising=False)
    store = MemoryStore(":memory:")
    install_bfs_depth_budget_meta_belief(store, now_ts=1700000000)
    assert resolve_bfs_depth_budget_with_meta(
        store, now_ts=1700000001
    ) == BFS_DEFAULT_MAX_DEPTH


def test_resolver_flag_on_cold_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flag-on + freshly-installed meta-belief at static_default=0.5.

    decode_bfs_depth_budget(0.5) = 2. BFS_DEFAULT_MAX_DEPTH is also 2,
    so flipping the flag on for the first time is byte-identical.
    """
    monkeypatch.setenv(ENV_META_BELIEF_BFS_DEPTH_BUDGET, "enabled")
    store = MemoryStore(":memory:")
    install_bfs_depth_budget_meta_belief(store, now_ts=1700000000)
    cold_start_depth = resolve_bfs_depth_budget_with_meta(
        store, now_ts=1700000001
    )
    assert cold_start_depth == decode_bfs_depth_budget(META_BFS_DEPTH_BUDGET_STATIC_DEFAULT)


def test_resolver_explicit_non_default_kwarg_bypasses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit non-default kwarg wins over meta-belief (bench override)."""
    monkeypatch.setenv(ENV_META_BELIEF_BFS_DEPTH_BUDGET, "enabled")
    store = MemoryStore(":memory:")
    install_bfs_depth_budget_meta_belief(store, now_ts=1700000000)
    assert resolve_bfs_depth_budget_with_meta(
        store, now_ts=1700000001, explicit=5,
    ) == 5


def test_resolver_flag_on_meta_uninstalled_falls_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flag-on but no meta-belief row: resolver falls through to
    BFS_DEFAULT_MAX_DEPTH cleanly. No spurious row materialises."""
    monkeypatch.setenv(ENV_META_BELIEF_BFS_DEPTH_BUDGET, "enabled")
    store = MemoryStore(":memory:")
    assert resolve_bfs_depth_budget_with_meta(
        store, now_ts=1700000000,
    ) == BFS_DEFAULT_MAX_DEPTH
    assert store.read_meta_belief_state(META_BFS_DEPTH_BUDGET_KEY) is None


# --- End-to-end: evidence responsiveness ----------------------------------

def test_strong_positive_latency_evidence_resolved_depth_ge_cold_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """100 strong-positive latency events (evidence=1.0) → the posterior
    should pull the resolved depth toward CEIL (6), so resolved depth
    must be >= the cold-start depth (2).

    This is the responsiveness gate from the #759 spec.
    """
    monkeypatch.setenv(ENV_META_BELIEF_BFS_DEPTH_BUDGET, "enabled")
    store = MemoryStore(":memory:")
    install_bfs_depth_budget_meta_belief(store, now_ts=1700000000)
    cold_start_depth = resolve_bfs_depth_budget_with_meta(
        store, now_ts=1700000000
    )
    base_ts = 1700000001
    for i in range(100):
        store.update_meta_belief(
            META_BFS_DEPTH_BUDGET_KEY,
            SIGNAL_LATENCY,
            evidence=1.0,
            now_ts=base_ts + i,
        )
    after_depth = resolve_bfs_depth_budget_with_meta(
        store, now_ts=base_ts + 100
    )
    assert after_depth >= cold_start_depth, (
        f"after 100 strong-positive events expected depth >= {cold_start_depth}, "
        f"got {after_depth}"
    )


# --- Default-OFF byte-identical retrieve_v2 --------------------------------

from aelfrice.models import (  # noqa: E402
    BELIEF_FACTUAL,
    LOCK_NONE,
    RETENTION_FACT,
    Belief,
)
from aelfrice.retrieval import retrieve_v2


def _mk_belief(bid: str, content: str) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2023-11-14T22:13:20+00:00",
        last_retrieved_at=None,
        retention_class=RETENTION_FACT,
    )


def _make_populated_store() -> MemoryStore:
    store = MemoryStore(":memory:")
    for i in range(5):
        store.insert_belief(_mk_belief(f"b{i}", f"alpha bravo charlie item {i}"))
    return store


def test_retrieve_v2_default_off_byte_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default-OFF: retrieve_v2 with the meta flag unset must return the
    same belief IDs in the same order with and without the meta-belief
    installed."""
    monkeypatch.delenv(ENV_META_BELIEF_BFS_DEPTH_BUDGET, raising=False)
    store = _make_populated_store()

    result_no_meta = retrieve_v2(
        store, "alpha bravo", budget=2000, now_ts=1700000000,
    )

    install_bfs_depth_budget_meta_belief(store, now_ts=1700000000)
    result_with_meta = retrieve_v2(
        store, "alpha bravo", budget=2000, now_ts=1700000000,
    )

    ids_no_meta = [b.id for b in result_no_meta.beliefs]
    ids_with_meta = [b.id for b in result_with_meta.beliefs]
    assert ids_no_meta == ids_with_meta, (
        f"flag-off retrieve_v2 changed output after meta-belief install: "
        f"{ids_no_meta} vs {ids_with_meta}"
    )


# --- Determinism: same evidence + same now_ts → same resolved depth ------

def test_determinism_same_evidence_same_depth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two fresh stores that receive the same evidence sequence with the
    same now_ts values must resolve to the same depth."""
    monkeypatch.setenv(ENV_META_BELIEF_BFS_DEPTH_BUDGET, "enabled")

    evidence_seq = [0.8, 0.9, 0.7, 1.0, 0.85, 0.6, 0.95, 0.75, 1.0, 0.8]
    base_ts = 1700000000

    def _build_store() -> MemoryStore:
        s = MemoryStore(":memory:")
        install_bfs_depth_budget_meta_belief(s, now_ts=base_ts)
        for i, ev in enumerate(evidence_seq):
            s.update_meta_belief(
                META_BFS_DEPTH_BUDGET_KEY,
                SIGNAL_LATENCY,
                evidence=ev,
                now_ts=base_ts + i + 1,
            )
        return s

    store_a = _build_store()
    store_b = _build_store()

    query_ts = base_ts + len(evidence_seq) + 1
    depth_a = resolve_bfs_depth_budget_with_meta(store_a, now_ts=query_ts)
    depth_b = resolve_bfs_depth_budget_with_meta(store_b, now_ts=query_ts)
    assert depth_a == depth_b, (depth_a, depth_b)
