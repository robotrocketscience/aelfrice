"""Tests for the #757 meta-belief consumer wiring on BM25F anchor_weight.

Sub-task C of umbrella #480. Pattern reuse of the #756
``temporal_half_life_seconds`` consumer — same shape, scoped to the
``anchor_weight`` integer that ``BM25IndexCache`` accepts.

Substrate-level coverage (Beta-Bernoulli update, decay, signal-class
enum) lives in ``tests/test_meta_beliefs.py``.
"""
from __future__ import annotations

import math

import pytest

from aelfrice.bm25 import BM25IndexCache, DEFAULT_ANCHOR_WEIGHT
from aelfrice.meta_beliefs import SIGNAL_BM25_L0_RATIO
from aelfrice.retrieval import (
    BM25F_ANCHOR_WEIGHT_CEIL,
    BM25F_ANCHOR_WEIGHT_FLOOR,
    ENV_META_BELIEF_BM25F_ANCHOR_WEIGHT,
    META_BM25F_ANCHOR_WEIGHT_KEY,
    META_BM25F_ANCHOR_WEIGHT_POSTERIOR_DECAY_SECONDS,
    META_BM25F_ANCHOR_WEIGHT_STATIC_DEFAULT,
    decode_meta_bm25f_anchor_weight,
    install_bm25f_anchor_weight_meta_belief,
    is_meta_belief_bm25f_anchor_weight_enabled,
    resolve_bm25f_anchor_weight_with_meta,
)
from aelfrice.store import MemoryStore


# --- Log-linear bounded encoding (#757 pattern reuse of #756 D3) -------

def test_decode_anchor_weight_floor() -> None:
    assert decode_meta_bm25f_anchor_weight(0.0) == BM25F_ANCHOR_WEIGHT_FLOOR


def test_decode_anchor_weight_ceil() -> None:
    assert decode_meta_bm25f_anchor_weight(1.0) == BM25F_ANCHOR_WEIGHT_CEIL


def test_decode_anchor_weight_mid_matches_static_default() -> None:
    """v=0.5 decodes to ``DEFAULT_ANCHOR_WEIGHT`` exactly (3).

    Cold-start parity invariant: on first install, the meta-belief's
    static_default of 0.5 must yield the same anchor_weight as the
    pre-#757 hardcoded constant. Otherwise installing the meta-belief
    would silently shift retrieval order before any evidence accrues.

    Log-linear midpoint between 1 and 10 is sqrt(1*10) ≈ 3.16; the
    rounding rule lands it on 3.
    """
    assert decode_meta_bm25f_anchor_weight(0.5) == DEFAULT_ANCHOR_WEIGHT
    assert decode_meta_bm25f_anchor_weight(0.5) == round(
        math.sqrt(BM25F_ANCHOR_WEIGHT_FLOOR * BM25F_ANCHOR_WEIGHT_CEIL)
    )


def test_decode_anchor_weight_clamps_out_of_range() -> None:
    """Substrate guarantees `value` in [0, 1] but defend at the boundary.

    Retrieval must never crash on a stale meta-belief row whose ``value``
    drifted outside the unit interval (e.g. a hand-written row pre-clamp).
    """
    assert decode_meta_bm25f_anchor_weight(-1.0) == decode_meta_bm25f_anchor_weight(0.0)
    assert decode_meta_bm25f_anchor_weight(2.0) == decode_meta_bm25f_anchor_weight(1.0)
    assert decode_meta_bm25f_anchor_weight(float("inf")) == decode_meta_bm25f_anchor_weight(1.0)


def test_decode_anchor_weight_monotonic_non_decreasing() -> None:
    """Higher value → larger anchor_weight, monotonic non-decreasing
    over `[0, 1]`. The rounding to int means strict monotonicity holds
    only at the float level; at the int level we get a step function
    that must never go backwards.
    """
    previous = decode_meta_bm25f_anchor_weight(0.0)
    for k in range(1, 101):
        current = decode_meta_bm25f_anchor_weight(k / 100.0)
        assert current >= previous, (k, previous, current)
        previous = current


def test_decode_anchor_weight_returns_int() -> None:
    """``BM25Index.build`` accepts only ``int`` anchor_weight (it
    replicates the anchor token stream that many times — see bm25.py
    line 243). Decode must therefore return ``int`` not ``float``.
    """
    for v in (0.0, 0.123, 0.5, 0.876, 1.0):
        result = decode_meta_bm25f_anchor_weight(v)
        assert isinstance(result, int), (v, type(result))


# --- Env feature flag (#757 default-OFF) -------------------------------

def test_meta_flag_unset_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV_META_BELIEF_BM25F_ANCHOR_WEIGHT, raising=False)
    assert is_meta_belief_bm25f_anchor_weight_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "enabled"])
def test_meta_flag_truthy_tokens(
    monkeypatch: pytest.MonkeyPatch, value: str,
) -> None:
    monkeypatch.setenv(ENV_META_BELIEF_BM25F_ANCHOR_WEIGHT, value)
    assert is_meta_belief_bm25f_anchor_weight_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "disabled", ""])
def test_meta_flag_falsy_tokens(
    monkeypatch: pytest.MonkeyPatch, value: str,
) -> None:
    monkeypatch.setenv(ENV_META_BELIEF_BM25F_ANCHOR_WEIGHT, value)
    assert is_meta_belief_bm25f_anchor_weight_enabled() is False


# --- Constants sanity --------------------------------------------------

def test_meta_constants_match_ratification() -> None:
    """Pin the #757 ratified defaults so a careless edit on the
    consumer-side surfaces here, not in a downstream surprise."""
    assert META_BM25F_ANCHOR_WEIGHT_KEY == "meta:retrieval.bm25f_anchor_weight"
    assert BM25F_ANCHOR_WEIGHT_FLOOR == 1
    assert BM25F_ANCHOR_WEIGHT_CEIL == 10
    assert META_BM25F_ANCHOR_WEIGHT_STATIC_DEFAULT == 0.5
    assert META_BM25F_ANCHOR_WEIGHT_POSTERIOR_DECAY_SECONDS == 30 * 24 * 3600


# --- Install helper (idempotency) --------------------------------------

def test_install_meta_belief_first_call_returns_true() -> None:
    store = MemoryStore(":memory:")
    assert install_bm25f_anchor_weight_meta_belief(
        store, now_ts=1700000000
    ) is True
    state = store.read_meta_belief_state(META_BM25F_ANCHOR_WEIGHT_KEY)
    assert state is not None


def test_install_meta_belief_second_call_no_op() -> None:
    """Re-install must NOT overwrite the existing row. The surfaced
    anchor_weight would silently shift under BM25F otherwise — and the
    posterior accumulated so far would be lost."""
    store = MemoryStore(":memory:")
    install_bm25f_anchor_weight_meta_belief(store, now_ts=1700000000)
    assert install_bm25f_anchor_weight_meta_belief(
        store, now_ts=1700000001
    ) is False


# --- Meta-aware resolver (precedence + fallback) -----------------------

def test_resolver_no_store_falls_through_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``store=None`` collapses to the static default unchanged. Lets
    bare callers (CLI tooling, smoke tests) keep working without a
    store handle."""
    monkeypatch.delenv(ENV_META_BELIEF_BM25F_ANCHOR_WEIGHT, raising=False)
    assert resolve_bm25f_anchor_weight_with_meta(
        None, now_ts=1700000000
    ) == DEFAULT_ANCHOR_WEIGHT


def test_resolver_flag_off_ignores_meta_belief(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default-OFF invariant: with the env flag unset, an installed
    meta-belief is invisible to the resolver."""
    monkeypatch.delenv(ENV_META_BELIEF_BM25F_ANCHOR_WEIGHT, raising=False)
    store = MemoryStore(":memory:")
    install_bm25f_anchor_weight_meta_belief(store, now_ts=1700000000)
    assert resolve_bm25f_anchor_weight_with_meta(
        store, now_ts=1700000001
    ) == DEFAULT_ANCHOR_WEIGHT


def test_resolver_flag_on_cold_start_matches_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flag-on + freshly-installed meta-belief at static_default=0.5
    yields the same anchor_weight as the pre-#757 hardcoded path.
    Bench-gate safety: flipping the flag on for the first time must
    not shift BM25F's output before any evidence accrues."""
    monkeypatch.setenv(ENV_META_BELIEF_BM25F_ANCHOR_WEIGHT, "enabled")
    store = MemoryStore(":memory:")
    install_bm25f_anchor_weight_meta_belief(store, now_ts=1700000000)
    assert resolve_bm25f_anchor_weight_with_meta(
        store, now_ts=1700000001
    ) == DEFAULT_ANCHOR_WEIGHT


def test_resolver_explicit_kwarg_wins_over_meta_belief(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit kwarg is the only operator-side override path. It must
    beat the meta-belief even when the flag is on."""
    monkeypatch.setenv(ENV_META_BELIEF_BM25F_ANCHOR_WEIGHT, "enabled")
    store = MemoryStore(":memory:")
    install_bm25f_anchor_weight_meta_belief(store, now_ts=1700000000)
    assert resolve_bm25f_anchor_weight_with_meta(
        store, now_ts=1700000001, explicit=7,
    ) == 7


def test_resolver_flag_on_meta_uninstalled_falls_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flag-on but no meta-belief row: resolver falls through to
    DEFAULT_ANCHOR_WEIGHT cleanly. No spurious row materialises."""
    monkeypatch.setenv(ENV_META_BELIEF_BM25F_ANCHOR_WEIGHT, "enabled")
    store = MemoryStore(":memory:")
    assert resolve_bm25f_anchor_weight_with_meta(
        store, now_ts=1700000000,
    ) == DEFAULT_ANCHOR_WEIGHT
    # And no row was implicitly installed.
    assert store.read_meta_belief_state(META_BM25F_ANCHOR_WEIGHT_KEY) is None


# --- _l1_hits integration (default-OFF + signal update) ----------------
#
# These exercise the actual BM25F branch of _l1_hits that #757 amends.
# The default-OFF case is the safety invariant — bench-gated features
# must not perturb retrieval output until operators opt in.

from aelfrice.models import (  # noqa: E402 — kept local to the integration block
    BELIEF_FACTUAL,
    LOCK_NONE,
    LOCK_USER,
    RETENTION_FACT,
    Belief,
)
from aelfrice.retrieval import _l1_hits


def _mk_belief(
    bid: str, content: str, *, lock_level: int = LOCK_NONE,
) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=lock_level,
        locked_at=None,
        created_at="2023-11-14T22:13:20+00:00",
        last_retrieved_at=None,
        retention_class=RETENTION_FACT,
    )


def _populate_bm25f_store() -> MemoryStore:
    """Store with two locked beliefs and four unlocked beliefs sharing
    the same query keyword. BM25F's top-K should normally surface the
    locked ones; that's what `bm25_l0_ratio = 1.0` looks like."""
    store = MemoryStore(":memory:")
    store.insert_belief(_mk_belief(
        "L1", "alpha probe locked content one", lock_level=LOCK_USER,
    ))
    store.insert_belief(_mk_belief(
        "L2", "alpha probe locked content two", lock_level=LOCK_USER,
    ))
    for i in range(4):
        store.insert_belief(_mk_belief(
            f"U{i}", f"alpha probe unlocked content {i}",
        ))
    return store


def test_l1_hits_default_off_uses_static_anchor_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bench-gate safety: with the meta-flag unset, _l1_hits constructs
    a BM25IndexCache identical to the pre-#757 BM25IndexCache(store)
    call — that is, with anchor_weight=DEFAULT_ANCHOR_WEIGHT."""
    monkeypatch.delenv(ENV_META_BELIEF_BM25F_ANCHOR_WEIGHT, raising=False)
    store = _populate_bm25f_store()
    # Install the meta-belief but don't enable the flag. Default-OFF
    # arm must still produce identical output to an arm without the
    # row at all.
    install_bm25f_anchor_weight_meta_belief(store, now_ts=1700000000)
    out = _l1_hits(
        store, "alpha probe", l1_limit=10,
        posterior_weight=0.0, use_bm25f_anchors=True,
    )
    assert out  # non-empty
    state = store.read_meta_belief_state(META_BM25F_ANCHOR_WEIGHT_KEY)
    # Flag-off: no signal update fires, so the row stays at the
    # cold-start posterior (empty posteriors dict).
    assert state is not None
    assert SIGNAL_BM25_L0_RATIO not in state.posteriors


def test_l1_hits_meta_flag_on_records_bm25_l0_ratio_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: with flag on + meta-belief installed, _l1_hits
    persists a bm25_l0_ratio sub-posterior on the BM25F branch."""
    monkeypatch.setenv(ENV_META_BELIEF_BM25F_ANCHOR_WEIGHT, "enabled")
    store = _populate_bm25f_store()
    install_bm25f_anchor_weight_meta_belief(store, now_ts=1700000000)
    state_before = store.read_meta_belief_state(META_BM25F_ANCHOR_WEIGHT_KEY)
    assert state_before is not None
    assert SIGNAL_BM25_L0_RATIO not in state_before.posteriors

    _l1_hits(
        store, "alpha probe", l1_limit=10,
        posterior_weight=0.0, use_bm25f_anchors=True,
    )

    state_after = store.read_meta_belief_state(META_BM25F_ANCHOR_WEIGHT_KEY)
    assert state_after is not None
    assert SIGNAL_BM25_L0_RATIO in state_after.posteriors
    # The cold-start prior mass for static_default=0.5 is (0.5, 0.5)
    # (mu * PRIOR_MASS, (1-mu) * PRIOR_MASS with PRIOR_MASS=1). Either
    # alpha or beta strictly grew on this single query.
    p = state_after.posteriors[SIGNAL_BM25_L0_RATIO]
    assert (p.alpha + p.beta) > 1.0, (p.alpha, p.beta)


def test_l1_hits_meta_flag_on_no_install_no_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flag-on but meta-belief uninstalled: _l1_hits must still return
    cleanly. update_meta_belief returns False on a missing key, so
    there's no crash and no row materialises."""
    monkeypatch.setenv(ENV_META_BELIEF_BM25F_ANCHOR_WEIGHT, "enabled")
    store = _populate_bm25f_store()  # no install
    out = _l1_hits(
        store, "alpha probe", l1_limit=10,
        posterior_weight=0.0, use_bm25f_anchors=True,
    )
    assert out  # non-empty
    assert store.read_meta_belief_state(META_BM25F_ANCHOR_WEIGHT_KEY) is None


def test_l1_hits_explicit_cache_bypasses_meta_belief(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller-supplied bm25f_cache is the bench harness's pin point;
    even with the flag on, the resolver must not override the explicit
    cache's anchor_weight. This is the contract bench tests depend on
    to compare anchor_weight A/B arms deterministically."""
    monkeypatch.setenv(ENV_META_BELIEF_BM25F_ANCHOR_WEIGHT, "enabled")
    store = _populate_bm25f_store()
    install_bm25f_anchor_weight_meta_belief(store, now_ts=1700000000)
    pinned_cache = BM25IndexCache(store, anchor_weight=7)
    out = _l1_hits(
        store, "alpha probe", l1_limit=10,
        posterior_weight=0.0, use_bm25f_anchors=True,
        bm25f_cache=pinned_cache,
    )
    assert out  # non-empty
    assert pinned_cache.anchor_weight == 7  # untouched by the resolver
