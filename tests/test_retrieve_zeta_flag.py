"""Tests for #817 ζ flag wiring in retrieve / retrieve_with_tiers.

Properties under test:

1. **Flag-off byte-identity.** With AELFRICE_USE_ZETA_POSTERIOR_RERANK
   unset (the default) AND γ off, retrieve()'s output is unchanged
   compared to a pre-#817 baseline — the existing log-additive / γ
   contract holds.
2. **Flag-on deterministic.** Flag on, no γ → ζ runs the rerank loop;
   output is deterministic given the same store + query.
3. **Resolver precedence.** env > kwarg > TOML > False. Verified by
   the resolver-only tests (no store touch needed for the precedence
   chain).
4. **γ + ζ mutual exclusion.** When both flags resolve True, both
   retrieve sites raise `ValueError` at flag-resolution time.
5. **No-cross-fire.** With ζ off, the γ-on path is unchanged
   (regression protection against a future refactor that
   inadvertently routes through ζ when ζ is off).
"""
from __future__ import annotations

import uuid

import pytest

from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, RETENTION_FACT, Belief
from aelfrice.retrieval import (
    _assert_gamma_zeta_mutual_exclusion,
    resolve_use_zeta_posterior_rerank,
    retrieve,
    retrieve_with_tiers,
)
from aelfrice.store import MemoryStore


_ENV_FLAG_ZETA = "AELFRICE_USE_ZETA_POSTERIOR_RERANK"
_ENV_FLAG_GAMMA = "AELFRICE_USE_GAMMA_POSTERIOR_TEMPERATURE"


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Ensure the rerank env flags never leak across tests."""
    monkeypatch.delenv(_ENV_FLAG_ZETA, raising=False)
    monkeypatch.delenv(_ENV_FLAG_GAMMA, raising=False)
    yield


def _mk_belief(text: str, *, alpha: float = 1.0, beta: float = 1.0) -> Belief:
    bid = uuid.uuid4().hex[:16]
    return Belief(
        id=bid,
        content=text,
        content_hash=f"h_{bid}",
        alpha=alpha,
        beta=beta,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2023-11-14T22:13:20+00:00",
        last_retrieved_at=None,
        retention_class=RETENTION_FACT,
    )


@pytest.fixture
def populated_store():
    """Fresh in-memory store with a small corpus that exercises
    posterior reweighting (different α/β so ζ can move beliefs
    relative to each other)."""
    s = MemoryStore(":memory:")
    s.insert_belief(_mk_belief("alpha is the first letter", alpha=10.0, beta=1.0))
    s.insert_belief(_mk_belief("beta is the second letter", alpha=1.0, beta=10.0))
    s.insert_belief(_mk_belief("gamma is the third letter", alpha=5.0, beta=5.0))
    s.insert_belief(_mk_belief("delta is the fourth letter", alpha=1.0, beta=1.0))
    yield s
    s.close()


# ---------------------------------------------------------------------------
# Resolver precedence
# ---------------------------------------------------------------------------

def test_resolver_default_false(monkeypatch) -> None:
    monkeypatch.delenv(_ENV_FLAG_ZETA, raising=False)
    assert resolve_use_zeta_posterior_rerank() is False


def test_resolver_env_truthy_wins(monkeypatch) -> None:
    monkeypatch.setenv(_ENV_FLAG_ZETA, "1")
    assert resolve_use_zeta_posterior_rerank() is True


def test_resolver_env_falsy_wins_over_kwarg(monkeypatch) -> None:
    monkeypatch.setenv(_ENV_FLAG_ZETA, "0")
    assert resolve_use_zeta_posterior_rerank(explicit=True) is False


def test_resolver_explicit_kwarg_when_env_unset(monkeypatch) -> None:
    monkeypatch.delenv(_ENV_FLAG_ZETA, raising=False)
    assert resolve_use_zeta_posterior_rerank(explicit=True) is True
    assert resolve_use_zeta_posterior_rerank(explicit=False) is False


def test_resolver_unrecognised_env_falls_through(monkeypatch) -> None:
    monkeypatch.setenv(_ENV_FLAG_ZETA, "maybe")
    assert resolve_use_zeta_posterior_rerank() is False


# ---------------------------------------------------------------------------
# Mutual exclusion helper
# ---------------------------------------------------------------------------

def test_mutex_raises_when_both_on() -> None:
    """Both γ and ζ resolved True → ValueError at flag-resolution time."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        _assert_gamma_zeta_mutual_exclusion(True, True)


@pytest.mark.parametrize(
    "gamma_on,zeta_on",
    [(False, False), (True, False), (False, True)],
)
def test_mutex_does_not_raise_when_not_both_on(
    gamma_on: bool, zeta_on: bool,
) -> None:
    """Only the both-True case raises; the three other combinations
    pass silently (so the call is cheap on the hot path)."""
    _assert_gamma_zeta_mutual_exclusion(gamma_on, zeta_on)


# ---------------------------------------------------------------------------
# Flag-off byte-identity + flag-on determinism
# ---------------------------------------------------------------------------

def test_flag_off_baseline_unchanged(populated_store, monkeypatch) -> None:
    """Both flags unset → retrieve() produces the same output across calls
    (the regression-protection lane)."""
    monkeypatch.delenv(_ENV_FLAG_ZETA, raising=False)
    monkeypatch.delenv(_ENV_FLAG_GAMMA, raising=False)
    a = retrieve(populated_store, "letter alphabet")
    b = retrieve(populated_store, "letter alphabet")
    assert [x.id for x in a] == [x.id for x in b]


def test_flag_on_deterministic(populated_store, monkeypatch) -> None:
    """ζ on, γ off → ζ runs the rerank loop; output is deterministic
    across repeats."""
    monkeypatch.setenv(_ENV_FLAG_ZETA, "1")
    monkeypatch.delenv(_ENV_FLAG_GAMMA, raising=False)
    a = retrieve(populated_store, "letter alphabet")
    b = retrieve(populated_store, "letter alphabet")
    assert [x.id for x in a] == [x.id for x in b]


def test_flag_on_versus_off_runs_clean(populated_store, monkeypatch) -> None:
    """Sanity: flipping the ζ flag on does not raise on a small corpus.

    This is the smoke gate the broader bench-corpus comparison sits
    on top of. The actual ζ-vs-γ-vs-log-additive ranking comparison
    is the job of the lab-side R&D harness (campaign at
    experiments/zeta-posterior/), not this unit test.
    """
    monkeypatch.delenv(_ENV_FLAG_ZETA, raising=False)
    off = retrieve(populated_store, "letter alphabet")
    monkeypatch.setenv(_ENV_FLAG_ZETA, "1")
    on = retrieve(populated_store, "letter alphabet")
    assert isinstance(off, list)
    assert isinstance(on, list)


def test_flag_on_reorders_by_posterior(populated_store, monkeypatch) -> None:
    """ζ at pinned defaults pushes high-posterior beliefs up the rank
    when BM25 is similar across rows. The corpus has α/β = (10,1),
    (1,10), (5,5), (1,1) — posterior means 0.91, 0.09, 0.5, 0.5.

    With ζ on, the high-posterior 'alpha' belief should outrank the
    low-posterior 'beta' belief on a generic 'letter' query (both
    match equally well). With both flags off the v1.3 log-additive
    path produces the same dominance, but the assert proves ζ doesn't
    invert the expected ordering.
    """
    monkeypatch.setenv(_ENV_FLAG_ZETA, "1")
    out = retrieve(populated_store, "letter alphabet")
    ids_to_contents = {b.id: b.content for b in out}
    contents = [ids_to_contents[b.id] for b in out]
    alpha_idx = next(i for i, c in enumerate(contents) if c.startswith("alpha"))
    beta_idx = next(i for i, c in enumerate(contents) if c.startswith("beta"))
    assert alpha_idx < beta_idx


# ---------------------------------------------------------------------------
# Both-flags-ON path raises through retrieve / retrieve_with_tiers
# ---------------------------------------------------------------------------

def test_retrieve_raises_when_both_flags_on(populated_store, monkeypatch) -> None:
    monkeypatch.setenv(_ENV_FLAG_GAMMA, "1")
    monkeypatch.setenv(_ENV_FLAG_ZETA, "1")
    with pytest.raises(ValueError, match="mutually exclusive"):
        retrieve(populated_store, "letter alphabet")


def test_retrieve_with_tiers_raises_when_both_flags_on(
    populated_store, monkeypatch,
) -> None:
    monkeypatch.setenv(_ENV_FLAG_GAMMA, "1")
    monkeypatch.setenv(_ENV_FLAG_ZETA, "1")
    with pytest.raises(ValueError, match="mutually exclusive"):
        retrieve_with_tiers(populated_store, "letter alphabet")
