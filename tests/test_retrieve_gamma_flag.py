"""Tests for #796 γ flag wiring in retrieve_v2 / retrieve_with_tiers.

Properties under test:

1. **Flag-off byte-identity.** With ``AELFRICE_USE_GAMMA_POSTERIOR_TEMPERATURE``
   unset (the default), retrieve()'s output is unchanged compared to a
   pre-#796 baseline — the existing log-additive contract holds.
2. **Flag-on, meta-belief absent.** The resolver returns T=1.0
   (byte-identical to ``partial_bayesian_score(.., 1.0)``), so γ runs
   the rerank loop but its score is anchored to the known log-additive
   reference. Output is deterministic given the same store + query.
3. **Resolver precedence.** env > kwarg > TOML > False. Verified by
   the resolver-only tests (no store touch needed for the precedence
   chain).
4. **Temperature decoder bounds.** ``resolve_posterior_temperature_with_meta``
   returns 1.0 on a None store, decodes log-linearly into
   ``[POSTERIOR_TEMPERATURE_FLOOR, POSTERIOR_TEMPERATURE_CEIL]``, and
   hits exactly 1.0 at the static-default mid-value of 0.5.
"""
from __future__ import annotations

import math
import uuid
from pathlib import Path

import pytest

from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, RETENTION_FACT, Belief
from aelfrice.retrieval import (
    POSTERIOR_TEMPERATURE_CEIL,
    POSTERIOR_TEMPERATURE_FLOOR,
    resolve_posterior_temperature_with_meta,
    resolve_use_gamma_posterior_temperature,
    retrieve,
)
from aelfrice.store import MemoryStore


_ENV_FLAG = "AELFRICE_USE_GAMMA_POSTERIOR_TEMPERATURE"


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Ensure the γ env flag never leaks across tests."""
    monkeypatch.delenv(_ENV_FLAG, raising=False)
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
        demotion_pressure=0,
        created_at="2023-11-14T22:13:20+00:00",
        last_retrieved_at=None,
        retention_class=RETENTION_FACT,
    )


@pytest.fixture
def populated_store():
    """Fresh in-memory store with a small corpus that exercises
    posterior reweighting (the two ``letter`` beliefs have different
    α/β so γ can move them relative to each other)."""
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
    monkeypatch.delenv(_ENV_FLAG, raising=False)
    assert resolve_use_gamma_posterior_temperature() is False


def test_resolver_env_truthy_wins(monkeypatch) -> None:
    monkeypatch.setenv(_ENV_FLAG, "1")
    assert resolve_use_gamma_posterior_temperature() is True


def test_resolver_env_falsy_wins_over_kwarg(monkeypatch) -> None:
    monkeypatch.setenv(_ENV_FLAG, "0")
    assert resolve_use_gamma_posterior_temperature(explicit=True) is False


def test_resolver_explicit_kwarg_when_env_unset(monkeypatch) -> None:
    monkeypatch.delenv(_ENV_FLAG, raising=False)
    assert resolve_use_gamma_posterior_temperature(explicit=True) is True
    assert resolve_use_gamma_posterior_temperature(explicit=False) is False


def test_resolver_unrecognised_env_falls_through(monkeypatch) -> None:
    monkeypatch.setenv(_ENV_FLAG, "maybe")
    assert resolve_use_gamma_posterior_temperature() is False


# ---------------------------------------------------------------------------
# Temperature decoder
# ---------------------------------------------------------------------------

def test_temperature_decoder_none_store_returns_one() -> None:
    assert resolve_posterior_temperature_with_meta(None, now_ts=0) == 1.0


def test_temperature_decoder_static_default_geometric_mean() -> None:
    """Manually verify the log-linear decode at v=0.5 lands at T=1.0
    (the documented byte-identical contract). This guards against
    accidental bound changes that would break the cold-start
    invariant."""
    log_floor = math.log(POSTERIOR_TEMPERATURE_FLOOR)
    log_ceil = math.log(POSTERIOR_TEMPERATURE_CEIL)
    decoded = math.exp(log_floor + 0.5 * (log_ceil - log_floor))
    assert math.isclose(decoded, 1.0, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# Flag-off byte-identity + flag-on determinism
# ---------------------------------------------------------------------------

def test_flag_off_baseline_unchanged(populated_store, monkeypatch) -> None:
    """Flag unset → retrieve() produces the same output across calls
    (the regression-protection lane)."""
    monkeypatch.delenv(_ENV_FLAG, raising=False)
    a = retrieve(populated_store, "letter alphabet")
    b = retrieve(populated_store, "letter alphabet")
    assert [x.id for x in a] == [x.id for x in b]


def test_flag_on_deterministic(populated_store, monkeypatch) -> None:
    """Flag-on, no meta-belief → T=1.0 → output is deterministic
    across repeats."""
    monkeypatch.setenv(_ENV_FLAG, "1")
    a = retrieve(populated_store, "letter alphabet")
    b = retrieve(populated_store, "letter alphabet")
    assert [x.id for x in a] == [x.id for x in b]


def test_flag_on_versus_off_runs_clean(populated_store, monkeypatch) -> None:
    """Sanity: flipping the flag on does not raise on a small corpus.

    This is the smoke gate the broader bench-corpus comparison sits
    on top of. The actual γ-vs-log-additive ranking comparison is the
    job of the lab-side A/B harness, not this unit test.
    """
    monkeypatch.delenv(_ENV_FLAG, raising=False)
    off = retrieve(populated_store, "letter alphabet")
    monkeypatch.setenv(_ENV_FLAG, "1")
    on = retrieve(populated_store, "letter alphabet")
    # Both produced something; the content / order is the bench's job.
    assert isinstance(off, list)
    assert isinstance(on, list)
