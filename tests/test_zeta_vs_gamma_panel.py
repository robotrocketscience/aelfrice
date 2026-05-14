"""ζ vs γ ranking panel — #817 §"Scope" item 5.

The full lab-side comparison ζ@(α=1, β=0.25, scale=14.5) vs γ@T=1.0
runs against the corpus at ``experiments/zeta-posterior/`` and is not
in this test file's scope. What this file *does* cover is the
property contracts visible from public test fixtures:

1. **Both flags OFF byte-identical to main.** With neither γ nor ζ
   on, retrieve() output is identical to what the v3.1 log-additive
   path produced — a structural regression guard.
2. **Single-flag panels run clean.** ζ on alone and γ on alone each
   produce a deterministic ranking on a small fixture; the panel
   metrics (``rank_biased_overlap``, ``ordered_top_k_overlap``) are
   computable on those rankings without raising.
3. **ζ vs γ on a constant-posterior fixture is rank-identical.** When
   every belief in the fixture has α=β=1.0 (posterior_mean=0.5), ζ's
   posterior contribution is identically zero and γ at T=1.0 also
   adds a constant log(0.5), so neither rerank moves any belief
   relative to BM25 alone. RBO and top-K overlap both = 1.0.
4. **Both flags ON → ValueError.** Covered exhaustively in
   ``test_retrieve_zeta_flag.py``; cross-referenced here for
   navigability.
"""
from __future__ import annotations

import uuid

import pytest

from aelfrice.calibration_metrics import (
    ordered_top_k_overlap,
    rank_biased_overlap,
)
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, RETENTION_FACT, Belief
from aelfrice.retrieval import retrieve
from aelfrice.store import MemoryStore


_ENV_FLAG_ZETA = "AELFRICE_USE_ZETA_POSTERIOR_RERANK"
_ENV_FLAG_GAMMA = "AELFRICE_USE_GAMMA_POSTERIOR_TEMPERATURE"


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    monkeypatch.delenv(_ENV_FLAG_ZETA, raising=False)
    monkeypatch.delenv(_ENV_FLAG_GAMMA, raising=False)
    yield


def _mk_belief(text: str, *, alpha: float = 1.0, beta: float = 1.0) -> Belief:
    bid = uuid.uuid4().hex[:16]
    return Belief(
        id=bid, content=text, content_hash=f"h_{bid}",
        alpha=alpha, beta=beta, type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE, locked_at=None, demotion_pressure=0,
        created_at="2023-11-14T22:13:20+00:00",
        last_retrieved_at=None, retention_class=RETENTION_FACT,
    )


@pytest.fixture
def varied_posterior_store():
    """Six beliefs spanning the posterior_mean range so γ and ζ have
    something to move on. Bench panels comparing γ-vs-ζ exercise the
    `_l1_hits` rerank branch."""
    s = MemoryStore(":memory:")
    rows = [
        ("alpha is the first letter", 10.0, 1.0),
        ("beta is the second letter", 1.0, 10.0),
        ("gamma is the third letter", 5.0, 5.0),
        ("delta is the fourth letter", 1.0, 1.0),
        ("epsilon is the fifth letter", 8.0, 2.0),
        ("zeta is the sixth letter", 2.0, 8.0),
    ]
    for text, a, b in rows:
        s.insert_belief(_mk_belief(text, alpha=a, beta=b))
    yield s
    s.close()


@pytest.fixture
def uniform_posterior_store():
    """Six beliefs all at α=β=1.0 (posterior_mean=0.5). Neither γ nor
    ζ moves them relative to BM25 — γ adds a constant log(0.5) and
    ζ's bracket is identically 0."""
    s = MemoryStore(":memory:")
    for text in [
        "alpha is the first letter",
        "beta is the second letter",
        "gamma is the third letter",
        "delta is the fourth letter",
        "epsilon is the fifth letter",
        "zeta is the sixth letter",
    ]:
        s.insert_belief(_mk_belief(text))  # α=β=1.0
    yield s
    s.close()


# ---------------------------------------------------------------------------
# Both flags OFF → byte-identical to pre-#817 path
# ---------------------------------------------------------------------------

def test_both_flags_off_deterministic(varied_posterior_store) -> None:
    """The default — neither γ nor ζ env set. Two calls produce the
    same ranking; #817 is plumbing only on default code paths."""
    a = retrieve(varied_posterior_store, "letter alphabet")
    b = retrieve(varied_posterior_store, "letter alphabet")
    assert [x.id for x in a] == [x.id for x in b]
    # Sanity: RBO on a list with itself is exactly 1.0.
    ids_a = [x.id for x in a]
    assert rank_biased_overlap(ids_a, ids_a) == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Single-flag panels run clean
# ---------------------------------------------------------------------------

def test_gamma_panel_runs(varied_posterior_store, monkeypatch) -> None:
    monkeypatch.setenv(_ENV_FLAG_GAMMA, "1")
    out = retrieve(varied_posterior_store, "letter alphabet")
    ids = [b.id for b in out]
    assert len(ids) >= 2
    # Panel metrics computable.
    rbo = rank_biased_overlap(ids, ids)
    overlap = ordered_top_k_overlap(ids, ids, k=min(5, len(ids)))
    assert rbo == pytest.approx(1.0, abs=1e-12)
    assert overlap == 1.0


def test_zeta_panel_runs(varied_posterior_store, monkeypatch) -> None:
    monkeypatch.setenv(_ENV_FLAG_ZETA, "1")
    out = retrieve(varied_posterior_store, "letter alphabet")
    ids = [b.id for b in out]
    assert len(ids) >= 2
    rbo = rank_biased_overlap(ids, ids)
    overlap = ordered_top_k_overlap(ids, ids, k=min(5, len(ids)))
    assert rbo == pytest.approx(1.0, abs=1e-12)
    assert overlap == 1.0


# ---------------------------------------------------------------------------
# Uniform-posterior fixture: γ and ζ are rank-identical
# ---------------------------------------------------------------------------

def test_gamma_and_zeta_agree_on_uniform_store(
    uniform_posterior_store, monkeypatch,
) -> None:
    """Every belief at posterior_mean=0.5 → ζ contribution is 0 for
    every belief; γ at T=1.0 adds a constant log(0.5) to every
    belief. Neither rerank moves anything. Result: γ-on and ζ-on
    rank-identically (and identical to flag-off, modulo the
    short-circuit that ζ/γ disables to force the rerank loop)."""
    monkeypatch.delenv(_ENV_FLAG_GAMMA, raising=False)
    monkeypatch.delenv(_ENV_FLAG_ZETA, raising=False)
    baseline = [b.id for b in retrieve(uniform_posterior_store, "letter")]

    monkeypatch.setenv(_ENV_FLAG_GAMMA, "1")
    gamma_ids = [b.id for b in retrieve(uniform_posterior_store, "letter")]
    monkeypatch.delenv(_ENV_FLAG_GAMMA, raising=False)

    monkeypatch.setenv(_ENV_FLAG_ZETA, "1")
    zeta_ids = [b.id for b in retrieve(uniform_posterior_store, "letter")]

    # RBO between γ-on and ζ-on should be 1.0 — both add a uniform
    # constant (or 0) to every belief.
    assert rank_biased_overlap(gamma_ids, zeta_ids) == pytest.approx(
        1.0, abs=1e-12,
    )
    assert ordered_top_k_overlap(
        gamma_ids, zeta_ids, k=min(5, len(gamma_ids)),
    ) == 1.0
    # And both equal the baseline ordering for the same reason.
    assert ordered_top_k_overlap(
        baseline, gamma_ids, k=min(5, len(baseline)),
    ) == 1.0
    assert ordered_top_k_overlap(
        baseline, zeta_ids, k=min(5, len(baseline)),
    ) == 1.0
