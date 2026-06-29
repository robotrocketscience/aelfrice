"""#1011 origin-tier rerank lane: de-rank bulk document-chunk beliefs
below user-stated facts that share keyword overlap.

The lane is default-OFF (a byte-identical no-op). When enabled it adds a
log-additive per-origin multiplier in the `_l1_hits` rerank, so a
`user_stated` fact outranks a `document_recent` chunk even when the chunk
has the stronger raw BM25 match. Weights are ablation-pending (LoCoMo)
before any default-on flip.
"""
from __future__ import annotations

import math

import pytest

from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief
from aelfrice.retrieval import (
    DEFAULT_ORIGIN_TIER_WEIGHTS,
    _origin_tier_boosted,
    resolve_origin_tier_rerank,
    retrieve,
)
from aelfrice.store import MemoryStore

_ENV = "AELFRICE_USE_ORIGIN_TIER_RERANK"


def _mk(bid: str, content: str, origin: str) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-05-11T00:00:00Z",
        last_retrieved_at=None,
        origin=origin,
    )


def _build() -> MemoryStore:
    """A document chunk with the stronger raw BM25 match (4x term
    frequency) plus a single user-stated fact on the same topic."""
    s = MemoryStore(":memory:")
    s.insert_belief(
        _mk("DOC", "job search job search job search job search market advice",
            "document_recent"),
    )
    s.insert_belief(
        _mk("USER", "job search update advanced to round two", "user_stated"),
    )
    return s


# --- resolver --------------------------------------------------------------


def test_resolver_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    assert resolve_origin_tier_rerank() is None


def test_resolver_env_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_ENV, "1")
    assert resolve_origin_tier_rerank() == DEFAULT_ORIGIN_TIER_WEIGHTS


def test_resolver_env_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_ENV, "0")
    assert resolve_origin_tier_rerank() is None


def test_resolver_explicit_overrides_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    assert resolve_origin_tier_rerank(explicit=True) == DEFAULT_ORIGIN_TIER_WEIGHTS


# --- _origin_tier_boosted --------------------------------------------------


def test_boost_noop_when_tiers_none() -> None:
    assert _origin_tier_boosted(5.0, "user_stated", None) == 5.0


def test_boost_lifts_user_stated() -> None:
    w = DEFAULT_ORIGIN_TIER_WEIGHTS
    assert _origin_tier_boosted(0.0, "user_stated", w) == pytest.approx(math.log(3.0))


def test_boost_demotes_document_recent() -> None:
    w = DEFAULT_ORIGIN_TIER_WEIGHTS
    assert _origin_tier_boosted(0.0, "document_recent", w) == pytest.approx(
        math.log(0.5),
    )


def test_boost_noop_for_unmapped_origin() -> None:
    w = DEFAULT_ORIGIN_TIER_WEIGHTS
    assert _origin_tier_boosted(2.0, "agent_inferred", w) == 2.0


# --- end-to-end through retrieve() -----------------------------------------


def test_lane_off_keeps_bm25_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default (off): the higher-BM25 document chunk ranks first, and the
    order is stable across calls — a byte-identical no-op."""
    monkeypatch.delenv(_ENV, raising=False)
    first = [b.id for b in retrieve(_build(), "job search")]
    second = [b.id for b in retrieve(_build(), "job search")]
    assert first == second
    assert first.index("DOC") < first.index("USER")


def test_lane_on_lifts_user_fact_above_document_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enabled: the user_stated fact is reranked above the document chunk
    despite the chunk's stronger raw BM25 match."""
    monkeypatch.setenv(_ENV, "1")
    ids = [b.id for b in retrieve(_build(), "job search")]
    assert ids.index("USER") < ids.index("DOC"), ids
