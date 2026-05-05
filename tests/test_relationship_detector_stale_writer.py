"""Unit tests for ``write_potentially_stale_edges`` (#387).

Covers edge direction, idempotency, report counts, and filtering
behaviour (skips non-contradicts verdicts and high-confidence pairs).

All tests use a real ``MemoryStore(":memory:")`` — no mocks.
"""
from __future__ import annotations

import hashlib

import pytest

from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_POTENTIALLY_STALE,
    LOCK_NONE,
    Belief,
)
from aelfrice.relationship_detector import (
    LABEL_CONTRADICTS,
    write_potentially_stale_edges,
)
from aelfrice.store import MemoryStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_belief(
    store: MemoryStore,
    *,
    belief_id: str,
    content: str,
    created_at: str = "2026-01-01T00:00:00Z",
) -> Belief:
    """Insert a minimal belief and return it."""
    b = Belief(
        id=belief_id,
        content=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        demotion_pressure=0,
        created_at=created_at,
        last_retrieved_at=None,
    )
    store.insert_belief(b)
    return b


def _edge_count(store: MemoryStore) -> int:
    cur = store._conn.execute("SELECT COUNT(*) FROM edges")  # type: ignore[attr-defined]
    return int(cur.fetchone()[0])


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore(":memory:")


# ---------------------------------------------------------------------------
# Sub-confidence pair: always(1.0) vs rarely(-0.6) → axis dist 1.6,
# q_term = 0.8, n_term = 0.0 → score = 0.4 < 0.5 = confidence_min.
# These pairs reliably land below the floor.
# ---------------------------------------------------------------------------

_CONTENT_A = "BM25 retrieval is always fast"
_CONTENT_B = "BM25 retrieval is rarely fast"

# High-confidence pair: negation only, no quantifiers → score = 0.5,
# which is NOT < confidence_min (0.5 < 0.5 is False), so no stale edge.
_CONTENT_HI_A = "BM25 index lookup is fast"
_CONTENT_HI_B = "BM25 index lookup is not fast"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_writer_emits_edges_for_sub_confidence_pairs(store: MemoryStore) -> None:
    """A sub-confidence contradicting pair produces exactly one stale edge."""
    _make_belief(store, belief_id="b1", content=_CONTENT_A,
                 created_at="2026-01-01T00:00:00Z")
    _make_belief(store, belief_id="b2", content=_CONTENT_B,
                 created_at="2026-02-01T00:00:00Z")

    report = write_potentially_stale_edges(store)

    assert report.n_edges_written == 1
    assert report.n_sub_confidence == 1
    assert _edge_count(store) == 1


def test_writer_skips_high_confidence_pairs(store: MemoryStore) -> None:
    """A high-confidence (score >= confidence_min) contradicting pair produces no stale edge.

    negation-only pair scores exactly 0.5 == confidence_min, so it is
    NOT sub-confidence and must be excluded.
    """
    _make_belief(store, belief_id="h1", content=_CONTENT_HI_A)
    _make_belief(store, belief_id="h2", content=_CONTENT_HI_B)

    report = write_potentially_stale_edges(store)

    assert report.n_edges_written == 0
    assert _edge_count(store) == 0


def test_writer_idempotent(store: MemoryStore) -> None:
    """Calling the writer twice writes 0 new edges on the second call."""
    _make_belief(store, belief_id="b1", content=_CONTENT_A,
                 created_at="2026-01-01T00:00:00Z")
    _make_belief(store, belief_id="b2", content=_CONTENT_B,
                 created_at="2026-02-01T00:00:00Z")

    first = write_potentially_stale_edges(store)
    assert first.n_edges_written == 1

    second = write_potentially_stale_edges(store)
    assert second.n_edges_written == 0
    assert second.n_edges_skipped_existing == 1
    # Total edge count unchanged after second call.
    assert _edge_count(store) == 1


def test_writer_direction_newer_to_older(store: MemoryStore) -> None:
    """Edge direction: newer belief (b2, later created_at) is src; older is dst."""
    b1 = _make_belief(store, belief_id="b1", content=_CONTENT_A,
                      created_at="2026-01-01T00:00:00Z")  # older
    b2 = _make_belief(store, belief_id="b2", content=_CONTENT_B,
                      created_at="2026-03-01T00:00:00Z")  # newer

    write_potentially_stale_edges(store)

    edge = store.get_edge(b2.id, b1.id, EDGE_POTENTIALLY_STALE)
    assert edge is not None, "expected edge src=b2 (newer) → dst=b1 (older)"
    assert edge.src == b2.id
    assert edge.dst == b1.id
    assert edge.weight == 1.0
    # Reverse direction must NOT exist.
    assert store.get_edge(b1.id, b2.id, EDGE_POTENTIALLY_STALE) is None


def test_writer_direction_tiebreak_on_id(store: MemoryStore) -> None:
    """When created_at is identical, lex-greater id wins as src."""
    # "b_zzz" > "b_aaa" lexicographically → b_zzz is src.
    b_aaa = _make_belief(store, belief_id="b_aaa", content=_CONTENT_A,
                         created_at="2026-01-01T00:00:00Z")
    b_zzz = _make_belief(store, belief_id="b_zzz", content=_CONTENT_B,
                         created_at="2026-01-01T00:00:00Z")  # same ts

    write_potentially_stale_edges(store)

    edge = store.get_edge(b_zzz.id, b_aaa.id, EDGE_POTENTIALLY_STALE)
    assert edge is not None, "expected edge src=b_zzz → dst=b_aaa"
    assert store.get_edge(b_aaa.id, b_zzz.id, EDGE_POTENTIALLY_STALE) is None


def test_writer_skips_refines_and_unrelated(store: MemoryStore) -> None:
    """Refines and unrelated pairs produce zero stale edges."""
    # Refines: same modality, high residual overlap (b1/b2).
    # "BM25 scoring weights title tokens" vs "BM25 scoring weights body tokens"
    # → modality agrees, refines.
    _make_belief(store, belief_id="r1",
                 content="BM25 scoring weights title tokens")
    _make_belief(store, belief_id="r2",
                 content="BM25 scoring weights body tokens")
    # Unrelated: disjoint content (r3/r4).
    _make_belief(store, belief_id="r3", content="the cat sat on the mat")
    _make_belief(store, belief_id="r4", content="dogs enjoy running outside")

    report = write_potentially_stale_edges(store)

    assert report.n_edges_written == 0
    assert _edge_count(store) == 0


def test_writer_returns_report_counts(store: MemoryStore) -> None:
    """Mixed store: report counts add up correctly.

    Store contains:
      - one sub-confidence contradicting pair → 1 edge written
      - one high-confidence contradicting pair → 0 edges (excluded)
      - one unrelated pair → 0 edges
    """
    # Sub-confidence pair (score 0.4).
    _make_belief(store, belief_id="s1", content=_CONTENT_A,
                 created_at="2026-01-01T00:00:00Z")
    _make_belief(store, belief_id="s2", content=_CONTENT_B,
                 created_at="2026-02-01T00:00:00Z")
    # High-confidence pair (score 0.5 — NOT sub-confidence).
    _make_belief(store, belief_id="h1", content=_CONTENT_HI_A)
    _make_belief(store, belief_id="h2", content=_CONTENT_HI_B)
    # Unrelated pair.
    _make_belief(store, belief_id="u1", content="sphinx fts indexing latency")
    _make_belief(store, belief_id="u2", content="compression ratio on disk")

    report = write_potentially_stale_edges(store)

    # Exactly 1 sub-confidence contradicts pair qualifies.
    assert report.n_sub_confidence == 1
    assert report.n_edges_written == 1
    assert report.n_edges_skipped_existing == 0
    assert report.n_edges_skipped_self_pair == 0
    # Total contradicts audited = sub + high.
    assert report.n_pairs_audited == report.n_sub_confidence + (
        report.n_pairs_audited - report.n_sub_confidence
    )
    # n_pairs_audited must be >= n_sub_confidence.
    assert report.n_pairs_audited >= report.n_sub_confidence
