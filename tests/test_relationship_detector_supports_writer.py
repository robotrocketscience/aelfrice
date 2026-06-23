"""Unit tests for the #999 SUPPORTS edge writer.

Covers ``write_supports_edges`` (SUPPORTS edges for REFINES / mutual-
agreement pairs — the agreement counterpart to the #988 CONTRADICTS
writer), idempotency, the per-belief write-gate, and the determinism
guarantee.

All tests use a real ``MemoryStore(":memory:")`` — no mocks.
"""
from __future__ import annotations

import hashlib

import pytest

from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_SUPPORTS,
    LOCK_NONE,
    Belief,
)
from aelfrice.relationship_detector import write_supports_edges
from aelfrice.store import MemoryStore

# REFINES pair: identical residual content, agreeing modality (both
# universal "always", no negation) → score 0.0, label refines.
_ALWAYS = "the deployment script always runs the database migration step"
_ALWAYS_LAUNCH = _ALWAYS + " before launch"
_ALWAYS_PROD = _ALWAYS + " in production"
# CONTRADICTS pair counterpart (universal affirm vs negate).
_NEVER = "the deployment script never runs the database migration step"
# Lexically distant filler that relates to nothing else.
_UNRELATED = "the harbor seals bask on the warm rocks at noon each day"


def _make_belief(
    store: MemoryStore,
    *,
    belief_id: str,
    content: str,
    created_at: str = "2026-01-01T00:00:00Z",
) -> Belief:
    b = Belief(
        id=belief_id,
        content=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at=created_at,
        last_retrieved_at=None,
    )
    store.insert_belief(b)
    return b


def _supports_edges(store: MemoryStore) -> list[tuple[str, str]]:
    """Return all SUPPORTS edges as sorted (src, dst) tuples."""
    rows = store._conn.execute(  # type: ignore[attr-defined]
        "SELECT src, dst FROM edges WHERE type = ? ORDER BY src, dst",
        (EDGE_SUPPORTS,),
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore(":memory:")


# ---------------------------------------------------------------------------
# write_supports_edges — write / scope / idempotency
# ---------------------------------------------------------------------------


def test_writes_supports_edge_for_refines_pair(store: MemoryStore) -> None:
    _make_belief(store, belief_id="b1", content=_ALWAYS)
    _make_belief(store, belief_id="b2", content=_ALWAYS_LAUNCH)
    report = write_supports_edges(store)
    assert report.n_refines == 1
    assert report.n_edges_written == 1
    # Symmetric relation: canonical direction src = min(id), dst = max(id).
    assert _supports_edges(store) == [("b1", "b2")]


def test_idempotent_second_run_writes_nothing(store: MemoryStore) -> None:
    _make_belief(store, belief_id="b1", content=_ALWAYS)
    _make_belief(store, belief_id="b2", content=_ALWAYS_LAUNCH)
    write_supports_edges(store)
    report = write_supports_edges(store)
    assert report.n_edges_written == 0
    assert report.n_edges_skipped_existing == 1
    assert _supports_edges(store) == [("b1", "b2")]


def test_contradicts_pair_writes_no_supports(store: MemoryStore) -> None:
    # Universal affirm vs negate → contradicts, which this SUPPORTS-only
    # writer ignores.
    _make_belief(store, belief_id="b1", content=_ALWAYS)
    _make_belief(store, belief_id="b2", content=_NEVER)
    report = write_supports_edges(store)
    assert report.n_edges_written == 0
    assert _supports_edges(store) == []


def test_unrelated_pair_writes_no_supports(store: MemoryStore) -> None:
    _make_belief(store, belief_id="b1", content=_ALWAYS)
    _make_belief(store, belief_id="b2", content=_UNRELATED)
    report = write_supports_edges(store)
    assert report.n_refines == 0
    assert report.n_edges_written == 0
    assert _supports_edges(store) == []


# ---------------------------------------------------------------------------
# Determinism — byte-equal edge sets across two fresh stores
# ---------------------------------------------------------------------------


def test_determinism_byte_equal_edges() -> None:
    def build() -> list[tuple[str, str]]:
        s = MemoryStore(":memory:")
        _make_belief(s, belief_id="b1", content=_ALWAYS)
        _make_belief(s, belief_id="b2", content=_ALWAYS_LAUNCH)
        _make_belief(s, belief_id="b3", content=_UNRELATED)
        write_supports_edges(s)
        return _supports_edges(s)

    assert build() == build()


# ---------------------------------------------------------------------------
# Write-gate — per-belief edge cap (Exp-48 dilution guard)
# ---------------------------------------------------------------------------


def test_write_gate_caps_per_belief_edges(store: MemoryStore) -> None:
    # Three mutually-refining beliefs → three REFINES pairs. With a cap of
    # 1, only the first pair in deterministic (a_id, b_id) order survives.
    _make_belief(store, belief_id="b1", content=_ALWAYS)
    _make_belief(store, belief_id="b2", content=_ALWAYS_LAUNCH)
    _make_belief(store, belief_id="b3", content=_ALWAYS_PROD)
    report = write_supports_edges(store, max_edges_per_belief=1)
    assert report.n_edges_written == 1
    assert report.n_edges_skipped_gated == 2
    assert _supports_edges(store) == [("b1", "b2")]
