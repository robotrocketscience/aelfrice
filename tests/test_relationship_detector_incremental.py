"""Tests for incremental edge detection in write_semantic_edges (#1000).

Covers:
1. Incremental == full-audit edge set (no cap pressure, across simulated turns).
2. Determinism of the incremental path.
3. Full-store mode (new_belief_ids=None) backward-compat / no-op parity.
4. Hard per-belief cap enforced across turns in incremental mode.
5. restrict_to_ids unit test on _jaccard_prefiltered_pairs directly.

All tests use real MemoryStore(":memory:") — no mocks.
Stdlib-only, deterministic, no LLM, no embeddings.
"""
from __future__ import annotations

import hashlib

import pytest

from aelfrice.dedup import _jaccard_prefiltered_pairs
from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_CONTRADICTS,
    LOCK_NONE,
    Belief,
)
from aelfrice.relationship_detector import (
    DEFAULT_MAX_EDGES_PER_BELIEF,
    write_semantic_edges,
)
from aelfrice.store import MemoryStore


# ---------------------------------------------------------------------------
# Belief content fixtures — universal-affirmation vs negation pairs generate
# high-confidence CONTRADICTS scores (score 1.0).  All share the same
# residual-content token pattern so the detector fires reliably.
# ---------------------------------------------------------------------------

# Turn-1 pair
_T1_ALWAYS = "the cache layer always stores the user session token"
_T1_NEVER = "the cache layer never stores the user session token"

# Turn-2 pair (different subject, no overlap with Turn-1 content)
_T2_ALWAYS = "the scheduler always runs the backup job at midnight"
_T2_NEVER = "the scheduler never runs the backup job at midnight"

# Turn-3 pair
_T3_ALWAYS = "the config loader always reads environment variables first"
_T3_NEVER = "the config loader never reads environment variables first"

# Lexically distant filler — does not contradict anything above.
_FILLER = "the harbor seals bask on the warm rocks at noon each day"


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


def _contradicts_edges(store: MemoryStore) -> list[tuple[str, str, str]]:
    """Return all CONTRADICTS edges as sorted (src, dst, type) tuples."""
    rows = store._conn.execute(  # type: ignore[attr-defined]
        "SELECT src, dst, type FROM edges WHERE type = ? ORDER BY src, dst",
        (EDGE_CONTRADICTS,),
    ).fetchall()
    return [(r[0], r[1], r[2]) for r in rows]


# ---------------------------------------------------------------------------
# 1. Incremental == full-audit edge set
# ---------------------------------------------------------------------------


def test_incremental_equals_full_audit_edge_set() -> None:
    """Incremental per-turn calls produce the same final edge set as a single
    full-store audit over all turns combined."""

    # Path A: insert all beliefs, then run one full-store audit.
    store_a = MemoryStore(":memory:")
    _make_belief(store_a, belief_id="a1", content=_T1_ALWAYS)
    _make_belief(store_a, belief_id="a2", content=_T1_NEVER)
    _make_belief(store_a, belief_id="b1", content=_T2_ALWAYS)
    _make_belief(store_a, belief_id="b2", content=_T2_NEVER)
    _make_belief(store_a, belief_id="c1", content=_T3_ALWAYS)
    _make_belief(store_a, belief_id="c2", content=_T3_NEVER)
    _make_belief(store_a, belief_id="f1", content=_FILLER)
    write_semantic_edges(store_a)  # full-store, new_belief_ids=None
    edges_full = _contradicts_edges(store_a)

    # Path B: insert turn-by-turn, run incremental after each turn.
    store_b = MemoryStore(":memory:")
    # Turn 1
    _make_belief(store_b, belief_id="a1", content=_T1_ALWAYS)
    _make_belief(store_b, belief_id="a2", content=_T1_NEVER)
    write_semantic_edges(store_b, new_belief_ids=["a1", "a2"])
    # Turn 2
    _make_belief(store_b, belief_id="b1", content=_T2_ALWAYS)
    _make_belief(store_b, belief_id="b2", content=_T2_NEVER)
    write_semantic_edges(store_b, new_belief_ids=["b1", "b2"])
    # Turn 3
    _make_belief(store_b, belief_id="c1", content=_T3_ALWAYS)
    _make_belief(store_b, belief_id="c2", content=_T3_NEVER)
    _make_belief(store_b, belief_id="f1", content=_FILLER)
    write_semantic_edges(store_b, new_belief_ids=["c1", "c2", "f1"])
    edges_incremental = _contradicts_edges(store_b)

    assert edges_incremental == edges_full, (
        f"Incremental path produced different edges.\n"
        f"  Full-store: {edges_full}\n"
        f"  Incremental: {edges_incremental}"
    )
    # Sanity: at least the three turn-pairs should each have one edge.
    assert len(edges_full) >= 3


# ---------------------------------------------------------------------------
# 2. Determinism — two incremental runs on identical input are byte-equal
# ---------------------------------------------------------------------------


def test_incremental_is_deterministic() -> None:
    """Two incremental builds on identical input produce byte-equal edge tables."""

    def build() -> list[tuple[str, str, str]]:
        s = MemoryStore(":memory:")
        _make_belief(s, belief_id="a1", content=_T1_ALWAYS)
        _make_belief(s, belief_id="a2", content=_T1_NEVER)
        write_semantic_edges(s, new_belief_ids=["a1", "a2"])
        _make_belief(s, belief_id="b1", content=_T2_ALWAYS)
        _make_belief(s, belief_id="b2", content=_T2_NEVER)
        write_semantic_edges(s, new_belief_ids=["b1", "b2"])
        return _contradicts_edges(s)

    assert build() == build()


# ---------------------------------------------------------------------------
# 3. Full-store mode (new_belief_ids=None) backward-compat parity
# ---------------------------------------------------------------------------


def test_full_store_mode_backward_compat() -> None:
    """write_semantic_edges with new_belief_ids=None is unchanged — same edges
    as expected from the original full-audit path."""
    store = MemoryStore(":memory:")
    _make_belief(store, belief_id="x1", content=_T1_ALWAYS)
    _make_belief(store, belief_id="x2", content=_T1_NEVER)
    _make_belief(store, belief_id="x3", content=_FILLER)

    report = write_semantic_edges(store)  # new_belief_ids=None (default)
    assert report.n_edges_written == 1
    edges = _contradicts_edges(store)
    assert edges == [("x1", "x2", EDGE_CONTRADICTS)]


def test_full_store_idempotent_parity() -> None:
    """Second call in full-store mode writes nothing (idempotency preserved)."""
    store = MemoryStore(":memory:")
    _make_belief(store, belief_id="x1", content=_T1_ALWAYS)
    _make_belief(store, belief_id="x2", content=_T1_NEVER)
    write_semantic_edges(store)
    report2 = write_semantic_edges(store)
    assert report2.n_edges_written == 0
    assert report2.n_edges_skipped_existing == 1


# ---------------------------------------------------------------------------
# 4. Hard per-belief cap enforced across turns in incremental mode
# ---------------------------------------------------------------------------


def test_incremental_hard_cap_across_turns() -> None:
    """Hub belief accretes at most max_edges_per_belief CONTRADICTS edges
    across multiple incremental turns."""
    cap = 3  # set low to stress the cross-turn enforcement

    # "hub" always contradicts "n_i" (never variant) for each i.
    # Introduce hub + n_1, n_2 in turn 1; n_3, n_4 in turn 2; n_5, n_6 in turn 3.
    # With cap=3, hub should never exceed 3 edges regardless of turn count.

    base_text = "the indexer always rebuilds the search index on startup"

    def _never_variant(i: int) -> str:
        # Vary by appending a token that doesn't break the residual content
        # overlap enough to drop below jaccard_min, but gives distinct IDs.
        return f"the indexer never rebuilds the search index on startup run{i}"

    store = MemoryStore(":memory:")
    hub_id = "hub"
    _make_belief(store, belief_id=hub_id, content=base_text)

    spoke_ids = [f"n{i}" for i in range(1, 7)]
    for sid in spoke_ids:
        _make_belief(
            store, belief_id=sid,
            content=_never_variant(int(sid[1:]))
        )

    # Simulate three turns introducing the spokes in pairs.
    turn_deltas = [
        [hub_id, "n1", "n2"],  # turn 1: hub + n1, n2
        ["n3", "n4"],          # turn 2
        ["n5", "n6"],          # turn 3
    ]
    for delta in turn_deltas:
        write_semantic_edges(store, new_belief_ids=delta, max_edges_per_belief=cap)

    # Count hub's CONTRADICTS edges directly from the store.
    hub_edge_count = store._conn.execute(  # type: ignore[attr-defined]
        "SELECT COUNT(*) FROM edges WHERE type = ? AND (src = ? OR dst = ?)",
        (EDGE_CONTRADICTS, hub_id, hub_id),
    ).fetchone()[0]

    assert hub_edge_count <= cap, (
        f"Hub accreted {hub_edge_count} edges, exceeding cap {cap}"
    )


# ---------------------------------------------------------------------------
# 5. restrict_to_ids unit test on _jaccard_prefiltered_pairs
# ---------------------------------------------------------------------------


def test_restrict_to_ids_filters_pairs() -> None:
    """With restrict_to_ids, every returned pair has at least one endpoint in
    the set; restrict_to_ids=None returns the superset."""
    # Build a small beliefs list: ids sorted ASC (as store.list_beliefs_for_indexing does).
    beliefs = [
        ("a", _T1_ALWAYS),
        ("b", _T1_NEVER),
        ("c", _T2_ALWAYS),
        ("d", _T2_NEVER),
    ]
    restrict = frozenset({"a", "b"})

    pairs_restricted, raw_restricted, _ = _jaccard_prefiltered_pairs(
        beliefs,
        jaccard_min=0.0,
        max_pairs=1000,
        restrict_to_ids=restrict,
    )
    pairs_full, raw_full, _ = _jaccard_prefiltered_pairs(
        beliefs,
        jaccard_min=0.0,
        max_pairs=1000,
        restrict_to_ids=None,
    )

    # Every restricted pair must have at least one endpoint in the set.
    for id_a, _, id_b, *_ in pairs_restricted:
        assert id_a in restrict or id_b in restrict, (
            f"Pair ({id_a}, {id_b}) has no endpoint in restrict set {restrict}"
        )

    # Full set must be at least as large as restricted set.
    assert raw_full >= raw_restricted
    assert len(pairs_full) >= len(pairs_restricted)

    # Pairs that are old-old (neither endpoint in restrict) must not appear.
    old_old_in_restricted = [
        (id_a, id_b)
        for id_a, _, id_b, *_ in pairs_restricted
        if id_a not in restrict and id_b not in restrict
    ]
    assert old_old_in_restricted == []


def test_restrict_to_ids_none_is_full_scan() -> None:
    """restrict_to_ids=None returns all pairs (same count as unrestricted call)."""
    beliefs = [
        ("a", _T1_ALWAYS),
        ("b", _T1_NEVER),
        ("c", _T2_ALWAYS),
    ]
    pairs_none, raw_none, _ = _jaccard_prefiltered_pairs(
        beliefs, jaccard_min=0.0, max_pairs=1000, restrict_to_ids=None
    )
    pairs_explicit_none, raw_explicit_none, _ = _jaccard_prefiltered_pairs(
        beliefs, jaccard_min=0.0, max_pairs=1000
    )
    # Both forms (omitted and explicit None) must give the same result.
    assert raw_none == raw_explicit_none
    assert pairs_none == pairs_explicit_none
