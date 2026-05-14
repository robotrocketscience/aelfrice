"""Tests for wonder_ingest and wonder_gc lifecycle (#548)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from aelfrice.models import (
    BELIEF_FACTUAL,
    BELIEF_SPECULATIVE,
    CORROBORATION_SOURCE_WONDER_INGEST,
    EDGE_RELATES_TO,
    EDGE_RESOLVES,
    LOCK_NONE,
    ORIGIN_SPECULATIVE,
    RETENTION_FACT,
    Belief,
    Edge,
    Phantom,
)
from aelfrice.store import MemoryStore
from aelfrice.wonder.lifecycle import (
    WonderGCResult,
    WonderIngestResult,
    wonder_gc,
    wonder_ingest,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_ALPHA_DEFAULT = 0.3
_BETA_DEFAULT = 1.0


def _constituent(bid: str) -> Belief:
    return Belief(
        id=bid,
        content=f"constituent content for {bid}",
        content_hash=f"ch_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-05-01T00:00:00+00:00",
        last_retrieved_at=None,
        retention_class=RETENTION_FACT,
    )


def _phantom(
    a_id: str,
    b_id: str,
    *,
    content: str = "speculative content",
    score: float = 0.75,
    generator: str = "bfs+wonder_consolidation",
) -> Phantom:
    return Phantom(
        constituent_belief_ids=(a_id, b_id),
        generator=generator,
        content=content,
        score=score,
    )


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore(":memory:")


@pytest.fixture
def store_with_constituents(store: MemoryStore) -> MemoryStore:
    store.insert_belief(_constituent("a"))
    store.insert_belief(_constituent("b"))
    store.insert_belief(_constituent("c"))
    return store


# ---------------------------------------------------------------------------
# wonder_ingest: schema correctness
# ---------------------------------------------------------------------------


def test_ingest_writes_belief_with_speculative_type(
    store_with_constituents: MemoryStore,
) -> None:
    store = store_with_constituents
    phantom = _phantom("a", "b")
    result = wonder_ingest(store, [phantom])

    assert isinstance(result, WonderIngestResult)
    assert result.inserted == 1
    assert result.skipped == 0

    beliefs = [store.get_belief(bid) for bid in store.list_belief_ids()]
    speculative = [b for b in beliefs if b is not None and b.type == BELIEF_SPECULATIVE]
    assert len(speculative) == 1

    s = speculative[0]
    assert s.origin == ORIGIN_SPECULATIVE
    assert s.alpha == pytest.approx(_ALPHA_DEFAULT)
    assert s.beta == pytest.approx(_BETA_DEFAULT)
    assert s.content == phantom.content
    assert s.lock_level == LOCK_NONE
    assert s.valid_to is None


def test_ingest_writes_belief_with_session_id(
    store_with_constituents: MemoryStore,
) -> None:
    store = store_with_constituents
    result = wonder_ingest(store, [_phantom("a", "b")], session_id="sess-1")
    assert result.inserted == 1

    beliefs = [store.get_belief(bid) for bid in store.list_belief_ids()]
    speculative = [b for b in beliefs if b is not None and b.type == BELIEF_SPECULATIVE]
    assert len(speculative) == 1
    assert speculative[0].session_id == "sess-1"


# ---------------------------------------------------------------------------
# wonder_ingest: RELATES_TO edges
# ---------------------------------------------------------------------------


def test_ingest_writes_relates_to_edges_to_all_constituents(
    store_with_constituents: MemoryStore,
) -> None:
    store = store_with_constituents
    phantom = Phantom(
        constituent_belief_ids=("a", "b", "c"),
        generator="bfs",
        content="three-way speculative",
        score=0.5,
    )
    result = wonder_ingest(store, [phantom])
    assert result.inserted == 1
    assert result.edges_created == 3

    beliefs = [store.get_belief(bid) for bid in store.list_belief_ids()]
    phantom_belief = next(
        b for b in beliefs if b is not None and b.type == BELIEF_SPECULATIVE
    )
    edges = store.edges_from(phantom_belief.id)
    relates = [e for e in edges if e.type == EDGE_RELATES_TO]
    assert len(relates) == 3
    assert {e.dst for e in relates} == {"a", "b", "c"}


def test_ingest_edge_count_matches_result(
    store_with_constituents: MemoryStore,
) -> None:
    store = store_with_constituents
    phantoms = [_phantom("a", "b"), _phantom("b", "c")]
    result = wonder_ingest(store, phantoms)
    assert result.inserted == 2
    assert result.edges_created == 4


# ---------------------------------------------------------------------------
# wonder_ingest: audit row
# ---------------------------------------------------------------------------


def test_ingest_writes_audit_corroboration_row(
    store_with_constituents: MemoryStore,
) -> None:
    store = store_with_constituents
    phantom = _phantom("a", "b", score=0.8765, generator="bfs+wonder_consolidation")
    wonder_ingest(store, [phantom])

    beliefs = [store.get_belief(bid) for bid in store.list_belief_ids()]
    speculative = next(
        b for b in beliefs if b is not None and b.type == BELIEF_SPECULATIVE
    )
    corroborations = store.list_corroborations(speculative.id)
    assert len(corroborations) == 1

    ingested_at, source_type, _session, source_path_hash = corroborations[0]
    assert source_type == CORROBORATION_SOURCE_WONDER_INGEST
    assert source_path_hash is not None
    assert "bfs+wonder_consolidation" in source_path_hash
    assert "0.8765" in source_path_hash


# ---------------------------------------------------------------------------
# wonder_ingest: idempotency
# ---------------------------------------------------------------------------


def test_ingest_is_idempotent_on_same_constituent_pair(
    store_with_constituents: MemoryStore,
) -> None:
    store = store_with_constituents
    phantom = _phantom("a", "b")

    r1 = wonder_ingest(store, [phantom])
    r2 = wonder_ingest(store, [phantom])

    assert r1.inserted == 1
    assert r1.skipped == 0
    assert r2.inserted == 0
    assert r2.skipped == 1

    beliefs = [store.get_belief(bid) for bid in store.list_belief_ids()]
    speculative = [b for b in beliefs if b is not None and b.type == BELIEF_SPECULATIVE]
    assert len(speculative) == 1


def test_ingest_distinct_constituent_pairs_are_not_deduped(
    store_with_constituents: MemoryStore,
) -> None:
    store = store_with_constituents
    phantom_ab = _phantom("a", "b", content="same text")
    phantom_bc = _phantom("b", "c", content="same text")

    r = wonder_ingest(store, [phantom_ab, phantom_bc])
    assert r.inserted == 2
    assert r.skipped == 0


def test_ingest_distinct_generators_are_not_deduped(
    store_with_constituents: MemoryStore,
) -> None:
    """Two phantoms with identical constituents but different generators
    persist as two distinct rows (v3.0 #644 option 2).

    The v1 key was generator-agnostic and would collapse these to one.
    Under the v2 key the generator is part of the hash basis, so each
    axis of a single --axes dispatch lands as its own phantom.
    """
    store = store_with_constituents
    phantom_gen_a = _phantom(
        "a", "b", content="axis A research", generator="subagent_dispatch:axis_A"
    )
    phantom_gen_b = _phantom(
        "a", "b", content="axis B research", generator="subagent_dispatch:axis_B"
    )

    r = wonder_ingest(store, [phantom_gen_a, phantom_gen_b])
    assert r.inserted == 2, "distinct generators must persist as distinct rows"
    assert r.skipped == 0

    beliefs = [store.get_belief(bid) for bid in store.list_belief_ids()]
    speculative = [
        b for b in beliefs if b is not None and b.type == BELIEF_SPECULATIVE
    ]
    assert len(speculative) == 2
    contents = {b.content for b in speculative}
    assert contents == {"axis A research", "axis B research"}


def test_ingest_same_generator_same_constituents_is_idempotent(
    store_with_constituents: MemoryStore,
) -> None:
    """Re-running the *same* dispatch (same generator, same constituents)
    is still a no-op under option 2 — generator-keyed dedup does not
    weaken the cross-run idempotency contract.
    """
    store = store_with_constituents
    phantom = _phantom("a", "b", generator="subagent_dispatch:axis_X")
    r1 = wonder_ingest(store, [phantom])
    r2 = wonder_ingest(store, [phantom])
    assert r1.inserted == 1
    assert r2.inserted == 0
    assert r2.skipped == 1


# ---------------------------------------------------------------------------
# wonder_gc: dry_run
# ---------------------------------------------------------------------------


def _old_ts(days_ago: int = 15) -> str:
    dt = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return dt.isoformat()


def _insert_old_speculative(store: MemoryStore, bid: str, days_ago: int = 15) -> str:
    """Insert a stale speculative belief directly (bypassing lifecycle for date control)."""
    b = Belief(
        id=bid,
        content=f"speculative content {bid}",
        content_hash=f"spec_hash_{bid}",
        alpha=_ALPHA_DEFAULT,
        beta=_BETA_DEFAULT,
        type=BELIEF_SPECULATIVE,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at=_old_ts(days_ago),
        last_retrieved_at=None,
        origin=ORIGIN_SPECULATIVE,
        retention_class="snapshot",
    )
    store.insert_belief(b)
    return bid


def test_gc_dry_run_reports_candidates_without_deleting(store: MemoryStore) -> None:
    _insert_old_speculative(store, "spec1")
    _insert_old_speculative(store, "spec2")

    result = wonder_gc(store, ttl_days=14, dry_run=True)

    assert isinstance(result, WonderGCResult)
    assert result.scanned == 2
    assert result.deleted == 0
    assert result.surviving == 2

    b1 = store.get_belief("spec1")
    assert b1 is not None
    assert b1.valid_to is None


# ---------------------------------------------------------------------------
# wonder_gc: non-dry-run sets valid_to
# ---------------------------------------------------------------------------


def test_gc_non_dry_run_sets_valid_to(store: MemoryStore) -> None:
    _insert_old_speculative(store, "spec1")

    result = wonder_gc(store, ttl_days=14, dry_run=False)

    assert result.scanned == 1
    assert result.deleted == 1
    assert result.surviving == 0

    b = store.get_belief("spec1")
    assert b is not None
    assert b.valid_to is not None


def test_gc_skips_fresh_beliefs(store: MemoryStore) -> None:
    b = Belief(
        id="fresh",
        content="fresh speculative",
        content_hash="fresh_hash",
        alpha=_ALPHA_DEFAULT,
        beta=_BETA_DEFAULT,
        type=BELIEF_SPECULATIVE,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at=datetime.now(timezone.utc).isoformat(),
        last_retrieved_at=None,
        origin=ORIGIN_SPECULATIVE,
        retention_class="snapshot",
    )
    store.insert_belief(b)

    result = wonder_gc(store, ttl_days=14, dry_run=False)
    assert result.deleted == 0

    b2 = store.get_belief("fresh")
    assert b2 is not None
    assert b2.valid_to is None


# ---------------------------------------------------------------------------
# wonder_gc: RESOLVES-edge preservation
# ---------------------------------------------------------------------------


def test_gc_preserves_phantom_with_outgoing_resolves_edge(store: MemoryStore) -> None:
    _insert_old_speculative(store, "spec1")
    store.insert_belief(_constituent("real"))
    store.insert_edge(Edge(src="spec1", dst="real", type=EDGE_RESOLVES, weight=1.0))

    result = wonder_gc(store, ttl_days=14, dry_run=False)
    assert result.deleted == 0

    b = store.get_belief("spec1")
    assert b is not None
    assert b.valid_to is None


def test_gc_preserves_phantom_with_incoming_resolves_edge(store: MemoryStore) -> None:
    _insert_old_speculative(store, "spec1")
    store.insert_belief(_constituent("real"))
    store.insert_edge(Edge(src="real", dst="spec1", type=EDGE_RESOLVES, weight=1.0))

    result = wonder_gc(store, ttl_days=14, dry_run=False)
    assert result.deleted == 0

    b = store.get_belief("spec1")
    assert b is not None
    assert b.valid_to is None


# ---------------------------------------------------------------------------
# wonder_gc: alpha-update preservation
# ---------------------------------------------------------------------------


def test_gc_preserves_phantom_with_alpha_update(store: MemoryStore) -> None:
    b = Belief(
        id="updated",
        content="updated alpha speculative",
        content_hash="updated_hash",
        alpha=0.5,
        beta=_BETA_DEFAULT,
        type=BELIEF_SPECULATIVE,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at=_old_ts(20),
        last_retrieved_at=None,
        origin=ORIGIN_SPECULATIVE,
        retention_class="snapshot",
    )
    store.insert_belief(b)

    result = wonder_gc(store, ttl_days=14, dry_run=False)
    assert result.deleted == 0

    b2 = store.get_belief("updated")
    assert b2 is not None
    assert b2.valid_to is None


# ---------------------------------------------------------------------------
# wonder_gc: idempotency
# ---------------------------------------------------------------------------


def test_gc_is_idempotent(store: MemoryStore) -> None:
    _insert_old_speculative(store, "spec1")
    _insert_old_speculative(store, "spec2")

    r1 = wonder_gc(store, ttl_days=14, dry_run=False)
    assert r1.deleted == 2

    r2 = wonder_gc(store, ttl_days=14, dry_run=False)
    assert r2.scanned == 0
    assert r2.deleted == 0
