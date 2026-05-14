"""#435 — retrieve_v2(with_doc_anchors=True) attaches anchor metadata.

`with_doc_anchors=False` (the default) keeps `RetrievalResult.doc_anchors`
empty so adapters that don't opt in see byte-identical wire shape.
`with_doc_anchors=True` populates a parallel list aligned with
`result.beliefs`.
"""
from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import pytest

from aelfrice.derivation_worker import run_worker
from aelfrice.doc_linker import (
    ANCHOR_INGEST,
    ANCHOR_MANUAL,
    link_belief_to_document,
)
from aelfrice.models import (
    BELIEF_FACTUAL,
    INGEST_SOURCE_FILESYSTEM,
    LOCK_NONE,
    ORIGIN_AGENT_INFERRED,
    RETENTION_FACT,
    Belief,
)
from aelfrice.retrieval import retrieve_v2
from aelfrice.store import MemoryStore


@pytest.fixture
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    s = MemoryStore(str(tmp_path / "ra.db"))
    yield s
    s.close()


def _seed_via_worker(store: MemoryStore, text: str, source_path: str) -> str:
    """Drive a belief through the worker so anchors are written through
    the production code path. Returns the resulting belief id."""
    log_id = store.record_ingest(
        source_kind=INGEST_SOURCE_FILESYSTEM,
        source_path=source_path,
        raw_text=text,
        raw_meta={"call_site": "filesystem_ingest"},
    )
    run_worker(store)
    row = store.get_ingest_log_entry(log_id)
    assert row is not None and row["derived_belief_ids"]
    return row["derived_belief_ids"][0]


def _seed_direct(store: MemoryStore, bid: str, content: str) -> Belief:
    ts = datetime.now(timezone.utc).isoformat()
    b = Belief(
        id=bid,
        content=content,
        content_hash=f"hash-{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at=ts,
        last_retrieved_at=None,
        session_id=None,
        origin=ORIGIN_AGENT_INFERRED,
        retention_class=RETENTION_FACT,
    )
    store.insert_belief(b)
    return b


def test_default_off_keeps_doc_anchors_empty(store: MemoryStore) -> None:
    """Hypothesis: default-off retrieve_v2 leaves doc_anchors=[] even when
    anchors exist in the store. Adapter wire shape stays byte-identical.
    Falsifiable by a non-empty `doc_anchors` field on default call."""
    bid = _seed_via_worker(
        store,
        "The architecture document explains storage.",
        "docs/architecture.md",
    )
    assert store.get_doc_anchors(bid)  # anchor exists in store

    result = retrieve_v2(store, "architecture document storage")
    assert result.doc_anchors == []


def test_with_doc_anchors_attaches_parallel_list(store: MemoryStore) -> None:
    """Hypothesis: with_doc_anchors=True returns a list of length
    len(beliefs); each entry carries every anchor for the corresponding
    belief; ordering is created_at ASC. Falsifiable by length mismatch
    or by a missing anchor."""
    bid = _seed_via_worker(
        store,
        "The architecture document explains storage.",
        "docs/architecture.md",
    )
    # Stack a manual anchor on the same belief.
    link_belief_to_document(
        store,
        bid,
        "https://example.com/architecture",
        anchor_type=ANCHOR_MANUAL,
    )

    result = retrieve_v2(
        store,
        "architecture document storage",
        with_doc_anchors=True,
    )
    assert result.beliefs, "retrieve must return at least one belief"
    assert len(result.doc_anchors) == len(result.beliefs)

    # Find the anchored belief in the result and check its anchors.
    for b, anchors in zip(result.beliefs, result.doc_anchors):
        if b.id == bid:
            uris = sorted(a.doc_uri for a in anchors)
            assert uris == [
                "file:docs/architecture.md",
                "https://example.com/architecture",
            ]
            assert any(a.anchor_type == ANCHOR_INGEST for a in anchors)
            assert any(a.anchor_type == ANCHOR_MANUAL for a in anchors)
            break
    else:
        pytest.fail(f"belief {bid} not in retrieve result")


def test_with_doc_anchors_handles_anchorless_beliefs(
    store: MemoryStore,
) -> None:
    """Beliefs that have no anchors get an empty list at the same index
    (not omitted). The parallel-list contract requires len(doc_anchors)
    == len(beliefs)."""
    bid_with_anchor = _seed_via_worker(
        store,
        "The system uses SQLite for storage.",
        "docs/storage.md",
    )
    # Anchor-free belief, written directly (no ingest log entry).
    _seed_direct(store, "naked", "Storage layer matters.")

    result = retrieve_v2(store, "storage", with_doc_anchors=True)
    assert len(result.doc_anchors) == len(result.beliefs)

    for b, anchors in zip(result.beliefs, result.doc_anchors):
        if b.id == bid_with_anchor:
            assert anchors and anchors[0].doc_uri == "file:docs/storage.md"
        elif b.id == "naked":
            assert anchors == []


def test_with_doc_anchors_off_does_not_query_store(
    store: MemoryStore,
) -> None:
    """Default off path performs no batched anchor SELECT — proxy: an
    empty store of beliefs still returns doc_anchors=[].

    This protects the byte-identical-wire-shape contract for adapters
    that don't opt in: turning the flag on must be the only path that
    reads `belief_documents`.
    """
    result = retrieve_v2(store, "anything")
    assert result.beliefs == []
    assert result.doc_anchors == []
