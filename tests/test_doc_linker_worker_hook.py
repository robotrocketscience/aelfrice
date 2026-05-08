"""#435 — derivation worker writes a doc anchor when source_path is set.

The worker calls the linker AFTER `insert_or_corroborate`, so the anchor
fires for both new beliefs and corroborations. Idempotent on
`(belief_id, doc_uri)` so re-derive of the same row produces no extra
anchors.

Skips when `source_path` is None (transcripts, lock without --doc).
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from aelfrice.derivation_worker import run_worker
from aelfrice.doc_linker import ANCHOR_INGEST, get_doc_anchors
from aelfrice.models import (
    CORROBORATION_SOURCE_FILESYSTEM_INGEST,
    CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
    INGEST_SOURCE_FILESYSTEM,
)
from aelfrice.store import MemoryStore


@pytest.fixture
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    s = MemoryStore(str(tmp_path / "wh.db"))
    yield s
    s.close()


def _record(
    store: MemoryStore,
    text: str,
    *,
    source_path: str | None,
    call_site: str = CORROBORATION_SOURCE_FILESYSTEM_INGEST,
) -> str:
    return store.record_ingest(
        source_kind=INGEST_SOURCE_FILESYSTEM,
        source_path=source_path,
        raw_text=text,
        raw_meta={"call_site": call_site},
    )


def test_worker_writes_anchor_when_source_path_is_set(
    store: MemoryStore,
) -> None:
    """Hypothesis: after `run_worker()`, every newly-derived belief whose
    log row carried `source_path` has exactly one `belief_documents` row
    pointing at that path. Falsifiable by a missing anchor or a wrong
    URI."""
    log_id = _record(
        store,
        "The system uses SQLite for storage.",
        source_path="docs/architecture.md",
    )

    result = run_worker(store)
    assert result.beliefs_inserted == 1

    row = store.get_ingest_log_entry(log_id)
    assert row is not None
    bid = row["derived_belief_ids"][0]
    anchors = get_doc_anchors(store, bid)
    assert len(anchors) == 1
    a = anchors[0]
    assert a.anchor_type == ANCHOR_INGEST
    assert a.doc_uri == "file:docs/architecture.md"
    assert a.position_hint is None


def test_worker_skips_anchor_when_source_path_is_none(
    store: MemoryStore,
) -> None:
    """Transcript-ingest rows have `source_path=None`. No anchor written."""
    log_id = _record(
        store,
        "I prefer atomic commits over batches.",
        source_path=None,
        call_site=CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
    )

    run_worker(store)

    row = store.get_ingest_log_entry(log_id)
    assert row is not None
    derived = row["derived_belief_ids"]
    if not derived:
        # Some classifier configurations skip transcript text; nothing
        # to assert about anchors. Test still demonstrates the
        # source_path=None path doesn't crash the worker.
        return
    bid = derived[0]
    assert get_doc_anchors(store, bid) == []


def test_worker_anchor_idempotent_on_re_derive(store: MemoryStore) -> None:
    """Hypothesis: re-running the worker on the same row produces no
    extra anchors. Falsifiable by anchor count > 1 after the second pass."""
    _record(
        store,
        "The system uses SQLite for storage.",
        source_path="docs/foo.md",
    )

    run_worker(store)
    # Force a second pass by clearing the stamp on the row.
    store._conn.execute(
        "UPDATE ingest_log SET derived_belief_ids = NULL"
    )
    store._conn.commit()
    run_worker(store)

    cur = store._conn.execute("SELECT id FROM beliefs")
    bids = [r["id"] for r in cur.fetchall()]
    assert len(bids) == 1
    anchors = get_doc_anchors(store, bids[0])
    assert len(anchors) == 1


def test_worker_anchor_persists_through_corroboration(
    store: MemoryStore,
) -> None:
    """A second ingest of the same content (same content_hash) corroborates
    rather than inserting. The first ingest's anchor remains; if the
    second ingest carries a *different* doc_uri, both anchors stack.

    Hypothesis: anchors are 1:1 with (belief_id, doc_uri), not (belief_id),
    so two ingests of the same text from two different files produce two
    anchors on one belief.
    """
    _record(
        store,
        "Same belief content for both ingests.",
        source_path="docs/a.md",
    )
    run_worker(store)
    _record(
        store,
        "Same belief content for both ingests.",
        source_path="docs/b.md",
    )
    run_worker(store)

    cur = store._conn.execute("SELECT id FROM beliefs")
    bids = [r["id"] for r in cur.fetchall()]
    assert len(bids) == 1, "expected one belief (corroborated)"
    anchors = get_doc_anchors(store, bids[0])
    uris = sorted(a.doc_uri for a in anchors)
    assert uris == ["file:docs/a.md", "file:docs/b.md"]
