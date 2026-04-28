"""ingest_triples idempotency, session_id propagation, self-edge handling."""
from __future__ import annotations

from aelfrice.models import EDGE_SUPPORTS
from aelfrice.store import MemoryStore
from aelfrice.triple_extractor import (
    Triple,
    extract_triples,
    ingest_triples,
)


def test_ingest_creates_beliefs_and_edge() -> None:
    store = MemoryStore(":memory:")
    try:
        triples = extract_triples("the new index supports faster queries")
        r = ingest_triples(store, triples)
        assert len(r.new_beliefs) == 2
        assert len(r.new_edges) == 1
        src, dst, etype = r.new_edges[0]
        edge = store.get_edge(src, dst, etype)
        assert edge is not None
        assert edge.type == EDGE_SUPPORTS
        assert "supports" in (edge.anchor_text or "")
    finally:
        store.close()


def test_ingest_idempotent_on_re_run() -> None:
    store = MemoryStore(":memory:")
    try:
        triples = extract_triples("the new index supports faster queries")
        r1 = ingest_triples(store, triples)
        r2 = ingest_triples(store, triples)
        assert r1.new_beliefs and r1.new_edges
        assert r2.new_beliefs == []
        assert r2.new_edges == []
        assert r2.skipped_duplicate_edges == 1
    finally:
        store.close()


def test_session_id_written_on_new_beliefs() -> None:
    store = MemoryStore(":memory:")
    try:
        sess = store.create_session(model="haiku").id
        triples = extract_triples("the new index supports faster queries")
        ingest_triples(store, triples, session_id=sess)
        rows = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
            "SELECT session_id FROM beliefs"
        ).fetchall()
        assert rows
        for row in rows:
            assert row["session_id"] == sess
    finally:
        store.close()


def test_no_session_id_leaves_field_null() -> None:
    store = MemoryStore(":memory:")
    try:
        triples = extract_triples("the new index supports faster queries")
        ingest_triples(store, triples)
        rows = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
            "SELECT session_id FROM beliefs"
        ).fetchall()
        for row in rows:
            assert row["session_id"] is None
    finally:
        store.close()


def test_self_edge_skipped() -> None:
    """A triple whose subject and object normalize to the same phrase
    must not produce an edge."""
    store = MemoryStore(":memory:")
    try:
        # Force a self-edge by hand — the extractor itself rarely emits these.
        t = Triple(
            subject="the parser", relation=EDGE_SUPPORTS,
            object="the parser", anchor_text="the parser supports the parser",
        )
        r = ingest_triples(store, [t])
        assert r.new_edges == []
        assert r.skipped_no_subject_or_object == 1
    finally:
        store.close()


def test_anchor_text_persisted_on_edge() -> None:
    store = MemoryStore(":memory:")
    try:
        triples = extract_triples(
            "the proposal cites the prior memo as background"
        )
        r = ingest_triples(store, triples)
        assert r.new_edges
        src, dst, etype = r.new_edges[0]
        edge = store.get_edge(src, dst, etype)
        assert edge is not None
        assert edge.anchor_text is not None
        assert "cites" in edge.anchor_text
    finally:
        store.close()


def test_same_phrase_resolves_to_same_belief_id_across_calls() -> None:
    """Triples extracted in different calls that mention the same noun
    phrase should land on the same belief — the canonical-id property."""
    store = MemoryStore(":memory:")
    try:
        ingest_triples(store, extract_triples(
            "the cache layer relates to retrieval"
        ))
        ingest_triples(store, extract_triples(
            "the cache layer supports the index"
        ))
        # The cache layer noun phrase appears in both — should be one belief.
        rows = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
            "SELECT id, content FROM beliefs WHERE content = ?",
            ("the cache layer",),
        ).fetchall()
        assert len(rows) == 1
    finally:
        store.close()
