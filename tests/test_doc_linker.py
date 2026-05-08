"""Tests for the doc linker (#435) — storage, idempotency, schema migration."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from aelfrice.doc_linker import (
    ANCHOR_DERIVED,
    ANCHOR_INGEST,
    ANCHOR_MANUAL,
    DocAnchor,
    file_uri_from_path,
    get_doc_anchors,
    link_belief_to_document,
)
from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    LOCK_USER,
    ORIGIN_AGENT_INFERRED,
    ORIGIN_USER_STATED,
    RETENTION_FACT,
    Belief,
)
from aelfrice.store import MemoryStore


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _belief(bid: str, content: str, *, lock: str = LOCK_NONE) -> Belief:
    ts = _ts()
    return Belief(
        id=bid,
        content=content,
        content_hash=f"hash-{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=lock,
        locked_at=ts if lock == LOCK_USER else None,
        demotion_pressure=0,
        created_at=ts,
        last_retrieved_at=None,
        session_id=None,
        origin=ORIGIN_USER_STATED if lock == LOCK_USER else ORIGIN_AGENT_INFERRED,
        retention_class=RETENTION_FACT,
    )


def test_link_belief_to_document_round_trip(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path / "rt.db"))
    try:
        store.insert_belief(_belief("b1", "hello world"))

        anchor = link_belief_to_document(
            store,
            "b1",
            "file:src/foo.py#L10-L20",
            anchor_type=ANCHOR_INGEST,
            position_hint="L10-L20",
        )

        assert isinstance(anchor, DocAnchor)
        assert anchor.belief_id == "b1"
        assert anchor.doc_uri == "file:src/foo.py#L10-L20"
        assert anchor.anchor_type == ANCHOR_INGEST
        assert anchor.position_hint == "L10-L20"
        assert anchor.created_at > 0

        out = get_doc_anchors(store, "b1")
        assert out == [anchor]
    finally:
        store.close()


def test_link_belief_to_document_idempotent(tmp_path: Path) -> None:
    """A3: idempotency — N writes of the same (belief_id, doc_uri) → one row."""
    store = MemoryStore(str(tmp_path / "idem.db"))
    try:
        store.insert_belief(_belief("b1", "x"))
        first = link_belief_to_document(
            store, "b1", "file:src/foo.py", position_hint="L1"
        )
        for _ in range(5):
            again = link_belief_to_document(
                store,
                "b1",
                "file:src/foo.py",
                anchor_type=ANCHOR_MANUAL,  # different! still no-op write
                position_hint="L99",
            )
            # Returned row reflects the canonical (first-write) state —
            # anchor_type / position_hint are NOT overwritten on no-op.
            assert again.created_at == first.created_at
            assert again.anchor_type == ANCHOR_INGEST
            assert again.position_hint == "L1"

        anchors = get_doc_anchors(store, "b1")
        assert len(anchors) == 1
        assert anchors[0].created_at == first.created_at
    finally:
        store.close()


def test_link_belief_to_document_multiple_uris(tmp_path: Path) -> None:
    """A belief can have many anchors. Ordering is created_at ASC."""
    store = MemoryStore(str(tmp_path / "multi.db"))
    try:
        store.insert_belief(_belief("b1", "x"))
        first = link_belief_to_document(
            store, "b1", "file:src/foo.py"
        )
        second = link_belief_to_document(
            store, "b1", "https://example.com/foo#section"
        )
        third = link_belief_to_document(
            store, "b1", "file:docs/bar.md", anchor_type=ANCHOR_MANUAL
        )

        anchors = get_doc_anchors(store, "b1")
        assert [a.doc_uri for a in anchors] == [
            first.doc_uri,
            second.doc_uri,
            third.doc_uri,
        ]
    finally:
        store.close()


def test_link_belief_to_document_rejects_empty_uri(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path / "empty.db"))
    try:
        store.insert_belief(_belief("b1", "x"))
        with pytest.raises(ValueError, match="non-empty"):
            link_belief_to_document(store, "b1", "")
    finally:
        store.close()


def test_link_belief_to_document_rejects_unknown_anchor_type(
    tmp_path: Path,
) -> None:
    store = MemoryStore(str(tmp_path / "type.db"))
    try:
        store.insert_belief(_belief("b1", "x"))
        with pytest.raises(ValueError, match="anchor_type"):
            link_belief_to_document(
                store, "b1", "file:foo", anchor_type="auto",
            )
    finally:
        store.close()


def test_link_belief_to_document_fk_required(tmp_path: Path) -> None:
    """Linking to a non-existent belief fails at the FK layer."""
    store = MemoryStore(str(tmp_path / "fk.db"))
    try:
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            link_belief_to_document(store, "b-missing", "file:foo.py")
    finally:
        store.close()


def test_get_doc_anchors_empty_for_unknown_belief(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path / "empty.db"))
    try:
        assert get_doc_anchors(store, "nope") == []
    finally:
        store.close()


def test_get_doc_anchors_batch(tmp_path: Path) -> None:
    """Batched fetch returns an entry per requested id, even with no anchors."""
    store = MemoryStore(str(tmp_path / "batch.db"))
    try:
        store.insert_belief(_belief("b1", "x"))
        store.insert_belief(_belief("b2", "y"))
        store.insert_belief(_belief("b3", "z"))
        link_belief_to_document(store, "b1", "file:a.py")
        link_belief_to_document(store, "b1", "file:b.py")
        link_belief_to_document(store, "b3", "file:c.py")

        out = store.get_doc_anchors_batch(["b1", "b2", "b3"])
        assert sorted(out.keys()) == ["b1", "b2", "b3"]
        assert [a.doc_uri for a in out["b1"]] == ["file:a.py", "file:b.py"]
        assert out["b2"] == []
        assert [a.doc_uri for a in out["b3"]] == ["file:c.py"]

        # Empty input → empty dict, no SQL.
        assert store.get_doc_anchors_batch([]) == {}
    finally:
        store.close()


def test_anchor_cascades_on_belief_delete(tmp_path: Path) -> None:
    """ON DELETE CASCADE: deleting a belief drops its anchors."""
    store = MemoryStore(str(tmp_path / "cascade.db"))
    try:
        store.insert_belief(_belief("b1", "x"))
        link_belief_to_document(store, "b1", "file:a.py")
        link_belief_to_document(store, "b1", "file:b.py")
        assert len(get_doc_anchors(store, "b1")) == 2

        # Direct DELETE bypasses the audit-row insertion path of `aelf
        # delete`; we just want the FK cascade behaviour confirmed.
        store._conn.execute("DELETE FROM beliefs WHERE id = ?", ("b1",))
        store._conn.commit()
        assert get_doc_anchors(store, "b1") == []
    finally:
        store.close()


def test_schema_migration_creates_table_on_existing_store(
    tmp_path: Path,
) -> None:
    """A4: schema migration — first open of a store with no
    `belief_documents` table creates it and populates anchors normally.

    Simulates a v1.7-era store by opening with sqlite3 directly, creating
    only the legacy schema, then re-opening through MemoryStore (which
    runs the additive `CREATE TABLE IF NOT EXISTS belief_documents` on
    every open).
    """
    db = tmp_path / "legacy.db"
    legacy_conn = sqlite3.connect(str(db))
    legacy_conn.execute(
        """
        CREATE TABLE beliefs (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            alpha REAL NOT NULL,
            beta REAL NOT NULL,
            type TEXT NOT NULL,
            lock_level TEXT NOT NULL,
            locked_at TEXT,
            demotion_pressure INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            last_retrieved_at TEXT
        )
        """
    )
    legacy_conn.commit()
    legacy_conn.close()

    store = MemoryStore(str(db))
    try:
        cur = store._conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='belief_documents'"
        )
        assert cur.fetchone() is not None, (
            "belief_documents table must be created on first open of legacy store"
        )

        store.insert_belief(_belief("b1", "x"))
        anchor = link_belief_to_document(store, "b1", "file:foo.py")
        assert get_doc_anchors(store, "b1") == [anchor]
    finally:
        store.close()


def test_anchor_type_enum_includes_derived_reservation(
    tmp_path: Path,
) -> None:
    """`anchor_type='derived'` is reserved at v2.0.0 — accepted by the writer.

    The spec defers the *writer* (retrieval-time inference) to v2.x but
    keeps the enum value live so a future revision lands without a
    schema migration. Verify the reservation by writing one row.
    """
    store = MemoryStore(str(tmp_path / "derived.db"))
    try:
        store.insert_belief(_belief("b1", "x"))
        anchor = link_belief_to_document(
            store,
            "b1",
            "file:src/foo.py",
            anchor_type=ANCHOR_DERIVED,
            position_hint="inferred",
        )
        assert anchor.anchor_type == ANCHOR_DERIVED
    finally:
        store.close()


# --- file_uri_from_path -------------------------------------------------


def test_file_uri_from_path_relative(tmp_path: Path) -> None:
    """When project_root contains source_path, URI is repo-relative."""
    src = tmp_path / "src" / "foo.py"
    src.parent.mkdir(parents=True)
    src.write_text("x")
    uri = file_uri_from_path(
        str(src),
        project_root=tmp_path,
        position_hint="L1-L5",
    )
    assert uri == "file:src/foo.py#L1-L5"


def test_file_uri_from_path_absolute_when_outside_project(
    tmp_path: Path,
) -> None:
    """source_path outside project_root falls back to the absolute path."""
    elsewhere = tmp_path / "elsewhere.py"
    elsewhere.write_text("x")
    proj = tmp_path / "proj"
    proj.mkdir()
    uri = file_uri_from_path(str(elsewhere), project_root=proj)
    # Falls through to the absolute path; "/abs/path" form gets file://
    assert uri.startswith("file://"), uri
    assert str(elsewhere) in uri


def test_file_uri_from_path_no_position_hint() -> None:
    uri = file_uri_from_path("docs/foo.md")
    assert uri == "file:docs/foo.md"
    assert "#" not in uri
