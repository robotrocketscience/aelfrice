"""#435 — `aelf lock --doc=URI` writes a manual anchor on the locked belief.

The lock entry point passes `source_path=None` to the derivation worker
(cli_remember has no canonical doc URI), so the worker's ingest-time
hook does NOT fire. The CLI handler is the only path that writes anchors
on the lock surface. Idempotent on re-lock with the same URI.
"""
from __future__ import annotations

import argparse
import io
from collections.abc import Iterator
from pathlib import Path

import pytest

from aelfrice.cli import _cmd_lock
from aelfrice.doc_linker import ANCHOR_MANUAL, get_doc_anchors
from aelfrice.store import MemoryStore


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[MemoryStore]:
    db = tmp_path / "lock-doc.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    s = MemoryStore(str(db))
    yield s
    s.close()


def _ns(
    statement: str,
    *,
    session_id: str | None = None,
    doc_uri: str | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        statement=statement, session_id=session_id, doc_uri=doc_uri,
    )


def _locked_belief_id(store: MemoryStore) -> str:
    rows = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "SELECT id FROM beliefs"
    ).fetchall()
    assert len(rows) == 1
    return str(rows[0]["id"])


def test_lock_without_doc_writes_no_anchor(store: MemoryStore) -> None:
    """Hypothesis: aelf lock without --doc produces no belief_documents
    rows. Falsifiable by any anchor on the locked belief."""
    rc = _cmd_lock(_ns("atomic commits beat batched"), io.StringIO())
    assert rc == 0

    bid = _locked_belief_id(store)
    assert get_doc_anchors(store, bid) == []


def test_lock_with_doc_writes_manual_anchor(store: MemoryStore) -> None:
    """Hypothesis: aelf lock --doc=URI produces exactly one manual
    anchor with the supplied URI. Falsifiable by missing anchor, wrong
    type, or wrong URI."""
    rc = _cmd_lock(
        _ns(
            "the sky is blue",
            doc_uri="https://example.com/notes#sky",
        ),
        io.StringIO(),
    )
    assert rc == 0

    bid = _locked_belief_id(store)
    anchors = get_doc_anchors(store, bid)
    assert len(anchors) == 1
    a = anchors[0]
    assert a.doc_uri == "https://example.com/notes#sky"
    assert a.anchor_type == ANCHOR_MANUAL


def test_relock_with_same_doc_is_idempotent(store: MemoryStore) -> None:
    """Re-locking the same statement with the same --doc adds no extra
    anchor rows. The lock body itself is re-applied (lock-upgrade); the
    anchor write is a no-op via INSERT OR IGNORE."""
    args = _ns("atomic commits beat batched", doc_uri="file:CLAUDE.md#commits")
    _cmd_lock(args, io.StringIO())
    _cmd_lock(args, io.StringIO())

    bid = _locked_belief_id(store)
    anchors = get_doc_anchors(store, bid)
    assert len(anchors) == 1
