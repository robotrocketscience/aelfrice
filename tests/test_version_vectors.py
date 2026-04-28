"""Tests for the v1.5.0 #204 version-vector forward-compat schema.

Acceptance criteria from the issue:

1. New beliefs and edges carry a non-trivial vector after the change.
2. `aelf:health` does not regress  (covered by the existing
   `tests/test_doctor_*` suite — no new tests here).
3. Migration applied to a v1.4 DB produces vectors `{<scope_id>: 1}`
   on all rows.

Plus invariants the spec implies:

- `local_scope_id` is stable across re-opens of the same DB.
- Two distinct DB files have distinct `local_scope_id` values.
- `vv[local_scope] += 1` on every write (insert AND update).
- delete_belief / delete_edge cascade is unchanged (version rows
  persist; reconcile at v3 will GC). The test here just guards the
  belief-delete path against an accidental crash.
"""
from __future__ import annotations

from pathlib import Path

from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief, Edge
from aelfrice.store import (
    SCHEMA_META_LOCAL_SCOPE_ID,
    SCHEMA_META_VERSION_VECTOR_BACKFILL,
    MemoryStore,
)


def _mk(bid: str, content: str = "x") -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        demotion_pressure=0,
        created_at="2026-04-28T00:00:00Z",
        last_retrieved_at=None,
    )


# --- Scope id init -------------------------------------------------------


def test_local_scope_id_generated_on_first_open(tmp_path: Path) -> None:
    db = str(tmp_path / "first.db")
    s = MemoryStore(db)
    scope = s.local_scope_id
    assert scope
    # Hex token, 32 chars.
    assert len(scope) == 32
    assert s.get_schema_meta(SCHEMA_META_LOCAL_SCOPE_ID) == scope
    s.close()


def test_local_scope_id_stable_across_reopens(tmp_path: Path) -> None:
    db = str(tmp_path / "stable.db")
    s = MemoryStore(db)
    first = s.local_scope_id
    s.close()
    s = MemoryStore(db)
    assert s.local_scope_id == first
    s.close()


def test_distinct_dbs_have_distinct_scope_ids(tmp_path: Path) -> None:
    a = MemoryStore(str(tmp_path / "a.db"))
    b = MemoryStore(str(tmp_path / "b.db"))
    assert a.local_scope_id != b.local_scope_id
    a.close()
    b.close()


# --- AC1: writes produce non-trivial vectors -----------------------------


def test_insert_belief_creates_version_one() -> None:
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("b1"))
    vv = s.get_belief_version_vector("b1")
    assert vv == {s.local_scope_id: 1}


def test_update_belief_increments_local_counter() -> None:
    s = MemoryStore(":memory:")
    b = _mk("b1", "first")
    s.insert_belief(b)
    b2 = _mk("b1", "second")
    s.update_belief(b2)
    vv = s.get_belief_version_vector("b1")
    assert vv == {s.local_scope_id: 2}


def test_three_updates_make_counter_four() -> None:
    """One insert + three updates -> counter == 4."""
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("b1"))
    for _ in range(3):
        s.update_belief(_mk("b1", "next"))
    assert s.get_belief_version_vector("b1") == {s.local_scope_id: 4}


def test_delete_belief_does_not_crash_with_version_row() -> None:
    """delete_belief leaves the version row alone (v3 reconcile will
    GC). The test only guards that the delete path itself does not
    raise."""
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("b1"))
    s.delete_belief("b1")
    assert s.get_belief("b1") is None
    # Version row persists for reconcile-at-v3 contract; reading it
    # returns a non-empty map.
    assert s.get_belief_version_vector("b1") == {s.local_scope_id: 1}


def test_insert_edge_creates_version_one() -> None:
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("b1"))
    s.insert_belief(_mk("b2"))
    s.insert_edge(Edge(src="b1", dst="b2", type="cites", weight=1.0,
                       anchor_text=None))
    vv = s.get_edge_version_vector("b1", "b2", "cites")
    assert vv == {s.local_scope_id: 1}


def test_update_edge_increments_local_counter() -> None:
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("b1"))
    s.insert_belief(_mk("b2"))
    e = Edge(src="b1", dst="b2", type="cites", weight=1.0, anchor_text=None)
    s.insert_edge(e)
    s.update_edge(Edge(src="b1", dst="b2", type="cites", weight=0.5,
                       anchor_text="updated"))
    vv = s.get_edge_version_vector("b1", "b2", "cites")
    assert vv == {s.local_scope_id: 2}


# --- AC3: backfill on legacy DB ------------------------------------------


def test_backfill_stamps_legacy_rows(tmp_path: Path) -> None:
    """Open a fresh DB, insert beliefs + edges, drop the
    backfill marker AND the version rows, re-open. The backfill
    should re-stamp `{scope: 1}` on every existing row."""
    db = str(tmp_path / "legacy.db")
    s = MemoryStore(db)
    scope = s.local_scope_id
    s.insert_belief(_mk("legacy_b1"))
    s.insert_belief(_mk("legacy_b2"))
    s.insert_edge(Edge(src="legacy_b1", dst="legacy_b2",
                       type="cites", weight=1.0, anchor_text=None))
    # Simulate a pre-#204 DB: drop version rows + backfill marker.
    s._conn.execute("DELETE FROM belief_versions")
    s._conn.execute("DELETE FROM edge_versions")
    s._conn.execute(
        "DELETE FROM schema_meta WHERE key = ?",
        (SCHEMA_META_VERSION_VECTOR_BACKFILL,),
    )
    s._conn.commit()
    s.close()
    # Re-open: backfill should run.
    s = MemoryStore(db)
    assert s.get_belief_version_vector("legacy_b1") == {scope: 1}
    assert s.get_belief_version_vector("legacy_b2") == {scope: 1}
    assert s.get_edge_version_vector(
        "legacy_b1", "legacy_b2", "cites",
    ) == {scope: 1}
    assert s.get_schema_meta(SCHEMA_META_VERSION_VECTOR_BACKFILL)
    s.close()


def test_backfill_idempotent_on_second_open(tmp_path: Path) -> None:
    """Backfill marker present -> second open is a no-op."""
    db = str(tmp_path / "idempotent.db")
    s = MemoryStore(db)
    s.insert_belief(_mk("b1"))
    s.close()
    # Second open: backfill marker already stamped, no extra writes.
    s = MemoryStore(db)
    assert s.get_belief_version_vector("b1") == {s.local_scope_id: 1}
    s.close()


def test_backfill_skipped_on_fresh_v15_store() -> None:
    """A fresh v1.5+ store with no rows still stamps the marker so
    later opens short-circuit, exactly like the entity-index
    backfill semantics."""
    s = MemoryStore(":memory:")
    assert s.get_schema_meta(SCHEMA_META_VERSION_VECTOR_BACKFILL)
    s.close()


# --- Invariants ----------------------------------------------------------


def test_get_belief_version_vector_unknown_id_returns_empty() -> None:
    s = MemoryStore(":memory:")
    assert s.get_belief_version_vector("never-existed") == {}


def test_get_edge_version_vector_unknown_edge_returns_empty() -> None:
    s = MemoryStore(":memory:")
    assert s.get_edge_version_vector("a", "b", "cites") == {}


def test_local_scope_id_is_only_scope_at_v15(tmp_path: Path) -> None:
    """Today's contract: the local scope id is the only key in any
    version vector. Federation lands at v3 with peer scope ids."""
    s = MemoryStore(str(tmp_path / "single.db"))
    s.insert_belief(_mk("b1"))
    vv = s.get_belief_version_vector("b1")
    assert list(vv) == [s.local_scope_id]
    s.close()
