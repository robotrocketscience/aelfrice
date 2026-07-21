"""#1135 hot-path indexes: unstamped ingest_log scan, locked-belief
tier, edge-type probe.

Asserts both existence (sqlite_master) and that the planner actually
uses each index for the exact query shape the hot path issues — an
index that exists but is not chosen is a silent regression.
"""
from __future__ import annotations

from aelfrice.store import MemoryStore


def _index_names(store: MemoryStore) -> set[str]:
    cur = store._conn.execute(  # type: ignore[attr-defined]
        "SELECT name FROM sqlite_master WHERE type = 'index'"
    )
    return {str(r["name"]) for r in cur.fetchall()}


def _plan(store: MemoryStore, sql: str) -> str:
    cur = store._conn.execute(  # type: ignore[attr-defined]
        "EXPLAIN QUERY PLAN " + sql
    )
    return " | ".join(str(r["detail"]) for r in cur.fetchall())


def test_hot_path_indexes_exist() -> None:
    store = MemoryStore(":memory:")
    try:
        names = _index_names(store)
        assert "idx_ingest_log_unstamped" in names
        assert "idx_beliefs_locked" in names
        assert "idx_edges_type" in names
    finally:
        store.close()


def test_unstamped_scan_uses_partial_index() -> None:
    store = MemoryStore(":memory:")
    try:
        plan = _plan(
            store,
            "SELECT id FROM ingest_log "
            "WHERE derived_belief_ids IS NULL ORDER BY id",
        )
        assert "idx_ingest_log_unstamped" in plan, plan
    finally:
        store.close()


def test_locked_beliefs_query_uses_partial_index() -> None:
    store = MemoryStore(":memory:")
    try:
        plan = _plan(
            store,
            "SELECT * FROM beliefs b "
            "WHERE b.lock_level != 'none' AND b.valid_to IS NULL "
            "ORDER BY b.locked_at DESC, b.id ASC",
        )
        assert "idx_beliefs_locked" in plan, plan
    finally:
        store.close()


def test_edge_type_probe_uses_index() -> None:
    store = MemoryStore(":memory:")
    try:
        plan = _plan(
            store,
            "SELECT 1 FROM edges WHERE type = 'TEMPORAL_NEXT' LIMIT 1",
        )
        assert "idx_edges_type" in plan, plan
    finally:
        store.close()
