"""ingest_jsonl: turns.jsonl -> beliefs + DERIVED_FROM edges, idempotent."""
from __future__ import annotations

import json
from pathlib import Path

from aelfrice.ingest import ingest_jsonl
from aelfrice.models import EDGE_DERIVED_FROM
from aelfrice.store import MemoryStore


def _write_jsonl(path: Path, lines: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")


def test_missing_file_returns_empty(tmp_path: Path) -> None:
    store = MemoryStore(":memory:")
    try:
        r = ingest_jsonl(store, tmp_path / "nope.jsonl")
        assert r.lines_read == 0
        assert r.turns_ingested == 0
    finally:
        store.close()


def test_ingest_writes_beliefs_with_session_id(tmp_path: Path) -> None:
    p = tmp_path / "turns.jsonl"
    _write_jsonl(p, [
        {
            "schema_version": 1, "ts": "2026-04-27T00:00:00Z",
            "role": "user", "text": "The project uses SQLite for storage.",
            "session_id": "S1", "turn_id": "t1", "context": {},
        },
    ])
    store = MemoryStore(":memory:")
    try:
        r = ingest_jsonl(store, p)
        assert r.turns_ingested == 1
        assert r.beliefs_inserted >= 1
        rows = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
            "SELECT session_id FROM beliefs"
        ).fetchall()
        for row in rows:
            assert row["session_id"] == "S1"
    finally:
        store.close()


def test_consecutive_turns_get_derived_from_edge(tmp_path: Path) -> None:
    p = tmp_path / "turns.jsonl"
    _write_jsonl(p, [
        {"role": "user", "text": "Pi equals 3.14.", "session_id": "S",
         "turn_id": "t1"},
        {"role": "assistant", "text": "Tau equals 6.28.", "session_id": "S",
         "turn_id": "t2"},
    ])
    store = MemoryStore(":memory:")
    try:
        r = ingest_jsonl(store, p)
        assert r.edges_inserted >= 1
        edges = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
            "SELECT * FROM edges WHERE type = ?", (EDGE_DERIVED_FROM,)
        ).fetchall()
        assert len(edges) >= 1
        # Anchor text is the *prior* turn's text.
        assert any(e["anchor_text"] == "Pi equals 3.14." for e in edges)
    finally:
        store.close()


def test_no_edge_across_different_sessions(tmp_path: Path) -> None:
    p = tmp_path / "turns.jsonl"
    _write_jsonl(p, [
        {"role": "user", "text": "Pi equals 3.14.", "session_id": "S1",
         "turn_id": "t1"},
        {"role": "user", "text": "Tau equals 6.28.", "session_id": "S2",
         "turn_id": "t2"},
    ])
    store = MemoryStore(":memory:")
    try:
        r = ingest_jsonl(store, p)
        assert r.edges_inserted == 0
    finally:
        store.close()


def test_no_edge_when_session_id_missing(tmp_path: Path) -> None:
    p = tmp_path / "turns.jsonl"
    _write_jsonl(p, [
        {"role": "user", "text": "Pi equals 3.14.", "turn_id": "t1"},
        {"role": "user", "text": "Tau equals 6.28.", "turn_id": "t2"},
    ])
    store = MemoryStore(":memory:")
    try:
        r = ingest_jsonl(store, p)
        assert r.edges_inserted == 0
    finally:
        store.close()


def test_idempotent_re_ingest(tmp_path: Path) -> None:
    p = tmp_path / "turns.jsonl"
    _write_jsonl(p, [
        {"role": "user", "text": "Pi equals 3.14.", "session_id": "S",
         "turn_id": "t1"},
        {"role": "assistant", "text": "Tau equals 6.28.", "session_id": "S",
         "turn_id": "t2"},
    ])
    store = MemoryStore(":memory:")
    try:
        r1 = ingest_jsonl(store, p)
        r2 = ingest_jsonl(store, p)
        assert r2.beliefs_inserted == 0
        assert r2.edges_inserted == 0
        assert r2.turns_ingested == r1.turns_ingested  # turns "ingested" still counted
    finally:
        store.close()


def test_compaction_markers_skipped(tmp_path: Path) -> None:
    p = tmp_path / "turns.jsonl"
    _write_jsonl(p, [
        {"event": "compaction_start", "ts": "2026-01-01"},
        {"role": "user", "text": "x.", "session_id": "S", "turn_id": "t1"},
        {"event": "compaction_complete", "ts": "2026-01-02"},
    ])
    store = MemoryStore(":memory:")
    try:
        r = ingest_jsonl(store, p)
        assert r.lines_read == 3
        assert r.turns_ingested == 1
        assert r.skipped_lines == 2
    finally:
        store.close()


def test_malformed_lines_counted_and_skipped(tmp_path: Path) -> None:
    p = tmp_path / "turns.jsonl"
    p.write_text(
        "{not json\n"
        "[]\n"
        '{"role": "user", "text": "ok.", "session_id": "S", "turn_id": "t1"}\n'
        "\n",
        encoding="utf-8",
    )
    store = MemoryStore(":memory:")
    try:
        r = ingest_jsonl(store, p)
        assert r.turns_ingested == 1
        assert r.skipped_lines == 3
    finally:
        store.close()


def test_anchor_text_truncated_at_cap(tmp_path: Path) -> None:
    from aelfrice.models import ANCHOR_TEXT_MAX_LEN

    overlong = "x" * (ANCHOR_TEXT_MAX_LEN + 200)
    p = tmp_path / "turns.jsonl"
    _write_jsonl(p, [
        {"role": "user", "text": overlong + ".", "session_id": "S",
         "turn_id": "t1"},
        {"role": "assistant", "text": "Reply.", "session_id": "S",
         "turn_id": "t2"},
    ])
    store = MemoryStore(":memory:")
    try:
        ingest_jsonl(store, p)
        edges = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
            "SELECT anchor_text FROM edges WHERE type = ?",
            (EDGE_DERIVED_FROM,),
        ).fetchall()
        for e in edges:
            assert len(e["anchor_text"]) <= ANCHOR_TEXT_MAX_LEN
    finally:
        store.close()


def test_source_label_passed_through(tmp_path: Path) -> None:
    p = tmp_path / "turns.jsonl"
    _write_jsonl(p, [
        {"role": "user", "text": "Pi equals 3.14.", "session_id": "S",
         "turn_id": "t1"},
    ])
    store = MemoryStore(":memory:")
    try:
        # Different source labels produce distinct belief ids -> two writes.
        r1 = ingest_jsonl(store, p, source_label="conv-A")
        r2 = ingest_jsonl(store, p, source_label="conv-B")
        assert r1.beliefs_inserted >= 1
        assert r2.beliefs_inserted >= 1
    finally:
        store.close()
