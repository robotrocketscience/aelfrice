"""#1650 part 1: a corroboration records who spoke.

Automatic phantom promotion may count only support that came from the
user (docs/design/feature-supports-writer.md). A transcript corroboration
row now keeps its speaker; every other source, and every row written
before the column existed, leaves it NULL, which never counts as the
user's. The column is not part of the #1020 unique key, so a second write
for the same source upgrades the row to the user's when either write came
from a user turn.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from aelfrice.ingest import ingest_jsonl
from aelfrice.models import (
    BELIEF_FACTUAL,
    CORROBORATION_SOURCE_COMMIT_INGEST,
    CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
    LOCK_NONE,
    Belief,
)
from aelfrice.store import MemoryStore


def _belief(bid: str = "b1", content: str = "The build uses uv.") -> Belief:
    return Belief(
        id=bid, content=content, content_hash=f"h_{bid}", alpha=1.0, beta=1.0,
        type=BELIEF_FACTUAL, lock_level=LOCK_NONE, locked_at=None,
        created_at="2026-10-01T00:00:00Z", last_retrieved_at=None,
    )


def _speakers(store: MemoryStore) -> list[str | None]:
    return [r[0] for r in store._conn.execute(  # noqa: SLF001
        "SELECT speaker FROM belief_corroborations ORDER BY id").fetchall()]


def _columns(store: MemoryStore) -> set[str]:
    return {str(r["name"]) for r in store._conn.execute(  # noqa: SLF001
        "PRAGMA table_info(belief_corroborations)").fetchall()}


def test_a_fresh_store_has_the_column(tmp_path: Path) -> None:
    s = MemoryStore(str(tmp_path / "m.db"))
    try:
        assert "speaker" in _columns(s)
    finally:
        s.close()


def test_an_older_store_gains_the_column_with_null_rows(tmp_path: Path) -> None:
    db = tmp_path / "m.db"
    s = MemoryStore(str(db))
    s.insert_belief(_belief())
    s.record_corroboration("b1", source_type=CORROBORATION_SOURCE_COMMIT_INGEST,
                           session_id="s1")
    s.close()
    con = sqlite3.connect(db)
    con.execute("ALTER TABLE belief_corroborations DROP COLUMN speaker")
    con.commit()
    con.close()
    s = MemoryStore(str(db))
    try:
        assert "speaker" in _columns(s)
        assert _speakers(s) == [None]
    finally:
        s.close()


def test_the_legacy_retype_keeps_the_column(tmp_path: Path) -> None:
    # The #762 retype rebuilds this table on a legacy store, after the ALTER
    # loop has run. Its DDL must carry the column or the rebuild drops it.
    db = tmp_path / "m.db"
    con = sqlite3.connect(db)
    con.executescript("""
        CREATE TABLE beliefs (
            id TEXT PRIMARY KEY, content TEXT NOT NULL, content_hash TEXT NOT NULL,
            alpha REAL NOT NULL, beta REAL NOT NULL, type TEXT NOT NULL,
            lock_level TEXT NOT NULL, locked_at TEXT,
            demotion_pressure INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL, last_retrieved_at TEXT, session_id TEXT,
            origin TEXT NOT NULL DEFAULT 'unknown');
        CREATE VIRTUAL TABLE beliefs_fts
            USING fts5(id UNINDEXED, content, tokenize='porter unicode61');
        CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE belief_corroborations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            belief_id INTEGER NOT NULL REFERENCES beliefs(id) ON DELETE CASCADE,
            ingested_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            source_type TEXT NOT NULL, session_id TEXT, source_path_hash TEXT);
    """)
    con.commit()
    con.close()
    s = MemoryStore(str(db))
    try:
        assert "speaker" in _columns(s)
        s.insert_belief(_belief())
        s.record_corroboration("b1", source_type=CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
                               session_id="s1", speaker="user")
        assert _speakers(s) == ["user"]
    finally:
        s.close()


def test_an_unknown_speaker_is_rejected(tmp_path: Path) -> None:
    s = MemoryStore(str(tmp_path / "m.db"))
    try:
        s.insert_belief(_belief())
        with pytest.raises(ValueError, match="speaker"):
            s.record_corroboration("b1", source_type=CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
                                   session_id="s1", speaker="system")
    finally:
        s.close()


@pytest.mark.parametrize(("first", "second", "expected"), [
    ("assistant", "user", "user"),
    ("user", "assistant", "user"),
    (None, "user", "user"),
    ("assistant", None, "assistant"),
])
def test_a_user_turn_wins_on_the_same_source(
    tmp_path: Path, first: str | None, second: str | None, expected: str,
) -> None:
    s = MemoryStore(str(tmp_path / "m.db"))
    try:
        s.insert_belief(_belief())
        for speaker in (first, second):
            s.record_corroboration("b1", source_type=CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
                                   session_id="s1", source_path_hash="p", speaker=speaker)
        assert _speakers(s) == [expected]
    finally:
        s.close()


def test_the_upgrade_touches_only_the_same_source(tmp_path: Path) -> None:
    s = MemoryStore(str(tmp_path / "m.db"))
    try:
        s.insert_belief(_belief())
        s.record_corroboration("b1", source_type=CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
                               session_id="s1", source_path_hash="p", speaker="assistant")
        s.record_corroboration("b1", source_type=CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
                               session_id="s2", source_path_hash="p", speaker="user")
        assert sorted(_speakers(s), key=str) == ["assistant", "user"]
    finally:
        s.close()


def _turn(session: str, text: str, ts: str) -> dict[str, object]:
    return {"schema_version": 1, "role": "user", "session_id": session, "ts": ts, "text": text}


def test_transcript_ingest_records_the_users_speaker(tmp_path: Path) -> None:
    log = tmp_path / "turns.jsonl"
    text = "The release checks must include pyright before every tag."
    log.write_text("\n".join(json.dumps(r) for r in [
        _turn("s1", text, "2026-10-01T10:00:00Z"),
        _turn("s2", text, "2026-10-02T10:00:00Z"),
    ]) + "\n", encoding="utf-8")
    s = MemoryStore(str(tmp_path / "m.db"))
    try:
        ingest_jsonl(s, log)
        rows = s._conn.execute(  # noqa: SLF001
            "SELECT source_type, speaker FROM belief_corroborations").fetchall()
        assert rows, "the repeated sentence did not corroborate"
        assert {(r[0], r[1]) for r in rows} == {(CORROBORATION_SOURCE_TRANSCRIPT_INGEST, "user")}
    finally:
        s.close()


def test_a_non_transcript_source_leaves_the_speaker_null(tmp_path: Path) -> None:
    s = MemoryStore(str(tmp_path / "m.db"))
    try:
        b = _belief()
        s.insert_or_corroborate(b, source_type=CORROBORATION_SOURCE_COMMIT_INGEST, session_id="s1")
        s.insert_or_corroborate(b, source_type=CORROBORATION_SOURCE_COMMIT_INGEST, session_id="s2")
        assert _speakers(s) == [None]
    finally:
        s.close()
