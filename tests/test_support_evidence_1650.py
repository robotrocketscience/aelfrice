"""#1650 part 5: the `belief_support_evidence` side table.

A SUPPORTS matcher records each match here, keyed (supporter_id,
target_id, kind). A match that was only logged has applied=0. The table
is in `_SCHEMA`, so a fresh store has it and an older store gains it on
its next open. The store methods write no edge and move no posterior.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    SUPPORT_EVIDENCE_KIND_REWORDED,
    Belief,
    SupportEvidence,
)
from aelfrice.store import (
    SCHEMA_META_CONTENT_HASH_DEDUP_COMPLETE,
    SCHEMA_META_CONTENT_HASH_UNIQUE_APPLIED,
    SCHEMA_META_STORE_GENERATION,
    MemoryStore,
)

_TABLE = "belief_support_evidence"
_INDEX = "idx_belief_support_evidence_target"
_COLUMNS = {
    "supporter_id", "target_id", "kind", "target_sentence_index",
    "supporter_session_id", "supporter_ingest_log_id", "matched_span",
    "applied", "created_at",
}


def _belief(bid: str) -> Belief:
    return Belief(
        id=bid, content=f"The build step {bid} runs under uv.",
        content_hash=f"h_{bid}", alpha=1.0, beta=1.0, type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE, locked_at=None,
        created_at="2026-10-01T00:00:00Z", last_retrieved_at=None,
    )


def _record(
    s: MemoryStore, supporter: str = "u1", target: str = "p1", *,
    index: int = 0, span: str = "build step runs under uv",
    applied: bool = False, ts: str = "2026-10-02T00:00:00+00:00",
) -> bool:
    return s.record_support_evidence(
        supporter, target, kind=SUPPORT_EVIDENCE_KIND_REWORDED,
        target_sentence_index=index, supporter_session_id="s2",
        supporter_ingest_log_id="log1", matched_span=span,
        applied=applied, ts=ts,
    )


def _store(path: Path, *ids: str) -> MemoryStore:
    s = MemoryStore(str(path))
    for bid in ids:
        s.insert_belief(_belief(bid))
    return s


def _ddl(s: MemoryStore) -> list[tuple[str, str]]:
    return [(str(r[0]), str(r[1])) for r in s._conn.execute(  # noqa: SLF001
        "SELECT name, sql FROM sqlite_master "
        "WHERE tbl_name = ? AND sql IS NOT NULL ORDER BY name", (_TABLE,))]


def _columns(s: MemoryStore) -> dict[str, int]:
    """Column name -> its position in the primary key (0 = not in it)."""
    return {str(r["name"]): int(r["pk"]) for r in s._conn.execute(  # noqa: SLF001
        f"PRAGMA table_info({_TABLE})")}


def _rows(s: MemoryStore) -> list[tuple[object, ...]]:
    return [tuple(r) for r in s._conn.execute(  # noqa: SLF001
        f"SELECT * FROM {_TABLE} ORDER BY supporter_id, target_id, kind")]


def _drop_content_hash_unique(con: sqlite3.Connection) -> None:
    """Rebuild `beliefs` without UNIQUE(content_hash), as before #219.

    Clears the #219 marker so the next open runs the swap again. Run on
    a raw connection: foreign keys are off there, so nothing cascades.
    """
    (ddl,) = con.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='beliefs'"
    ).fetchone()
    assert ddl.count("UNIQUE") == 1
    con.executescript(
        ddl.replace("UNIQUE", "").replace(
            "CREATE TABLE beliefs", "CREATE TABLE beliefs_legacy", 1)
        + ";\nINSERT INTO beliefs_legacy SELECT * FROM beliefs;"
        "\nDROP TABLE beliefs;"
        "\nALTER TABLE beliefs_legacy RENAME TO beliefs;"
    )
    con.execute("DELETE FROM schema_meta WHERE key = ?",
                (SCHEMA_META_CONTENT_HASH_UNIQUE_APPLIED,))


# --- Schema and migration ------------------------------------------------


def test_a_fresh_store_has_the_table_key_and_target_index(tmp_path: Path) -> None:
    s = MemoryStore(str(tmp_path / "m.db"))
    try:
        cols = _columns(s)
        assert set(cols) == _COLUMNS
        assert {c: p for c, p in cols.items() if p} == {
            "supporter_id": 1, "target_id": 2, "kind": 3}
        assert [name for name, _ in _ddl(s)] == [_TABLE, _INDEX]
    finally:
        s.close()


def test_a_store_without_the_table_gains_it_on_open(tmp_path: Path) -> None:
    # github/main's schema is this schema minus the table and its index.
    db = tmp_path / "m.db"
    s = _store(db, "u1", "p1")
    s.close()
    con = sqlite3.connect(db)
    con.execute(f"DROP TABLE {_TABLE}")
    con.commit()
    assert con.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE tbl_name = ?", (_TABLE,)
    ).fetchone() == (0,)
    con.close()
    s = MemoryStore(str(db))
    try:
        assert set(_columns(s)) == _COLUMNS
        assert [name for name, _ in _ddl(s)] == [_TABLE, _INDEX]
        assert s.get_belief("p1") is not None
        assert _record(s) is True
    finally:
        s.close()


def test_a_legacy_store_gains_the_table(tmp_path: Path) -> None:
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
    """)
    con.commit()
    con.close()
    s = MemoryStore(str(db))
    try:
        assert set(_columns(s)) == _COLUMNS
        assert [name for name, _ in _ddl(s)] == [_TABLE, _INDEX]
    finally:
        s.close()


def test_opening_twice_changes_neither_the_ddl_nor_the_rows(tmp_path: Path) -> None:
    db = tmp_path / "m.db"
    s = _store(db, "u1", "p1")
    _record(s)
    ddl, rows = _ddl(s), _rows(s)
    s._apply_open_schema()  # noqa: SLF001 - the open-time pass, run again
    assert (_ddl(s), _rows(s)) == (ddl, rows)
    s.close()
    s = MemoryStore(str(db))
    try:
        assert (_ddl(s), _rows(s)) == (ddl, rows)
    finally:
        s.close()


def test_rows_survive_the_content_hash_table_swap(tmp_path: Path) -> None:
    # The #219 swap rebuilds `beliefs` on a store that predates
    # UNIQUE(content_hash). DROP TABLE beliefs with foreign keys on would
    # cascade into this table and delete every row.
    db = tmp_path / "m.db"
    s = _store(db, "u1", "p1")
    _record(s)
    rows = _rows(s)
    s.close()
    con = sqlite3.connect(db)
    _drop_content_hash_unique(con)
    con.commit()
    con.close()
    s = MemoryStore(str(db))
    try:
        assert s.get_schema_meta(SCHEMA_META_CONTENT_HASH_UNIQUE_APPLIED)
        swapped = s._conn.execute(  # noqa: SLF001
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='beliefs'"
        ).fetchone()[0]
        assert "UNIQUE" in swapped
        assert _rows(s) == rows
    finally:
        s.close()


def test_the_content_hash_dedup_moves_rows_to_the_canonical_belief(
    tmp_path: Path,
) -> None:
    # The #219 dedup pass deletes a duplicate belief with foreign keys
    # on, which cascades into this table. Rows on the duplicate move to
    # the canonical belief first, as edges do. A row that would collide
    # with a canonical row on either side is dropped, and so is a match
    # inside the group in either direction.
    db = tmp_path / "m.db"
    s = _store(db, "u1", "u2", "u3", "p1", "p1dup")
    _record(s, "u1", "p1dup", span="moved target")
    _record(s, "p1dup", "u3", span="moved supporter")
    _record(s, "u2", "p1", span="kept target", ts="2026-10-02T00:00:00+00:00")
    _record(s, "u2", "p1dup", span="collides", ts="2026-10-05T00:00:00+00:00")
    _record(s, "p1", "u2", span="kept supporter")
    _record(s, "p1dup", "u2", span="collides supporter")
    _record(s, "p1dup", "p1", span="intra-group")
    _record(s, "p1", "p1dup", span="intra-group reversed")
    _record(s, "u1", "u2", span="untouched")
    s.close()
    con = sqlite3.connect(db)
    _drop_content_hash_unique(con)
    con.execute(
        "UPDATE beliefs SET content_hash = 'h_p1', "
        "created_at = '2026-10-03T00:00:00Z' WHERE id = 'p1dup'")
    con.execute("DELETE FROM schema_meta WHERE key = ?",
                (SCHEMA_META_CONTENT_HASH_DEDUP_COMPLETE,))
    con.commit()
    con.close()
    s = MemoryStore(str(db))
    try:
        assert s.get_schema_meta(SCHEMA_META_CONTENT_HASH_DEDUP_COMPLETE)
        assert s.get_belief("p1dup") is None
        assert [(r[0], r[1], r[6]) for r in _rows(s)] == [
            ("p1", "u2", "kept supporter"),
            ("p1", "u3", "moved supporter"),
            ("u1", "p1", "moved target"),
            ("u1", "u2", "untouched"),
            ("u2", "p1", "kept target"),
        ]
    finally:
        s.close()


def test_the_target_lookup_uses_the_index(tmp_path: Path) -> None:
    s = MemoryStore(str(tmp_path / "m.db"))
    try:
        plan = " | ".join(str(r["detail"]) for r in s._conn.execute(  # noqa: SLF001
            f"EXPLAIN QUERY PLAN SELECT * FROM {_TABLE} WHERE target_id = ?",
            ("p1",)))
        assert _INDEX in plan, plan
    finally:
        s.close()


# --- Store methods -------------------------------------------------------


def test_a_repeat_on_the_key_is_ignored(tmp_path: Path) -> None:
    s = _store(tmp_path / "m.db", "u1", "p1")
    try:
        assert _record(s) is True
        assert _record(s, index=3, span="another span", applied=True,
                       ts="2026-10-09T00:00:00+00:00") is False
        assert s.list_support_evidence("p1") == [SupportEvidence(
            supporter_id="u1", target_id="p1",
            kind=SUPPORT_EVIDENCE_KIND_REWORDED, target_sentence_index=0,
            supporter_session_id="s2", supporter_ingest_log_id="log1",
            matched_span="build step runs under uv", applied=False,
            created_at="2026-10-02T00:00:00+00:00",
        )]
        assert s.count_support_evidence(applied=False) == 1
        assert s.count_support_evidence(applied=True) == 0
    finally:
        s.close()


def test_rows_differing_in_one_key_column_are_kept(tmp_path: Path) -> None:
    s = _store(tmp_path / "m.db", "u1", "u2", "p1")
    try:
        assert _record(s, "u1", "p1") is True
        assert _record(s, "u2", "p1") is True
        assert _record(s, "u1", "u2") is True
        assert len(_rows(s)) == 3
    finally:
        s.close()


def test_list_is_per_target_in_a_total_order(tmp_path: Path) -> None:
    s = _store(tmp_path / "m.db", "u1", "u2", "u3", "p1", "p2")
    try:
        _record(s, "u3", "p1", ts="2026-10-02T00:00:00+00:00")
        _record(s, "u2", "p1", ts="2026-10-03T00:00:00+00:00")
        _record(s, "u1", "p1", ts="2026-10-02T00:00:00+00:00")
        _record(s, "u1", "p2", ts="2026-10-01T00:00:00+00:00")
        assert [e.supporter_id for e in s.list_support_evidence("p1")] == [
            "u1", "u3", "u2"]
        assert [e.target_id for e in s.list_support_evidence("p2")] == ["p2"]
        assert s.list_support_evidence("u1") == []
    finally:
        s.close()


def test_count_splits_on_the_applied_flag(tmp_path: Path) -> None:
    s = _store(tmp_path / "m.db", "u1", "u2", "u3", "p1")
    try:
        _record(s, "u1", applied=False)
        _record(s, "u2", applied=True)
        _record(s, "u3", applied=False)
        assert s.count_support_evidence(applied=False) == 2
        assert s.count_support_evidence(applied=True) == 1
        assert [e.applied for e in s.list_support_evidence("p1")] == [
            False, True, False]
    finally:
        s.close()


def test_recording_writes_no_edge_posterior_or_generation(tmp_path: Path) -> None:
    s = _store(tmp_path / "m.db", "u1", "p1")
    try:
        generation = s.get_schema_meta(SCHEMA_META_STORE_GENERATION)
        _record(s, applied=True)
        assert s._conn.execute(  # noqa: SLF001
            "SELECT COUNT(*) FROM edges").fetchone()[0] == 0
        for bid in ("u1", "p1"):
            b = s.get_belief(bid)
            assert b is not None and (b.alpha, b.beta) == (1.0, 1.0)
        assert s._conn.execute(  # noqa: SLF001
            "SELECT COUNT(*) FROM feedback_history").fetchone()[0] == 0
        assert s.get_schema_meta(SCHEMA_META_STORE_GENERATION) == generation
    finally:
        s.close()


def test_a_write_without_ts_stamps_the_clock(tmp_path: Path) -> None:
    s = _store(tmp_path / "m.db", "u1", "p1")
    try:
        assert s.record_support_evidence(
            "u1", "p1", kind=SUPPORT_EVIDENCE_KIND_REWORDED,
            target_sentence_index=0, supporter_session_id=None,
            supporter_ingest_log_id=None, matched_span="x",
        ) is True
        (row,) = s.list_support_evidence("p1")
        assert row.created_at.startswith("20")
        assert row.supporter_session_id is None
        assert row.supporter_ingest_log_id is None
    finally:
        s.close()


@pytest.mark.parametrize(("kwargs", "match"), [
    ({"kind": "praise"}, "kind"),
    ({"target_sentence_index": -1}, "target_sentence_index"),
    ({"target_id": "u1"}, "itself"),
])
def test_bad_input_is_rejected(
    tmp_path: Path, kwargs: dict[str, object], match: str,
) -> None:
    s = _store(tmp_path / "m.db", "u1", "p1")
    args: dict[str, object] = {
        "supporter_id": "u1", "target_id": "p1",
        "kind": SUPPORT_EVIDENCE_KIND_REWORDED, "target_sentence_index": 0,
        "supporter_session_id": "s2", "supporter_ingest_log_id": "log1",
        "matched_span": "x",
    }
    args.update(kwargs)
    try:
        with pytest.raises(ValueError, match=match):
            s.record_support_evidence(**args)  # type: ignore[arg-type]
        assert _rows(s) == []
    finally:
        s.close()


@pytest.mark.parametrize(("column", "value"), [
    ("target_sentence_index", -1),
    ("applied", 2),
    ("applied", -1),
])
def test_the_table_checks_refuse_a_raw_insert(
    tmp_path: Path, column: str, value: int,
) -> None:
    # The method checks its input in Python; the table's CHECK
    # constraints hold for a writer that skips the method.
    s = _store(tmp_path / "m.db", "u1", "p1")
    row: dict[str, object] = {
        "supporter_id": "u1", "target_id": "p1",
        "kind": SUPPORT_EVIDENCE_KIND_REWORDED, "target_sentence_index": 0,
        "matched_span": "x", "applied": 0,
        "created_at": "2026-10-02T00:00:00+00:00",
    }
    row[column] = value
    sql = (f"INSERT INTO {_TABLE} ({', '.join(row)}) "
           f"VALUES ({', '.join('?' * len(row))})")
    try:
        with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint"):
            s._conn.execute(sql, tuple(row.values()))  # noqa: SLF001
        s._conn.rollback()  # noqa: SLF001
        assert _rows(s) == []
        row[column] = 0
        s._conn.execute(sql, tuple(row.values()))  # noqa: SLF001
        s._conn.commit()  # noqa: SLF001
        assert len(_rows(s)) == 1
    finally:
        s.close()


@pytest.mark.parametrize("missing", ["supporter", "target"])
def test_an_unknown_belief_id_is_refused(tmp_path: Path, missing: str) -> None:
    s = _store(tmp_path / "m.db", "u1", "p1")
    try:
        with pytest.raises(sqlite3.IntegrityError):
            if missing == "supporter":
                _record(s, "nope", "p1")
            else:
                _record(s, "u1", "nope")
    finally:
        s.close()


@pytest.mark.parametrize("missing", ["supporter", "target"])
def test_a_refused_id_releases_the_write_lock(
    tmp_path: Path, missing: str,
) -> None:
    # The failed INSERT runs inside the implicit transaction the driver
    # opened for it. Left open, it holds the write lock for the life of
    # the handle, and every other writer on the store times out.
    db = tmp_path / "m.db"
    s = _store(db, "u1", "p1")
    try:
        with pytest.raises(sqlite3.IntegrityError):
            if missing == "supporter":
                _record(s, "nope", "p1")
            else:
                _record(s, "u1", "nope")
        assert s.transaction_open is False
        other = sqlite3.connect(db, timeout=0)
        try:
            other.execute(
                "INSERT INTO schema_meta (key, value) VALUES ('t1650', '1')")
            other.commit()
        finally:
            other.close()
        assert s.get_schema_meta("t1650") == "1"
    finally:
        s.close()


def test_a_refused_id_inside_a_transaction_keeps_the_outer_group(
    tmp_path: Path,
) -> None:
    s = _store(tmp_path / "m.db", "u1", "u2", "p1")
    try:
        with s.transaction():
            assert _record(s, "u1", "p1") is True
            with pytest.raises(sqlite3.IntegrityError):
                _record(s, "u2", "nope")
            assert _record(s, "u2", "p1") is True
        assert [r[0] for r in _rows(s)] == ["u1", "u2"]
    finally:
        s.close()


@pytest.mark.parametrize("deleted", ["u1", "p1"])
def test_deleting_either_belief_removes_its_rows(
    tmp_path: Path, deleted: str,
) -> None:
    s = _store(tmp_path / "m.db", "u1", "u2", "p1", "p2")
    try:
        _record(s, "u1", "p1")
        _record(s, "u2", "p2")
        s.delete_belief(deleted)
        assert [(r[0], r[1]) for r in _rows(s)] == [("u2", "p2")]
    finally:
        s.close()
