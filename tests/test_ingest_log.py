"""Tests for the v2.0 #205 ingest_log table + ULID generator + record_ingest API.

Each test states a falsifiable hypothesis in its docstring and asserts
the property that would falsify it. This is the v2.0 first-slice
parallel-write phase: no view-flip yet, so tests target the LOG side
only — entry-point integration tests follow in subsequent commits.
"""
from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import pytest

from aelfrice.models import (
    INGEST_SOURCE_CLI_REMEMBER,
    INGEST_SOURCE_FILESYSTEM,
    INGEST_SOURCE_GIT,
    INGEST_SOURCE_LEGACY_UNKNOWN,
    INGEST_SOURCE_MCP_REMEMBER,
)
from aelfrice.store import MemoryStore
from aelfrice.ulid import _ULID_LEN, make_generator, ulid


# Crockford base32, lowercase i/l/o/u removed.
_ULID_RE = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")


@pytest.fixture
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    s = MemoryStore(str(tmp_path / "ingest_log.db"))
    yield s
    s.close()


# --- ULID generator -----------------------------------------------------


def test_ulid_returns_26_char_crockford_base32() -> None:
    """Hypothesis: every ulid() return value is a 26-char Crockford-base32
    string. Falsifiable by any output that doesn't match the format."""
    for _ in range(50):
        v = ulid()
        assert len(v) == _ULID_LEN
        assert _ULID_RE.match(v), f"not Crockford base32: {v!r}"


def test_ulid_strictly_monotone_within_process() -> None:
    """Hypothesis: within one process, ulid() output is strictly
    increasing under lexicographic order even across same-millisecond
    bursts. Falsifiable by any pair where ids[i] >= ids[i+1]."""
    ids = [ulid() for _ in range(2000)]
    for a, b in zip(ids, ids[1:]):
        assert a < b, f"non-monotone: {a!r} >= {b!r}"


def test_ulid_seeded_generator_is_deterministic() -> None:
    """Hypothesis: a seeded make_generator() with deterministic time
    and rand sources produces a fixed output sequence. Falsifiable if
    two generators with identical seeds disagree on any element."""
    def fixed_time() -> float:
        return 1_700_000_000.0  # fixed second

    def fixed_rand_factory():
        # Counter so each call returns a distinct deterministic 10-byte seq.
        n = [0]

        def f(k: int) -> bytes:
            n[0] += 1
            return n[0].to_bytes(k, "big")
        return f

    g1 = make_generator(rand_source=fixed_rand_factory(), time_source=fixed_time)
    g2 = make_generator(rand_source=fixed_rand_factory(), time_source=fixed_time)
    seq1 = [g1() for _ in range(20)]
    seq2 = [g2() for _ in range(20)]
    assert seq1 == seq2


# --- Schema migration ---------------------------------------------------


def test_ingest_log_table_exists_on_fresh_store(store: MemoryStore) -> None:
    """Hypothesis: opening a fresh store creates the `ingest_log` table.
    Falsifiable if `sqlite_master` lacks the row."""
    cur = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "SELECT name FROM sqlite_master WHERE type='table' AND name='ingest_log'"
    )
    assert cur.fetchone() is not None


def test_ingest_log_indexes_exist(store: MemoryStore) -> None:
    """Hypothesis: spec-required `(source_kind, source_path)` index
    plus the `session_id` index exist on a fresh store. Falsifiable
    by any missing index."""
    rows = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "SELECT name FROM sqlite_master WHERE type='index' "
        "AND tbl_name='ingest_log'"
    ).fetchall()
    names = {r["name"] for r in rows}
    assert "idx_ingest_log_source" in names
    assert "idx_ingest_log_session" in names


def test_ingest_log_migration_idempotent(tmp_path: Path) -> None:
    """Hypothesis: closing and reopening the store does not error on
    the additive ingest_log migration. Falsifiable if the second open
    raises."""
    db = tmp_path / "i.db"
    s1 = MemoryStore(str(db))
    s1.close()
    s2 = MemoryStore(str(db))
    s2.close()


# --- record_ingest API --------------------------------------------------


def test_record_ingest_returns_valid_ulid(store: MemoryStore) -> None:
    """Hypothesis: record_ingest() returns a 26-char Crockford base32
    string. Falsifiable by any other shape."""
    log_id = store.record_ingest(
        source_kind=INGEST_SOURCE_CLI_REMEMBER,
        raw_text="user remembered something",
    )
    assert _ULID_RE.match(log_id), log_id


def test_record_ingest_persists_all_fields(store: MemoryStore) -> None:
    """Hypothesis: every field passed to record_ingest is round-tripped
    via get_ingest_log_entry, including JSON-encoded raw_meta and
    derived ids. Falsifiable by any field that drops or mutates."""
    log_id = store.record_ingest(
        source_kind=INGEST_SOURCE_FILESYSTEM,
        raw_text="some sentence from a file",
        source_path="/tmp/x/file.md",
        raw_meta={"line": 42, "node": "Heading"},
        derived_belief_ids=["b1", "b2"],
        derived_edge_ids=[("b1", "b2", "SUPPORTS")],
        classifier_version="v1.0",
        rule_set_hash="abcd1234",
        session_id="sess-1",
        ts="2026-04-28T00:00:00+00:00",
    )
    entry = store.get_ingest_log_entry(log_id)
    assert entry is not None
    assert entry["source_kind"] == INGEST_SOURCE_FILESYSTEM
    assert entry["raw_text"] == "some sentence from a file"
    assert entry["source_path"] == "/tmp/x/file.md"
    assert entry["raw_meta"] == {"line": 42, "node": "Heading"}
    assert entry["derived_belief_ids"] == ["b1", "b2"]
    # JSON arrays of tuples come back as lists of lists.
    assert entry["derived_edge_ids"] == [["b1", "b2", "SUPPORTS"]]
    assert entry["classifier_version"] == "v1.0"
    assert entry["rule_set_hash"] == "abcd1234"
    assert entry["session_id"] == "sess-1"
    assert entry["ts"] == "2026-04-28T00:00:00+00:00"


def test_record_ingest_optional_fields_default_to_none(
    store: MemoryStore,
) -> None:
    """Hypothesis: only source_kind and raw_text are required; everything
    else defaults to None. Falsifiable if a missing optional field
    raises or persists a non-None value."""
    log_id = store.record_ingest(
        source_kind=INGEST_SOURCE_GIT,
        raw_text="commit message body",
    )
    entry = store.get_ingest_log_entry(log_id)
    assert entry is not None
    for field in (
        "source_path", "raw_meta", "derived_belief_ids",
        "derived_edge_ids", "classifier_version", "rule_set_hash",
        "session_id",
    ):
        assert entry[field] is None, f"{field} should be None, got {entry[field]!r}"


def test_record_ingest_rejects_unknown_source_kind(store: MemoryStore) -> None:
    """Hypothesis: a source_kind outside INGEST_SOURCE_KINDS raises
    ValueError before any write. Falsifiable if the call succeeds."""
    with pytest.raises(ValueError, match="Unknown source_kind"):
        store.record_ingest(source_kind="not_a_real_kind", raw_text="x")


def test_record_ingest_accepts_legacy_unknown(store: MemoryStore) -> None:
    """Hypothesis: legacy_unknown is a valid source_kind for migration
    rows. Falsifiable if record_ingest rejects it."""
    log_id = store.record_ingest(
        source_kind=INGEST_SOURCE_LEGACY_UNKNOWN,
        raw_text="pre-v2.0 belief content",
    )
    assert log_id


def test_update_ingest_derived_ids_post_classification(
    store: MemoryStore,
) -> None:
    """Hypothesis: the ingest path can write a log row first, then
    UPDATE derived_belief_ids after classification produces them.
    Falsifiable if the update doesn't land or clobbers other fields."""
    log_id = store.record_ingest(
        source_kind=INGEST_SOURCE_MCP_REMEMBER,
        raw_text="some user-stated fact",
    )
    store.update_ingest_derived_ids(log_id, derived_belief_ids=["b-x"])
    entry = store.get_ingest_log_entry(log_id)
    assert entry is not None
    assert entry["derived_belief_ids"] == ["b-x"]
    # raw_text still present; derived_edge_ids still null.
    assert entry["raw_text"] == "some user-stated fact"
    assert entry["derived_edge_ids"] is None


def test_iter_ingest_log_for_belief_returns_only_matching(
    store: MemoryStore,
) -> None:
    """Hypothesis: iter_ingest_log_for_belief returns exactly the log
    rows whose derived_belief_ids contains the queried id. Falsifiable
    by any false positive or false negative."""
    a = store.record_ingest(
        source_kind=INGEST_SOURCE_FILESYSTEM,
        raw_text="row a",
        derived_belief_ids=["b-1", "b-2"],
    )
    b = store.record_ingest(
        source_kind=INGEST_SOURCE_FILESYSTEM,
        raw_text="row b",
        derived_belief_ids=["b-2"],
    )
    store.record_ingest(
        source_kind=INGEST_SOURCE_FILESYSTEM,
        raw_text="row c",
        derived_belief_ids=["b-3"],
    )
    rows_for_b1 = store.iter_ingest_log_for_belief("b-1")
    rows_for_b2 = store.iter_ingest_log_for_belief("b-2")
    assert {r["id"] for r in rows_for_b1} == {a}
    assert {r["id"] for r in rows_for_b2} == {a, b}


def test_count_ingest_log_zero_on_fresh_store(store: MemoryStore) -> None:
    """Hypothesis: a freshly-created store has zero ingest_log rows.
    Falsifiable by any non-zero count."""
    assert store.count_ingest_log() == 0


def test_stale_experimental_ingest_log_dropped_on_open(tmp_path: Path) -> None:
    """Hypothesis: an empty pre-#205 experimental `ingest_log` table
    (id INTEGER PK, raw_meta_json instead of raw_meta) is replaced by
    the canonical schema on the next MemoryStore open. Falsifiable if
    the canonical migration fails or the old shape persists."""
    import sqlite3
    db = tmp_path / "stale.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE ingest_log ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "ts TEXT NOT NULL, "
        "source_kind TEXT NOT NULL, "
        "raw_text TEXT NOT NULL, "
        "raw_meta_json TEXT)"
    )
    conn.commit()
    conn.close()
    s = MemoryStore(str(db))
    try:
        # Canonical schema present and writable.
        log_id = s.record_ingest(
            source_kind=INGEST_SOURCE_FILESYSTEM, raw_text="post-drop write",
        )
        assert log_id  # ULID, not INTEGER
    finally:
        s.close()


def test_nonempty_stale_ingest_log_is_left_alone(tmp_path: Path) -> None:
    """Hypothesis: if the stale table is non-empty, the bootstrap
    refuses to drop it (data preservation > migration). Falsifiable
    if stale rows disappear after open."""
    import sqlite3
    db = tmp_path / "stale_nonempty.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE ingest_log ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "ts TEXT NOT NULL, "
        "source_kind TEXT NOT NULL, "
        "raw_text TEXT NOT NULL, "
        "raw_meta_json TEXT)"
    )
    conn.execute(
        "INSERT INTO ingest_log (ts, source_kind, raw_text) VALUES (?, ?, ?)",
        ("2026-01-01T00:00:00Z", "filesystem", "stale row"),
    )
    conn.commit()
    conn.close()
    # Open should error rather than silently destroy the stale row.
    with pytest.raises(Exception):
        MemoryStore(str(db))


def test_record_ingest_stamps_version_vector(store: MemoryStore) -> None:
    """Hypothesis: every record_ingest call writes one row to log_versions
    keyed (log_id, local_scope_id) with counter == 1. Mirrors #204
    behavior on beliefs/edges. Falsifiable if the VV is missing or
    counter != 1 for a single-write log row."""
    log_id = store.record_ingest(
        source_kind=INGEST_SOURCE_FILESYSTEM, raw_text="x",
    )
    vv = store.get_log_version_vector(log_id)
    assert len(vv) == 1
    assert list(vv.values()) == [1]


def test_log_version_backfill_stamps_existing_rows(tmp_path: Path) -> None:
    """Hypothesis: opening a store that already contains ingest_log rows
    without log_versions entries triggers a one-shot backfill that
    stamps `{local_scope: 1}` on each row. Falsifiable if any
    pre-existing log row remains without a version vector after open."""
    import sqlite3
    db = tmp_path / "backfill.db"
    s = MemoryStore(str(db))
    log_id = s.record_ingest(
        source_kind=INGEST_SOURCE_FILESYSTEM, raw_text="x",
    )
    # Manually nuke the version row + the backfill marker to simulate
    # a store that pre-dates the v2.0 backfill.
    s._conn.execute("DELETE FROM log_versions WHERE log_id = ?", (log_id,))  # pyright: ignore[reportPrivateUsage]
    s._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "DELETE FROM schema_meta WHERE key = 'log_version_vector_backfill_complete'"
    )
    s._conn.commit()  # pyright: ignore[reportPrivateUsage]
    s.close()
    # Re-open should backfill.
    s2 = MemoryStore(str(db))
    try:
        vv = s2.get_log_version_vector(log_id)
        assert len(vv) == 1
        assert list(vv.values()) == [1]
    finally:
        s2.close()


def test_count_ingest_log_increments_per_record(store: MemoryStore) -> None:
    """Hypothesis: each successful record_ingest call adds exactly one
    row. Falsifiable by any count drift."""
    for i in range(5):
        store.record_ingest(
            source_kind=INGEST_SOURCE_CLI_REMEMBER, raw_text=f"row {i}",
        )
    assert store.count_ingest_log() == 5


# --- Entry-point parallel-write contract --------------------------------


def test_ingest_turn_writes_log_row_per_new_belief(
    store: MemoryStore,
) -> None:
    """Hypothesis: every belief inserted by ingest_turn has at least
    one ingest_log row that references its id in derived_belief_ids.
    Falsifiable by any orphan belief (the spec's reachability check)."""
    from aelfrice.ingest import ingest_turn
    text = (
        "The configuration file lives at /etc/aelfrice/conf. "
        "The default port is 8080 for the dashboard."
    )
    n = ingest_turn(store, text, source="user", session_id="s-1")
    assert n == 2
    for belief_id in store.list_belief_ids():
        rows = store.iter_ingest_log_for_belief(belief_id)
        assert rows, f"belief {belief_id} has no log row"
        # Spec: source_kind + raw_text + session_id stamped.
        assert all(r["source_kind"] == "filesystem" for r in rows)
        assert all(r["session_id"] == "s-1" for r in rows)


def test_ingest_turn_dedup_does_not_double_log(store: MemoryStore) -> None:
    """Hypothesis: re-ingesting the same (source, sentence) skips the
    belief insert AND skips the log row (dedup is idempotent on both).
    Falsifiable if log count grows on the no-op second call."""
    from aelfrice.ingest import ingest_turn
    text = "The default port is 8080 for the dashboard service."
    ingest_turn(store, text, source="user")
    log_after_first = store.count_ingest_log()
    ingest_turn(store, text, source="user")
    log_after_second = store.count_ingest_log()
    assert log_after_first == log_after_second


def test_scan_repo_writes_log_row_per_new_belief(
    tmp_path: Path,
) -> None:
    """Hypothesis: every belief inserted by scan_repo has at least one
    ingest_log row referencing its id. Falsifiable by any orphan belief."""
    from aelfrice.scanner import scan_repo
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text(
        "This project must use uv for environment management.\n\n"
        "We always prefer atomic commits over batched commits.\n"
    )
    s = MemoryStore(":memory:")
    try:
        scan_repo(s, repo, now="2026-04-28T00:00:00Z")
        ids = s.list_belief_ids()
        assert ids, "expected scan_repo to insert beliefs"
        for belief_id in ids:
            rows = s.iter_ingest_log_for_belief(belief_id)
            assert rows, f"belief {belief_id} has no log row"
            assert all(r["source_kind"] == "filesystem" for r in rows)
    finally:
        s.close()
