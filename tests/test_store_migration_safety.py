"""No one-shot store migration may leave the store unopenable (#1161).

The defect this module pins was the only failure mode in the store that
destroyed access to a user's whole memory corpus.

`MemoryStore.__init__` runs ten one-shot migrations, and every one stamps
its completion marker *after* doing the work. The #219 consolidation pass
rewrote foreign keys with a bare `UPDATE edges SET src = ?`, but `edges`
is keyed `PRIMARY KEY (src, dst, type)`. Whenever a duplicate belief and
its canonical row shared an edge of the same type to the same neighbour —
the expected shape, since duplicates are the same content ingested twice
and the edge builders derive edges from content — the rewrite raised
`UNIQUE constraint failed: edges.src, edges.dst, edges.type`. That
exception escaped the constructor before the marker was written, so the
next open re-ran the same pass and raised again. Measured against a
regressed real store: `aelf stats`, `aelf health` and `aelf doctor` all
died with the same IntegrityError, and there is no recovery path through
the package because every entry point opens the store.

Three defects are covered here:

1. **The edge rewrite was not collision-safe.** Both the `src` and the
   `dst` rewrite could raise. Fixed with `UPDATE OR IGNORE` plus a
   cleanup delete, and a `NOT IN (canon, dupe)` guard so an intra-group
   edge is never rewritten into a `(canon, canon)` self-loop.
2. **Parameter binding was unbounded.** `... IN ({ph})` bound one
   parameter per affected row, so a store with more duplicates than
   SQLITE_LIMIT_VARIABLE_NUMBER raised `too many SQL variables` — the
   same unopenable-store failure by a different route. Fixed by
   chunking through `_param_chunks`.
3. **Any migration failure was fatal.** Fixed by `_run_guarded_migration`
   at the `__init__` call sites, which records the failure in
   `schema_meta`, leaves the completion marker unset so a fixed build
   retries, and lets the open succeed. The migration methods themselves
   still raise when called directly.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pytest

from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    ORIGIN_UNKNOWN,
    Belief,
    Edge,
)
from aelfrice.store import (
    SCHEMA_META_CONTENT_HASH_DEDUP_COMPLETE,
    SCHEMA_META_CONTENT_HASH_UNIQUE_APPLIED,
    SCHEMA_META_MIGRATION_FAILED_PREFIX,
    MemoryStore,
    _param_chunks,
)

CANON = "a" * 16
DUPE = "d" * 16
DUPE2 = "e" * 16
OTHER = "b" * 16

# The canonical row must sort first: oldest created_at, then id ASC.
_CANON_CREATED = "2026-01-01T00:00:00+00:00"
_DUPE_CREATED = "2099-01-01T00:00:00+00:00"

_DEDUP_MARKERS = (
    SCHEMA_META_CONTENT_HASH_DEDUP_COMPLETE,
    SCHEMA_META_CONTENT_HASH_UNIQUE_APPLIED,
)


def _belief(bid: str, content: str, *, content_hash: str | None = None) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=content_hash or f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at=_CANON_CREATED,
        last_retrieved_at=None,
        origin=ORIGIN_UNKNOWN,
    )


def _drop_content_hash_unique(conn: sqlite3.Connection) -> list[str]:
    """Rebuild `beliefs` without UNIQUE(content_hash); return its columns.

    Reproduces the pre-#219 shape that the consolidation migration exists
    to bring forward. Derived from the live DDL rather than a hardcoded
    CREATE TABLE so it keeps tracking columns added by `_MIGRATIONS`.
    """
    live = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name='beliefs'"
    ).fetchone()[0]
    cols = [r[1] for r in conn.execute("PRAGMA table_info(beliefs)")]
    legacy = live.replace("TEXT NOT NULL UNIQUE", "TEXT NOT NULL", 1)
    legacy = legacy.replace(
        "CREATE TABLE beliefs", "CREATE TABLE beliefs_legacy", 1
    )
    assert "UNIQUE" not in legacy, "UNIQUE(content_hash) was not dropped"
    collist = ", ".join(cols)
    conn.executescript(
        f"PRAGMA foreign_keys=OFF; BEGIN; {legacy};"
        f" INSERT INTO beliefs_legacy ({collist}) SELECT {collist} FROM beliefs;"
        f" DROP TABLE beliefs;"
        f" ALTER TABLE beliefs_legacy RENAME TO beliefs; COMMIT;"
    )
    return cols


def _make_legacy_store(
    db: Path,
    *,
    seed=None,
    raw=None,
    n_dupes: int = 1,
) -> list[str]:
    """Build a pre-#219 store with `n_dupes` duplicates of CANON.

    `seed(store)` runs against the canonical store before the regression
    (use the public API); `raw(conn, dupe_ids)` runs after, against a
    plain connection, for rows the public API would reject. Returns the
    duplicate ids. Both #219 markers are cleared so the migration runs.
    """
    st = MemoryStore(str(db))
    st.insert_belief(_belief(CANON, "the deploy target is us-east-1"))
    st.insert_belief(_belief(OTHER, "the region has three availability zones"))
    if seed is not None:
        seed(st)
    st.close()

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    cols = _drop_content_hash_unique(conn)
    template = dict(
        conn.execute("SELECT * FROM beliefs WHERE id = ?", (CANON,)).fetchone()
    )
    dupe_ids = [DUPE, DUPE2][:n_dupes]
    for dupe_id in dupe_ids:
        row = dict(template)
        row["id"] = dupe_id
        row["created_at"] = _DUPE_CREATED
        conn.execute(
            f"INSERT INTO beliefs ({', '.join(cols)}) "
            f"VALUES ({','.join('?' * len(cols))})",
            [row[c] for c in cols],
        )
    if raw is not None:
        raw(conn, dupe_ids)
    conn.execute(
        "DELETE FROM schema_meta WHERE key IN (?, ?)", _DEDUP_MARKERS
    )
    conn.commit()
    conn.close()
    return dupe_ids


def _raw_edge(
    conn: sqlite3.Connection,
    src: str,
    dst: str,
    type_: str = "SUPPORTS",
    weight: float = 0.5,
) -> None:
    conn.execute(
        "INSERT INTO edges (src, dst, type, weight) VALUES (?, ?, ?, ?)",
        (src, dst, type_, weight),
    )


def _edges(store: MemoryStore) -> set[tuple[str, str, str]]:
    return {
        (str(r["src"]), str(r["dst"]), str(r["type"]))
        for r in store._conn.execute("SELECT src, dst, type FROM edges")
    }


# --- The headline: collisions must not brick the store ------------------


@pytest.mark.parametrize(
    "name,seed,raw",
    [
        (
            # The duplicate and the canonical row both point at OTHER
            # with the same edge type. The `src` rewrite collides.
            "src",
            lambda st: st.insert_edge(
                Edge(src=CANON, dst=OTHER, type="SUPPORTS", weight=0.8)
            ),
            lambda conn, dupes: _raw_edge(conn, dupes[0], OTHER),
        ),
        (
            # Mirror image: OTHER points at both. The `dst` rewrite
            # collides.
            "dst",
            lambda st: st.insert_edge(
                Edge(src=OTHER, dst=CANON, type="SUPPORTS", weight=0.8)
            ),
            lambda conn, dupes: _raw_edge(conn, OTHER, dupes[0]),
        ),
    ],
)
def test_a_colliding_edge_rewrite_does_not_brick_the_store(
    tmp_path: Path, name: str, seed, raw
) -> None:
    db = tmp_path / f"{name}.db"
    _make_legacy_store(db, seed=seed, raw=raw)
    # Pre-#1161 this raised sqlite3.IntegrityError on every open.
    store = MemoryStore(str(db))
    assert store.get_belief(CANON) is not None
    assert store.get_belief(DUPE) is None
    store.close()


def test_the_store_stays_openable_on_every_subsequent_open(
    tmp_path: Path,
) -> None:
    """The brick was permanent, so one successful open is not enough."""
    db = tmp_path / "repeat.db"
    _make_legacy_store(
        db,
        seed=lambda st: st.insert_edge(
            Edge(src=CANON, dst=OTHER, type="SUPPORTS", weight=0.8)
        ),
        raw=lambda conn, dupes: _raw_edge(conn, dupes[0], OTHER),
    )
    for _ in range(3):
        store = MemoryStore(str(db))
        assert store.count_beliefs() == 2
        store.close()


def test_two_duplicates_sharing_one_target_collide_with_each_other(
    tmp_path: Path,
) -> None:
    """The second duplicate collides with the first duplicate's rewrite.

    The canonical row has no edge of its own here, so the collision is
    between two rewrites within the same executemany rather than against
    a pre-existing canonical edge.
    """
    db = tmp_path / "two_dupes.db"
    _make_legacy_store(
        db,
        raw=lambda conn, dupes: [
            _raw_edge(conn, dupes[0], OTHER),
            _raw_edge(conn, dupes[1], OTHER),
        ],
        n_dupes=2,
    )
    store = MemoryStore(str(db))
    assert _edges(store) == {(CANON, OTHER, "SUPPORTS")}
    store.close()


def test_a_corroboration_collision_does_not_brick_the_store(
    tmp_path: Path,
) -> None:
    """`belief_corroborations` gains a partial UNIQUE index under #1020.

    A duplicate corroborated by the same (session, source) as the
    canonical row collides on the FK rewrite the same way edges do.
    """
    db = tmp_path / "corrob.db"

    def raw(conn: sqlite3.Connection, dupes: list[str]) -> None:
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS
                uq_belief_corroborations_source
            ON belief_corroborations(
                belief_id, COALESCE(session_id, ''),
                COALESCE(source_path_hash, ''), source_type
            )
            WHERE source_type != 'consolidation_migration'
            """
        )
        for bid in (CANON, dupes[0]):
            conn.execute(
                "INSERT INTO belief_corroborations "
                "(belief_id, ingested_at, source_type, session_id, "
                " source_path_hash) VALUES (?, ?, 'transcript', 's1', 'p1')",
                (bid, _CANON_CREATED),
            )

    _make_legacy_store(db, raw=raw)
    store = MemoryStore(str(db))
    assert store.get_belief(CANON) is not None
    remaining = store._conn.execute(
        "SELECT COUNT(*) FROM belief_corroborations WHERE belief_id = ?",
        (DUPE,),
    ).fetchone()[0]
    # The loser stayed on the duplicate and was reaped by ON DELETE CASCADE.
    assert remaining == 0
    store.close()


# --- Correctness of the rewrite, not just survival ----------------------


def test_a_noncolliding_edge_moves_to_the_canonical_row(
    tmp_path: Path,
) -> None:
    """Survival must not come from silently dropping every edge."""
    db = tmp_path / "moves.db"
    _make_legacy_store(
        db, raw=lambda conn, dupes: _raw_edge(conn, dupes[0], OTHER)
    )
    store = MemoryStore(str(db))
    assert _edges(store) == {(CANON, OTHER, "SUPPORTS")}
    store.close()


def test_distinct_edge_types_are_both_preserved(tmp_path: Path) -> None:
    """Only the primary key collides — differing `type` is not a conflict."""
    db = tmp_path / "types.db"
    _make_legacy_store(
        db,
        seed=lambda st: st.insert_edge(
            Edge(src=CANON, dst=OTHER, type="SUPPORTS", weight=0.8)
        ),
        raw=lambda conn, dupes: _raw_edge(conn, dupes[0], OTHER, "CONTRADICTS"),
    )
    store = MemoryStore(str(db))
    assert _edges(store) == {
        (CANON, OTHER, "SUPPORTS"),
        (CANON, OTHER, "CONTRADICTS"),
    }
    store.close()


def test_no_edge_survives_pointing_at_a_consumed_duplicate(
    tmp_path: Path,
) -> None:
    """`edges` has no FK to `beliefs`, so orphans are not swept for us."""
    db = tmp_path / "orphans.db"
    _make_legacy_store(
        db,
        seed=lambda st: st.insert_edge(
            Edge(src=CANON, dst=OTHER, type="SUPPORTS", weight=0.8)
        ),
        raw=lambda conn, dupes: _raw_edge(conn, dupes[0], OTHER),
    )
    store = MemoryStore(str(db))
    dangling = [
        e for e in _edges(store) if DUPE in (e[0], e[1])
    ]
    assert dangling == []
    store.close()


def test_an_intra_group_edge_does_not_become_a_self_loop(
    tmp_path: Path,
) -> None:
    """A duplicate -> canonical edge would rewrite to (canon, canon).

    Such an edge asserts a relationship between two rows that turned out
    to be the same belief, so it carries no information and must be
    dropped rather than preserved as a self-loop.
    """
    db = tmp_path / "selfloop.db"
    _make_legacy_store(
        db, raw=lambda conn, dupes: _raw_edge(conn, dupes[0], CANON)
    )
    store = MemoryStore(str(db))
    assert _edges(store) == set()
    store.close()


def test_a_preexisting_self_loop_on_the_canonical_row_is_kept(
    tmp_path: Path,
) -> None:
    """The self-loop guard must not reach edges the migration didn't make."""
    db = tmp_path / "keep_selfloop.db"
    _make_legacy_store(
        db,
        seed=lambda st: st.insert_edge(
            Edge(src=CANON, dst=CANON, type="SUPPORTS", weight=0.8)
        ),
        raw=lambda conn, dupes: _raw_edge(conn, dupes[0], OTHER),
    )
    store = MemoryStore(str(db))
    assert (CANON, CANON, "SUPPORTS") in _edges(store)
    store.close()


def test_edge_versions_follow_the_rewritten_edge(tmp_path: Path) -> None:
    """The version sidecar mirrors `edges`' composite key.

    Leaving it un-rewritten strands federation version vectors on edges
    the migration moved or deleted.
    """
    db = tmp_path / "edge_versions.db"

    def raw(conn: sqlite3.Connection, dupes: list[str]) -> None:
        _raw_edge(conn, dupes[0], OTHER)
        conn.execute(
            "INSERT INTO edge_versions (src, dst, type, scope_id, counter) "
            "VALUES (?, ?, 'SUPPORTS', 'scope-x', 3)",
            (dupes[0], OTHER),
        )

    _make_legacy_store(db, raw=raw)
    store = MemoryStore(str(db))
    rows = {
        (str(r["src"]), str(r["dst"]))
        for r in store._conn.execute("SELECT src, dst FROM edge_versions")
    }
    assert (CANON, OTHER) in rows
    assert (DUPE, OTHER) not in rows
    store.close()


def test_feedback_history_follows_the_canonical_row(tmp_path: Path) -> None:
    """The one rewrite with no uniqueness constraint must still happen."""
    db = tmp_path / "feedback.db"
    _make_legacy_store(
        db,
        raw=lambda conn, dupes: conn.execute(
            "INSERT INTO feedback_history "
            "(belief_id, valence, source, created_at) "
            "VALUES (?, 1.0, 'test', ?)",
            (dupes[0], _CANON_CREATED),
        ),
    )
    store = MemoryStore(str(db))
    owners = {
        str(r["belief_id"])
        for r in store._conn.execute("SELECT belief_id FROM feedback_history")
    }
    assert owners == {CANON}
    store.close()


def test_evidence_is_summed_onto_the_canonical_row(tmp_path: Path) -> None:
    """Guard the pre-existing merge semantics against the rewrite."""
    db = tmp_path / "evidence.db"
    _make_legacy_store(
        db, raw=lambda conn, dupes: _raw_edge(conn, dupes[0], OTHER)
    )
    store = MemoryStore(str(db))
    belief = store.get_belief(CANON)
    assert belief is not None
    # Two rows, each with the Jeffreys prior alpha=1, beta=1.
    assert (belief.alpha, belief.beta) == (2.0, 2.0)
    store.close()


# --- Parameter chunking -------------------------------------------------


def test_param_chunks_yields_nothing_for_an_empty_list() -> None:
    """An empty chunk would render `IN ()`, which is a syntax error."""
    assert list(_param_chunks([])) == []


def test_param_chunks_partitions_without_loss_or_overlap() -> None:
    ids = [f"id{i}" for i in range(7)]
    chunks = list(_param_chunks(ids, size=3))
    assert chunks == [
        ["id0", "id1", "id2"],
        ["id3", "id4", "id5"],
        ["id6"],
    ]
    assert [i for c in chunks for i in c] == ids


def test_param_chunks_default_stays_under_the_oldest_sqlite_limit() -> None:
    """999 is the SQLITE_LIMIT_VARIABLE_NUMBER on builds before 3.32.

    The cleanup delete binds each chunk twice (`src IN (...) OR dst IN
    (...)`), so the default must clear half the limit, not all of it.
    """
    from aelfrice.store import _MIGRATION_PARAM_CHUNK

    assert _MIGRATION_PARAM_CHUNK * 2 <= 999


def test_consolidation_is_correct_across_a_chunk_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Shrink the chunk instead of building 32k duplicates.

    Binding one parameter per duplicate raised `too many SQL variables`
    past the host's limit — the same unopenable-store outcome. Chunk
    size is the variable under test, so driving it to 1 exercises every
    boundary deterministically without a slow fixture.
    """
    monkeypatch.setattr("aelfrice.store._MIGRATION_PARAM_CHUNK", 1)
    db = tmp_path / "chunked.db"
    _make_legacy_store(
        db,
        raw=lambda conn, dupes: [
            _raw_edge(conn, dupes[0], OTHER),
            _raw_edge(conn, dupes[1], OTHER, "CONTRADICTS"),
        ],
        n_dupes=2,
    )
    store = MemoryStore(str(db))
    assert store.count_beliefs() == 2
    assert _edges(store) == {
        (CANON, OTHER, "SUPPORTS"),
        (CANON, OTHER, "CONTRADICTS"),
    }
    store.close()


# --- The guard: a failing migration degrades, it does not brick --------


def _break_dedup(monkeypatch: pytest.MonkeyPatch, err: Exception) -> None:
    """Replace the consolidation pass with one that raises `err`.

    The replacement keeps the original `__name__` because
    `_run_guarded_migration` keys its schema_meta row on it.
    """
    original = MemoryStore._maybe_consolidate_content_hash_duplicates

    def exploding(self: MemoryStore) -> int:
        raise err

    exploding.__name__ = original.__name__
    monkeypatch.setattr(
        MemoryStore,
        "_maybe_consolidate_content_hash_duplicates",
        exploding,
    )


def test_a_raising_migration_does_not_prevent_the_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "guarded.db"
    MemoryStore(str(db)).close()
    _break_dedup(monkeypatch, RuntimeError("synthetic failure"))
    store = MemoryStore(str(db))
    # Reads and writes still work; the store is degraded, not dead.
    store.insert_belief(_belief(CANON, "still writable"))
    assert store.count_beliefs() == 1
    store.close()


def test_a_raising_migration_leaves_its_completion_marker_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This is what makes the degradation recoverable rather than stuck."""
    db = tmp_path / "unmarked.db"
    MemoryStore(str(db)).close()
    conn = sqlite3.connect(str(db))
    conn.execute(
        "DELETE FROM schema_meta WHERE key = ?",
        (SCHEMA_META_CONTENT_HASH_DEDUP_COMPLETE,),
    )
    conn.commit()
    conn.close()
    _break_dedup(monkeypatch, RuntimeError("synthetic failure"))
    store = MemoryStore(str(db))
    assert (
        store.get_schema_meta(SCHEMA_META_CONTENT_HASH_DEDUP_COMPLETE) is None
    )
    store.close()


def test_a_raising_migration_is_recorded_and_logged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Degrading silently would be worse than bricking, not better."""
    db = tmp_path / "recorded.db"
    MemoryStore(str(db)).close()
    _break_dedup(monkeypatch, RuntimeError("synthetic failure"))
    with caplog.at_level(logging.ERROR, logger="aelfrice"):
        store = MemoryStore(str(db))
    failures = store.failed_migrations()
    assert list(failures) == ["_maybe_consolidate_content_hash_duplicates"]
    assert "synthetic failure" in failures[
        "_maybe_consolidate_content_hash_duplicates"
    ]
    assert any("synthetic failure" in r.getMessage() for r in caplog.records)
    store.close()


def test_a_repaired_build_clears_the_recorded_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "repaired.db"
    MemoryStore(str(db)).close()
    _break_dedup(monkeypatch, RuntimeError("synthetic failure"))
    store = MemoryStore(str(db))
    assert store.failed_migrations()
    store.close()
    monkeypatch.undo()
    store = MemoryStore(str(db))
    assert store.failed_migrations() == {}
    assert store.get_schema_meta(SCHEMA_META_CONTENT_HASH_DEDUP_COMPLETE)
    store.close()


def test_failed_migrations_is_empty_on_a_healthy_store(
    tmp_path: Path,
) -> None:
    store = MemoryStore(str(tmp_path / "healthy.db"))
    assert store.failed_migrations() == {}
    store.close()


def test_calling_a_migration_directly_still_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard wraps the `__init__` call sites, not the methods.

    A repair tool or a test that drives one pass on purpose must still
    see the exception, or the guard would hide real bugs.
    """
    db = tmp_path / "direct.db"
    store = MemoryStore(str(db))
    _break_dedup(monkeypatch, RuntimeError("synthetic failure"))
    with pytest.raises(RuntimeError, match="synthetic failure"):
        store._maybe_consolidate_content_hash_duplicates()
    store.close()


def test_the_unique_swap_is_skipped_while_duplicates_remain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One incomplete migration must not cascade into a second failure.

    The swap used to treat its precondition as guaranteed. Now that the
    dedup pass is allowed to fail, attempting the swap over surviving
    duplicates would raise a UNIQUE violation from inside `__init__`.
    """
    db = tmp_path / "precondition.db"
    _make_legacy_store(
        db, raw=lambda conn, dupes: _raw_edge(conn, dupes[0], OTHER)
    )
    _break_dedup(monkeypatch, RuntimeError("dedup unavailable"))
    store = MemoryStore(str(db))
    assert store.count_beliefs() == 3, "the duplicate should still be present"
    assert (
        store.get_schema_meta(SCHEMA_META_CONTENT_HASH_UNIQUE_APPLIED) is None
    )
    ddl = store._conn.execute(
        "SELECT sql FROM sqlite_master WHERE name='beliefs'"
    ).fetchone()[0]
    assert "UNIQUE" not in ddl
    # The load-bearing assertion. Without the precondition check the swap
    # attempts the table copy, raises a UNIQUE violation, and the guard
    # records a *second* failed migration on the same open — every other
    # assertion above still holds, because the guard absorbs it either
    # way. A clean skip is the difference between one incomplete
    # migration and two.
    assert list(store.failed_migrations()) == [
        "_maybe_consolidate_content_hash_duplicates"
    ]
    store.close()

    # Both passes retry once a build that can complete them arrives.
    monkeypatch.undo()
    store = MemoryStore(str(db))
    assert store.count_beliefs() == 2
    assert store.get_schema_meta(SCHEMA_META_CONTENT_HASH_UNIQUE_APPLIED)
    assert store.failed_migrations() == {}
    store.close()


# --- Doctor surfaces the degraded state --------------------------------


def test_doctor_reports_an_incomplete_migration(tmp_path: Path) -> None:
    from aelfrice.doctor import diagnose, format_report

    db = tmp_path / "reported.db"
    store = MemoryStore(str(db))
    store.set_schema_meta(
        f"{SCHEMA_META_MIGRATION_FAILED_PREFIX}_maybe_rehash_speculative_v2",
        "IntegrityError('boom')",
    )
    store.close()
    report = diagnose(
        user_settings=tmp_path / "absent-settings.json", store_path=str(db)
    )
    assert list(report.failed_store_migrations) == [
        "_maybe_rehash_speculative_v2"
    ]
    rendered = format_report(report)
    assert "store migration(s) INCOMPLETE" in rendered
    assert "_maybe_rehash_speculative_v2" in rendered
    assert "IntegrityError('boom')" in rendered


def test_doctor_is_quiet_about_migrations_on_a_healthy_store(
    tmp_path: Path,
) -> None:
    from aelfrice.doctor import diagnose, format_report

    db = tmp_path / "quiet.db"
    MemoryStore(str(db)).close()
    report = diagnose(
        user_settings=tmp_path / "absent-settings.json", store_path=str(db)
    )
    assert report.failed_store_migrations == {}
    assert "INCOMPLETE" not in format_report(report)


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda p: None, id="missing-file"),
        pytest.param(lambda p: p.write_bytes(b"not a database"), id="garbage"),
        pytest.param(lambda p: p.write_bytes(b""), id="empty"),
    ],
)
def test_reading_failed_migrations_is_fail_soft(tmp_path: Path, make) -> None:
    """A diagnostic that raises is worse than one that reports nothing."""
    from aelfrice.doctor import _read_failed_store_migrations

    db = tmp_path / "broken.db"
    make(db)
    assert _read_failed_store_migrations(str(db)) == {}


def test_reading_failed_migrations_does_not_run_migrations(
    tmp_path: Path,
) -> None:
    """Doctor must not mutate the store or pay the migration cost.

    Constructing a `MemoryStore` to read the marker would retry every
    pending one-shot pass and take the write lock.
    """
    from aelfrice.doctor import _read_failed_store_migrations

    db = tmp_path / "readonly.db"
    _make_legacy_store(
        db, raw=lambda conn, dupes: _raw_edge(conn, dupes[0], OTHER)
    )
    before = db.stat().st_mtime_ns
    assert _read_failed_store_migrations(str(db)) == {}
    assert db.stat().st_mtime_ns == before
    conn = sqlite3.connect(str(db))
    # The dedup pass has not run: the duplicate and both cleared markers
    # are exactly as the fixture left them.
    assert conn.execute("SELECT COUNT(*) FROM beliefs").fetchone()[0] == 3
    assert conn.execute(
        "SELECT COUNT(*) FROM schema_meta WHERE key IN (?, ?)", _DEDUP_MARKERS
    ).fetchone()[0] == 0
    conn.close()
