"""A retired belief keeps no lock (#1945).

Before the fix, `aelf retire --force` on a locked belief set only
`valid_to`, so the retired row kept `lock_level = 'user'`. `aelf status`
counted it, `aelf locked` hid it, and `aelf unlock` reported it missing.

These tests pin each part of the fix:

* retiring a locked row clears `lock_level`, `locked_at` and
  `lock_expires_at` and writes one `lock:retire` audit row;
* `count_locked` excludes retired rows, as `list_locked_beliefs` does;
* `promotion.unlock` reaches a retired row an older release left locked;
* a one-time open migration unlocks such rows, one audit row each;
* `aelf restore` brings a formerly locked belief back unlocked.
"""
from __future__ import annotations

import io
import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest

from aelfrice.cli import main
from aelfrice.models import (
    BELIEF_FACTUAL,
    FEEDBACK_SOURCE_LOCK_RETIRE,
    LOCK_NONE,
    LOCK_USER,
    ORIGIN_USER_STATED,
    Belief,
)
from aelfrice.promotion import unlock
from aelfrice.store import SCHEMA_META_RETIRED_LOCK_CLEANUP, MemoryStore

_BID = "aabbccddeeff1945"
_LOCKED_AT = "2026-05-05T00:00:00Z"
_EXPIRES_AT = "2099-01-01T00:00:00+00:00"
_RETIRED_AT = "2026-06-01T00:00:00+00:00"


@pytest.fixture(autouse=True)
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    p = tmp_path / "aelf.db"
    monkeypatch.setenv("AELFRICE_DB", str(p))
    return p


def _run(*argv: str) -> tuple[int, str]:
    buf = io.StringIO()
    code = main(argv=list(argv), out=buf)
    return code, buf.getvalue()


def _seed_locked(db: Path, bid: str = _BID, content: str = "synthetic") -> str:
    s = MemoryStore(str(db))
    try:
        s.insert_belief(Belief(
            id=bid,
            content=content,
            content_hash="hash_" + bid,
            alpha=1.0,
            beta=1.0,
            type=BELIEF_FACTUAL,
            lock_level=LOCK_USER,
            locked_at=_LOCKED_AT,
            created_at=_LOCKED_AT,
            last_retrieved_at=None,
            origin=ORIGIN_USER_STATED,
            lock_expires_at=_EXPIRES_AT,
        ))
    finally:
        s.close()
    return bid


def _seed_legacy_retired_lock(
    db: Path, bid: str = _BID, expires_at: str = _EXPIRES_AT,
) -> str:
    """Leave the row an older release's `retire --force` left behind.

    Seeds a locked belief, then sets `valid_to` with raw SQL (the old
    `soft_delete_belief` did exactly this and nothing more) and clears
    the migration marker, so the next open sees a pre-fix store.
    """
    _seed_locked(db, bid)
    s = MemoryStore(str(db))
    try:
        s._conn.execute(  # noqa: SLF001, reproduce the pre-fix row
            "UPDATE beliefs SET valid_to = ?, lock_expires_at = ? WHERE id = ?",
            (_RETIRED_AT, expires_at, bid),
        )
        s._conn.execute(  # noqa: SLF001
            "DELETE FROM beliefs_fts WHERE id = ?", (bid,)
        )
        s._conn.execute(  # noqa: SLF001
            "DELETE FROM schema_meta WHERE key = ?",
            (SCHEMA_META_RETIRED_LOCK_CLEANUP,),
        )
        s._conn.commit()  # noqa: SLF001
    finally:
        s.close()
    return bid


def _raw_lock_fields(db: Path, bid: str) -> tuple[str, str | None, str | None]:
    s = MemoryStore(str(db))
    try:
        row = s._conn.execute(  # noqa: SLF001
            "SELECT lock_level, locked_at, lock_expires_at FROM beliefs "
            "WHERE id = ?",
            (bid,),
        ).fetchone()
        return (row["lock_level"], row["locked_at"], row["lock_expires_at"])
    finally:
        s.close()


def _sources(db: Path, bid: str) -> list[tuple[str, float]]:
    s = MemoryStore(str(db))
    try:
        rows = s._conn.execute(  # noqa: SLF001
            "SELECT source, valence FROM feedback_history WHERE belief_id = ? "
            "ORDER BY id",
            (bid,),
        ).fetchall()
        return [(str(r["source"]), float(r["valence"])) for r in rows]
    finally:
        s.close()


def _in_fts(db: Path, bid: str) -> bool:
    s = MemoryStore(str(db))
    try:
        row = s._conn.execute(  # noqa: SLF001
            "SELECT 1 FROM beliefs_fts WHERE id = ?", (bid,)
        ).fetchone()
        return row is not None
    finally:
        s.close()


# --- retire clears the lock --------------------------------------------------


def test_retire_force_clears_lock_fields(isolated_db: Path) -> None:
    bid = _seed_locked(isolated_db)
    code, _ = _run("retire", "--force", bid)
    assert code == 0
    assert _raw_lock_fields(isolated_db, bid) == (LOCK_NONE, None, None)


def test_retire_force_writes_one_lock_retire_audit_row(isolated_db: Path) -> None:
    bid = _seed_locked(isolated_db)
    _run("retire", "--force", bid)
    assert _sources(isolated_db, bid) == [
        ("user_retired_force", -1.0),
        (FEEDBACK_SOURCE_LOCK_RETIRE, 0.0),
    ]


def test_retire_unlocked_writes_no_lock_retire_row(isolated_db: Path) -> None:
    bid = _seed_locked(isolated_db)
    s = MemoryStore(str(isolated_db))
    try:
        s._conn.execute(  # noqa: SLF001
            "UPDATE beliefs SET lock_level = ?, locked_at = NULL WHERE id = ?",
            (LOCK_NONE, bid),
        )
        s._conn.commit()  # noqa: SLF001
    finally:
        s.close()
    _run("retire", bid)
    assert _sources(isolated_db, bid) == [("user_retired", -1.0)]


def test_soft_delete_clears_lock_for_every_caller(isolated_db: Path) -> None:
    """The store primitive clears the lock, so `/aelf:review` remove and
    the other soft-delete callers cannot leave a retired lock either."""
    bid = _seed_locked(isolated_db)
    s = MemoryStore(str(isolated_db))
    try:
        s.soft_delete_belief(bid, ts=_RETIRED_AT)
        assert s.count_locked() == 0
        events = s.list_feedback_events(belief_id=bid)
        assert [(e.source, e.created_at) for e in events] == [
            (FEEDBACK_SOURCE_LOCK_RETIRE, _RETIRED_AT)
        ]
    finally:
        s.close()
    assert _raw_lock_fields(isolated_db, bid) == (LOCK_NONE, None, None)


def test_status_after_retire_force_reports_zero_locks(isolated_db: Path) -> None:
    bid = _seed_locked(isolated_db)
    _run("retire", "--force", bid)
    s = MemoryStore(str(isolated_db))
    try:
        assert s.count_locked() == 0
        assert s.list_locked_beliefs() == []
    finally:
        s.close()


# --- count_locked uses the lifecycle filter ----------------------------------


def test_count_locked_excludes_retired_rows(isolated_db: Path) -> None:
    _seed_legacy_retired_lock(isolated_db)
    s = MemoryStore(str(isolated_db))
    try:
        # Undo what the open migration just did, so the count itself is
        # what this test measures.
        s._conn.execute(  # noqa: SLF001
            "UPDATE beliefs SET lock_level = ? WHERE id = ?", (LOCK_USER, _BID)
        )
        s._conn.commit()  # noqa: SLF001
        assert s.count_locked() == len(s.list_locked_beliefs()) == 0
    finally:
        s.close()


# --- unlock reaches a retired row --------------------------------------------


def test_unlock_reaches_retired_locked_row(isolated_db: Path) -> None:
    _seed_legacy_retired_lock(isolated_db)
    s = MemoryStore(str(isolated_db))
    try:
        s._conn.execute(  # noqa: SLF001, undo the migration for this test
            "UPDATE beliefs SET lock_level = ?, locked_at = ?, "
            "lock_expires_at = ? WHERE id = ?",
            (LOCK_USER, _LOCKED_AT, _EXPIRES_AT, _BID),
        )
        s._conn.commit()  # noqa: SLF001
        result = unlock(s, _BID, now="2026-07-01T00:00:00Z")
    finally:
        s.close()
    assert result.already_unlocked is False
    assert result.audit_event_id is not None
    assert _raw_lock_fields(isolated_db, _BID) == (LOCK_NONE, None, None)


def test_unlock_of_retired_row_keeps_it_out_of_search(isolated_db: Path) -> None:
    _seed_legacy_retired_lock(isolated_db)
    s = MemoryStore(str(isolated_db))
    try:
        s._conn.execute(  # noqa: SLF001, undo the migration for this test
            "UPDATE beliefs SET lock_level = ? WHERE id = ?", (LOCK_USER, _BID)
        )
        s._conn.commit()  # noqa: SLF001
        unlock(s, _BID)
        b = s.get_belief(_BID, include_retired=True)
        assert b is not None and b.valid_to == _RETIRED_AT
    finally:
        s.close()
    assert not _in_fts(isolated_db, _BID)


def test_unlock_cli_on_retired_locked_row_exits_zero(isolated_db: Path) -> None:
    _seed_legacy_retired_lock(isolated_db)
    s = MemoryStore(str(isolated_db))
    try:
        s._conn.execute(  # noqa: SLF001, undo the migration for this test
            "UPDATE beliefs SET lock_level = ? WHERE id = ?", (LOCK_USER, _BID)
        )
        s._conn.commit()  # noqa: SLF001
    finally:
        s.close()
    code, _ = _run("unlock", _BID)
    assert code == 0
    assert _raw_lock_fields(isolated_db, _BID)[0] == LOCK_NONE


# --- one-time migration ------------------------------------------------------


def test_migration_unlocks_retired_locked_rows(isolated_db: Path) -> None:
    _seed_legacy_retired_lock(isolated_db)
    MemoryStore(str(isolated_db)).close()
    assert _raw_lock_fields(isolated_db, _BID) == (LOCK_NONE, None, None)
    assert _sources(isolated_db, _BID) == [(FEEDBACK_SOURCE_LOCK_RETIRE, 0.0)]


def test_migration_stamps_marker_and_runs_once(isolated_db: Path) -> None:
    _seed_legacy_retired_lock(isolated_db)
    MemoryStore(str(isolated_db)).close()
    s = MemoryStore(str(isolated_db))
    try:
        assert s.get_schema_meta(SCHEMA_META_RETIRED_LOCK_CLEANUP) is not None
        # A retired lock that appears after the marker is not swept again
        # on open: the pass is one-time.
        s._conn.execute(  # noqa: SLF001
            "UPDATE beliefs SET lock_level = ? WHERE id = ?", (LOCK_USER, _BID)
        )
        s._conn.commit()  # noqa: SLF001
    finally:
        s.close()
    MemoryStore(str(isolated_db)).close()
    assert _raw_lock_fields(isolated_db, _BID)[0] == LOCK_USER
    assert _sources(isolated_db, _BID) == [(FEEDBACK_SOURCE_LOCK_RETIRE, 0.0)]


def test_migration_is_idempotent_when_rerun(isolated_db: Path) -> None:
    _seed_legacy_retired_lock(isolated_db)
    s = MemoryStore(str(isolated_db))
    try:
        s._conn.execute(  # noqa: SLF001
            "DELETE FROM schema_meta WHERE key = ?",
            (SCHEMA_META_RETIRED_LOCK_CLEANUP,),
        )
        s._conn.commit()  # noqa: SLF001
        assert s._maybe_clear_retired_locks() == 0  # noqa: SLF001
    finally:
        s.close()
    assert _sources(isolated_db, _BID) == [(FEEDBACK_SOURCE_LOCK_RETIRE, 0.0)]


def test_migration_leaves_active_locks_alone(isolated_db: Path) -> None:
    active = _seed_locked(isolated_db, bid="1122334455667788", content="kept")
    _seed_legacy_retired_lock(isolated_db)
    MemoryStore(str(isolated_db)).close()
    assert _raw_lock_fields(isolated_db, active) == (
        LOCK_USER, _LOCKED_AT, _EXPIRES_AT,
    )
    assert _sources(isolated_db, active) == []


# --- restore of a formerly locked belief -------------------------------------


def test_restore_of_formerly_locked_belief_is_active_and_unlocked(
    isolated_db: Path,
) -> None:
    bid = _seed_locked(isolated_db)
    _run("retire", "--force", bid)
    code, _ = _run("restore", bid)
    assert code == 0
    s = MemoryStore(str(isolated_db))
    try:
        b = s.get_belief(bid)
        assert b is not None and b.valid_to is None
        assert b.lock_level == LOCK_NONE
        assert s.count_locked() == 0
    finally:
        s.close()
    assert [src for src, _ in _sources(isolated_db, bid)] == [
        "user_retired_force", FEEDBACK_SOURCE_LOCK_RETIRE, "user_restored",
    ]


# --- review round 1 ------------------------------------------------------------


def test_restore_of_formerly_locked_belief_prints_relock_hint(
    isolated_db: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    bid = _seed_locked(isolated_db)
    _run("retire", "--force", bid)
    capsys.readouterr()
    code, out = _run("restore", bid)
    err = capsys.readouterr().err
    assert code == 0
    assert out == f"restored: {bid}\n"
    lines = [line for line in err.splitlines() if "aelf lock" in line]
    assert len(lines) == 1
    assert bid in lines[0]


def test_restore_of_never_locked_belief_prints_no_hint(
    isolated_db: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    bid = _seed_locked(isolated_db)
    s = MemoryStore(str(isolated_db))
    try:
        s._conn.execute(  # noqa: SLF001
            "UPDATE beliefs SET lock_level = ?, locked_at = NULL WHERE id = ?",
            (LOCK_NONE, bid),
        )
        s._conn.commit()  # noqa: SLF001
    finally:
        s.close()
    _run("retire", bid)
    capsys.readouterr()
    code, _ = _run("restore", bid)
    assert code == 0
    assert "aelf lock" not in capsys.readouterr().err


def test_migration_hint_after_restore_of_legacy_row(
    isolated_db: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    _seed_legacy_retired_lock(isolated_db)
    MemoryStore(str(isolated_db)).close()
    capsys.readouterr()
    code, _ = _run("restore", _BID)
    assert code == 0
    assert "aelf lock" in capsys.readouterr().err


class _BeforeFirstLockUpdate:
    """Connection proxy: run `hook` once, just before the first
    lock-clearing UPDATE, so another writer's change lands between this
    connection's select and its update."""

    def __init__(self, conn: object, hook: Callable[[], None]) -> None:
        self._c = conn
        self._hook = hook
        self.fired = False

    def execute(self, sql: str, *args: object) -> object:
        if not self.fired and sql.startswith("UPDATE beliefs SET lock_level"):
            self.fired = True
            self._hook()
        return self._c.execute(sql, *args)  # type: ignore[attr-defined]

    def __getattr__(self, name: str) -> object:
        return getattr(self._c, name)


def test_migration_writes_one_audit_row_when_two_openers_race(
    isolated_db: Path,
) -> None:
    _seed_legacy_retired_lock(isolated_db)
    b = MemoryStore(str(isolated_db))
    try:
        # Put the pre-fix state back, so B's own pass is due again.
        b._conn.execute(  # noqa: SLF001
            "UPDATE beliefs SET lock_level = ? WHERE id = ?", (LOCK_USER, _BID)
        )
        b._conn.execute("DELETE FROM feedback_history")  # noqa: SLF001
        b._conn.execute(  # noqa: SLF001
            "DELETE FROM schema_meta WHERE key = ?",
            (SCHEMA_META_RETIRED_LOCK_CLEANUP,),
        )
        b._conn.commit()  # noqa: SLF001
        real = b._conn  # noqa: SLF001
        proxy = _BeforeFirstLockUpdate(
            real, lambda: MemoryStore(str(isolated_db)).close()
        )
        b._conn = proxy  # type: ignore[assignment]  # noqa: SLF001
        try:
            cleared = b._maybe_clear_retired_locks()  # noqa: SLF001
        finally:
            b._conn = real  # noqa: SLF001
        assert proxy.fired
        assert cleared == 0
    finally:
        b.close()
    assert _sources(isolated_db, _BID) == [(FEEDBACK_SOURCE_LOCK_RETIRE, 0.0)]


def test_migration_runs_before_the_expiry_sweep(isolated_db: Path) -> None:
    """A retired lock past its window is recorded as ended by the retire.

    The sweep has no `valid_to` filter, so if it ran first it would flip
    the row and write `lock:expire`, leaving the migration nothing to do.
    """
    _seed_legacy_retired_lock(isolated_db, expires_at="2026-01-01T00:00:00+00:00")
    MemoryStore(str(isolated_db)).close()
    assert _sources(isolated_db, _BID) == [(FEEDBACK_SOURCE_LOCK_RETIRE, 0.0)]


def test_clear_lock_bumps_belief_version_and_fires_invalidation(
    isolated_db: Path,
) -> None:
    bid = _seed_locked(isolated_db)
    s = MemoryStore(str(isolated_db))
    try:
        def counter() -> int:
            row = s._conn.execute(  # noqa: SLF001
                "SELECT COALESCE(SUM(counter), 0) AS n FROM belief_versions "
                "WHERE belief_id = ?",
                (bid,),
            ).fetchone()
            return int(row["n"])

        fired: list[int] = []
        s.add_invalidation_callback(lambda: fired.append(1))
        before = counter()
        assert s.clear_lock(bid) is True
        assert counter() == before + 1
        assert fired == [1]
    finally:
        s.close()


def test_unlock_writes_nothing_when_clear_lock_clears_nothing(
    isolated_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale locked snapshot, with the row already unlocked underneath
    (a concurrent unlock), must not produce a second `lock:unlock` row."""
    bid = _seed_locked(isolated_db)
    s = MemoryStore(str(isolated_db))
    try:
        stale = s.get_belief(bid)
        assert stale is not None and stale.lock_level == LOCK_USER
        s._conn.execute(  # noqa: SLF001
            "UPDATE beliefs SET lock_level = ? WHERE id = ?", (LOCK_NONE, bid)
        )
        s._conn.commit()  # noqa: SLF001
        monkeypatch.setattr(s, "get_belief", lambda *a, **k: stale)
        result = unlock(s, bid)
    finally:
        s.close()
    assert result.already_unlocked is True
    assert result.audit_event_id is None
    assert _sources(isolated_db, bid) == []


def test_migration_leaves_a_row_restored_after_its_select_locked(
    isolated_db: Path,
) -> None:
    _seed_legacy_retired_lock(isolated_db)
    b = MemoryStore(str(isolated_db))
    try:
        b._conn.execute(  # noqa: SLF001, put the pre-fix state back
            "UPDATE beliefs SET lock_level = ? WHERE id = ?", (LOCK_USER, _BID)
        )
        b._conn.execute("DELETE FROM feedback_history")  # noqa: SLF001
        b._conn.execute(  # noqa: SLF001
            "DELETE FROM schema_meta WHERE key = ?",
            (SCHEMA_META_RETIRED_LOCK_CLEANUP,),
        )
        b._conn.commit()  # noqa: SLF001

        def restore_elsewhere() -> None:
            # A second process restores the row, raw, so its own open
            # migration cannot clear the lock first.
            c = sqlite3.connect(str(isolated_db))
            try:
                c.execute("UPDATE beliefs SET valid_to = NULL WHERE id = ?", (_BID,))
                c.commit()
            finally:
                c.close()

        real = b._conn  # noqa: SLF001
        proxy = _BeforeFirstLockUpdate(real, restore_elsewhere)
        b._conn = proxy  # type: ignore[assignment]  # noqa: SLF001
        try:
            cleared = b._maybe_clear_retired_locks()  # noqa: SLF001
        finally:
            b._conn = real  # noqa: SLF001
        assert proxy.fired
        assert cleared == 0
    finally:
        b.close()
    assert _raw_lock_fields(isolated_db, _BID)[0] == LOCK_USER
    assert _sources(isolated_db, _BID) == []
