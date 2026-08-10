"""Read-only CLI commands against a non-writable store (#1416).

`aelf search` died in `MemoryStore.__init__` with `sqlite3.Operational
Error: attempt to write a readonly database` before retrieval began.
That is the everyday shape of a Codex workspace-write session: the
workspace is writable, `.git/` — where the repo store lives — is not.

What the reproduction actually proves, established before the fix was
designed: the failing statement is a *read* (`SELECT name FROM
sqlite_master`), and it fails because the store is in WAL mode and
SQLite must create the `-shm` shared-memory sidecar **in the database's
directory** before any page can be read. So there are two distinct
regimes, and this file pins both:

* usable `-wal`/`-shm` present -> a `mode=ro` handle reads fine, and
  the observational commands must succeed;
* sidecars absent and the directory not writable -> SQLite cannot open
  the store at all, and the command must say so in one line instead of
  unwinding a traceback. `immutable=1` would open it, and is refused on
  purpose: it promises the engine the file never changes while an
  aelfrice hook outside the sandbox may still be writing it.

**#1416's acceptance criterion 1 is therefore NOT met, and this file
says so rather than around it.** AC1 asks that `aelf search` *succeed*
with the database at 0444 and its directory at 0555. That regime is the
second one above: with no live writer there are no sidecars, and
`test_missing_sidecars_report_instead_of_tracebacking` asserts exit 1
with a message. Independently confirmed at the engine, below aelfrice:
on SQLite 3.50.4 a plain `connect` and a `mode=ro` connect both fail
`SQLITE_READONLY_DIRECTORY` against such a directory, and only
`immutable=1` opens it. Serving AC1 as written means taking the
`immutable=1` promise, which is an operator decision about a
correctness/availability trade, not one to make silently inside a
bugfix — so the issue stays open on AC1.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from aelfrice.bm25 import sidecar_path_for
from aelfrice.cli import main
from aelfrice.db_paths import open_store_for_read
from aelfrice.store import (
    _SCHEMA,
    READ_ONLY_REQUIRED_TABLES,
    MemoryStore,
    ReadOnlyStoreUnavailable,
    StoreSchemaTooOld,
)

pytestmark = pytest.mark.skipif(
    os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="POSIX permission bits; root ignores them",
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed(db: Path) -> None:
    """Create a current store holding one locked belief."""
    store = MemoryStore(str(db))
    store.close()
    assert main(["lock", "codex scratch fact"]) == 0


@pytest.fixture()
def store_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """A seeded store directory whose permissions are always restored.

    Restoring in a fixture rather than the test body matters: a test that
    fails mid-way would otherwise leave a mode-555 directory behind and
    break tmp cleanup for the whole session.
    """
    d = tmp_path / "store"
    d.mkdir()
    monkeypatch.setenv("AELFRICE_DB", str(d / "memory.db"))
    _seed(d / "memory.db")
    try:
        yield d
    finally:
        d.chmod(0o755)
        for child in d.iterdir():
            child.chmod(0o644)


def _freeze(d: Path) -> None:
    """Make the directory and everything in it read-only."""
    for child in d.iterdir():
        child.chmod(0o444)
    d.chmod(0o555)


def _hold_sidecars(db: Path) -> sqlite3.Connection:
    """Keep a connection open so `-wal`/`-shm` exist on disk.

    SQLite deletes both when the last connection closes, so a store at
    rest has neither. A live writer elsewhere in the machine — exactly
    the aelfrice hook running outside the sandbox — is what leaves them
    there for a read-only reader to use.
    """
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("SELECT count(*) FROM beliefs").fetchone()
    return conn


# --- regime 1: sidecars present, directory frozen --------------------------


def test_search_succeeds_against_a_frozen_store(
    store_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = store_dir / "memory.db"
    holder = _hold_sidecars(db)
    try:
        # An older binary's store: one table the DDL battery would
        # recreate, so the writable open must attempt a real write.
        sqlite3.connect(str(db)).executescript(
            "DROP TABLE IF EXISTS exploration_events;"
        )
        _freeze(store_dir)
        before = _digest(db)
        capsys.readouterr()
        rc = main(["search", "codex"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "codex scratch fact" in out
        assert _digest(db) == before
    finally:
        holder.close()


def test_frozen_store_reads_through_a_read_only_handle(
    store_dir: Path,
) -> None:
    db = store_dir / "memory.db"
    holder = _hold_sidecars(db)
    try:
        sqlite3.connect(str(db)).executescript(
            "DROP TABLE IF EXISTS exploration_events;"
        )
        _freeze(store_dir)
        store = open_store_for_read()
        try:
            assert store.read_only is True
            assert store.count_beliefs() == 1
        finally:
            store.close()
    finally:
        holder.close()


def test_status_and_locked_survive_a_frozen_store(
    store_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = store_dir / "memory.db"
    holder = _hold_sidecars(db)
    try:
        sqlite3.connect(str(db)).executescript(
            "DROP TABLE IF EXISTS exploration_events;"
        )
        _freeze(store_dir)
        before = _digest(db)
        capsys.readouterr()
        assert main(["status"]) == 0
        assert "beliefs:" in capsys.readouterr().out
        assert main(["locked"]) == 0
        assert "codex scratch fact" in capsys.readouterr().out
        assert _digest(db) == before
    finally:
        holder.close()


def test_the_other_observational_commands_survive_a_frozen_store(
    store_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """#1416 AC3: the rest of the declared read-only surface.

    `speculative`, `core`, `stale`, `introspect`, `graph` and `feed` all
    advertise an observational contract, and every one of them opened
    the store read-write. `feed` is in the list because it is user-
    visible as a read command, though it reads a JSONL log rather than
    the database.

    The dropped table is load-bearing, not decoration: a frozen store
    whose schema is already complete gives the *writable* open nothing
    to write, so it succeeds and the fallback is never reached. Removing
    one table the open-time DDL battery recreates is what forces the
    permission failure this test is about.
    """
    db = store_dir / "memory.db"
    holder = _hold_sidecars(db)
    try:
        sqlite3.connect(str(db)).executescript(
            "DROP TABLE IF EXISTS exploration_events;"
        )
        _freeze(store_dir)
        before = _digest(db)
        for argv in (
            ["speculative"],
            ["core"],
            ["stale"],
            ["introspect"],
            ["graph", "codex"],
            ["feed"],
        ):
            capsys.readouterr()
            assert main(list(argv)) == 0, f"{argv} failed"
            assert "Traceback" not in capsys.readouterr().err
        assert _digest(db) == before
    finally:
        holder.close()


def test_observational_read_writes_no_bm25f_sidecar(
    store_dir: Path,
) -> None:
    """`mode=ro` binds the engine, not the files aelfrice writes beside it.

    A read-only retrieval was observed creating `memory.db.bm25f` in a
    directory the caller was only meant to be reading. Only the *write*
    is suppressed — a read-only handle still resolves the sidecar for
    loading, so an existing index is not thrown away.
    """
    db = store_dir / "memory.db"
    store = MemoryStore(str(db), read_only=True)
    try:
        assert sidecar_path_for(store, for_write=True) is None
        assert sidecar_path_for(store) is not None, "the load path stays open"
        from aelfrice.retrieval import retrieve

        assert [b.content for b in retrieve(store, "codex", token_budget=800)]
    finally:
        store.close()
    assert not (store_dir / "memory.db.bm25f").exists()


# --- regime 2: sidecars absent, directory frozen ---------------------------


def test_missing_sidecars_report_instead_of_tracebacking(
    store_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The filed reproduction, verbatim: no live writer, frozen directory."""
    db = store_dir / "memory.db"
    assert not (store_dir / "memory.db-shm").exists()
    _freeze(store_dir)
    before = _digest(db)
    capsys.readouterr()
    rc = main(["search", "codex"])
    err = capsys.readouterr().err
    assert rc == 1
    assert str(db) in err
    assert "write access" in err
    assert "Traceback" not in err
    assert _digest(db) == before


def test_read_only_open_raises_the_typed_error_at_the_open_site(
    store_dir: Path,
) -> None:
    """Not `sqlite3.OperationalError` from deep inside retrieval."""
    db = store_dir / "memory.db"
    _freeze(store_dir)
    with pytest.raises(ReadOnlyStoreUnavailable) as excinfo:
        MemoryStore(str(db), read_only=True)
    message = str(excinfo.value)
    assert "-shm" in message
    assert str(store_dir) in message


# --- regime 3: the store is readable but older than this binary ------------


def test_an_outdated_schema_reports_a_migration_instead_of_tracebacking(
    store_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """#1416 AC4. A read-only handle runs no migration, by construction.

    Before this check the shortfall surfaced as
    `sqlite3.OperationalError: no such table: edges`, raised from
    `count_edges` four frames below the open — the same traceback class
    #1416 was filed about, relocated rather than removed.
    """
    db = store_dir / "memory.db"
    holder = _hold_sidecars(db)
    try:
        sqlite3.connect(str(db)).executescript("DROP TABLE edges;")
        _freeze(store_dir)
        before = _digest(db)
        capsys.readouterr()
        rc = main(["status"])
        err = capsys.readouterr().err
        assert rc == 1
        assert "Traceback" not in err
        assert "no such table" not in err
        assert "edges" in err
        assert "write access" in err
        assert _digest(db) == before
    finally:
        holder.close()


def test_the_outdated_schema_error_is_typed_at_the_open_site(
    store_dir: Path,
) -> None:
    db = store_dir / "memory.db"
    holder = _hold_sidecars(db)
    try:
        sqlite3.connect(str(db)).executescript("DROP TABLE belief_entities;")
        _freeze(store_dir)
        with pytest.raises(StoreSchemaTooOld) as excinfo:
            MemoryStore(str(db), read_only=True)
        assert "belief_entities" in str(excinfo.value)
        # Caught by the CLI's existing #1416 handler, hence the subclass.
        assert isinstance(excinfo.value, ReadOnlyStoreUnavailable)
    finally:
        holder.close()


def test_a_table_no_read_command_touches_does_not_block_the_read(
    store_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The required set is a measured floor, not the whole schema.

    Refusing to read a store that merely lacks `exploration_events` —
    which not one observational command queries — would deny exactly the
    service #1416 asks for. This is the arm that fails if the required
    set is widened to "every table `_SCHEMA` creates".
    """
    db = store_dir / "memory.db"
    holder = _hold_sidecars(db)
    try:
        sqlite3.connect(str(db)).executescript(
            "DROP TABLE exploration_events;"
        )
        _freeze(store_dir)
        capsys.readouterr()
        assert main(["search", "codex"]) == 0
        assert "codex scratch fact" in capsys.readouterr().out
    finally:
        holder.close()


def test_every_required_table_is_one_this_schema_creates() -> None:
    """Drift guard: a rename must break here, not at a user's open."""
    fresh = sqlite3.connect(":memory:")
    for statement in _SCHEMA:
        fresh.execute(statement)
    created = {
        str(row[0])
        for row in fresh.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        )
    }
    assert READ_ONLY_REQUIRED_TABLES <= created
    assert READ_ONLY_REQUIRED_TABLES  # a silently emptied set gates nothing


def test_writable_store_still_gets_a_writable_handle(
    store_dir: Path,
) -> None:
    """The fallback must not degrade the ordinary case.

    A read-only handle runs no migration and no expired-lock sweep, so
    silently preferring it would change `aelf locked`'s semantics for
    every user whose store is perfectly writable.
    """
    store = open_store_for_read()
    try:
        assert store.read_only is False
    finally:
        store.close()
