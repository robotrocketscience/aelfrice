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
  the four sanctioned commands (`search`, `status`, `locked`,
  `speculative`) must succeed;
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


def test_speculative_survives_a_frozen_store(
    store_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """#1416 AC3, fourth and last of the sanctioned commands.

    `search`, `status`, `locked` and `speculative` are the whole of the
    partial the 2026-08-09 ruling sanctioned; the other observational
    commands are held (see the scope test below).

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
        capsys.readouterr()
        assert main(["speculative"]) == 0
        assert "Traceback" not in capsys.readouterr().err
        assert _digest(db) == before
    finally:
        holder.close()


def test_only_the_sanctioned_commands_take_the_read_only_path() -> None:
    """The routed set is exactly the commands audited one at a time.

    #1416's 2026-08-09 operator ruling sanctioned this partial for
    `search`, `status`, `locked` and `speculative`, and held the rest —
    "each needing its own read-only audit rather than a blanket change".
    The 2026-09-17 ruling releases four more, and their audits are below.

    `_cmd_show` is the fifth. The operator's ruling on #1553 was to ship
    the command on this existing routing with honest documentation, and
    to re-file the store-open cost as its own issue covering *every*
    read-only verb — not to bless the routing as satisfying #1553's
    AC5. So this is a per-command decision about one more handler
    joining the set, not the blanket change the #1416 ruling refused,
    and the per-command audit that ruling asks for is below.

    **#1553's AC5 is NOT met by this branch, and this file says so
    rather than around it.** AC5 asks that the store open read-only so
    that printing one row pays no DDL battery, no migration, no scope-id
    mint and no expired-lock sweep. `open_store_for_read` attempts the
    *writable* open first and falls back to `mode=ro` only on a
    permission failure, so against an ordinary writable store `show`
    pays all of it, exactly as `search`, `status`, `locked` and
    `speculative` do. What the routing genuinely buys is the other half:
    the command changes no belief and reads a store the caller cannot
    write, proved by
    `tests/test_cli_show_1553.py::test_show_reads_a_store_it_cannot_write`.
    The cost half is tracked in the re-filed issue, not here.

    The audit itself: `show` reads one `beliefs` row and prints it, so
    no migration is needed — it projects `b.*`, and `_row_to_belief`
    already defaults every post-v1.6 column an older store lacks. The
    audit's one finding is that on the *fallback* handle no expired-lock
    sweep has run (#1314), so a lock past its `lock_expires_at` still
    prints `lock: user` — the same staleness `aelf locked` carries
    there, on a command that reports a stored field rather than acting
    on it.

    The held commands are not merely unfinished: on the fallback path a
    read-only handle runs no migration and no expired-lock sweep, which
    is the per-command semantics the audit is *for*. `feed` never opens
    the store at all (it reads a JSONL log), so it is held trivially.

    The four the 2026-09-17 ruling releases, audited one at a time. Each
    has exactly one store-open call site, and each was mapped back to its
    enclosing `def` by AST before the edit, because the issue comment's
    line numbers point one handler off — at `_cmd_confirm`
    (`apply_feedback(..., respect_lock=False)`) and `_cmd_resolve`
    (`auto_resolve_all_contradictions`), which are writers. Routing
    either onto a read-only handle would ship a bug, so neither is here.

    * `_cmd_graph` resolves seeds (`get_belief`, `find_foreign_owner`, or
      a BM25 top-1), walks `expand_bfs`, and serialises through
      `graph_export`. Reads only. On the fallback handle the walk sees
      whatever `edges` rows exist, unmigrated.
    * `_cmd_stale` runs one `SELECT` over `beliefs` with two date
      thresholds. Reads only. Its one finding: no expired-lock sweep has
      run there, so `--locked-only` still lists a belief past its
      `lock_expires_at` — the same staleness `aelf locked` carries.
    * `_cmd_introspect` hands the store to `introspect.build_report`,
      which projects beliefs and their entities. Reads only.
    * `_cmd_core` composes `list_locked_beliefs`, `list_belief_ids` and
      `get_belief`. Reads only, and carries the same expired-lock finding
      as `stale`, since `locked` is one of the three sets it unions.

    The audit is asserted, not just asserted-to: `test_observational_
    commands_report_instead_of_tracebacking` drives all four against a
    store they cannot write, and the regime-1 arm proves the file's
    digest is unchanged afterwards, which is what "reads only" means.

    Asserted over the handlers' source rather than by driving them,
    because the observable difference only appears on a store that is
    unwritable — a routing added back for a *writable* store is
    invisible behaviourally, which is exactly how the first version of
    this branch shipped it.
    """
    import inspect

    from aelfrice import cli

    routed = {
        name
        for name in dir(cli)
        if name.startswith("_cmd_")
        and callable(getattr(cli, name))
        and getattr(getattr(cli, name), "__module__", "") == cli.__name__
        and "open_store_for_read()" in inspect.getsource(getattr(cli, name))
    }
    assert routed == {
        "_cmd_search", "_cmd_stats", "_cmd_locked", "_cmd_speculative",
        "_cmd_show", "_cmd_graph", "_cmd_stale", "_cmd_introspect",
        "_cmd_core",
    }, "amend the #1416 ruling before routing another command"
    # The adjacent writers, named so a future edit that lands one line
    # off is caught here rather than in a user's store.
    for writer in ("_cmd_confirm", "_cmd_resolve"):
        source = inspect.getsource(getattr(cli, writer))
        assert "open_store_for_read()" not in source
        assert "_open_store()" in source


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


# --- regime 4: a URI metacharacter in the store path -----------------------


@pytest.fixture()
def metachar_store(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Path]:
    """A seeded store under a directory literally named `request.param`.

    Yields the store directory; its parent — where a truncated path lands
    — is `tmp_path`, which stays writable on purpose. Freezing it too
    would mask the defect by denying the stray write rather than not
    attempting it.
    """
    d = tmp_path / str(request.param)
    d.mkdir()
    monkeypatch.setenv("AELFRICE_DB", str(d / "memory.db"))
    _seed(d / "memory.db")
    try:
        yield d
    finally:
        d.chmod(0o755)
        for child in d.iterdir():
            child.chmod(0o644)


@pytest.mark.parametrize(
    "metachar_store", ["store#1", "store?1", "store%41"], indirect=True,
)
def test_a_uri_metacharacter_in_the_path_opens_the_intended_database(
    metachar_store: Path,
) -> None:
    """`file:{path}?mode=ro` is string concatenation, and paths are data.

    `#` starts a URI fragment and `?` starts a second query string, so
    either one truncates the path **and** discards `mode=ro`; SQLite then
    opens the shorter path read-write and creates it as a zero-byte file.
    `%` starts a percent-escape, which decodes to a path that does not
    exist. Measured before the fix on SQLite 3.50.4, and the three cases
    fail differently, so they are asserted together only on the outcome
    they share *after* it:

    * `store#1`, `store?1` -> opened an empty database and left a stray
      `<tmp>/store` behind, so the #1416 schema floor raised
      `StoreSchemaTooOld` and told the user to migrate a store that was
      already current;
    * `store%41` -> `sqlite3.OperationalError: unable to open database
      file`, because `%41` decodes to `A`. The escape must be a complete
      two-hex-digit one to bite: a bare `%1` is malformed, SQLite leaves
      it alone, and such a path opens either way — which is why the
      parameters are whole directory names rather than a bare
      metacharacter spliced into one.

    A space, an apostrophe and a non-ASCII letter open correctly with the
    plain concatenation, so they are not regression coverage for this and
    are deliberately absent.
    """
    db = metachar_store / "memory.db"
    parent_of_parent = metachar_store.parent
    before = sorted(p.name for p in parent_of_parent.iterdir())

    store = MemoryStore(str(db), read_only=True)
    try:
        assert store.read_only is True
        assert [b.content for b in store.list_locked_beliefs()] == [
            "codex scratch fact"
        ]
    finally:
        store.close()

    assert sorted(p.name for p in parent_of_parent.iterdir()) == before, (
        "a truncated URI opened a different path and created it"
    )


@pytest.mark.parametrize("metachar_store", ["store#1"], indirect=True)
def test_search_under_a_hash_path_reports_the_real_failure(
    metachar_store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The user-visible half: the wrong diagnosis, on the filed command.

    Regime 2 (no sidecars, frozen directory) has one correct answer —
    `ReadOnlyStoreUnavailable`, "grant write access to that directory".
    Before the fix the truncated URI opened an empty database somewhere
    else entirely, which passed the sidecar probe and failed the schema
    floor, so the command instead demanded a migration of a store that
    needed none. `StoreSchemaTooOld` subclasses `ReadOnlyStoreUnavailable`,
    so the type alone does not separate them; the message does.
    """
    db = metachar_store / "memory.db"
    parent_of_parent = metachar_store.parent
    assert not (metachar_store / "memory.db-shm").exists()
    _freeze(metachar_store)
    before = sorted(p.name for p in parent_of_parent.iterdir())
    capsys.readouterr()

    rc = main(["search", "codex"])

    err = capsys.readouterr().err
    assert rc == 1
    assert "Traceback" not in err
    assert "-shm" in err, "the real failure is the missing WAL sidecars"
    assert "needs a migration" not in err, (
        "a truncated URI made an empty file look like an old schema"
    )
    assert sorted(p.name for p in parent_of_parent.iterdir()) == before


def test_the_read_only_uri_is_built_by_the_uri_builder() -> None:
    """Percent-encoding, and the abspath-not-resolve choice, pinned.

    `_db_path` keeps the caller's spelling — it places the `.bm25f`
    sidecar and fills the error strings — so the URI is absolutised
    lexically rather than resolved: `resolve()` would follow symlinks and
    let the engine's path drift from the one the store reports.
    """
    from aelfrice.store import read_only_uri

    uri = read_only_uri("/tmp/store#1/memory.db")
    assert uri == "file:///tmp/store%231/memory.db?mode=ro"
    assert read_only_uri("/tmp/store?1/x.db").startswith(
        "file:///tmp/store%3F1/"
    )
    assert read_only_uri("/tmp/store%1/x.db").startswith(
        "file:///tmp/store%251/"
    )
    assert read_only_uri("relative.db").startswith("file:///")


# --- the four observational commands released on 2026-09-17 ----------------

#: `(argv, exit code on the happy path)` for the commands the 2026-09-17
#: ruling routes. `graph` exits 2 when its anchor matches nothing, which
#: is a resolution failure, not a store failure — and the seeded store
#: holds one locked belief, which `expand_bfs` reaches.
_RELEASED_COMMANDS: list[tuple[list[str], int]] = [
    (["core"], 0),
    (["introspect"], 0),
    (["graph", "codex"], 0),
    (["stale", "--older-than", "0", "--cold-for", "0"], 0),
]


def _deny_write_control(d: Path) -> None:
    """Prove the fixture is actually frozen before trusting the result.

    A `chmod` that silently did not take — a permissive umask, an ACL, a
    filesystem mounted without permission support — would turn every
    assertion below into a tautology about a perfectly writable store.
    """
    canary = d / "write-denial-canary"
    with pytest.raises(OSError):
        canary.touch()
    assert not canary.exists()


@pytest.mark.parametrize(
    "argv", [c for c, _ in _RELEASED_COMMANDS], ids=lambda a: a[0]
)
def test_observational_commands_report_instead_of_tracebacking(
    store_dir: Path, capsys: pytest.CaptureFixture[str], argv: list[str]
) -> None:
    """Regime 2 for the four handlers the 2026-09-17 ruling releases.

    Before the routing these called `_open_store()`, so a store the
    caller cannot write dumped a raw `sqlite3.OperationalError` traceback
    out of `MemoryStore.__init__` — `main` catches
    `ReadOnlyStoreUnavailable` and nothing else, so the exception escaped
    the CLI entirely. The assertion that separates the two is the call
    itself: unrouted, `main` *raises* here rather than returning 1.
    """
    db = store_dir / "memory.db"
    assert not (store_dir / "memory.db-shm").exists()
    _freeze(store_dir)
    _deny_write_control(store_dir)
    manifest = sorted(p.name for p in store_dir.iterdir())
    before = _digest(db)
    capsys.readouterr()

    rc = main(argv)

    captured = capsys.readouterr()
    assert rc == 1
    assert "Traceback" not in captured.err
    # The typed message quotes the SQLite error rather than being it, so
    # the discriminator is the `aelf <cmd>: ` prefix the CLI's #1416
    # handler adds and the remediation sentence, not the engine's words.
    assert captured.err.startswith(f"aelf {argv[0]}: ")
    assert str(db) in captured.err
    assert "write access" in captured.err
    assert "-shm" in captured.err
    assert sorted(p.name for p in store_dir.iterdir()) == manifest
    assert _digest(db) == before


@pytest.mark.parametrize(
    "argv,code",
    _RELEASED_COMMANDS,
    ids=[argv[0] for argv, _ in _RELEASED_COMMANDS],
)
def test_observational_commands_read_a_store_they_cannot_write(
    store_dir: Path,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    code: int,
) -> None:
    """Regime 1: the half the routing buys, rather than a politer failure.

    The dropped table is load-bearing: a frozen store whose schema is
    already complete gives the *writable* open nothing to write, so it
    succeeds and the fallback is never exercised. `exploration_events` is
    one the open-time DDL battery recreates and no observational command
    queries.
    """
    db = store_dir / "memory.db"
    holder = _hold_sidecars(db)
    try:
        sqlite3.connect(str(db)).executescript(
            "DROP TABLE IF EXISTS exploration_events;"
        )
        _freeze(store_dir)
        _deny_write_control(store_dir)
        manifest = sorted(p.name for p in store_dir.iterdir())
        before = _digest(db)
        capsys.readouterr()

        rc = main(argv)

        captured = capsys.readouterr()
        assert rc == code, captured.err
        assert "Traceback" not in captured.err
        assert _digest(db) == before, "an observational command wrote"
        assert sorted(p.name for p in store_dir.iterdir()) == manifest
    finally:
        holder.close()
