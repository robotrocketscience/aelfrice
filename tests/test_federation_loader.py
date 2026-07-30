"""Unit tests for `aelfrice.federation` loader + open helpers (#655)."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from aelfrice.federation import (
    ENV_KNOWLEDGE_DEPS,
    ForeignBeliefError,
    PeerDep,
    load_peer_deps,
    open_peer_connection,
    resolve_knowledge_deps_path,
)


def _write_deps(tmp_path: Path, deps: list[dict[str, str]]) -> Path:
    p = tmp_path / "knowledge_deps.json"
    p.write_text(json.dumps({"version": 1, "deps": deps}), encoding="utf-8")
    return p


def test_load_peer_deps_returns_empty_when_no_path():
    """No env override and no git tree → empty list (no crash)."""
    assert load_peer_deps(deps_path=None) == []


def test_load_peer_deps_missing_file(tmp_path: Path):
    """A configured path that doesn't exist → empty list, not raise."""
    p = tmp_path / "knowledge_deps.json"
    assert not p.exists()
    assert load_peer_deps(deps_path=p) == []


def test_load_peer_deps_parses_absolute_path(tmp_path: Path):
    peer = tmp_path / "peer.db"
    peer.write_bytes(b"")  # exists; not a valid SQLite file but path resolves
    deps_path = _write_deps(tmp_path, [{"name": "global", "path": str(peer)}])
    [dep] = load_peer_deps(deps_path=deps_path)
    assert dep.name == "global"
    assert dep.path == peer
    assert dep.reachable is True


def test_load_peer_deps_resolves_relative_path(tmp_path: Path):
    """Relative paths resolve from the deps-file directory, not cwd."""
    sub = tmp_path / "shared"
    sub.mkdir()
    peer = sub / "memory.db"
    peer.write_bytes(b"")
    deps_path = _write_deps(
        tmp_path, [{"name": "team", "path": "shared/memory.db"}]
    )
    [dep] = load_peer_deps(deps_path=deps_path)
    assert dep.path == peer.resolve()
    assert dep.reachable is True


def test_load_peer_deps_expands_tilde(tmp_path: Path, monkeypatch):
    """`~` in path expands to $HOME — required for the
    `~/.aelfrice/shared/global/memory.db` form in the issue."""
    monkeypatch.setenv("HOME", str(tmp_path))
    deps_path = _write_deps(
        tmp_path, [{"name": "global", "path": "~/global.db"}]
    )
    [dep] = load_peer_deps(deps_path=deps_path)
    assert dep.path == tmp_path / "global.db"
    assert dep.reachable is False  # file does not actually exist


def test_load_peer_deps_marks_missing_unreachable(tmp_path: Path):
    """Missing peer file → reachable=False, not raise. Federation is opportunistic."""
    deps_path = _write_deps(
        tmp_path, [{"name": "ghost", "path": "/nonexistent/aelfrice.db"}]
    )
    [dep] = load_peer_deps(deps_path=deps_path)
    assert dep.reachable is False


def test_load_peer_deps_rejects_unsupported_version(tmp_path: Path):
    p = tmp_path / "knowledge_deps.json"
    p.write_text(json.dumps({"version": 99, "deps": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported version"):
        load_peer_deps(deps_path=p)


def test_load_peer_deps_rejects_malformed_json(tmp_path: Path):
    p = tmp_path / "knowledge_deps.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_peer_deps(deps_path=p)


def test_load_peer_deps_rejects_duplicate_names(tmp_path: Path):
    deps_path = _write_deps(
        tmp_path,
        [
            {"name": "a", "path": "x.db"},
            {"name": "a", "path": "y.db"},
        ],
    )
    with pytest.raises(ValueError, match="duplicate dep name"):
        load_peer_deps(deps_path=deps_path)


def test_load_peer_deps_rejects_missing_name(tmp_path: Path):
    p = tmp_path / "knowledge_deps.json"
    p.write_text(
        json.dumps({"version": 1, "deps": [{"path": "x.db"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing non-empty 'name'"):
        load_peer_deps(deps_path=p)


def test_load_peer_deps_rejects_top_level_array(tmp_path: Path):
    p = tmp_path / "knowledge_deps.json"
    p.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
    with pytest.raises(ValueError, match="top-level must be a JSON object"):
        load_peer_deps(deps_path=p)


def test_load_peer_deps_empty_deps_array(tmp_path: Path):
    deps_path = _write_deps(tmp_path, [])
    assert load_peer_deps(deps_path=deps_path) == []


def test_resolve_knowledge_deps_path_env_override(tmp_path: Path, monkeypatch):
    target = tmp_path / "custom.json"
    monkeypatch.setenv(ENV_KNOWLEDGE_DEPS, str(target))
    assert resolve_knowledge_deps_path() == target


def test_foreign_belief_error_carries_metadata():
    err = ForeignBeliefError("abc123", "team-shared")
    assert err.belief_id == "abc123"
    assert err.owning_scope == "team-shared"
    assert "team-shared" in str(err)
    assert isinstance(err, ValueError)


def test_open_peer_connection_is_read_only(tmp_path: Path):
    """Peer handles must reject writes — guards against accidental mutations."""
    peer_path = tmp_path / "peer.db"
    # Materialise the peer with one table so the open succeeds.
    bootstrap = sqlite3.connect(str(peer_path))
    bootstrap.execute("CREATE TABLE t (x INTEGER)")
    bootstrap.execute("INSERT INTO t VALUES (1)")
    bootstrap.commit()
    bootstrap.close()

    conn = open_peer_connection(peer_path)
    try:
        rows = list(conn.execute("SELECT x FROM t"))
        assert [r["x"] for r in rows] == [1]
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("INSERT INTO t VALUES (2)")
            conn.commit()
    finally:
        conn.close()


def test_peer_dep_is_frozen():
    """PeerDep is immutable — prevents accidental in-place edits in caches."""
    dep = PeerDep(name="x", path=Path("/x"), reachable=False)
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        dep.name = "y"  # type: ignore[misc]


def _wal_peer(path: Path, rows: int = 30) -> sqlite3.Connection:
    """Build a WAL-mode peer and return the writer, still open.

    Leaving the writer open is the point: the commits stay in
    `memory.db-wal` with nothing checkpointing them, which is the shape
    of a store held open by a running hook. Caller closes.
    """
    w = sqlite3.connect(str(path))
    w.execute("PRAGMA journal_mode=WAL")
    w.execute("CREATE TABLE beliefs (id TEXT PRIMARY KEY)")
    w.executemany(
        "INSERT INTO beliefs VALUES (?)", [(f"b{i}",) for i in range(rows)]
    )
    w.commit()
    return w


def test_peer_connection_reads_an_uncheckpointed_wal(tmp_path: Path):
    """Committed-but-uncheckpointed rows must be visible (#1198).

    `immutable=1` promises SQLite the file cannot change, and SQLite
    honours that by ignoring the WAL entirely — so a peer whose schema
    is still in the WAL raised `no such table`, and one whose rows were
    still in the WAL under-read with no error at all.
    """
    peer_path = tmp_path / "peer.db"
    writer = _wal_peer(peer_path)
    try:
        wal = peer_path.with_name(peer_path.name + "-wal")
        assert wal.exists() and wal.stat().st_size > 0, "WAL not live"

        conn = open_peer_connection(peer_path)
        try:
            assert conn.execute(
                "SELECT count(*) FROM beliefs"
            ).fetchone()[0] == 30
        finally:
            conn.close()
    finally:
        writer.close()


def test_peer_connection_is_read_only_against_a_live_wal(tmp_path: Path):
    """Dropping `immutable=1` must not weaken the read-only guarantee."""
    peer_path = tmp_path / "peer.db"
    writer = _wal_peer(peer_path)
    try:
        conn = open_peer_connection(peer_path)
        try:
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                conn.execute("INSERT INTO beliefs VALUES ('x')")
                conn.commit()
        finally:
            conn.close()
    finally:
        writer.close()


def test_peer_connection_falls_back_on_read_only_media(tmp_path: Path):
    """A read-only directory must still be readable.

    A WAL-mode DB needs to create `-shm` even when fully checkpointed,
    so plain `mode=ro` fails there with `attempt to write a readonly
    database`. The `immutable=1` fallback is truthful on a read-only
    medium and is the only way to read the peer at all.
    """
    peer_dir = tmp_path / "ro"
    peer_dir.mkdir()
    peer_path = peer_dir / "peer.db"
    _wal_peer(peer_path).close()  # clean close checkpoints and drops the WAL

    peer_dir.chmod(0o555)
    try:
        # A root user ignores the directory mode, which would make this
        # test pass without ever reaching the fallback. Confirm the
        # plain form really is blocked here before asserting on it.
        probe = sqlite3.connect(f"file:{peer_path}?mode=ro", uri=True)
        try:
            probe.execute("PRAGMA schema_version").fetchone()
            pytest.skip("read-only directory not enforced (running as root?)")
        except sqlite3.OperationalError:
            # Expected, and the precondition this test needs: the plain
            # `mode=ro` form really is blocked here, so reaching the
            # fallback below is the behaviour under test rather than an
            # accident of the environment.
            pass
        finally:
            probe.close()

        conn = open_peer_connection(peer_path)
        try:
            assert conn.execute(
                "SELECT count(*) FROM beliefs"
            ).fetchone()[0] == 30
        finally:
            conn.close()
    finally:
        peer_dir.chmod(0o755)


def test_a_non_readonly_error_does_not_fall_back_to_immutable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lock or FS error must not hand back a WAL-blind handle (#1198).

    The immutable fallback is safe only because it is reached after an
    honest read has already failed *for the one reason that justifies
    it*. Routing every `OperationalError` there would silently restore
    the defect this function exists to remove — and `_peer_conn` caches
    the handle and swallows failures, so it would stay restored for the
    life of the process.
    """
    peer_path = tmp_path / "peer.db"
    writer = _wal_peer(peer_path)

    class _Locked:
        """Stands in for a handle whose first statement hits a lock."""

        def __init__(self, inner: sqlite3.Connection) -> None:
            self._inner = inner
            self.row_factory = None

        def execute(self, *_a: object, **_kw: object) -> object:
            raise sqlite3.OperationalError("database is locked")

        def close(self) -> None:
            self._inner.close()

    try:
        real_connect = sqlite3.connect
        opened: list[str] = []

        def fake_connect(dsn: str, *a: object, **kw: object) -> object:
            # First parameter is deliberately not named `uri` — the real
            # call passes `uri=True` as a keyword, which would collide.
            opened.append(str(dsn))
            conn = real_connect(dsn, *a, **kw)  # type: ignore[arg-type]
            return conn if "immutable=1" in str(dsn) else _Locked(conn)

        monkeypatch.setattr(sqlite3, "connect", fake_connect)

        with pytest.raises(sqlite3.OperationalError, match="locked"):
            open_peer_connection(peer_path)

        assert not any("immutable=1" in u for u in opened), (
            f"fell back to a WAL-blind handle on a non-readonly error: {opened}"
        )
    finally:
        writer.close()
