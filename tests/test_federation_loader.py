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
