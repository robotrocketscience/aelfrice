"""Acceptance tests for v3.0 #688 — scope field on beliefs.

Covers:
* Schema migration: scope column added on fresh DB and on a pre-#688 DB
  opened for the first time (idempotency check).
* Belief dataclass default: scope defaults to 'project'.
* Write-time validation: malformed scope values raise ValueError.
* Two-scope smoke: project A has 1 global + 1 project belief; project B
  (with A in its knowledge_deps.json) sees only the global one via
  search_peer_beliefs.
* shared-group overlay: same setup with shared:foo instead of global
  works when the requesting DB lists shared:foo as a dep.
* per-scope counts via count_beliefs_by_scope().
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from aelfrice.models import (
    BELIEF_FACTUAL,
    BELIEF_SCOPE_GLOBAL,
    BELIEF_SCOPE_PROJECT,
    BELIEF_SCOPE_RE,
    LOCK_NONE,
    ORIGIN_AGENT_INFERRED,
    Belief,
    validate_belief_scope,
)
from aelfrice.store import MemoryStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _belief(
    id_: str,
    content: str,
    *,
    scope: str = BELIEF_SCOPE_PROJECT,
) -> Belief:
    return Belief(
        id=id_,
        content=content,
        content_hash=id_ + "-hash",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        demotion_pressure=0,
        created_at="2026-05-11T00:00:00+00:00",
        last_retrieved_at=None,
        session_id=None,
        origin=ORIGIN_AGENT_INFERRED,
        scope=scope,
    )


def _seed_peer(path: Path, beliefs: list[Belief]) -> None:
    """Create a peer DB at path and populate it with beliefs."""
    store = MemoryStore(str(path))
    try:
        for b in beliefs:
            store.insert_belief(b)
    finally:
        store.close()


def _wire_peers(tmp_path: Path, deps: list[dict], monkeypatch) -> Path:
    """Materialise knowledge_deps.json with the given dep list."""
    deps_file = tmp_path / "knowledge_deps.json"
    deps_file.write_text(
        json.dumps({"version": 1, "deps": deps}),
        encoding="utf-8",
    )
    monkeypatch.setenv("AELFRICE_KNOWLEDGE_DEPS", str(deps_file))
    return deps_file


# ---------------------------------------------------------------------------
# 1. Schema migration
# ---------------------------------------------------------------------------

def test_fresh_db_has_scope_column() -> None:
    """A freshly created DB has the scope column with default 'project'."""
    store = MemoryStore(":memory:")
    try:
        cols = {
            r["name"]
            for r in store._conn.execute(
                "PRAGMA table_info(beliefs)"
            ).fetchall()
        }
        assert "scope" in cols
    finally:
        store.close()


def test_migration_adds_scope_to_existing_db(tmp_path: Path) -> None:
    """Opening a pre-#688 DB (no scope column) adds the column idempotently.

    Simulates a v2.x store that has the beliefs table but no scope column.
    The migration runner should ADD the column without error and existing
    rows should have scope='project' (the SQL DEFAULT).
    """
    db_path = tmp_path / "legacy.db"

    # Build a minimal pre-#688 schema: beliefs without scope column.
    raw = sqlite3.connect(str(db_path))
    raw.execute(
        """
        CREATE TABLE beliefs (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            content_hash TEXT NOT NULL UNIQUE,
            alpha REAL NOT NULL,
            beta REAL NOT NULL,
            type TEXT NOT NULL,
            lock_level TEXT NOT NULL,
            locked_at TEXT,
            demotion_pressure INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            last_retrieved_at TEXT,
            session_id TEXT,
            origin TEXT NOT NULL DEFAULT 'unknown',
            hibernation_score REAL,
            activation_condition TEXT,
            retention_class TEXT NOT NULL DEFAULT 'unknown',
            valid_to TEXT
        )
        """
    )
    raw.execute(
        "INSERT INTO beliefs "
        "(id, content, content_hash, alpha, beta, type, lock_level, "
        "demotion_pressure, created_at, origin, retention_class) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("old-1", "legacy content", "old-1-hash",
         1.0, 1.0, "factual", "none",
         0, "2026-01-01T00:00:00+00:00", "agent_inferred", "fact"),
    )
    raw.commit()
    raw.close()

    # Opening with MemoryStore runs the migration.
    store = MemoryStore(str(db_path))
    try:
        cols = {
            r["name"]
            for r in store._conn.execute(
                "PRAGMA table_info(beliefs)"
            ).fetchall()
        }
        assert "scope" in cols

        # Existing row should have scope='project' from the DEFAULT.
        row = store._conn.execute(
            "SELECT scope FROM beliefs WHERE id = 'old-1'"
        ).fetchone()
        assert row is not None
        assert row["scope"] == "project"
    finally:
        store.close()


def test_migration_is_idempotent(tmp_path: Path) -> None:
    """Opening a store that already has the scope column does not error."""
    db_path = tmp_path / "modern.db"
    store1 = MemoryStore(str(db_path))
    store1.close()
    # Second open should not raise (duplicate-column ALTER is caught).
    store2 = MemoryStore(str(db_path))
    store2.close()


# ---------------------------------------------------------------------------
# 2. Dataclass default
# ---------------------------------------------------------------------------

def test_belief_scope_defaults_to_project() -> None:
    """Belief constructed without scope= gets scope='project'."""
    b = _belief("b1", "content")
    assert b.scope == BELIEF_SCOPE_PROJECT


def test_belief_scope_round_trips_through_store() -> None:
    """scope value persists through insert → get_belief."""
    store = MemoryStore(":memory:")
    try:
        b = _belief("b1", "compiler optimisations", scope=BELIEF_SCOPE_GLOBAL)
        store.insert_belief(b)
        got = store.get_belief("b1")
        assert got is not None
        assert got.scope == BELIEF_SCOPE_GLOBAL
    finally:
        store.close()


# ---------------------------------------------------------------------------
# 3. Validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("valid_scope", [
    "project",
    "global",
    "shared:teamalpha",
    "shared:team-beta",
    "shared:x1",
    "shared:a_b-c",
])
def test_valid_scopes_pass_validation(valid_scope: str) -> None:
    validate_belief_scope(valid_scope)  # must not raise


@pytest.mark.parametrize("bad_scope", [
    "",
    "Project",
    "GLOBAL",
    "shared:",
    "shared:UPPER",
    "shared:has space",
    "shared:has.dot",
    "local",
    "public",
    "project:extra",
])
def test_invalid_scopes_raise_value_error(bad_scope: str) -> None:
    with pytest.raises(ValueError):
        validate_belief_scope(bad_scope)


def test_insert_belief_rejects_invalid_scope() -> None:
    """insert_belief raises ValueError on a malformed scope."""
    store = MemoryStore(":memory:")
    try:
        b = _belief("bad-scope", "content", scope="not-valid-scope")
        with pytest.raises(ValueError, match="invalid belief scope"):
            store.insert_belief(b)
    finally:
        store.close()


def test_update_belief_rejects_invalid_scope() -> None:
    """update_belief raises ValueError on a malformed scope."""
    store = MemoryStore(":memory:")
    try:
        b = _belief("b1", "content")
        store.insert_belief(b)
        got = store.get_belief("b1")
        assert got is not None
        bad = Belief(**{**got.__dict__, "scope": "bad!value"})
        with pytest.raises(ValueError, match="invalid belief scope"):
            store.update_belief(bad)
    finally:
        store.close()


def test_scope_regex_pattern() -> None:
    """BELIEF_SCOPE_RE accepts exactly the documented pattern."""
    assert BELIEF_SCOPE_RE.match("project")
    assert BELIEF_SCOPE_RE.match("global")
    assert BELIEF_SCOPE_RE.match("shared:foo")
    assert BELIEF_SCOPE_RE.match("shared:foo-bar_baz")
    assert not BELIEF_SCOPE_RE.match("shared:")
    assert not BELIEF_SCOPE_RE.match("shared:Foo")
    assert not BELIEF_SCOPE_RE.match("unknown")


# ---------------------------------------------------------------------------
# 4. Two-scope overlay smoke: global visibility
# ---------------------------------------------------------------------------

def test_global_belief_surfaces_in_peer_search(
    tmp_path: Path, monkeypatch
) -> None:
    """project A has 1 global + 1 project belief; project B sees only global."""
    peer_path = tmp_path / "peerA.db"
    _seed_peer(
        peer_path,
        [
            _belief("a-global", "hash tables use buckets", scope="global"),
            _belief("a-project", "hash tables local secret", scope="project"),
        ],
    )
    _wire_peers(
        tmp_path,
        [{"name": "peerA", "path": str(peer_path)}],
        monkeypatch,
    )

    local = MemoryStore(str(tmp_path / "local.db"))
    try:
        hits = local.search_peer_beliefs("hash tables")
        ids = {b.id for b, _, _ in hits}
        # global belief must surface
        assert "a-global" in ids
        # project belief must NOT surface
        assert "a-project" not in ids
    finally:
        local.close()


def test_project_belief_not_surfaced_to_any_peer(
    tmp_path: Path, monkeypatch
) -> None:
    """A project-scope belief is never returned by search_peer_beliefs."""
    peer_path = tmp_path / "peerA.db"
    _seed_peer(
        peer_path,
        [_belief("a-private", "private implementation detail", scope="project")],
    )
    _wire_peers(
        tmp_path,
        [{"name": "peerA", "path": str(peer_path)}],
        monkeypatch,
    )

    local = MemoryStore(str(tmp_path / "local.db"))
    try:
        hits = local.search_peer_beliefs("private implementation")
        assert len(hits) == 0
    finally:
        local.close()


# ---------------------------------------------------------------------------
# 5. Two-scope overlay smoke: shared:<name> group membership
# ---------------------------------------------------------------------------

def test_shared_group_belief_surfaces_when_dep_listed(
    tmp_path: Path, monkeypatch
) -> None:
    """shared:foo belief on peer A is visible to B when B lists shared:foo dep."""
    peer_path = tmp_path / "peerA.db"
    _seed_peer(
        peer_path,
        [
            _belief("a-shared", "binary search trees have log depth",
                    scope="shared:teamfoo"),
            _belief("a-local", "binary search local only", scope="project"),
        ],
    )
    # local DB declares shared:teamfoo in its deps (group membership convention)
    _wire_peers(
        tmp_path,
        [
            {"name": "peerA", "path": str(peer_path)},
            {"name": "shared:teamfoo", "path": str(peer_path)},
        ],
        monkeypatch,
    )

    local = MemoryStore(str(tmp_path / "local.db"))
    try:
        hits = local.search_peer_beliefs("binary search")
        ids = {b.id for b, _, _ in hits}
        assert "a-shared" in ids
        assert "a-local" not in ids
    finally:
        local.close()


def test_shared_group_belief_hidden_when_dep_not_listed(
    tmp_path: Path, monkeypatch
) -> None:
    """shared:bar belief on peer A is NOT visible to B when B has no shared:bar dep."""
    peer_path = tmp_path / "peerA.db"
    _seed_peer(
        peer_path,
        [_belief("a-shared-bar", "dynamic programming overlap", scope="shared:bar")],
    )
    # local DB does NOT declare shared:bar
    _wire_peers(
        tmp_path,
        [{"name": "peerA", "path": str(peer_path)}],
        monkeypatch,
    )

    local = MemoryStore(str(tmp_path / "local.db"))
    try:
        hits = local.search_peer_beliefs("dynamic programming")
        assert len(hits) == 0
    finally:
        local.close()


# ---------------------------------------------------------------------------
# 6. per-scope counts
# ---------------------------------------------------------------------------

def test_count_beliefs_by_scope_empty() -> None:
    """count_beliefs_by_scope on empty store returns {}."""
    store = MemoryStore(":memory:")
    try:
        assert store.count_beliefs_by_scope() == {}
    finally:
        store.close()


def test_count_beliefs_by_scope_mixed() -> None:
    """count_beliefs_by_scope returns correct per-scope distribution."""
    store = MemoryStore(":memory:")
    try:
        store.insert_belief(_belief("p1", "alpha", scope="project"))
        store.insert_belief(_belief("p2", "beta", scope="project"))
        store.insert_belief(_belief("g1", "gamma", scope="global"))
        store.insert_belief(_belief("s1", "delta", scope="shared:team"))
        counts = store.count_beliefs_by_scope()
        assert counts["project"] == 2
        assert counts["global"] == 1
        assert counts["shared:team"] == 1
        assert len(counts) == 3
    finally:
        store.close()
