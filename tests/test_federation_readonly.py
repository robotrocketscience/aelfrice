"""Two-scope read-only federation acceptance tests (#655).

Each test creates a "project A" peer DB on disk, populates it with at
least one belief, then opens a fresh "project B" store and points it at
A via a tmp knowledge_deps.json (via the AELFRICE_KNOWLEDGE_DEPS env
override so the test doesn't depend on git toplevel).

Covers the acceptance bullets from #655:

* peer belief surfaces in B's `search_peer_beliefs` (overlay)
* feedback/lock/delete/unlock/promote on a foreign id raise
  ForeignBeliefError
* missing peer DB warns rather than crashes
* aelf health JSON reports peers with reachability + scope_id
* no regression on single-DB workflows (zero peers configured)

Updated for #688 scope field: beliefs that should be visible to peers
must use scope='global' (or 'shared:<name>'). The factory default is
now 'project' per the column default, but tests that assert peer
visibility pass scope='global' explicitly.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aelfrice.federation import ForeignBeliefError
from aelfrice.feedback import apply_feedback
from aelfrice.models import (
    LOCK_NONE,
    LOCK_USER,
    ORIGIN_AGENT_INFERRED,
    Belief,
)
from aelfrice.promotion import promote, unlock
from aelfrice.store import MemoryStore


def _belief(
    id_: str,
    content: str,
    *,
    lock_level: str = LOCK_NONE,
    origin: str = ORIGIN_AGENT_INFERRED,
    scope: str = "global",
) -> Belief:
    """Construct a test Belief.

    scope defaults to 'global' so existing tests that assert peer
    visibility continue to pass after #688 introduced the scope filter
    on search_peer_beliefs. Tests that exercise the project/global
    distinction should pass scope= explicitly.
    """
    return Belief(
        id=id_,
        content=content,
        content_hash=id_ + "-hash",
        alpha=1.0,
        beta=1.0,
        type="factual",
        lock_level=lock_level,
        locked_at=None,
        created_at="2026-05-11T00:00:00+00:00",
        last_retrieved_at=None,
        session_id=None,
        origin=origin,
        scope=scope,
    )


def _seed_peer(path: Path, beliefs: list[Belief]) -> None:
    """Create a peer DB at `path` and populate it with `beliefs`."""
    store = MemoryStore(str(path))
    try:
        for b in beliefs:
            store.insert_belief(b)
    finally:
        store.close()


def _wire_peer(tmp_path: Path, peer_path: Path, monkeypatch) -> Path:
    """Materialise knowledge_deps.json and point the env override at it."""
    deps_file = tmp_path / "knowledge_deps.json"
    deps_file.write_text(
        json.dumps(
            {
                "version": 1,
                "deps": [
                    {"name": "peerA", "path": str(peer_path)},
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AELFRICE_KNOWLEDGE_DEPS", str(deps_file))
    return deps_file


def test_two_scope_smoke_peer_belief_surfaces_in_local_search(
    tmp_path: Path, monkeypatch
):
    """Acceptance bullet 1: project B search surfaces project A's belief."""
    peer_path = tmp_path / "peerA.db"
    _seed_peer(
        peer_path,
        [_belief("peer-belief-1", "the cat sat on the mat")],
    )
    _wire_peer(tmp_path, peer_path, monkeypatch)

    local = MemoryStore(str(tmp_path / "local.db"))
    try:
        hits = local.search_peer_beliefs("cat mat")
        assert len(hits) == 1
        belief, peer_name, score = hits[0]
        assert belief.id == "peer-belief-1"
        assert belief.content == "the cat sat on the mat"
        assert peer_name == "peerA"
        # bm25 returns a non-positive score
        assert score <= 0.0
    finally:
        local.close()


def test_foreign_owner_resolves_for_peer_only_belief(
    tmp_path: Path, monkeypatch
):
    peer_path = tmp_path / "peerA.db"
    _seed_peer(peer_path, [_belief("foreign-1", "owned by A")])
    _wire_peer(tmp_path, peer_path, monkeypatch)

    local = MemoryStore(str(tmp_path / "local.db"))
    try:
        assert local.find_foreign_owner("foreign-1") == "peerA"
        # A purely-local belief is not "foreign".
        local.insert_belief(_belief("local-1", "owned by B"))
        assert local.find_foreign_owner("local-1") is None
        # An id that doesn't exist anywhere returns None (caller
        # handles "not found" via its own get_belief path).
        assert local.find_foreign_owner("never-existed") is None
    finally:
        local.close()


def test_apply_feedback_rejects_foreign_belief(tmp_path: Path, monkeypatch):
    """Acceptance bullet 2: aelf feedback <foreign_id> returns a clean error."""
    peer_path = tmp_path / "peerA.db"
    _seed_peer(peer_path, [_belief("foreign-1", "foreign content")])
    _wire_peer(tmp_path, peer_path, monkeypatch)

    local = MemoryStore(str(tmp_path / "local.db"))
    try:
        with pytest.raises(ForeignBeliefError) as exc_info:
            apply_feedback(
                store=local,
                belief_id="foreign-1",
                valence=1.0,
                source="test",
            )
        assert exc_info.value.owning_scope == "peerA"
        assert exc_info.value.belief_id == "foreign-1"
        # Subclass of ValueError so existing callers keep their handler.
        assert isinstance(exc_info.value, ValueError)
    finally:
        local.close()


def test_unlock_rejects_foreign_belief(tmp_path: Path, monkeypatch):
    """Acceptance bullet 3: aelf unlock <foreign_id> ditto."""
    peer_path = tmp_path / "peerA.db"
    _seed_peer(
        peer_path,
        [_belief("foreign-locked", "locked over there", lock_level=LOCK_USER)],
    )
    _wire_peer(tmp_path, peer_path, monkeypatch)

    local = MemoryStore(str(tmp_path / "local.db"))
    try:
        with pytest.raises(ForeignBeliefError) as exc_info:
            unlock(local, "foreign-locked")
        assert exc_info.value.owning_scope == "peerA"
    finally:
        local.close()


def test_promote_rejects_foreign_belief(tmp_path: Path, monkeypatch):
    peer_path = tmp_path / "peerA.db"
    _seed_peer(
        peer_path,
        [_belief("foreign-agent", "to be promoted elsewhere")],
    )
    _wire_peer(tmp_path, peer_path, monkeypatch)

    local = MemoryStore(str(tmp_path / "local.db"))
    try:
        with pytest.raises(ForeignBeliefError):
            promote(local, "foreign-agent")
    finally:
        local.close()


def test_missing_peer_warns_does_not_crash(tmp_path: Path, monkeypatch):
    """Acceptance bullet 4: missing peer file → warning + skip, not crash."""
    deps_file = tmp_path / "knowledge_deps.json"
    deps_file.write_text(
        json.dumps(
            {
                "version": 1,
                "deps": [
                    {"name": "ghost", "path": "/no/such/aelfrice.db"},
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AELFRICE_KNOWLEDGE_DEPS", str(deps_file))

    local = MemoryStore(str(tmp_path / "local.db"))
    try:
        # Search must succeed with the missing peer skipped, not crash.
        assert local.search_peer_beliefs("anything") == []
        # peer_health surfaces the unreachable peer.
        snapshot = local.peer_health()
        assert len(snapshot) == 1
        assert snapshot[0]["name"] == "ghost"
        assert snapshot[0]["reachable"] is False
        assert snapshot[0]["scope_id"] is None
        # find_foreign_owner doesn't blow up either.
        assert local.find_foreign_owner("anything") is None
    finally:
        local.close()


def test_peer_health_reports_scope_id_when_reachable(
    tmp_path: Path, monkeypatch
):
    """Acceptance bullet 5: aelf health reports peer DBs + reachability."""
    peer_path = tmp_path / "peerA.db"
    _seed_peer(peer_path, [_belief("foreign-1", "x")])
    # Capture peer's scope_id for cross-check
    peer_store = MemoryStore(str(peer_path))
    expected_scope = peer_store.local_scope_id
    peer_store.close()
    _wire_peer(tmp_path, peer_path, monkeypatch)

    local = MemoryStore(str(tmp_path / "local.db"))
    try:
        snapshot = local.peer_health()
        assert len(snapshot) == 1
        entry = snapshot[0]
        assert entry["name"] == "peerA"
        assert entry["reachable"] is True
        assert entry["scope_id"] == expected_scope
    finally:
        local.close()


def test_no_regression_when_no_peers_configured(tmp_path: Path, monkeypatch):
    """Acceptance bullet 6: single-DB workflows unchanged when no peers set."""
    # Explicit unset — guards against test-process env leakage.
    monkeypatch.delenv("AELFRICE_KNOWLEDGE_DEPS", raising=False)

    local = MemoryStore(str(tmp_path / "local.db"))
    try:
        assert local.peer_deps() == []
        assert local.peer_health() == []
        assert local.search_peer_beliefs("anything") == []
        # Local feedback works exactly as before.
        local.insert_belief(_belief("local-1", "local content"))
        result = apply_feedback(
            store=local,
            belief_id="local-1",
            valence=1.0,
            source="test",
        )
        assert result.new_alpha > result.prior_alpha
    finally:
        local.close()


def test_peer_search_skips_non_aelfrice_sqlite_file(
    tmp_path: Path, monkeypatch
):
    """An attached SQLite file without aelfrice's schema is skipped silently.

    Federation must tolerate arbitrary user input at the peer-path
    layer without crashing the host query.
    """
    import sqlite3

    peer_path = tmp_path / "alien.db"
    alien = sqlite3.connect(str(peer_path))
    alien.execute("CREATE TABLE notes (text TEXT)")
    alien.execute("INSERT INTO notes VALUES ('not an aelfrice db')")
    alien.commit()
    alien.close()
    _wire_peer(tmp_path, peer_path, monkeypatch)

    local = MemoryStore(str(tmp_path / "local.db"))
    try:
        # No crash; just empty results for the alien peer.
        assert local.search_peer_beliefs("anything") == []
        # Health still reports the peer as reachable (file exists)
        # but with no scope_id (no schema_meta row).
        snapshot = local.peer_health()
        assert snapshot[0]["reachable"] is True
        assert snapshot[0]["scope_id"] is None
    finally:
        local.close()


def test_cli_search_annotates_peer_hits_with_scope_tag(
    tmp_path: Path, monkeypatch
):
    """aelf search surfaces peer hits prefixed by [scope:<name>]."""
    import io

    from aelfrice.cli import main

    peer_path = tmp_path / "peerA.db"
    _seed_peer(
        peer_path,
        [_belief("peer-belief-search", "the cat sat on the mat")],
    )
    _wire_peer(tmp_path, peer_path, monkeypatch)
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "local.db"))
    monkeypatch.setenv("AELF_NO_UPDATE_CHECK", "1")

    buf = io.StringIO()
    rc = main(["search", "cat mat"], out=buf)
    assert rc == 0
    output = buf.getvalue()
    assert "[scope:peerA]" in output
    assert "peer-belief-search" in output


def test_cli_delete_rejects_foreign_id(tmp_path: Path, monkeypatch, capsys):
    """aelf delete <foreign_id> exits 1 with a foreign-scope message."""
    import io

    from aelfrice.cli import main

    peer_path = tmp_path / "peerA.db"
    _seed_peer(peer_path, [_belief("foreign-delete", "do not delete me")])
    _wire_peer(tmp_path, peer_path, monkeypatch)
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "local.db"))
    monkeypatch.setenv("AELF_NO_UPDATE_CHECK", "1")

    buf = io.StringIO()
    rc = main(["delete", "foreign-delete", "--yes"], out=buf)
    assert rc == 1
    captured = capsys.readouterr()
    assert "peerA" in captured.err
    assert "foreign" in captured.err.lower()


def test_local_id_wins_over_peer_id_in_owner_lookup(
    tmp_path: Path, monkeypatch
):
    """If the same belief id is in both, local ownership takes precedence.

    Avoids a stray peer collision raising ForeignBeliefError on a
    legitimately-local belief.
    """
    peer_path = tmp_path / "peerA.db"
    _seed_peer(peer_path, [_belief("collision", "from peer")])
    _wire_peer(tmp_path, peer_path, monkeypatch)

    local = MemoryStore(str(tmp_path / "local.db"))
    try:
        local.insert_belief(_belief("collision", "from local"))
        assert local.find_foreign_owner("collision") is None
        # And feedback works.
        result = apply_feedback(
            store=local, belief_id="collision", valence=1.0, source="test",
        )
        assert result.new_alpha > result.prior_alpha
    finally:
        local.close()
