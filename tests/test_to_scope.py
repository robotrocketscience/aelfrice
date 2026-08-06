"""Tests for the --to-scope flag on aelf promote / aelf demote (#689).

Covers:
* CLI happy path: promote --to-scope flips scope, writes audit row
* CLI happy path: demote --to-scope flips scope, writes audit row
* promote --to-scope on already-validated belief: only scope changes
* promote --to-scope + agent_inferred: both origin flip and scope flip
* --to-scope foo!bar rejected at parse time (argparse error, exit 2)
* Foreign belief id raises ForeignBeliefError, exits nonzero
* scope unchanged (already == target) reports no-op
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from aelfrice.cli import main
from aelfrice.models import (
    BELIEF_FACTUAL,
    BELIEF_SCOPE_GLOBAL,
    BELIEF_SCOPE_PROJECT,
    LOCK_NONE,
    LOCK_USER,
    ORIGIN_AGENT_INFERRED,
    ORIGIN_USER_VALIDATED,
    Belief,
)
from aelfrice.store import MemoryStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _belief(
    id_: str,
    content: str,
    *,
    origin: str = ORIGIN_AGENT_INFERRED,
    lock_level: str = LOCK_NONE,
    scope: str = BELIEF_SCOPE_PROJECT,
) -> Belief:
    return Belief(
        id=id_,
        content=content,
        content_hash=id_ + "-hash",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=lock_level,
        locked_at="2026-05-11T00:00:00Z" if lock_level == LOCK_USER else None,
        created_at="2026-05-11T00:00:00Z",
        last_retrieved_at=None,
        origin=origin,
        scope=scope,
    )


@pytest.fixture(autouse=True)
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    p = tmp_path / "aelf.db"
    monkeypatch.setenv("AELFRICE_DB", str(p))
    return p


def _run(*argv: str) -> tuple[int, str]:
    buf = io.StringIO()
    code = main(argv=list(argv), out=buf)
    return code, buf.getvalue()


def _seed(db: Path, belief: Belief) -> None:
    s = MemoryStore(str(db))
    try:
        s.insert_belief(belief)
    finally:
        s.close()


def _get(db: Path, bid: str) -> Belief | None:
    s = MemoryStore(str(db))
    try:
        return s.get_belief(bid)
    finally:
        s.close()


def _audit_events(db: Path, bid: str) -> list[dict]:
    s = MemoryStore(str(db))
    try:
        events = s.list_feedback_events(belief_id=bid)
        return [
            {"source": e.source, "valence": e.valence}
            for e in events
        ]
    finally:
        s.close()


# ---------------------------------------------------------------------------
# CLI: promote --to-scope
# ---------------------------------------------------------------------------

def test_promote_to_scope_flips_scope_field(isolated_db: Path) -> None:
    """aelf promote <id> --to-scope global flips the scope field."""
    _seed(isolated_db, _belief("b001", "test belief", scope=BELIEF_SCOPE_PROJECT))
    code, out = _run("promote", "b001", "--to-scope", "global")
    assert code == 0, f"unexpected exit: {out}"
    b = _get(isolated_db, "b001")
    assert b is not None
    assert b.scope == BELIEF_SCOPE_GLOBAL


def test_promote_to_scope_writes_audit_row(isolated_db: Path) -> None:
    """aelf promote --to-scope writes a zero-valence 'scope:project->global' audit row."""
    _seed(isolated_db, _belief("b002", "audit row test", scope=BELIEF_SCOPE_PROJECT))
    _run("promote", "b002", "--to-scope", "global")
    events = _audit_events(isolated_db, "b002")
    scope_events = [e for e in events if e["source"].startswith("scope:")]
    assert len(scope_events) == 1
    assert scope_events[0]["source"] == "scope:project->global"
    assert scope_events[0]["valence"] == 0.0


def test_promote_to_scope_agent_inferred_also_flips_origin(
    isolated_db: Path,
) -> None:
    """When belief is agent_inferred, promote --to-scope also flips origin."""
    _seed(
        isolated_db,
        _belief("b003", "both flip", origin=ORIGIN_AGENT_INFERRED, scope=BELIEF_SCOPE_PROJECT),
    )
    code, out = _run("promote", "b003", "--to-scope", "global")
    assert code == 0, f"unexpected exit: {out}"
    b = _get(isolated_db, "b003")
    assert b is not None
    assert b.scope == BELIEF_SCOPE_GLOBAL
    assert b.origin == ORIGIN_USER_VALIDATED


def test_promote_to_scope_user_validated_only_flips_scope(
    isolated_db: Path,
) -> None:
    """When belief is already user_validated, promote --to-scope only flips scope."""
    _seed(
        isolated_db,
        _belief("b004", "already validated", origin=ORIGIN_USER_VALIDATED, scope=BELIEF_SCOPE_PROJECT),
    )
    code, out = _run("promote", "b004", "--to-scope", "global")
    assert code == 0, f"unexpected exit: {out}"
    b = _get(isolated_db, "b004")
    assert b is not None
    assert b.scope == BELIEF_SCOPE_GLOBAL
    # origin remains user_validated; no origin-flip audit row
    events = _audit_events(isolated_db, "b004")
    origin_events = [e for e in events if e["source"].startswith("promotion:")]
    assert len(origin_events) == 0, (
        "expected no promotion audit row when belief was already user_validated"
    )


def test_promote_to_scope_invalid_rejected_at_parse_time(
    isolated_db: Path,
) -> None:
    """--to-scope foo!bar should be rejected (invalid scope)."""
    _seed(isolated_db, _belief("b005", "scope test"))
    code, _ = _run("promote", "b005", "--to-scope", "foo!bar")
    assert code != 0


def test_promote_to_scope_shared_name(isolated_db: Path) -> None:
    """aelf promote --to-scope shared:team-a is valid and accepted."""
    _seed(isolated_db, _belief("b006", "shared scope test", scope=BELIEF_SCOPE_PROJECT))
    code, out = _run("promote", "b006", "--to-scope", "shared:team-a")
    assert code == 0, f"unexpected exit: {out}"
    b = _get(isolated_db, "b006")
    assert b is not None
    assert b.scope == "shared:team-a"


def test_promote_to_scope_unchanged_is_noop(isolated_db: Path) -> None:
    """--to-scope to the same value reports unchanged and writes no audit row."""
    _seed(isolated_db, _belief("b007", "already global", scope=BELIEF_SCOPE_GLOBAL))
    code, out = _run("promote", "b007", "--to-scope", "global")
    assert code == 0
    assert "unchanged" in out
    events = _audit_events(isolated_db, "b007")
    scope_events = [e for e in events if e["source"].startswith("scope:")]
    assert len(scope_events) == 0


# ---------------------------------------------------------------------------
# CLI: demote --to-scope
# ---------------------------------------------------------------------------

def test_demote_to_scope_flips_scope_field(isolated_db: Path) -> None:
    """aelf demote <id> --to-scope project flips scope from global to project."""
    _seed(isolated_db, _belief("b010", "demote scope", scope=BELIEF_SCOPE_GLOBAL))
    code, out = _run("demote", "b010", "--to-scope", "project")
    assert code == 0, f"unexpected exit: {out}"
    b = _get(isolated_db, "b010")
    assert b is not None
    assert b.scope == BELIEF_SCOPE_PROJECT


def test_demote_to_scope_writes_audit_row(isolated_db: Path) -> None:
    """aelf demote --to-scope writes a zero-valence 'scope:global->project' audit row."""
    _seed(isolated_db, _belief("b011", "audit row demote", scope=BELIEF_SCOPE_GLOBAL))
    _run("demote", "b011", "--to-scope", "project")
    events = _audit_events(isolated_db, "b011")
    scope_events = [e for e in events if e["source"].startswith("scope:")]
    assert len(scope_events) == 1
    assert scope_events[0]["source"] == "scope:global->project"
    assert scope_events[0]["valence"] == 0.0


def test_demote_to_scope_invalid_rejected(isolated_db: Path) -> None:
    """--to-scope BADVAL rejected with nonzero exit."""
    _seed(isolated_db, _belief("b012", "invalid scope demote"))
    code, _ = _run("demote", "b012", "--to-scope", "INVALID SCOPE!")
    assert code != 0


# ---------------------------------------------------------------------------
# CLI: foreign-id rejection
# ---------------------------------------------------------------------------

def test_promote_to_scope_foreign_id_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_db: Path,
) -> None:
    """promote --to-scope on a foreign belief id exits nonzero."""
    # Create a peer DB with a belief.
    peer_db = tmp_path / "peer.db"
    peer_store = MemoryStore(str(peer_db))
    try:
        peer_store.insert_belief(_belief("foreign-01", "peer belief", scope="global"))
    finally:
        peer_store.close()

    # Point local store at the peer via knowledge_deps.
    deps_file = tmp_path / "knowledge_deps.json"
    deps_file.write_text(
        json.dumps({
            "version": 1,
            "deps": [{"name": "peer", "path": str(peer_db)}],
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("AELFRICE_KNOWLEDGE_DEPS", str(deps_file))

    # Bootstrap local DB (open/close creates schema).
    local_store = MemoryStore(str(isolated_db))
    local_store.close()

    code, err_out = _run("promote", "foreign-01", "--to-scope", "project")
    assert code != 0, "expected nonzero exit for foreign belief id"
