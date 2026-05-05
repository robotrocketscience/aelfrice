"""Tests for `aelf delete` CLI subcommand (#440).

Unit tests use the in-process `main(argv=..., out=...)` harness.
Integration tests open a real MemoryStore and verify feedback_history rows
and edge cascade behaviour.
Each test is isolated via the `isolated_db` fixture (AELFRICE_DB envvar).
"""
from __future__ import annotations

import io
from pathlib import Path

import pytest

from aelfrice.cli import main
from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_SUPPORTS,
    LOCK_NONE,
    LOCK_USER,
    ORIGIN_AGENT_INFERRED,
    Belief,
    Edge,
)
from aelfrice.store import MemoryStore


@pytest.fixture(autouse=True)
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Every test gets its own throwaway DB at <tmp>/aelf.db."""
    p = tmp_path / "aelf.db"
    monkeypatch.setenv("AELFRICE_DB", str(p))
    return p


def _run(*argv: str) -> tuple[int, str]:
    buf = io.StringIO()
    code = main(argv=list(argv), out=buf)
    return code, buf.getvalue()


def _seed_belief(
    db: Path,
    content: str,
    bid: str = "aabbccddeeff1122",
    lock_level: str = LOCK_NONE,
) -> str:
    """Insert one belief into the store and return its id."""
    s = MemoryStore(str(db))
    try:
        s.insert_belief(Belief(
            id=bid,
            content=content,
            content_hash="testhash_" + bid[:4],
            alpha=1.0,
            beta=1.0,
            type=BELIEF_FACTUAL,
            lock_level=lock_level,
            locked_at=None,
            demotion_pressure=0,
            created_at="2026-05-05T00:00:00Z",
            last_retrieved_at=None,
            origin=ORIGIN_AGENT_INFERRED,
        ))
    finally:
        s.close()
    return bid


# --- unit: belief not found --------------------------------------------------


def test_delete_not_found_exits_one(isolated_db: Path) -> None:
    code, _ = _run("delete", "doesnotexist", "--yes")
    assert code == 1


def test_delete_not_found_message(isolated_db: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _run("delete", "doesnotexist", "--yes")
    captured = capsys.readouterr()
    assert "belief not found: doesnotexist" in captured.err


# --- unit: locked belief without --force -------------------------------------


def test_delete_locked_without_force_exits_one(isolated_db: Path) -> None:
    bid = _seed_belief(isolated_db, "do not delete me", lock_level=LOCK_USER)
    code, _ = _run("delete", bid, "--yes")
    assert code == 1


def test_delete_locked_without_force_message(
    isolated_db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bid = _seed_belief(isolated_db, "do not delete me", lock_level=LOCK_USER)
    _run("delete", bid, "--yes")
    captured = capsys.readouterr()
    assert "belief is locked (lock_level=user)" in captured.err
    assert "--force" in captured.err


def test_delete_locked_belief_is_not_deleted(isolated_db: Path) -> None:
    bid = _seed_belief(isolated_db, "still alive", lock_level=LOCK_USER)
    _run("delete", bid, "--yes")
    s = MemoryStore(str(isolated_db))
    try:
        assert s.get_belief(bid) is not None
    finally:
        s.close()


# --- unit: confirmation prompt — mismatch ------------------------------------


def test_delete_prompt_mismatch_exits_one(
    isolated_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bid = _seed_belief(isolated_db, "some content")
    # Provide wrong prefix
    monkeypatch.setattr("builtins.input", lambda _: "wronginp")
    code, _ = _run("delete", bid)
    assert code == 1


def test_delete_prompt_mismatch_message(
    isolated_db: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    bid = _seed_belief(isolated_db, "some content")
    monkeypatch.setattr("builtins.input", lambda _: "wronginp")
    _run("delete", bid)
    captured = capsys.readouterr()
    assert "aborted: confirmation did not match" in captured.err


def test_delete_prompt_mismatch_belief_survives(
    isolated_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bid = _seed_belief(isolated_db, "should survive mismatch")
    monkeypatch.setattr("builtins.input", lambda _: "wronginp")
    _run("delete", bid)
    s = MemoryStore(str(isolated_db))
    try:
        assert s.get_belief(bid) is not None
    finally:
        s.close()


def test_delete_prompt_empty_input_exits_one(
    isolated_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bid = _seed_belief(isolated_db, "empty input test")
    monkeypatch.setattr("builtins.input", lambda _: "")
    code, _ = _run("delete", bid)
    assert code == 1


# --- unit: confirmation prompt — match (happy path) --------------------------


def test_delete_prompt_match_exits_zero(
    isolated_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bid = _seed_belief(isolated_db, "belief to delete via prompt")
    monkeypatch.setattr("builtins.input", lambda _: bid[:8])
    code, out = _run("delete", bid)
    assert code == 0
    assert f"deleted: {bid}" in out


def test_delete_prompt_match_belief_gone(
    isolated_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bid = _seed_belief(isolated_db, "gone after prompt match")
    monkeypatch.setattr("builtins.input", lambda _: bid[:8])
    _run("delete", bid)
    s = MemoryStore(str(isolated_db))
    try:
        assert s.get_belief(bid) is None
    finally:
        s.close()


# --- unit: --yes path --------------------------------------------------------


def test_delete_yes_exits_zero(isolated_db: Path) -> None:
    bid = _seed_belief(isolated_db, "delete with --yes")
    code, out = _run("delete", bid, "--yes")
    assert code == 0
    assert f"deleted: {bid}" in out


def test_delete_yes_belief_gone(isolated_db: Path) -> None:
    bid = _seed_belief(isolated_db, "gone after --yes")
    _run("delete", bid, "--yes")
    s = MemoryStore(str(isolated_db))
    try:
        assert s.get_belief(bid) is None
    finally:
        s.close()


# --- unit: --force path (locked belief) --------------------------------------


def test_delete_force_yes_locked_belief_exits_zero(isolated_db: Path) -> None:
    bid = _seed_belief(isolated_db, "locked but force-deleted", lock_level=LOCK_USER)
    code, out = _run("delete", bid, "--force", "--yes")
    assert code == 0
    assert f"deleted: {bid}" in out


def test_delete_force_yes_locked_belief_gone(isolated_db: Path) -> None:
    bid = _seed_belief(isolated_db, "locked, gone after --force --yes", lock_level=LOCK_USER)
    _run("delete", bid, "--force", "--yes")
    s = MemoryStore(str(isolated_db))
    try:
        assert s.get_belief(bid) is None
    finally:
        s.close()


def test_delete_force_without_yes_still_prompts(
    isolated_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--force alone does not skip the confirmation prompt."""
    bid = _seed_belief(isolated_db, "locked needs prompt", lock_level=LOCK_USER)
    monkeypatch.setattr("builtins.input", lambda _: bid[:8])
    code, out = _run("delete", bid, "--force")
    assert code == 0
    assert f"deleted: {bid}" in out


# --- integration: audit row written to feedback_history ----------------------


def test_delete_writes_audit_row(isolated_db: Path) -> None:
    bid = _seed_belief(isolated_db, "audit row test")
    _run("delete", bid, "--yes")
    s = MemoryStore(str(isolated_db))
    try:
        events = s.list_feedback_events(belief_id=bid)
    finally:
        s.close()
    assert len(events) == 1
    assert events[0].source == "user_deleted"
    assert events[0].valence == -1.0
    assert events[0].belief_id == bid


def test_delete_force_writes_force_source(isolated_db: Path) -> None:
    bid = _seed_belief(isolated_db, "force audit row", lock_level=LOCK_USER)
    _run("delete", bid, "--force", "--yes")
    s = MemoryStore(str(isolated_db))
    try:
        events = s.list_feedback_events(belief_id=bid)
    finally:
        s.close()
    assert len(events) == 1
    assert events[0].source == "user_deleted_force"
    assert events[0].valence == -1.0


def test_delete_audit_row_survives_cascade(isolated_db: Path) -> None:
    """feedback_history has no FK to beliefs; orphan row persists after delete."""
    bid = _seed_belief(isolated_db, "orphan audit row")
    _run("delete", bid, "--yes")
    s = MemoryStore(str(isolated_db))
    try:
        # Belief is gone.
        assert s.get_belief(bid) is None
        # But the audit row is still there (orphan-tolerant table).
        events = s.list_feedback_events(belief_id=bid)
        assert len(events) == 1
        assert events[0].source == "user_deleted"
    finally:
        s.close()


# --- integration: cascade removes edges --------------------------------------


def test_delete_cascades_outgoing_edges(isolated_db: Path) -> None:
    """Deleting src belief removes its outgoing edges."""
    src_id = "src0001aabbccdd00"
    dst_id = "dst0001aabbccdd00"
    s = MemoryStore(str(isolated_db))
    try:
        s.insert_belief(Belief(
            id=src_id, content="source belief",
            content_hash="hash_src", alpha=1.0, beta=1.0,
            type=BELIEF_FACTUAL, lock_level=LOCK_NONE,
            locked_at=None, demotion_pressure=0,
            created_at="2026-05-05T00:00:00Z",
            last_retrieved_at=None, origin=ORIGIN_AGENT_INFERRED,
        ))
        s.insert_belief(Belief(
            id=dst_id, content="destination belief",
            content_hash="hash_dst", alpha=1.0, beta=1.0,
            type=BELIEF_FACTUAL, lock_level=LOCK_NONE,
            locked_at=None, demotion_pressure=0,
            created_at="2026-05-05T00:00:00Z",
            last_retrieved_at=None, origin=ORIGIN_AGENT_INFERRED,
        ))
        s.insert_edge(Edge(src=src_id, dst=dst_id, type=EDGE_SUPPORTS, weight=0.8))
    finally:
        s.close()

    _run("delete", src_id, "--yes")

    s = MemoryStore(str(isolated_db))
    try:
        assert s.get_belief(src_id) is None
        assert s.get_edge(src_id, dst_id, EDGE_SUPPORTS) is None
        # dst belief is intact.
        assert s.get_belief(dst_id) is not None
    finally:
        s.close()


def test_delete_cascades_incoming_edges(isolated_db: Path) -> None:
    """Deleting dst belief removes its incoming edges."""
    src_id = "src0002aabbccdd00"
    dst_id = "dst0002aabbccdd00"
    s = MemoryStore(str(isolated_db))
    try:
        s.insert_belief(Belief(
            id=src_id, content="source belief for incoming test",
            content_hash="hash_src2", alpha=1.0, beta=1.0,
            type=BELIEF_FACTUAL, lock_level=LOCK_NONE,
            locked_at=None, demotion_pressure=0,
            created_at="2026-05-05T00:00:00Z",
            last_retrieved_at=None, origin=ORIGIN_AGENT_INFERRED,
        ))
        s.insert_belief(Belief(
            id=dst_id, content="destination belief for incoming test",
            content_hash="hash_dst2", alpha=1.0, beta=1.0,
            type=BELIEF_FACTUAL, lock_level=LOCK_NONE,
            locked_at=None, demotion_pressure=0,
            created_at="2026-05-05T00:00:00Z",
            last_retrieved_at=None, origin=ORIGIN_AGENT_INFERRED,
        ))
        s.insert_edge(Edge(src=src_id, dst=dst_id, type=EDGE_SUPPORTS, weight=0.8))
    finally:
        s.close()

    _run("delete", dst_id, "--yes")

    s = MemoryStore(str(isolated_db))
    try:
        assert s.get_belief(dst_id) is None
        assert s.get_edge(src_id, dst_id, EDGE_SUPPORTS) is None
        # src belief is intact.
        assert s.get_belief(src_id) is not None
    finally:
        s.close()


# --- unit: output string exactness -------------------------------------------


def test_delete_success_output_format(isolated_db: Path) -> None:
    bid = _seed_belief(isolated_db, "output format check")
    code, out = _run("delete", bid, "--yes")
    assert code == 0
    assert out.strip() == f"deleted: {bid}"
