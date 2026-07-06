"""Tests for `aelf retire` / `aelf restore` — reversible soft-delete (#1081).

`retire` is the gentle sibling of `delete`: it sets `valid_to` (the belief
drops out of retrieval/search) but preserves the evidence trail (edges,
entities, corroborations). `restore` clears `valid_to` and re-indexes the
belief for FTS search. Together they make curation reversible.

Unit tests use the in-process `main(argv=..., out=...)` harness; integration
tests open a real MemoryStore and assert `valid_to`, FTS membership,
feedback_history rows, and the edge-preservation contrast with `delete`.
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
            created_at="2026-05-05T00:00:00Z",
            last_retrieved_at=None,
            origin=ORIGIN_AGENT_INFERRED,
        ))
    finally:
        s.close()
    return bid


def _valid_to(db: Path, bid: str) -> str | None:
    s = MemoryStore(str(db))
    try:
        b = s.get_belief(bid)
        return None if b is None else b.valid_to
    finally:
        s.close()


def _in_fts(db: Path, bid: str) -> bool:
    s = MemoryStore(str(db))
    try:
        row = s._conn.execute(  # noqa: SLF001 — direct FTS membership probe
            "SELECT 1 FROM beliefs_fts WHERE id = ?", (bid,)
        ).fetchone()
        return row is not None
    finally:
        s.close()


# --- retire: not found -------------------------------------------------------


def test_retire_not_found_exits_one(isolated_db: Path) -> None:
    code, _ = _run("retire", "doesnotexist")
    assert code == 1


def test_retire_not_found_message(
    isolated_db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _run("retire", "doesnotexist")
    captured = capsys.readouterr()
    assert "belief not found: doesnotexist" in captured.err


# --- retire: locked without --force ------------------------------------------


def test_retire_locked_without_force_exits_one(isolated_db: Path) -> None:
    bid = _seed_belief(isolated_db, "do not retire me", lock_level=LOCK_USER)
    code, _ = _run("retire", bid)
    assert code == 1


def test_retire_locked_without_force_message(
    isolated_db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bid = _seed_belief(isolated_db, "do not retire me", lock_level=LOCK_USER)
    _run("retire", bid)
    captured = capsys.readouterr()
    assert "belief is locked (lock_level=user)" in captured.err
    assert "--force" in captured.err


def test_retire_locked_without_force_belief_stays_active(isolated_db: Path) -> None:
    bid = _seed_belief(isolated_db, "still active", lock_level=LOCK_USER)
    _run("retire", bid)
    assert _valid_to(isolated_db, bid) is None


# --- retire: happy path ------------------------------------------------------


def test_retire_exits_zero_and_reports(isolated_db: Path) -> None:
    bid = _seed_belief(isolated_db, "retire me")
    code, out = _run("retire", bid)
    assert code == 0
    assert f"retired: {bid}" in out
    assert "restore" in out  # points at the reversal


def test_retire_sets_valid_to(isolated_db: Path) -> None:
    bid = _seed_belief(isolated_db, "soft-deleted")
    _run("retire", bid)
    assert _valid_to(isolated_db, bid) is not None


def test_retire_drops_from_active_list(isolated_db: Path) -> None:
    bid = _seed_belief(isolated_db, "gone from active")
    _run("retire", bid)
    s = MemoryStore(str(isolated_db))
    try:
        active_ids = {b.id for b in s.list_active_beliefs()}
    finally:
        s.close()
    assert bid not in active_ids


def test_retire_prunes_fts_row(isolated_db: Path) -> None:
    bid = _seed_belief(isolated_db, "searchable widget")
    assert _in_fts(isolated_db, bid) is True
    _run("retire", bid)
    assert _in_fts(isolated_db, bid) is False


def test_retire_writes_audit_row(isolated_db: Path) -> None:
    bid = _seed_belief(isolated_db, "audit on retire")
    _run("retire", bid)
    s = MemoryStore(str(isolated_db))
    try:
        events = s.list_feedback_events(belief_id=bid)
    finally:
        s.close()
    assert len(events) == 1
    assert events[0].source == "user_retired"
    assert events[0].valence == -1.0


def test_retire_force_locked_exits_zero(isolated_db: Path) -> None:
    bid = _seed_belief(isolated_db, "locked but retired", lock_level=LOCK_USER)
    code, out = _run("retire", bid, "--force")
    assert code == 0
    assert f"retired: {bid}" in out


def test_retire_force_writes_force_source(isolated_db: Path) -> None:
    bid = _seed_belief(isolated_db, "force audit", lock_level=LOCK_USER)
    _run("retire", bid, "--force")
    s = MemoryStore(str(isolated_db))
    try:
        events = s.list_feedback_events(belief_id=bid)
    finally:
        s.close()
    assert events[0].source == "user_retired_force"


def test_retire_already_retired_is_noop(isolated_db: Path) -> None:
    bid = _seed_belief(isolated_db, "retire twice")
    _run("retire", bid)
    code, out = _run("retire", bid)
    assert code == 0
    assert f"already retired: {bid}" in out
    # No second audit row.
    s = MemoryStore(str(isolated_db))
    try:
        events = s.list_feedback_events(belief_id=bid)
    finally:
        s.close()
    assert len(events) == 1


# --- retire: preserves the evidence trail (contrast with delete) -------------


def test_retire_preserves_edges(isolated_db: Path) -> None:
    """Unlike `delete`, `retire` keeps edges so the belief can be restored."""
    src_id = "src9001aabbccdd00"
    dst_id = "dst9001aabbccdd00"
    s = MemoryStore(str(isolated_db))
    try:
        for bid, content, h in (
            (src_id, "source belief", "hash_src9"),
            (dst_id, "destination belief", "hash_dst9"),
        ):
            s.insert_belief(Belief(
                id=bid, content=content, content_hash=h,
                alpha=1.0, beta=1.0, type=BELIEF_FACTUAL,
                lock_level=LOCK_NONE, locked_at=None,
                created_at="2026-05-05T00:00:00Z",
                last_retrieved_at=None, origin=ORIGIN_AGENT_INFERRED,
            ))
        s.insert_edge(Edge(src=src_id, dst=dst_id, type=EDGE_SUPPORTS, weight=0.8))
    finally:
        s.close()

    _run("retire", src_id)

    s = MemoryStore(str(isolated_db))
    try:
        # Edge survives the soft-delete — the evidence trail is intact.
        assert s.get_edge(src_id, dst_id, EDGE_SUPPORTS) is not None
    finally:
        s.close()


# --- restore: happy path -----------------------------------------------------


def test_restore_exits_zero_and_reports(isolated_db: Path) -> None:
    bid = _seed_belief(isolated_db, "bring me back")
    _run("retire", bid)
    code, out = _run("restore", bid)
    assert code == 0
    assert f"restored: {bid}" in out


def test_restore_clears_valid_to(isolated_db: Path) -> None:
    bid = _seed_belief(isolated_db, "reactivated")
    _run("retire", bid)
    _run("restore", bid)
    assert _valid_to(isolated_db, bid) is None


def test_restore_reinstates_fts_row(isolated_db: Path) -> None:
    bid = _seed_belief(isolated_db, "searchable again")
    _run("retire", bid)
    assert _in_fts(isolated_db, bid) is False
    _run("restore", bid)
    assert _in_fts(isolated_db, bid) is True


def test_restore_returns_to_active_list(isolated_db: Path) -> None:
    bid = _seed_belief(isolated_db, "active again")
    _run("retire", bid)
    _run("restore", bid)
    s = MemoryStore(str(isolated_db))
    try:
        active_ids = {b.id for b in s.list_active_beliefs()}
    finally:
        s.close()
    assert bid in active_ids


def test_restore_writes_audit_row(isolated_db: Path) -> None:
    bid = _seed_belief(isolated_db, "audit on restore")
    _run("retire", bid)
    _run("restore", bid)
    s = MemoryStore(str(isolated_db))
    try:
        events = s.list_feedback_events(belief_id=bid)
    finally:
        s.close()
    sources = [e.source for e in events]
    assert "user_restored" in sources
    restore_ev = next(e for e in events if e.source == "user_restored")
    assert restore_ev.valence == 1.0


# --- restore: no-op cases ----------------------------------------------------


def test_restore_active_belief_exits_one(isolated_db: Path) -> None:
    bid = _seed_belief(isolated_db, "already active")
    code, _ = _run("restore", bid)
    assert code == 1


def test_restore_active_belief_message(
    isolated_db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bid = _seed_belief(isolated_db, "already active")
    _run("restore", bid)
    captured = capsys.readouterr()
    assert "not restorable" in captured.err


def test_restore_unknown_id_exits_one(isolated_db: Path) -> None:
    code, _ = _run("restore", "doesnotexist")
    assert code == 1


def test_restore_active_belief_writes_no_audit_row(isolated_db: Path) -> None:
    bid = _seed_belief(isolated_db, "no spurious audit")
    _run("restore", bid)
    s = MemoryStore(str(isolated_db))
    try:
        events = s.list_feedback_events(belief_id=bid)
    finally:
        s.close()
    assert events == []


# --- round trip --------------------------------------------------------------


def test_retire_restore_round_trip_preserves_content(isolated_db: Path) -> None:
    bid = _seed_belief(isolated_db, "durable content survives the round trip")
    _run("retire", bid)
    _run("restore", bid)
    s = MemoryStore(str(isolated_db))
    try:
        b = s.get_belief(bid)
    finally:
        s.close()
    assert b is not None
    assert b.valid_to is None
    assert b.content == "durable content survives the round trip"
