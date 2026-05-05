"""Tests for `aelf confirm` CLI subcommand (#441).

Unit tests use the in-process `main(argv=..., out=...)` harness.
Integration tests open a real MemoryStore and verify feedback_history rows.
Each test is isolated via the `isolated_db` fixture (AELFRICE_DB envvar).
"""
from __future__ import annotations

import io
from pathlib import Path

import pytest

from aelfrice.cli import main
from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    ORIGIN_AGENT_INFERRED,
    Belief,
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


def _seed_belief(db: Path, content: str, bid: str = "aabbccddeeff") -> str:
    """Insert one agent_inferred belief into the store and return its id."""
    s = MemoryStore(str(db))
    try:
        s.insert_belief(Belief(
            id=bid, content=content, content_hash="testhash",
            alpha=1.0, beta=1.0, type=BELIEF_FACTUAL,
            lock_level=LOCK_NONE, locked_at=None,
            demotion_pressure=0,
            created_at="2026-05-05T00:00:00Z",
            last_retrieved_at=None,
            origin=ORIGIN_AGENT_INFERRED,
        ))
    finally:
        s.close()
    return bid


# --- unit: happy path -------------------------------------------------------


def test_confirm_happy_path_exits_zero(isolated_db: Path) -> None:
    bid = _seed_belief(isolated_db, "Python uses indentation for blocks")
    code, out = _run("confirm", bid)
    assert code == 0
    assert "confirmed" in out
    assert bid in out


def test_confirm_shows_alpha_transition(isolated_db: Path) -> None:
    bid = _seed_belief(isolated_db, "git commits should be atomic")
    code, out = _run("confirm", bid)
    assert code == 0
    # Prior alpha is 1.000; after +1.0 it should be 2.000.
    assert "1.000->2.000" in out


def test_confirm_shows_posterior_mean(isolated_db: Path) -> None:
    bid = _seed_belief(isolated_db, "always write tests before shipping")
    code, out = _run("confirm", bid)
    assert code == 0
    # alpha=2, beta=1 -> mean = 2/3 ≈ 0.667
    assert "mean" in out
    assert "0.667" in out


# --- unit: unknown belief ---------------------------------------------------


def test_confirm_unknown_belief_exits_one(isolated_db: Path) -> None:
    code, _ = _run("confirm", "doesnotexist")
    assert code == 1


# --- unit: --note flag ------------------------------------------------------


def test_confirm_note_appears_in_output(isolated_db: Path) -> None:
    bid = _seed_belief(isolated_db, "use uv for Python environment management")
    code, out = _run("confirm", bid, "--note", "verified in prod")
    assert code == 0
    assert "verified in prod" in out


def test_confirm_note_not_persisted(isolated_db: Path) -> None:
    """Note must NOT be stored in feedback_history."""
    bid = _seed_belief(isolated_db, "sign every commit with SSH")
    _run("confirm", bid, "--note", "check this note is gone")
    s = MemoryStore(str(isolated_db))
    try:
        events = s.list_feedback_events(belief_id=bid)
    finally:
        s.close()
    assert len(events) == 1
    # There is no 'note' column in feedback_history — just verify the row exists.
    assert events[0].source == "user_confirmed"


# --- unit: --source override ------------------------------------------------


def test_confirm_source_override_written_to_history(isolated_db: Path) -> None:
    bid = _seed_belief(isolated_db, "all secrets go in environment variables")
    code, _ = _run("confirm", bid, "--source", "ci_automation")
    assert code == 0
    s = MemoryStore(str(isolated_db))
    try:
        events = s.list_feedback_events(belief_id=bid)
    finally:
        s.close()
    assert len(events) == 1
    assert events[0].source == "ci_automation"


# --- integration: end-to-end ------------------------------------------------


def test_confirm_writes_feedback_history_row(isolated_db: Path) -> None:
    """Full integration: ingest via lock, confirm, assert row in feedback_history."""
    # Use `aelf lock` to insert a belief with a known content string.
    lock_code, lock_out = _run("lock", "always review diffs before merging")
    assert lock_code == 0

    # Retrieve the belief id from the store.
    s = MemoryStore(str(isolated_db))
    try:
        locked = s.list_locked_beliefs()
        assert len(locked) == 1
        bid = locked[0].id
        pre_alpha = locked[0].alpha
        pre_events = s.count_feedback_events(bid)
    finally:
        s.close()

    code, out = _run("confirm", bid)
    assert code == 0
    assert "confirmed" in out

    s = MemoryStore(str(isolated_db))
    try:
        b = s.get_belief(bid)
        assert b is not None
        # Alpha must have increased.
        assert b.alpha > pre_alpha
        # Exactly one new feedback_history row.
        assert s.count_feedback_events(bid) == pre_events + 1
        events = s.list_feedback_events(belief_id=bid)
        latest = events[0]
        assert latest.source == "user_confirmed"
        assert latest.valence == 1.0
    finally:
        s.close()


def test_confirm_does_not_write_belief_corroborations(isolated_db: Path) -> None:
    """confirm writes feedback_history, NOT belief_corroborations (#441/#190)."""
    bid = _seed_belief(isolated_db, "never skip the linter")
    _run("confirm", bid)
    s = MemoryStore(str(isolated_db))
    try:
        # belief_corroborations only populated on re-ingest dedup, not confirm.
        # Verify via direct SQL; if the table doesn't exist that's also fine.
        cur = s._conn.execute(  # pyright: ignore[reportPrivateUsage]
            "SELECT COUNT(*) FROM sqlite_master "
            "WHERE type='table' AND name='belief_corroborations'"
        )
        table_exists = cur.fetchone()[0] > 0
        if table_exists:
            cur2 = s._conn.execute(  # pyright: ignore[reportPrivateUsage]
                "SELECT COUNT(*) FROM belief_corroborations WHERE belief_id = ?",
                (bid,),
            )
            assert cur2.fetchone()[0] == 0, (
                "confirm must not write to belief_corroborations"
            )
        # Either way, feedback_history must have exactly one row.
        assert s.count_feedback_events(bid) == 1
    finally:
        s.close()
