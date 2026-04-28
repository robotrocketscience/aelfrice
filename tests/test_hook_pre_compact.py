"""PreCompact hook entry-point: payload parse, source resolution, output."""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from aelfrice.context_rebuilder import (
    CLOSE_TAG as REBUILD_CLOSE_TAG,
    OPEN_TAG as REBUILD_OPEN_TAG,
)
from aelfrice.hook import pre_compact
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
from aelfrice.store import MemoryStore


def _mk(
    bid: str,
    content: str,
    lock_level: str = LOCK_NONE,
    locked_at: str | None = None,
) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=lock_level,
        locked_at=locked_at,
        demotion_pressure=0,
        created_at="2026-04-26T00:00:00Z",
        last_retrieved_at=None,
    )


def _seed_db(db_path: Path, beliefs: list[Belief]) -> None:
    store = MemoryStore(str(db_path))
    try:
        for b in beliefs:
            store.insert_belief(b)
    finally:
        store.close()


def _set_db(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    monkeypatch.setenv("AELFRICE_DB", str(path))


def _aelfrice_log(cwd: Path, lines: list[dict[str, object]]) -> Path:
    p = cwd / ".git" / "aelfrice" / "transcripts" / "turns.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    return p


def _claude_transcript(path: Path, lines: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    return path


def _payload(
    *,
    cwd: Path | None = None,
    transcript_path: Path | None = None,
) -> str:
    return json.dumps(
        {
            "session_id": "s1",
            "transcript_path": str(transcript_path) if transcript_path else "",
            "cwd": str(cwd) if cwd else "",
            "hook_event_name": "PreCompact",
        }
    )


# ---- exit code contract -------------------------------------------------


def test_pre_compact_returns_zero_on_empty_stdin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    _seed_db(db, [])
    _set_db(monkeypatch, db)
    sin = io.StringIO("")
    sout = io.StringIO()
    serr = io.StringIO()
    rc = pre_compact(stdin=sin, stdout=sout, stderr=serr)
    assert rc == 0
    assert sout.getvalue() == ""


def test_pre_compact_returns_zero_on_malformed_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    _seed_db(db, [])
    _set_db(monkeypatch, db)
    sin = io.StringIO("{not valid json")
    sout = io.StringIO()
    serr = io.StringIO()
    rc = pre_compact(stdin=sin, stdout=sout, stderr=serr)
    assert rc == 0
    assert sout.getvalue() == ""


def test_pre_compact_returns_zero_when_no_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No aelfrice log, no claude transcript -> empty turns -> empty body."""
    db = tmp_path / "memory.db"
    _seed_db(db, [])
    _set_db(monkeypatch, db)
    no_git = tmp_path / "no_git"
    no_git.mkdir()
    sin = io.StringIO(_payload(cwd=no_git))
    sout = io.StringIO()
    serr = io.StringIO()
    rc = pre_compact(stdin=sin, stdout=sout, stderr=serr)
    assert rc == 0
    # No locked beliefs, no recent turns -> rebuild() emits the empty
    # frame (just <continue/> wrapped in tags). No <recent-turns> or
    # <retrieved-beliefs> sections.
    out = sout.getvalue()
    assert REBUILD_OPEN_TAG in out
    assert REBUILD_CLOSE_TAG in out
    assert "<recent-turns>" not in out
    assert "<retrieved-beliefs" not in out


# ---- aelfrice log preferred over claude transcript ---------------------


def test_pre_compact_prefers_aelfrice_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "kitchen has bananas")])
    _set_db(monkeypatch, db)
    cwd = tmp_path / "repo"
    (cwd / ".git").mkdir(parents=True)
    _aelfrice_log(cwd, [{"role": "user", "text": "kitchen contents"}])
    # Provide an unrelated claude transcript that, if used, would
    # surface different content -- verifies aelfrice log wins.
    other = tmp_path / "claude.jsonl"
    _claude_transcript(other, [{
        "type": "user",
        "message": {"role": "user", "content": "lawnmower repair"},
    }])
    sin = io.StringIO(_payload(cwd=cwd, transcript_path=other))
    sout = io.StringIO()
    serr = io.StringIO()
    rc = pre_compact(stdin=sin, stdout=sout, stderr=serr)
    assert rc == 0
    out = sout.getvalue()
    assert 'kitchen contents' in out
    # The fallback's content must NOT appear because aelfrice log won.
    assert "lawnmower" not in out


def test_pre_compact_falls_back_to_claude_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No .git anywhere -> use transcript_path."""
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "kitchen has bananas")])
    _set_db(monkeypatch, db)
    no_git = tmp_path / "no_git"
    no_git.mkdir()
    transcript = tmp_path / "claude.jsonl"
    _claude_transcript(transcript, [
        {"type": "user", "message": {"role": "user", "content": "kitchen check"}},
    ])
    sin = io.StringIO(_payload(cwd=no_git, transcript_path=transcript))
    sout = io.StringIO()
    serr = io.StringIO()
    rc = pre_compact(stdin=sin, stdout=sout, stderr=serr)
    assert rc == 0
    out = sout.getvalue()
    assert "kitchen check" in out


# ---- output emits expected tags ----------------------------------------


def test_pre_compact_writes_rebuild_block_with_hits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "kitchen has bananas")])
    _set_db(monkeypatch, db)
    cwd = tmp_path / "repo"
    (cwd / ".git").mkdir(parents=True)
    _aelfrice_log(cwd, [
        {"role": "user", "text": "kitchen contents"},
        {"role": "assistant", "text": "checking now"},
    ])
    sin = io.StringIO(_payload(cwd=cwd))
    sout = io.StringIO()
    serr = io.StringIO()
    rc = pre_compact(stdin=sin, stdout=sout, stderr=serr)
    assert rc == 0
    out = sout.getvalue()
    assert REBUILD_OPEN_TAG in out
    assert REBUILD_CLOSE_TAG in out
    assert "<recent-turns>" in out
    assert 'id="F1"' in out
    assert "<continue/>" in out


# ---- locked beliefs surface even with no transcript ---------------------


def test_pre_compact_returns_locked_when_no_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    _seed_db(
        db,
        [_mk(
            "L1", "user-locked baseline",
            lock_level=LOCK_USER, locked_at="2026-04-26T00:00:00Z",
        )],
    )
    _set_db(monkeypatch, db)
    no_git = tmp_path / "no_git"
    no_git.mkdir()
    sin = io.StringIO(_payload(cwd=no_git))
    sout = io.StringIO()
    serr = io.StringIO()
    rc = pre_compact(stdin=sin, stdout=sout, stderr=serr)
    assert rc == 0
    out = sout.getvalue()
    assert 'id="L1"' in out
    assert 'locked="true"' in out
