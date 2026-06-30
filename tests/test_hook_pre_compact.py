"""PreCompact neuter + post-compaction SessionStart rebuild (#1031).

The host harness rejects `additionalContext` emitted from a
PreCompact hook, so `pre_compact` is now a no-op on stdout and the
rebuild block is carried by the SessionStart hook on
`source == "compact"`. These tests pin both halves: PreCompact stays
silent, and SessionStart:compact emits the rebuild block (preferring
the canonical aelfrice log, falling back to the Claude transcript).
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from aelfrice.context_rebuilder import (
    CLOSE_TAG as REBUILD_CLOSE_TAG,
    OPEN_TAG as REBUILD_OPEN_TAG,
)
from aelfrice.hook import pre_compact, session_start
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
from aelfrice.store import MemoryStore


def _rebuild_block(stdout_value: str) -> str | None:
    """Slice the `<aelfrice-rebuild>` block out of SessionStart stdout.

    SessionStart emits the locked baseline block followed by the rebuild
    block on a compact fire; this isolates the rebuild portion. Returns
    None when no rebuild block was written.
    """
    if REBUILD_OPEN_TAG not in stdout_value:
        return None
    start = stdout_value.index(REBUILD_OPEN_TAG)
    end = stdout_value.index(REBUILD_CLOSE_TAG) + len(REBUILD_CLOSE_TAG)
    return stdout_value[start:end]


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
    # Auto-fire is the ship default since #746; we still write the
    # threshold mode explicitly so the test is self-documenting and
    # robust to any future default flip. Disable the v1.7 (#364)
    # relevance floor so these tests aren't gated on weak-overlap BM25.
    cfg = cwd / ".aelfrice.toml"
    if not cfg.exists():
        cfg.write_text(
            '[rebuilder]\ntrigger_mode = "threshold"\n'
            "[rebuild_floor]\nsession = 0.0\nl1 = 0.0\n",
            encoding="utf-8",
        )
    return p


def _claude_transcript(path: Path, lines: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    return path


def _payload(
    *,
    cwd: Path | None = None,
    transcript_path: Path | None = None,
    source: str = "",
    hook_event_name: str = "PreCompact",
) -> str:
    body: dict[str, object] = {
        "session_id": "s1",
        "transcript_path": str(transcript_path) if transcript_path else "",
        "cwd": str(cwd) if cwd else "",
        "hook_event_name": hook_event_name,
    }
    if source:
        body["source"] = source
    return json.dumps(body)


def _start_compact(
    *, cwd: Path | None = None, transcript_path: Path | None = None
) -> str:
    """Run the SessionStart hook with a `source=="compact"` payload."""
    sin = io.StringIO(
        _payload(
            cwd=cwd,
            transcript_path=transcript_path,
            source="compact",
            hook_event_name="SessionStart",
        )
    )
    sout = io.StringIO()
    serr = io.StringIO()
    rc = session_start(stdin=sin, stdout=sout, stderr=serr)
    assert rc == 0
    return sout.getvalue()


# ---- PreCompact is now a no-op on stdout (#1031) ------------------------


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


def test_pre_compact_silent_even_with_turns_and_hits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#1031: PreCompact emits nothing even when a rebuild would be
    produced — a populated store, a non-empty transcript, threshold
    mode. The harness rejects PreCompact `additionalContext`, so the
    block must not be written here (it moves to SessionStart:compact).
    """
    db = tmp_path / "memory.db"
    _seed_db(
        db,
        [_mk("F1", "kitchen has bananas",
             lock_level=LOCK_USER, locked_at="2026-04-26T00:00:00Z")],
    )
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
    assert sout.getvalue() == ""


# ---- SessionStart:compact emits the rebuild block ----------------------


def test_session_start_compact_prefers_aelfrice_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    # v1.7 (#364) post-floor contract: lock F1 so the L0 always-pack
    # lane keeps the block emitted regardless of FTS5 state. The
    # assertion is "the rebuild picked the aelfrice log over the
    # alternate transcript path".
    _seed_db(
        db,
        [_mk("F1", "kitchen contents",
             lock_level=LOCK_USER, locked_at="2026-04-26T00:00:00Z")],
    )
    _set_db(monkeypatch, db)
    cwd = tmp_path / "repo"
    (cwd / ".git").mkdir(parents=True)
    _aelfrice_log(cwd, [{"role": "user", "text": "kitchen contents"}])
    # Unrelated claude transcript that, if used, would surface different
    # content -- verifies the aelfrice log wins.
    other = tmp_path / "claude.jsonl"
    _claude_transcript(other, [{
        "type": "user",
        "message": {"role": "user", "content": "lawnmower repair"},
    }])
    out = _start_compact(cwd=cwd, transcript_path=other)
    block = _rebuild_block(out)
    assert block is not None
    assert "kitchen contents" in block
    # The fallback's content must NOT appear because the aelfrice log won.
    assert "lawnmower" not in block


def test_session_start_compact_falls_back_to_claude_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No .git anywhere -> use transcript_path."""
    db = tmp_path / "memory.db"
    _seed_db(
        db,
        [_mk("F1", "kitchen check fixture",
             lock_level=LOCK_USER, locked_at="2026-04-26T00:00:00Z")],
    )
    _set_db(monkeypatch, db)
    no_git = tmp_path / "no_git"
    no_git.mkdir()
    (no_git / ".aelfrice.toml").write_text(
        '[rebuilder]\ntrigger_mode = "threshold"\n'
        "[rebuild_floor]\nsession = 0.0\nl1 = 0.0\n",
        encoding="utf-8",
    )
    transcript = tmp_path / "claude.jsonl"
    _claude_transcript(transcript, [
        {"type": "user", "message": {"role": "user", "content": "kitchen check"}},
    ])
    out = _start_compact(cwd=no_git, transcript_path=transcript)
    block = _rebuild_block(out)
    assert block is not None
    assert "kitchen check" in block


def test_session_start_compact_writes_rebuild_block_with_hits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    _seed_db(
        db,
        [_mk("F1", "kitchen has bananas",
             lock_level=LOCK_USER, locked_at="2026-04-26T00:00:00Z")],
    )
    _set_db(monkeypatch, db)
    cwd = tmp_path / "repo"
    (cwd / ".git").mkdir(parents=True)
    _aelfrice_log(cwd, [
        {"role": "user", "text": "kitchen contents"},
        {"role": "assistant", "text": "checking now"},
    ])
    out = _start_compact(cwd=cwd)
    block = _rebuild_block(out)
    assert block is not None
    assert REBUILD_OPEN_TAG in block
    assert REBUILD_CLOSE_TAG in block
    assert "<recent-turns>" in block
    assert "<continue/>" in block


def test_session_start_compact_no_rebuild_when_no_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No transcript -> no rebuild block (even with locked beliefs).

    The locked baseline still emits via the SessionStart L0 channel;
    the rebuilder is for resuming a session in progress, so an empty
    transcript yields no `<aelfrice-rebuild>` block.
    """
    db = tmp_path / "memory.db"
    _seed_db(
        db,
        [_mk("L1", "user-locked baseline",
             lock_level=LOCK_USER, locked_at="2026-04-26T00:00:00Z")],
    )
    _set_db(monkeypatch, db)
    no_git = tmp_path / "no_git"
    no_git.mkdir()
    out = _start_compact(cwd=no_git)
    assert _rebuild_block(out) is None


def test_session_start_non_compact_source_emits_no_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A normal (non-compact) SessionStart never emits the rebuild block,
    even with a populated transcript — only `source == "compact"` does.
    """
    db = tmp_path / "memory.db"
    _seed_db(
        db,
        [_mk("F1", "kitchen has bananas",
             lock_level=LOCK_USER, locked_at="2026-04-26T00:00:00Z")],
    )
    _set_db(monkeypatch, db)
    cwd = tmp_path / "repo"
    (cwd / ".git").mkdir(parents=True)
    _aelfrice_log(cwd, [
        {"role": "user", "text": "kitchen contents"},
        {"role": "assistant", "text": "checking now"},
    ])
    sin = io.StringIO(
        _payload(cwd=cwd, source="startup", hook_event_name="SessionStart")
    )
    sout = io.StringIO()
    serr = io.StringIO()
    rc = session_start(stdin=sin, stdout=sout, stderr=serr)
    assert rc == 0
    assert _rebuild_block(sout.getvalue()) is None
