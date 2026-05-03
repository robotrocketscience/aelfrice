"""PreCompact hook entry-point: payload parse, source resolution, output."""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import cast

import pytest

from aelfrice.context_rebuilder import (
    CLOSE_TAG as REBUILD_CLOSE_TAG,
    HOOK_EVENT_NAME,
    OPEN_TAG as REBUILD_OPEN_TAG,
)
from aelfrice.hook import pre_compact
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
from aelfrice.store import MemoryStore


def _additional_context(stdout_value: str) -> str | None:
    """Parse the PreCompact JSON envelope and return additionalContext.

    Returns None when stdout is empty (the documented "no rebuild
    block" exit). Raises a clear AssertionError on shape mismatch.
    """
    if not stdout_value:
        return None
    raw = json.loads(stdout_value)
    assert isinstance(raw, dict)
    payload = cast(dict[str, object], raw)
    spec_obj = payload.get("hookSpecificOutput")
    assert isinstance(spec_obj, dict)
    spec = cast(dict[str, object], spec_obj)
    assert spec.get("hookEventName") == HOOK_EVENT_NAME
    ctx = spec.get("additionalContext")
    assert isinstance(ctx, str)
    return ctx


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
    # The v1.4 ship default for `trigger_mode` is "manual"; these
    # tests exercise the auto-fire path, so opt them into "threshold".
    cfg = cwd / ".aelfrice.toml"
    if not cfg.exists():
        # Disable v1.7 (#364) relevance floor in tests by default;
        # tests that exercise floor behavior set their own thresholds
        # via this knob. Keeps pre-floor pre-compact tests valid
        # without an audit of every weak-overlap query.
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
    """No aelfrice log, no claude transcript -> empty turns -> no envelope.

    Issue #139 acceptance: empty transcript exits 0 with no
    `additionalContext` written. The tool path is unaffected.
    """
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
    # Empty transcript -> empty stdout (no JSON envelope).
    assert sout.getvalue() == ""


# ---- aelfrice log preferred over claude transcript ---------------------


def test_pre_compact_prefers_aelfrice_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    # v1.7 (#364) post-floor contract: rebuild_v14 returns "" when no
    # candidate survives the floor and there are no locks. The fixture
    # belief F1 is intentionally unrelated to the recent turn (FTS5 is
    # not populated for in-memory inserts in this test path), so we
    # lock it to exercise the L0-always-packs guarantee. The test's
    # assertion is "the rebuild path picked the aelfrice log over the
    # alternate transcript path", not "weak BM25 hit packs".
    _seed_db(
        db,
        [_mk("F1", "kitchen contents",
             lock_level=LOCK_USER, locked_at="2026-04-26T00:00:00Z")],
    )
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
    ctx = _additional_context(sout.getvalue())
    assert ctx is not None
    assert 'kitchen contents' in ctx
    # The fallback's content must NOT appear because aelfrice log won.
    assert "lawnmower" not in ctx


def test_pre_compact_falls_back_to_claude_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No .git anywhere -> use transcript_path."""
    db = tmp_path / "memory.db"
    # See test_pre_compact_prefers_aelfrice_log for the L0-lock rationale
    # post v1.7 (#364) floor.
    _seed_db(
        db,
        [_mk("F1", "kitchen check fixture",
             lock_level=LOCK_USER, locked_at="2026-04-26T00:00:00Z")],
    )
    _set_db(monkeypatch, db)
    no_git = tmp_path / "no_git"
    no_git.mkdir()
    # v1.4 default trigger_mode is "manual"; opt into auto-fire. Floor
    # disabled so this test's assertion ("kitchen check" appears in
    # rebuild output) is not gated on FTS5 indexing of the fixture.
    (no_git / ".aelfrice.toml").write_text(
        '[rebuilder]\ntrigger_mode = "threshold"\n'
        "[rebuild_floor]\nsession = 0.0\nl1 = 0.0\n",
        encoding="utf-8",
    )
    transcript = tmp_path / "claude.jsonl"
    _claude_transcript(transcript, [
        {"type": "user", "message": {"role": "user", "content": "kitchen check"}},
    ])
    sin = io.StringIO(_payload(cwd=no_git, transcript_path=transcript))
    sout = io.StringIO()
    serr = io.StringIO()
    rc = pre_compact(stdin=sin, stdout=sout, stderr=serr)
    assert rc == 0
    ctx = _additional_context(sout.getvalue())
    assert ctx is not None
    assert "kitchen check" in ctx


# ---- output emits expected tags ----------------------------------------


def test_pre_compact_writes_rebuild_block_with_hits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    # v1.7 (#364) post-floor contract: lock the fixture so the L0
    # always-pack lane keeps the block emitted regardless of FTS5
    # state. The assertion target is the block's structural tags,
    # not retrieval quality.
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
    ctx = _additional_context(sout.getvalue())
    assert ctx is not None
    assert REBUILD_OPEN_TAG in ctx
    assert REBUILD_CLOSE_TAG in ctx
    assert "<recent-turns>" in ctx
    assert "<continue/>" in ctx


# ---- locked beliefs surface even with no transcript ---------------------


def test_pre_compact_no_envelope_when_no_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No transcript -> no rebuild block, even if the store has locked beliefs.

    Per issue #139: an empty transcript exits 0 with no
    `additionalContext`. L0 baseline is the SessionStart hook's
    channel; the rebuilder is for resuming a session in progress.
    """
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
    assert sout.getvalue() == ""
