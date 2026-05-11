"""Tests for the #288 phase-1a rebuild_log on the UserPromptSubmit path.

Phase-1a originally instrumented only `rebuild_v14` (the PreCompact
call site). The UserPromptSubmit hook calls `search_for_prompt`
directly, so the high-frequency rebuild path produced no log rows
and phase-1b operator-week data could not accumulate. These tests
cover the UPS-side wiring: schema parity with the PreCompact log,
dedup-drop visibility, env opt-out, TOML opt-out, and fail-soft
behaviour when the log path can't be derived.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from aelfrice.context_rebuilder import (
    REBUILD_LOG_ENV,
    record_user_prompt_submit_log,
)
from aelfrice.hook import user_prompt_submit
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
from aelfrice.store import MemoryStore


# ---- helpers -----------------------------------------------------------


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
        created_at="2026-04-28T00:00:00Z",
        last_retrieved_at=None,
    )


def _seed_db(db_path: Path, beliefs: list[Belief]) -> None:
    store = MemoryStore(str(db_path))
    try:
        for b in beliefs:
            store.insert_belief(b)
    finally:
        store.close()


def _payload(prompt: str, session_id: str = "ups-sess-1") -> str:
    return json.dumps(
        {
            "session_id": session_id,
            "transcript_path": "/dev/null",
            "cwd": "/tmp",
            "hook_event_name": "UserPromptSubmit",
            "prompt": prompt,
        }
    )


def _read_log(log_path: Path) -> list[dict[str, object]]:
    return [
        json.loads(ln)
        for ln in log_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]


# ---- end-to-end through the hook entry point ---------------------------


def test_user_prompt_submit_writes_rebuild_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "the kitchen is full of bananas")])
    monkeypatch.setenv("AELFRICE_DB", str(db))

    sout = io.StringIO()
    rc = user_prompt_submit(
        stdin=io.StringIO(_payload("are there bananas in the kitchen", "ups-sess-1")),
        stdout=sout,
        stderr=io.StringIO(),
    )
    assert rc == 0
    assert sout.getvalue() != ""

    log_path = tmp_path / "rebuild_logs" / "ups-sess-1.jsonl"
    assert log_path.exists(), (
        "UPS hook must emit a rebuild_log row when retrieval returns "
        "any candidate"
    )
    rows = _read_log(log_path)
    assert len(rows) == 1
    rec = rows[0]
    assert rec["session_id"] == "ups-sess-1"
    assert isinstance(rec["ts"], str) and rec["ts"].endswith("Z")
    # Schema parity with the PreCompact rebuild_log: same input/
    # candidates/pack_summary keys with the same `_empty_scores`
    # block per candidate.
    assert set(rec["input"]) == {
        "recent_turns_hash", "n_recent_turns",
        "extracted_query", "extracted_entities", "extracted_intent",
    }
    assert rec["input"]["n_recent_turns"] == 1
    assert isinstance(rec["candidates"], list)
    assert len(rec["candidates"]) >= 1
    cand = rec["candidates"][0]
    assert set(cand["scores"]) == {
        "bm25", "posterior_mean", "reranker", "final",
    }
    assert cand["decision"] == "packed"
    assert cand["reason"] is None
    assert rec["pack_summary"]["n_candidates"] >= 1
    assert rec["pack_summary"]["n_packed"] >= 1


def test_user_prompt_submit_no_log_when_no_hits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No retrieval hits means no candidate set, hence no row —
    same contract as `rebuild_v14` on an empty store."""
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "elephants are large")])
    monkeypatch.setenv("AELFRICE_DB", str(db))

    rc = user_prompt_submit(
        stdin=io.StringIO(_payload("dogs", "no-hit-sess")),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    assert rc == 0
    log_path = tmp_path / "rebuild_logs" / "no-hit-sess.jsonl"
    assert not log_path.exists()


def test_user_prompt_submit_env_opt_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "the kitchen is full of bananas")])
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.setenv(REBUILD_LOG_ENV, "0")

    rc = user_prompt_submit(
        stdin=io.StringIO(_payload("bananas", "opt-out")),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    assert rc == 0
    log_path = tmp_path / "rebuild_logs" / "opt-out.jsonl"
    assert not log_path.exists()


def test_user_prompt_submit_toml_opt_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "the kitchen is full of bananas")])
    monkeypatch.setenv("AELFRICE_DB", str(db))
    cfg = tmp_path / ".aelfrice.toml"
    cfg.write_text(
        "[rebuild_log]\nenabled = false\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    rc = user_prompt_submit(
        stdin=io.StringIO(_payload("bananas", "toml-off")),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    assert rc == 0
    log_path = tmp_path / "rebuild_logs" / "toml-off.jsonl"
    assert not log_path.exists()


def test_user_prompt_submit_no_log_when_session_id_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The on-disk JSONL is keyed by session_id; without it there's
    no file path to write to. Drop silently rather than fabricating
    a session id."""
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "the kitchen is full of bananas")])
    monkeypatch.setenv("AELFRICE_DB", str(db))
    payload = json.dumps(
        {
            # no session_id field
            "transcript_path": "/dev/null",
            "cwd": "/tmp",
            "hook_event_name": "UserPromptSubmit",
            "prompt": "bananas",
        }
    )

    rc = user_prompt_submit(
        stdin=io.StringIO(payload),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    assert rc == 0
    log_dir = tmp_path / "rebuild_logs"
    assert not log_dir.exists() or not any(log_dir.iterdir())


# ---- direct unit coverage of the helper --------------------------------


def test_record_user_prompt_submit_log_marks_dedup_drops(
    tmp_path: Path,
) -> None:
    """Pre-dedup hits with duplicate content collapse to one
    `packed` survivor; the dropped duplicates carry
    `content_hash_collision_with:<survivor>` as their reason."""
    log_path = tmp_path / "rebuild_logs" / "dedup.jsonl"
    pre = [
        _mk("A", "same content"),
        _mk("B", "same content"),  # collides with A
        _mk("C", "different content"),
    ]
    post = [pre[0], pre[2]]  # B dropped by dedup
    record_user_prompt_submit_log(
        prompt="anything",
        session_id="dedup",
        hits_pre_dedup=pre,
        hits_post_dedup=post,
        log_path=log_path,
        enabled=True,
        stderr=io.StringIO(),
    )
    rows = _read_log(log_path)
    assert len(rows) == 1
    rec = rows[0]
    assert rec["pack_summary"]["n_candidates"] == 3
    assert rec["pack_summary"]["n_packed"] == 2
    assert rec["pack_summary"]["n_dropped_by_dedup"] == 1
    by_id = {c["belief_id"]: c for c in rec["candidates"]}
    assert by_id["A"]["decision"] == "packed"
    assert by_id["B"]["decision"] == "dropped"
    assert by_id["B"]["reason"] == "content_hash_collision_with:A"
    assert by_id["C"]["decision"] == "packed"


def test_record_user_prompt_submit_log_lock_level_passthrough(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "rebuild_logs" / "locks.jsonl"
    pre = [
        _mk("L0", "x", lock_level=LOCK_USER, locked_at="2026-04-28T00:00:00Z"),
        _mk("L1", "y"),
    ]
    record_user_prompt_submit_log(
        prompt="q",
        session_id="locks",
        hits_pre_dedup=pre,
        hits_post_dedup=pre,
        log_path=log_path,
        enabled=True,
        stderr=io.StringIO(),
    )
    rec = _read_log(log_path)[0]
    by_id = {c["belief_id"]: c for c in rec["candidates"]}
    assert by_id["L0"]["lock_level"] == "user"
    assert by_id["L1"]["lock_level"] == "none"


def test_record_user_prompt_submit_log_disabled_writes_nothing(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "rebuild_logs" / "off.jsonl"
    record_user_prompt_submit_log(
        prompt="q",
        session_id="off",
        hits_pre_dedup=[_mk("A", "x")],
        hits_post_dedup=[_mk("A", "x")],
        log_path=log_path,
        enabled=False,
        stderr=io.StringIO(),
    )
    assert not log_path.exists()


def test_record_user_prompt_submit_log_no_path_is_noop(
    tmp_path: Path,
) -> None:
    """`log_path=None` means we couldn't derive a per-session file
    (e.g. the brain-graph DB is in-memory). Must be a silent no-op."""
    record_user_prompt_submit_log(
        prompt="q",
        session_id="x",
        hits_pre_dedup=[_mk("A", "x")],
        hits_post_dedup=[_mk("A", "x")],
        log_path=None,
        enabled=True,
        stderr=io.StringIO(),
    )
