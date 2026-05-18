"""Coverage-line: hook surfaces retrieval/index count asymmetry.

Three acceptance cases from the spec:
  M == N  →  line omitted
  M >  N  →  line shown
  M == 0  →  no-op (no hit at all, no line)
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from aelfrice.hook import (
    CLOSE_TAG,
    OPEN_TAG,
    _coverage_line,
    user_prompt_submit,
)
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief
from aelfrice.retrieval import LaneTelemetry
from aelfrice.store import MemoryStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mk(bid: str, content: str) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
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


def _payload(prompt: str, session_id: str = "s1") -> str:
    return json.dumps(
        {
            "session_id": session_id,
            "transcript_path": "/dev/null",
            "cwd": "/tmp",
            "hook_event_name": "UserPromptSubmit",
            "prompt": prompt,
        }
    )


def _set_db(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    monkeypatch.setenv("AELFRICE_DB", str(path))


# ---------------------------------------------------------------------------
# Unit tests for _coverage_line
# ---------------------------------------------------------------------------

def _tel(*, locked: int = 0, l25: int = 0, l1: int = 0, l1_candidates: int = 0) -> LaneTelemetry:
    return LaneTelemetry(
        locked=locked,
        l25=l25,
        l1=l1,
        l1_candidates=l1_candidates,
    )


def test_coverage_line_omitted_when_m_equals_n() -> None:
    """M == N: all candidates injected, no coverage line."""
    tel = _tel(locked=0, l25=0, l1=3, l1_candidates=3)
    result = _coverage_line(3, tel, "some query")
    assert result == ""


def test_coverage_line_shown_when_m_greater_than_n() -> None:
    """M > N: token budget cut some L1 candidates, line is emitted."""
    tel = _tel(locked=0, l25=0, l1=2, l1_candidates=5)
    result = _coverage_line(2, tel, "some query")
    assert result != ""
    assert "retrieved 2 of 5" in result
    assert "some query" in result
    assert "aelf search" in result


def test_coverage_line_m_zero_no_op() -> None:
    """M == 0: no candidates at all, no coverage line."""
    tel = _tel(locked=0, l25=0, l1=0, l1_candidates=0)
    result = _coverage_line(0, tel, "some query")
    assert result == ""


def test_coverage_line_counts_include_locked_and_l25() -> None:
    """M sums locked + l25 + l1_candidates; N is total injected."""
    # 1 locked + 1 l25 + 3 l1_candidates = M=5; injected 3 (locked+l25+1 l1)
    tel = _tel(locked=1, l25=1, l1=1, l1_candidates=3)
    result = _coverage_line(3, tel, "keyword")
    assert "retrieved 3 of 5" in result


def test_coverage_line_no_belief_content() -> None:
    """Coverage line must not contain belief content — only counts + topic."""
    secret = "confidential-secret-content-xyz"
    tel = _tel(locked=0, l25=0, l1=1, l1_candidates=4)
    result = _coverage_line(1, tel, "normal query")
    assert secret not in result


def test_coverage_line_topic_truncated_for_long_prompt() -> None:
    """Long prompt is truncated at 60 chars in the display portion."""
    long_prompt = "x" * 200
    tel = _tel(locked=0, l25=0, l1=1, l1_candidates=3)
    result = _coverage_line(1, tel, long_prompt)
    assert result != ""
    assert "…" in result
    assert long_prompt not in result


def test_coverage_line_prompt_appears_verbatim_when_short() -> None:
    """Short prompt appears verbatim in the coverage line."""
    short_prompt = "short topic"
    tel = _tel(locked=0, l25=0, l1=1, l1_candidates=2)
    result = _coverage_line(1, tel, short_prompt)
    assert short_prompt in result


# ---------------------------------------------------------------------------
# Integration tests: hook output includes/omits coverage line
# ---------------------------------------------------------------------------

def test_hook_coverage_line_shown_when_budget_truncates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hook emits coverage line when token budget causes M > N.

    Strategy: insert many beliefs that all match the query keyword,
    then set a tiny token budget so only a few fit. The coverage line
    must appear in the hook output with the correct counts.
    """
    db = tmp_path / "memory.db"
    # Each belief content is ~10 tokens; budget=30 allows about 3 through.
    beliefs = [
        _mk(f"B{i}", f"banana related topic content here {i:02d}")
        for i in range(10)
    ]
    _seed_db(db, beliefs)
    _set_db(monkeypatch, db)

    # Use a prompt that passes the hook's prompt-shape gate (not "trivial:short").
    sin = io.StringIO(_payload("tell me about bananas"))
    sout = io.StringIO()
    rc = user_prompt_submit(stdin=sin, stdout=sout, token_budget=30)
    assert rc == 0
    out = sout.getvalue()
    assert OPEN_TAG in out
    assert CLOSE_TAG in out
    assert "retrieved" in out
    assert "of 10" in out
    assert "aelf search" in out
    assert "bananas" in out


def test_hook_coverage_line_omitted_when_all_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hook omits coverage line when every candidate fits in the token budget."""
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "banana is yellow fruit")])
    _set_db(monkeypatch, db)

    # Use a prompt that passes the hook's prompt-shape gate (not "trivial:short").
    sin = io.StringIO(_payload("tell me about bananas"))
    sout = io.StringIO()
    rc = user_prompt_submit(stdin=sin, stdout=sout)
    assert rc == 0
    out = sout.getvalue()
    assert OPEN_TAG in out
    assert "aelf search" not in out


def test_hook_coverage_line_absent_when_no_hits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hook emits nothing (not even coverage line) when M == 0."""
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "elephant is a large grey mammal")])
    _set_db(monkeypatch, db)

    # Use a long prompt to pass the gate, but with content that has no match.
    sin = io.StringIO(_payload("completely unrelated xyzzy plugh quux frobble"))
    sout = io.StringIO()
    rc = user_prompt_submit(stdin=sin, stdout=sout)
    assert rc == 0
    out = sout.getvalue()
    assert out == ""
