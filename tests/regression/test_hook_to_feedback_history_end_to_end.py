"""Regression: hook firing → feedback_history grows.

End-to-end coverage of the v1.0.1 hook→feedback-history loop closure.
A UserPromptSubmit payload is fed to user_prompt_submit() against an
on-disk DB seeded with three matching beliefs. After the call:

  - feedback_history contains rows tagged source='hook'.
  - Each row has positive valence.
  - Per-belief alpha increased; beta unchanged.
  - User-locked beliefs in the surrounding graph are not auto-demoted
    (propagate=False semantics through the hook path).
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from aelfrice.hook import user_prompt_submit
from aelfrice.hook_search import HOOK_FEEDBACK_SOURCE, HOOK_RETRIEVAL_VALENCE
from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_CONTRADICTS,
    LOCK_NONE,
    LOCK_USER,
    Belief,
    Edge,
)
from aelfrice.store import MemoryStore


pytestmark = pytest.mark.regression


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


def _payload(prompt: str) -> str:
    return json.dumps(
        {
            "session_id": "s1",
            "transcript_path": "/dev/null",
            "cwd": "/tmp",
            "hook_event_name": "UserPromptSubmit",
            "prompt": prompt,
        }
    )


def _set_db(monkeypatch: pytest.MonkeyPatch, db: Path) -> None:
    monkeypatch.setenv("AELFRICE_DB", str(db))


def test_hook_fire_writes_feedback_rows_tagged_hook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    s = MemoryStore(str(db))
    s.insert_belief(_mk("F1", "the kitchen is full of bananas"))
    s.insert_belief(_mk("F2", "bananas ripen on the counter"))
    s.insert_belief(_mk("F3", "unrelated content about coffee"))
    s.close()
    _set_db(monkeypatch, db)

    rc = user_prompt_submit(
        stdin=io.StringIO(_payload("tell me about the bananas in the kitchen")),
        stdout=io.StringIO(),
    )
    assert rc == 0

    s2 = MemoryStore(str(db))
    try:
        events = s2.list_feedback_events()
    finally:
        s2.close()

    hook_events = [e for e in events if e.source == HOOK_FEEDBACK_SOURCE]
    assert len(hook_events) >= 2
    assert all(e.valence == HOOK_RETRIEVAL_VALENCE for e in hook_events)
    assert {e.belief_id for e in hook_events} >= {"F1", "F2"}


def test_hook_fire_increments_alpha_not_beta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    s = MemoryStore(str(db))
    s.insert_belief(_mk("F1", "the kitchen is full of bananas"))
    s.close()
    _set_db(monkeypatch, db)

    user_prompt_submit(
        stdin=io.StringIO(_payload("are there bananas in the kitchen")),
        stdout=io.StringIO(),
    )

    s2 = MemoryStore(str(db))
    try:
        b = s2.get_belief("F1")
    finally:
        s2.close()
    assert b is not None
    assert b.alpha == 1.0 + HOOK_RETRIEVAL_VALENCE
    assert b.beta == 1.0


def test_hook_fire_does_not_pressure_locked_contradictors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The non-corrective contract: hook exposure of a belief that
    contradicts a locked belief must NOT pressure the locked belief.
    Otherwise every prompt that mentions the contradictor silently
    erodes the lock."""
    db = tmp_path / "memory.db"
    s = MemoryStore(str(db))
    s.insert_belief(_mk("CONTRADICTOR", "alpha is greater than beta"))
    s.insert_belief(
        _mk("LOCKED", "beta is greater than alpha",
            lock_level=LOCK_USER, locked_at="2026-04-26T01:00:00Z")
    )
    s.insert_edge(
        Edge(src="CONTRADICTOR", dst="LOCKED",
             type=EDGE_CONTRADICTS, weight=1.0)
    )
    s.close()
    _set_db(monkeypatch, db)

    # Fire 10 prompts that all hit CONTRADICTOR.
    for _ in range(10):
        user_prompt_submit(
            stdin=io.StringIO(_payload("is alpha greater than beta always")),
            stdout=io.StringIO(),
        )

    s2 = MemoryStore(str(db))
    try:
        locked = s2.get_belief("LOCKED")
    finally:
        s2.close()
    assert locked is not None
    assert locked.demotion_pressure == 0
    assert locked.lock_level == LOCK_USER


def test_hook_fire_records_locked_hits_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Locked beliefs auto-loaded by L0 are recorded just like L1 hits."""
    db = tmp_path / "memory.db"
    s = MemoryStore(str(db))
    s.insert_belief(
        _mk("L1", "always use scripts/publish.sh",
            lock_level=LOCK_USER, locked_at="2026-04-26T01:00:00Z")
    )
    s.close()
    _set_db(monkeypatch, db)

    user_prompt_submit(
        stdin=io.StringIO(_payload("which scripts should I use to publish")),
        stdout=io.StringIO(),
    )

    s2 = MemoryStore(str(db))
    try:
        events = s2.list_feedback_events()
    finally:
        s2.close()
    assert any(
        e.belief_id == "L1" and e.source == HOOK_FEEDBACK_SOURCE
        for e in events
    )


def test_hook_fire_no_match_writes_no_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    s = MemoryStore(str(db))
    s.insert_belief(_mk("F1", "the kitchen is full of bananas"))
    s.close()
    _set_db(monkeypatch, db)

    user_prompt_submit(
        stdin=io.StringIO(
            _payload("unrelated content about typewriters and jellyfish")
        ),
        stdout=io.StringIO(),
    )

    s2 = MemoryStore(str(db))
    try:
        events = s2.list_feedback_events()
    finally:
        s2.close()
    assert events == []
