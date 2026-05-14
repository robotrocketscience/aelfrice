"""PreToolUse search-tool hook dedup against session-ring (#740)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from aelfrice import hook_search_tool as hk
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
from aelfrice.session_ring import append_ids, read_ring_state
from aelfrice.store import MemoryStore


@pytest.fixture
def per_repo_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    return db


def _payload(
    *,
    pattern: str,
    tool_name: str = "Grep",
    cwd: str | None = None,
    session_id: str = "sess-A",
) -> dict[str, object]:
    return {
        "hook_event_name": "PreToolUse",
        "tool_name": tool_name,
        "tool_input": {"pattern": pattern},
        "cwd": cwd or "/tmp",
        "session_id": session_id,
    }


def _run(payload: dict[str, object]) -> tuple[int, str, str]:
    sin = io.StringIO(json.dumps(payload))
    sout = io.StringIO()
    serr = io.StringIO()
    rc = hk.main(stdin=sin, stdout=sout, stderr=serr)
    return rc, sout.getvalue(), serr.getvalue()


def _seed(db_path: Path, bid: str, content: str, *, locked: bool = False) -> str:
    b = Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_USER if locked else LOCK_NONE,
        locked_at="2026-05-13T00:00:00Z" if locked else None,
        created_at="2026-05-13T00:00:00Z",
        last_retrieved_at=None,
    )
    store = MemoryStore(str(db_path))
    try:
        store.insert_belief(b)
    finally:
        store.close()
    return bid


def _outer_payload(out: str) -> dict[str, object]:
    return json.loads(out)


def test_first_fire_emits_full_block_and_populates_ring(
    per_repo_db: Path,
) -> None:
    _seed(per_repo_db, "DEDUPA1", "delta delta delta pattern marker")
    rc, out, _ = _run(_payload(pattern="delta-pattern"))
    assert rc == 0
    ctx = _outer_payload(out)["hookSpecificOutput"]["additionalContext"]
    assert "DEDUPA1"[:16] in ctx
    assert "note=" not in ctx  # not a dedup-pointer block
    state = read_ring_state("sess-A")
    ids = {e["id"] for e in state["ring"]}
    assert "DEDUPA1" in ids


def test_repeat_fire_emits_pointer_when_all_recent(
    per_repo_db: Path,
) -> None:
    _seed(per_repo_db, "DEDUPB1", "echo echo echo pattern marker")
    # Pre-populate the ring as if a UPS fire already injected DEDUPB1.
    append_ids("sess-A", ["DEDUPB1"])
    rc, out, _ = _run(_payload(pattern="echo-pattern"))
    assert rc == 0
    ctx = _outer_payload(out)["hookSpecificOutput"]["additionalContext"]
    # Expect the pointer form: no body listing, with note + suppressed.
    assert 'note="answer already in prompt context"' in ctx
    assert 'suppressed="1"' in ctx
    # Belief content should NOT appear (it was suppressed).
    assert "echo echo echo" not in ctx


def test_mixed_fire_emits_new_with_trailing_count(
    per_repo_db: Path,
) -> None:
    _seed(per_repo_db, "MIXOLD1", "foxtrot foxtrot match alpha distinct")
    _seed(per_repo_db, "MIXNEW1", "foxtrot foxtrot match beta distinct")
    append_ids("sess-A", ["MIXOLD1"])  # one already injected
    rc, out, _ = _run(_payload(pattern="foxtrot-match"))
    assert rc == 0
    ctx = _outer_payload(out)["hookSpecificOutput"]["additionalContext"]
    # The new belief is rendered; the old one is suppressed.
    assert "MIXNEW1"[:16] in ctx
    assert "MIXOLD1"[:16] not in ctx
    assert "already in prompt" in ctx  # trailing count phrase
    state = read_ring_state("sess-A")
    ids = {e["id"] for e in state["ring"]}
    assert {"MIXOLD1", "MIXNEW1"} <= ids


def test_locked_belief_always_re_emits(per_repo_db: Path) -> None:
    _seed(per_repo_db, "LK01", "golf golf golf locked surface marker", locked=True)
    # Even if the locked id is already in the ring, it must re-emit.
    append_ids("sess-A", ["LK01"], locked_ids={"LK01"})
    rc, out, _ = _run(_payload(pattern="golf-locked"))
    assert rc == 0
    ctx = _outer_payload(out)["hookSpecificOutput"]["additionalContext"]
    assert "LK01" in ctx
    assert "[L0]" in ctx


def test_new_session_id_clears_dedup(per_repo_db: Path) -> None:
    _seed(per_repo_db, "HOTEL1", "hotel hotel hotel pattern marker indelible")
    append_ids("sess-A", ["HOTEL1"])
    # Fire under sess-B — sess-A's ring is stale, so HOTEL1 re-emits.
    rc, out, _ = _run(_payload(pattern="hotel-pattern", session_id="sess-B"))
    assert rc == 0
    ctx = _outer_payload(out)["hookSpecificOutput"]["additionalContext"]
    assert "HOTEL1"[:16] in ctx
    assert "note=" not in ctx
    state = read_ring_state("sess-B")
    ids = {e["id"] for e in state["ring"]}
    assert "HOTEL1" in ids
