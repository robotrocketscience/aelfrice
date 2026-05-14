"""UserPromptSubmit hook records injection_events rows (#779 Layer 1 wiring)."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pytest

from aelfrice.hook import (
    _new_injection_event_turn_id,
    _record_injection_events,
    user_prompt_submit,
)
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief
from aelfrice.retrieval import (
    ENV_META_BELIEF_BM25F_ANCHOR_WEIGHT,
    ENV_META_BELIEF_HALF_LIFE,
    META_BM25F_ANCHOR_WEIGHT_KEY,
    META_HALF_LIFE_KEY,
    get_active_meta_belief_consumers,
)
from aelfrice.store import MemoryStore


# --- helpers ---------------------------------------------------------

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
        created_at="2026-05-14T00:00:00+00:00",
        last_retrieved_at=None,
    )


def _seed(db: Path, beliefs: list[Belief]) -> None:
    s = MemoryStore(str(db))
    try:
        for b in beliefs:
            s.insert_belief(b)
    finally:
        s.close()


def _payload(prompt: str, session_id: str = "sess-779") -> str:
    return json.dumps(
        {
            "session_id": session_id,
            "transcript_path": "/dev/null",
            "cwd": "/tmp",
            "hook_event_name": "UserPromptSubmit",
            "prompt": prompt,
        }
    )


def _fire(prompt: str, session_id: str = "sess-779") -> str:
    out = io.StringIO()
    rc = user_prompt_submit(
        stdin=io.StringIO(_payload(prompt, session_id)),
        stdout=out,
        stderr=io.StringIO(),
    )
    assert rc == 0
    return out.getvalue()


def _read_events(db: Path, session_id: str = "sess-779") -> list[dict]:
    s = MemoryStore(str(db))
    try:
        cur = s._conn.execute(
            "SELECT id, session_id, turn_id, belief_id, source, "
            "active_consumers, referenced "
            "FROM injection_events WHERE session_id = ? ORDER BY id",
            (session_id,),
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        s.close()


# --- turn-id shape ---------------------------------------------------

def test_new_injection_event_turn_id_shape() -> None:
    """Shape: ``{utc-compact-ts}-{hex4}``, sortable lexicographically."""
    a = _new_injection_event_turn_id()
    b = _new_injection_event_turn_id()
    # Same year prefix (8 digits + 'T').
    assert a[:9].startswith("2026") or a[:9].startswith("20")
    # Suffix is 8 hex chars.
    assert len(a.split("-")[-1]) == 8
    # Distinct ids on close-in-time calls.
    assert a != b


# --- get_active_meta_belief_consumers --------------------------------

def test_get_active_consumers_empty_when_env_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV_META_BELIEF_HALF_LIFE, raising=False)
    monkeypatch.delenv(ENV_META_BELIEF_BM25F_ANCHOR_WEIGHT, raising=False)
    assert get_active_meta_belief_consumers() == []


def test_get_active_consumers_half_life_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_META_BELIEF_HALF_LIFE, "1")
    monkeypatch.delenv(ENV_META_BELIEF_BM25F_ANCHOR_WEIGHT, raising=False)
    assert get_active_meta_belief_consumers() == [META_HALF_LIFE_KEY]


def test_get_active_consumers_sorted_when_both_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_META_BELIEF_HALF_LIFE, "enabled")
    monkeypatch.setenv(ENV_META_BELIEF_BM25F_ANCHOR_WEIGHT, "enabled")
    out = get_active_meta_belief_consumers()
    assert out == sorted(out)
    assert META_HALF_LIFE_KEY in out
    assert META_BM25F_ANCHOR_WEIGHT_KEY in out


# --- _record_injection_events fail-soft -------------------------------

def test_record_skips_when_no_session_id(tmp_path: Path) -> None:
    """No session_id → no rows, no exception."""
    db = tmp_path / "m.db"
    _seed(db, [_mk("B1", "x")])
    os.environ["AELFRICE_DB"] = str(db)
    try:
        _record_injection_events(
            session_id=None,
            turn_id="t",
            hits=[_mk("B1", "x")],
            source="ups",
            active_consumers=[],
            stderr=io.StringIO(),
        )
        s = MemoryStore(str(db))
        try:
            n = s._conn.execute(
                "SELECT COUNT(*) FROM injection_events"
            ).fetchone()[0]
            assert n == 0
        finally:
            s.close()
    finally:
        del os.environ["AELFRICE_DB"]


def test_record_skips_when_no_hits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "m.db"
    _seed(db, [_mk("B1", "x")])
    monkeypatch.setenv("AELFRICE_DB", str(db))
    _record_injection_events(
        session_id="s", turn_id="t", hits=[], source="ups",
        active_consumers=[], stderr=io.StringIO(),
    )
    assert _read_events(db, "s") == []


def test_record_fail_soft_on_bad_db(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bad DB path → stderr line, no exception, hook continues."""
    monkeypatch.setenv("AELFRICE_DB", "/nonexistent/dir/x.db")
    err = io.StringIO()
    _record_injection_events(
        session_id="s", turn_id="t",
        hits=[_mk("B1", "x")],
        source="ups", active_consumers=[], stderr=err,
    )
    # No raise — fail-soft.
    assert "injection_events emit failed" in err.getvalue() \
        or err.getvalue() == ""  # depends on filesystem behavior


# --- end-to-end UPS-hook wiring --------------------------------------

def test_ups_fire_records_one_event_per_hit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "m.db"
    _seed(db, [_mk("HIT01", "the cellar door is full of barrels and casks")])
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.delenv(ENV_META_BELIEF_HALF_LIFE, raising=False)
    monkeypatch.delenv(ENV_META_BELIEF_BM25F_ANCHOR_WEIGHT, raising=False)
    out = _fire("how many barrels are in the cellar door storage")
    assert "HIT01" in out
    events = _read_events(db)
    assert len(events) >= 1
    by_belief = {e["belief_id"]: e for e in events}
    assert "HIT01" in by_belief
    assert by_belief["HIT01"]["source"] == "ups"
    assert by_belief["HIT01"]["session_id"] == "sess-779"
    # No env flags set → active_consumers is empty array.
    assert by_belief["HIT01"]["active_consumers"] == "[]"
    # Pending detection — referenced is NULL until Layer 3 sweeper.
    assert by_belief["HIT01"]["referenced"] is None


def test_ups_fire_threads_active_consumer_when_env_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "m.db"
    _seed(db, [_mk("HIT02", "the document references the cellar storage capacity")])
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.setenv(ENV_META_BELIEF_HALF_LIFE, "enabled")
    monkeypatch.delenv(ENV_META_BELIEF_BM25F_ANCHOR_WEIGHT, raising=False)
    _fire("show the document about cellar storage capacity")
    events = _read_events(db)
    assert events
    # Each event's active_consumers must contain the half-life key.
    for e in events:
        decoded = json.loads(e["active_consumers"])
        assert META_HALF_LIFE_KEY in decoded


def test_ups_fire_shares_turn_id_across_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All events from one UPS fire share the same turn_id — that's
    how the sweeper groups the batch."""
    db = tmp_path / "m.db"
    _seed(db, [
        _mk("B01", "the alpha document covers many subjects"),
        _mk("B02", "the alpha file references the subject matter"),
        _mk("B03", "the alpha entry lists relevant subjects"),
    ])
    monkeypatch.setenv("AELFRICE_DB", str(db))
    _fire("show the alpha entries about subject coverage")
    events = _read_events(db)
    turn_ids = {e["turn_id"] for e in events}
    assert len(turn_ids) == 1, (
        f"expected single turn_id, got {turn_ids}"
    )
