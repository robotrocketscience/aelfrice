"""Tests for `aelf tail` (#321) — reader/pretty-printer over hook_audit.jsonl.

Covers:
  - filter parser (valid / invalid / unknown key)
  - since parser (s/m/h/d, malformed)
  - record_matches_filters (hook + lane semantics, missing field falsifies)
  - format_record (header + per-belief lines, --no-blob)
  - tail_audit one-shot (--no-follow) over a seeded file
  - tail_audit follow mode picks up new lines
  - tail_audit follow mode survives rotation (inode flip)
  - tail_audit --since backfills both rotated and live
  - end-to-end: hook fires write, `aelf tail --no-follow` reads them
"""
from __future__ import annotations

import io
import json
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from aelfrice.hook import (
    AUDIT_FILENAME,
    AUDIT_ROTATED_SUFFIX,
    _audit_path_for_db,
    _write_hook_audit_record,
)
from aelfrice.hook_tail import (
    format_record,
    parse_filter,
    parse_since,
    record_matches_filters,
    tail_audit,
)
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
from aelfrice.store import MemoryStore


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _mk(
    bid: str,
    content: str,
    lock_level: str = LOCK_NONE,
) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=2.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=lock_level,
        locked_at=("2026-04-26T00:00:00Z" if lock_level == LOCK_USER else None),
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


def _write_record(
    audit_path: Path,
    *,
    hook: str = "user_prompt_submit",
    beliefs: list[Belief] | None = None,
    rendered_block: str = "<aelfrice-memory>...</aelfrice-memory>\n",
    latency_ms: int = 12,
    ts: str | None = None,
) -> None:
    """Write one audit record by calling `_write_hook_audit_record`. The
    ts override is only used for `--since` tests; default ts is now()."""
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    # Monkey the global db_path lookup by temporarily setting AELFRICE_DB
    # to the parent so _audit_path_for_db lands at the right place.
    bs = beliefs or []
    n_locked = sum(1 for b in bs if b.lock_level == LOCK_USER)
    # Going through the live writer keeps schema in sync with production.
    db = audit_path.parent / "memory.db"
    os.environ["AELFRICE_DB"] = str(db)
    _write_hook_audit_record(
        hook=hook,
        prompt="bananas",
        rendered_block=rendered_block,
        n_beliefs=len(bs),
        n_locked=n_locked,
        session_id="sid",
        beliefs=bs,
        latency_ms=latency_ms,
    )
    if ts is not None:
        # Patch the most-recent record's ts in place. Used by --since
        # tests that need an artificially-old record.
        lines = audit_path.read_text(encoding="utf-8").splitlines()
        last = json.loads(lines[-1])
        last["ts"] = ts
        lines[-1] = json.dumps(last)
        audit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# parse_filter
# ---------------------------------------------------------------------------


def test_parse_filter_valid_hook() -> None:
    assert parse_filter("hook=user_prompt_submit") == (
        "hook", "user_prompt_submit",
    )


def test_parse_filter_valid_lane() -> None:
    assert parse_filter("lane=L0") == ("lane", "L0")


def test_parse_filter_strips_whitespace() -> None:
    assert parse_filter("  hook = session_start  ") == (
        "hook", "session_start",
    )


def test_parse_filter_no_equals_raises() -> None:
    with pytest.raises(ValueError, match="key=value"):
        parse_filter("hook")


def test_parse_filter_unknown_key_raises() -> None:
    with pytest.raises(ValueError, match="not recognized"):
        parse_filter("score=0.5")


def test_parse_filter_empty_value_raises() -> None:
    with pytest.raises(ValueError, match="no value"):
        parse_filter("hook=")


# ---------------------------------------------------------------------------
# parse_since
# ---------------------------------------------------------------------------


def test_parse_since_seconds() -> None:
    assert parse_since("30s") == timedelta(seconds=30)


def test_parse_since_minutes() -> None:
    assert parse_since("5m") == timedelta(minutes=5)


def test_parse_since_hours() -> None:
    assert parse_since("2h") == timedelta(hours=2)


def test_parse_since_days() -> None:
    assert parse_since("1d") == timedelta(days=1)


def test_parse_since_malformed_raises() -> None:
    with pytest.raises(ValueError):
        parse_since("five-minutes")


def test_parse_since_missing_unit_raises() -> None:
    with pytest.raises(ValueError):
        parse_since("30")


# ---------------------------------------------------------------------------
# record_matches_filters
# ---------------------------------------------------------------------------


def test_filter_hook_match() -> None:
    rec = {"hook": "user_prompt_submit"}
    assert record_matches_filters(rec, [("hook", "user_prompt_submit")])
    assert not record_matches_filters(rec, [("hook", "session_start")])


def test_filter_lane_match_when_belief_present() -> None:
    rec = {
        "hook": "user_prompt_submit",
        "beliefs": [{"lane": "L0"}, {"lane": "L1"}],
    }
    assert record_matches_filters(rec, [("lane", "L0")])
    assert record_matches_filters(rec, [("lane", "L1")])
    assert not record_matches_filters(rec, [("lane", "L2")])


def test_filter_lane_no_beliefs_field_falsifies() -> None:
    rec = {"hook": "user_prompt_submit"}
    assert not record_matches_filters(rec, [("lane", "L0")])


def test_filter_compound_AND() -> None:
    rec = {
        "hook": "session_start",
        "beliefs": [{"lane": "L0"}],
    }
    assert record_matches_filters(
        rec, [("hook", "session_start"), ("lane", "L0")],
    )
    assert not record_matches_filters(
        rec, [("hook", "session_start"), ("lane", "L1")],
    )


# ---------------------------------------------------------------------------
# format_record
# ---------------------------------------------------------------------------


def test_format_record_header_and_per_belief_lines() -> None:
    rec = {
        "ts": "2026-05-02T14:22:11Z",
        "hook": "user_prompt_submit",
        "tokens": 412,
        "latency_ms": 18,
        "beliefs": [
            {
                "id": "ab96e9d3501b1c14",
                "lane": "L0",
                "locked": True,
                "snippet": "Content sourced from somewhere",
            },
            {
                "id": "032197a1cccccccc",
                "lane": "L1",
                "locked": False,
                "snippet": "BM25 hit body",
            },
        ],
    }
    text = format_record(rec, include_blob=True)
    lines = text.splitlines()
    # Header carries hook, tokens, latency, lane counts.
    assert "user_prompt_submit" in lines[0]
    assert "412 tok" in lines[0]
    assert "18 ms" in lines[0]
    assert "L0×1 L1×1" in lines[0]
    # Per-belief lines: short id, lane, snippet.
    assert any("ab96e9d3" in line and "L0" in line for line in lines[1:])
    assert any("locked" in line for line in lines[1:])
    assert any("BM25 hit body" in line for line in lines[1:])


def test_format_record_no_blob_omits_snippets() -> None:
    rec = {
        "ts": "2026-05-02T14:22:11Z",
        "hook": "user_prompt_submit",
        "tokens": 100,
        "latency_ms": 5,
        "beliefs": [
            {
                "id": "abc",
                "lane": "L0",
                "locked": True,
                "snippet": "secret content body",
            },
        ],
    }
    text = format_record(rec, include_blob=False)
    assert "secret content body" not in text
    assert "abc" in text  # id still shown
    assert "L0" in text


def test_format_record_handles_missing_optional_fields() -> None:
    """Backward-compatible: pre-#321 records lack beliefs/tokens/latency."""
    rec = {
        "ts": "2026-05-02T14:22:11Z",
        "hook": "user_prompt_submit",
        "n_beliefs": 0,
        "n_locked": 0,
    }
    text = format_record(rec, include_blob=True)
    # Header still emits without crashing.
    assert "user_prompt_submit" in text
    # No "tok" / "ms" tokens since fields are missing.
    assert "tok" not in text
    assert " ms" not in text


# ---------------------------------------------------------------------------
# tail_audit one-shot (no follow)
# ---------------------------------------------------------------------------


def test_tail_no_follow_dumps_current_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "alpha")])
    monkeypatch.setenv("AELFRICE_DB", str(db))
    audit_path = _audit_path_for_db(db)
    monkeypatch.delenv("AELFRICE_HOOK_AUDIT", raising=False)

    _write_record(audit_path, beliefs=[_mk("B1", "first")])
    _write_record(audit_path, beliefs=[_mk("B2", "second")])

    out = io.StringIO()
    rc = tail_audit(
        audit_path=audit_path, follow=False, out=out,
    )
    assert rc == 0
    text = out.getvalue()
    assert "B1" in text
    assert "B2" in text


def test_tail_no_follow_filter_hook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "alpha")])
    monkeypatch.setenv("AELFRICE_DB", str(db))
    audit_path = _audit_path_for_db(db)
    monkeypatch.delenv("AELFRICE_HOOK_AUDIT", raising=False)

    _write_record(audit_path, hook="user_prompt_submit",
                  beliefs=[_mk("X1", "ups body")])
    _write_record(audit_path, hook="session_start",
                  beliefs=[_mk("X2", "ss body")])

    out = io.StringIO()
    tail_audit(
        audit_path=audit_path,
        filters=[("hook", "session_start")],
        follow=False,
        out=out,
    )
    text = out.getvalue()
    assert "X2" in text
    assert "X1" not in text


def test_tail_no_follow_filter_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "alpha")])
    monkeypatch.setenv("AELFRICE_DB", str(db))
    audit_path = _audit_path_for_db(db)
    monkeypatch.delenv("AELFRICE_HOOK_AUDIT", raising=False)

    _write_record(
        audit_path,
        beliefs=[_mk("L1ONLY", "unlocked", lock_level=LOCK_NONE)],
    )
    _write_record(
        audit_path,
        beliefs=[_mk("L0HIT", "locked one", lock_level=LOCK_USER)],
    )

    out = io.StringIO()
    tail_audit(
        audit_path=audit_path,
        filters=[("lane", "L0")],
        follow=False,
        out=out,
    )
    text = out.getvalue()
    assert "L0HIT" in text
    assert "L1ONLY" not in text


# ---------------------------------------------------------------------------
# tail_audit follow mode
# ---------------------------------------------------------------------------


def test_tail_follow_emits_new_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Seed one record, then run tail_audit in follow mode in a thread.
    Write two more records and assert they show up in the output."""
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "alpha")])
    monkeypatch.setenv("AELFRICE_DB", str(db))
    audit_path = _audit_path_for_db(db)
    monkeypatch.delenv("AELFRICE_HOOK_AUDIT", raising=False)

    _write_record(audit_path, beliefs=[_mk("OLD", "before tail")])

    out = io.StringIO()

    def _run() -> None:
        tail_audit(
            audit_path=audit_path, follow=True, out=out,
            poll_interval=0.02, max_iters=80,
        )

    th = threading.Thread(target=_run)
    th.start()
    time.sleep(0.05)  # let tail_audit reach end-of-file
    _write_record(audit_path, beliefs=[_mk("NEW1", "after tail")])
    time.sleep(0.05)
    _write_record(audit_path, beliefs=[_mk("NEW2", "even later")])
    th.join(timeout=4.0)
    text = out.getvalue()
    # OLD was already in the file when tail started → should NOT show.
    assert "OLD" not in text
    assert "NEW1" in text
    assert "NEW2" in text


def test_tail_follow_survives_rotation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulate a rotation: rename live → .1 mid-tail, recreate live,
    write a new record. The follower must pick it up."""
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "alpha")])
    monkeypatch.setenv("AELFRICE_DB", str(db))
    audit_path = _audit_path_for_db(db)
    monkeypatch.delenv("AELFRICE_HOOK_AUDIT", raising=False)

    _write_record(audit_path, beliefs=[_mk("PRE", "pre-rotation")])

    out = io.StringIO()

    def _run() -> None:
        tail_audit(
            audit_path=audit_path, follow=True, out=out,
            poll_interval=0.02, max_iters=120,
        )

    th = threading.Thread(target=_run)
    th.start()
    time.sleep(0.05)

    # Rotate: rename live to .1, then write a fresh live record.
    rotated = audit_path.with_name(audit_path.name + AUDIT_ROTATED_SUFFIX)
    os.replace(audit_path, rotated)
    _write_record(audit_path, beliefs=[_mk("POSTROT", "after rotation")])
    th.join(timeout=4.0)
    text = out.getvalue()
    assert "POSTROT" in text


# ---------------------------------------------------------------------------
# tail_audit --since
# ---------------------------------------------------------------------------


def test_tail_since_backfills_recent_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "alpha")])
    monkeypatch.setenv("AELFRICE_DB", str(db))
    audit_path = _audit_path_for_db(db)
    monkeypatch.delenv("AELFRICE_HOOK_AUDIT", raising=False)

    # An old record (10 hours ago) and a fresh one.
    old_ts = (
        datetime.now(timezone.utc) - timedelta(hours=10)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    _write_record(
        audit_path, beliefs=[_mk("ANCIENT", "old")], ts=old_ts,
    )
    _write_record(
        audit_path, beliefs=[_mk("RECENT", "new")],
    )

    out = io.StringIO()
    tail_audit(
        audit_path=audit_path, since=timedelta(minutes=30),
        follow=False, out=out,
    )
    text = out.getvalue()
    assert "RECENT" in text
    assert "ANCIENT" not in text


def test_tail_since_reads_rotated_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--since` should backfill from .1 (rotated) AND live."""
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "alpha")])
    monkeypatch.setenv("AELFRICE_DB", str(db))
    audit_path = _audit_path_for_db(db)
    monkeypatch.delenv("AELFRICE_HOOK_AUDIT", raising=False)

    # Put one record in .1 (rotated), one in live, both within window.
    _write_record(audit_path, beliefs=[_mk("ROT", "in rotated")])
    rotated = audit_path.with_name(audit_path.name + AUDIT_ROTATED_SUFFIX)
    os.replace(audit_path, rotated)
    _write_record(audit_path, beliefs=[_mk("LIVE", "in live")])

    out = io.StringIO()
    tail_audit(
        audit_path=audit_path, since=timedelta(hours=1),
        follow=False, out=out,
    )
    text = out.getvalue()
    assert "ROT" in text
    assert "LIVE" in text


# ---------------------------------------------------------------------------
# end-to-end: hook fire → tail reads
# ---------------------------------------------------------------------------


def test_end_to_end_hook_fire_then_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real hook fire must produce a record that `aelf tail` renders."""
    from aelfrice.hook import user_prompt_submit

    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("HIT", "the kitchen is full of bananas")])
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.delenv("AELFRICE_HOOK_AUDIT", raising=False)
    audit_path = _audit_path_for_db(db)

    payload = json.dumps({
        "session_id": "S",
        "transcript_path": "/dev/null",
        "cwd": "/tmp",
        "hook_event_name": "UserPromptSubmit",
        "prompt": "what is in the kitchen full of bananas",
    })
    rc = user_prompt_submit(stdin=io.StringIO(payload), stdout=io.StringIO())
    assert rc == 0

    out = io.StringIO()
    rc = tail_audit(audit_path=audit_path, follow=False, out=out)
    assert rc == 0
    text = out.getvalue()
    # Header lane counts derived from beliefs[].
    assert "L1×1" in text
    assert "user_prompt_submit" in text
    # Per-belief snippet visible.
    assert "kitchen is full of bananas" in text
    assert "HIT" in text
