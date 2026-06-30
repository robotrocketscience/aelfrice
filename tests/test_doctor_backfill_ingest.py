"""Acceptance tests for `aelf doctor --backfill-ingest` (#1011).

Folds the canonical `turns.jsonl` into beliefs — the companion fix to
the #1034 `ingest_gap` warning. Idempotent (dedupes per source+sentence,
default `transcript` label, matching the Stop/PreCompact flush), and it
does not rotate the live log.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

import aelfrice.cli as cli_module
from aelfrice.store import MemoryStore


def _turns_path(db: Path) -> Path:
    p = db.parent / "transcripts" / "turns.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _write_turns(db: Path, lines: list[dict[str, object]]) -> Path:
    p = _turns_path(db)
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    return p


def _run(monkeypatch: pytest.MonkeyPatch, db: Path) -> tuple[int, str]:
    monkeypatch.setenv("AELFRICE_DB", str(db))
    buf = io.StringIO()
    code = cli_module.main(argv=["doctor", "--backfill-ingest"], out=buf)
    return code, buf.getvalue()


def _active_count(db: Path) -> int:
    s = MemoryStore(str(db))
    try:
        return len(s.list_active_beliefs())
    finally:
        s.close()


def test_backfill_ingests_logged_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    MemoryStore(str(db)).close()
    _write_turns(db, [
        {"role": "user", "text": "The widget factory ships blue gadgets on Tuesdays.",
         "session_id": "s1", "ts": "2026-06-30T10:00:00+00:00"},
        {"role": "user", "text": "The mascot for the project is a friendly otter.",
         "session_id": "s1", "ts": "2026-06-30T10:01:00+00:00"},
    ])
    assert _active_count(db) == 0
    code, out = _run(monkeypatch, db)
    assert code == 0
    assert "backfill-ingest:" in out
    assert _active_count(db) > 0


def test_backfill_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    MemoryStore(str(db)).close()
    _write_turns(db, [
        {"role": "user", "text": "The garden has twelve tomato plants this season.",
         "session_id": "s1", "ts": "2026-06-30T10:00:00+00:00"},
    ])
    _run(monkeypatch, db)
    after_first = _active_count(db)
    code, out = _run(monkeypatch, db)
    assert code == 0
    assert "0 new belief(s)" in out
    assert _active_count(db) == after_first  # no inflation


def test_backfill_no_log_is_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    MemoryStore(str(db)).close()
    code, out = _run(monkeypatch, db)
    assert code == 0
    assert "no transcript log" in out
