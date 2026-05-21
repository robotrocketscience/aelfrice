"""Integration tests for #876 p3_velocity dispatcher wiring — UPS side."""
from __future__ import annotations

import io
import json
import textwrap
from pathlib import Path

import pytest

from aelfrice import hook
from aelfrice.cadence import (
    CONFIG_FILENAME,
    ENV_CADENCE_P3_VELOCITY_THRESHOLD,
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "AELFRICE_CADENCE_ENABLED",
        "AELFRICE_CADENCE_POLICY",
        ENV_CADENCE_P3_VELOCITY_THRESHOLD,
    ):
        monkeypatch.delenv(var, raising=False)


def _seed_db(db: Path) -> None:
    from aelfrice.store import MemoryStore
    MemoryStore(str(db)).close()


def _write_toml(repo: Path, body: str) -> None:
    (repo / CONFIG_FILENAME).write_text(textwrap.dedent(body))


def _write_ring_state(
    state_dir: Path,
    session_id: str,
    *,
    next_fire_idx: int,
    bytes_at_last_fire: int = 0,
    fire_idx_at_last_fire: int = 0,
) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "session_id": session_id, "ring": [], "ring_max": 200,
        "next_fire_idx": next_fire_idx, "evicted_total": 0,
        "bytes_at_last_fire": bytes_at_last_fire,
        "fire_idx_at_last_fire": fire_idx_at_last_fire,
        "classifications": [],
    }
    (state_dir / "session_injected_ids.json").write_text(json.dumps(payload))


def _write_transcript(path: Path, size_bytes: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    filler = json.dumps({
        "message": {"role": "assistant", "content": "x" * 800},
    }) + "\n"
    n = max(1, size_bytes // len(filler))
    with path.open("w", encoding="utf-8") as f:
        for _ in range(n):
            f.write(filler)


def _stub_rebuild_and_format(monkeypatch: pytest.MonkeyPatch) -> list:
    calls: list = []

    def _stub(recent, token_budget, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(1)
        return "<body>"

    monkeypatch.setattr(hook, "_rebuild_and_format", _stub)
    return calls


def _stub_read_recent(monkeypatch: pytest.MonkeyPatch) -> None:
    from aelfrice.context_rebuilder import RecentTurn
    monkeypatch.setattr(
        hook, "_read_recent_for_pre_compact",
        lambda _payload, _n: [RecentTurn(role="user", text="t")],
    )


def test_ups_velocity_fires_returns_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    _seed_db(db)
    _write_ring_state(
        db.parent, "sess-U", next_fire_idx=10,
        bytes_at_last_fire=0, fire_idx_at_last_fire=0,
    )
    _write_toml(tmp_path, """
        [cadence]
        enabled = true
        policy = "p3_velocity"
        p3_velocity_threshold = 3000
    """)
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(transcript, 50_000)
    calls = _stub_rebuild_and_format(monkeypatch)
    _stub_read_recent(monkeypatch)
    payload = {
        "session_id": "sess-U",
        "cwd": str(tmp_path),
        "transcript_path": str(transcript),
    }
    serr = io.StringIO()
    body = hook._maybe_run_ups_cadence_checkpoint(payload, "sess-U", serr)
    assert body == "<body>"
    assert len(calls) == 1
    assert "ups cadence checkpoint fired" in serr.getvalue()
    assert "p3_velocity" in serr.getvalue()


def test_ups_velocity_below_threshold_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    _seed_db(db)
    _write_ring_state(
        db.parent, "sess-U", next_fire_idx=10,
        bytes_at_last_fire=0, fire_idx_at_last_fire=0,
    )
    _write_toml(tmp_path, """
        [cadence]
        enabled = true
        policy = "p3_velocity"
        p3_velocity_threshold = 3000
    """)
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(transcript, 10_000)
    calls = _stub_rebuild_and_format(monkeypatch)
    _stub_read_recent(monkeypatch)
    payload = {
        "session_id": "sess-U",
        "cwd": str(tmp_path),
        "transcript_path": str(transcript),
    }
    serr = io.StringIO()
    body = hook._maybe_run_ups_cadence_checkpoint(payload, "sess-U", serr)
    assert body is None
    assert calls == []


def test_ups_velocity_state_update_on_fire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    _seed_db(db)
    _write_ring_state(
        db.parent, "sess-U", next_fire_idx=15,
        bytes_at_last_fire=5_000, fire_idx_at_last_fire=5,
    )
    _write_toml(tmp_path, """
        [cadence]
        enabled = true
        policy = "p3_velocity"
        p3_velocity_threshold = 1000
    """)
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(transcript, 50_000)
    _stub_rebuild_and_format(monkeypatch)
    _stub_read_recent(monkeypatch)
    payload = {
        "session_id": "sess-U",
        "cwd": str(tmp_path),
        "transcript_path": str(transcript),
    }
    hook._maybe_run_ups_cadence_checkpoint(payload, "sess-U", io.StringIO())
    persisted = json.loads(
        (db.parent / "session_injected_ids.json").read_text()
    )
    assert persisted["fire_idx_at_last_fire"] == 15
    assert persisted["bytes_at_last_fire"] > 5_000


def test_ups_velocity_empty_session_id_skips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    serr = io.StringIO()
    body = hook._maybe_run_ups_cadence_checkpoint(
        {"cwd": str(tmp_path)}, "", serr,
    )
    assert body is None
