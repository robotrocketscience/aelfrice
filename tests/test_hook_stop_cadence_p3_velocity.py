"""Integration tests for #876 p3_velocity dispatcher wiring — Stop side.

PR 3 of 5 in the #876 stack. Covers the new POLICY_P3_VELOCITY branch
in _maybe_fire_cadence_checkpoint: predicate evaluation, ring state
read, on-fire updates to both p3_velocity state slots, fail-soft
behaviour.
"""
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
        "AELF_AUTOLOCK_CORRECTIONS",
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
        "session_id": session_id,
        "ring": [],
        "ring_max": 200,
        "next_fire_idx": next_fire_idx,
        "evicted_total": 0,
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


def _stub_rebuild_and_format(
    monkeypatch: pytest.MonkeyPatch,
    body: str = "<rebuilder synthesis body>",
) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    def _stub(recent, token_budget, **kwargs):  # type: ignore[no-untyped-def]
        calls.append({"n_recent": len(recent), "token_budget": token_budget})
        return body

    monkeypatch.setattr(hook, "_rebuild_and_format", _stub)
    return calls


def _stub_read_recent(monkeypatch: pytest.MonkeyPatch, n_turns: int = 1) -> None:
    from aelfrice.context_rebuilder import RecentTurn
    canned = [RecentTurn(role="user", text=f"turn-{i}") for i in range(n_turns)]
    monkeypatch.setattr(
        hook, "_read_recent_for_pre_compact", lambda _payload, _n: canned,
    )


def _stop_payload(session_id: str, cwd: Path, transcript_path: Path) -> str:
    return json.dumps({
        "session_id": session_id,
        "cwd": str(cwd),
        "transcript_path": str(transcript_path),
    })


def _setup_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    next_fire_idx: int,
    bytes_at_last_fire: int,
    fire_idx_at_last_fire: int,
    transcript_bytes: int,
    threshold: int = 3000,
) -> tuple[io.StringIO, list, Path]:
    db = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    _seed_db(db)
    _write_ring_state(
        db.parent, "sess-V",
        next_fire_idx=next_fire_idx,
        bytes_at_last_fire=bytes_at_last_fire,
        fire_idx_at_last_fire=fire_idx_at_last_fire,
    )
    _write_toml(tmp_path, f"""
        [cadence]
        enabled = true
        policy = "p3_velocity"
        p3_velocity_threshold = {threshold}
    """)
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(transcript, transcript_bytes)
    calls = _stub_rebuild_and_format(monkeypatch)
    _stub_read_recent(monkeypatch)
    serr = io.StringIO()
    rc = hook.stop(
        stdin=io.StringIO(_stop_payload("sess-V", tmp_path, transcript)),
        stderr=serr,
    )
    assert rc == 0
    return serr, calls, db.parent


# --- Tests ----------------------------------------------------------------


def test_velocity_above_threshold_fires(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At density >= threshold the dispatch fires and updates state."""
    serr, calls, state_dir = _setup_test(
        tmp_path, monkeypatch,
        next_fire_idx=10, bytes_at_last_fire=0, fire_idx_at_last_fire=0,
        transcript_bytes=50_000, threshold=3000,
    )
    # 50k bytes / 10 turns = 5000 bytes/turn >= 3000 threshold
    assert len(calls) == 1, "rebuilder should fire"
    assert "cadence checkpoint fired" in serr.getvalue()
    assert "p3_velocity" in serr.getvalue()
    # State slots updated atomically
    persisted = json.loads(
        (state_dir / "session_injected_ids.json").read_text()
    )
    assert persisted["fire_idx_at_last_fire"] == 10
    assert persisted["bytes_at_last_fire"] >= 40_000  # actual transcript size


def test_velocity_below_threshold_does_not_fire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    serr, calls, state_dir = _setup_test(
        tmp_path, monkeypatch,
        next_fire_idx=10, bytes_at_last_fire=0, fire_idx_at_last_fire=0,
        transcript_bytes=10_000, threshold=3000,
    )
    # 10k / 10 turns = 1000 bytes/turn < 3000
    assert calls == []
    assert "cadence checkpoint" not in serr.getvalue()
    persisted = json.loads(
        (state_dir / "session_injected_ids.json").read_text()
    )
    # Unchanged
    assert persisted["bytes_at_last_fire"] == 0
    assert persisted["fire_idx_at_last_fire"] == 0


def test_velocity_zero_turns_since_last_fire_skips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Immediately after a fire (turns_since_last_fire == 0), no new fire."""
    serr, calls, _ = _setup_test(
        tmp_path, monkeypatch,
        next_fire_idx=10, bytes_at_last_fire=20_000, fire_idx_at_last_fire=10,
        transcript_bytes=50_000, threshold=3000,
    )
    assert calls == []


def test_velocity_disabled_no_fire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    _seed_db(db)
    _write_ring_state(
        db.parent, "sess-V", next_fire_idx=10,
        bytes_at_last_fire=0, fire_idx_at_last_fire=0,
    )
    # cadence section absent: enabled defaults False
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(transcript, 100_000)
    calls = _stub_rebuild_and_format(monkeypatch)
    _stub_read_recent(monkeypatch)
    serr = io.StringIO()
    rc = hook.stop(
        stdin=io.StringIO(_stop_payload("sess-V", tmp_path, transcript)),
        stderr=serr,
    )
    assert rc == 0
    assert calls == []


def test_velocity_env_threshold_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Env override pushes the threshold above the actual velocity → no fire."""
    db = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    _seed_db(db)
    _write_ring_state(
        db.parent, "sess-V", next_fire_idx=10,
        bytes_at_last_fire=0, fire_idx_at_last_fire=0,
    )
    _write_toml(tmp_path, """
        [cadence]
        enabled = true
        policy = "p3_velocity"
        p3_velocity_threshold = 1000
    """)
    monkeypatch.setenv(ENV_CADENCE_P3_VELOCITY_THRESHOLD, "10000")
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(transcript, 50_000)
    calls = _stub_rebuild_and_format(monkeypatch)
    _stub_read_recent(monkeypatch)
    serr = io.StringIO()
    rc = hook.stop(
        stdin=io.StringIO(_stop_payload("sess-V", tmp_path, transcript)),
        stderr=serr,
    )
    assert rc == 0
    # 50k/10 = 5000 bytes/turn; env threshold 10000 ⇒ no fire
    assert calls == []


def test_velocity_missing_session_state_skips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No ring file at all — predicate cannot read state, skips cleanly."""
    db = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    _seed_db(db)
    # NO _write_ring_state call
    _write_toml(tmp_path, """
        [cadence]
        enabled = true
        policy = "p3_velocity"
        p3_velocity_threshold = 1000
    """)
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(transcript, 50_000)
    calls = _stub_rebuild_and_format(monkeypatch)
    _stub_read_recent(monkeypatch)
    serr = io.StringIO()
    rc = hook.stop(
        stdin=io.StringIO(_stop_payload("sess-V", tmp_path, transcript)),
        stderr=serr,
    )
    assert rc == 0
    # next_fire_idx unset → turns_since_last_fire = 0 - 0 = 0 → no fire
    assert calls == []
