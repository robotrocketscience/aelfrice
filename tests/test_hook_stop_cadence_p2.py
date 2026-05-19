"""Integration tests for P2 cadence + resume cache in the Stop hook (#871).

P1 integration tests live in :mod:`tests.test_hook_stop_cadence` and
remain unchanged. This module covers the new P2 dispatch branch and
the shared resume-cache writer.
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
    ENV_CADENCE_CTX_BYTE_WINDOW,
    ENV_CADENCE_CTX_THRESHOLD,
    POLICY_P1_EVERY_K_TURNS,
    POLICY_P2_CTX_THRESHOLD,
)


# --- Fixtures -------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "AELFRICE_CADENCE_ENABLED",
        "AELFRICE_CADENCE_POLICY",
        "AELFRICE_CADENCE_K",
        ENV_CADENCE_CTX_THRESHOLD,
        ENV_CADENCE_CTX_BYTE_WINDOW,
        "AELF_AUTOLOCK_CORRECTIONS",
    ):
        monkeypatch.delenv(var, raising=False)


def _set_db(monkeypatch: pytest.MonkeyPatch, db: Path) -> None:
    monkeypatch.setenv("AELFRICE_DB", str(db))


def _seed_db(db: Path) -> None:
    from aelfrice.store import MemoryStore
    MemoryStore(str(db)).close()


def _write_toml(repo: Path, body: str) -> None:
    (repo / CONFIG_FILENAME).write_text(textwrap.dedent(body))


def _stub_rebuild_and_format(
    monkeypatch: pytest.MonkeyPatch,
    body: str = "<rebuilder synthesis body>",
) -> list[dict[str, object]]:
    """Replace ``_rebuild_and_format`` with a recorder; return calls."""
    calls: list[dict[str, object]] = []

    def _stub(recent, token_budget, **kwargs):  # type: ignore[no-untyped-def]
        calls.append({
            "n_recent": len(recent),
            "token_budget": token_budget,
            **kwargs,
        })
        return body

    monkeypatch.setattr(hook, "_rebuild_and_format", _stub)
    return calls


def _stub_read_recent(monkeypatch: pytest.MonkeyPatch, n_turns: int = 1) -> None:
    from aelfrice.context_rebuilder import RecentTurn

    canned = [
        RecentTurn(role="user", text=f"turn-{i}")
        for i in range(n_turns)
    ]
    monkeypatch.setattr(
        hook, "_read_recent_for_pre_compact", lambda _payload, _n: canned,
    )


def _write_transcript(path: Path, size_bytes: int, last_user_prompt: str) -> None:
    """Write a transcript jsonl of approximately ``size_bytes`` ending
    with a user-role line containing ``last_user_prompt``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Fill with assistant noise to inflate the byte count.
    filler_line = json.dumps({
        "message": {
            "role": "assistant",
            "content": "x" * 800,
        },
    }) + "\n"
    last_line = json.dumps({
        "message": {"role": "user", "content": last_user_prompt},
    }) + "\n"
    n_fillers = max(0, (size_bytes - len(last_line)) // len(filler_line))
    with path.open("w", encoding="utf-8") as f:
        for _ in range(n_fillers):
            f.write(filler_line)
        f.write(last_line)


def _stop_payload(
    session_id: str,
    cwd: Path,
    transcript_path: Path | None = None,
) -> str:
    payload: dict[str, object] = {"session_id": session_id, "cwd": str(cwd)}
    if transcript_path is not None:
        payload["transcript_path"] = str(transcript_path)
    return json.dumps(payload)


# --- P2 firing predicate at the Stop boundary ----------------------------


def test_p2_below_threshold_does_not_fire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    _seed_db(db)
    _write_toml(tmp_path, """
        [cadence]
        enabled = true
        policy = "p2_ctx_threshold"
        ctx_threshold = 0.5
        ctx_byte_window = 600000
    """)
    transcript = tmp_path / "transcript.jsonl"
    # Far below 300k watermark.
    _write_transcript(transcript, size_bytes=10_000, last_user_prompt="done")
    calls = _stub_rebuild_and_format(monkeypatch)
    _stub_read_recent(monkeypatch)

    serr = io.StringIO()
    rc = hook.stop(
        stdin=io.StringIO(_stop_payload("sess-P2-1", tmp_path, transcript)),
        stderr=serr,
    )
    assert rc == 0
    assert calls == []
    assert "cadence boundary" not in serr.getvalue()


def test_p2_above_threshold_no_boundary_does_not_fire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    _seed_db(db)
    _write_toml(tmp_path, """
        [cadence]
        enabled = true
        policy = "p2_ctx_threshold"
        ctx_threshold = 0.001
        ctx_byte_window = 600000
    """)
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(
        transcript,
        size_bytes=5_000,
        last_user_prompt="please refactor this function",
    )
    calls = _stub_rebuild_and_format(monkeypatch)
    _stub_read_recent(monkeypatch)

    serr = io.StringIO()
    rc = hook.stop(
        stdin=io.StringIO(_stop_payload("sess-P2-2", tmp_path, transcript)),
        stderr=serr,
    )
    assert rc == 0
    assert calls == []
    assert "cadence boundary" not in serr.getvalue()


def test_p2_above_threshold_with_boundary_fires(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    _seed_db(db)
    _write_toml(tmp_path, """
        [cadence]
        enabled = true
        policy = "p2_ctx_threshold"
        ctx_threshold = 0.001
        ctx_byte_window = 600000
    """)
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(transcript, size_bytes=5_000, last_user_prompt="done")
    calls = _stub_rebuild_and_format(monkeypatch)
    _stub_read_recent(monkeypatch)

    serr = io.StringIO()
    rc = hook.stop(
        stdin=io.StringIO(_stop_payload("sess-P2-3", tmp_path, transcript)),
        stderr=serr,
    )
    assert rc == 0
    assert len(calls) == 1, "rebuilder should fire when both conditions met"
    err = serr.getvalue()
    assert "cadence boundary" in err
    assert "ctx≈" in err
    assert "/clear now" in err
    assert "'done'" in err


def test_p2_missing_transcript_path_does_not_fire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No transcript_path in payload → 0 bytes → below threshold → no fire."""
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    _seed_db(db)
    _write_toml(tmp_path, """
        [cadence]
        enabled = true
        policy = "p2_ctx_threshold"
        ctx_threshold = 0.001
        ctx_byte_window = 600000
    """)
    calls = _stub_rebuild_and_format(monkeypatch)
    _stub_read_recent(monkeypatch)

    serr = io.StringIO()
    rc = hook.stop(
        stdin=io.StringIO(_stop_payload("sess-P2-4", tmp_path, transcript_path=None)),
        stderr=serr,
    )
    assert rc == 0
    assert calls == []


def test_p2_env_overrides_toml_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Env ctx_threshold beats TOML; transition from fire→no-fire."""
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    _seed_db(db)
    _write_toml(tmp_path, """
        [cadence]
        enabled = true
        policy = "p2_ctx_threshold"
        ctx_threshold = 0.001
        ctx_byte_window = 600000
    """)
    # Override threshold to require 99% of window — won't fire on 5k.
    monkeypatch.setenv(ENV_CADENCE_CTX_THRESHOLD, "0.99")
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(transcript, size_bytes=5_000, last_user_prompt="done")
    calls = _stub_rebuild_and_format(monkeypatch)
    _stub_read_recent(monkeypatch)

    serr = io.StringIO()
    rc = hook.stop(
        stdin=io.StringIO(_stop_payload("sess-P2-5", tmp_path, transcript)),
        stderr=serr,
    )
    assert rc == 0
    assert calls == []


def test_p2_unknown_policy_no_fire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Policy="off" with enabled=true is a no-op for P2 path too."""
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    _seed_db(db)
    _write_toml(tmp_path, """
        [cadence]
        enabled = true
        policy = "off"
    """)
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(transcript, size_bytes=600_000, last_user_prompt="done")
    calls = _stub_rebuild_and_format(monkeypatch)
    _stub_read_recent(monkeypatch)

    serr = io.StringIO()
    rc = hook.stop(
        stdin=io.StringIO(_stop_payload("sess-P2-6", tmp_path, transcript)),
        stderr=serr,
    )
    assert rc == 0
    assert calls == []


# --- Resume cache write --------------------------------------------------


def test_p2_fire_writes_resume_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P2 fire writes <db_dir>/cadence_resume_cache.json with expected schema."""
    db = tmp_path / "aelfrice" / "memory.db"
    db.parent.mkdir(parents=True)
    _set_db(monkeypatch, db)
    _seed_db(db)
    _write_toml(tmp_path, """
        [cadence]
        enabled = true
        policy = "p2_ctx_threshold"
        ctx_threshold = 0.001
        ctx_byte_window = 600000
    """)
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(transcript, size_bytes=5_000, last_user_prompt="done")
    _stub_rebuild_and_format(monkeypatch, body="<P2 rebuilder body>")
    _stub_read_recent(monkeypatch)

    serr = io.StringIO()
    hook.stop(
        stdin=io.StringIO(_stop_payload("sess-cache-A", tmp_path, transcript)),
        stderr=serr,
    )

    cache = db.parent / "cadence_resume_cache.json"
    assert cache.exists(), "resume cache should be written on P2 fire"
    record = json.loads(cache.read_text())
    assert record["session_id"] == "sess-cache-A"
    assert record["policy"] == POLICY_P2_CTX_THRESHOLD
    assert record["body"] == "<P2 rebuilder body>"
    assert "ts" in record


def test_p1_fire_also_writes_resume_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P1 cadence also writes the resume cache (consistent across policies)."""
    db = tmp_path / "aelfrice" / "memory.db"
    db.parent.mkdir(parents=True)
    _set_db(monkeypatch, db)
    _seed_db(db)
    # Seed ring with next_fire_idx=15 (P1 fires).
    state = {
        "session_id": "sess-P1-cache",
        "ring": [],
        "ring_max": 200,
        "next_fire_idx": 15,
        "evicted_total": 0,
    }
    (db.parent / "session_injected_ids.json").write_text(json.dumps(state))
    _write_toml(tmp_path, """
        [cadence]
        enabled = true
        policy = "p1_every_k_turns"
        k = 15
    """)
    _stub_rebuild_and_format(monkeypatch, body="<P1 rebuilder body>")
    _stub_read_recent(monkeypatch)

    serr = io.StringIO()
    hook.stop(
        stdin=io.StringIO(_stop_payload("sess-P1-cache", tmp_path)),
        stderr=serr,
    )

    cache = db.parent / "cadence_resume_cache.json"
    assert cache.exists()
    record = json.loads(cache.read_text())
    assert record["policy"] == POLICY_P1_EVERY_K_TURNS
    assert record["body"] == "<P1 rebuilder body>"


def test_resume_cache_overwrite_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second fire overwrites the cache; no stale .tmp left behind."""
    db = tmp_path / "aelfrice" / "memory.db"
    db.parent.mkdir(parents=True)
    _set_db(monkeypatch, db)
    _seed_db(db)
    _write_toml(tmp_path, """
        [cadence]
        enabled = true
        policy = "p2_ctx_threshold"
        ctx_threshold = 0.001
        ctx_byte_window = 600000
    """)
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(transcript, size_bytes=5_000, last_user_prompt="done")
    bodies = ["<body 1>", "<body 2>"]
    call_idx = {"i": 0}

    def _stub(recent, token_budget, **kwargs):  # type: ignore[no-untyped-def]
        b = bodies[call_idx["i"]]
        call_idx["i"] += 1
        return b

    monkeypatch.setattr(hook, "_rebuild_and_format", _stub)
    _stub_read_recent(monkeypatch)

    for sid in ["fire-1", "fire-2"]:
        serr = io.StringIO()
        hook.stop(
            stdin=io.StringIO(_stop_payload(sid, tmp_path, transcript)),
            stderr=serr,
        )

    cache = db.parent / "cadence_resume_cache.json"
    tmp_cache = db.parent / "cadence_resume_cache.tmp"
    assert cache.exists()
    assert not tmp_cache.exists(), "tmp file should be renamed away"
    record = json.loads(cache.read_text())
    assert record["body"] == "<body 2>"
    assert record["session_id"] == "fire-2"


def test_p2_stderr_format_includes_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Operator nudge line carries ctx-pct, bytes ratio, boundary snippet."""
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    _seed_db(db)
    _write_toml(tmp_path, """
        [cadence]
        enabled = true
        policy = "p2_ctx_threshold"
        ctx_threshold = 0.5
        ctx_byte_window = 600000
    """)
    transcript = tmp_path / "transcript.jsonl"
    # ~400k bytes → ~67% of 600k window.
    _write_transcript(transcript, size_bytes=400_000, last_user_prompt="ok thanks")
    _stub_rebuild_and_format(monkeypatch)
    _stub_read_recent(monkeypatch)

    serr = io.StringIO()
    hook.stop(
        stdin=io.StringIO(_stop_payload("sess-fmt", tmp_path, transcript)),
        stderr=serr,
    )
    err = serr.getvalue()
    assert "cadence boundary" in err
    assert "ctx≈67%" in err or "ctx≈66%" in err  # rounding tolerance
    assert "/600000 bytes" in err
    assert "ok thanks" in err  # boundary snippet appears (possibly truncated)


def test_resume_cache_skipped_for_in_memory_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An :memory: DB skips the cache write cleanly (no crash, no file)."""
    monkeypatch.setenv("AELFRICE_DB", ":memory:")
    _write_toml(tmp_path, """
        [cadence]
        enabled = true
        policy = "p2_ctx_threshold"
        ctx_threshold = 0.001
        ctx_byte_window = 600000
    """)
    transcript = tmp_path / "transcript.jsonl"
    _write_transcript(transcript, size_bytes=5_000, last_user_prompt="done")
    _stub_rebuild_and_format(monkeypatch)
    _stub_read_recent(monkeypatch)

    serr = io.StringIO()
    rc = hook.stop(
        stdin=io.StringIO(_stop_payload("sess-mem", tmp_path, transcript)),
        stderr=serr,
    )
    assert rc == 0
    # No cache file should have been created anywhere we can check.
    assert not any(p.name == "cadence_resume_cache.json" for p in tmp_path.rglob("*"))


def test_p2_does_not_regress_p1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Switching policy to p1 still fires correctly post-#871."""
    db = tmp_path / "aelfrice" / "memory.db"
    db.parent.mkdir(parents=True)
    _set_db(monkeypatch, db)
    _seed_db(db)
    state = {
        "session_id": "sess-regress",
        "ring": [],
        "ring_max": 200,
        "next_fire_idx": 15,
        "evicted_total": 0,
    }
    (db.parent / "session_injected_ids.json").write_text(json.dumps(state))
    _write_toml(tmp_path, """
        [cadence]
        enabled = true
        policy = "p1_every_k_turns"
        k = 15
    """)
    calls = _stub_rebuild_and_format(monkeypatch)
    _stub_read_recent(monkeypatch)

    serr = io.StringIO()
    hook.stop(
        stdin=io.StringIO(_stop_payload("sess-regress", tmp_path)),
        stderr=serr,
    )
    assert len(calls) == 1
    assert "cadence checkpoint fired" in serr.getvalue()
    assert "fire_idx=15" in serr.getvalue()
