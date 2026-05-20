"""Unit tests for UPS-side cadence dispatch (#870).

Targets ``_maybe_run_ups_cadence_checkpoint`` directly — exercises the
dispatch in isolation from ``user_prompt_submit``'s broader flow. The
end-to-end injection through ``user_prompt_submit`` is covered by the
integration suite (``test_hook_ups_cadence_wiring.py``).
"""
from __future__ import annotations

import io
import json
import textwrap
from pathlib import Path

import pytest

from aelfrice import hook
from aelfrice.cadence import CONFIG_FILENAME


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "AELFRICE_CADENCE_ENABLED",
        "AELFRICE_CADENCE_POLICY",
        "AELFRICE_CADENCE_K",
        "AELFRICE_CADENCE_CTX_THRESHOLD",
        "AELFRICE_CADENCE_CTX_BYTE_WINDOW",
    ):
        monkeypatch.delenv(var, raising=False)


def _set_db(monkeypatch: pytest.MonkeyPatch, db: Path) -> None:
    monkeypatch.setenv("AELFRICE_DB", str(db))


def _seed_db(db: Path) -> None:
    from aelfrice.store import MemoryStore
    MemoryStore(str(db)).close()


def _write_toml(repo: Path, body: str) -> None:
    (repo / CONFIG_FILENAME).write_text(textwrap.dedent(body))


def _write_ring_state(state_dir: Path, session_id: str, next_fire_idx: int) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "session_id": session_id,
        "ring": [],
        "ring_max": 200,
        "next_fire_idx": next_fire_idx,
        "evicted_total": 0,
    }
    (state_dir / "session_injected_ids.json").write_text(json.dumps(payload))


def _stub_read_recent(monkeypatch: pytest.MonkeyPatch, n_turns: int = 1) -> None:
    from aelfrice.context_rebuilder import RecentTurn
    canned = [RecentTurn(role="user", text=f"turn-{i}") for i in range(n_turns)]
    monkeypatch.setattr(
        hook, "_read_recent_for_pre_compact", lambda _payload, _n: canned,
    )


def _stub_rebuild_and_format(
    monkeypatch: pytest.MonkeyPatch, body: str = "REBUILD-BODY",
) -> list[dict]:
    calls: list[dict] = []

    def _stub(recent, token_budget, **kwargs):  # type: ignore[no-untyped-def]
        calls.append({"n_recent": len(recent), "token_budget": token_budget})
        return body

    monkeypatch.setattr(hook, "_rebuild_and_format", _stub)
    return calls


def _payload(cwd: Path) -> dict[str, object]:
    return {"cwd": str(cwd), "session_id": "sess-A"}


# --- Tests ----------------------------------------------------------------


def test_returns_none_when_session_id_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty session_id short-circuits — no ring read, no fire."""
    serr = io.StringIO()
    out = hook._maybe_run_ups_cadence_checkpoint(_payload(tmp_path), "", serr)
    assert out is None
    assert serr.getvalue() == ""


def test_disabled_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No [cadence] section, no env → returns None."""
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    _seed_db(db)
    _write_ring_state(db.parent, "sess-A", next_fire_idx=15)

    serr = io.StringIO()
    out = hook._maybe_run_ups_cadence_checkpoint(_payload(tmp_path), "sess-A", serr)

    assert out is None
    assert "cadence checkpoint" not in serr.getvalue()


def test_policy_off_does_not_fire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    _seed_db(db)
    _write_ring_state(db.parent, "sess-A", next_fire_idx=15)
    _write_toml(tmp_path, """
        [cadence]
        enabled = true
    """)
    serr = io.StringIO()
    out = hook._maybe_run_ups_cadence_checkpoint(_payload(tmp_path), "sess-A", serr)
    assert out is None


def test_p1_fires_at_multiple_of_k(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    _seed_db(db)
    _write_toml(tmp_path, """
        [cadence]
        enabled = true
        policy = "p1_every_k_turns"
        k = 5
    """)
    _stub_read_recent(monkeypatch)
    calls = _stub_rebuild_and_format(monkeypatch, body="X-BODY")

    for fire_idx, expected_fired in [
        (1, False), (4, False), (5, True), (6, False),
        (9, False), (10, True), (15, True),
    ]:
        calls.clear()
        _write_ring_state(db.parent, "sess-A", next_fire_idx=fire_idx)
        serr = io.StringIO()
        out = hook._maybe_run_ups_cadence_checkpoint(
            _payload(tmp_path), "sess-A", serr,
        )
        if expected_fired:
            assert out == "X-BODY", f"fire_idx={fire_idx}: expected fire"
            assert f"fire_idx={fire_idx}" in serr.getvalue()
            assert "ups cadence checkpoint fired" in serr.getvalue()
            assert len(calls) == 1
        else:
            assert out is None, f"fire_idx={fire_idx}: expected no fire"
            assert "cadence checkpoint" not in serr.getvalue()
            assert calls == []


def test_p1_fire_idx_zero_does_not_fire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cold-start guard: fire_idx=0 must not fire (0 % k == 0)."""
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    _seed_db(db)
    _write_toml(tmp_path, """
        [cadence]
        enabled = true
        policy = "p1_every_k_turns"
        k = 5
    """)
    _write_ring_state(db.parent, "sess-A", next_fire_idx=0)
    _stub_read_recent(monkeypatch)
    calls = _stub_rebuild_and_format(monkeypatch)

    serr = io.StringIO()
    out = hook._maybe_run_ups_cadence_checkpoint(_payload(tmp_path), "sess-A", serr)

    assert out is None
    assert calls == []


def test_p1_missing_ring_state_skips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    _seed_db(db)
    _write_toml(tmp_path, """
        [cadence]
        enabled = true
        policy = "p1_every_k_turns"
        k = 5
    """)
    _stub_read_recent(monkeypatch)
    calls = _stub_rebuild_and_format(monkeypatch)

    serr = io.StringIO()
    out = hook._maybe_run_ups_cadence_checkpoint(_payload(tmp_path), "sess-A", serr)

    assert out is None
    assert calls == []


def test_p1_empty_recent_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """fire_idx is on boundary, but recent-turns window is empty → no body."""
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    _seed_db(db)
    _write_toml(tmp_path, """
        [cadence]
        enabled = true
        policy = "p1_every_k_turns"
        k = 5
    """)
    _write_ring_state(db.parent, "sess-A", next_fire_idx=5)
    monkeypatch.setattr(
        hook, "_read_recent_for_pre_compact", lambda _payload, _n: [],
    )
    calls = _stub_rebuild_and_format(monkeypatch)

    serr = io.StringIO()
    out = hook._maybe_run_ups_cadence_checkpoint(_payload(tmp_path), "sess-A", serr)

    assert out is None
    assert calls == [], "rebuilder should not be called on empty recent"


def test_p1_missing_db_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cadence on, fire boundary hit, recent populated — but DB file is absent."""
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    # Note: no _seed_db; the file does not exist on disk.
    _write_toml(tmp_path, """
        [cadence]
        enabled = true
        policy = "p1_every_k_turns"
        k = 5
    """)
    _write_ring_state(db.parent, "sess-A", next_fire_idx=5)
    _stub_read_recent(monkeypatch)
    calls = _stub_rebuild_and_format(monkeypatch)

    serr = io.StringIO()
    out = hook._maybe_run_ups_cadence_checkpoint(_payload(tmp_path), "sess-A", serr)

    assert out is None
    assert calls == []


def test_p2_fires_at_boundary_signal_and_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P2: transcript bytes ≥ threshold AND last user prompt is boundary."""
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    _seed_db(db)
    _write_toml(tmp_path, """
        [cadence]
        enabled = true
        policy = "p2_ctx_threshold"
        ctx_threshold = 0.5
        ctx_byte_window = 100
    """)
    transcript = tmp_path / "transcript.jsonl"
    boundary_line = json.dumps({
        "message": {"role": "user", "content": "next task"},
    })
    transcript.write_text(boundary_line + "\n" + "x" * 100)
    _stub_read_recent(monkeypatch)
    calls = _stub_rebuild_and_format(monkeypatch, body="P2-BODY")

    payload = {
        "cwd": str(tmp_path),
        "session_id": "sess-A",
        "transcript_path": str(transcript),
    }
    serr = io.StringIO()
    out = hook._maybe_run_ups_cadence_checkpoint(payload, "sess-A", serr)

    assert out == "P2-BODY"
    assert "ups cadence checkpoint fired" in serr.getvalue()
    assert "p2_ctx_threshold" in serr.getvalue()
    assert len(calls) == 1


def test_p2_no_boundary_signal_does_not_fire(
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
        ctx_byte_window = 100
    """)
    transcript = tmp_path / "transcript.jsonl"
    non_boundary = json.dumps({
        "message": {"role": "user", "content": "please refactor this function"},
    })
    transcript.write_text(non_boundary + "\n" + "x" * 200)
    _stub_read_recent(monkeypatch)
    calls = _stub_rebuild_and_format(monkeypatch)

    payload = {
        "cwd": str(tmp_path),
        "session_id": "sess-A",
        "transcript_path": str(transcript),
    }
    serr = io.StringIO()
    out = hook._maybe_run_ups_cadence_checkpoint(payload, "sess-A", serr)

    assert out is None
    assert calls == []


def test_env_override_disables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AELFRICE_CADENCE_ENABLED=0 overrides TOML enabled=true."""
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    _seed_db(db)
    _write_toml(tmp_path, """
        [cadence]
        enabled = true
        policy = "p1_every_k_turns"
        k = 5
    """)
    _write_ring_state(db.parent, "sess-A", next_fire_idx=5)
    monkeypatch.setenv("AELFRICE_CADENCE_ENABLED", "0")
    _stub_read_recent(monkeypatch)
    calls = _stub_rebuild_and_format(monkeypatch)

    serr = io.StringIO()
    out = hook._maybe_run_ups_cadence_checkpoint(_payload(tmp_path), "sess-A", serr)

    assert out is None
    assert calls == []
