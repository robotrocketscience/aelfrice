"""Integration tests for #875 cadence shadow-evaluation mode in Stop hook."""
from __future__ import annotations

import io
import json
import textwrap
from pathlib import Path

import pytest

from aelfrice import hook
from aelfrice.cadence import (
    CADENCE_SHADOW_DIRNAME,
    CONFIG_FILENAME,
    POLICY_OFF,
    POLICY_P1_EVERY_K_TURNS,
    POLICY_P2_CTX_THRESHOLD,
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "AELFRICE_CADENCE_ENABLED",
        "AELFRICE_CADENCE_POLICY",
        "AELFRICE_CADENCE_K",
        "AELFRICE_CADENCE_CTX_THRESHOLD",
        "AELFRICE_CADENCE_CTX_BYTE_WINDOW",
        "AELFRICE_CADENCE_SHADOW_MODE_ENABLED",
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


def _stop_payload(session_id: str, cwd: Path, transcript: Path | None = None) -> str:
    p: dict[str, object] = {"session_id": session_id, "cwd": str(cwd)}
    if transcript is not None:
        p["transcript_path"] = str(transcript)
    return json.dumps(p)


def _stub_rebuild_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hook, "_rebuild_and_format", lambda *a, **k: "")
    from aelfrice.context_rebuilder import RecentTurn
    monkeypatch.setattr(
        hook, "_read_recent_for_pre_compact",
        lambda _payload, _n: [RecentTurn(role="user", text="turn")],
    )


def _shadow_log(db: Path, session_id: str) -> Path:
    return db.parent / CADENCE_SHADOW_DIRNAME / f"{session_id}.jsonl"


def test_shadow_disabled_default_writes_no_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    _seed_db(db)
    _write_toml(tmp_path, """\
        [cadence]
        enabled = true
        policy = "p1_every_k_turns"
        k = 15
    """)
    _write_ring_state(db.parent, "sess-A", next_fire_idx=10)
    _stub_rebuild_no_op(monkeypatch)

    rc = hook.stop(
        stdin=io.StringIO(_stop_payload("sess-A", tmp_path)),
        stderr=io.StringIO(),
    )
    assert rc == 0
    assert not _shadow_log(db, "sess-A").exists()


def test_shadow_enabled_logs_row_for_p1_no_fire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    _seed_db(db)
    _write_toml(tmp_path, """\
        [cadence]
        enabled = true
        policy = "p1_every_k_turns"
        k = 15
        shadow_mode_enabled = true
    """)
    _write_ring_state(db.parent, "sess-A", next_fire_idx=7)
    _stub_rebuild_no_op(monkeypatch)

    rc = hook.stop(
        stdin=io.StringIO(_stop_payload("sess-A", tmp_path)),
        stderr=io.StringIO(),
    )
    assert rc == 0
    log = _shadow_log(db, "sess-A")
    assert log.exists()
    rows = [json.loads(line) for line in log.read_text().splitlines() if line]
    assert len(rows) == 1
    row = rows[0]
    assert row["session_id"] == "sess-A"
    assert row["selected"] == POLICY_P1_EVERY_K_TURNS
    assert row["fired"] is False
    assert row["shadow"][POLICY_P1_EVERY_K_TURNS]["would_fire"] is False
    assert "fire_idx=7" in row["shadow"][POLICY_P1_EVERY_K_TURNS]["reason"]
    assert row["shadow"][POLICY_P2_CTX_THRESHOLD]["would_fire"] is False


def test_shadow_enabled_logs_row_for_p1_fire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    _seed_db(db)
    _write_toml(tmp_path, """\
        [cadence]
        enabled = true
        policy = "p1_every_k_turns"
        k = 15
        shadow_mode_enabled = true
    """)
    _write_ring_state(db.parent, "sess-B", next_fire_idx=15)
    _stub_rebuild_no_op(monkeypatch)

    rc = hook.stop(
        stdin=io.StringIO(_stop_payload("sess-B", tmp_path)),
        stderr=io.StringIO(),
    )
    assert rc == 0
    log = _shadow_log(db, "sess-B")
    assert log.exists()
    rows = [json.loads(line) for line in log.read_text().splitlines() if line]
    assert len(rows) == 1
    assert rows[0]["fired"] is True
    assert rows[0]["shadow"][POLICY_P1_EVERY_K_TURNS]["would_fire"] is True


def test_shadow_enabled_p2_selected_logs_both_policies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    _seed_db(db)
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text("x" * 200)
    _write_toml(tmp_path, """\
        [cadence]
        enabled = true
        policy = "p2_ctx_threshold"
        k = 15
        ctx_threshold = 0.5
        ctx_byte_window = 100
        shadow_mode_enabled = true
    """)
    _write_ring_state(db.parent, "sess-C", next_fire_idx=7)
    _stub_rebuild_no_op(monkeypatch)

    rc = hook.stop(
        stdin=io.StringIO(_stop_payload("sess-C", tmp_path, transcript=transcript)),
        stderr=io.StringIO(),
    )
    assert rc == 0
    log = _shadow_log(db, "sess-C")
    rows = [json.loads(line) for line in log.read_text().splitlines() if line]
    assert len(rows) == 1
    row = rows[0]
    assert row["selected"] == POLICY_P2_CTX_THRESHOLD
    assert row["shadow"][POLICY_P2_CTX_THRESHOLD]["would_fire"] is False
    assert row["shadow"][POLICY_P1_EVERY_K_TURNS]["would_fire"] is False


def test_shadow_enabled_policy_off_logs_fired_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    _seed_db(db)
    _write_toml(tmp_path, """\
        [cadence]
        enabled = true
        policy = "off"
        k = 15
        shadow_mode_enabled = true
    """)
    _write_ring_state(db.parent, "sess-D", next_fire_idx=15)
    _stub_rebuild_no_op(monkeypatch)

    rc = hook.stop(
        stdin=io.StringIO(_stop_payload("sess-D", tmp_path)),
        stderr=io.StringIO(),
    )
    assert rc == 0
    log = _shadow_log(db, "sess-D")
    rows = [json.loads(line) for line in log.read_text().splitlines() if line]
    assert len(rows) == 1
    row = rows[0]
    assert row["selected"] == POLICY_OFF
    assert row["fired"] is False
    assert row["shadow"][POLICY_P1_EVERY_K_TURNS]["would_fire"] is True


def test_shadow_two_ticks_append_two_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    _seed_db(db)
    _write_toml(tmp_path, """\
        [cadence]
        enabled = true
        policy = "p1_every_k_turns"
        k = 15
        shadow_mode_enabled = true
    """)
    _write_ring_state(db.parent, "sess-E", next_fire_idx=3)
    _stub_rebuild_no_op(monkeypatch)

    for _ in range(2):
        hook.stop(
            stdin=io.StringIO(_stop_payload("sess-E", tmp_path)),
            stderr=io.StringIO(),
        )
    log = _shadow_log(db, "sess-E")
    rows = [line for line in log.read_text().splitlines() if line]
    assert len(rows) == 2


def test_shadow_cadence_disabled_writes_no_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    _seed_db(db)
    _write_toml(tmp_path, """\
        [cadence]
        enabled = false
        policy = "p1_every_k_turns"
        k = 15
        shadow_mode_enabled = true
    """)
    _write_ring_state(db.parent, "sess-F", next_fire_idx=15)
    _stub_rebuild_no_op(monkeypatch)

    hook.stop(
        stdin=io.StringIO(_stop_payload("sess-F", tmp_path)),
        stderr=io.StringIO(),
    )
    assert not _shadow_log(db, "sess-F").exists()


def test_shadow_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    _seed_db(db)
    _write_toml(tmp_path, """\
        [cadence]
        enabled = true
        policy = "p1_every_k_turns"
        k = 15
    """)
    _write_ring_state(db.parent, "sess-G", next_fire_idx=5)
    _stub_rebuild_no_op(monkeypatch)
    monkeypatch.setenv("AELFRICE_CADENCE_SHADOW_MODE_ENABLED", "1")

    hook.stop(
        stdin=io.StringIO(_stop_payload("sess-G", tmp_path)),
        stderr=io.StringIO(),
    )
    assert _shadow_log(db, "sess-G").exists()
