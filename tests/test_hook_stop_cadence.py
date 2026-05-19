"""Integration tests for P1 every-K-turns cadence in the Stop hook (#749)."""
from __future__ import annotations

import io
import json
import textwrap
from pathlib import Path

import pytest

from aelfrice import hook
from aelfrice.cadence import (
    CONFIG_FILENAME,
    POLICY_OFF,
    POLICY_P1_EVERY_K_TURNS,
)


# --- Fixtures -------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure no inherited env tilts cadence on/off accidentally.
    for var in (
        "AELFRICE_CADENCE_ENABLED",
        "AELFRICE_CADENCE_POLICY",
        "AELFRICE_CADENCE_K",
        "AELF_AUTOLOCK_CORRECTIONS",
    ):
        monkeypatch.delenv(var, raising=False)


def _set_db(monkeypatch: pytest.MonkeyPatch, db: Path) -> None:
    monkeypatch.setenv("AELFRICE_DB", str(db))


def _seed_db(db: Path) -> None:
    # Touch the DB so :func:`_open_store` succeeds (the file must exist
    # for the Stop hook's existing lock-prompt path; cadence has its
    # own `p.exists()` guard).
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


def _stop_payload(session_id: str, cwd: Path) -> str:
    return json.dumps({"session_id": session_id, "cwd": str(cwd)})


def _stub_rebuild_and_format(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Replace `_rebuild_and_format` with a recorder; return the call list."""
    calls: list[dict] = []

    def _stub(recent, token_budget, **kwargs):  # type: ignore[no-untyped-def]
        calls.append({
            "n_recent": len(recent),
            "token_budget": token_budget,
            **kwargs,
        })
        return ""

    monkeypatch.setattr(hook, "_rebuild_and_format", _stub)
    return calls


def _stub_read_recent(monkeypatch: pytest.MonkeyPatch, n_turns: int = 1) -> None:
    """Replace `_read_recent_for_pre_compact` to return a canned list.

    Cadence needs at least one recent turn to proceed past the
    empty-transcript guard; tests should not depend on a real
    transcript file.
    """
    from aelfrice.context_rebuilder import RecentTurn

    canned = [
        RecentTurn(role="user", text=f"turn-{i}")
        for i in range(n_turns)
    ]
    monkeypatch.setattr(
        hook, "_read_recent_for_pre_compact", lambda _payload, _n: canned,
    )


# --- Tests ----------------------------------------------------------------


def test_cadence_disabled_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No [cadence] section + no env → cadence never fires."""
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    _seed_db(db)
    _write_ring_state(db.parent, "sess-A", next_fire_idx=15)
    calls = _stub_rebuild_and_format(monkeypatch)

    serr = io.StringIO()
    rc = hook.stop(
        stdin=io.StringIO(_stop_payload("sess-A", tmp_path)),
        stderr=serr,
    )

    assert rc == 0
    assert calls == [], "cadence should be inert when disabled"
    assert "cadence checkpoint" not in serr.getvalue()


def test_cadence_enabled_policy_off_does_not_fire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    _seed_db(db)
    _write_ring_state(db.parent, "sess-A", next_fire_idx=15)
    _write_toml(tmp_path, """
        [cadence]
        enabled = true
        # policy defaults to "off" when unset, even with enabled=true
    """)
    calls = _stub_rebuild_and_format(monkeypatch)

    serr = io.StringIO()
    rc = hook.stop(
        stdin=io.StringIO(_stop_payload("sess-A", tmp_path)),
        stderr=serr,
    )

    assert rc == 0
    assert calls == []
    assert "cadence checkpoint" not in serr.getvalue()


def test_cadence_p1_fires_at_multiples_of_k(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    _seed_db(db)
    _write_toml(tmp_path, """
        [cadence]
        enabled = true
        policy = "p1_every_k_turns"
        k = 15
    """)
    _stub_read_recent(monkeypatch)
    calls = _stub_rebuild_and_format(monkeypatch)

    for fire_idx, expected_fired in [
        (1, False), (14, False), (15, True), (16, False),
        (29, False), (30, True), (45, True),
    ]:
        calls.clear()
        _write_ring_state(db.parent, "sess-A", next_fire_idx=fire_idx)
        serr = io.StringIO()
        rc = hook.stop(
            stdin=io.StringIO(_stop_payload("sess-A", tmp_path)),
            stderr=serr,
        )
        assert rc == 0
        if expected_fired:
            assert len(calls) == 1, f"fire_idx={fire_idx}: expected fire"
            assert f"fire_idx={fire_idx}" in serr.getvalue()
            assert "cadence checkpoint fired" in serr.getvalue()
        else:
            assert calls == [], f"fire_idx={fire_idx}: expected no fire"
            assert "cadence checkpoint" not in serr.getvalue()


def test_cadence_fire_idx_zero_does_not_fire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cold-start guard: fire_idx=0 must not trip cadence even though 0%k==0."""
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    _seed_db(db)
    _write_toml(tmp_path, """
        [cadence]
        enabled = true
        policy = "p1_every_k_turns"
        k = 15
    """)
    _write_ring_state(db.parent, "sess-A", next_fire_idx=0)
    _stub_read_recent(monkeypatch)
    calls = _stub_rebuild_and_format(monkeypatch)

    serr = io.StringIO()
    rc = hook.stop(
        stdin=io.StringIO(_stop_payload("sess-A", tmp_path)),
        stderr=serr,
    )

    assert rc == 0
    assert calls == []
    assert "cadence checkpoint" not in serr.getvalue()


def test_cadence_missing_ring_state_skips_cadence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No ring sentinel → no fire_idx → cadence is a no-op."""
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    _seed_db(db)
    _write_toml(tmp_path, """
        [cadence]
        enabled = true
        policy = "p1_every_k_turns"
        k = 15
    """)
    # NB: no _write_ring_state call.
    calls = _stub_rebuild_and_format(monkeypatch)

    serr = io.StringIO()
    rc = hook.stop(
        stdin=io.StringIO(_stop_payload("sess-A", tmp_path)),
        stderr=serr,
    )

    assert rc == 0
    assert calls == []


def test_cadence_session_mismatch_skips_cadence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ring written under sess-B but Stop fires for sess-A → no cadence."""
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    _seed_db(db)
    _write_toml(tmp_path, """
        [cadence]
        enabled = true
        policy = "p1_every_k_turns"
        k = 15
    """)
    _write_ring_state(db.parent, "sess-B", next_fire_idx=15)
    calls = _stub_rebuild_and_format(monkeypatch)

    serr = io.StringIO()
    rc = hook.stop(
        stdin=io.StringIO(_stop_payload("sess-A", tmp_path)),
        stderr=serr,
    )

    assert rc == 0
    assert calls == []
    assert "cadence checkpoint" not in serr.getvalue()


def test_cadence_empty_recent_skips_fire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cadence is gated on having a recent-turns slice; empty → no fire."""
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    _seed_db(db)
    _write_toml(tmp_path, """
        [cadence]
        enabled = true
        policy = "p1_every_k_turns"
        k = 15
    """)
    _write_ring_state(db.parent, "sess-A", next_fire_idx=15)
    # No _stub_read_recent — default returns [] when transcript is missing.
    monkeypatch.setattr(
        hook, "_read_recent_for_pre_compact", lambda _payload, _n: [],
    )
    calls = _stub_rebuild_and_format(monkeypatch)

    serr = io.StringIO()
    rc = hook.stop(
        stdin=io.StringIO(_stop_payload("sess-A", tmp_path)),
        stderr=serr,
    )

    assert rc == 0
    assert calls == []


def test_cadence_env_override_beats_toml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TOML says enabled=false; env flips it on. Cadence fires."""
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    _seed_db(db)
    _write_toml(tmp_path, """
        [cadence]
        enabled = false
        policy = "p1_every_k_turns"
        k = 15
    """)
    monkeypatch.setenv("AELFRICE_CADENCE_ENABLED", "1")
    _write_ring_state(db.parent, "sess-A", next_fire_idx=15)
    _stub_read_recent(monkeypatch)
    calls = _stub_rebuild_and_format(monkeypatch)

    serr = io.StringIO()
    rc = hook.stop(
        stdin=io.StringIO(_stop_payload("sess-A", tmp_path)),
        stderr=serr,
    )

    assert rc == 0
    assert len(calls) == 1
    assert "cadence checkpoint fired" in serr.getvalue()


def test_cadence_fires_even_with_no_lock_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: pre-#749 the Stop hook early-returned on empty
    lock-candidate sets, which would have suppressed cadence too. The
    restructured stop() must run cadence regardless."""
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    _seed_db(db)
    # NB: zero beliefs in the store → zero lock candidates.
    _write_toml(tmp_path, """
        [cadence]
        enabled = true
        policy = "p1_every_k_turns"
        k = 15
    """)
    _write_ring_state(db.parent, "sess-A", next_fire_idx=15)
    _stub_read_recent(monkeypatch)
    calls = _stub_rebuild_and_format(monkeypatch)

    serr = io.StringIO()
    rc = hook.stop(
        stdin=io.StringIO(_stop_payload("sess-A", tmp_path)),
        stderr=serr,
    )

    assert rc == 0
    assert len(calls) == 1, "cadence must fire when policy says so, even with no lock candidates"


def test_cadence_emits_policy_and_k_in_stderr_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stderr observability line includes policy + k for debugging."""
    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    _seed_db(db)
    _write_toml(tmp_path, """
        [cadence]
        enabled = true
        policy = "p1_every_k_turns"
        k = 7
    """)
    _write_ring_state(db.parent, "sess-A", next_fire_idx=14)
    _stub_read_recent(monkeypatch)
    _stub_rebuild_and_format(monkeypatch)

    serr = io.StringIO()
    rc = hook.stop(
        stdin=io.StringIO(_stop_payload("sess-A", tmp_path)),
        stderr=serr,
    )

    assert rc == 0
    out = serr.getvalue()
    assert "policy=p1_every_k_turns" in out
    assert "k=7" in out
    assert "fire_idx=14" in out


def test_cadence_does_not_break_existing_lock_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When both lock-prompt candidates exist AND cadence fires,
    the stderr stream carries both outputs and rc is still 0."""
    from aelfrice.models import (
        BELIEF_CORRECTION,
        LOCK_NONE,
        Belief,
    )
    from aelfrice.store import MemoryStore

    db = tmp_path / "memory.db"
    _set_db(monkeypatch, db)
    _seed_db(db)
    # Insert one correction-class lock candidate.
    s = MemoryStore(str(db))
    try:
        s.insert_belief(Belief(
            id="b1",
            content="user asserted X about Y",
            content_hash="h_b1",
            alpha=1.0, beta=1.0,
            type=BELIEF_CORRECTION,
            lock_level=LOCK_NONE,
            locked_at=None,
            created_at="2026-01-01T00:00:00Z",
            last_retrieved_at=None,
            session_id="sess-A",
            origin="user_stated",
        ))
    finally:
        s.close()
    _write_toml(tmp_path, """
        [cadence]
        enabled = true
        policy = "p1_every_k_turns"
        k = 15
    """)
    _write_ring_state(db.parent, "sess-A", next_fire_idx=15)
    _stub_read_recent(monkeypatch)
    calls = _stub_rebuild_and_format(monkeypatch)

    serr = io.StringIO()
    rc = hook.stop(
        stdin=io.StringIO(_stop_payload("sess-A", tmp_path)),
        stderr=serr,
    )

    assert rc == 0
    out = serr.getvalue()
    assert hook.STOP_PROMPT_OPEN_TAG in out  # lock-prompt block emitted
    assert "cadence checkpoint fired" in out  # cadence emitted
    assert len(calls) == 1
