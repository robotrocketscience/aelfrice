"""Integration tests for UPS-side cadence wiring (#870).

Exercises ``user_prompt_submit`` end-to-end with cadence enabled and
fire_idx positioned on a P1 boundary; asserts the ``<cadence-checkpoint>``
block lands on stdout ahead of any retrieval body. Companion to the
unit suite ``test_hook_ups_cadence.py``, which targets the dispatch
function in isolation.
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
    monkeypatch: pytest.MonkeyPatch, body: str = "REBUILDER-SYNTH",
) -> None:
    monkeypatch.setattr(
        hook, "_rebuild_and_format",
        lambda recent, token_budget, **kwargs: body,
    )


def _ups_payload(prompt: str, cwd: Path, session_id: str = "sess-A") -> str:
    return json.dumps({
        "session_id": session_id,
        "transcript_path": "/dev/null",
        "cwd": str(cwd),
        "hook_event_name": "UserPromptSubmit",
        "prompt": prompt,
    })


def _run_ups(payload: str) -> tuple[str, str]:
    sout = io.StringIO()
    serr = io.StringIO()
    rc = hook.user_prompt_submit(
        stdin=io.StringIO(payload),
        stdout=sout,
        stderr=serr,
    )
    assert rc == 0
    return sout.getvalue(), serr.getvalue()


# --- Tests ----------------------------------------------------------------


def test_disabled_emits_no_checkpoint_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default-OFF: no cadence-checkpoint block on stdout, even at boundary."""
    db = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    _seed_db(db)
    _write_ring_state(db.parent, "sess-A", next_fire_idx=15)

    out, serr = _run_ups(_ups_payload("hello world", tmp_path))

    assert "<cadence-checkpoint>" not in out
    assert "ups cadence checkpoint" not in serr


def test_p1_boundary_emits_checkpoint_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cadence-enabled + P1 + fire_idx % k == 0 → <cadence-checkpoint> on stdout."""
    db = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    _seed_db(db)
    _write_toml(tmp_path, """
        [cadence]
        enabled = true
        policy = "p1_every_k_turns"
        k = 5
    """)
    _write_ring_state(db.parent, "sess-A", next_fire_idx=5)
    _stub_read_recent(monkeypatch)
    _stub_rebuild_and_format(monkeypatch, body="REBUILDER-SYNTH")

    out, serr = _run_ups(_ups_payload("hello world", tmp_path))

    assert "<cadence-checkpoint>\nREBUILDER-SYNTH\n</cadence-checkpoint>" in out
    assert "ups cadence checkpoint fired" in serr
    assert "fire_idx=5" in serr


def test_p1_non_boundary_emits_no_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cadence-enabled + P1 + fire_idx not at boundary → no checkpoint block."""
    db = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    _seed_db(db)
    _write_toml(tmp_path, """
        [cadence]
        enabled = true
        policy = "p1_every_k_turns"
        k = 5
    """)
    _write_ring_state(db.parent, "sess-A", next_fire_idx=3)
    _stub_read_recent(monkeypatch)
    _stub_rebuild_and_format(monkeypatch, body="REBUILDER-SYNTH")

    out, serr = _run_ups(_ups_payload("hello world", tmp_path))

    assert "<cadence-checkpoint>" not in out
    assert "ups cadence checkpoint" not in serr


def test_checkpoint_block_precedes_retrieval_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Augment mode: when both blocks emit, cadence-checkpoint is first."""
    from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief
    from aelfrice.store import MemoryStore

    db = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))

    # Seed a hit so retrieval has something to emit.
    store = MemoryStore(str(db))
    try:
        store.insert_belief(Belief(
            id="HIT01",
            content="cellar door barrel inventory",
            content_hash="h_HIT01",
            alpha=1.0, beta=1.0,
            type=BELIEF_FACTUAL, lock_level=LOCK_NONE,
            locked_at=None,
            created_at="2026-05-20T00:00:00Z",
            last_retrieved_at=None,
        ))
    finally:
        store.close()

    _write_toml(tmp_path, """
        [cadence]
        enabled = true
        policy = "p1_every_k_turns"
        k = 5
    """)
    _write_ring_state(db.parent, "sess-A", next_fire_idx=5)
    _stub_read_recent(monkeypatch)
    _stub_rebuild_and_format(monkeypatch, body="CADENCE-BODY-MARKER")

    out, _ = _run_ups(_ups_payload(
        "how many barrels are in the cellar door storage", tmp_path,
    ))

    cadence_pos = out.find("CADENCE-BODY-MARKER")
    hit_pos = out.find("HIT01")
    assert cadence_pos != -1, f"cadence body missing from stdout: {out!r}"
    assert hit_pos != -1, f"retrieval hit missing from stdout: {out!r}"
    assert cadence_pos < hit_pos, (
        "expected cadence-checkpoint to precede retrieval body; "
        f"cadence@{cadence_pos}, hit@{hit_pos}"
    )


def test_missing_session_id_no_fire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No session_id in payload → cadence cannot resolve ring state, skips."""
    db = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    _seed_db(db)
    _write_toml(tmp_path, """
        [cadence]
        enabled = true
        policy = "p1_every_k_turns"
        k = 5
    """)
    _write_ring_state(db.parent, "sess-A", next_fire_idx=5)
    _stub_read_recent(monkeypatch)
    _stub_rebuild_and_format(monkeypatch)

    # Payload omits session_id.
    payload = json.dumps({
        "transcript_path": "/dev/null",
        "cwd": str(tmp_path),
        "hook_event_name": "UserPromptSubmit",
        "prompt": "hello",
    })
    out, serr = _run_ups(payload)

    assert "<cadence-checkpoint>" not in out
    assert "ups cadence checkpoint" not in serr
