"""Integration tests for #980 phantom-trigger wiring in UserPromptSubmit.

Exercises ``hook.user_prompt_submit`` end-to-end and asserts the
``<aelfrice-phantom-opportunity>`` block lands on stdout only when the
opt-in flag is on and a trigger fires. Companion to the unit suite
``test_phantom_trigger.py``.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from aelfrice import hook
from aelfrice.phantom_trigger import ENV_PHANTOM_GENERATION


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_PHANTOM_GENERATION, raising=False)


def _seed_db(db: Path) -> None:
    from aelfrice.store import MemoryStore

    MemoryStore(str(db)).close()


def _payload(prompt: str, cwd: Path, session_id: str = "sess-P") -> str:
    return json.dumps(
        {
            "session_id": session_id,
            "transcript_path": "/dev/null",
            "cwd": str(cwd),
            "hook_event_name": "UserPromptSubmit",
            "prompt": prompt,
        }
    )


def _run_ups(payload: str) -> tuple[str, str]:
    sout = io.StringIO()
    serr = io.StringIO()
    rc = hook.user_prompt_submit(
        stdin=io.StringIO(payload), stdout=sout, stderr=serr
    )
    assert rc == 0
    return sout.getvalue(), serr.getvalue()


def test_phantom_block_default_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    _seed_db(db)
    out, _ = _run_ups(
        _payload("explain the FrobnicatorWidget subsystem here", tmp_path)
    )
    assert "<aelfrice-phantom-opportunity>" not in out


def test_phantom_block_fires_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.setenv(ENV_PHANTOM_GENERATION, "1")
    _seed_db(db)
    # Empty store + novel identifier -> gap (zero hits) AND new_entity fire.
    out, _ = _run_ups(
        _payload("explain the FrobnicatorWidget subsystem here", tmp_path)
    )
    assert "<aelfrice-phantom-opportunity>" in out
    assert "gap" in out
    assert "FrobnicatorWidget" in out
    assert "Consider running /aelf:wonder" in out


def test_phantom_block_deduped_second_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.setenv(ENV_PHANTOM_GENERATION, "1")
    _seed_db(db)
    prompt = _payload("explain the FrobnicatorWidget subsystem here", tmp_path)
    _run_ups(prompt)
    # Same prompt, same session -> all opportunities deduped -> no block.
    out2, _ = _run_ups(prompt)
    assert "<aelfrice-phantom-opportunity>" not in out2


def test_phantom_block_non_blocking_rc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.setenv(ENV_PHANTOM_GENERATION, "1")
    _seed_db(db)
    # Even if anything inside the phantom path raised, rc must be 0.
    out, _ = _run_ups(_payload("a perfectly ordinary prompt sentence", tmp_path))
    # No identifier here; gap may fire on the empty store, but the contract
    # under test is simply that the hook returns 0 and does not crash.
    assert isinstance(out, str)
