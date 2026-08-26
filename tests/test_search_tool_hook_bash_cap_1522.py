"""#1522 — the Bash per-turn fire cap must bind across OS processes.

`aelf-search-tool-hook` is registered as a `"type": "command"` hook, so
the host spawns one process per fire. A process-global counter is empty
at the start of every fire and `BASH_FIRE_CAP_PER_TURN` could never be
reached in any deployed configuration.

Every cap assertion here is therefore driven through **subprocesses** —
one `python -m aelfrice.hook_search_tool` per fire, exactly as the host
runs it. An in-process test structurally cannot observe this defect: it
passes against the dead process-global counter and against the wired
ring alike, which is what let the cap ship non-functional.

Each fire carries a distinct search token, so the #740 session-ring
content dedup cannot be what suppresses a block.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from aelfrice.hook_search_tool import BASH_FIRE_CAP_PER_TURN
from aelfrice.session_ring import (
    SESSION_RING_FILENAME,
    read_bash_fire_state,
    record_bash_fire,
    stamp_bash_turn,
)
from aelfrice.store import MemoryStore

SESSION_ID = "cap-1522-session"

# Generous: each fire is a cold interpreter start plus a store open.
_SUBPROCESS_TIMEOUT_S = 60.0


@pytest.fixture
def hook_env(tmp_path: Path) -> dict[str, str]:
    """Env for a hook subprocess, pointed at a throwaway empty store.

    An empty store is enough: the cap decides whether a block is emitted
    at all, before retrieval, so "sentinel block" vs "no output" is the
    observable. It also keeps nine interpreter launches affordable.
    """
    db = tmp_path / "aelfrice" / "memory.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    MemoryStore(str(db)).close()
    env = dict(os.environ)
    env["AELFRICE_DB"] = str(db)
    env["AELF_NO_UPDATE_CHECK"] = "1"
    return env


def _fire(env: dict[str, str], token: str, cwd: Path) -> bool:
    """Run one PreToolUse Bash fire in its own process.

    Returns True when the hook emitted an `additionalContext` block.
    """
    payload = json.dumps({
        "hook_event_name": "PreToolUse",
        "tool_name": "Bash",
        "tool_input": {"command": f"rg {token} src/"},
        "cwd": str(cwd),
        "session_id": SESSION_ID,
    })
    proc = subprocess.run(
        [sys.executable, "-m", "aelfrice.hook_search_tool"],
        input=payload,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        cwd=str(cwd),
        timeout=_SUBPROCESS_TIMEOUT_S,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout.strip()
    if not out:
        return False
    envelope = json.loads(out)
    assert "hookSpecificOutput" in envelope, out
    return True


def _submit_prompt(env: dict[str, str], cwd: Path) -> None:
    """Run the real UserPromptSubmit hook — the shipped turn stamp."""
    payload = json.dumps({
        "hook_event_name": "UserPromptSubmit",
        "prompt": "next turn please",
        "cwd": str(cwd),
        "session_id": SESSION_ID,
    })
    proc = subprocess.run(
        [sys.executable, "-m", "aelfrice.hook"],
        input=payload,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        cwd=str(cwd),
        timeout=_SUBPROCESS_TIMEOUT_S,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


@pytest.mark.timeout(300)
def test_cap_binds_across_separate_processes(
    hook_env: dict[str, str], tmp_path: Path,
) -> None:
    """The issue's repro: five processes, one session, one turn.

    Fires 1..3 emit, 4..5 are suppressed. Before #1522 all five emitted.
    """
    tokens = [
        "needlealpha", "needlebravo", "needlecharlie",
        "needledelta", "needleecho",
    ]
    emitted = [_fire(hook_env, t, tmp_path) for t in tokens]
    assert emitted == [True, True, True, False, False], emitted
    assert sum(emitted) == BASH_FIRE_CAP_PER_TURN


@pytest.mark.timeout(300)
def test_new_turn_resets_the_cap(
    hook_env: dict[str, str], tmp_path: Path,
) -> None:
    """A stamped turn boundary re-opens the budget.

    Without this the fix is a permanent per-session cap of three, which
    the issue calls out as a worse bug than the dead cap.
    """
    turn_one = [
        _fire(hook_env, t, tmp_path)
        for t in ("alphaone", "bravoone", "charlieone", "deltaone")
    ]
    assert turn_one == [True, True, True, False], turn_one

    _submit_prompt(hook_env, tmp_path)

    turn_two = [
        _fire(hook_env, t, tmp_path)
        for t in ("alphatwo", "bravotwo", "charlietwo", "deltatwo")
    ]
    assert turn_two == [True, True, True, False], turn_two


@pytest.mark.timeout(300)
def test_unusable_ring_degrades_to_no_cap(
    hook_env: dict[str, str], tmp_path: Path,
) -> None:
    """Fail-soft: an unusable ring leaves behaviour exactly as today.

    The ring path is occupied by a directory, so every read raises
    `IsADirectoryError` and every `os.replace` write fails. The hook must
    keep firing and keep exiting 0 — a hook that suppressed retrieval
    because it could not read a file would be the worse defect.
    """
    ring_path = Path(hook_env["AELFRICE_DB"]).parent / SESSION_RING_FILENAME
    ring_path.mkdir(parents=True, exist_ok=False)
    (ring_path / "keep").write_text("not a ring", encoding="utf-8")

    tokens = [
        "brokenalpha", "brokenbravo", "brokencharlie",
        "brokendelta", "brokenecho",
    ]
    emitted = [_fire(hook_env, t, tmp_path) for t in tokens]
    assert emitted == [True] * 5, emitted


def test_malformed_ring_reads_as_no_fires(
    hook_env: dict[str, str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A corrupt ring file reads as "no cap", not as "cap reached"."""
    monkeypatch.setenv("AELFRICE_DB", hook_env["AELFRICE_DB"])
    ring_path = Path(hook_env["AELFRICE_DB"]).parent / SESSION_RING_FILENAME
    ring_path.write_text("{not json at all", encoding="utf-8")
    assert read_bash_fire_state(SESSION_ID) == {}


def test_counted_fires_survive_a_read_but_not_a_new_turn(
    hook_env: dict[str, str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The turn comparison, not a second write, is what zeroes the count.

    `stamp_bash_turn` only bumps the turn id; the recorded count keeps
    naming the turn it was taken in, so the reader reports 0 fires for
    the new turn while the raw field still holds the old count.
    """
    monkeypatch.setenv("AELFRICE_DB", hook_env["AELFRICE_DB"])
    assert record_bash_fire(SESSION_ID) == {"turn_id": 0, "fires": 1}
    assert record_bash_fire(SESSION_ID) == {"turn_id": 0, "fires": 2}
    assert read_bash_fire_state(SESSION_ID) == {"turn_id": 0, "fires": 2}

    assert stamp_bash_turn(SESSION_ID) is True
    assert read_bash_fire_state(SESSION_ID) == {"turn_id": 1, "fires": 0}
    assert record_bash_fire(SESSION_ID) == {"turn_id": 1, "fires": 1}


def test_bool_ring_fields_read_as_the_default(
    hook_env: dict[str, str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`bool` is an `int` subclass; `true` must not read as one fire."""
    monkeypatch.setenv("AELFRICE_DB", hook_env["AELFRICE_DB"])
    ring_path = Path(hook_env["AELFRICE_DB"]).parent / SESSION_RING_FILENAME
    ring_path.write_text(
        json.dumps({
            "session_id": SESSION_ID,
            "bash_turn_id": True,
            "bash_fires": True,
            "bash_fires_turn_id": True,
        }),
        encoding="utf-8",
    )
    assert read_bash_fire_state(SESSION_ID) == {"turn_id": 0, "fires": 0}
