"""#1522 — the Bash per-turn fire cap must bind across processes *and*
across the sessions that share one repo checkout.

`aelf-search-tool-hook` is registered as a `"type": "command"` hook, so
the host spawns one process per fire. A process-global counter is empty
at the start of every fire and `BASH_FIRE_CAP_PER_TURN` could never be
reached in any deployed configuration. Moving it onto the session ring
*record* fixes that and introduces the second defect: the record is one
per repo, keyed by a single `session_id`, and every git worktree of a
repo shares one ring file, so any concurrent session's fire zeroed the
count — A,A,A,B,A,A,A emitted 7 of 7.

Every cap assertion here is therefore driven through **subprocesses** —
one `python -m aelfrice.hook_search_tool` per fire, exactly as the host
runs it. An in-process test structurally cannot observe either defect:
it passes against the dead process-global counter and against the wired
ring alike, which is what let the cap ship non-functional, and it
cannot interleave two sessions against one ring file at all.

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
    BASH_STATE_MAX_SESSIONS,
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


def _fire(
    env: dict[str, str], token: str, cwd: Path, session: str = SESSION_ID,
) -> bool:
    """Run one PreToolUse Bash fire in its own process.

    Returns True when the hook emitted an `additionalContext` block.
    """
    payload = json.dumps({
        "hook_event_name": "PreToolUse",
        "tool_name": "Bash",
        "tool_input": {"command": f"rg {token} src/"},
        "cwd": str(cwd),
        "session_id": session,
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


def _submit_prompt(
    env: dict[str, str], cwd: Path, session: str = SESSION_ID,
) -> None:
    """Run the real UserPromptSubmit hook — the shipped turn stamp."""
    payload = json.dumps({
        "hook_event_name": "UserPromptSubmit",
        "prompt": "next turn please",
        "cwd": str(cwd),
        "session_id": session,
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
def test_cap_binds_when_two_sessions_share_a_checkout(
    hook_env: dict[str, str], tmp_path: Path,
) -> None:
    """Two sessions on one repo checkout each keep their own budget.

    Every git worktree of a repo shares one
    `<git-common-dir>/aelfrice/session_injected_ids.json`, and the
    operator's normal configuration is several concurrent sessions
    against it. The #740 dedup ring is one record per repo keyed by a
    single `session_id`, so a fire from session B rewrites the record
    for session A. If the fire counters live in that record, B's single
    fire zeroes A's count and A gets an unbounded budget: the reviewer's
    A,A,A,B,A,A,A interleave emitted 7 of 7 blocks, six of them A's,
    against a cap of 3.

    The sequence below is that interleave. A's fourth through sixth
    fires must stay suppressed across B's, and B must still get its own
    full cap — the failure this guards against is fail-soft in the
    "no cap" direction, so an assertion that only counted A's would
    also pass a fix that simply capped everyone to three per repo.
    """
    sess_a, sess_b = "cap-1522-worktree-A", "cap-1522-worktree-B"
    interleave = [
        (sess_a, "alphazero"), (sess_a, "alphaone"), (sess_a, "alphatwo"),
        (sess_b, "bravozero"),
        (sess_a, "alphathree"), (sess_a, "alphafour"), (sess_a, "alphafive"),
        (sess_b, "bravoone"), (sess_b, "bravotwo"), (sess_b, "bravothree"),
    ]
    emitted = [
        _fire(hook_env, token, tmp_path, session=sid)
        for sid, token in interleave
    ]
    by_session: dict[str, list[bool]] = {sess_a: [], sess_b: []}
    for (sid, _token), fired in zip(interleave, emitted, strict=True):
        by_session[sid].append(fired)
    assert by_session[sess_a] == [True, True, True, False, False, False]
    assert by_session[sess_b] == [True, True, True, False]
    assert sum(emitted) == 2 * BASH_FIRE_CAP_PER_TURN


@pytest.mark.timeout(300)
def test_a_second_sessions_turn_stamp_leaves_the_first_capped(
    hook_env: dict[str, str], tmp_path: Path,
) -> None:
    """Session B starting a new turn does not re-open session A's budget.

    `stamp_bash_turn` fires from `UserPromptSubmit`, which is per
    session, not per repo. A turn boundary in one session must not be
    read as a turn boundary in another — otherwise a busy neighbour
    hands session A a fresh cap on every prompt it submits.
    """
    sess_a, sess_b = "cap-1522-stamp-A", "cap-1522-stamp-B"
    first = [
        _fire(hook_env, t, tmp_path, session=sess_a)
        for t in ("sa0", "sa1", "sa2", "sa3")
    ]
    assert first == [True, True, True, False], first

    _submit_prompt(hook_env, tmp_path, session=sess_b)

    assert _fire(hook_env, "sa4", tmp_path, session=sess_a) is False
    assert _fire(hook_env, "sb0", tmp_path, session=sess_b) is True

    _submit_prompt(hook_env, tmp_path, session=sess_a)
    assert _fire(hook_env, "sa5", tmp_path, session=sess_a) is True


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
            "bash": {
                SESSION_ID: {
                    "turn_id": True,
                    "fires": True,
                    "fires_turn_id": True,
                    "seq": True,
                },
            },
        }),
        encoding="utf-8",
    )
    assert read_bash_fire_state(SESSION_ID) == {"turn_id": 0, "fires": 0}


def test_bash_state_survives_a_ring_reset_by_another_session(
    hook_env: dict[str, str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dedup ring resets on a session switch; the Bash map does not.

    This is the unit-level statement of the multi-session defect: a
    write under session B rewrites the ring record wholesale, and A's
    fire count has to come through it intact.
    """
    monkeypatch.setenv("AELFRICE_DB", hook_env["AELFRICE_DB"])
    other = SESSION_ID + "-neighbour"
    assert record_bash_fire(SESSION_ID) == {"turn_id": 0, "fires": 1}
    assert record_bash_fire(SESSION_ID) == {"turn_id": 0, "fires": 2}

    assert record_bash_fire(other) == {"turn_id": 0, "fires": 1}
    ring = json.loads(
        (Path(hook_env["AELFRICE_DB"]).parent / SESSION_RING_FILENAME)
        .read_text(encoding="utf-8")
    )
    assert ring["session_id"] == other, "the dedup ring did switch session"

    assert read_bash_fire_state(SESSION_ID) == {"turn_id": 0, "fires": 2}
    assert stamp_bash_turn(other) is True
    assert read_bash_fire_state(SESSION_ID) == {"turn_id": 0, "fires": 2}


def test_bash_state_map_is_bounded(
    hook_env: dict[str, str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The per-session map evicts least-recently-touched beyond the cap.

    One ring file serves a whole machine's worth of sessions over time,
    so the map that survives session switches has to be bounded or it
    grows without limit.
    """
    monkeypatch.setenv("AELFRICE_DB", hook_env["AELFRICE_DB"])
    sessions = [f"bounded-{i:02d}" for i in range(BASH_STATE_MAX_SESSIONS + 3)]
    for sid in sessions:
        assert record_bash_fire(sid) == {"turn_id": 0, "fires": 1}
    ring = json.loads(
        (Path(hook_env["AELFRICE_DB"]).parent / SESSION_RING_FILENAME)
        .read_text(encoding="utf-8")
    )
    assert sorted(ring["bash"]) == sessions[-BASH_STATE_MAX_SESSIONS:]
    # Eviction is fail-soft in the "no cap" direction, never a phantom cap.
    assert read_bash_fire_state(sessions[0]) == {}


def test_eviction_victim_is_the_coldest_entry_not_the_newest(
    hook_env: dict[str, str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Eviction follows recency, and never takes the entry just written.

    `test_bash_state_map_is_bounded` inserts its sessions in ascending
    id order, so recency order and lexicographic order coincide there
    and the id tiebreak alone reproduces the right answer. This case
    makes the two disagree: `z1` is the oldest insertion and the newest
    touch, and the newcomer sorts before every held id while being the
    most recently written.

    Evicting the freshly written entry would leave that session with no
    record at all, which reinstates the pre-#1522 defect — an uncapped
    lane — for whichever session fires next.
    """
    monkeypatch.setenv("AELFRICE_DB", hook_env["AELFRICE_DB"])
    held = [f"z{i}" for i in range(1, BASH_STATE_MAX_SESSIONS + 1)]
    for sid in held:
        assert record_bash_fire(sid) == {"turn_id": 0, "fires": 1}
    # Keep z1 live. z2 becomes the least recently touched entry.
    for fires in range(2, 7):
        assert record_bash_fire("z1") == {"turn_id": 0, "fires": fires}
    # One over the bound, and lexicographically first of all of them.
    assert record_bash_fire("a-newcomer") == {"turn_id": 0, "fires": 1}

    ring = json.loads(
        (Path(hook_env["AELFRICE_DB"]).parent / SESSION_RING_FILENAME)
        .read_text(encoding="utf-8")
    )
    assert sorted(ring["bash"]) == sorted({*held[2:], "z1", "a-newcomer"})
    assert read_bash_fire_state("a-newcomer") == {"turn_id": 0, "fires": 1}
    assert read_bash_fire_state("z1") == {"turn_id": 0, "fires": 6}
    assert read_bash_fire_state("z2") == {}
