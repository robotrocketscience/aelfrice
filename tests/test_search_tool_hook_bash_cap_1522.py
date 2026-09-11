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

from aelfrice.hook_search_tool import (
    BASH_FIRE_CAP_PER_TURN,
    _bash_fire_cap_reached,
)
from aelfrice.session_ring import (
    BASH_STATE_MAX_SESSIONS,
    SESSION_RING_FILENAME,
    append_ids,
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


# Runs one Bash fire in a clean interpreter and reports whether the cap
# lane's module landed in the import graph. Must be a subprocess: pytest
# has already imported `aelfrice.session_ring` by the time this module is
# collected, so an in-process check would pass either way.
_IMPORT_PROBE = """
import io, json, sys
import aelfrice.hook_search_tool as hook
payload = json.dumps({
    "hook_event_name": "PreToolUse",
    "tool_name": "Bash",
    "tool_input": {"command": sys.argv[1]},
    "cwd": sys.argv[2],
    "session_id": "import-probe-1522",
})
hook.main(
    stdin=io.StringIO(payload), stdout=io.StringIO(), stderr=io.StringIO()
)
print("aelfrice.session_ring" in sys.modules)
"""


def _session_ring_imported(
    command: str, env: dict[str, str], cwd: Path,
) -> bool:
    """Return True if `command` pulled `session_ring` into `sys.modules`."""
    proc = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE, command, str(cwd)],
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
    assert out in {"True", "False"}, f"{out!r} / {proc.stderr}"
    return out == "True"


def _submit_prompt(
    env: dict[str, str],
    cwd: Path,
    session: str = SESSION_ID,
    prompt: str = "next turn please",
) -> None:
    """Run the real UserPromptSubmit hook — the shipped turn stamp."""
    payload = json.dumps({
        "hook_event_name": "UserPromptSubmit",
        "prompt": prompt,
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

    # `append_ids` is the #740 injection write: it owns the record and
    # reshapes it for whoever calls. A Bash counter write deliberately
    # does not (see the co-tenant test below), so the switch has to be
    # driven from the path that really performs one.
    assert append_ids(other, ["neighbour-belief"]) >= 0
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


def test_invalid_utf8_ring_is_not_read_as_cap_reached(
    hook_env: dict[str, str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ring that raises on decode still reads as "no cap".

    `_read_ring_unlocked` decodes the file as UTF-8 and catches only
    `json.JSONDecodeError` and `OSError`, so a stray non-UTF-8 byte
    raises `UnicodeDecodeError` straight out of it — the #1441 class.
    That is what reaches `read_bash_fire_state`'s blanket handler, the
    one branch whose whole job is to fail open. A handler that returned
    a count instead would suppress this session's retrieval for as long
    as the unreadable file sat there.
    """
    monkeypatch.setenv("AELFRICE_DB", hook_env["AELFRICE_DB"])
    ring_path = Path(hook_env["AELFRICE_DB"]).parent / SESSION_RING_FILENAME
    sid = SESSION_ID.encode("ascii")
    ring_path.write_bytes(
        b'{"session_id": "' + sid + b'", "bash": {"' + sid
        + b'": {"turn_id": 0, "fires": \x81}}}'
    )
    assert read_bash_fire_state(SESSION_ID) == {}
    assert _bash_fire_cap_reached(SESSION_ID) is False


@pytest.mark.timeout(180)
def test_a_non_search_bash_call_does_not_import_the_session_ring(
    hook_env: dict[str, str], tmp_path: Path,
) -> None:
    """The cap check runs after extraction, so most Bash calls skip it.

    Reading the cap imports `aelfrice.session_ring`, which pulls
    `db_paths` and `store`. Every Bash call in a session runs this hook
    and the overwhelming majority are not searches, so the order of the
    two statements decides whether that import lands on all of them or
    only on the ones about to retrieve.

    The assertion is on the module set, not on wall-clock milliseconds,
    for the reason `test_hook_import_cost_1351.py` gives: a timing
    budget here is a flake generator under CI contention. The `rg` case
    is the control — without it this test would also pass if the Bash
    lane stopped consulting the ring at all.
    """
    assert _session_ring_imported("cat notes.txt", hook_env, tmp_path) is False
    assert _session_ring_imported("rg needle src/", hook_env, tmp_path) is True


@pytest.mark.timeout(300)
def test_a_turn_stamp_leaves_a_co_tenants_ring_record_intact(
    hook_env: dict[str, str], tmp_path: Path,
) -> None:
    """Session A's prompt must not wipe session B's ring record.

    Two sessions share one repo checkout — the configuration the whole
    per-session map exists for. B has injected beliefs, so B owns the
    record. A then submits a bare prompt that injects nothing.

    The turn stamp runs on every `UserPromptSubmit`, so if it went
    through the path that reshapes the record for the calling session it
    would reset B's #740 dedup ring, `next_fire_idx` and #876 cadence
    state on every prompt A submitted, and B would re-inject beliefs it
    had already injected. The stamp is asserted to have landed as well,
    so a stamp that simply stopped writing cannot pass this.
    """
    sess_a, sess_b = "cap-1522-tenant-A", "cap-1522-tenant-B"
    ring_path = Path(hook_env["AELFRICE_DB"]).parent / SESSION_RING_FILENAME
    before = {
        "session_id": sess_b,
        "ring": [{"id": f"tenant-b-{i}", "fire_idx": i} for i in range(4)],
        "ring_max": 40,
        "next_fire_idx": 4,
        "evicted_total": 2,
        "bytes_at_last_fire": 9999,
        "fire_idx_at_last_fire": 3,
        "classifications": [True, False, True],
        "phantom_fires": 1,
        "phantom_dedup": ["tenant-b-phantom"],
        "phantom_contradicts": ["x|y"],
        "phantom_init": True,
        "promotion_fires": 1,
        "promotion_dedup": ["tenant-b-promotion"],
        "bash": {
            sess_b: {
                "turn_id": 5, "fires": 2, "fires_turn_id": 5, "seq": 1,
            },
        },
    }
    ring_path.parent.mkdir(parents=True, exist_ok=True)
    ring_path.write_text(json.dumps(before), encoding="utf-8")

    _submit_prompt(hook_env, tmp_path, session=sess_a, prompt="ok")

    after = json.loads(ring_path.read_text(encoding="utf-8"))
    for field, want in before.items():
        if field == "bash":
            continue
        assert after[field] == want, field
    assert after["bash"][sess_b] == before["bash"][sess_b]
    # The stamp did happen: A's own entry advanced to turn 1.
    assert after["bash"][sess_a]["turn_id"] == 1, after["bash"]


def test_a_hand_edited_bash_map_is_bounded_on_the_next_write(
    hook_env: dict[str, str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An oversize map read off disk comes back trimmed, not carried.

    `BASH_STATE_MAX_SESSIONS` is enforced when the map is *written* too,
    so every path that adds an entry stays bounded on its own. This pins
    the other half: a ring file that already holds more entries than the
    cap — hand-edited, or left by an older build — is trimmed the next
    time the map is normalized, rather than being written back oversize
    by a path that adds no entry of its own.
    """
    monkeypatch.setenv("AELFRICE_DB", hook_env["AELFRICE_DB"])
    owner = "hand-edited-owner"
    oversize = 20
    assert oversize > BASH_STATE_MAX_SESSIONS
    ring_path = Path(hook_env["AELFRICE_DB"]).parent / SESSION_RING_FILENAME
    ring_path.parent.mkdir(parents=True, exist_ok=True)
    ring_path.write_text(
        json.dumps({
            "session_id": owner,
            "ring": [],
            "next_fire_idx": 0,
            "bash": {
                f"hand-{i:02d}": {
                    "turn_id": 1, "fires": 1, "fires_turn_id": 1, "seq": i,
                }
                for i in range(oversize)
            },
        }),
        encoding="utf-8",
    )

    # An injection write for the session that already owns the record:
    # it normalizes the map without touching any entry in it.
    assert append_ids(owner, ["hand-edited-belief"]) >= 0

    after = json.loads(ring_path.read_text(encoding="utf-8"))
    assert sorted(after["bash"]) == [
        f"hand-{i:02d}"
        for i in range(oversize - BASH_STATE_MAX_SESSIONS, oversize)
    ], sorted(after["bash"])
