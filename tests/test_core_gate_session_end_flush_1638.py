"""The session-end core-gate batch across real ingest flushes (#1638).

Drives the installed entry points the host runs, in the host's order:
`aelf-transcript-logger` on UserPromptSubmit and on Stop, then
`aelf-stop-hook` on Stop, for 15 turns. The transcript logger folds the
session's turns into beliefs only when `STOP_FLUSH_TURNS` turn lines have
built up, by spawning `aelf ingest-transcript` in the background, so the
session has no beliefs until the first flush. The batch must fire on the
first Stop after that flush, stay quiet until the next one, and then fire
again with only the new beliefs.

A once-per-session trigger fails this: it is spent on the first Stop,
before any belief exists (the defect the 2026-10-06 ruling on #1638
fixed). Every store and transcript is under `tmp_path`.
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from aelfrice.transcript_logger import DEFAULT_STOP_FLUSH_TURNS

pytestmark = pytest.mark.timeout(300)

SESSION = "widgetsession-flush"
TURNS = 15
BIN = Path(sys.executable).parent


def _bin(name: str) -> str:
    path = BIN / name
    if not path.exists():
        pytest.fail(f"{name} is not installed beside {sys.executable}")
    return str(path)


def _run(name: str, payload: dict[str, object], env: dict[str, str], cwd: Path) -> str:
    proc = subprocess.run(
        [_bin(name)],
        input=json.dumps(payload).encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=cwd,
        check=False,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    return proc.stdout.decode("utf-8")


def _session_belief_ids(db: Path) -> set[str]:
    if not db.exists():
        return set()
    conn = sqlite3.connect(str(db), timeout=30)
    try:
        return {
            str(r[0]) for r in conn.execute(
                "SELECT id FROM beliefs WHERE session_id = ?", (SESSION,),
            )
        }
    except sqlite3.OperationalError:
        return set()
    finally:
        conn.close()


def _batched_ids(db: Path) -> list[set[str]]:
    conn = sqlite3.connect(str(db), timeout=30)
    try:
        rows = conn.execute(
            "SELECT items_json FROM core_gate_batches ORDER BY rowid"
        ).fetchall()
    finally:
        conn.close()
    return [{i["belief_id"] for i in json.loads(r[0])} for r in rows]


def _wait_for_beliefs(db: Path, want: int) -> set[str]:
    """Wait for the background ingest the flush spawned to land."""
    deadline = time.monotonic() + 120
    while True:
        ids = _session_belief_ids(db)
        if len(ids) >= want:
            return ids
        if time.monotonic() > deadline:
            pytest.fail(f"ingest landed {len(ids)} of {want} beliefs")
        time.sleep(0.2)


def test_batch_fires_after_each_ingest_flush(tmp_path: Path) -> None:
    """Killed by: claiming a once-per-session marker on the first Stop
    (the first fire never comes), or not claiming batched beliefs (every
    Stop after the first flush fires)."""
    work = tmp_path / "work"
    work.mkdir()
    db = tmp_path / "store" / "memory.db"
    env = {
        k: v for k, v in os.environ.items()
        if k not in {"CLAUDE_CODE_ENTRYPOINT", "AELFRICE_CORE_GATE_SESSION_END",
                     "AELFRICE_INGEST_STOP_FLUSH_TURNS", "AELF_SESSION_ID"}
    }
    env.update({
        "AELFRICE_DB": str(db),
        "AELFRICE_TRANSCRIPTS_DIR": str(tmp_path / "transcripts"),
        "AELF_NO_UPDATE_CHECK": "1",
        "AELFRICE_NO_AUTO_INSTALL": "1",
        "PATH": f"{BIN}{os.pathsep}{env.get('PATH', '')}",
    })
    flush_every = DEFAULT_STOP_FLUSH_TURNS // 2  # a turn writes two lines
    fired_at: list[int] = []
    seen: set[str] = set()
    for turn in range(1, TURNS + 1):
        base = {"session_id": SESSION, "cwd": str(work)}
        _run("aelf-transcript-logger", {
            **base, "hook_event_name": "UserPromptSubmit",
            "prompt": f"The widgetprompt cache in region {turn} holds {turn * 7} entries.",
        }, env, work)
        stop_payload = {
            **base, "hook_event_name": "Stop", "stop_hook_active": False,
            "transcript_path": None,
            "last_assistant_message": f"Noted region {turn}.",
        }
        _run("aelf-transcript-logger", stop_payload, env, work)
        if turn % flush_every == 0:
            # Ingested user turns so far; the assistant lines make no belief.
            seen = _wait_for_beliefs(db, turn)
        out = _run("aelf-stop-hook", stop_payload, env, work)
        if out.strip():
            obj = json.loads(out)
            assert obj["hookSpecificOutput"]["hookEventName"] == "Stop"
            fired_at.append(turn)
            batch = _batched_ids(db)[-1]
            already = set().union(*_batched_ids(db)[:-1])
            assert batch == seen - already
    assert fired_at == [flush_every, 2 * flush_every]
    batches = _batched_ids(db)
    assert len(batches) == 2
    assert batches[0].isdisjoint(batches[1])
    assert len(batches[0]) == flush_every
    assert len(batches[1]) == flush_every
