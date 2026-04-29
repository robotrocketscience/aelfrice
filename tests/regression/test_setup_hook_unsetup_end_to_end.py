"""End-to-end regression: aelf setup -> spawn hook subprocess -> aelf unsetup.

Cumulative integration scenario added at v0.7.0. Exercises every
step of the Claude Code wiring round-trip:

  1. `aelf setup --scope project --project-root <tmp>` writes a
     UserPromptSubmit entry into <tmp>/.claude/settings.json.
  2. The recorded `command` is read back out of settings.json and
     spawned as a real subprocess with a Claude Code
     UserPromptSubmit JSON payload on stdin.
  3. The subprocess emits an <aelfrice-memory>...</aelfrice-memory>
     block on stdout containing seeded retrieval hits.
  4. `aelf unsetup` strips the entry; settings.json is left clean
     with no `UserPromptSubmit` entries.

The recorded command is overridden via `--command "<python> -m
aelfrice.hook"` (using `sys.executable`) rather than relying on the
CLI default. That keeps the test:
  - hermetic across `python` / `aelf-hook` PATH availability,
  - independent of CLI-default churn (the default may flip between
    `python -m aelfrice.hook` and `aelf-hook` depending on whether
    the script-entry PR has merged at the time this test runs).
"""
from __future__ import annotations

import io
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from aelfrice.cli import main as cli_main
from aelfrice.hook import CLOSE_TAG, OPEN_TAG
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
from aelfrice.store import MemoryStore

pytestmark = pytest.mark.regression


def _mk(
    bid: str,
    content: str,
    lock_level: str = LOCK_NONE,
    locked_at: str | None = None,
) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=lock_level,
        locked_at=locked_at,
        demotion_pressure=0,
        created_at="2026-04-26T00:00:00Z",
        last_retrieved_at=None,
    )


@pytest.fixture
def seeded_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """One locked belief + one FTS5-matchable unlocked belief."""
    db = tmp_path / "memory.db"
    store = MemoryStore(str(db))
    try:
        store.insert_belief(
            _mk(
                "L1",
                "the user pinned this as ground truth",
                lock_level=LOCK_USER,
                locked_at="2026-04-26T01:00:00Z",
            )
        )
        store.insert_belief(_mk("F1", "the kitchen is full of bananas"))
    finally:
        store.close()
    monkeypatch.setenv("AELFRICE_DB", str(db))
    return db


def _run_cli(*argv: str) -> tuple[int, str]:
    buf = io.StringIO()
    code = cli_main(argv=list(argv), out=buf)
    return code, buf.getvalue()


def _project_settings(tmp_path: Path) -> Path:
    return tmp_path / ".claude" / "settings.json"


def _read_recorded_command(settings: Path) -> str:
    raw = settings.read_text(encoding="utf-8")
    parsed = json.loads(raw)  # pyright: ignore[reportAny]
    assert isinstance(parsed, dict)
    data = cast(dict[str, object], parsed)
    hooks = data["hooks"]
    assert isinstance(hooks, dict)
    hooks_typed = cast(dict[str, object], hooks)
    entries = hooks_typed["UserPromptSubmit"]
    assert isinstance(entries, list)
    entries_typed = cast(list[dict[str, object]], entries)
    inner = entries_typed[0]["hooks"]
    assert isinstance(inner, list)
    inner_typed = cast(list[dict[str, object]], inner)
    cmd = inner_typed[0]["command"]
    assert isinstance(cmd, str)
    return cmd


@pytest.mark.timeout(30)
def test_setup_then_invoke_hook_then_unsetup_round_trip(
    seeded_db: Path, tmp_path: Path
) -> None:
    explicit_command = f"{shlex.quote(sys.executable)} -m aelfrice.hook"

    code, output = _run_cli(
        "setup",
        "--scope", "project",
        "--project-root", str(tmp_path),
        "--command", explicit_command,
    )
    assert code == 0
    settings = _project_settings(tmp_path)
    assert settings.exists()
    assert _read_recorded_command(settings) == explicit_command
    assert "installed" in output

    payload = json.dumps(
        {
            "session_id": "s-e2e",
            "transcript_path": "/dev/null",
            "cwd": str(tmp_path),
            "hook_event_name": "UserPromptSubmit",
            "prompt": "bananas",
        }
    )
    proc = subprocess.run(
        shlex.split(_read_recorded_command(settings)),
        input=payload,
        text=True,
        capture_output=True,
        timeout=20,
        env={**os.environ, "AELFRICE_DB": str(seeded_db)},
    )
    assert proc.returncode == 0, (
        f"hook subprocess exited {proc.returncode}; stderr={proc.stderr!r}"
    )
    assert proc.stdout.startswith(OPEN_TAG + "\n"), (
        f"missing open tag; stdout={proc.stdout!r}"
    )
    assert CLOSE_TAG in proc.stdout
    assert (
        '<belief id="L1" lock="user">'
        "the user pinned this as ground truth</belief>"
    ) in proc.stdout
    assert (
        '<belief id="F1" lock="none">'
        "the kitchen is full of bananas</belief>"
    ) in proc.stdout

    code, output = _run_cli(
        "unsetup",
        "--scope", "project",
        "--project-root", str(tmp_path),
        "--command", explicit_command,
    )
    assert code == 0
    assert "removed 1" in output
    raw = settings.read_text(encoding="utf-8")
    parsed = json.loads(raw)  # pyright: ignore[reportAny]
    assert isinstance(parsed, dict)
    data = cast(dict[str, object], parsed)
    hooks = data.get("hooks")
    if isinstance(hooks, dict):
        hooks_typed = cast(dict[str, object], hooks)
        ups = hooks_typed.get("UserPromptSubmit")
        if isinstance(ups, list):
            assert ups == []
