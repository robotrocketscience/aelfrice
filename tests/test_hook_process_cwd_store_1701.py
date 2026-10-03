"""#1701: the search-tool and agent-context hooks read the process-cwd store.

Both hooks used to read the payload `cwd` and pass it to `db_path(cwd=...)`
behind a signature probe. `db_path()` has never taken `cwd`, so the probe
was always false and the payload `cwd` was dead. #1630 settled the rule
for a whole turn: the store resolves from the hook process's cwd, which
the host sets to the session's directory. These tests pin that rule for
both hooks with a payload `cwd` that names a different repository, so a
change that starts honouring the payload `cwd` fails here.
"""
from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path

import pytest

from aelfrice import hook_agent_context, hook_search_tool
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief
from aelfrice.store import MemoryStore


def _repo(root: Path) -> Path:
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True, timeout=30)
    return root


def _seed(repo: Path, bid: str, content: str) -> None:
    db = repo / ".git" / "aelfrice" / "memory.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(str(db))
    try:
        store.insert_belief(Belief(
            id=bid, content=content, content_hash=f"h_{bid}", alpha=1.0,
            beta=1.0, type=BELIEF_FACTUAL, lock_level=LOCK_NONE,
            locked_at=None, created_at="2026-01-01T00:00:00Z",
            last_retrieved_at=None,
        ))
    finally:
        store.close()


@pytest.fixture
def two_repos(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Process cwd in repo A, a different repo B for the payload `cwd`."""
    a = _repo(tmp_path / "a")
    b = _repo(tmp_path / "b")
    _seed(a, "A0000001", "widgetprompt alpha lives in the process cwd store")
    _seed(b, "B0000001", "widgetprompt beta lives in the payload cwd store")
    monkeypatch.delenv("AELFRICE_DB", raising=False)
    monkeypatch.chdir(a)
    return a, b


def _run(module: object, payload: dict[str, object]) -> str:
    out = io.StringIO()
    rc = module.main(  # type: ignore[attr-defined]
        stdin=io.StringIO(json.dumps(payload)), stdout=out, stderr=io.StringIO(),
    )
    assert rc == 0
    return out.getvalue()


@pytest.mark.timeout(60)
def test_search_tool_hook_reads_the_process_cwd_store(
    two_repos: tuple[Path, Path],
) -> None:
    _, b = two_repos
    out = _run(hook_search_tool, {
        "hook_event_name": "PreToolUse", "tool_name": "Grep",
        "tool_input": {"pattern": "widgetprompt"}, "cwd": str(b),
        "session_id": "s1701",
    })
    assert "widgetprompt alpha" in out
    assert "widgetprompt beta" not in out


@pytest.mark.timeout(60)
def test_agent_context_hook_reads_the_process_cwd_store(
    two_repos: tuple[Path, Path],
) -> None:
    _, b = two_repos
    out = _run(hook_agent_context, {
        "hook_event_name": "PreToolUse", "tool_name": "Agent",
        "tool_input": {"description": "d", "prompt": "check widgetprompt"},
        "cwd": str(b), "session_id": "s1701",
    })
    assert "widgetprompt alpha" in out
    assert "widgetprompt beta" not in out
