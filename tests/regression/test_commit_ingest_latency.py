"""commit-ingest hook latency regression test.

Spec target (docs/commit_ingest_hook.md):
    median <= 30 ms, p95 <= 100 ms

Cold-start Python eats most of that on a real subprocess run, so the
spec target is for the in-process hook body. We measure exactly that:
build a payload pointing at a real commit in a tmp git repo, then
time the hook's `_do_ingest` end-to-end across N=20 iterations.

Asserts a generous 200ms p95 envelope so the test catches a 10x
regression without flaking on shared CI runners. Median is reported
but not asserted on — assertion would be too noisy.
"""
from __future__ import annotations

import statistics
import subprocess
import time
from pathlib import Path

import pytest

from aelfrice import hook_commit_ingest as hk

P95_BUDGET_MS = 200.0
ITERATIONS = 20


def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=False,
    )
    if r.returncode != 0:
        raise RuntimeError(f"git {args!r} failed: {r.stderr}")
    return r.stdout


@pytest.fixture
def commit_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "f").write_text("x")
    _git(repo, "add", "f")
    msg = (
        "the new index supports faster queries. "
        "the proposal cites the prior memo. "
        "the spec is derived from the earlier draft."
    )
    msg_file = repo / ".msg"
    msg_file.write_text(msg)
    _git(repo, "commit", "-q", "-F", str(msg_file))
    short = _git(repo, "rev-parse", "--short", "HEAD").strip()
    branch = _git(repo, "symbolic-ref", "--short", "HEAD").strip()
    db = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    return {
        "hook_event_name": "PostToolUse",
        "tool_name": "Bash",
        "tool_input": {"command": "git commit -F .msg"},
        "tool_response": {
            "stdout": f"[{branch} {short}] {msg.splitlines()[0]}",
            "stderr": "",
            "isError": False,
            "interrupted": False,
        },
        "cwd": str(repo),
    }


def test_in_process_p95_under_budget(commit_payload: dict[str, object]) -> None:
    timings_ms: list[float] = []
    # Warm up once so the first run does not skew the bucket (lazy
    # imports happen on first call).
    hk._do_ingest(commit_payload)  # pyright: ignore[reportPrivateUsage]
    for _ in range(ITERATIONS):
        t0 = time.perf_counter()
        hk._do_ingest(commit_payload)  # pyright: ignore[reportPrivateUsage]
        timings_ms.append((time.perf_counter() - t0) * 1000.0)
    timings_ms.sort()
    p95 = timings_ms[int(len(timings_ms) * 0.95)]
    median = statistics.median(timings_ms)
    assert p95 < P95_BUDGET_MS, (
        f"p95={p95:.2f}ms exceeds {P95_BUDGET_MS}ms regression budget; "
        f"median={median:.2f}ms"
    )
