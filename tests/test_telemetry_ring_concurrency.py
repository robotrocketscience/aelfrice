"""#1145 telemetry ring: concurrent appenders don't lose records.

Both `_append_telemetry` implementations (`hook`, `hook_search_tool`)
do a read-all → trim → atomic-rewrite. Without an inter-process lock a
writer's rewrite is based on a snapshot taken before a sibling's
rewrite, silently dropping the sibling's record. The exclusive advisory
lock added in #1145 serialises the read-modify-write so every record up
to the ring cap survives contention.

These tests race real OS threads: `read_text` and `os.replace` are
distinct syscalls with the GIL released in between, so an unlocked
implementation drops updates here — the lock is what makes them pass.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from aelfrice.hook import _append_telemetry as _append_uprompt
from aelfrice.hook import read_user_prompt_submit_telemetry
from aelfrice.hook_search_tool import _append_telemetry as _append_search
from aelfrice.hook_search_tool import read_telemetry

_WORKERS = 8
_PER_WORKER = 25  # 200 total, well under the 1000 ring cap

# These tests contend on an advisory file lock, so they carry their own
# budget rather than relying on the suite default; every blocking call
# below has its own ceiling under this budget, so the test ends on an
# assertion rather than on a timeout.
_RACE_BUDGET_SECONDS = 30
_BARRIER_TIMEOUT_SECONDS = 10
_JOIN_TIMEOUT_SECONDS = 15


def _race(target) -> None:
    barrier = threading.Barrier(_WORKERS)
    errors: list[Exception] = []
    lock = threading.Lock()

    def worker(wid: int) -> None:
        try:
            # Bounded: an unbounded wait here strands the siblings if any
            # worker dies before reaching it, and non-daemon threads then
            # block interpreter shutdown — pytest prints its summary and
            # the process still never exits.
            barrier.wait(timeout=_BARRIER_TIMEOUT_SECONDS)
            for i in range(_PER_WORKER):
                target(wid, i)
        except Exception as exc:  # noqa: BLE001 - asserted below
            barrier.abort()
            with lock:
                errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(w,), daemon=True)
        for w in range(_WORKERS)
    ]
    for t in threads:
        t.start()
    # One deadline for the whole loop, not one per thread — see the same
    # note in tests/test_feedback_atomicity.py.
    deadline = time.monotonic() + _JOIN_TIMEOUT_SECONDS
    stuck = []
    for t in threads:
        t.join(timeout=max(0.0, deadline - time.monotonic()))
        if t.is_alive():
            stuck.append(t.name)
    assert not stuck, (
        f"workers did not finish within {_JOIN_TIMEOUT_SECONDS}s: {stuck}. "
        "The advisory lock is the thing under test; a hang must fail as an "
        "assertion, not as a suite timeout."
    )
    assert not errors, f"workers raised: {errors!r}"


@pytest.mark.timeout(_RACE_BUDGET_SECONDS)
def test_uprompt_appender_loses_no_records_under_contention(
    tmp_path: Path,
) -> None:
    tel = tmp_path / "user_prompt_submit.jsonl"

    def target(wid: int, i: int) -> None:
        _append_uprompt(tel, {"query": f"w{wid}-r{i}"})

    _race(target)

    records = read_user_prompt_submit_telemetry(tel)
    queries = {r["query"] for r in records}
    expected = {
        f"w{w}-r{i}" for w in range(_WORKERS) for i in range(_PER_WORKER)
    }
    assert len(records) == _WORKERS * _PER_WORKER
    assert queries == expected


@pytest.mark.timeout(_RACE_BUDGET_SECONDS)
def test_search_tool_appender_loses_no_records_under_contention(
    tmp_path: Path,
) -> None:
    tel = tmp_path / "search_tool.jsonl"

    def target(wid: int, i: int) -> None:
        _append_search(
            tel,
            session_id="s",
            command="rg",
            query=f"w{wid}-r{i}",
            latency_ms=1.0,
            injected_l1=0,
            injected_l0=0,
        )

    _race(target)

    records = read_telemetry(tel)
    queries = {r["query"] for r in records}
    expected = {
        f"w{w}-r{i}" for w in range(_WORKERS) for i in range(_PER_WORKER)
    }
    assert len(records) == _WORKERS * _PER_WORKER
    assert queries == expected
