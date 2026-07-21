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
from pathlib import Path

from aelfrice.hook import _append_telemetry as _append_uprompt
from aelfrice.hook import read_user_prompt_submit_telemetry
from aelfrice.hook_search_tool import _append_telemetry as _append_search
from aelfrice.hook_search_tool import read_telemetry

_WORKERS = 8
_PER_WORKER = 25  # 200 total, well under the 1000 ring cap


def _race(target) -> None:
    barrier = threading.Barrier(_WORKERS)

    def worker(wid: int) -> None:
        barrier.wait()  # maximise overlap on the read-modify-write
        for i in range(_PER_WORKER):
            target(wid, i)

    threads = [
        threading.Thread(target=worker, args=(w,)) for w in range(_WORKERS)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


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
