"""`aelf core-gate accept <batch-id>` (#1638).

Each test names the mutation that kills it. Every store is under
`tmp_path` through `AELFRICE_DB`; the reply text is synthetic.
"""
from __future__ import annotations

import io
import json
import sqlite3
from pathlib import Path

import pytest

from aelfrice.cli import main
from aelfrice.core_gate import CLASSIFIER_VERSION
from aelfrice.models import CoreGateBatchItem
from aelfrice.store import MemoryStore


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(path))
    monkeypatch.setenv("AELF_NO_UPDATE_CHECK", "1")
    monkeypatch.setenv("AELFRICE_NO_AUTO_INSTALL", "1")
    return path


def _make_batch(db: Path, version: str = CLASSIFIER_VERSION, n: int = 3) -> str:
    store = MemoryStore(str(db))
    try:
        return store.create_core_gate_batch(
            [
                CoreGateBatchItem(index=i, belief_id=f"widget{i}", content_hash=f"hash{i}")
                for i in range(n)
            ],
            classifier_version=version, origin="doctor", session_id=None,
            created_at="2026-10-05T00:00:00+00:00",
        )
    finally:
        store.close()


def _reply(labels: dict[int, str]) -> str:
    return json.dumps([{"index": i, "label": v} for i, v in labels.items()])


def _run(
    monkeypatch: pytest.MonkeyPatch, batch_id: str, stdin: str,
) -> tuple[int, str]:
    monkeypatch.setattr("sys.stdin", io.StringIO(stdin))
    buf = io.StringIO()
    code = main(["core-gate", "accept", batch_id], out=buf)
    return code, buf.getvalue()


def _state(db: Path) -> tuple[list[tuple[str, str, str]], list[tuple[str, object]]]:
    conn = sqlite3.connect(str(db))
    try:
        labels = [
            (str(r[0]), str(r[1]), str(r[2])) for r in conn.execute(
                "SELECT content_hash, classifier_version, label "
                "FROM core_gate_labels ORDER BY content_hash"
            )
        ]
        batches = [
            (str(r[0]), r[1]) for r in conn.execute(
                "SELECT batch_id, accepted_at FROM core_gate_batches "
                "ORDER BY batch_id"
            )
        ]
    finally:
        conn.close()
    return labels, batches


def test_accept_caches_the_labels_and_prints_the_counts(
    db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Killed by: returning 0 before `accept_core_gate_batch` is called,
    or miscounting a label in the summary."""
    bid = _make_batch(db)
    code, out = _run(monkeypatch, bid, _reply({2: "C", 0: "A", 1: "A"}))
    assert code == 0
    assert out.strip() == (
        f"core-gate: accepted batch {bid}: 3 labels (A 2, B 0, C 1)"
    )
    labels, batches = _state(db)
    assert labels == [
        ("hash0", CLASSIFIER_VERSION, "A"),
        ("hash1", CLASSIFIER_VERSION, "A"),
        ("hash2", CLASSIFIER_VERSION, "C"),
    ]
    assert batches[0][1] is not None


def test_a_stale_batch_is_refused_ahead_of_a_parse_error(
    db: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A batch from another classifier version is refused as stale, even
    when the reply is also malformed, and nothing is written.

    Killed by: removing the CLI's version check (the parse error is then
    reported instead).
    """
    bid = _make_batch(db, version="core-gate-0000000000000000")
    before = _state(db)
    code, _ = _run(monkeypatch, bid, "not json")
    assert code == 1
    assert "emit a new batch" in capsys.readouterr().err
    assert _state(db) == before


def test_a_stale_batch_with_a_valid_reply_writes_nothing(
    db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Killed by: bypassing the version gate (dropping the CLI check and
    passing the batch's own version to the store)."""
    bid = _make_batch(db, version="core-gate-0000000000000000")
    before = _state(db)
    code, _ = _run(monkeypatch, bid, _reply({0: "A", 1: "A", 2: "A"}))
    assert code == 1
    assert _state(db) == before


@pytest.mark.parametrize(
    "stdin",
    [
        "",
        "widgetprompt reply without json",
        _reply({0: "A", 1: "B"}),
        _reply({0: "A", 1: "B", 2: "C", 3: "A"}),
        _reply({0: "A", 1: "B", 2: "D"}),
        json.dumps({"index": 0, "label": "A"}),
    ],
    ids=["empty", "not-json", "partial", "extra-index", "bad-label", "not-array"],
)
def test_a_malformed_or_partial_reply_is_refused_with_nothing_written(
    db: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], stdin: str,
) -> None:
    """Killed by: swallowing the `parse_labels` ValueError and accepting an
    empty label set (the store's coverage check then answers, with a
    different message)."""
    bid = _make_batch(db)
    before = _state(db)
    code, out = _run(monkeypatch, bid, stdin)
    assert code == 1
    assert out == ""
    assert "reply refused, nothing written" in capsys.readouterr().err
    assert _state(db) == before


def test_a_second_accept_of_one_batch_is_refused(
    db: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The first accept's labels stand.

    Killed by: removing the CLI's already-accepted check together with
    the store's check and its `accepted_at IS NULL` guard. The layers
    back each other up, so removing one alone is caught by the next.
    """
    bid = _make_batch(db)
    assert _run(monkeypatch, bid, _reply({0: "A", 1: "A", 2: "A"}))[0] == 0
    first = _state(db)
    capsys.readouterr()
    code, _ = _run(monkeypatch, bid, _reply({0: "C", 1: "C", 2: "C"}))
    assert code == 1
    assert "already accepted" in capsys.readouterr().err
    assert _state(db) == first


def test_an_unknown_batch_is_refused(
    db: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Killed by: removing the `batch is None` check (AttributeError)."""
    MemoryStore(str(db)).close()
    code, _ = _run(monkeypatch, "0123456789abcdef", _reply({0: "A"}))
    assert code == 1
    assert "no core-gate batch" in capsys.readouterr().err
    assert _state(db) == ([], [])


def test_the_store_refusal_exits_one(
    db: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A refusal raised inside the accept transaction (here, a concurrent
    accept between the CLI's read and the store's) exits 1, not 0 and
    not a traceback.

    Killed by: removing the `except CoreGateAcceptRefused` handler, or
    returning 0 from it.
    """
    from aelfrice.store import CoreGateAcceptRefused

    bid = _make_batch(db)

    def _refuse(self: MemoryStore, *_a: object, **_k: object) -> None:
        raise CoreGateAcceptRefused("widget refusal")

    monkeypatch.setattr(MemoryStore, "accept_core_gate_batch", _refuse)
    code, out = _run(monkeypatch, bid, _reply({0: "A", 1: "A", 2: "A"}))
    assert code == 1
    assert out == ""
    assert "widget refusal" in capsys.readouterr().err


def test_core_gate_accept_is_a_registered_action(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Killed by: renaming or removing the `accept` subparser (argparse
    then exits 2 on an invalid choice)."""
    with pytest.raises(SystemExit) as exc:
        main(["core-gate", "accept", "--help"])
    assert exc.value.code == 0
    assert "batch_id" in capsys.readouterr().out


_RACE_SCRIPT = """
import sys, time
from aelfrice.cli import main
start = float(sys.argv[2])
while time.time() < start:
    time.sleep(0.001)
sys.exit(main(["core-gate", "accept", sys.argv[1]]))
"""


@pytest.mark.timeout(120)
def test_two_concurrent_accepts_of_one_batch_have_exactly_one_winner(
    db: Path,
) -> None:
    """Two processes race on one batch: one writes its labels, one is refused.

    Three guards each stop a second accept: the CLI's pre-check, the
    re-check under `BEGIN IMMEDIATE`, and the `accepted_at IS NULL` update.
    Killed by: removing all three (both processes then win). Any one alone
    keeps this green, which is the point of having three.
    """
    import os
    import subprocess
    import sys
    import time

    bid = _make_batch(db, n=3)
    start = str(time.time() + 2.0)
    replies = (
        {0: "A", 1: "B", 2: "C"},
        {0: "C", 1: "C", 2: "C"},
    )
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _RACE_SCRIPT, bid, start],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env={**os.environ}, text=True, encoding="utf-8",
        )
        for _ in replies
    ]
    codes: list[int] = []
    for proc, reply in zip(procs, replies, strict=True):
        proc.communicate(_reply(reply), timeout=60)
        codes.append(proc.returncode)
    assert sorted(codes) == [0, 1], codes
    labels, batches = _state(db)
    winner = replies[codes.index(0)]
    assert labels == [
        (f"hash{i}", CLASSIFIER_VERSION, winner[i]) for i in range(3)
    ]
    assert len(batches) == 1 and batches[0][1] is not None
