"""`aelf doctor core-gate`: the core admission gate backlog drain (#1638).

The backlog is the active, unlocked beliefs that meet today's non-lock
core rule and have no label under the current classifier version.
`--emit` batches it for the host's classifier; a second emit prints a
still-open batch again instead of batching its beliefs twice. Plain
`aelf doctor` reports label coverage, and `aelf core-gate accept` runs the
spec's C-share self-check over the accepted batches of one emit run.

Each test names the mutation that kills it. The fixture text is neutral
and synthetic; every store lives under `tmp_path` through `AELFRICE_DB`.
"""
from __future__ import annotations

import io
import json
import re
import sqlite3
from pathlib import Path

import pytest

from aelfrice import core_gate
from aelfrice.cli import main
from aelfrice.core_gate import CLASSIFIER_VERSION, MAX_BATCH, build_prompt
from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    LOCK_USER,
    ORIGIN_AGENT_INFERRED,
    Belief,
    CoreGateBatchItem,
)
from aelfrice.store import MemoryStore

OLD_VERSION = "core-gate-0000000000000000"


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(path))
    monkeypatch.setenv("AELF_NO_UPDATE_CHECK", "1")
    monkeypatch.setenv("AELFRICE_NO_AUTO_INSTALL", "1")
    return path


def _bid(i: int) -> str:
    return f"widget{i:010d}"


def _hash(i: int) -> str:
    return f"h_widget{i:010d}"


def _add(store: MemoryStore, i: int, *, core: bool = True,
         locked: bool = False) -> None:
    """A belief in core through the posterior arm (alpha 9, beta 1), or
    in no arm (alpha 1, beta 1)."""
    store.insert_belief(Belief(
        id=_bid(i), content=f"the widgetprompt part {i} is blue",
        content_hash=_hash(i), alpha=9.0 if core else 1.0, beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_USER if locked else LOCK_NONE,
        locked_at="2026-08-01T09:00:00+00:00" if locked else None,
        created_at="2026-08-01T09:00:00+00:00", last_retrieved_at=None,
        origin=ORIGIN_AGENT_INFERRED,
    ))


def _store(db: Path, n: int) -> None:
    """`n` unlabeled core candidates, ids 0..n-1."""
    store = MemoryStore(str(db))
    try:
        for i in range(n):
            _add(store, i)
    finally:
        store.close()


def _label(db: Path, labels: dict[int, str], version: str = CLASSIFIER_VERSION) -> None:
    store = MemoryStore(str(db))
    try:
        store.put_core_gate_labels(
            {_hash(i): v for i, v in labels.items()},
            classifier_version=version, batch_id=None,
            labeled_at="2026-10-05T00:00:00+00:00",
        )
    finally:
        store.close()


def _run(*argv: str) -> tuple[int, str]:
    buf = io.StringIO()
    code = main(list(argv), out=buf)
    return code, buf.getvalue()


def _emit(*extra: str) -> dict[str, object]:
    code, out = _run("doctor", "core-gate", "--emit", "--json", *extra)
    assert code == 0
    return json.loads(out)


def _batches(db: Path) -> list[tuple[str, str, str | None, str, list[dict[str, object]]]]:
    conn = sqlite3.connect(str(db))
    try:
        return [
            (str(r[0]), str(r[1]), r[2], str(r[3]), json.loads(r[4]))
            for r in conn.execute(
                "SELECT batch_id, origin, session_id, classifier_version, "
                "items_json FROM core_gate_batches ORDER BY rowid"
            )
        ]
    finally:
        conn.close()


def _batched_ids(db: Path) -> list[str]:
    return [
        str(item["belief_id"])
        for _, _, _, _, items in _batches(db) for item in items
    ]


# --- the backlog --------------------------------------------------------


def test_emit_batches_only_unlabeled_candidates(db: Path) -> None:
    """A candidate with a label under the current version is not batched;
    one labeled only under an older version is.

    Killed by: dropping the `content_hash not in labels` filter in
    `emit_core_gate_batches`.
    """
    _store(db, 4)
    _label(db, {0: "A", 1: "C"})
    _label(db, {2: "A"}, version=OLD_VERSION)
    report = _emit()
    assert report["backlog"] == 2
    assert _batched_ids(db) == [_bid(2), _bid(3)]


def test_emit_skips_locked_retired_and_non_core_beliefs(db: Path) -> None:
    """Killed by: dropping the `lock_level != LOCK_NONE` skip in
    `core_gate_candidates`, or the `qualifies` test there."""
    store = MemoryStore(str(db))
    try:
        _add(store, 0)
        _add(store, 1, locked=True)
        _add(store, 2, core=False)
        _add(store, 3)
        store.soft_delete_belief(_bid(3))
    finally:
        store.close()
    _emit()
    assert _batched_ids(db) == [_bid(0)]


def test_emit_splits_the_backlog_evenly_in_id_order(db: Path) -> None:
    """53 candidates make two batches of 27 and 26, not 50 and 3, each
    indexed from 0 in ascending belief id order.

    Killed by: chunking at a fixed `MAX_BATCH` (sizes 50 and 3), or
    reversing the backlog.
    """
    _store(db, MAX_BATCH + 3)
    report = _emit()
    sizes = [b["size"] for b in report["batches"]]  # type: ignore[index]
    assert sizes == [27, 26]
    rows = _batches(db)
    flat = [(int(str(i["index"])), str(i["belief_id"])) for r in rows for i in r[4]]
    assert flat == (
        [(k, _bid(k)) for k in range(27)]
        + [(k, _bid(27 + k)) for k in range(26)]
    )


@pytest.mark.parametrize(
    ("n", "sizes"),
    [
        (0, []),
        (1, [1]),
        (MAX_BATCH, [MAX_BATCH]),
        (MAX_BATCH + 1, [26, 25]),
        (151, [38, 38, 38, 37]),
        (200, [50, 50, 50, 50]),
    ],
)
def test_balanced_chunks_sizes(n: int, sizes: list[int]) -> None:
    """ceil(n / MAX_BATCH) chunks, sizes within one of each other, larger
    first, items in order.

    Killed by: putting the extra item in the last chunks instead of the
    first, or using one chunk too many (`n // MAX_BATCH + 1`).
    """
    from aelfrice.doctor import balanced_chunks

    chunks = balanced_chunks(list(range(n)), MAX_BATCH)
    assert [len(c) for c in chunks] == sizes
    assert [x for c in chunks for x in c] == list(range(n))


def test_a_healthy_run_with_a_would_be_remainder_is_not_flagged(
    db: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """151 beliefs labeled with the same C share (every fifth snippet C)
    across one emit run: no batch is flagged. A fixed 50/50/50/1 split
    would leave a one-snippet batch whose share is 1.0.

    Killed by: chunking at a fixed `MAX_BATCH` in `emit_core_gate_batches`.
    """
    _store(db, 151)
    report = _emit()
    batches = list(report["batches"])  # type: ignore[arg-type]
    assert len(batches) == 4
    err = ""
    for b in batches:
        capsys.readouterr()
        size = int(b["size"])
        labels = ["C" if i % 5 == 0 else "A" for i in range(size)]
        assert _accept(monkeypatch, str(b["batch_id"]), labels) == 0
        err = capsys.readouterr().err
        assert _FLAG.findall(err) == []
    # The last accept ran the check (four accepted batches), not a skip.
    assert "self-check skipped" not in err


def test_emitted_batches_are_doctor_batches_of_one_run(db: Path) -> None:
    """Every new batch is origin `doctor`, has no session, carries the
    current classifier version, and shares the run's one `created_at`.

    Killed by: stamping each batch with its own timestamp (moving the
    `datetime.now` call into the batch loop), or origin `session_end`.
    """
    _store(db, MAX_BATCH + 1)
    _emit()
    rows = _batches(db)
    assert [r[1:4] for r in rows] == [("doctor", None, CLASSIFIER_VERSION)] * 2
    conn = sqlite3.connect(str(db))
    try:
        stamps = {
            str(r[0]) for r in conn.execute(
                "SELECT created_at FROM core_gate_batches"
            )
        }
    finally:
        conn.close()
    assert len(stamps) == 1


def test_emit_writes_only_batch_rows(db: Path) -> None:
    """Killed by: any write emit makes outside `core_gate_batches`, for
    example caching a placeholder label for each emitted belief."""
    _store(db, 3)

    def dump() -> dict[str, list[tuple[object, ...]]]:
        conn = sqlite3.connect(str(db))
        try:
            names = [
                str(r[0]) for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' "
                    "AND name NOT LIKE 'sqlite_%' AND name NOT LIKE '%fts%' "
                    "AND name != 'core_gate_batches' ORDER BY name"
                )
            ]
            return {
                n: sorted(conn.execute(f'SELECT * FROM "{n}"').fetchall(), key=repr)
                for n in names
            }
        finally:
            conn.close()

    _run("status")  # settle any open-time one-shots before the snapshot
    before = dump()
    _emit()
    assert dump() == before
    assert len(_batches(db)) == 1


def test_a_concurrent_emit_cannot_batch_the_same_beliefs(db: Path) -> None:
    """A second handle's emit, run while the first emit is between its
    reads and its first batch write, is refused by the write lock, and no
    belief ends up in two open batches.

    Killed by: running the emit outside `store.transaction(immediate=True)`
    (the second emit then batches every belief, and the first batches them
    again: 2 open batches holding each belief twice).
    """
    from aelfrice.cli import default_core_rule
    from aelfrice.doctor import emit_core_gate_batches

    _store(db, 3)
    a = MemoryStore(str(db))
    b = MemoryStore(str(db))
    b._conn.execute("PRAGMA busy_timeout=0")  # pyright: ignore[reportPrivateUsage]
    outcome: list[str] = []
    real_create = a.create_core_gate_batch

    def create_after_b(*args: object, **kwargs: object) -> str:
        if not outcome:
            try:
                emit_core_gate_batches(
                    b, default_core_rule, limit=None,
                    created_at="2026-10-05T05:00:00+00:00",
                )
                outcome.append("b emitted")
            except sqlite3.OperationalError:
                outcome.append("b locked out")
        return real_create(*args, **kwargs)  # type: ignore[arg-type]

    a.create_core_gate_batch = create_after_b  # type: ignore[method-assign]
    try:
        emit_core_gate_batches(
            a, default_core_rule, limit=None,
            created_at="2026-10-05T04:00:00+00:00",
        )
    finally:
        a.close()
        b.close()
    assert outcome == ["b locked out"]
    ids = _batched_ids(db)
    assert sorted(ids) == sorted(set(ids)) == [_bid(i) for i in range(3)]


# --- re-emit ------------------------------------------------------------


def test_a_second_emit_prints_the_open_batch_again(db: Path) -> None:
    """An open batch whose items are all still backlog is printed again,
    with the same id and prompt, and no new batch is created.

    Killed by: skipping the open-batch loop in `emit_core_gate_batches`
    (the beliefs are then batched a second time).
    """
    _store(db, 3)
    first = _emit()
    second = _emit()
    assert len(_batches(db)) == 1
    [a] = first["batches"]  # type: ignore[misc]
    [b] = second["batches"]  # type: ignore[misc]
    assert (b["batch_id"], b["prompt"], b["reused"]) == (a["batch_id"], a["prompt"], True)
    assert a["reused"] is False


def test_an_open_batch_with_a_labeled_item_is_set_aside(db: Path) -> None:
    """When one item of an open batch has been labeled since, the batch is
    not printed again; its remaining backlog goes into a new batch.

    Killed by: reusing an open batch when any item is still backlog
    (`all(` to `any(`), which would print it with a labeled item.
    """
    _store(db, 3)
    first = _emit()
    _label(db, {1: "A"})
    second = _emit()
    [b] = second["batches"]  # type: ignore[misc]
    assert b["reused"] is False
    assert b["batch_id"] != first["batches"][0]["batch_id"]  # type: ignore[index]
    assert _batched_ids(db)[3:] == [_bid(0), _bid(2)]


def _set_lock(db: Path, i: int, locked: bool) -> None:
    conn = sqlite3.connect(str(db))
    try:
        conn.execute(
            "UPDATE beliefs SET lock_level = ?, locked_at = ? WHERE id = ?",
            (
                LOCK_USER if locked else LOCK_NONE,
                "2026-10-05T00:00:00+00:00" if locked else None,
                _bid(i),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def test_two_open_batches_never_print_one_belief_twice(db: Path) -> None:
    """Emit 10, lock one (the first batch is set aside and a 9-belief
    batch is made), unlock it, emit again: both open batches are wholly
    backlog again, but only the older one is printed, so the backlog of
    10 prints 10 snippets.

    Killed by: dropping `item.content_hash not in claimed` from the
    open-batch reuse test (both are printed: 19 snippets).
    """
    _store(db, 10)
    first = _emit()
    _set_lock(db, 4, True)
    second = _emit()
    assert [b["size"] for b in second["batches"]] == [9]  # type: ignore[index, union-attr]
    _set_lock(db, 4, False)
    third = _emit()
    assert third["backlog"] == 10
    assert [(b["batch_id"], b["size"]) for b in third["batches"]] == [  # type: ignore[index, union-attr]
        (first["batches"][0]["batch_id"], 10),  # type: ignore[index]
    ]
    assert len(_batches(db)) == 2


def test_limit_caps_the_printed_batches_and_creates_none_past_it(db: Path) -> None:
    """`--limit` counts batches over the even split: 53 candidates split
    27/26, `--limit 1` creates only the first, and a second `--limit 1`
    prints only that open batch, creates nothing, and counts the other 26
    as left.

    Killed by: dropping the truncation of the new chunks under `limit`.
    """
    _store(db, MAX_BATCH + 3)
    _emit("--limit", "1")
    assert len(_batches(db)) == 1
    report = _emit("--limit", "1")
    assert [b["reused"] for b in report["batches"]] == [True]  # type: ignore[index]
    assert report["left"] == 26
    assert len(_batches(db)) == 1


# --- output -------------------------------------------------------------


def test_text_output_prints_each_prompt_and_accept_command(db: Path) -> None:
    """Killed by: dropping the prompt print or the `accept:` line."""
    _store(db, 2)
    code, out = _run("doctor", "core-gate", "--emit")
    assert code == 0
    [(bid, _, _, _, items)] = _batches(db)
    prompt = build_prompt([
        (int(str(i["index"])), f"the widgetprompt part {k} is blue")
        for k, i in enumerate(items)
    ])
    assert f"----- prompt -----\n{prompt}----- end of prompt -----\n" in out
    assert f"\naccept: aelf core-gate accept {bid}\n" in out


def test_out_writes_one_prompt_file_per_batch(db: Path, tmp_path: Path) -> None:
    """With `--out`, each prompt goes to `core-gate-<id>.txt` and stdout
    names the file instead of printing the prompt.

    Killed by: skipping the file write (the path is printed, no file).
    """
    _store(db, MAX_BATCH + 1)
    out_dir = tmp_path / "prompts"
    code, out = _run("doctor", "core-gate", "--emit", "--out", str(out_dir))
    assert code == 0
    assert "----- prompt -----" not in out
    ids = [r[0] for r in _batches(db)]
    assert sorted(p.name for p in out_dir.iterdir()) == sorted(
        f"core-gate-{i}.txt" for i in ids
    )
    for bid in ids:
        text = (out_dir / f"core-gate-{bid}.txt").read_text(encoding="utf-8")
        assert text.startswith(core_gate.PROMPT_HEADER)
        assert f"prompt: {out_dir / f'core-gate-{bid}.txt'}" in out


def test_an_empty_backlog_emits_nothing(db: Path) -> None:
    """Killed by: creating a batch with no items (the store refuses an
    empty batch, so the command raises)."""
    _store(db, 1)
    _label(db, {0: "A"})
    code, out = _run("doctor", "core-gate", "--emit")
    assert code == 0
    assert "nothing to emit." in out
    assert _batches(db) == []


@pytest.mark.parametrize(
    "argv",
    [
        ("doctor", "--emit"),
        ("doctor", "graph", "--limit", "2"),
        ("doctor", "core-gate", "--limit", "2"),
        ("doctor", "core-gate", "--out", "widgetdir"),
        ("doctor", "core-gate", "--emit", "--limit", "0"),
    ],
    ids=["emit-without-scope", "limit-other-scope", "limit-without-emit",
         "out-without-emit", "limit-zero"],
)
def test_misplaced_flags_exit_2_and_write_nothing(
    db: Path, argv: tuple[str, ...],
) -> None:
    """Killed by: removing the flag checks in `_cmd_doctor` and
    `_cmd_doctor_core_gate` (the command then runs and exits 0)."""
    _store(db, 1)
    code, _ = _run(*argv)
    assert code == 2
    assert _batches(db) == []


# --- coverage report ----------------------------------------------------


def _coverage_store(db: Path) -> None:
    store = MemoryStore(str(db))
    try:
        for i in range(5):
            _add(store, i)
        _add(store, 5, locked=True)
        _add(store, 6, core=False)
    finally:
        store.close()
    _label(db, {0: "A", 1: "B", 2: "C", 5: "C", 6: "A"})
    _label(db, {3: "A"}, version=OLD_VERSION)


EXPECTED_COVERAGE = {
    "classifier_version": CLASSIFIER_VERSION,
    "candidates": 5,
    "labeled": {"A": 1, "B": 1, "C": 1},
    "unlabeled": 2,
}


def test_core_gate_report_counts_labeled_and_unlabeled_candidates(db: Path) -> None:
    """Locked and non-core beliefs are not counted, whatever their label;
    a label under an older version counts as unlabeled. The report writes
    no batch.

    Killed by: counting a labeled candidate as unlabeled (dropping the
    `labeled[label] += 1` branch), or counting locked beliefs.
    """
    _coverage_store(db)
    code, out = _run("doctor", "core-gate", "--json")
    assert code == 0
    assert json.loads(out) == EXPECTED_COVERAGE
    assert _batches(db) == []


def test_health_reports_core_gate_coverage_in_text_and_json(db: Path) -> None:
    """The graph report (`aelf doctor graph`, `aelf health --json`)
    carries the coverage block and the `core_gate` key, and the unlabeled
    backlog does not change the exit code.

    Killed by: dropping the coverage block from `_cmd_health`'s text or
    JSON output.
    """
    _coverage_store(db)
    code, out = _run("doctor", "graph")
    assert code == 0
    assert (
        f"core admission gate ({CLASSIFIER_VERSION}):\n"
        "  5 unlocked core candidates: 3 labeled (A 1, B 1, C 1), 2 unlabeled\n"
    ) in out
    code, out = _run("health", "--json")
    assert code == 0
    assert json.loads(out)["core_gate"] == EXPECTED_COVERAGE


# --- accept-time self-check ---------------------------------------------


RUN_AT = "2026-10-05T01:00:00+00:00"


def _run_batches(db: Path, sizes: list[int], *, created_at: str = RUN_AT,
                 origin: str = "doctor", start: int = 0) -> list[str]:
    store = MemoryStore(str(db))
    ids: list[str] = []
    k = start
    try:
        for size in sizes:
            ids.append(store.create_core_gate_batch(
                [
                    CoreGateBatchItem(index=i, belief_id=_bid(k + i), content_hash=_hash(k + i))
                    for i in range(size)
                ],
                classifier_version=CLASSIFIER_VERSION, origin=origin,
                session_id=None if origin == "doctor" else "widgetsession",
                created_at=created_at,
            ))
            k += size
    finally:
        store.close()
    return ids


def _accept(
    monkeypatch: pytest.MonkeyPatch, batch_id: str, labels: list[str],
) -> int:
    reply = json.dumps([{"index": i, "label": v} for i, v in enumerate(labels)])
    monkeypatch.setattr("sys.stdin", io.StringIO(reply))
    return main(["core-gate", "accept", batch_id], out=io.StringIO())


_FLAG = re.compile(r"core-gate: self-check: batch (\w+) labeled ([\d.]+)")


def test_self_check_flags_the_outlier_batch_of_a_run(
    db: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Four batches of one run: three with no C label, one all C. The
    accept of the last flags only it and still exits 0.

    Killed by: never flagging (raising `SELF_CHECK_MAX_C_SHARE_GAP`
    above 1, or dropping the outlier print).
    """
    a, b, c, d = _run_batches(db, [4, 4, 4, 4])
    assert _accept(monkeypatch, a, ["A"] * 4) == 0
    assert _accept(monkeypatch, b, ["A", "B", "A", "A"]) == 0
    assert _accept(monkeypatch, c, ["B"] * 4) == 0
    capsys.readouterr()
    assert _accept(monkeypatch, d, ["C"] * 4) == 0
    assert _FLAG.findall(capsys.readouterr().err) == [(d, "1.00")]


def test_self_check_compares_only_batches_of_the_same_run(
    db: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Batches from another emit run (another `created_at`) are not
    siblings: a lone C batch after a run of four all-A batches is in a
    run of one, so the check is skipped and nothing is flagged.

    Killed by: dropping `b.created_at = run.created_at` from
    `core_gate_run_label_counts` (the five batches then form one run and
    the C batch is flagged).
    """
    run = _run_batches(db, [4, 4, 4, 4])
    [c] = _run_batches(db, [4], created_at="2026-10-05T02:00:00+00:00", start=16)
    for bid in run:
        _accept(monkeypatch, bid, ["A"] * 4)
    capsys.readouterr()
    assert _accept(monkeypatch, c, ["C"] * 4) == 0
    err = capsys.readouterr().err
    assert _FLAG.findall(err) == []
    assert "has 1 accepted batches" in err


def test_self_check_accepts_print_nothing_for_session_end_batches(
    db: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Session-end batches have no emit run, so their accepts print no
    self-check line, not even the skip line.

    Killed by: dropping the `batch.origin != CORE_GATE_ORIGIN_DOCTOR`
    return in `_cmd_core_gate` (the skip line then prints).
    """
    ids = _run_batches(db, [4, 4, 4, 4], origin="session_end")
    for bid in ids[:3]:
        _accept(monkeypatch, bid, ["A"] * 4)
    capsys.readouterr()
    assert _accept(monkeypatch, ids[3], ["C"] * 4) == 0
    assert "self-check" not in capsys.readouterr().err


def test_run_label_counts_are_empty_for_a_session_end_batch(
    db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Killed by: dropping the `run.origin = ?` condition from
    `core_gate_run_label_counts`."""
    ids = _run_batches(db, [4, 4], origin="session_end")
    for bid in ids:
        _accept(monkeypatch, bid, ["A"] * 4)
    store = MemoryStore(str(db))
    try:
        assert store.core_gate_run_label_counts(ids[0]) == {}
    finally:
        store.close()


def test_self_check_is_skipped_below_four_accepted_batches(
    db: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Three accepted batches, one all C: nothing is flagged, and one
    stderr line says the check was skipped and why.

    Killed by: lowering `SELF_CHECK_MIN_BATCHES` to 3 (all three are then
    flagged), or dropping the skip line.
    """
    a, b, c = _run_batches(db, [4, 4, 4])
    _accept(monkeypatch, a, ["A"] * 4)
    _accept(monkeypatch, b, ["A"] * 4)
    capsys.readouterr()
    assert _accept(monkeypatch, c, ["C"] * 4) == 0
    err = capsys.readouterr().err
    assert _FLAG.findall(err) == []
    assert (
        f"core-gate: self-check skipped: the emit run of batch {c} has 3 "
        "accepted batches, and the check needs at least 4."
    ) in err


@pytest.mark.parametrize(
    ("shares", "flagged"),
    [
        ({"x": 0.9}, []),
        ({"x": 0.0, "y": 0.0, "z": 0.25}, []),
        ({"x": 0.0, "y": 0.0, "z": 0.26}, [("z", 0.26, 0.0)]),
        ({"w": 0.1, "x": 0.2, "y": 0.3, "z": 0.9}, [("z", 0.9, 0.2)]),
        (
            {"x": 0.8, "y": 0.4, "z": 0.6},
            [("x", 0.8, 0.5), ("y", 0.4, 0.7)],
        ),
    ],
    ids=["single", "at-threshold", "over-threshold", "odd-median",
         "even-median"],
)
def test_c_share_outliers_uses_the_median_of_the_other_batches(
    shares: dict[str, float], flagged: list[tuple[str, float, float]],
) -> None:
    """A gap of exactly 0.25 is not flagged, the median is over the other
    batches only, and an even count takes the mean of the middle two.

    Killed by: `>` to `>=` in the gap test, or taking the median over all
    batches including the one tested.
    """
    got = [
        (b, round(s, 6), round(m, 6))
        for b, s, m in core_gate.c_share_outliers(shares)
    ]
    assert got == flagged


# --- re-run of one accepted batch ---------------------------------------


def _emitted_and_accepted(
    db: Path, monkeypatch: pytest.MonkeyPatch, n: int, label: str = "C",
) -> tuple[str, str]:
    """`n` core beliefs, one emitted batch, accepted with all `label`.
    Returns the batch id and its `created_at`."""
    _store(db, n)
    [batch] = _emit()["batches"]  # type: ignore[misc]
    assert _accept(monkeypatch, str(batch["batch_id"]), [label] * n) == 0
    return str(batch["batch_id"]), str(batch["created_at"])


def _labels(db: Path) -> list[tuple[str, str | None]]:
    conn = sqlite3.connect(str(db))
    try:
        return [
            (str(r[0]), r[1]) for r in conn.execute(
                "SELECT content_hash, batch_id FROM core_gate_labels "
                "ORDER BY content_hash"
            )
        ]
    finally:
        conn.close()


def _batch_column(db: Path, batch_id: str, column: str) -> object:
    conn = sqlite3.connect(str(db))
    try:
        row = conn.execute(
            f"SELECT {column} FROM core_gate_batches WHERE batch_id = ?",
            (batch_id,),
        ).fetchone()
    finally:
        conn.close()
    return None if row is None else row[0]


def _rerun(*extra: str) -> tuple[int, dict[str, object]]:
    code, out = _run("doctor", "core-gate", "--json", "--rerun", *extra)
    return code, (json.loads(out) if code == 0 else {})


def _retire(db: Path, *ids: int) -> None:
    store = MemoryStore(str(db))
    try:
        for i in ids:
            store.soft_delete_belief(_bid(i))
    finally:
        store.close()


def _items_of(db: Path, batch_id: str) -> list[str]:
    rows = {r[0]: r for r in _batches(db)}
    return [str(i["belief_id"]) for i in rows[batch_id][4]]


def test_rerun_of_a_whole_batch_reopens_it_in_its_run(
    db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With every belief still current, the re-run drops the batch's
    labels and reopens the same batch (its id derives from the run's
    `created_at` and the same hashes), which can then be accepted again.

    Killed by: skipping `reopen_core_gate_batch` (the second accept is
    refused as already accepted).
    """
    bid, _ = _emitted_and_accepted(db, monkeypatch, 3)
    code, report = _rerun(bid)
    assert code == 0
    assert (report["labels_dropped"], report["kept"], report["reopened"]) == (3, 3, True)
    assert [b["batch_id"] for b in report["batches"]] == [bid]  # type: ignore[index, union-attr]
    # The re-run is never re-split: it keeps the original batch's size.
    assert [b["size"] for b in report["batches"]] == [3]  # type: ignore[index, union-attr]
    assert _labels(db) == []
    assert _batch_column(db, bid, "accepted_at") is None
    assert _accept(monkeypatch, bid, ["A"] * 3) == 0
    assert [owner for _, owner in _labels(db)] == [bid] * 3


def test_rerun_of_a_partly_current_batch_makes_a_new_batch_in_the_same_run(
    db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retired belief drops out; the others go into a new batch that
    carries the original `created_at`, so it joins the original emit run.

    Killed by: creating the new batch with a fresh timestamp instead of
    the original batch's `created_at`.
    """
    bid, created_at = _emitted_and_accepted(db, monkeypatch, 3)
    _retire(db, 1)
    code, report = _rerun(bid)
    assert code == 0
    [fresh] = report["batches"]  # type: ignore[misc]
    fresh_id = str(fresh["batch_id"])
    assert fresh_id != bid
    assert _items_of(db, fresh_id) == [_bid(0), _bid(2)]
    assert _batch_column(db, fresh_id, "created_at") == created_at
    assert _labels(db) == []


def test_rerun_drops_only_labels_the_batch_still_owns(
    db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A label a later batch wrote for one of the hashes stays, and that
    belief is not re-batched.

    Killed by: deleting the labels without the `batch_id` condition, or
    dropping the `content_hash not in owned` skip (the relabeled belief is
    then re-batched).
    """
    bid, _ = _emitted_and_accepted(db, monkeypatch, 3)
    store = MemoryStore(str(db))
    try:
        store.put_core_gate_labels(
            {_hash(0): "A"}, classifier_version=CLASSIFIER_VERSION,
            batch_id="widgetlater00000", labeled_at="2026-10-05T03:00:00+00:00",
        )
    finally:
        store.close()
    code, report = _rerun(bid)
    assert code == 0
    assert report["labels_dropped"] == 2
    assert _labels(db) == [(_hash(0), "widgetlater00000")]
    [fresh] = report["batches"]  # type: ignore[misc]
    assert _items_of(db, str(fresh["batch_id"])) == [_bid(1), _bid(2)]


def test_rerun_with_no_belief_still_current_drops_labels_and_emits_nothing(
    db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Killed by: dropping the `if not kept: return report` early return
    (an empty batch is refused, so the command exits 1)."""
    bid, _ = _emitted_and_accepted(db, monkeypatch, 2)
    _retire(db, 0, 1)
    code, report = _rerun(bid)
    assert code == 0
    assert (report["labels_dropped"], report["kept"], report["batches"]) == (2, 0, [])
    assert len(_batches(db)) == 1


def test_rerun_is_one_transaction(
    db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When creating the fresh batch fails, the dropped labels come back.

    Killed by: running the re-run outside `store.transaction(...)` (the
    delete is then committed before the failure).
    """
    bid, _ = _emitted_and_accepted(db, monkeypatch, 3)
    _retire(db, 2)
    before = _labels(db)

    def refuse(*_a: object, **_k: object) -> str:
        raise ValueError("widgetprompt refusal")

    monkeypatch.setattr(MemoryStore, "create_core_gate_batch", refuse)
    code, _ = _rerun(bid)
    assert code == 1
    assert _labels(db) == before


def test_rerun_writes_the_prompt_file_with_out(
    db: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Killed by: not passing `--out` through to the prompt writer in
    `_cmd_doctor_core_gate_rerun`."""
    bid, _ = _emitted_and_accepted(db, monkeypatch, 2)
    out_dir = tmp_path / "rerun"
    code, report = _rerun(bid, "--out", str(out_dir))
    assert code == 0
    [b] = report["batches"]  # type: ignore[misc]
    path = out_dir / f"core-gate-{bid}.txt"
    assert b["prompt_file"] == str(path)
    assert path.read_text(encoding="utf-8").startswith(core_gate.PROMPT_HEADER)


def test_rerun_text_output_prints_the_batch_and_accept_command(
    db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Killed by: dropping the `_print_core_gate_batches` call from the
    re-run's text output."""
    bid, _ = _emitted_and_accepted(db, monkeypatch, 2)
    code, out = _run("doctor", "core-gate", "--rerun", bid)
    assert code == 0
    assert "----- prompt -----" in out
    assert f"\naccept: aelf core-gate accept {bid}\n" in out


def _refusal_case(db: Path, case: str) -> tuple[str, str]:
    """Build the batch for one refusal case; return its id and reason."""
    items = [
        CoreGateBatchItem(index=i, belief_id=_bid(i), content_hash=_hash(i))
        for i in range(2)
    ]
    if case == "unknown":
        return "widget0000000000", "no core-gate batch"
    version = OLD_VERSION if case == "other-version" else CLASSIFIER_VERSION
    origin = "session_end" if case == "session-end" else "doctor"
    store = MemoryStore(str(db))
    try:
        bid = store.create_core_gate_batch(
            items, classifier_version=version, origin=origin,
            session_id="widgetsession" if origin == "session_end" else None,
            created_at=RUN_AT,
        )
        if case != "not-accepted":
            store.accept_core_gate_batch(
                bid, {0: "C", 1: "C"}, classifier_version=version,
                accepted_at=RUN_AT,
            )
        if case == "no-labels":
            store.delete_core_gate_labels_of_batch(bid, version)
    finally:
        store.close()
    return bid, {
        "not-accepted": "never accepted",
        "other-version": "emitted under",
        "session-end": "session_end batch",
        "no-labels": "owns no labels",
    }[case]


@pytest.mark.parametrize(
    "case",
    ["unknown", "not-accepted", "other-version", "session-end", "no-labels"],
)
def test_rerun_refusals_exit_1_and_write_nothing(
    db: Path, capsys: pytest.CaptureFixture[str], case: str,
) -> None:
    """Each refusal names its reason and leaves the store unchanged.

    Killed by: removing any one of the five checks in
    `rerun_core_gate_batch` (a check another one masks is caught by its
    reason).
    """
    _store(db, 2)
    bid, reason = _refusal_case(db, case)
    before = (_labels(db), _batches(db), _batch_column(db, bid, "accepted_at"))
    capsys.readouterr()
    code, _ = _run("doctor", "core-gate", "--rerun", bid)
    assert code == 1
    assert reason in capsys.readouterr().err
    assert (_labels(db), _batches(db), _batch_column(db, bid, "accepted_at")) == before


@pytest.mark.parametrize(
    "argv",
    [
        ("doctor", "core-gate", "--rerun", "widget0000000000", "--emit"),
        ("doctor", "core-gate", "--rerun", "widget0000000000", "--limit", "1"),
        ("doctor", "graph", "--rerun", "widget0000000000"),
    ],
    ids=["with-emit", "with-limit", "other-scope"],
)
def test_rerun_with_emit_limit_or_another_scope_exits_2(
    db: Path, argv: tuple[str, ...],
) -> None:
    """Killed by: removing the `--rerun` combination check in
    `_cmd_doctor_core_gate`, or `--rerun` from `_cmd_doctor`'s scope
    check."""
    _store(db, 1)
    code, _ = _run(*argv)
    assert code == 2
    assert _batches(db) == []
