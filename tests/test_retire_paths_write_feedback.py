"""#2019: every path that retires a belief writes one feedback row.

Wonder GC, the scheduled-prompt pass, and `aelf doctor --prune-noise` set
`valid_to` on a belief. Each one also writes one `feedback_history` row
per belief it retires, at valence 0.0, under a source of its own, in the
same transaction as the retirement. A dry run writes nothing.

These tests don't name the new sources. They read the source each path
writes from the rows it lands, and check that the three differ, that no
source is an exposure-only one, and that wonder GC ignores each when it
picks candidates.

Every store here is a temporary file. `AELFRICE_DB`, `AELFRICE_USER_DB`,
`HOME`, and `AELFRICE_TRANSCRIPTS_DIR` point under `tmp_path` for every
test.
"""
from __future__ import annotations

import contextlib
import io
import json
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

import aelfrice.cli as cli_module
from aelfrice import hook as hook_module
from aelfrice.ingest import ingest_jsonl
from aelfrice.models import (
    BELIEF_FACTUAL,
    BELIEF_SPECULATIVE,
    EXPOSURE_ONLY_FEEDBACK_SOURCES,
    LOCK_NONE,
    ORIGIN_AGENT_INFERRED,
    ORIGIN_SPECULATIVE,
    ORIGIN_USER_TRANSCRIPT,
    Belief,
)
from aelfrice.store import MemoryStore
from aelfrice.wonder.lifecycle import wonder_gc

_REAL_CONNECT = sqlite3.connect

# The value on main when #2019 was filed. The fix must leave it alone.
EXPOSURE_ONLY_ON_MAIN = frozenset({"hook"})

SESSION = "sess-2019"
SCHEDULED_ONLY = "The release checklist requires a signed tag before publishing."
TYPED_ONLY = "The staging database lives on the second replica host."

# Noise that `--prune-noise` retires, and a fact it keeps. The shapes come
# from tests/test_doctor_prune_noise.py.
NOISE = {
    "n_tag": "<tool-use-id>toolu_abc</tool-use-id>",
    "n_q": "Want me to run it?",
    "n_ack": "Polling.",
}
KEEP_FACT = "The project stores beliefs in SQLite under .git."


# --- Environment -------------------------------------------------------


@pytest.fixture(autouse=True)
def _pin_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "default" / "memory.db"))
    monkeypatch.setenv("AELFRICE_USER_DB", str(tmp_path / "user" / "memory.db"))
    monkeypatch.setenv("AELFRICE_TRANSCRIPTS_DIR", str(tmp_path / "transcripts"))
    monkeypatch.setenv("AELFRICE_WONDER_AUTOGC", "1")
    monkeypatch.delenv("AELFRICE_WONDER_AUTOGC_TTL_DAYS", raising=False)


def _use_db(monkeypatch: pytest.MonkeyPatch, db: Path) -> Path:
    db.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AELFRICE_DB", str(db))
    return db


def _open(db: Path) -> contextlib.closing[MemoryStore]:
    return contextlib.closing(MemoryStore(str(db)))


# --- Reading the store -------------------------------------------------


def _max_feedback_id(db: Path) -> int:
    conn = _REAL_CONNECT(str(db))
    try:
        row = conn.execute("SELECT COALESCE(MAX(id), 0) FROM feedback_history").fetchone()
        return int(row[0])
    finally:
        conn.close()


def _rows_after(db: Path, after_id: int) -> list[tuple[str, float, str]]:
    """`(belief_id, valence, source)` for each row with an id past `after_id`."""
    conn = _REAL_CONNECT(str(db))
    try:
        return [
            (str(r[0]), float(r[1]), str(r[2]))
            for r in conn.execute(
                "SELECT belief_id, valence, source FROM feedback_history "
                "WHERE id > ? ORDER BY id",
                (after_id,),
            )
        ]
    finally:
        conn.close()


def _state(db: Path, ids: list[str]) -> dict[str, tuple[str | None, float, float]]:
    """`(valid_to, alpha, beta)` per belief id, retired or not."""
    conn = _REAL_CONNECT(str(db))
    try:
        out: dict[str, tuple[str | None, float, float]] = {}
        for bid in ids:
            row = conn.execute(
                "SELECT valid_to, alpha, beta FROM beliefs WHERE id = ?", (bid,),
            ).fetchone()
            assert row is not None, bid
            out[bid] = (row[0], float(row[1]), float(row[2]))
        return out
    finally:
        conn.close()


def _retired(db: Path, ids: list[str]) -> set[str]:
    return {bid for bid, (vt, _, _) in _state(db, ids).items() if vt is not None}


def _assert_one_row_per_retired(
    rows: list[tuple[str, float, str]], retired: set[str],
) -> str:
    """Check the rows a run wrote and return the one source they share."""
    assert retired, "the run retired nothing, so the check would be vacuous"
    by_belief: dict[str, int] = {}
    for bid, _, _ in rows:
        by_belief[bid] = by_belief.get(bid, 0) + 1
    assert by_belief == {bid: 1 for bid in retired}, rows
    assert {valence for _, valence, _ in rows} == {0.0}, rows
    sources = {source for _, _, source in rows}
    assert len(sources) == 1, rows
    source = sources.pop()
    assert source, rows
    assert source not in EXPOSURE_ONLY_FEEDBACK_SOURCES, source
    return source


# --- Seeding -----------------------------------------------------------


def _ts(days_ago: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


def _phantom(bid: str, content: str, *, days_ago: int = 20) -> Belief:
    return Belief(
        id=bid, content=content, content_hash=f"wonder_{bid}", alpha=0.3,
        beta=1.0, type=BELIEF_SPECULATIVE, lock_level=LOCK_NONE,
        locked_at=None, created_at=_ts(days_ago), last_retrieved_at=None,
        session_id="s0", origin=ORIGIN_SPECULATIVE,
    )


def _seed_phantoms(db: Path) -> tuple[list[str], list[str]]:
    """Two stale phantoms GC collects and one fresh phantom it keeps."""
    with _open(db) as store:
        store.insert_belief(_phantom("p_old1", "The cache expires after ten minutes."))
        store.insert_belief(_phantom("p_old2", "Writes bypass the cache entirely."))
        store.insert_belief(
            _phantom("p_new", "The queue drains every hour.", days_ago=1),
        )
    return ["p_old1", "p_old2"], ["p_new"]


def _agent_belief(bid: str, content: str) -> Belief:
    return Belief(
        id=bid, content=content, content_hash=f"h_{bid}", alpha=1.0, beta=1.0,
        type=BELIEF_FACTUAL, lock_level=LOCK_NONE, locked_at=None,
        created_at="2026-06-01T00:00:00Z", last_retrieved_at=None,
        origin=ORIGIN_AGENT_INFERRED,
    )


def _seed_noise(db: Path) -> tuple[list[str], list[str]]:
    with _open(db) as store:
        for bid, content in NOISE.items():
            store.insert_belief(_agent_belief(bid, content))
        store.insert_belief(_agent_belief("keep_fact", KEEP_FACT))
    return sorted(NOISE), ["keep_fact"]


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def _host(text: str, *, scheduled: bool) -> dict[str, object]:
    rec: dict[str, object] = {
        "type": "user", "sessionId": SESSION,
        "timestamp": "2026-10-11T15:00:00.000Z",
        "message": {"role": "user", "content": text},
    }
    if scheduled:
        rec.update(
            isMeta=True, promptSource="system", turnOrigin="scheduled",
            scheduledTaskId="task-1", queuePriority="later",
        )
    else:
        rec.update(promptSource="typed", turnOrigin="human")
    return rec


def _seed_scheduled(db: Path, work: Path) -> tuple[list[str], list[str], Path]:
    """A belief only a scheduled turn made, one a typed turn made, and the host file."""
    turns = _write_jsonl(work / "turns.jsonl", [
        {"role": "user", "text": text, "session_id": SESSION,
         "ts": f"2026-10-11T15:0{minute}:00.000000+00:00",
         "turn_id": f"t{minute}", "schema_version": 1}
        for minute, text in ((1, SCHEDULED_ONLY), (2, TYPED_ONLY))
    ])
    with _open(db) as store:
        ingest_jsonl(store, turns)
        by_text = {b.content: b for b in store.list_active_beliefs()}
    scheduled, typed = by_text[SCHEDULED_ONLY], by_text[TYPED_ONLY]
    assert scheduled.origin == ORIGIN_USER_TRANSCRIPT
    assert typed.origin == ORIGIN_USER_TRANSCRIPT
    host = _write_jsonl(work / "host" / f"{SESSION}.jsonl", [
        _host(SCHEDULED_ONLY, scheduled=True),
        _host(TYPED_ONLY, scheduled=False),
    ])
    return [scheduled.id], [typed.id], host


# --- Entry points ------------------------------------------------------


def _run_gc_hook() -> str:
    """Wonder GC as the SessionStart hook runs it."""
    err = io.StringIO()
    hook_module._maybe_run_wonder_autogc(err)  # pyright: ignore[reportPrivateUsage]
    return err.getvalue()


def _cli(*argv: str) -> tuple[int, str]:
    buf = io.StringIO()
    code = cli_module.main(argv=list(argv), out=buf)
    return code, buf.getvalue()


def _run_prune_noise(*, apply: bool) -> tuple[int, str]:
    return _cli("doctor", "--prune-noise", *(["--apply"] if apply else []))


def _run_scheduled(host: Path, *, apply: bool) -> tuple[int, str]:
    return _cli(
        "doctor", "--retire-scheduled-prompts", str(host),
        *(["--apply"] if apply else []),
    )


# Each helper below runs one path once on its own store and returns the
# source it wrote, so a test can plant or compare sources without naming
# them.


def _gc_source(monkeypatch: pytest.MonkeyPatch, work: Path) -> str:
    db = _use_db(monkeypatch, work / "gc" / "memory.db")
    retire, keep = _seed_phantoms(db)
    before = _max_feedback_id(db)
    _run_gc_hook()
    assert _retired(db, retire + keep) == set(retire)
    return _assert_one_row_per_retired(_rows_after(db, before), set(retire))


def _prune_source(monkeypatch: pytest.MonkeyPatch, work: Path) -> str:
    db = _use_db(monkeypatch, work / "prune" / "memory.db")
    retire, keep = _seed_noise(db)
    before = _max_feedback_id(db)
    code, _ = _run_prune_noise(apply=True)
    assert code == 0
    assert _retired(db, retire + keep) == set(retire)
    return _assert_one_row_per_retired(_rows_after(db, before), set(retire))


def _scheduled_source(monkeypatch: pytest.MonkeyPatch, work: Path) -> str:
    db = _use_db(monkeypatch, work / "sched" / "memory.db")
    retire, keep, host = _seed_scheduled(db, work / "sched")
    before = _max_feedback_id(db)
    code, _ = _run_scheduled(host, apply=True)
    assert code == 0
    assert _retired(db, retire + keep) == set(retire)
    return _assert_one_row_per_retired(_rows_after(db, before), set(retire))


# --- One row per retired belief, valence 0.0, none on a dry run --------


def test_wonder_gc_writes_one_row_per_collected_phantom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _use_db(monkeypatch, tmp_path / "memory.db")
    retire, keep = _seed_phantoms(db)
    before = _max_feedback_id(db)
    _run_gc_hook()
    assert _retired(db, retire + keep) == set(retire)
    _assert_one_row_per_retired(_rows_after(db, before), set(retire))

    # A second sweep retires nothing, so it writes nothing.
    after_first = _max_feedback_id(db)
    _run_gc_hook()
    assert _rows_after(db, after_first) == []


def test_wonder_gc_dry_run_writes_no_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _use_db(monkeypatch, tmp_path / "memory.db")
    retire, keep = _seed_phantoms(db)
    before = _max_feedback_id(db)
    with _open(db) as store:
        result = wonder_gc(store, ttl_days=14, dry_run=True)
    assert result.scanned == len(retire)
    assert _retired(db, retire + keep) == set()
    assert _rows_after(db, before) == []


def test_prune_noise_writes_one_row_per_retired_belief(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _use_db(monkeypatch, tmp_path / "memory.db")
    retire, keep = _seed_noise(db)
    before = _max_feedback_id(db)
    code, out = _run_prune_noise(apply=True)
    assert code == 0, out
    assert _retired(db, retire + keep) == set(retire)
    _assert_one_row_per_retired(_rows_after(db, before), set(retire))

    after_first = _max_feedback_id(db)
    code, _ = _run_prune_noise(apply=True)
    assert code == 0
    assert _rows_after(db, after_first) == []


def test_prune_noise_dry_run_writes_no_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _use_db(monkeypatch, tmp_path / "memory.db")
    retire, keep = _seed_noise(db)
    before = _max_feedback_id(db)
    code, out = _run_prune_noise(apply=False)
    assert code == 0
    assert f"{len(retire)} active agent_inferred" in out
    assert _retired(db, retire + keep) == set()
    assert _rows_after(db, before) == []


def test_scheduled_prompts_write_one_row_per_retired_belief(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _use_db(monkeypatch, tmp_path / "memory.db")
    retire, keep, host = _seed_scheduled(db, tmp_path)
    before = _max_feedback_id(db)
    code, out = _run_scheduled(host, apply=True)
    assert code == 0, out
    assert _retired(db, retire + keep) == set(retire)
    _assert_one_row_per_retired(_rows_after(db, before), set(retire))


def test_scheduled_prompts_dry_run_writes_no_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _use_db(monkeypatch, tmp_path / "memory.db")
    retire, keep, host = _seed_scheduled(db, tmp_path)
    before = _max_feedback_id(db)
    code, out = _run_scheduled(host, apply=False)
    assert code == 0
    assert retire[0] in out
    assert _retired(db, retire + keep) == set()
    assert _rows_after(db, before) == []


def test_each_path_writes_its_own_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = [
        _gc_source(monkeypatch, tmp_path),
        _prune_source(monkeypatch, tmp_path),
        _scheduled_source(monkeypatch, tmp_path),
    ]
    assert len(set(sources)) == 3, sources


# --- Posteriors --------------------------------------------------------


def test_retiring_leaves_alpha_and_beta_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    gc_db = _use_db(monkeypatch, tmp_path / "gc" / "memory.db")
    gc_ids = [bid for group in _seed_phantoms(gc_db) for bid in group]
    gc_before = _state(gc_db, gc_ids)
    gc_rows_from = _max_feedback_id(gc_db)
    _run_gc_hook()

    prune_db = _use_db(monkeypatch, tmp_path / "prune" / "memory.db")
    prune_ids = [bid for group in _seed_noise(prune_db) for bid in group]
    prune_before = _state(prune_db, prune_ids)
    prune_rows_from = _max_feedback_id(prune_db)
    assert _run_prune_noise(apply=True)[0] == 0

    sched_db = _use_db(monkeypatch, tmp_path / "sched" / "memory.db")
    retire, keep, host = _seed_scheduled(sched_db, tmp_path / "sched")
    sched_ids = retire + keep
    sched_before = _state(sched_db, sched_ids)
    sched_rows_from = _max_feedback_id(sched_db)
    assert _run_scheduled(host, apply=True)[0] == 0

    for db, ids, before, rows_from in (
        (gc_db, gc_ids, gc_before, gc_rows_from),
        (prune_db, prune_ids, prune_before, prune_rows_from),
        (sched_db, sched_ids, sched_before, sched_rows_from),
    ):
        after = _state(db, ids)
        # Guard: each path retired something and wrote its row, so the
        # posterior check below runs against a real retirement.
        assert _retired(db, ids), db
        assert _rows_after(db, rows_from), db
        for bid in ids:
            assert after[bid][1:] == before[bid][1:], (db, bid)


def test_exposure_only_sources_are_unchanged() -> None:
    assert EXPOSURE_ONLY_FEEDBACK_SOURCES == EXPOSURE_ONLY_ON_MAIN


# --- An injected failure leaves neither the retirement nor the row -----


_FEEDBACK_INSERT = re.compile(
    r"\bINSERT\b.*?\bINTO\s+[\"'`\[]?feedback_history\b", re.IGNORECASE | re.DOTALL,
)


def _fail_feedback_inserts(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Make every SQL insert into `feedback_history` raise.

    The patch sits on `sqlite3.connect`, under whatever helper the fix
    writes the row with, so it doesn't depend on that helper's name. The
    retirement is the first write and the row is the second. Returns a
    list that gets one entry each time the patch refuses an insert.
    """
    fired: list[str] = []

    def check(sql: object) -> None:
        if isinstance(sql, str) and _FEEDBACK_INSERT.search(sql):
            fired.append(sql)
            raise sqlite3.OperationalError("injected #2019 feedback insert failure")

    class _Cursor(sqlite3.Cursor):
        def execute(self, sql: str, *args: Any, **kwargs: Any) -> sqlite3.Cursor:  # pyright: ignore[reportIncompatibleMethodOverride]
            check(sql)
            return super().execute(sql, *args, **kwargs)

        def executemany(self, sql: str, *args: Any, **kwargs: Any) -> sqlite3.Cursor:  # pyright: ignore[reportIncompatibleMethodOverride]
            check(sql)
            return super().executemany(sql, *args, **kwargs)

        def executescript(self, sql: str, *args: Any, **kwargs: Any) -> sqlite3.Cursor:  # pyright: ignore[reportIncompatibleMethodOverride]
            check(sql)
            return super().executescript(sql, *args, **kwargs)

    class _Conn(sqlite3.Connection):
        def cursor(self, factory: Any = _Cursor) -> Any:  # pyright: ignore[reportIncompatibleMethodOverride]
            return super().cursor(factory)

        def execute(self, sql: str, *args: Any, **kwargs: Any) -> sqlite3.Cursor:  # pyright: ignore[reportIncompatibleMethodOverride]
            check(sql)
            return super().execute(sql, *args, **kwargs)

        def executemany(self, sql: str, *args: Any, **kwargs: Any) -> sqlite3.Cursor:  # pyright: ignore[reportIncompatibleMethodOverride]
            check(sql)
            return super().executemany(sql, *args, **kwargs)

        def executescript(self, sql: str, *args: Any, **kwargs: Any) -> sqlite3.Cursor:  # pyright: ignore[reportIncompatibleMethodOverride]
            check(sql)
            return super().executescript(sql, *args, **kwargs)

    def connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        kwargs.setdefault("factory", _Conn)
        return _REAL_CONNECT(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", connect)
    return fired


def _assert_nothing_landed(
    db: Path, ids: list[str], before: int, fired: list[str],
) -> None:
    assert _retired(db, ids) == set(), "the retirement landed without its row"
    assert _rows_after(db, before) == []
    assert fired, "the path never tried to write a feedback row"


def test_gc_failure_between_retirement_and_row_leaves_neither(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _use_db(monkeypatch, tmp_path / "memory.db")
    # One candidate, so a fix that commits per belief can't land an
    # earlier belief before the failure.
    with _open(db) as store:
        store.insert_belief(_phantom("p_old1", "The cache expires after ten minutes."))
    before = _max_feedback_id(db)
    fired = _fail_feedback_inserts(monkeypatch)
    # The hook swallows the failure so SessionStart still returns 0.
    _run_gc_hook()
    _assert_nothing_landed(db, ["p_old1"], before, fired)


def test_prune_noise_failure_between_retirement_and_row_leaves_neither(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _use_db(monkeypatch, tmp_path / "memory.db")
    with _open(db) as store:
        store.insert_belief(_agent_belief("n_tag", NOISE["n_tag"]))
    before = _max_feedback_id(db)
    fired = _fail_feedback_inserts(monkeypatch)
    with contextlib.suppress(sqlite3.Error):
        _run_prune_noise(apply=True)
    _assert_nothing_landed(db, ["n_tag"], before, fired)


def test_scheduled_failure_between_retirement_and_row_leaves_neither(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _use_db(monkeypatch, tmp_path / "memory.db")
    retire, _keep, host = _seed_scheduled(db, tmp_path)
    before = _max_feedback_id(db)
    fired = _fail_feedback_inserts(monkeypatch)
    with contextlib.suppress(sqlite3.Error):
        _run_scheduled(host, apply=True)
    _assert_nothing_landed(db, retire, before, fired)


# --- GC ignores the retire rows when it picks candidates ---------------


def _gc_collects(db: Path, bid: str) -> bool:
    _run_gc_hook()
    return _retired(db, [bid]) == {bid}


def test_a_gc_collected_then_restored_phantom_is_a_candidate_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _use_db(monkeypatch, tmp_path / "memory.db")
    with _open(db) as store:
        store.insert_belief(_phantom("p1", "The cache expires after ten minutes."))
    before = _max_feedback_id(db)
    assert _gc_collects(db, "p1")
    # Guard: GC wrote its row, so the second sweep below has to look past it.
    assert [bid for bid, _, _ in _rows_after(db, before)] == ["p1"]
    with _open(db) as store:
        assert store.restore_belief("p1")
    assert _gc_collects(db, "p1")


@pytest.mark.parametrize("path", ["prune_noise", "scheduled_prompts"])
def test_a_restored_phantom_with_a_planted_retire_row_is_a_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path: str,
) -> None:
    # The row is planted because neither path can retire a phantom in a
    # real run: prune-noise takes only agent_inferred beliefs, and the
    # scheduled-prompt pass takes only user_transcript ones, while GC
    # takes only origin speculative. The source comes from a real run of
    # that path on another store, so the test doesn't name it.
    if path == "prune_noise":
        source = _prune_source(monkeypatch, tmp_path / "probe")
    else:
        source = _scheduled_source(monkeypatch, tmp_path / "probe")
    db = _use_db(monkeypatch, tmp_path / "memory.db")
    with _open(db) as store:
        store.insert_belief(_phantom("p1", "The cache expires after ten minutes."))
        store.soft_delete_belief("p1", ts=_ts(3))
        store.insert_feedback_event("p1", 0.0, source, _ts(3))
        assert store.restore_belief("p1")
    assert _gc_collects(db, "p1")


# --- aelf migrate ------------------------------------------------------


def test_migrate_writes_no_retire_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = {
        _gc_source(monkeypatch, tmp_path / "probe"),
        _prune_source(monkeypatch, tmp_path / "probe"),
        _scheduled_source(monkeypatch, tmp_path / "probe"),
    }
    # A source store whose beliefs were retired before the change, so
    # they carry no retire row.
    legacy = tmp_path / "legacy" / "memory.db"
    legacy.parent.mkdir(parents=True)
    with _open(legacy) as store:
        store.insert_belief(_phantom("p_old1", "The cache expires after ten minutes."))
        store.insert_belief(_agent_belief("n_tag", NOISE["n_tag"]))
        user = _agent_belief("u1", SCHEDULED_ONLY)
        user.origin = ORIGIN_USER_TRANSCRIPT
        store.insert_belief(user)
        store.insert_belief(_agent_belief("keep_fact", KEEP_FACT))
        for bid in ("p_old1", "n_tag", "u1"):
            store.soft_delete_belief(bid, ts=_ts(5))
    assert _rows_after(legacy, 0) == []

    target = _use_db(monkeypatch, tmp_path / "target" / "memory.db")
    code, out = _cli("migrate", "--from", str(legacy), "--apply", "--all", "--yes")
    assert code == 0, out
    with _open(target) as store:
        copied = {
            bid for bid in ("p_old1", "n_tag", "u1", "keep_fact")
            if store.get_belief(bid, include_retired=True) is not None
        }
    # Guard: the beliefs came across, so the empty log below means something.
    assert copied == {"p_old1", "n_tag", "u1", "keep_fact"}
    rows = _rows_after(target, 0)
    assert [r for r in rows if r[2] in sources] == []
    assert rows == []
