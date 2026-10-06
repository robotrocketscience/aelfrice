"""The session-end core-gate continuation in the Stop hook (#1638).

On a Stop that finds this session's core candidates that no label or open
batch covers, the hook records one classifier batch and writes a Stop
continuation, `{"hookSpecificOutput": {"hookEventName": "Stop",
"additionalContext": ...}}`, to stdout. Each test names the mutation that
kills it. Every store is under `tmp_path` through `AELFRICE_DB`, and the
belief text is synthetic. The real-entry-point run is in
`tests/test_core_gate_session_end_flush_1638.py`.
"""
from __future__ import annotations

import io
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

import aelfrice.hook as hook
from aelfrice.cli import main as cli_main
from aelfrice.core_gate import CLASSIFIER_VERSION, MAX_BATCH, build_prompt
from aelfrice.hook_payload import HOOK_PAYLOAD_CHAR_LIMIT
from aelfrice.models import (
    BELIEF_FACTUAL,
    CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
    LOCK_NONE,
    LOCK_USER,
    ORIGIN_USER_TRANSCRIPT,
    Belief,
    CoreGateBatchItem,
)
from aelfrice.store import MemoryStore

pytestmark = pytest.mark.timeout(60)

SESSION = "widgetsession-now"
OTHER = "widgetsession-earlier"


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(path))
    monkeypatch.setenv("AELF_NO_UPDATE_CHECK", "1")
    monkeypatch.setenv("AELFRICE_NO_AUTO_INSTALL", "1")
    monkeypatch.delenv(hook.CORE_GATE_SESSION_END_ENV, raising=False)
    return path


def _belief(
    bid: str,
    *,
    session: str | None = SESSION,
    core: bool = True,
    lock: str = LOCK_NONE,
    content: str | None = None,
) -> Belief:
    """A belief; `core=True` gives it the (3, 1) prior, which passes the
    posterior arm (alpha + beta >= 4, mean >= 2/3)."""
    return Belief(
        id=bid,
        content=content if content is not None else f"The widgetprompt {bid} setting is stable.",
        content_hash=f"hash-{bid}",
        alpha=3.0 if core else 1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=lock,
        locked_at="2026-10-01T00:00:00Z" if lock == LOCK_USER else None,
        created_at="2026-10-01T00:00:00Z",
        last_retrieved_at=None,
        session_id=session,
        origin=ORIGIN_USER_TRANSCRIPT,
    )


def _seed(db: Path, beliefs: list[Belief]) -> None:
    store = MemoryStore(str(db))
    try:
        for b in beliefs:
            store.insert_belief(b)
    finally:
        store.close()


def _stop(
    tmp_path: Path,
    *,
    session: str = SESSION,
    env: dict[str, str] | None = None,
    extra: dict[str, object] | None = None,
) -> tuple[str, str]:
    payload: dict[str, object] = {"session_id": session, "cwd": str(tmp_path)}
    payload.update(extra or {})
    out, err = io.StringIO(), io.StringIO()
    assert hook.stop(
        stdin=io.StringIO(json.dumps(payload)), stdout=out, stderr=err,
        env=env if env is not None else {},
    ) == 0
    return out.getvalue(), err.getvalue()


def _context(out: str) -> str:
    obj = json.loads(out)
    assert obj["hookSpecificOutput"]["hookEventName"] == "Stop"
    return str(obj["hookSpecificOutput"]["additionalContext"])


def _batches(db: Path) -> list[tuple[str, str, str | None, list[str]]]:
    """(batch_id, origin, session_id, belief ids in index order) per batch,
    in creation order."""
    if not db.exists():
        return []
    conn = sqlite3.connect(str(db))
    try:
        rows = conn.execute(
            "SELECT batch_id, origin, session_id, items_json "
            "FROM core_gate_batches ORDER BY rowid"
        ).fetchall()
    finally:
        conn.close()
    out: list[tuple[str, str, str | None, list[str]]] = []
    for bid, origin, sess, items_json in rows:
        items = sorted(json.loads(items_json), key=lambda i: i["index"])
        out.append((bid, origin, sess, [i["belief_id"] for i in items]))
    return out


def _batch_ids(db: Path) -> list[list[str]]:
    return [ids for *_, ids in _batches(db)]


def _doctor_batch(
    db: Path, bid: str, *, version: str = CLASSIFIER_VERSION,
    created_at: str = "2026-10-01T00:00:00Z",
) -> None:
    store = MemoryStore(str(db))
    try:
        store.create_core_gate_batch(
            [CoreGateBatchItem(index=0, belief_id=bid, content_hash=f"hash-{bid}")],
            classifier_version=version, origin="doctor", session_id=None,
            created_at=created_at,
        )
    finally:
        store.close()


def test_n_min_is_one() -> None:
    """Pins the stated floor. Killed by: changing
    `hook.CORE_GATE_SESSION_END_N_MIN` to anything but 1."""
    assert hook.CORE_GATE_SESSION_END_N_MIN == 1


def test_char_budget_is_the_payload_limit() -> None:
    """Killed by: a budget above the host's inline limit."""
    assert hook.CORE_GATE_SESSION_END_CHAR_BUDGET == HOOK_PAYLOAD_CHAR_LIMIT
    assert hook.CORE_GATE_SESSION_END_CHAR_BUDGET <= 10_000


def test_fires_with_candidates(db: Path, tmp_path: Path) -> None:
    """Killed by: returning before the batch is created, or comparing the
    candidate count with `<=` against N_MIN (a single candidate fires)."""
    _seed(db, [_belief("w1")])
    out, _ = _stop(tmp_path)
    assert _context(out)
    assert [(o, s, ids) for _, o, s, ids in _batches(db)] == [
        ("session_end", SESSION, ["w1"]),
    ]


def test_no_candidates_no_output(db: Path, tmp_path: Path) -> None:
    """Killed by: emitting without a batch (an empty prompt raises, which
    the hook logs, so this checks stdout and stderr both stay quiet)."""
    _seed(db, [_belief("w1", core=False)])
    out, err = _stop(tmp_path)
    assert out == ""
    assert "core-gate" not in err
    assert _batches(db) == []


def test_fires_again_only_for_new_candidates(db: Path, tmp_path: Path) -> None:
    """Killed by: dropping the open-batch claim from the candidate query.
    A Stop with nothing new stays quiet; a later flush's beliefs fire a
    second batch holding only them."""
    _seed(db, [_belief("w1"), _belief("w2")])
    assert _stop(tmp_path)[0]
    quiet, _ = _stop(tmp_path)
    assert quiet == ""
    _seed(db, [_belief("w3")])
    again, _ = _stop(tmp_path)
    assert _context(again)
    assert [sorted(ids) for ids in _batch_ids(db)] == [["w1", "w2"], ["w3"]]


def test_open_doctor_batch_claims_a_belief(db: Path, tmp_path: Path) -> None:
    """Killed by: claiming only `session_end` batches. A belief already in
    an open doctor batch is not asked about again."""
    _seed(db, [_belief("w1"), _belief("w2")])
    _doctor_batch(db, "w1")
    _stop(tmp_path)
    assert _batch_ids(db)[-1] == ["w2"]


def test_doctor_batch_older_than_the_session_is_not_read(
    db: Path, tmp_path: Path,
) -> None:
    """Only doctor batches created at or after this session's earliest
    belief are read. A batch emitted before the belief existed can't hold
    it in practice; this one is built to hold it anyway, so the test can
    see that the bound is applied.

    Killed by: dropping the `julianday(cb.created_at) >= ...` bound (the
    old batch then claims the belief and nothing fires).
    """
    _seed(db, [_belief("w1")])  # created 2026-10-01T00:00:00Z
    # A whole second earlier: julianday() resolves milliseconds at best,
    # and SQLite builds differ on rounding below that.
    _doctor_batch(db, "w1", created_at="2026-09-30T23:59:59+00:00")
    out, _ = _stop(tmp_path)
    assert _context(out)
    assert _batch_ids(db)[-1] == ["w1"]


def test_doctor_batch_bound_compares_instants_not_strings(
    db: Path, tmp_path: Path,
) -> None:
    """A doctor batch stamped at the same instant as the belief, in the
    `+00:00` form `--emit` writes, is read and claims the belief.

    Killed by: comparing the timestamps as strings (`+` sorts before `Z`,
    so the batch reads as older), or `>=` changed to `>`.
    """
    _seed(db, [_belief("w1")])  # created 2026-10-01T00:00:00Z
    _doctor_batch(db, "w1", created_at="2026-10-01T00:00:00+00:00")
    out, _ = _stop(tmp_path)
    assert out == ""
    assert len(_batches(db)) == 1


def test_accepted_batch_that_owns_no_labels_claims_nothing(
    db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A session-end batch accepted, then re-run by `aelf doctor core-gate
    --rerun` while its belief was locked (so nothing was re-batched),
    owns no labels. Once the belief is unlocked it is unlabeled, and the
    accepted batch must not claim it.

    Killed by: dropping `accepted_at IS NULL` from the claim query (the
    accepted batch then claims the belief forever).
    """
    _seed(db, [_belief("w1")])
    assert _context(_stop(tmp_path)[0])
    [(batch_id, *_)] = _batches(db)
    monkeypatch.setattr("sys.stdin", io.StringIO('[{"index": 0, "label": "A"}]'))
    assert cli_main(["core-gate", "accept", batch_id], out=io.StringIO()) == 0
    _set_lock(db, "w1", LOCK_USER)
    rerun_out = io.StringIO()
    assert cli_main(
        ["doctor", "core-gate", "--json", "--rerun", batch_id], out=rerun_out,
    ) == 0
    assert json.loads(rerun_out.getvalue())["kept"] == 0
    _set_lock(db, "w1", LOCK_NONE)
    out, _ = _stop(tmp_path)
    assert _context(out)
    assert _batch_ids(db)[-1] == ["w1"]


def _set_lock(db: Path, bid: str, level: str) -> None:
    store = MemoryStore(str(db))
    try:
        b = store.get_belief(bid)
        assert b is not None
        b.lock_level = level
        b.locked_at = "2026-10-02T00:00:00Z" if level == LOCK_USER else None
        store.update_belief(b)
    finally:
        store.close()


def test_stale_version_batch_claims_nothing(db: Path, tmp_path: Path) -> None:
    """Killed by: dropping the classifier-version conjunct from the claim.
    A batch under another version can't be accepted, so its beliefs are
    still asked about."""
    _seed(db, [_belief("w1")])
    _doctor_batch(db, "w1", version="core-gate-widgetold")
    out, _ = _stop(tmp_path)
    assert _context(out)
    assert _batch_ids(db)[-1] == ["w1"]


def test_accepted_batch_excludes_through_its_labels(
    db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Killed by: dropping the label conjunct. Accepting a batch closes it
    and writes labels; the labels keep its beliefs out."""
    _seed(db, [_belief("w1")])
    _stop(tmp_path)
    [(batch_id, *_)] = _batches(db)
    monkeypatch.setattr("sys.stdin", io.StringIO('[{"index": 0, "label": "A"}]'))
    assert cli_main(["core-gate", "accept", batch_id], out=io.StringIO()) == 0
    out, _ = _stop(tmp_path)
    assert out == ""
    assert len(_batches(db)) == 1


def test_stop_hook_active_does_not_continue(
    db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Killed by: deleting the `stop_hook_active` check."""
    _seed(db, [_belief("w1")])
    out, _ = _stop(tmp_path, extra={"stop_hook_active": True})
    assert out == ""
    assert _batches(db) == []


@pytest.mark.parametrize("entrypoint", ["sdk-cli", "sdk-ts", "sdk-py"])
def test_headless_does_not_continue_or_query(
    db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entrypoint: str,
) -> None:
    """Killed by: deleting the headless check, or moving it after the
    candidate query (a headless Stop must not read the store for this)."""
    _seed(db, [_belief("w1")])
    queries: list[str] = []
    real = MemoryStore.list_core_gate_session_candidates

    def spy(self: MemoryStore, *a: str) -> list[Belief]:
        queries.append("q")
        return real(self, *a)

    monkeypatch.setattr(MemoryStore, "list_core_gate_session_candidates", spy)
    out, _ = _stop(tmp_path, env={"CLAUDE_CODE_ENTRYPOINT": entrypoint})
    assert out == ""
    assert queries == []
    assert _batches(db) == []


def test_interactive_entrypoint_fires(db: Path, tmp_path: Path) -> None:
    """Killed by: treating every set entrypoint as headless."""
    _seed(db, [_belief("w1")])
    out, _ = _stop(tmp_path, env={"CLAUDE_CODE_ENTRYPOINT": "cli"})
    assert _context(out)


def test_codex_host_does_not_continue(db: Path, tmp_path: Path) -> None:
    """Killed by: deleting the Codex check. Codex's Stop input carries
    `turn_id`, and Codex doesn't document `additionalContext` on Stop."""
    _seed(db, [_belief("w1")])
    out, _ = _stop(tmp_path, extra={"turn_id": "widgetturn-1"})
    assert out == ""
    assert _batches(db) == []


@pytest.mark.parametrize("value", ["0", "false", "no", "off"])
def test_env_opt_out_does_not_continue(
    db: Path, tmp_path: Path, value: str,
) -> None:
    """Killed by: deleting the env-var opt-out check."""
    _seed(db, [_belief("w1")])
    out, _ = _stop(tmp_path, env={hook.CORE_GATE_SESSION_END_ENV: value})
    assert out == ""
    assert _batches(db) == []


def test_unrecognised_env_value_falls_through(db: Path, tmp_path: Path) -> None:
    """Killed by: reading any value other than an on-value as off."""
    _seed(db, [_belief("w1")])
    out, _ = _stop(tmp_path, env={hook.CORE_GATE_SESSION_END_ENV: "widgetjunk"})
    assert _context(out)


def test_toml_opt_out_does_not_continue(db: Path, tmp_path: Path) -> None:
    """Killed by: deleting the `[core_gate] session_end` check, or reading
    a key other than `session_end`."""
    (tmp_path / ".aelfrice.toml").write_text(
        "[core_gate]\nsession_end = false\n", encoding="utf-8",
    )
    _seed(db, [_belief("w1")])
    out, _ = _stop(tmp_path)
    assert out == ""
    assert _batches(db) == []


def test_env_on_overrides_toml_opt_out(db: Path, tmp_path: Path) -> None:
    """Killed by: reading the TOML key even when the env var says on."""
    (tmp_path / ".aelfrice.toml").write_text(
        "[core_gate]\nsession_end = false\n", encoding="utf-8",
    )
    _seed(db, [_belief("w1")])
    out, _ = _stop(tmp_path, env={hook.CORE_GATE_SESSION_END_ENV: "1"})
    assert _context(out)


def test_batch_holds_only_this_sessions_candidates(
    db: Path, tmp_path: Path,
) -> None:
    """Killed by: listing candidates without the session conjunct, or
    dropping the core-rule filter. An older session's core belief, a
    belief with no session, a retired one, a locked one, and this
    session's non-core one are all left out."""
    retired = _belief("w-retired")
    retired.valid_to = "2026-10-02T00:00:00Z"
    _seed(db, [
        _belief("w-old", session=OTHER),
        _belief("w-none", session=None),
        _belief("w-weak", core=False),
        _belief("w-locked", lock=LOCK_USER),
        retired,
        _belief("w-mine"),
    ])
    out, _ = _stop(tmp_path)
    assert _context(out)
    assert _batch_ids(db) == [["w-mine"]]


def test_labels_decide_by_version(db: Path, tmp_path: Path) -> None:
    """Killed by: dropping the label conjunct, or matching labels under
    any version. A label under another classifier version doesn't count."""
    _seed(db, [_belief("w-labeled"), _belief("w-stale"), _belief("w-fresh")])
    store = MemoryStore(str(db))
    try:
        store.put_core_gate_labels(
            {"hash-w-labeled": "A"}, classifier_version=CLASSIFIER_VERSION,
            batch_id=None, labeled_at="2026-10-01T00:00:00Z",
        )
        store.put_core_gate_labels(
            {"hash-w-stale": "C"}, classifier_version="core-gate-widgetold",
            batch_id=None, labeled_at="2026-10-01T00:00:00Z",
        )
    finally:
        store.close()
    _stop(tmp_path)
    assert [sorted(ids) for ids in _batch_ids(db)] == [["w-fresh", "w-stale"]]


def _corroborate(db: Path, bid: str, stamps: tuple[str, ...]) -> None:
    store = MemoryStore(str(db))
    try:
        for n, ts in enumerate(stamps):
            store.record_corroboration(
                bid,
                source_type=CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
                session_id=f"s{n}",
                ts=ts,
            )
    finally:
        store.close()


def test_corroboration_arm_candidate_is_included(
    db: Path, tmp_path: Path,
) -> None:
    """Killed by: dropping the `corroboration_episodes()` fallback, so a
    belief that qualifies only on the corroboration arm is missed. A
    belief with two rows in one episode stays out."""
    _seed(db, [_belief("w-corr", core=False), _belief("w-burst", core=False)])
    _corroborate(db, "w-corr", ("2026-10-01T00:00:00+00:00", "2026-10-01T05:00:00+00:00"))
    _corroborate(db, "w-burst", ("2026-10-01T00:00:00+00:00", "2026-10-01T00:10:00+00:00"))
    _stop(tmp_path)
    assert _batch_ids(db) == [["w-corr"]]


def test_episodes_are_not_read_without_a_corroboration_candidate(
    db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Killed by: loading `corroboration_episodes()` eagerly. It reads
    every corroboration row in the store, so a Stop whose beliefs can't
    pass on that arm must not read it."""
    _seed(db, [_belief("w1"), _belief("w2", core=False)])
    calls: list[str] = []
    real = MemoryStore.corroboration_episodes

    def spy(self: MemoryStore) -> dict[str, int]:
        calls.append("episodes")
        return real(self)

    monkeypatch.setattr(MemoryStore, "corroboration_episodes", spy)
    out, _ = _stop(tmp_path)
    assert _context(out)
    assert calls == []


def test_batch_is_capped_at_max_batch(db: Path, tmp_path: Path) -> None:
    """The newest MAX_BATCH beliefs are the batch, whether the whole-set
    try takes them or, past a belief too long to fit, the walk does.

    Killed by: dropping the `MAX_BATCH` slice of the whole-set try or the
    `MAX_BATCH` stop in the walk (the store refuses an oversized batch,
    so nothing would fire), or omitting the left-over count.
    """
    extra = 3
    ids = [f"w{i:03d}" for i in range(MAX_BATCH + extra)]
    _seed(db, [_belief(i, content=f"Widget {i} holds.") for i in ids])
    out, _ = _stop(tmp_path)
    context = _context(out)
    assert _batch_ids(db) == [list(reversed(ids))[:MAX_BATCH]]
    assert f"{extra} more from this session" in context


@pytest.mark.parametrize("over", [0, 1], ids=["exactly-the-budget", "one-over"])
def test_walk_budget_boundary_is_exact(
    db: Path, tmp_path: Path, over: int,
) -> None:
    """Past a belief too long for any batch, the walk takes a trial whose
    context is exactly the budget, and not one character more.

    Killed by: any slack in the walk's budget check (`<=` to `<`, or the
    budget plus a margin).
    """
    small = "Widget small holds."
    filler = "Widget big " + "z" * 1000
    base = _context_length([filler, small], 1)
    big = filler + "z" * (hook.CORE_GATE_SESSION_END_CHAR_BUDGET - base + over)
    assert _context_length([big, small], 1) == hook.CORE_GATE_SESSION_END_CHAR_BUDGET + over
    huge = _belief("w-huge", content="Widget " + "y" * hook.CORE_GATE_SESSION_END_CHAR_BUDGET)
    _seed(db, [
        _belief("w-small", content=small), _belief("w-big", content=big), huge,
    ])
    out, _ = _stop(tmp_path)
    assert len(_context(out)) <= hook.CORE_GATE_SESSION_END_CHAR_BUDGET
    expected = [["w-big", "w-small"]] if over == 0 else [["w-big"]]
    assert _batch_ids(db) == expected


def test_walk_is_capped_at_max_batch(db: Path, tmp_path: Path) -> None:
    """With the newest belief too long for any batch, the walk skips it
    and takes the next MAX_BATCH, not all of them.

    Killed by: dropping the `MAX_BATCH` stop in the walk of
    `_fit_core_gate_batch`.
    """
    ids = [f"w{i:03d}" for i in range(MAX_BATCH + 2)]
    huge = _belief("w-huge", content="Widget " + "y" * hook.CORE_GATE_SESSION_END_CHAR_BUDGET)
    _seed(db, [_belief(i, content=f"Widget {i} holds.") for i in ids] + [huge])
    out, _ = _stop(tmp_path)
    assert _context(out)
    assert _batch_ids(db) == [list(reversed(ids))[:MAX_BATCH]]


def test_batch_fits_the_char_budget(db: Path, tmp_path: Path) -> None:
    """Killed by: dropping the char-budget check, or truncating snippets
    instead of leaving them out. Each snippet is whole, the context fits,
    and the rest are counted and left unclaimed for the next Stop."""
    ids = [f"w{i}" for i in range(8)]
    _seed(db, [_belief(i, content=f"Widget {i} " + "x" * 1500 + ".") for i in ids])
    out, _ = _stop(tmp_path)
    context = _context(out)
    assert len(context) <= hook.CORE_GATE_SESSION_END_CHAR_BUDGET
    [first] = _batch_ids(db)
    assert 0 < len(first) < len(ids)
    store = MemoryStore(str(db))
    try:
        for bid in first:
            assert store.get_belief(bid).content in context  # type: ignore[union-attr]
    finally:
        store.close()
    assert f"{len(ids) - len(first)} more from this session" in context
    again, _ = _stop(tmp_path)
    assert _context(again)
    assert set(_batch_ids(db)[1]).isdisjoint(first)


def _context_length(texts: list[str], left: int) -> int:
    """Length of the context `_fit_core_gate_batch` would render for
    `texts` in index order; the batch id is a fixed 16 characters."""
    return len(hook._format_core_gate_context(
        "0" * 16, build_prompt(list(enumerate(texts))), len(texts), left,
    ))


@pytest.mark.parametrize("over", [0, 1], ids=["exactly-the-budget", "one-over"])
def test_char_budget_boundary_is_exact(
    db: Path, tmp_path: Path, over: int,
) -> None:
    """A two-belief context of exactly the budget is batched whole. One
    character more doesn't fit, and the newer, larger belief doesn't fit
    alone beside the line counting the one left out, so the smaller one
    goes first and the larger one gets the next Stop to itself.

    Killed by: any slack in the budget check (`<=` to `<`, or the budget
    plus a margin), or dropping the whole-set try in `_fit_core_gate_batch`
    (the walk alone skips the larger belief even when both fit).
    """
    small = "Widget small holds."
    filler = "Widget big " + "z" * 1000
    base = _context_length([filler, small], 0)
    big = filler + "z" * (hook.CORE_GATE_SESSION_END_CHAR_BUDGET - base + over)
    assert _context_length([big, small], 0) == hook.CORE_GATE_SESSION_END_CHAR_BUDGET + over
    _seed(db, [_belief("w-small", content=small), _belief("w-big", content=big)])
    out, _ = _stop(tmp_path)
    context = _context(out)
    assert len(context) <= hook.CORE_GATE_SESSION_END_CHAR_BUDGET
    if over == 0:
        assert _batch_ids(db) == [["w-big", "w-small"]]
        return
    assert _batch_ids(db) == [["w-small"]]
    again, _ = _stop(tmp_path)
    assert len(_context(again)) <= hook.CORE_GATE_SESSION_END_CHAR_BUDGET
    assert _batch_ids(db) == [["w-small"], ["w-big"]]


def test_an_oversized_belief_is_skipped_not_blocking(
    db: Path, tmp_path: Path,
) -> None:
    """Killed by: stopping at the first candidate that doesn't fit. The
    newest belief is too long for any batch; the older one still goes."""
    _seed(db, [
        _belief("w-small"),
        _belief("w-huge", content="Widget " + "y" * hook.CORE_GATE_SESSION_END_CHAR_BUDGET),
    ])
    out, _ = _stop(tmp_path)
    assert _context(out)
    assert _batch_ids(db) == [["w-small"]]


def test_nothing_fits_means_no_batch(db: Path, tmp_path: Path) -> None:
    """Killed by: trying to record a batch when no candidate fits (the
    store refuses an empty batch, and the hook would log that failure on
    every Stop)."""
    _seed(db, [_belief("w-huge", content="Widget " + "y" * hook.CORE_GATE_SESSION_END_CHAR_BUDGET)])
    out, err = _stop(tmp_path)
    assert out == ""
    assert "core-gate" not in err
    assert _batches(db) == []


def test_continuation_shape_and_accept_round_trip(
    db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Killed by: emitting `decision: "block"`, adding keys, building the
    prompt out of index order, or naming another batch id in the accept
    command. The emitted batch then accepts a synthetic reply and the
    labels land under the beliefs' content hashes."""
    _seed(db, [_belief("w-a"), _belief("w-b")])
    out, _ = _stop(tmp_path)
    assert out.endswith("\n") and out.count("\n") == 1
    obj = json.loads(out)
    assert obj == {
        "hookSpecificOutput": {
            "hookEventName": "Stop",
            "additionalContext": obj["hookSpecificOutput"]["additionalContext"],
        },
    }
    context = _context(out)
    [(batch_id, _, _, ids)] = _batches(db)
    store = MemoryStore(str(db))
    try:
        texts = [store.get_belief(i).content for i in ids]  # type: ignore[union-attr]
    finally:
        store.close()
    assert build_prompt(list(enumerate(texts))).rstrip("\n") in context
    assert f"aelf core-gate accept {batch_id} <<'AELFRICE_LABELS'" in context
    assert context.index("aelf core-gate accept") < context.index(
        "--- classifier prompt ---"
    )

    reply = json.dumps([{"index": 0, "label": "A"}, {"index": 1, "label": "C"}])
    monkeypatch.setattr("sys.stdin", io.StringIO(reply))
    assert cli_main(["core-gate", "accept", batch_id], out=io.StringIO()) == 0
    store = MemoryStore(str(db))
    try:
        labels = store.core_gate_labels_for(
            [f"hash-{i}" for i in ids], CLASSIFIER_VERSION,
        )
    finally:
        store.close()
    assert labels == {f"hash-{ids[0]}": "A", f"hash-{ids[1]}": "C"}


def test_failure_after_batch_insert_leaves_no_batch(
    db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Killed by: creating the batch outside the transaction. A failure
    after the insert rolls it back, so no batch is left that no prompt
    named, and the next Stop asks again."""
    _seed(db, [_belief("w1")])
    real = MemoryStore.create_core_gate_batch

    def create_then_fail(self: MemoryStore, *a: object, **k: object) -> str:
        real(self, *a, **k)  # type: ignore[arg-type]
        raise RuntimeError("widgetfailure")

    monkeypatch.setattr(MemoryStore, "create_core_gate_batch", create_then_fail)
    out, err = _stop(tmp_path)
    assert out == ""
    assert "widgetfailure" in err
    assert _batches(db) == []
    monkeypatch.setattr(MemoryStore, "create_core_gate_batch", real)
    assert _context(_stop(tmp_path)[0])


def test_concurrent_stop_does_not_double_batch(
    db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Killed by: batching the pre-check's candidates instead of
    collecting again under the write lock. A second Stop that batches
    first, between this Stop's pre-check and its transaction, leaves this
    Stop nothing to batch."""
    _seed(db, [_belief("w1")])
    real_txn = MemoryStore.transaction
    raced: list[str] = []

    @contextmanager
    def racing_txn(self: MemoryStore, *, immediate: bool = False) -> Iterator[None]:
        if immediate and not raced:
            raced.append("other")
            other_out, _ = _stop(tmp_path)
            assert _context(other_out)
        with real_txn(self, immediate=immediate):
            yield

    monkeypatch.setattr(MemoryStore, "transaction", racing_txn)
    out, _ = _stop(tmp_path)
    assert raced == ["other"]
    assert out == ""
    assert _batch_ids(db) == [["w1"]]


def test_exceptions_are_swallowed(
    db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Killed by: removing the try/except around the session-end call in
    `hook.stop()`. The failure is logged to stderr, nothing reaches stdout, and
    the hook still returns 0."""
    _seed(db, [_belief("w1")])

    def boom(*_a: object, **_k: object) -> list[Belief]:
        raise RuntimeError("widgetfailure")

    monkeypatch.setattr(hook, "_collect_core_gate_session_candidates", boom)
    out, err = _stop(tmp_path)
    assert out == ""
    assert "core-gate session-end batch failed" in err
    assert "widgetfailure" in err


def test_non_firing_path_is_one_query_on_the_shared_handle(
    db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Killed by: opening a store of its own, reading the config before
    the candidate query, or taking the write lock with no candidates.
    Once a flush's candidates are batched, a Stop opens only the store
    the lock listing always opened, and the session-end path runs one
    candidate query on it and nothing else."""
    _seed(db, [_belief("w1")])
    assert _stop(tmp_path)[0]
    opens: list[str] = []
    calls: list[str] = []
    real_init = MemoryStore.__init__
    real_query = MemoryStore.list_core_gate_session_candidates
    real_txn = MemoryStore.transaction

    def counting_init(self: MemoryStore, *a: object, **k: object) -> None:
        opens.append("open")
        real_init(self, *a, **k)  # type: ignore[arg-type]

    def counting_query(self: MemoryStore, *a: str) -> list[Belief]:
        calls.append("query")
        return real_query(self, *a)

    @contextmanager
    def counting_txn(self: MemoryStore, *, immediate: bool = False) -> Iterator[None]:
        calls.append("txn")
        with real_txn(self, immediate=immediate):
            yield

    def counting_toml(*_a: object, **_k: object) -> dict[str, object]:
        calls.append("toml")
        return {}

    monkeypatch.setattr(MemoryStore, "__init__", counting_init)
    monkeypatch.setattr(MemoryStore, "list_core_gate_session_candidates", counting_query)
    monkeypatch.setattr(MemoryStore, "transaction", counting_txn)
    monkeypatch.setattr(hook, "_load_aelfrice_toml", counting_toml)
    out, _ = _stop(tmp_path)
    assert out == ""
    assert opens == ["open"]
    assert calls == ["query"]
