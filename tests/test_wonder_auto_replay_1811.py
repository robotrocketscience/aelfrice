"""The #1811 replay harness: prepare, replay, run, and score over store copies.

Every store, transcript, reply, and label is synthetic and lives under
`tmp_path`. No test runs a model: replies are written by hand in the shape
the host CLI prints, and the runner `replay` starts is a fake on disk. Each
test names the mutation that kills it.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import signal
import sqlite3
import stat
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

import aelfrice.wonder.auto_run as wb
from aelfrice import feed_log, file_lock
from aelfrice.models import (
    BELIEF_FACTUAL,
    CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
    INGEST_SOURCE_TRANSCRIPT,
    CORROBORATION_SPEAKER_USER,
    LOCK_NONE,
    ORIGIN_USER_TRANSCRIPT,
    Belief,
)
from aelfrice.store import MemoryStore
from aelfrice.transcript_logger import DEFAULT_STOP_FLUSH_TURNS
from benchmarks import real_store_ab_1795 as ab
from benchmarks import wonder_auto_replay as rp
from benchmarks.graded_gate import MIN_GRADED

pytestmark = pytest.mark.timeout(120)

MARK = "zorblaxquill"
"""In every belief, prompt, claim, and document: stdout must never show it."""
OLD, SHORT, BARE, NEW = "sess-old", "sess-short", "sess-bare", "sess-new"
SNAP = "s0"
BASE, PHANTOM = rp.BASE_ARM, rp.PHANTOM_ARM


@pytest.fixture(autouse=True)
def pinned_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A process cwd and a home folder of the test's own, which holds the spend ledger."""
    work = tmp_path / "proc-cwd"
    work.mkdir()
    monkeypatch.chdir(work)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))


def _belief(bid: str, content: str, session: str | None) -> Belief:
    return Belief(
        id=bid, content=content, content_hash=hashlib.sha256(content.encode()).hexdigest(),
        alpha=3.0, beta=1.0, type=BELIEF_FACTUAL, lock_level=LOCK_NONE, locked_at=None,
        created_at="2026-10-01T00:00:00Z", last_retrieved_at=None, session_id=session,
        origin=ORIGIN_USER_TRANSCRIPT,
    )


def _turn(role: str, text: str, sid: str, n: int) -> dict[str, Any]:
    return {"schema_version": 1, "ts": f"2026-10-{1 + n // 20:02d}T00:00:{n % 20:02d}Z",
            "role": role, "text": text, "session_id": sid, "turn_id": f"t{n:03d}"}


def _session(sid: str, prompts: Sequence[str], start: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, text in enumerate(prompts):
        rows.append(_turn("user", text, sid, start + 2 * i))
        rows.append(_turn("assistant", "ok, noted", sid, start + 2 * i + 1))
    return rows


EXCHANGES = rp.HORIZON_TURNS // 2


def _prompts(word: str) -> list[str]:
    return [f"how does the {MARK} {word} handle step {i}" for i in range(EXCHANGES)]


def _store(tmp_path: Path, name: str = "proj") -> Path:
    """A store with two sessions the hook would run for, one short, one with no belief.

    The newest session starts on 2026-10-10 and ends on 2026-10-11.
    """
    path = tmp_path / name / "aelfrice" / "memory.db"
    path.parent.mkdir(parents=True)
    s = MemoryStore(str(path))
    try:
        for i, (word, sid) in enumerate((("cache", OLD), ("manifest", NEW), ("plugin", NEW))):
            s.insert_belief(_belief(f"w{i}a", f"The {MARK} {word} flushes every ten minutes.", sid))
            s.insert_belief(_belief(f"w{i}b", f"The {MARK} {word} stores parsed data on disk.", sid))
        for i in range(30):
            s.insert_belief(_belief(f"f{i:02d}", " ".join(f"qv{i}{c}" for c in "abcde"), None))
        for sid in (OLD, NEW):
            s.record_ingest(source_kind=INGEST_SOURCE_TRANSCRIPT, raw_text="widget", session_id=sid)
    finally:
        s.close()
    rows = [
        *_session(OLD, _prompts("cache"), 0),
        *_session(SHORT, [f"what about the {MARK} cache"], 100),
        *_session(BARE, _prompts("bare"), 120),
        *_session(NEW, [*_prompts("manifest"), f"and the {MARK} plugin zeta after the horizon"], 190),
    ]
    tdir = path.parent / "transcripts"
    tdir.mkdir()
    (tdir / "turns.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    _mark_copy(path)
    return path


def _mark_copy(path: Path) -> None:
    """What `prepare` leaves beside a copy it finished."""
    (path.parent / rp.COPY_MARK_FILE).write_text("", encoding="utf-8")


def _main(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str]:
    code = rp.main(list(argv))
    out, err = capsys.readouterr()
    assert MARK not in err
    return code, out if code == 0 else out + err


def _requests(tmp_path: Path, capsys: pytest.CaptureFixture[str], store: Path,
              *extra: str) -> list[dict[str, Any]]:
    out = tmp_path / "scratch" / "requests.jsonl"
    code, text = _main(capsys, "requests", "--store", str(store), "--out", str(out), *extra)
    assert code == 0, text
    assert MARK not in text
    return [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]


def _claims_for(rid: str) -> list[str]:
    return [f"The {MARK} cache and its manifest change together in run {rid[:4]}.",
            f"A flush of the {MARK} cache forces a manifest read in run {rid[:4]}."]


def _reply(row: Mapping[str, Any], *, cost: float = 0.03, body: object = None,
           code: int = 0) -> dict[str, Any]:
    ids = list(row["anchor_ids"])
    rid = str(row["request_id"])
    if body is None:
        body = {"axes": [{
            "axis": row["axes"][0],
            "document": f"The {MARK} cache and manifest move as one unit in run {rid[:4]}.",
            "claims": [{"text": t, "anchors": [ids[i % len(ids)]]}
                       for i, t in enumerate(_claims_for(rid))],
        }]}
    result = body if isinstance(body, str) else json.dumps(body)
    return {"request_id": rid, "returncode": code,
            "stdout": json.dumps({"type": "result", "result": result, "total_cost_usd": cost})}


def _write_replies(path: Path, replies: Sequence[Mapping[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in replies), encoding="utf-8")
    return path


def _flow(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> tuple[Path, Path, str]:
    store = _store(tmp_path)
    rows = _requests(tmp_path, capsys, store)
    reqs = [r for r in rows if r["kind"] == "request"]
    replies = _write_replies(tmp_path / "scratch" / "replies.jsonl", [_reply(r) for r in reqs])
    out = tmp_path / "scratch" / "run"
    code, text = _main(capsys, "run", "--requests", str(tmp_path / "scratch" / "requests.jsonl"),
                       "--replies", str(replies), "--store", str(store), "--out", str(out))
    assert code == 0, text
    return store, out, text


# --- requests -----------------------------------------------------------------


def test_requests_pick_the_sessions_the_hook_would_run_for(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Killed by: dropping the length or belief guard, counting a guarded
    session against `--sessions`, ordering requests newest first, or
    counting a session on the day it ended."""
    rows = _requests(tmp_path, capsys, _store(tmp_path))
    store_row, *reqs = rows
    assert store_row["kind"] == "store"
    assert store_row["outcomes"] == {rp.S_SHORT: 1, rp.S_NO_BELIEFS: 1, rp.S_REQUEST: 2}
    assert store_row["considered"] == 2
    assert [r["session_id"] for r in reqs] == [OLD, NEW]
    assert store_row["hook_sessions_by_day"] == {"2026-10-01": 1, "2026-10-10": 1}
    capped = _requests(tmp_path / "again", capsys, _store(tmp_path / "again"), "--sessions", "1")
    assert [r["session_id"] for r in capped[1:]] == [NEW]
    assert capped[0]["hook_sessions_by_day"] == {"2026-10-01": 1, "2026-10-10": 1}


def test_a_request_is_the_writers_own_prompt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Killed by: building the prompt any way but `session_prompts` and
    `pick_topic`, or reading past the session's first `HORIZON_TURNS` lines."""
    store = _store(tmp_path)
    reqs = _requests(tmp_path, capsys, store)[1:]
    new = next(r for r in reqs if r["session_id"] == NEW)
    log = tmp_path / "seen.jsonl"
    lines = (store.parent / "transcripts" / "turns.jsonl").read_text(encoding="utf-8").splitlines()
    seen = [ln for ln in lines if json.loads(ln)["session_id"] == NEW][:rp.HORIZON_TURNS]
    log.write_text("\n".join(seen) + "\n", encoding="utf-8")
    s = MemoryStore(str(store))
    try:
        prompts = wb.session_prompts(log, NEW)
        picked = wb.pick_topic(s, prompts)
    finally:
        s.close()
    assert picked is not None
    _payload, prompt = picked
    assert new["input"] == prompt.text
    assert new["anchor_ids"] == list(prompt.anchor_ids)
    assert new["axes"] == list(prompt.axes)
    assert new["prompts"] == len(prompts) == EXCHANGES
    assert "zeta" not in new["input"]
    assert rp.HORIZON_TURNS == DEFAULT_STOP_FLUSH_TURNS + 2


def test_horizon_zero_reads_the_whole_session(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Killed by: ignoring `--horizon 0`, or applying the length guard to it."""
    rows = _requests(tmp_path, capsys, _store(tmp_path), "--horizon", "0")
    new = next(r for r in rows[1:] if r["session_id"] == NEW)
    assert new["prompts"] == EXCHANGES + 1
    assert rows[0]["outcomes"].get(rp.S_SHORT) is None


def test_requests_leave_the_store_unchanged(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Killed by: building prompts on the store instead of a copy."""
    store = _store(tmp_path)
    before = sorted(p.name for p in store.parent.iterdir())
    digest = hashlib.sha256(store.read_bytes()).hexdigest()
    _requests(tmp_path, capsys, store)
    assert hashlib.sha256(store.read_bytes()).hexdigest() == digest
    assert sorted(p.name for p in store.parent.iterdir()) == before


# --- run ----------------------------------------------------------------------


def _report(out: Path) -> dict[str, Any]:
    return json.loads((out / rp.REPORT_FILE).read_text(encoding="utf-8"))


def _key(out: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in (out / rp.KEY_FILE).read_text(encoding="utf-8").splitlines()]


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()]


def _scores(path: Path, ids: Sequence[str], score: int = 2) -> Path:
    path.write_text("".join(json.dumps({"id": i, "score": score}) + "\n" for i in ids),
                    encoding="utf-8")
    return path


def _release(tmp_path: Path, capsys: pytest.CaptureFixture[str], out: Path) -> tuple[Path, Path]:
    """Both graders' labels of every retrieval row, and the claims packet released on them."""
    ids = [r["id"] for r in _rows(out / rp.RETRIEVAL_DIR / rp.PACKET_FILE)]
    a, c = (_scores(tmp_path / f"retrieval-{g}.jsonl", ids) for g in "ac")
    code, text = _main(capsys, "claims", "--run", str(out), "--labels", str(a), str(c))
    assert code == 0, text
    return a, c


def test_run_ingests_each_reply_through_the_writer(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Killed by: skipping `apply_reply`, writing to the base copy, or
    losing the CLI's reported cost."""
    store, out, _ = _flow(tmp_path, capsys)
    (rep,) = _report(out)["stores"]
    assert rep["requests"] == 2
    assert rep["run_outcomes"] == {wb.OUTCOME_ACCEPTED: 2}
    assert rep["runs_with_claim"] == 2
    assert (rep["claims_written"], rep["documents_written"]) == (4, 2)
    assert rep["costs"] == [0.03, 0.03]
    claims = _rows(out / rp.CLAIMS_HELD_FILE)
    rows = (tmp_path / "scratch" / "requests.jsonl").read_text(encoding="utf-8").splitlines()
    assert sorted(c["claim"] for c in claims) == sorted(
        t for r in map(json.loads, rows[1:]) for t in _claims_for(r["request_id"]))
    assert all(c["notes"] and all(MARK in n for n in c["notes"]) for c in claims)
    kinds = [r["kind"] for r in _key(out)]
    assert kinds.count(rp.KEY_CLAIM) == 4
    assert kinds.count(ab.KEY_STORE) == 1
    with sqlite3.connect(ab.read_only_uri(str(store)), uri=True) as conn:
        assert conn.execute("SELECT COUNT(*) FROM beliefs WHERE type = 'speculative'").fetchone() == (0,)


def test_a_refused_reply_is_counted_and_costs_its_cap(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Killed by: not starting a record at the run's cap, or writing a refused reply."""
    store = _store(tmp_path)
    reqs = _requests(tmp_path, capsys, store)[1:]
    bad = {"request_id": reqs[0]["request_id"], "returncode": 0,
           "stdout": json.dumps({"type": "result", "result": "not json"})}
    replies = _write_replies(tmp_path / "scratch" / "replies.jsonl", [bad, _reply(reqs[1], cost=0.04)])
    out = tmp_path / "scratch" / "run"
    code, text = _main(capsys, "run", "--requests", str(tmp_path / "scratch" / "requests.jsonl"),
                       "--replies", str(replies), "--store", str(store), "--out", str(out))
    assert code == 0, text
    (rep,) = _report(out)["stores"]
    assert rep["run_outcomes"] == {wb.OUTCOME_ACCEPTED: 1, wb.OUTCOME_BAD_REPLY: 1}
    assert rep["costs"] == [float(wb.RUNNER_BUDGET_USD), 0.04]
    assert rep["claims_written"] == 2
    assert rep["runs_with_claim"] == 1


def test_a_run_that_wrote_only_its_document_wrote_no_claim(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Killed by: counting a run as supply when only its document was new,
    since `inserted` counts the document too."""
    store = _store(tmp_path)
    reqs = _requests(tmp_path, capsys, store)[1:]
    first = _reply(reqs[0])
    repeat = {"axes": [{
        "axis": reqs[1]["axes"][0],
        "document": f"The {MARK} manifest is a new document in the second run.",
        "claims": [{"text": t, "anchors": [reqs[1]["anchor_ids"][0]]}
                   for t in _claims_for(str(reqs[0]["request_id"]))],
    }]}
    replies = _write_replies(tmp_path / "scratch" / "replies.jsonl",
                             [first, _reply(reqs[1], body=repeat)])
    out = tmp_path / "scratch" / "run"
    code, text = _main(capsys, "run", "--requests", str(tmp_path / "scratch" / "requests.jsonl"),
                       "--replies", str(replies), "--store", str(store), "--out", str(out))
    assert code == 0, text
    (rep,) = _report(out)["stores"]
    assert rep["run_outcomes"] == {wb.OUTCOME_ACCEPTED: 2}
    assert (rep["claims_written"], rep["documents_written"]) == (2, 2)
    assert rep["runs_with_claim"] == 1


def test_the_retrieval_packet_judges_what_one_copy_packed_and_the_other_did_not(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Killed by: building the packet without the phantom-against-base pair,
    or retrieving the phantom arm from the base copy."""
    _store_path, out, _ = _flow(tmp_path, capsys)
    items = [r for r in _key(out) if r["kind"] == ab.KEY_ITEM]
    phantom_only = [r for r in items if r["arms"] is not None and set(r["arms"]) == {PHANTOM}]
    assert phantom_only
    assert _report(out)["stores"][0]["phantom_hits"] >= len(phantom_only)
    instructions = (out / rp.RETRIEVAL_DIR / rp.INSTRUCTIONS_FILE).read_text(encoding="utf-8")
    assert '"score"' in instructions and "0 to 3" in instructions


def _body(row: Mapping[str, Any], claims: Sequence[str], document: str) -> dict[str, Any]:
    ids = list(row["anchor_ids"])
    return {"axes": [{"axis": row["axes"][0], "document": document, "claims": [
        {"text": t, "anchors": [ids[i % len(ids)]]} for i, t in enumerate(claims)]}]}


def _twin(store: Path, text: str, tag: str) -> None:
    """A typed twin of `text`, restated in two more sessions, all dated after any phantom.

    Three supports from three sessions: enough for evidence promotion, so
    wonder GC keeps a phantom with this text.
    """
    s = MemoryStore(str(store))
    try:
        twin = _belief(f"twin-{tag}", text, "sess-twin-0")
        s.insert_belief(dataclasses.replace(twin, created_at="2099-01-01T00:00:00Z"))
        for n in (1, 2):
            s.record_corroboration(twin.id, source_type=CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
                                   session_id=f"sess-twin-{n}", ts=f"2099-01-0{n + 1}T00:00:00Z",
                                   speaker=CORROBORATION_SPEAKER_USER)
    finally:
        s.close()


def test_run_reads_its_own_phantoms_against_promotion_gc_narration_and_usage(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Killed by: counting the base copy's phantoms as written, promotion of
    a phantom this replay didn't write, counting GC's kept phantoms as
    collected, reading narration from no ingest row, or a usage report
    with no phantom hits."""
    store = _store(tmp_path)
    with sqlite3.connect(str(store)) as conn:
        conn.execute("INSERT INTO ingest_log (id, ts, source_kind, raw_text, raw_meta, "
                     "derived_belief_ids) VALUES ('n1', '2026-10-01', 'transcript', 'x', ?, ?)",
                     (json.dumps({"role": "assistant"}), json.dumps(["w0a", "w0b", "w1a", "w1b"])))
    old = _requests(tmp_path / "prior", capsys, store)[1]
    prior = f"The {MARK} cache was rebuilt by an earlier pass of the writer."
    reply = _reply(old, body=_body(old, [prior, f"The {MARK} cache predates this replay run."],
                                   f"The {MARK} cache document from the earlier pass."))
    with rp.pinned_store_env(store):
        s = MemoryStore(str(store))
        try:
            wb.apply_reply(s, "sess-prior", rp.to_prompt(old), subprocess.CompletedProcess(
                [], 0, reply["stdout"], ""), {})
        finally:
            s.close()
    backed = f"The {MARK} manifest is read again after every cache flush."
    _twin(store, prior, "prior")
    _twin(store, backed, "new")
    reqs = _requests(tmp_path, capsys, store)[1:]
    texts = [[backed, f"The {MARK} manifest lists each cache entry by name."],
             [f"The {MARK} plugin loads after the manifest is parsed.",
              f"The {MARK} plugin holds a handle on the cache."]]
    replies = _write_replies(tmp_path / "scratch" / "replies.jsonl", [
        _reply(r, body=_body(r, t, f"The {MARK} document of run {n}.")) for n, (r, t)
        in enumerate(zip(reqs, texts, strict=True))])
    out = tmp_path / "scratch" / "run"
    code, text = _run_cli(capsys, tmp_path / "scratch" / "requests.jsonl", replies, out, store)
    assert code == 0, text
    (rep,) = _report(out)["stores"]
    assert (rep["claims_written"], rep["documents_written"]) == (4, 2)
    claims = {r["claim"] for r in _rows(out / rp.CLAIMS_HELD_FILE)}
    assert claims == {t for pair in texts for t in pair}
    assert rep["promotable"] == 1
    assert rep["gc_collected_at_ttl"] == 6 - 1
    assert all(rep["narration"][arm]["narration"] > 0 for arm in (BASE, PHANTOM))
    assert rep["usage"]["phantom_hits"] > 0 and rep["usage"]["other_hits"] > 0
    assert rep["phantom_hits"] >= rep["usage"]["phantom_hits"]


def test_h3_samples_no_prompt_the_pass_read(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Killed by: sampling from every prompt, which grades a phantom on the
    prompts it was written to answer, or holding out a whole session."""
    _store_path, out, _ = _flow(tmp_path, capsys)
    (rep,) = _report(out)["stores"]
    read = {*_prompts("cache"), *_prompts("manifest")}
    assert rep["prompts_held_out"] == len(read) == 2 * EXCHANGES
    sampled = {r["prompt"] for r in _key(out) if r["kind"] == ab.KEY_ITEM}
    assert not sampled & read
    assert f"and the {MARK} plugin zeta after the horizon" in sampled
    assert rep["prompts_sampled"] == len(sampled) == EXCHANGES + 2


def test_the_pass_reads_up_to_the_horizon_and_holds_out_stripped_prompts() -> None:
    """Killed by: reading one line short of the horizon, which drops a
    prompt on its last line from the request and the hold-out, or holding
    out raw text, which lets a prompt that ends in a newline back into
    H3's sample."""
    texts = [f"how does step {n} of the {MARK} build run\n" for n in range(rp.HORIZON_TURNS + 2)]
    turns = [ab.Turn(seq=n, role="user", text=t, session_id="s1", ts=None, turn_id=f"t{n}")
             for n, t in enumerate(texts)]
    assert len(rp.seen_turns(turns, rp.HORIZON_TURNS)) == rp.HORIZON_TURNS
    held = rp.read_by_pass(turns, [{"session_id": "s1"}], rp.HORIZON_TURNS)
    assert held == {t.strip() for t in texts[:rp.HORIZON_TURNS]}
    sampled = ab.sample_prompts(turns, 1, len(texts), exclude=held)
    assert [p["text"] for p in sampled] == texts[rp.HORIZON_TURNS:]


def test_a_phantom_packed_past_the_top_ten_is_judged(tmp_path: Path) -> None:
    """Killed by: building the packet from the top 10s alone, without the
    phantom-against-base pair."""
    db = tmp_path / "p.db"
    s = MemoryStore(str(db))
    try:
        for i in range(20):
            s.insert_belief(_belief(f"b{i:02d}", f"qv belief number {i} of the packet store.", None))
    finally:
        s.close()
    ranked = {BASE: [f"b{i:02d}" for i in range(11)],
              PHANTOM: [*(f"b{i:02d}" for i in range(10)), "b11", "b12"]}
    runs = {arm: {"arm": arm, "snapshot": SNAP, "prompts": [
        {"prompt_id": "p0", "prompt": "a qv prompt", "ranked": ids}]} for arm, ids in ranked.items()}
    _packet, key = rp.retrieval_packet(runs, db, 3, snapshot=SNAP)
    arms = {r["belief_id"]: r["arms"] for r in key if r["kind"] == ab.KEY_ITEM and r["arms"]}
    assert arms["b10"] == {BASE: 11}
    assert (arms["b11"], arms["b12"]) == ({PHANTOM: 11}, {PHANTOM: 12})


def test_top_counts_read_each_prompts_top_ten() -> None:
    """Killed by: counting past the top 10 for the narration share."""
    run = {"prompts": [{"ranked": [f"x{i}" for i in range(12)]}, {"ranked": ["x0"]}]}
    assert rp.top_counts(run, {"x0", "x10", "x11"}) == (2, 11)


def test_the_replay_never_writes_the_machines_feed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Killed by: not pinning `AELFRICE_DB` to the copy, or leaving the feed on."""
    home = tmp_path / "home-store"
    home.mkdir()
    monkeypatch.setenv("AELFRICE_DB", str(home / "memory.db"))
    monkeypatch.setenv(feed_log.ENV_FEED_LOG, "1")
    _flow(tmp_path, capsys)
    assert not feed_log.feed_path(home).exists()
    assert os.environ["AELFRICE_DB"] == str(home / "memory.db")
    with rp.pinned_store_env(tmp_path / "copy.db"):
        assert not feed_log.is_enabled()
        assert feed_log.feed_path().parent == tmp_path
    assert os.environ[feed_log.ENV_FEED_LOG] == "1"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX mode bits")
def test_run_writes_owner_only_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Killed by: writing the packets, the held claims, or the key with the default modes."""
    _store_path, out, _ = _flow(tmp_path, capsys)
    _release(tmp_path, capsys, out)
    assert (out / rp.CLAIMS_DIR / rp.PACKET_FILE).is_file() and (out / rp.RELEASE_FILE).is_file()
    for p in (out, *out.rglob("*")):
        assert stat.S_IMODE(p.stat().st_mode) in (0o700, 0o600), p


# --- refusals -----------------------------------------------------------------


def _setup_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> tuple[Path, Path, list[dict[str, Any]]]:
    store = _store(tmp_path)
    reqs = _requests(tmp_path, capsys, store)[1:]
    return store, tmp_path / "scratch" / "requests.jsonl", reqs


def _run_cli(capsys: pytest.CaptureFixture[str], requests: Path, replies: Path,
             out: Path, *stores: Path) -> tuple[int, str]:
    return _main(capsys, "run", "--requests", str(requests), "--replies", str(replies),
                 "--store", *(str(s) for s in stores), "--out", str(out))


def test_run_refuses_a_missing_reply(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Killed by: running the requests that have replies and skipping the rest."""
    store, requests, reqs = _setup_run(tmp_path, capsys)
    replies = _write_replies(tmp_path / "r.jsonl", [_reply(reqs[0])])
    code, text = _run_cli(capsys, requests, replies, tmp_path / "out", store)
    assert code == 1
    assert "no reply to request" in text
    assert not (tmp_path / "out").exists()


def test_run_refuses_a_reply_to_no_request(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Killed by: ignoring a reply that matches no request."""
    store, requests, reqs = _setup_run(tmp_path, capsys)
    stray = {**_reply(reqs[0]), "request_id": "0" * 16}
    replies = _write_replies(tmp_path / "r.jsonl", [*(_reply(r) for r in reqs), stray])
    code, text = _run_cli(capsys, requests, replies, tmp_path / "out", store)
    assert (code, "reply to unknown request" in text) == (1, True)


def test_run_refuses_missing_inputs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Killed by: treating a missing requests or replies file, or a missing
    store, as empty."""
    store, requests, reqs = _setup_run(tmp_path, capsys)
    replies = _write_replies(tmp_path / "r.jsonl", [_reply(r) for r in reqs])
    assert _run_cli(capsys, tmp_path / "none.jsonl", replies, tmp_path / "o1", store)[0] == 1
    assert _run_cli(capsys, requests, tmp_path / "none.jsonl", tmp_path / "o2", store)[0] == 1
    assert _run_cli(capsys, requests, replies, tmp_path / "o3", tmp_path / "none.db")[0] == 1
    other = _store(tmp_path / "other", "other")
    s = MemoryStore(str(other))
    try:
        s.insert_belief(_belief("extra", "An extra qv belief for another snapshot.", None))
    finally:
        s.close()
    code, text = _run_cli(capsys, requests, replies, tmp_path / "o4", other)
    assert code == 1 and "no --store holds" in text
    assert not any((tmp_path / f"o{i}").exists() for i in range(1, 5))


def test_run_refuses_a_store_that_changed_since_its_requests(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Killed by: dropping the snapshot match, which grades replies to prompts
    built on another store."""
    store, requests, reqs = _setup_run(tmp_path, capsys)
    replies = _write_replies(tmp_path / "r.jsonl", [_reply(r) for r in reqs])
    s = MemoryStore(str(store))
    try:
        s.insert_belief(_belief("late", f"A late {MARK} belief.", None))
    finally:
        s.close()
    code, text = _run_cli(capsys, requests, replies, tmp_path / "out", store)
    assert code == 1
    assert "changed since its requests" in text


def test_a_store_that_changes_during_run_is_refused(tmp_path: Path) -> None:
    """Killed by: dropping `run_store`'s own snapshot check, which catches a
    write between `run_all`'s match and the copy it ingests on."""
    store = _store(tmp_path)
    work = tmp_path / "w"
    work.mkdir()
    with pytest.raises(ab.InputError, match="changed since its requests"):
        rp.run_store(store, {"snapshot": "0" * 64}, [], {}, work, seed=1, max_prompts=5)


def _jsonl_file(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_requests_and_replies_that_do_not_pair_up_are_refused(tmp_path: Path) -> None:
    """Killed by: dropping the check on a repeated request id, a request with
    no store row, a repeated reply, or a returncode that isn't an int."""
    store_row = {"kind": rp.ROW_STORE, "snapshot": SNAP}
    req = {"kind": rp.ROW_REQUEST, "snapshot": SNAP, "request_id": "r1", "session_id": "s",
           "topic": "t", "input": "i", "axes": [], "anchor_ids": []}
    good = rp.read_requests(_jsonl_file(tmp_path / "q0.jsonl", [store_row, req]))
    assert [r["request_id"] for r in good.requests] == ["r1"]
    with pytest.raises(ab.InputError, match="repeats a request id"):
        rp.read_requests(_jsonl_file(tmp_path / "q1.jsonl", [store_row, req, req]))
    with pytest.raises(ab.InputError, match="have no store row"):
        rp.read_requests(_jsonl_file(tmp_path / "q2.jsonl",
                                     [store_row, {**req, "snapshot": "s9"}]))
    reply = {"request_id": "r1", "stdout": "{}", "returncode": 0}
    assert rp.read_replies(_jsonl_file(tmp_path / "r0.jsonl", [reply]), good) == {"r1": ("{}", 0)}
    with pytest.raises(ab.InputError, match="two replies to request"):
        rp.read_replies(_jsonl_file(tmp_path / "r1.jsonl", [reply, reply]), good)
    for code in ("0", True):
        with pytest.raises(ab.InputError, match="needs request_id, stdout, and returncode"):
            rp.read_replies(_jsonl_file(tmp_path / "r2.jsonl", [{**reply, "returncode": code}]), good)


def test_run_refuses_a_store_given_twice_or_one_with_no_requests(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Killed by: dropping the same-snapshot or the extra-store check in `run_all`."""
    store, requests, reqs = _setup_run(tmp_path, capsys)
    replies = _write_replies(tmp_path / "r.jsonl", [_reply(r) for r in reqs])
    code, text = _run_cli(capsys, requests, replies, tmp_path / "o1", store, store)
    assert code == 1 and "are one store" in text
    with pytest.raises(rp.InputError, match="same snapshot"):
        rp.run_all(rp.read_requests(requests), {}, [store, store], None, seed=1, max_prompts=5)
    other = _store(tmp_path / "other", "other")
    code, text = _run_cli(capsys, requests, replies, tmp_path / "o2", store, other)
    assert code == 1 and "has no requests" in text
    assert not (tmp_path / "o1").exists() and not (tmp_path / "o2").exists()


def test_run_refuses_a_packet_id_collision(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Killed by: pooling packets whose row ids collide."""
    store, requests, reqs = _setup_run(tmp_path, capsys)
    replies = rp.read_replies(_write_replies(tmp_path / "r.jsonl", [_reply(r) for r in reqs]),
                              rp.read_requests(requests))

    def colliding(*_args: object, **_kw: object) -> rp.StoreRun:
        return rp.StoreRun(retrieval=[{"id": "x"}], claims=[{"id": "x"}], key=[], report={})

    monkeypatch.setattr(rp, "run_store", colliding)
    with pytest.raises(ab.InputError, match="packet id collision"):
        rp.run_all(rp.read_requests(requests), replies, [store], None, seed=1, max_prompts=5)


def test_a_key_or_report_that_does_not_fit_is_refused(tmp_path: Path) -> None:
    """Killed by: accepting a key store row for other arms, or a report
    whose stores are not the key's."""
    row = {"kind": ab.KEY_STORE, "snapshot": SNAP, "prompts_sampled": 1,
           "arms_run": sorted(rp.ARMS_RUN)}
    key = rp.read_key(_jsonl_file(tmp_path / "k0.jsonl", [row]))
    with pytest.raises(ab.InputError, match="malformed or repeated store row"):
        rp.read_key(_jsonl_file(tmp_path / "k1.jsonl", [{**row, "arms_run": [BASE]}]))
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"stores": [{"snapshot": SNAP}]}), encoding="utf-8")
    assert rp.read_report(report, key)["stores"] == [{"snapshot": SNAP}]
    report.write_text(json.dumps({"stores": [{"snapshot": "s9"}]}), encoding="utf-8")
    with pytest.raises(ab.InputError, match="not the key's stores"):
        rp.read_report(report, key)


def test_bad_counts_and_existing_outputs_are_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Killed by: dropping the `--sessions` or `--max` floor, or writing over
    an output that exists."""
    store, requests, reqs = _setup_run(tmp_path, capsys)
    code, text = _main(capsys, "requests", "--store", str(store), "--sessions", "0",
                       "--out", str(tmp_path / "q.jsonl"))
    assert code == 1 and "--sessions must be at least 1" in text
    replies = _write_replies(tmp_path / "r.jsonl", [_reply(r) for r in reqs])
    code, text = _main(capsys, "run", "--requests", str(requests), "--replies", str(replies),
                       "--store", str(store), "--max", "0", "--out", str(tmp_path / "o"),
                       "--dry-run")
    assert code == 1 and "--max must be at least 1" in text
    before = requests.read_bytes()
    code, text = _main(capsys, "requests", "--store", str(store), "--out", str(requests))
    assert code == 1 and "exists; pick a new file" in text
    assert requests.read_bytes() == before
    taken = tmp_path / "taken"
    taken.mkdir()
    code, text = _run_cli(capsys, requests, replies, taken, store)
    assert code == 1 and "exists; pick a new directory" in text
    assert not any(taken.iterdir())


def test_a_score_dry_run_refuses_labels_that_miss_a_row(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Killed by: a `score --dry-run` that skips the label coverage check."""
    _store_path, out, _ = _flow(tmp_path, capsys)
    a, c = _release(tmp_path, capsys, out)
    claim_ids = [r["id"] for r in _rows(out / rp.CLAIMS_DIR / rp.PACKET_FILE)]
    full = _scores(tmp_path / "claims.jsonl", claim_ids)
    part = _scores(tmp_path / "part.jsonl", claim_ids[1:])
    code, text = _main(capsys, "score", "--run", str(out), "--labels", str(a), str(c),
                       "--claim-labels", str(full), str(full), "--dry-run")
    assert code == 0, text
    code, text = _main(capsys, "score", "--run", str(out), "--labels", str(a), str(c),
                       "--claim-labels", str(full), str(part), "--dry-run")
    assert code == 1 and "lack a label" in text


def test_outputs_inside_the_repository_are_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Killed by: dropping the repository check on either output. A name of
    this run's own, removed afterward: a fixed one left behind by a mutant
    failed this test in every later run of that tree."""
    store, requests, reqs = _setup_run(tmp_path, capsys)
    inside = rp.REPO_ROOT.joinpath(f"replay-out-{tmp_path.name}")  # path-filter: none
    try:
        code, text = _main(capsys, "requests", "--store", str(store), "--out", str(inside / "q.jsonl"))
        assert code == 1 and "inside the repository" in text
        replies = _write_replies(tmp_path / "r.jsonl", [_reply(r) for r in reqs])
        code, text = _run_cli(capsys, requests, replies, inside, store)
        assert code == 1 and "inside the repository" in text
        assert not inside.exists()
    finally:
        shutil.rmtree(inside, ignore_errors=True)


def test_a_work_dir_inside_the_repository_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fake_runner: Path,
) -> None:
    """The temporary folder holds store copies. Killed by: leaving any
    subcommand's `--work-dir` unchecked."""
    store, requests, reqs = _setup_run(tmp_path, capsys)
    replies = _write_replies(tmp_path / "r.jsonl", [_reply(r) for r in reqs])
    inside = rp.REPO_ROOT.joinpath(f"replay-work-{tmp_path.name}")  # path-filter: none
    try:
        for argv in (
            ("replay", "--store", str(store), "--runner", str(fake_runner), "--out", str(tmp_path / "out1")),
            ("requests", "--store", str(store), "--out", str(tmp_path / "out2.jsonl")),
            ("run", "--requests", str(requests), "--replies", str(replies), "--store", str(store),
             "--out", str(tmp_path / "out3")),
        ):
            code, text = _main(capsys, *argv, "--work-dir", str(inside))
            assert code == 1, argv[0]
            assert "inside the repository" in text, argv[0]
        assert _calls(fake_runner) == []
        assert not inside.exists()
    finally:
        shutil.rmtree(inside, ignore_errors=True)


def test_a_link_into_the_repository_is_refused(tmp_path: Path) -> None:
    """Killed by: comparing only the path as given. Written `link/..`, it
    collapses to a folder outside the repository, and the system's own walk
    lands inside it."""
    root = tmp_path / "repo"
    (root / "sub").mkdir(parents=True)
    link = tmp_path / "looks-outside"
    link.symlink_to(root / "sub")
    for path in (link / "results", link / ".." / "results"):
        with pytest.raises(rp.InputError, match="inside the repository"):
            rp.require_outside_repo(path, root)


def test_a_checkout_removed_during_the_check_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sibling worktrees come and go on a busy machine. Killed by: a comparison
    that raises on a checkout gone since the listing, which reads as a failure
    of the command instead of as no match."""
    gone = tmp_path / "gone-worktree"
    monkeypatch.setattr(rp, "repository_roots", lambda _root: [gone])
    rp.require_outside_repo(tmp_path / "results", tmp_path)


def test_a_worktree_file_removed_during_the_listing_is_skipped(tmp_path: Path) -> None:
    """Killed by: raising on a `gitdir` file that went away after the glob."""
    main = tmp_path / "main"
    (main / ".git" / "worktrees" / "gone" / "gitdir").mkdir(parents=True)
    assert rp.repository_roots(main) == [main, main]


def test_outputs_in_any_checkout_of_the_repository_are_refused(tmp_path: Path) -> None:
    """Killed by: checking only the checkout the harness runs from, which
    lets a run from a linked worktree write into the main checkout or a
    sibling worktree, or failing on a worktree whose directory is gone."""
    main = tmp_path / "main"
    trees = {name: tmp_path / "trees" / name for name in ("here", "sibling", "gone")}
    for name, wt in trees.items():
        meta = main / ".git" / "worktrees" / name
        meta.mkdir(parents=True)
        (meta / "commondir").write_text("../..\n", encoding="utf-8")
        (meta / "gitdir").write_text(f"{wt / '.git'}\n", encoding="utf-8")
        if name != "gone":
            wt.mkdir(parents=True)
            (wt / ".git").write_text(f"gitdir: {meta}\n", encoding="utf-8")
    here, sibling = trees["here"], trees["sibling"]
    assert set(rp.repository_roots(here)) == {here, main, *trees.values()}
    assert set(rp.repository_roots(main)) == set(rp.repository_roots(here))
    for root in (here, main):
        for inside in (main / "q.jsonl", main / "benchmarks" / "new" / "run", sibling / "run",
                       here / "run"):
            with pytest.raises(ab.InputError, match="inside the repository"):
                rp.require_outside_repo(inside, root=root)
        rp.require_outside_repo(tmp_path / "scratch" / "run", root=root)
    lone = tmp_path / "lone"
    lone.mkdir()
    assert rp.repository_roots(lone) == [lone]
    bare_meta = tmp_path / "bare" / "r.git" / "worktrees" / "b"
    bare_meta.mkdir(parents=True)
    bare_tree = tmp_path / "bare-tree"
    bare_tree.mkdir()
    (bare_meta / "commondir").write_text("../..\n", encoding="utf-8")
    (bare_meta / "gitdir").write_text(f"{bare_tree / '.git'}\n", encoding="utf-8")
    (bare_tree / ".git").write_text(f"gitdir: {bare_meta}\n", encoding="utf-8")
    assert set(rp.repository_roots(bare_tree)) == {bare_tree}
    rp.require_outside_repo(tmp_path / "bare" / "run", root=bare_tree)


def test_dry_runs_write_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Killed by: a `--dry-run` that writes its output or copies a store."""
    store, requests, reqs = _setup_run(tmp_path, capsys)
    q = tmp_path / "dry" / "q.jsonl"
    code, text = _main(capsys, "requests", "--store", str(store), "--out", str(q), "--dry-run")
    assert code == 0 and json.loads(text)["dry_run"] is True
    replies = _write_replies(tmp_path / "r.jsonl", [_reply(r) for r in reqs])
    out = tmp_path / "dry" / "run"
    code, text = _main(capsys, "run", "--requests", str(requests), "--replies", str(replies),
                       "--store", str(store), "--out", str(out), "--dry-run")
    assert code == 0 and json.loads(text)["requests"] == 2
    assert not (tmp_path / "dry").exists()
    code, text = _main(capsys, "run", "--requests", str(requests), "--replies",
                       str(_write_replies(tmp_path / "r1.jsonl", [_reply(reqs[0])])),
                       "--store", str(store), "--out", str(out), "--dry-run")
    assert code == 1


def test_no_belief_or_prompt_text_reaches_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Killed by: printing a topic, a prompt, a snippet, a claim, or a note."""
    _store_path, out, text = _flow(tmp_path, capsys)
    assert MARK not in text
    ids = [r["id"] for r in _rows(out / rp.RETRIEVAL_DIR / rp.PACKET_FILE)]
    a, c = (_scores(tmp_path / f"retrieval-{g}.jsonl", ids) for g in "ac")
    code, text = _main(capsys, "claims", "--run", str(out), "--labels", str(a), str(c))
    assert code == 0 and MARK not in text
    claim_ids = [r["id"] for r in _rows(out / rp.CLAIMS_DIR / rp.PACKET_FILE)]
    ca, cc = (_scores(tmp_path / f"claims-{g}.jsonl", claim_ids) for g in "ac")
    code, text = _main(capsys, "score", "--run", str(out), "--labels", str(a), str(c),
                       "--claim-labels", str(ca), str(cc))
    assert code == 0, text
    assert MARK not in text
    assert json.loads(text)["claim_rows"] == 4
    packet = (out / rp.CLAIMS_DIR / rp.PACKET_FILE).read_text(encoding="utf-8")
    assert MARK in packet


def test_the_claims_packet_waits_for_every_retrieval_label(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Killed by: writing the claims packet with the retrieval packet, which
    shows a grader which retrieval rows only the phantom copy packed,
    releasing it before both graders scored every retrieval row, or
    scoring retrieval labels changed after the release."""
    _store_path, out, _ = _flow(tmp_path, capsys)
    assert not (out / rp.CLAIMS_DIR).exists()
    held = _rows(out / rp.CLAIMS_HELD_FILE)
    retrieval = _rows(out / rp.RETRIEVAL_DIR / rp.PACKET_FILE)
    assert {r["claim"] for r in held} & {r["belief"] for r in retrieval}
    ids = [r["id"] for r in retrieval]
    a = _scores(tmp_path / "a.jsonl", ids)
    short = _scores(tmp_path / "c-short.jsonl", ids[1:])
    code, text = _main(capsys, "claims", "--run", str(out), "--labels", str(a), str(short))
    assert code == 1 and "lack a label" in text
    assert not (out / rp.CLAIMS_DIR).exists() and not (out / rp.RELEASE_FILE).exists()
    c = _scores(tmp_path / "c.jsonl", ids, score=3)
    code, text = _main(capsys, "claims", "--run", str(out), "--labels", str(a), str(c),
                       "--dry-run")
    assert (code, json.loads(text)["claim_rows"]) == (0, 4)
    assert not (out / rp.CLAIMS_DIR).exists() and not (out / rp.RELEASE_FILE).exists()
    ca = _scores(tmp_path / "claims-a.jsonl", [r["id"] for r in held])
    code, text = _main(capsys, "score", "--run", str(out), "--labels", str(a), str(c),
                       "--claim-labels", str(ca), str(ca))
    assert code == 1 and "release the claims packet" in text
    code, text = _main(capsys, "claims", "--run", str(out), "--labels", str(a), str(c))
    assert (code, json.loads(text)["claim_rows"]) == (0, 4)
    assert _rows(out / rp.CLAIMS_DIR / rp.PACKET_FILE) == held
    assert (out / rp.CLAIMS_DIR / rp.INSTRUCTIONS_FILE).read_text(
        encoding="utf-8") == rp.CLAIM_INSTRUCTIONS
    code, text = _main(capsys, "claims", "--run", str(out), "--labels", str(a), str(c))
    assert code == 1 and "already released" in text
    code, text = _main(capsys, "score", "--run", str(out), "--labels", str(a), str(c),
                       "--claim-labels", str(ca), str(ca))
    assert code == 0, text
    _scores(c, ids, score=0)
    code, text = _main(capsys, "score", "--run", str(out), "--labels", str(a), str(c),
                       "--claim-labels", str(ca), str(ca))
    assert code == 1 and "retrieval labels differ" in text


def test_the_claims_release_refuses_held_rows_that_are_not_the_keys_claims(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Killed by: releasing held rows without matching them to the key."""
    _store_path, out, _ = _flow(tmp_path, capsys)
    ids = [r["id"] for r in _rows(out / rp.RETRIEVAL_DIR / rp.PACKET_FILE)]
    a = _scores(tmp_path / "a.jsonl", ids)
    held = out / rp.CLAIMS_HELD_FILE
    lines = held.read_text(encoding="utf-8").splitlines()
    held.write_text("".join(ln + "\n" for ln in lines[1:]), encoding="utf-8")
    code, text = _main(capsys, "claims", "--run", str(out), "--labels", str(a), str(a))
    assert code == 1 and "not the key's claims" in text
    assert not (out / rp.CLAIMS_DIR).exists() and not (out / rp.RELEASE_FILE).exists()


# --- labels -------------------------------------------------------------------


def test_scores_are_read_at_the_cut_of_two(tmp_path: Path) -> None:
    """Killed by: another cut, or reading a binary label file as scores."""
    p = tmp_path / "s.jsonl"
    p.write_text("".join(json.dumps({"id": f"r{s}", "score": s}) + "\n" for s in (0, 1, 2, 3))
                 + json.dumps({"id": "rn", "score": None}) + "\n", encoding="utf-8")
    assert rp.read_scores(p) == {"r0": False, "r1": False, "r2": True, "r3": True, "rn": None}
    p.write_text(json.dumps({"id": "r", "correct": True}) + "\n", encoding="utf-8")
    with pytest.raises(ab.InputError):
        rp.read_scores(p)


# --- H1 to H7 -----------------------------------------------------------------


def _store_report(**kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "snapshot": SNAP, "eligible_sessions": 10, "requests": 10,
        "run_outcomes": {wb.OUTCOME_ACCEPTED: 10},
        "session_outcomes": {rp.S_REQUEST: 10}, "runs_with_claim": 9, "costs": [0.03] * 10,
        "claims_written": 20, "documents_written": 10, "phantom_hits": 5, "promotable": 0,
        "gc_collected_at_ttl": 30, "hook_sessions_by_day": {"2026-10-01": 3},
        "narration": {BASE: {"narration": 2, "top": 100}, PHANTOM: {"narration": 2, "top": 100}},
        "usage": {"phantom_hits": 4, "phantom_used": 1, "other_hits": 8, "other_used": 2},
    }
    base.update(kw)
    return base


def test_h1_parse_failures_against_their_bar() -> None:
    """Killed by: another bar, a strict comparison at the bar, or counting
    any outcome but `bad_reply` as a parse failure."""
    at = rp.h1_supply([_store_report(run_outcomes={wb.OUTCOME_ACCEPTED: 8, wb.OUTCOME_BAD_REPLY: 1,
                                                   wb.OUTCOME_RUN_FAILED: 1})])
    assert (at["parse_failures"], at["within_bar"], at["supply"]) == (1, True, 0.9)
    over = rp.h1_supply([_store_report(requests=9, run_outcomes={wb.OUTCOME_BAD_REPLY: 1,
                                                                wb.OUTCOME_ACCEPTED: 8})])
    assert over["within_bar"] is False
    assert rp.PARSE_FAILURE_BAR == 0.10
    two = rp.h1_supply([_store_report(session_outcomes={rp.S_REQUEST: 2, rp.S_SHORT: 1}),
                        _store_report(session_outcomes={rp.S_REQUEST: 3})])
    assert two["session_outcomes"] == {rp.S_REQUEST: 5, rp.S_SHORT: 1}
    assert (two["requests"], two["runs_with_claim"]) == (20, 18)


def test_h1_supply_is_a_share_of_eligible_sessions() -> None:
    """Killed by: dividing by requests, which drops an eligible session the
    pass found no topic in, or by one store's sessions."""
    h1 = rp.h1_supply([_store_report(eligible_sessions=12, requests=10, runs_with_claim=9),
                       _store_report(eligible_sessions=3, requests=2, runs_with_claim=2,
                                     run_outcomes={wb.OUTCOME_ACCEPTED: 2})])
    assert (h1["eligible_sessions"], h1["requests"], h1["runs_with_claim"]) == (15, 12, 11)
    assert h1["supply"] == pytest.approx(11 / 15)
    assert h1["parse_failure_share"] == 0.0


def test_run_counts_a_session_with_no_topic_as_eligible(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Killed by: reporting a store's requests as its eligible sessions."""
    store = _store(tmp_path)
    s = MemoryStore(str(store))
    try:
        s.insert_belief(_belief("nt1", "The qqqfoo grommet is blue.", "sess-notopic"))
        s.record_ingest(source_kind=INGEST_SOURCE_TRANSCRIPT, raw_text="widget", session_id="sess-notopic")
    finally:
        s.close()
    log = store.parent / "transcripts" / "turns.jsonl"
    with log.open("a", encoding="utf-8") as f:
        for row in _session("sess-notopic", [f"plover xyzzy number {i}" for i in range(EXCHANGES)],
                            300):
            f.write(json.dumps(row) + "\n")
    rows = _requests(tmp_path, capsys, store)
    assert (rows[0]["outcomes"][rp.S_NO_TOPIC], rows[0]["considered"]) == (1, 3)
    replies = _write_replies(tmp_path / "scratch" / "replies.jsonl",
                             [_reply(r) for r in rows[1:]])
    out = tmp_path / "scratch" / "run"
    code, text = _run_cli(capsys, tmp_path / "scratch" / "requests.jsonl", replies, out, store)
    assert code == 0, text
    (rep,) = _report(out)["stores"]
    assert (rep["eligible_sessions"], rep["requests"], rep["runs_with_claim"]) == (3, 2, 2)
    assert rp.h1_supply([rep])["supply"] == pytest.approx(2 / 3)
    assert json.loads(text)["H1"]["supply"] == pytest.approx(2 / 3)


def _labels(ids: Sequence[str], good: int) -> dict[str, bool | None]:
    return {i: n < good for n, i in enumerate(ids)}


def test_h2_needs_seventy_percent_from_each_grader() -> None:
    """Killed by: another bar, a strict comparison, or passing on one grader."""
    ids = [f"c{i:02d}" for i in range(30)]
    assert rp.h2_precision(ids, _labels(ids, 21), _labels(ids, 21))["verdict"] == rp.V_PASS
    assert rp.h2_precision(ids, _labels(ids, 20), _labels(ids, 20))["verdict"] == rp.V_FAIL
    worse = _labels(ids, 21)
    worse["c20"] = False
    assert rp.h2_precision(ids, _labels(ids, 21), worse)["verdict"] == rp.V_FAIL
    assert rp.CLAIM_PRECISION_BAR == 0.70


def test_h2_is_unreliable_under_the_kappa_bar_and_insufficient_under_min_graded() -> None:
    """Killed by: dropping the kappa check or the `MIN_GRADED` floor, or
    applying either rule to one grader's position only."""
    ids = [f"c{i:02d}" for i in range(40)]
    a = _labels(ids, 30)
    c = {i: (not v) if n % 3 == 0 else v for n, (i, v) in enumerate(a.items())}
    assert rp.h2_precision(ids, a, c)["verdict"] == rp.V_UNRELIABLE
    assert rp.h2_precision(ids, c, a)["verdict"] == rp.V_UNRELIABLE
    few = ids[: MIN_GRADED - 1]
    assert rp.h2_precision(few, _labels(few, 25), _labels(few, 25))["verdict"] == rp.V_INSUFFICIENT
    scant = {i: None if n < len(ids) - (MIN_GRADED - 1) else n < len(ids) - 1
             for n, i in enumerate(ids)}
    full = _labels(ids, len(ids) - 1)
    for x, y in ((full, scant), (scant, full)):
        h2 = rp.h2_precision(ids, x, y)
        assert (h2["kappa"], h2["verdict"]) == (1.0, rp.V_INSUFFICIENT)
    assert rp.KAPPA_BAR == 0.60


def test_h2_passes_at_a_kappa_of_exactly_the_bar() -> None:
    """Killed by: a strict comparison at the kappa bar."""
    ids = [f"c{i:02d}" for i in range(40)]
    a = {i: n < 30 for n, i in enumerate(ids)}
    c = {i: n < 27 or 30 <= n < 33 for n, i in enumerate(ids)}
    h2 = rp.h2_precision(ids, a, c)
    assert h2["kappa"] == pytest.approx(0.6)
    assert h2["verdict"] == rp.V_PASS


def test_h2_fails_when_every_grader_is_under_the_bar_whatever_the_kappa() -> None:
    """Killed by: calling the labels unreliable before reading the shares,
    so a pass both graders score at 0% or 20% is not killed."""
    ids = [f"c{i:02d}" for i in range(40)]
    none = rp.h2_precision(ids, _labels(ids, 0), _labels(ids, 0))
    assert (none["kappa"], none["verdict"]) == (None, rp.V_FAIL)
    assert rp.decide(none["verdict"], rp.V_PASS, rp.V_PASS, 0) == rp.DECISION_KILL
    a = {i: n < 8 for n, i in enumerate(ids)}
    c = {i: n >= 32 for n, i in enumerate(ids)}
    low = rp.h2_precision(ids, a, c)
    assert low["kappa"] == pytest.approx(-0.25)
    assert low["verdict"] == rp.V_FAIL
    every = rp.h2_precision(ids, _labels(ids, 40), _labels(ids, 40))
    assert (every["kappa"], every["verdict"]) == (None, rp.V_UNRELIABLE)


def _rkey(n: int, gain: int, *, extra_irrelevant: int = 0, base_only: int = 0,
          sampled: int | None = None) -> ab.Key:
    """A key over `n` keyed prompts of `sampled` (default `n`) sampled ones."""
    items: list[ab.KeyItem] = []
    for i in range(n):
        pid = f"p{i:02d}"
        items.append(ab.KeyItem(f"both{i}", pid, {BASE: 1, PHANTOM: 1}, True, SNAP))
        items.append(ab.KeyItem(f"null{i}", pid, None, False, SNAP))
        if i < gain:
            items.append(ab.KeyItem(f"ph{i}", pid, {PHANTOM: 2}, True, SNAP))
        if i < extra_irrelevant:
            items.append(ab.KeyItem(f"junk{i}", pid, {PHANTOM: 3}, False, SNAP))
        if i < base_only:
            items.append(ab.KeyItem(f"lost{i}", pid, {BASE: 2}, True, SNAP))
    return ab.make_key(items, {SNAP: ab.KeyStore(prompts_sampled=sampled or n,
                                                 arms_run=(BASE, PHANTOM))})


def _truth(key: ab.Key) -> dict[str, bool | None]:
    return {i.id: not (i.id.startswith("null") or i.id.startswith("junk")) for i in key.items}


def test_h3_gain_is_phantom_only_minus_base_only_against_the_bar() -> None:
    """Killed by: another bar, comparing against the wrong control, or
    counting beliefs both copies packed."""
    for gain, verdict in ((4, rp.V_PASS), (3, rp.V_FAIL)):
        key = _rkey(40, gain)
        _h0, h3, _h4 = rp.h3_h4(key, _truth(key), _truth(key), [_store_report()], 1811)
        assert h3["verdict"] == verdict, gain
        assert h3["graders"][0]["mean_gain"] == pytest.approx(gain / 40)
    key = _rkey(40, 4, base_only=4)
    _h0, h3, _h4 = rp.h3_h4(key, _truth(key), _truth(key), [_store_report()], 1811)
    assert h3["graders"][0]["mean_gain"] == 0.0
    assert h3["verdict"] == rp.V_FAIL
    assert rp.VALUE_BAR == 0.1


def test_h3_counts_a_sampled_prompt_with_no_keyed_item_as_no_gain() -> None:
    """Killed by: averaging the gain over the keyed prompts only."""
    key = _rkey(40, 4, sampled=80)
    _h0, h3, _h4 = rp.h3_h4(key, _truth(key), _truth(key), [_store_report()], 1811)
    assert h3["graders"][0]["mean_gain"] == pytest.approx(4 / 80)
    assert h3["verdict"] == rp.V_FAIL


def test_h3_needs_the_interval_above_zero() -> None:
    """Killed by: deciding on the mean alone, or letting the interval touch 0."""
    fig = {"graded": 40, "graded_a0": 40, "mean_gain": 0.2}
    assert rp.value_verdict({**fig, "ci95": [0.01, 0.4]}) == rp.V_PASS
    assert rp.value_verdict({**fig, "ci95": [0.0, 0.4]}) == rp.V_FAIL
    assert rp.value_verdict({**fig, "mean_gain": 0.0999, "ci95": [0.01, 0.4]}) == rp.V_FAIL
    assert rp.value_verdict({**fig, "graded_a0": MIN_GRADED - 1, "ci95": [0.01, 0.4]}) == (
        rp.V_INSUFFICIENT)


def test_h3_needs_min_graded_on_both_copies() -> None:
    """Killed by: dropping the floor on the phantom copy, or a strict floor."""
    fig = {"mean_gain": 0.2, "ci95": [0.01, 0.4]}
    assert rp.value_verdict({**fig, "graded": MIN_GRADED - 1, "graded_a0": 40}) == (
        rp.V_INSUFFICIENT)
    assert rp.value_verdict({**fig, "graded": MIN_GRADED, "graded_a0": MIN_GRADED}) == rp.V_PASS


def test_h3_and_h4_precision_are_unreliable_when_the_null_arm_fails() -> None:
    """Killed by: dropping #1795's H0 from the retrieval verdicts."""
    key = _rkey(40, 4)
    loose = {**_truth(key), **{f"null{i}": True for i in range(9)}}
    h0, h3, h4 = rp.h3_h4(key, loose, loose, [_store_report()], 1811)
    assert h0["verdict"] == rp.V_FAIL
    assert (h3["verdict"], h4["precision_verdict"]) == (rp.V_UNRELIABLE, rp.V_UNRELIABLE)
    tight = {**_truth(key), **{f"null{i}": True for i in range(8)}}
    assert rp.h3_h4(key, tight, tight, [_store_report()], 1811)[0]["verdict"] == rp.V_PASS
    split = {**_truth(key), **{f"both{i}": False for i in range(20)}}
    h0, h3, _h4 = rp.h3_h4(key, _truth(key), split, [_store_report()], 1811)
    assert (h0["verdict"], h3["verdict"]) == (rp.V_FAIL, rp.V_UNRELIABLE)
    assert h0["kappa"] < rp.KAPPA_BAR
    assert rp.NULL_RATE_CEILING == 0.20


def test_h0_holds_at_a_kappa_of_exactly_the_bar() -> None:
    """Killed by: a strict comparison at the kappa bar."""
    key = _rkey(40, 4)
    a, c = _truth(key), _truth(key)
    for i in range(24, 28):
        c[f"both{i}"] = False
    for i in range(28, 40):
        a[f"both{i}"] = False
    for i in range(4):
        a[f"ph{i}"] = c[f"ph{i}"] = False
    reliable, h0 = rp.retrieval_reliable(key, a, c)
    assert h0["kappa"] == pytest.approx(0.6)
    assert (reliable, h0["verdict"]) == (True, rp.V_PASS)


def test_h0_fails_when_no_grader_scored_the_null_arm() -> None:
    """Killed by: dividing by an empty null arm instead of failing H0."""
    key = _rkey(40, 4)
    a = {**_truth(key), **{f"null{i}": None for i in range(40)}, "both0": False}
    reliable, h0 = rp.retrieval_reliable(key, a, dict(a))
    assert h0["kappa"] == 1.0
    assert (reliable, h0["verdict"]) == (False, rp.V_FAIL)


def test_h4_precision_drop_holds_at_five_points() -> None:
    """Killed by: another drop bar, or scoring precision past the top 10."""
    for junk, verdict in ((2, rp.V_PASS), (3, rp.V_FAIL)):
        key = _rkey(40, 0, extra_irrelevant=junk)
        _h0, _h3, h4 = rp.h3_h4(key, _truth(key), _truth(key), [_store_report()], 1811)
        assert (h4["precision_verdict"], h4["verdict"]) == (verdict, verdict), junk
    assert rp.PRECISION_DROP_BAR == 0.05


def test_h4_narration_share_may_not_rise() -> None:
    """Killed by: letting the share rise, a strict comparison, or a
    narration failure that a passing precision hides."""
    def report(pn: int, pt: int) -> dict[str, Any]:
        return _store_report(narration={BASE: {"narration": 2, "top": 100},
                                        PHANTOM: {"narration": pn, "top": pt}})
    assert rp.narration_verdict([report(2, 100)])["verdict"] == rp.V_PASS
    assert rp.narration_verdict([report(3, 150)])["verdict"] == rp.V_PASS
    assert rp.narration_verdict([report(3, 100)])["verdict"] == rp.V_FAIL
    assert rp.narration_verdict([report(0, 0)])["verdict"] == rp.V_NOT_MEASURED
    key = _rkey(40, 4)
    _h0, _h3, h4 = rp.h3_h4(key, _truth(key), _truth(key), [report(3, 100)], 1811)
    assert (h4["precision_verdict"], h4["verdict"]) == (rp.V_PASS, rp.V_FAIL)


def test_h4_is_inconclusive_when_narration_was_not_measured() -> None:
    """Killed by: letting a passing precision pass H4 with no top 10 to
    read narration from."""
    key = _rkey(40, 4)
    empty = _store_report(narration={BASE: {"narration": 0, "top": 0},
                                     PHANTOM: {"narration": 0, "top": 0}})
    _h0, _h3, h4 = rp.h3_h4(key, _truth(key), _truth(key), [empty], 1811)
    assert (h4["precision_verdict"], h4["verdict"]) == (rp.V_PASS, rp.V_INCONCLUSIVE)


def test_narration_is_a_belief_only_assistant_rows_derived(tmp_path: Path) -> None:
    """Killed by: counting a belief a user row also derived, or one with no role."""
    db = tmp_path / "n.db"
    MemoryStore(str(db)).close()
    rows = [("l1", {"role": "assistant"}, ["a", "mixed"]), ("l2", {"role": "user"}, ["mixed", "u"]),
            ("l3", {"call_site": "x"}, ["none"]), ("l4", {"role": "assistant"}, ["a"])]
    with sqlite3.connect(str(db)) as conn:
        for lid, meta, ids in rows:
            conn.execute("INSERT INTO ingest_log (id, ts, source_kind, raw_text, raw_meta, "
                         "derived_belief_ids) VALUES (?, '2026-10-01', 'transcript', 'x', ?, ?)",
                         (lid, json.dumps(meta), json.dumps(ids)))
    assert rp.narration_ids(db) == {"a"}


def test_h5_compares_phantom_hits_with_the_other_hits_of_their_prompts() -> None:
    """Killed by: counting prompts with no phantom hit, or swapping the sides."""
    report = {"prompts": [
        {"arms": {PHANTOM: [{"id": "ph", "used": True}, {"id": "x", "used": False}]}},
        {"arms": {PHANTOM: [{"id": "y", "used": True}]}},
    ]}
    counts = rp.usage_counts(report, {"ph"})
    assert counts == {"phantom_hits": 1, "phantom_used": 1, "other_hits": 1, "other_used": 0}
    h5 = rp.h5_usage([_store_report(usage=counts)])
    assert (h5["phantom_rate"], h5["other_rate"], h5["at_least_as_often"]) == (1.0, 0.0, True)
    assert rp.h5_usage([_store_report()])["at_least_as_often"] is True


def test_h6_reports_conversion_against_zero_promoted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Killed by: leaving documents out of the written count, or reading
    survival as the collected share."""
    h6 = rp.h6_conversion([_store_report(gc_collected_at_ttl=27)])
    assert (h6["promotable"], h6["survival_lower_bound"]) == (0, pytest.approx(0.1))
    assert h6["support_events"] == rp.V_NOT_MEASURED
    two = rp.h6_conversion([_store_report(), _store_report(
        claims_written=3, documents_written=2, phantom_hits=7, promotable=1, gc_collected_at_ttl=4)])
    assert {k: two[k] for k in ("claims_written", "documents_written", "phantom_hits", "promotable",
                                "gc_collected_at_ttl")} == {
        "claims_written": 23, "documents_written": 12, "phantom_hits": 12, "promotable": 1,
        "gc_collected_at_ttl": 34}
    assert two["survival_lower_bound"] == pytest.approx(1 / 35)
    _store_path, out, _ = _flow(tmp_path, capsys)
    (rep,) = _report(out)["stores"]
    assert rep["gc_collected_at_ttl"] == rep["claims_written"] + rep["documents_written"] == 6


def test_h7_counts_runs_over_their_cap_and_projects_the_busiest_day() -> None:
    """Killed by: another cap, counting a run at its cap as over it, or
    projecting from the quietest day."""
    rep = _store_report(costs=[0.04, 0.10, 0.11],
                        hook_sessions_by_day={"2026-10-01": 2, "2026-10-02": 5})
    h7 = rp.h7_cost([rep])
    assert (h7["runs"], h7["over_cap"], h7["busiest_day_sessions"]) == (3, 1, 5)
    assert h7["mean_usd"] == pytest.approx(0.25 / 3)
    assert h7["projected_busiest_day_usd"] == pytest.approx(0.25 / 3 * 5)
    assert h7["mean_within_bar"] is False
    assert rp.h7_cost([_store_report(costs=[0.05, 0.05])])["mean_within_bar"] is True
    assert rp.COST_MEAN_BAR == 0.05


def test_the_decision_kills_on_a_failed_must_pass_or_a_run_over_its_cap() -> None:
    """Killed by: passing without all three, or ignoring a run over its cap."""
    p, f, u = rp.V_PASS, rp.V_FAIL, rp.V_UNRELIABLE
    assert rp.decide(p, p, p, 0) == rp.DECISION_PASS
    assert rp.decide(p, p, p, 1) == rp.DECISION_KILL
    for args in ((f, p, p), (p, f, p), (p, p, f), (u, f, p)):
        assert rp.decide(*args, 0) == rp.DECISION_KILL
    assert rp.decide(u, p, p, 0) == rp.DECISION_OPEN
    assert rp.decide(p, rp.V_INSUFFICIENT, p, 0) == rp.DECISION_OPEN


def test_score_decides_from_h2_h3_h4_and_the_cost_cap() -> None:
    """Killed by: deciding without H2's, H3's, or H4's verdict, or without
    the runs over their cap."""
    ids = [f"c{i:02d}" for i in range(40)]

    def decision(gain: int = 4, good: int = 30, **report: Any) -> str:
        key = _rkey(40, gain)
        retrieval = {**_truth(key), **_labels(ids, 0)}
        claims = _labels(ids, good)
        out = rp.score(rp.ReplayKey(retrieval=key, claims=tuple(ids)),
                       {"stores": [_store_report(**report)]}, (retrieval, dict(retrieval)),
                       (claims, dict(claims)), 1811)
        return str(out["decision"])

    assert decision() == rp.DECISION_PASS
    assert decision(good=20) == rp.DECISION_KILL
    assert decision(gain=3) == rp.DECISION_KILL
    assert decision(narration={BASE: {"narration": 2, "top": 100},
                               PHANTOM: {"narration": 3, "top": 100}}) == rp.DECISION_KILL
    assert decision(costs=[0.03, 0.11]) == rp.DECISION_KILL


def test_score_refuses_labels_that_miss_a_row(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Killed by: scoring a claim or a retrieval row no grader scored as not
    correct or not relevant."""
    ids = [f"c{i:02d}" for i in range(40)]
    key = _rkey(40, 4)
    claims = _labels(ids, 30)
    partial = {k: v for k, v in _truth(key).items() if k != "both0"}
    with pytest.raises(ValueError, match="retrieval rows lack a label"):
        rp.score(rp.ReplayKey(retrieval=key, claims=tuple(ids)), {"stores": [_store_report()]},
                 (partial, _truth(key)), (claims, claims), 1811)
    no_first = {k: v for k, v in claims.items() if k != ids[0]}
    with pytest.raises(ValueError, match="claims lack a label"):
        rp.score(rp.ReplayKey(retrieval=key, claims=tuple(ids)), {"stores": [_store_report()]},
                 (_truth(key), _truth(key)), (claims, no_first), 1811)
    _store_path, out, _ = _flow(tmp_path, capsys)
    a, c = _release(tmp_path, capsys, out)
    code, text = _main(capsys, "score", "--run", str(out), "--labels", str(a), str(c),
                       "--claim-labels", str(a), str(c))
    assert code == 1 and "claims lack a label" in text


# --- prepare ------------------------------------------------------------------


def _real_project(root: Path, name: str, *, turns: bool = True) -> Path:
    """A project folder laid out the way a real one is, with a synthetic store."""
    db = root / name / ".git" / "aelfrice" / "memory.db"
    db.parent.mkdir(parents=True)
    store = MemoryStore(str(db))
    try:
        store.insert_belief(_belief(f"{name}-1", f"The {MARK} {name} cache flushes.", OLD))
    finally:
        store.close()
    if turns:
        tdir = db.parent / "transcripts"
        (tdir / "archive").mkdir(parents=True)
        (tdir / "turns.jsonl").write_text(json.dumps(_turn("user", "live", OLD, 1)) + "\n", encoding="utf-8")
        (tdir / "archive" / "turns-1.jsonl").write_text(
            json.dumps(_turn("user", "archived", OLD, 0)) + "\n", encoding="utf-8",
        )
    return db


def test_prepare_copies_stores_and_never_opens_an_excluded_project(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Planted stores in two excluded projects are mode 000, so a read of
    either raises. A third is readable, behind a link. Killed by: dropping
    an exclusion, comparing case-sensitively, following a link into an
    excluded project, or looking inside one first."""
    root = tmp_path / "projects"
    keep = _real_project(root, "widget-one")
    _real_project(root, "widget-two", turns=False)
    for name in ("widget-private", "WIDGET-Secret"):
        planted = _real_project(root, name)
        planted.chmod(0o000)
        planted.parent.chmod(0o000)
    readable = _real_project(root, "widget-hidden")
    assert readable.is_file()
    (root / "link-to-hidden").symlink_to(root / "widget-hidden")
    _real_project(tmp_path / "elsewhere", "real-folder")
    (root / "widget-aliased").symlink_to(tmp_path / "elsewhere" / "real-folder")
    out = tmp_path / "scratch" / "stores"
    try:
        code, text = _main(
            capsys, "prepare", "--projects-root", str(root), "--out", str(out),
            "--exclude", "widget-private", "--exclude", "widget-secret", "--exclude", "WIDGET-hidden",
            "--exclude", "widget-aliased",
        )
    finally:
        for name in ("widget-private", "WIDGET-Secret"):
            (root / name / ".git" / "aelfrice").chmod(0o700)
            (root / name / ".git" / "aelfrice" / "memory.db").chmod(0o600)
    assert code == 0, text
    assert sorted(p.name for p in out.iterdir()) == ["widget-one", "widget-two"]
    assert MARK not in text and "widget-private" not in text
    copied = out / "widget-one" / "memory.db"
    assert stat.S_IMODE(copied.stat().st_mode) == 0o600
    assert (out / "widget-one" / rp.COPY_MARK_FILE).is_file()
    rp.require_store_copy(copied)
    assert stat.S_IMODE((out / "widget-one").stat().st_mode) == 0o700
    assert (out / "widget-one" / "transcripts" / "turns.jsonl").is_file()
    assert (out / "widget-one" / "transcripts" / "archive" / "turns-1.jsonl").is_file()
    conn = sqlite3.connect(copied)
    try:
        assert conn.execute("SELECT id FROM beliefs").fetchall() == [("widget-one-1",)]
    finally:
        conn.close()
    assert sorted(p.name for p in keep.parent.iterdir() if p.name.endswith(("-wal", "-shm"))) == []


@pytest.mark.parametrize("link", [
    ".git", ".git/aelfrice", ".git/aelfrice/memory.db", ".git/aelfrice/memory.db-wal",
    ".git/aelfrice/transcripts", ".git/aelfrice/transcripts/archive", ".git/aelfrice/transcripts/turns.jsonl",
    ".git/aelfrice/transcripts/archive/turns-9.jsonl",
])
def test_a_link_inside_a_project_into_an_excluded_one_drops_it(tmp_path: Path, link: str) -> None:
    """A linked `.git` or `.git/aelfrice` copied an excluded store in the
    review. Killed by: checking only the project folder's own link."""
    root = tmp_path / "projects"
    hidden = _real_project(root, "widget-hidden")
    hidden.with_name(hidden.name + "-wal").write_bytes(b"")
    (hidden.parent / "transcripts" / "archive" / "turns-9.jsonl").write_text("", encoding="utf-8")
    keep = _real_project(root, "widget-one")
    sneak = _real_project(root, "widget-sneak")
    assert sneak != keep
    place = root / "widget-sneak" / link
    if place.is_dir():
        shutil.rmtree(place)
    else:
        place.unlink(missing_ok=True)
    place.parent.mkdir(parents=True, exist_ok=True)
    place.symlink_to(root / "widget-hidden" / link)
    assert [name for name, _db in rp.project_stores(root, frozenset({"widget-hidden"}))] == ["widget-one"]


def test_a_transcripts_link_into_an_excluded_project_is_never_listed(tmp_path: Path) -> None:
    """The planted folder is mode 000, so listing it raises. Killed by: listing
    the transcripts before deciding on the link."""
    root = tmp_path / "projects"
    hidden = _real_project(root, "widget-hidden")
    archive = hidden.parent / "transcripts" / "archive"
    sneak = _real_project(root, "widget-sneak", turns=False)
    (sneak.parent / "transcripts").mkdir()
    (sneak.parent / "transcripts" / "archive").symlink_to(archive)
    archive.chmod(0o000)
    try:
        assert rp.project_stores(root, frozenset({"widget-hidden"})) == []
    finally:
        archive.chmod(0o700)


def test_a_link_into_an_excluded_project_in_another_case_drops_it(tmp_path: Path) -> None:
    """macOS folds case on disk. Killed by: comparing the resolved parts
    case-sensitively."""
    root = tmp_path / "projects"
    _real_project(root, "WIDGET-Hidden")
    _real_project(root, "widget-one")
    (root / "widget-sneak").mkdir()
    (root / "widget-sneak" / ".git").symlink_to(root / "WIDGET-Hidden" / ".git")
    assert [name for name, _db in rp.project_stores(root, rp.excluded_names(["widget-hidden"]))] == ["widget-one"]


def test_a_link_loop_drops_the_project(tmp_path: Path) -> None:
    """Killed by: a loop raising out of `prepare`, on Python versions where
    `resolve` raises."""
    root = tmp_path / "projects"
    _real_project(root, "widget-one")
    (root / "widget-loop").mkdir()
    (root / "widget-loop" / ".git").symlink_to(root / "widget-loop" / ".git")
    assert [name for name, _db in rp.project_stores(root, frozenset({"nope"}))] == ["widget-one"]


def test_prepare_needs_an_exclusion_and_the_repo_names_none(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Killed by: a default that copies every project when the flag is
    forgotten, or a built-in list, which would publish private project names."""
    root = tmp_path / "projects"
    _real_project(root, "widget-one")
    out = tmp_path / "scratch" / "stores"
    with pytest.raises(SystemExit) as info:
        rp.main(["prepare", "--projects-root", str(root), "--out", str(out)])
    assert info.value.code == 1
    assert "--exclude" in capsys.readouterr().err
    assert not out.exists()
    assert rp.excluded_names(None) == frozenset()


def test_prepare_is_idempotent_and_dry_run_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Killed by: a dry run that copies, or a second run that fails on a
    project already copied or overwrites it."""
    root = tmp_path / "projects"
    _real_project(root, "widget-one")
    out = tmp_path / "scratch" / "stores"
    code, text = _main(capsys, "prepare", "--projects-root", str(root), "--out", str(out), "--exclude", "widget-no", "--dry-run")
    assert code == 0, text
    assert json.loads(text)["projects"] == ["widget-one"]
    assert not out.exists()
    code, text = _main(capsys, "prepare", "--projects-root", str(root), "--out", str(out), "--exclude", "widget-no")
    assert code == 0, text
    first = json.loads(text)
    on_disk = sum(p.stat().st_size for p in (out / "widget-one").rglob("*") if p.is_file())
    assert (first["copied"], first["already_copied"], first["bytes_copied"]) == (1, 0, on_disk)
    marker = out / "widget-one" / "memory.db"
    before = marker.stat().st_mtime_ns
    code, text = _main(capsys, "prepare", "--projects-root", str(root), "--out", str(out), "--exclude", "widget-no")
    assert code == 0, text
    second = json.loads(text)
    assert (second["projects"], second["copied"], second["already_copied"], second["bytes_copied"]) == (1, 0, 1, 0)
    assert marker.stat().st_mtime_ns == before


def test_prepare_stops_at_a_store_with_no_mark_and_keeps_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """A copy missing its mark exited 0 and stayed refused in the review, and
    `bytes_copied` counted it. It may also be a real store. Killed by: exiting
    0, copying the other projects first, or deleting it."""
    root = tmp_path / "projects"
    _real_project(root, "widget-one")
    _real_project(root, "widget-two")
    out = tmp_path / "scratch" / "stores"
    (out / "widget-two").mkdir(parents=True)
    (out / "widget-two" / "memory.db").write_bytes(b"not known to be a copy")
    code, text = _main(capsys, "prepare", "--projects-root", str(root), "--out", str(out), "--exclude", "widget-no")
    assert code != 0
    assert "no " + rp.COPY_MARK_FILE in text
    assert (out / "widget-two" / "memory.db").read_bytes() == b"not known to be a copy"
    assert not (out / "widget-one").exists()


def test_the_disk_check_counts_only_what_it_will_copy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Killed by: a dry run reporting bytes for a copy already finished."""
    root = tmp_path / "projects"
    _real_project(root, "widget-one")
    out = tmp_path / "scratch" / "stores"
    assert _main(capsys, "prepare", "--projects-root", str(root), "--out", str(out), "--exclude", "widget-no")[0] == 0
    code, text = _main(capsys, "prepare", "--projects-root", str(root), "--out", str(out), "--exclude", "widget-no", "--dry-run")
    assert code == 0, text
    assert json.loads(text)["bytes_needed"] == 0


def test_prepare_refuses_without_the_disk(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Killed by: starting copies the disk can't hold. A copy filled this disk once."""
    root = tmp_path / "projects"
    _real_project(root, "widget-one")
    free = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(rp.shutil, "disk_usage", lambda _p: free._replace(free=1))
    out = tmp_path / "scratch" / "stores"
    code, text = _main(capsys, "prepare", "--projects-root", str(root), "--out", str(out), "--exclude", "widget-no")
    assert code == 1
    assert "free some disk" in text
    assert not out.exists()


@pytest.mark.parametrize("path", [
    "projects/widget/.git/aelfrice/memory.db",
    "scratch/widget-private/memory.db",
    "scratch/WIDGET-PRIVATE/memory.db",
])
def test_store_refuses_a_real_store_or_an_excluded_project(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], path: str,
) -> None:
    """Planted at mode 000, so opening one raises. Killed by: opening a
    `--store` before checking it, or leaving any subcommand unchecked."""
    db = tmp_path / path
    db.parent.mkdir(parents=True)
    db.write_bytes(b"")
    db.chmod(0o000)
    try:
        for argv in (
            ("replay", "--store", str(db), "--runner", "/bin/echo", "--out", str(tmp_path / "scratch" / "r")),
            ("requests", "--store", str(db), "--out", str(tmp_path / "scratch" / "q.jsonl")),
        ):
            code, text = _main(capsys, *argv, "--exclude", "widget-private")
            assert code == 1
            assert "excluded project" in text or "real store" in text
    finally:
        db.chmod(0o600)


def test_store_refuses_a_marker_store(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A `[store]` marker places a real store anywhere. Killed by: taking a
    store that `prepare` didn't copy."""
    db = _store(tmp_path, "marked")
    (db.parent / rp.COPY_MARK_FILE).unlink()
    project = db.parent.parent
    (project / ".aelfrice.toml").write_text(f'[store]\npath = "{db.parent.name}"\n', encoding="utf-8")
    code, text = _main(capsys, "requests", "--store", str(db), "--out", str(tmp_path / "scratch" / "q.jsonl"))
    assert code == 1
    assert "isn't a copy" in text
    assert not (tmp_path / "scratch" / "q.jsonl").exists()


def test_store_refuses_a_hard_link(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A hard link has no path that names a real store, and writes through
    to it. Killed by: taking one because it has a mark beside it."""
    db = _store(tmp_path, "real")
    twin = tmp_path / "twin" / "memory.db"
    twin.parent.mkdir()
    os.link(db, twin)
    _mark_copy(twin)
    code, text = _main(capsys, "requests", "--store", str(twin), "--out", str(tmp_path / "scratch" / "q.jsonl"))
    assert code == 1
    assert "another name" in text


@pytest.mark.parametrize("folder", [".git", ".GIT", ".aelfrice", ".AELFRICE", ".Aelfrice"])
def test_store_refuses_a_link_to_a_real_store_with_a_mark_on_both_sides(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], folder: str,
) -> None:
    """Killed by: checking only the path as given, which a link hides."""
    home = Path.home()
    real = home / folder / "memory.db" if "aelfrice" in folder.lower() else tmp_path / "proj" / folder / "memory.db"
    real.parent.mkdir(parents=True)
    real.write_bytes(b"")
    _mark_copy(real)
    link = tmp_path / "copies" / "memory.db"
    link.parent.mkdir()
    link.symlink_to(real)
    _mark_copy(link)
    code, text = _main(capsys, "requests", "--store", str(link), "--out", str(tmp_path / "scratch" / "q.jsonl"))
    assert code == 1
    assert "real store" in text


@pytest.mark.parametrize("name", [".aelfrice", ".AELFRICE", ".aelFRICE"])
def test_store_refuses_the_home_store(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, name: str,
) -> None:
    """Killed by: treating `~/.aelfrice/memory.db` as a copy, or comparing its
    path by case, which macOS folds."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    db = tmp_path / "home" / name / "memory.db"
    db.parent.mkdir(parents=True)
    db.write_bytes(b"")
    code, text = _main(capsys, "requests", "--store", str(db), "--out", str(tmp_path / "scratch" / "q.jsonl"))
    assert code == 1
    assert "real store" in text


# --- replay -------------------------------------------------------------------


FAKE_RUNNER = """#!{python}
import json, os, sys
root = os.path.dirname(os.path.abspath(sys.argv[0]))
config = json.load(open(os.path.join(root, "config.json"), encoding="utf-8"))
stdin = sys.stdin.read()
with open(os.path.join(root, "calls.jsonl"), "a", encoding="utf-8") as f:
    f.write(json.dumps({{"argv": sys.argv[1:], "cwd": os.getcwd()}}) + "\\n")
def line(prefix):
    return next(x[len(prefix):] for x in stdin.splitlines() if x.startswith(prefix))
axes = json.loads(line("Axes: "))
ids = [r["id"] for r in json.loads(line("Snippets: "))]
tag = "".join(c for c in line("Topic: ") if c.isalnum())[-24:]
n = sum(1 for _ in open(os.path.join(root, "calls.jsonl"), encoding="utf-8"))
mode = config.get("mode", "ok")
if mode == "exit" and n == 1:
    sys.exit(4)
reply = {{"axes": [{{"axis": axes[0]["name"],
    "document": f"The cache and manifest for {{tag}} change together in run number {{n}}.",
    "claims": [{{"text": f"The {MARK} cache for {{tag}} bounds the manifest in run number {{n}}.",
                "anchors": ids[:1]}}]}}]}}
out = {{"type": "result", "result": json.dumps(reply)}}
if "cost" in config:
    out["total_cost_usd"] = config["cost"]
print(json.dumps(out))
""".replace("{MARK}", MARK)


@pytest.fixture
def fake_runner(tmp_path: Path) -> Path:
    root = tmp_path / "fake"
    root.mkdir()
    runner = root / "hostcli"
    runner.write_text(FAKE_RUNNER.format(python=sys.executable), encoding="utf-8")
    runner.chmod(0o700)
    _configure(runner, cost=0.03)
    return runner


def _configure(runner: Path, **config: object) -> None:
    """What the fake does. A file beside it: the runner gets no test's variables."""
    (runner.parent / "config.json").write_text(json.dumps(config), encoding="utf-8")


def _calls(runner: Path) -> list[dict[str, Any]]:
    path = runner.parent / "calls.jsonl"
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def _replay(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], runner: Path, stores: Sequence[Path],
    *extra: str, name: str = "one",
) -> tuple[int, str]:
    return _main(
        capsys, "replay", "--store", *map(str, stores), "--runner", str(runner),
        "--out", str(tmp_path / "scratch" / "replay" / name),
        *([] if "--budget-usd" in extra else ["--budget-usd", "0.5"]), *extra,
    )


@pytest.mark.timeout(120)
def test_replay_runs_the_pass_and_records_claims_and_costs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fake_runner: Path,
) -> None:
    """Killed by: not running the runner, other flags than the writer's, a
    runner started from a project folder, or a result with no claims,
    costs, grader packet, or grading steps."""
    store = _store(tmp_path)
    code, text = _replay(tmp_path, capsys, fake_runner, [store])
    assert code == 0, text
    assert MARK not in text
    out = tmp_path / "scratch" / "replay" / "one"
    calls = _calls(fake_runner)
    assert len(calls) == 2
    argv = calls[0]["argv"]
    assert argv[argv.index("--max-budget-usd") + 1] == wb.RUNNER_BUDGET_USD
    assert argv[argv.index("--tools") + 1] == ""
    assert "--no-session-persistence" in argv
    assert Path(calls[0]["cwd"]).resolve() == out.resolve()
    summary = json.loads(text)
    assert summary["ran_now"] == 2
    assert summary["spent_usd"] == pytest.approx(0.06)
    assert summary["H7"]["total_usd"] == pytest.approx(0.06)
    ledger = [json.loads(x) for x in rp.spend_ledger_path().read_text(encoding="utf-8").splitlines()]
    assert [r["state"] for r in ledger] == ["started", "done"] * 2
    assert {r["out"] for r in ledger} == {str(out.resolve())}
    assert stat.S_IMODE(rp.spend_ledger_path().stat().st_mode) == 0o600
    assert stat.S_IMODE(rp.spend_ledger_path().parent.stat().st_mode) == 0o700
    run = out / rp.RUN_DIR
    phantoms = [json.loads(x) for x in (run / rp.PHANTOMS_FILE).read_text(encoding="utf-8").splitlines()]
    assert sum(not r["document"] for r in phantoms) == 2
    assert all(MARK in r["text"] or r["document"] for r in phantoms)
    assert all(r["anchors"] for r in phantoms)
    assert (run / rp.RETRIEVAL_DIR / rp.PACKET_FILE).is_file()
    assert (run / rp.RETRIEVAL_DIR / rp.INSTRUCTIONS_FILE).is_file()
    assert str(run) in (run / rp.GRADING_FILE).read_text(encoding="utf-8")
    assert sorted(p.name for p in (run / rp.RETRIEVAL_DIR).iterdir()) == [rp.INSTRUCTIONS_FILE, rp.PACKET_FILE]


def test_the_store_copy_is_left_as_it_was(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fake_runner: Path,
) -> None:
    """Killed by: writing phantoms to the store it was given rather than to
    its own copy, or dropping a feed row beside it."""
    store = _store(tmp_path)
    before = store.read_bytes()
    listing = sorted(p.name for p in store.parent.iterdir())
    assert _replay(tmp_path, capsys, fake_runner, [store])[0] == 0
    assert store.read_bytes() == before
    assert sorted(p.name for p in store.parent.iterdir()) == listing


def test_replay_dry_run_runs_nothing_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fake_runner: Path,
) -> None:
    """Killed by: a dry run that starts the runner or writes a file."""
    store = _store(tmp_path)
    code, text = _replay(tmp_path, capsys, fake_runner, [store], "--dry-run")
    assert code == 0, text
    plan = json.loads(text)
    assert plan["dry_run"] is True
    assert plan["requests"] == 2
    assert plan["plan"]["runs_at_the_cap"] == 2
    assert plan["plan"]["runs_at_most"] == 2
    assert _calls(fake_runner) == []
    assert not (tmp_path / "scratch" / "replay").exists()


@pytest.mark.parametrize("budget", ["1.01", "2", "0", "-0.1", "nan"])
def test_replay_refuses_a_budget_past_the_ceiling(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fake_runner: Path, budget: str,
) -> None:
    """Killed by: accepting more than the ruled $0.50, a budget of zero or
    less, or a NaN that every comparison lets through."""
    code, text = _replay(tmp_path, capsys, fake_runner, [_store(tmp_path)], "--budget-usd", budget)
    assert code == 1
    assert "--budget-usd" in text
    assert _calls(fake_runner) == []


def _many_sessions(tmp_path: Path, n: int) -> Path:
    """One store with `n` eligible sessions."""
    path = tmp_path / "many" / "aelfrice" / "memory.db"
    path.parent.mkdir(parents=True)
    store = MemoryStore(str(path))
    rows: list[dict[str, Any]] = []
    try:
        for i in range(n):
            sid = f"sess-{i:02d}"
            store.insert_belief(_belief(f"m{i}a", f"The {MARK} cache{i} flushes every ten minutes.", sid))
            store.record_ingest(source_kind=INGEST_SOURCE_TRANSCRIPT, raw_text="widget", session_id=sid)
            rows += _session(sid, [f"how does the {MARK} cache{i} handle step {j}" for j in range(EXCHANGES)], 40 * i)
        for i in range(30):
            store.insert_belief(_belief(f"f{i:02d}", " ".join(f"qv{i}{c}" for c in "abcde"), None))
    finally:
        store.close()
    tdir = path.parent / "transcripts"
    tdir.mkdir()
    (tdir / "turns.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    _mark_copy(path)
    return path


@pytest.mark.timeout(120)
def test_the_budget_stop_holds_and_a_resume_keeps_the_total(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fake_runner: Path,
) -> None:
    """Killed by: no stop; a stop that ignores the reserve; a resume or a
    second `--out` beside the first that starts the total over; or a stop
    that lets a later request through.

    At $0.10 a run, $0.50 admits runs while the total plus $0.13 fits: 4."""
    store = _many_sessions(tmp_path, 8)
    _configure(fake_runner, cost=0.10)
    code, text = _replay(tmp_path, capsys, fake_runner, [store], "--sessions", "8", "--max-calls", "5")
    assert code == 0, text
    summary = json.loads(text)
    assert (summary["ran_now"], summary["budget_stopped"]) == (4, 4)
    assert rp.total_spend() == pytest.approx(0.40)
    assert len(_calls(fake_runner)) == 4
    out = tmp_path / "scratch" / "replay" / "one"
    rows = rp.read_requests(out / rp.ADMITTED_FILE)
    assert len(rows.requests) == 4
    (store_row,) = rows.stores.values()
    assert store_row["considered"] == 4
    assert store_row["outcomes"][rp.S_BUDGET_STOPPED] == 4
    assert summary["H1"]["eligible_sessions"] == 4
    # A new --out beside the first still sees the $0.40.
    code, text = _replay(tmp_path, capsys, fake_runner, [store], "--sessions", "8", "--max-calls", "5", name="two")
    assert code == 0, text
    assert json.loads(text)["ran_now"] == 0
    assert len(_calls(fake_runner)) == 4
    assert rp.total_spend() <= rp.TOTAL_CEILING_USD


@pytest.mark.timeout(120)
@pytest.mark.parametrize("second", [
    "elsewhere/one", "scratch/replay/one/nested", "scratch/cousin/deep/two", "scratch/replay/two",
])
def test_no_output_folder_gets_a_budget_of_its_own(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fake_runner: Path, second: str,
) -> None:
    """The review's new parent, nested, and cousin folders each got the full
    amount. Killed by: reading the total from anything `--out` can change."""
    store = _many_sessions(tmp_path, 8)
    _configure(fake_runner, cost=0.10)
    assert _replay(tmp_path, capsys, fake_runner, [store], "--sessions", "8", "--max-calls", "5")[0] == 0
    assert len(_calls(fake_runner)) == 4
    code, text = _main(capsys, "replay", "--store", str(store), "--runner", str(fake_runner),
                       "--sessions", "8", "--budget-usd", "0.5", "--max-calls", "5",
                       "--out", str(tmp_path / second))
    assert code == 0, text
    assert json.loads(text)["ran_now"] == 0
    assert len(_calls(fake_runner)) == 4
    assert rp.total_spend() == pytest.approx(0.40)


@pytest.mark.timeout(120)
@pytest.mark.parametrize("cost", [float("nan"), float("inf"), float("-inf"), -1.0, True, "0.01", 10 ** 400])
def test_a_cost_that_is_no_usable_number_charges_the_cap(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fake_runner: Path, cost: object,
) -> None:
    """NaN, negative, and -Infinity ran 16 of 16 in the review. Killed by:
    accepting any of them, in the ledger or in what the kill rule reads."""
    store = _many_sessions(tmp_path, 6)
    _configure(fake_runner, cost=cost)
    code, text = _replay(tmp_path, capsys, fake_runner, [store], "--sessions", "6", "--max-calls", "5")
    assert code == 0, text
    summary = json.loads(text)
    assert (summary["ran_now"], summary["budget_stopped"]) == (4, 2)
    assert rp.total_spend() == pytest.approx(0.40)
    assert summary["H7"]["over_cap"] == 4
    assert summary["H7"]["cost_unusable"] == 4
    assert rp.decide(ab.V_PASS, ab.V_PASS, ab.V_PASS, summary["H7"]["over_cap"]) == rp.DECISION_KILL


@pytest.mark.timeout(120)
def test_a_run_over_its_cap_stops_the_next_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fake_runner: Path,
) -> None:
    """A runner that ignores its cap is only stopped afterward, as the docs
    say. Killed by: charging the cap instead of what it reported."""
    store = _many_sessions(tmp_path, 4)
    _configure(fake_runner, cost=5.0)
    code, text = _replay(tmp_path, capsys, fake_runner, [store], "--sessions", "4", "--max-calls", "5")
    assert code == 0, text
    summary = json.loads(text)
    assert (summary["ran_now"], summary["budget_stopped"]) == (1, 3)
    assert rp.total_spend() == pytest.approx(5.0)
    assert summary["H7"]["over_cap"] == 1
    assert "timeout" in (rp.__doc__ or "")


@pytest.mark.timeout(120)
def test_a_run_a_little_over_its_cap_stays_under_the_ceiling(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fake_runner: Path,
) -> None:
    """$0.13 a run reached $0.52 in the review. Killed by: a start check with
    less room than the overshoot it claims to allow."""
    store = _many_sessions(tmp_path, 6)
    _configure(fake_runner, cost=0.13)
    code, text = _replay(tmp_path, capsys, fake_runner, [store], "--sessions", "6", "--max-calls", "5")
    assert code == 0, text
    summary = json.loads(text)
    assert summary["ran_now"] == 3
    assert summary["spent_usd"] == pytest.approx(0.39)
    assert rp.RUN_RESERVE_USD == 0.13


@pytest.mark.timeout(120)
def test_a_reported_zero_is_bounded_by_max(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fake_runner: Path,
) -> None:
    """A reported $0 admitted 12 of 12 in the review. Killed by: no cap on the
    number of runs, or one a new `--out` starts over."""
    store = _many_sessions(tmp_path, 6)
    _configure(fake_runner, cost=0)
    code, text = _replay(tmp_path, capsys, fake_runner, [store], "--sessions", "6", "--max-calls", "4")
    assert code == 0, text
    summary = json.loads(text)
    assert (summary["ran_now"], summary["budget_stopped"]) == (4, 2)
    assert rp.total_spend() == 0.0
    code, text = _replay(tmp_path, capsys, fake_runner, [store], "--sessions", "6", "--max-calls", "4", name="two")
    assert code == 0, text
    assert json.loads(text)["ran_now"] == 0
    assert len(_calls(fake_runner)) == 4
    assert "`--max`" in (rp.__doc__ or "")


@pytest.mark.timeout(120)
def test_the_dry_run_says_what_the_real_run_can_do(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fake_runner: Path,
) -> None:
    """It said 4 and 12 ran at $0.03 in the review. Killed by: a figure that
    a cheaper run passes."""
    store = _many_sessions(tmp_path, 12)
    _configure(fake_runner, cost=0.03)
    code, text = _replay(tmp_path, capsys, fake_runner, [store], "--sessions", "12", "--max-calls", "40",
                         "--dry-run")
    assert code == 0, text
    plan = json.loads(text)["plan"]
    assert plan["runs_at_the_cap"] == 4
    assert plan["runs_at_most"] == 12
    assert plan["last_run_starts_at_or_under_usd"] == pytest.approx(0.37)
    assert plan["each_run_within_budget_up_to_usd"] == 0.13
    code, text = _replay(tmp_path, capsys, fake_runner, [store], "--sessions", "12", "--max-calls", "40")
    assert code == 0, text
    assert plan["runs_at_the_cap"] <= json.loads(text)["ran_now"] <= plan["runs_at_most"]


def test_the_plan_prints_before_the_first_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fake_runner: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Killed by: printing the plan only after the money is spent."""
    store = _store(tmp_path)
    seen: list[str] = []
    real = rp.wb.execute_run

    def execute_run(*a: Any, **kw: Any) -> Any:
        seen.append(capsys.readouterr().err)
        return real(*a, **kw)

    monkeypatch.setattr(rp.wb, "execute_run", execute_run)
    assert _replay(tmp_path, capsys, fake_runner, [store])[0] == 0
    assert '"claims_target": 30' in seen[0]
    assert "warning" in seen[0]


def _plan_row(i: int, axes: int) -> dict[str, Any]:
    return {"snapshot": SNAP, "request_id": f"r{i:02d}", "axes": ["a", "b"][:axes]}


@pytest.mark.parametrize(("axes", "n", "budget", "max_calls", "at_cap", "at_most", "cap_claims", "most_claims"), [
    (2, 8, 0.5, 40, 4, 8, 24, 48),
    (1, 8, 0.5, 40, 4, 8, 12, 24),
    (2, 8, 0.25, 40, 2, 8, 12, 48),
    (2, 8, 0.10, 40, 0, 0, 0, 0),
    (2, 8, 0.5, 3, 3, 3, 18, 18),
    (2, 2, 0.5, 40, 2, 2, 12, 12),
])
def test_the_plan_counts_runs_and_claims(
    tmp_path: Path, axes: int, n: int, budget: float, max_calls: int, at_cap: int, at_most: int,
    cap_claims: int, most_claims: int,
) -> None:
    """Killed by: counting runs without the reserve, the cap, `--max`, or the
    requests, or claims with no regard to the axes offered."""
    rows = [_plan_row(i, axes) for i in range(n)]
    plan = rp.run_plan(rows, tmp_path, budget=budget, max_calls=max_calls, seed=1)
    assert (plan["runs_at_the_cap"], plan["runs_at_most"]) == (at_cap, at_most)
    assert (plan["claims_at_the_cap"], plan["claims_at_most"]) == (cap_claims, most_claims)
    assert plan["claims_target"] == 30
    assert plan["target_reachable_at_the_cap"] is (cap_claims >= 30)
    assert plan["target_reachable_at_most"] is (most_claims >= 30)


def test_the_plan_warns_that_the_cap_misses_the_target_and_prices_it() -> None:
    """4 runs of 6 claims is 24, under H2's 30. Killed by: no warning, or a
    price that doesn't admit the fifth run."""
    plan = rp.run_plan([_plan_row(i, 2) for i in range(8)], Path("widget"), budget=0.5, max_calls=40, seed=1)
    assert "24" in plan["warning"] and "30" in plan["warning"]
    assert plan["cost_per_run_for_the_target_usd"] == pytest.approx(0.0925)
    assert plan["cost_per_run_for_the_target_usd"] * 4 + rp.RUN_RESERVE_USD <= 0.5 + 1e-9


def test_the_plan_warns_when_no_price_reaches_the_target() -> None:
    """One axis a run is 3 claims. Killed by: pricing a target the requests can't reach."""
    plan = rp.run_plan([_plan_row(i, 1) for i in range(8)], Path("widget"), budget=0.5, max_calls=40, seed=1)
    assert "24" in plan["warning"]
    assert "more sessions or stores" in plan["warning"]
    assert plan["cost_per_run_for_the_target_usd"] is None


def test_the_plan_is_quiet_when_the_cap_reaches_the_target() -> None:
    """Killed by: a warning printed when the target is in reach, even if
    every reply answers one axis."""
    rows = [_plan_row(i, 2) for i in range(12)]
    rp.spend_ledger_path().parent.mkdir(parents=True)
    plan = rp.run_plan(rows, Path("widget"), budget=1.0, max_calls=40, seed=1)
    assert "warning" in plan
    out = str(Path("widget").resolve())
    rp.spend_ledger_path().write_text(
        json.dumps({"out": out, "call": 1, "request_id": "r00", "state": rp.STATE_DONE, "cost_usd": 0.0}) + "\n",
        encoding="utf-8",
    )
    plan = rp.run_plan(rows, Path("widget"), budget=1.0, max_calls=40, seed=1)
    assert plan["requests_left"] == 11
    assert plan["claims_at_the_cap"] == 6 + 9 * 6
    assert plan["claims_at_the_cap_one_axis_each"] == 3 + 9 * 3
    assert plan["target_reachable_at_the_cap"] is True
    assert "warning" not in plan


def test_the_ceiling_is_the_operators_dollar_and_admits_9_runs_at_the_cap() -> None:
    """The operator raised the replay's total to $1.00. Killed by: another
    ceiling, or a default call cap that isn't what the budget admits."""
    assert rp.TOTAL_CEILING_USD == 1.00
    assert rp.calls_admitted_at_the_cap(1.00) == 9
    assert rp.calls_admitted_at_the_cap(0.50) == 4
    assert rp.calls_admitted_at_the_cap(0.10) == 0


def test_at_a_dollar_one_axis_a_run_falls_short_and_the_plan_says_so() -> None:
    """9 runs at one axis give 27 claims, under H2's 30. Killed by: a plan that
    counts only the axes offered."""
    plan = rp.run_plan([_plan_row(i, 2) for i in range(12)], Path("widget"), budget=1.0, max_calls=9, seed=1)
    assert (plan["runs_at_the_cap"], plan["claims_at_the_cap"], plan["claims_at_the_cap_one_axis_each"]) == (9, 54, 27)
    assert plan["target_reachable_at_the_cap"] is True
    assert "27 to 54" in plan["warning"]
    assert "one axis" in plan["warning"]


@pytest.mark.timeout(120)
def test_the_dry_run_prints_both_limits_and_defaults_the_call_cap_to_the_budget(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fake_runner: Path,
) -> None:
    """`--max` once set both, so capping calls at 15 cut the prompts to 15.
    Killed by: one flag for both, or a default call cap not drawn from the budget."""
    store = _many_sessions(tmp_path, 12)
    code, text = _replay(tmp_path, capsys, fake_runner, [store], "--sessions", "12", "--budget-usd", "1.00",
                         "--dry-run")
    assert code == 0, text
    out = json.loads(text)
    assert (out["max_calls"], out["max_prompts"], out["plan"]["max_calls"]) == (9, 40, 9)
    assert out["plan"]["runs_at_most"] == 9
    code, text = _replay(tmp_path, capsys, fake_runner, [store], "--sessions", "12", "--budget-usd", "1.00",
                         "--max-calls", "15", "--dry-run", name="two")
    assert code == 0, text
    assert (json.loads(text)["max_calls"], json.loads(text)["max_prompts"]) == (15, 40)


def test_the_plan_reads_what_the_ledger_already_spent() -> None:
    """Killed by: planning as if the ledger were empty."""
    assert rp.admit_run("other", "x", 0.5, 40) == 1
    assert rp.admit_run("other", "y", 0.5, 40) == 2
    plan = rp.run_plan([_plan_row(i, 2) for i in range(8)], Path("widget"), budget=0.5, max_calls=4, seed=1)
    assert plan["spent_usd"] == pytest.approx(2 * rp.RUN_RESERVE_USD)
    assert plan["runs_at_the_cap"] == 2
    assert plan["runs_at_most"] == 2


def test_no_run_starts_unless_the_rest_covers_its_cap(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fake_runner: Path,
) -> None:
    """Killed by: a check that lets a run start on less than its own cap."""
    store = _store(tmp_path)
    code, text = _replay(tmp_path, capsys, fake_runner, [store], "--budget-usd", "0.10")
    assert code == 0, text
    assert json.loads(text)["ran_now"] == 0
    assert _calls(fake_runner) == []
    assert rp.RUN_RESERVE_USD >= float(wb.RUNNER_BUDGET_USD)


@pytest.mark.parametrize("how", ["path", "link", "bytes"])
def test_a_store_listed_twice_is_refused_before_any_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fake_runner: Path, how: str,
) -> None:
    """Listing one store three times made 6 real calls, charged 2 in the
    ledger, and then exit 1 with no output, in the review. Killed by: checking
    only one of a path, a link to it, or a second file of the same bytes."""
    store = _many_sessions(tmp_path, 2)
    twin = tmp_path / "twin" / "memory.db"
    twin.parent.mkdir()
    if how == "bytes":
        shutil.copyfile(store, twin)
        _mark_copy(twin)
    elif how == "link":
        twin.symlink_to(store)
        _mark_copy(twin)
    else:
        twin = store
    code, text = _replay(tmp_path, capsys, fake_runner, [store, twin], "--sessions", "2")
    assert code == 1
    assert "are one store" in text
    assert _calls(fake_runner) == []
    assert rp.total_spend() == 0.0


def test_requests_that_do_not_pair_up_are_refused_before_any_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fake_runner: Path,
) -> None:
    """A saved requests file holding one snapshot twice. Killed by: finding
    that out only after the money is spent, at `run`."""
    store = _store(tmp_path)
    out = tmp_path / "scratch" / "replay" / "one"
    out.mkdir(parents=True)
    rows = _requests(tmp_path, capsys, store)
    store_row = next(r for r in rows if r["kind"] == rp.ROW_STORE)
    (out / rp.REQUESTS_FILE).write_text(
        "".join(json.dumps(r) + "\n" for r in [store_row, *rows]), encoding="utf-8",
    )
    code, text = _replay(tmp_path, capsys, fake_runner, [store])
    assert code == 1
    assert "two store rows" in text
    assert _calls(fake_runner) == []


def test_a_repeated_request_runs_once(tmp_path: Path) -> None:
    """Killed by: a run not marking its request done, so a repeated row runs again."""
    calls: list[str] = []
    row = {"snapshot": SNAP, "request_id": "r1", "session_id": "s1", "topic": "t", "axes": ["a"],
           "anchor_ids": ["b"], "input": "widget"}

    def execute_run(prompt: Any, *, cli: str, record: dict[str, object], run: Any) -> Any:
        calls.append(prompt.text)
        record["cost_usd"] = 0.10
        return subprocess.CompletedProcess[str]([], 0, json.dumps({"total_cost_usd": 0.02}), "")

    real = rp.wb.execute_run
    rp.wb.execute_run = execute_run  # type: ignore[assignment]
    try:
        assert rp.run_requests([row, row], runner="widget", out=tmp_path, budget=0.5, seed=1, max_calls=40) == (1, 0)
    finally:
        rp.wb.execute_run = real  # type: ignore[assignment]
    assert len(calls) == 1
    assert rp.total_spend() == pytest.approx(0.02)


def test_the_ledger_charges_every_call_of_one_request() -> None:
    """Killed by: keying a charge by request id, so a later call overwrites an
    earlier one: 6 real calls were charged as 2 in the review."""
    rows = [
        {"out": "a", "call": 1, "request_id": "r", "state": rp.STATE_STARTED},
        {"out": "a", "call": 1, "request_id": "r", "state": rp.STATE_DONE, "cost_usd": 0.10},
        {"out": "a", "call": 2, "request_id": "r", "state": rp.STATE_STARTED},
        {"out": "a", "call": 2, "request_id": "r", "state": rp.STATE_DONE, "cost_usd": 0.02},
        {"out": "a", "call": 3, "request_id": "r", "state": rp.STATE_STARTED},
    ]
    calls = rp.ledger_calls(rows)
    assert sorted(c.charge or 0.0 for c in calls.values()) == [0.0, 0.02, 0.10]
    done, open_calls = rp.ledger_state(rows)
    assert done == {"r": pytest.approx(0.12)}
    assert open_calls == {}


def test_a_dead_run_is_closed_at_the_reserve_and_an_open_one_counts_at_it(tmp_path: Path) -> None:
    """The review: a resumed run was charged its $0.10 cap, below the $0.13
    the ceiling allows a run. Killed by: charging the cap for a run nobody
    saw finish, open or closed on a resume."""
    out = tmp_path / "out"
    out.mkdir()
    rp.spend_ledger_path().parent.mkdir(parents=True, exist_ok=True)
    assert rp.admit_run(str(out.resolve()), "dead", 1.0, 40) == 1
    assert rp.total_spend() == pytest.approx(rp.RUN_RESERVE_USD)
    assert rp.run_requests([], runner="widget", out=out, budget=1.0, seed=1, max_calls=40) == (0, 0)
    (row,) = [r for r in rp.read_ledger(rp.spend_ledger_path()) if r.get("state") == rp.STATE_DONE]
    assert row["cost_usd"] == pytest.approx(rp.RUN_RESERVE_USD)
    assert rp.total_spend() == pytest.approx(rp.RUN_RESERVE_USD)


def test_a_late_row_for_a_call_can_only_raise_its_charge() -> None:
    """A resume closes a dead call at the cap, and a reply the dead harness
    wrote can close it again. Killed by: the last row winning."""
    rows = [
        {"out": "a", "call": 1, "request_id": "r", "state": rp.STATE_STARTED},
        {"out": "a", "call": 1, "request_id": "r", "state": rp.STATE_DONE, "cost_usd": 0.10},
        {"out": "a", "call": 1, "request_id": "r", "state": rp.STATE_DONE, "cost_usd": 0.01},
    ]
    assert rp.ledger_calls(rows)[1].charge == pytest.approx(0.10)


def test_the_ledger_total_counts_every_call(tmp_path: Path) -> None:
    """Killed by: a total that dedupes on the request id or the output folder."""
    rp.spend_ledger_path().parent.mkdir(parents=True)
    rows = [
        {"out": "a", "call": 1, "request_id": "r", "state": rp.STATE_DONE, "cost_usd": 0.10},
        {"out": "b", "call": 2, "request_id": "r", "state": rp.STATE_DONE, "cost_usd": 0.10},
        {"out": "a", "call": 3, "request_id": "r", "state": rp.STATE_DONE, "cost_usd": 0.05},
        {"out": "a", "call": 4, "request_id": "q", "state": rp.STATE_STARTED},
        {"out": "a", "request_id": "no-number", "state": rp.STATE_STARTED},
        {"out": "a", "request_id": "no-number", "state": rp.STATE_STARTED},
    ]
    rp.spend_ledger_path().write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert rp.total_spend() == pytest.approx(0.10 + 0.10 + 0.05 + 3 * rp.RUN_RESERVE_USD)


def test_call_numbers_rise_and_a_closed_call_keeps_its_number() -> None:
    """Killed by: a reused number, which would close the wrong call."""
    first = rp.admit_run("a", "r1", 0.5, 40)
    second = rp.admit_run("a", "r2", 0.5, 40)
    assert (first, second) == (1, 2)
    rp.close_run("a", "r1", first, 0.01)
    assert rp.admit_run("a", "r3", 0.5, 40) == 3
    assert rp.ledger_calls(rp.read_ledger(rp.spend_ledger_path()))[1].charge == 0.01


def test_the_lock_spans_the_check_and_the_claim(monkeypatch: pytest.MonkeyPatch) -> None:
    """8 parallel siblings reached $0.54 in the review. Killed by: checking
    the total or writing the claim outside the lock."""
    held: dict[str, bool] = {}

    def lock_is_held() -> bool:
        fd = os.open(rp.spend_ledger_path().with_name(rp.SPEND_LOCK_FILE), os.O_RDWR)
        try:
            file_lock.try_lock_exclusive(fd)
        except BlockingIOError:
            return True
        finally:
            os.close(fd)
        return False

    real_total, real_append = rp.total_spend, rp._append_line  # pyright: ignore[reportPrivateUsage]

    def total() -> float:
        held["check"] = lock_is_held()
        return real_total()

    def append(path: Path, row: Mapping[str, Any]) -> None:
        held["claim"] = lock_is_held()
        real_append(path, row)

    monkeypatch.setattr(rp, "total_spend", total)
    monkeypatch.setattr(rp, "_append_line", append)
    assert rp.admit_run("widget-out", "widget-request", 0.5, 40) == 1
    assert held == {"check": True, "claim": True}


@pytest.mark.timeout(120)
def test_parallel_replays_share_the_ceiling(tmp_path: Path) -> None:
    """Real processes on one ledger: open runs count at the $0.13 reserve, so
    $0.50 admits 3. Killed by: a check-then-claim that can interleave."""
    code = (
        "import sys\n"
        "from benchmarks import wonder_auto_replay as rp\n"
        "print(rp.admit_run(sys.argv[1], 'widget-request', 0.5, 40) is not None)\n"
    )

    def one(i: int) -> str:
        return subprocess.run(  # noqa: S603 - this interpreter on the code above
            [sys.executable, "-c", code, f"widget-out-{i}"], capture_output=True, text=True,
            encoding="utf-8", cwd=rp.REPO_ROOT, timeout=100, check=True,
        ).stdout.strip()

    with ThreadPoolExecutor(8) as pool:
        answers = list(pool.map(one, range(8)))
    assert answers.count("True") == 3
    assert rp.total_spend() == pytest.approx(3 * rp.RUN_RESERVE_USD)


def test_one_replay_runs_at_a_time_whatever_its_output_folder(tmp_path: Path) -> None:
    """In a probe, 8 parallel replays of a runner reporting $5.00 charged $20.
    Killed by: letting a second replay start a runner, or closing a live
    replay's open runs as dead."""
    out = tmp_path / "out"
    out.mkdir()
    rp.spend_ledger_path().parent.mkdir(parents=True)
    fd = os.open(rp.spend_ledger_path().with_name(rp.REPLAY_LOCK_FILE), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        file_lock.try_lock_exclusive(fd)
        with pytest.raises(rp.InputError, match="another replay"):
            rp.run_requests([], runner="widget", out=out, budget=0.5, seed=1, max_calls=40)
    finally:
        os.close(fd)
    assert rp.run_requests([], runner="widget", out=out, budget=0.5, seed=1, max_calls=40) == (0, 0)


@pytest.mark.timeout(120)
@pytest.mark.parametrize("output", [
    b'{"total_cost_usd": 1' + b"0" * 5000 + b"}",
    b"\xff\xfe not text",
    b"[" * 100000,
], ids=["5001 digits", "undecodable", "deep nesting"])
def test_output_nothing_can_read_charges_the_cap_and_the_replay_goes_on(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], output: bytes,
) -> None:
    """Both ended the replay with rc 1 in the review, after spending. Killed
    by: a decode or parse error reaching the harness, or charging it as free."""
    store = _many_sessions(tmp_path, 2)
    folder = tmp_path / "bad"
    folder.mkdir()
    (folder / "output").write_bytes(output)
    runner = folder / "hostcli"
    runner.write_text(f'#!/bin/sh\ncat > /dev/null\ncat "{folder / "output"}"\n', encoding="utf-8")
    runner.chmod(0o700)
    code, text = _replay(tmp_path, capsys, runner, [store], "--sessions", "2")
    assert code == 0, text
    assert json.loads(text)["ran_now"] == 2
    assert rp.total_spend() == pytest.approx(2 * float(wb.RUNNER_BUDGET_USD))


SLOW_RUNNER = """#!/bin/sh
echo "$$ $PPID" > "$(dirname "$0")/pids"
cat > /dev/null
sleep 60
"""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_for(ready: Any, what: str) -> None:
    for _ in range(600):
        if ready():
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


def _replay_in_a_process(store: Path, runner: Path, out: Path) -> "subprocess.CompletedProcess[str]":
    """The harness in its own process, as a person runs it. It ends when a test signals it."""
    return subprocess.run(  # noqa: S603 - this interpreter on the harness
        [sys.executable, "-m", "benchmarks.wonder_auto_replay", "replay", "--store", str(store),
         "--runner", str(runner), "--sessions", "2", "--out", str(out)],
        capture_output=True, text=True, encoding="utf-8", cwd=rp.REPO_ROOT, timeout=100, check=False,
    )


def _slow_runner(tmp_path: Path) -> Path:
    runner = tmp_path / "slow" / "hostcli"
    runner.parent.mkdir()
    runner.write_text(SLOW_RUNNER, encoding="utf-8")
    runner.chmod(0o700)
    return runner


@pytest.mark.timeout(120)
@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP, signal.SIGINT, signal.SIGQUIT, signal.SIGUSR1])
def test_a_signal_to_the_replay_ends_the_runner(tmp_path: Path, sig: signal.Signals) -> None:
    """Killed by: Python's default for the signal, which ends the harness and
    leaves the runner alive beside the next replay."""
    store = _many_sessions(tmp_path, 2)
    runner = _slow_runner(tmp_path)
    pids = runner.parent / "pids"
    with ThreadPoolExecutor(1) as pool:
        run = pool.submit(_replay_in_a_process, store, runner, tmp_path / "out")
        _wait_for(lambda: pids.exists() and len(pids.read_text().split()) == 2, "the runner to start")
        pid, harness = (int(x) for x in pids.read_text().split())
        os.kill(harness, sig)
        done = run.result(timeout=100)
    assert done.returncode != 0
    assert done.returncode in (128 + sig, 1, -sig)
    _wait_for(lambda: not _alive(pid), "the runner to end")
    assert rp.total_spend() == pytest.approx(rp.RUN_RESERVE_USD)


@pytest.mark.timeout(120)
def test_sigkill_of_the_replay_leaves_the_lock_with_its_runner(tmp_path: Path) -> None:
    """Nothing can end the runner after SIGKILL of the harness, as the docs
    say, but a second replay ran beside it in the review. Killed by: not passing
    the lock to the runner."""
    store = _many_sessions(tmp_path, 2)
    runner = _slow_runner(tmp_path)
    pids = runner.parent / "pids"
    with ThreadPoolExecutor(1) as pool:
        run = pool.submit(_replay_in_a_process, store, runner, tmp_path / "out")
        _wait_for(lambda: pids.exists() and len(pids.read_text().split()) == 2, "the runner to start")
        pid, harness = (int(x) for x in pids.read_text().split())
        os.kill(harness, signal.SIGKILL)
        run.result(timeout=100)
    try:
        assert _alive(pid)
        with pytest.raises(rp.InputError, match="still alive"), rp.replay_lock():
            pass
    finally:
        os.killpg(pid, signal.SIGKILL)

    def free() -> bool:
        # The runner's `sleep` holds the lock too, and can outlast the shell
        # by a moment under load.
        try:
            with rp.replay_lock():
                return True
        except rp.InputError:
            return False

    _wait_for(free, "the lock to come free once the runner's group is gone")
    assert "SIGKILL" in (rp.__doc__ or "")


def test_a_resume_closes_only_its_own_open_runs_and_repeats_no_reply(tmp_path: Path) -> None:
    """Killed by: closing another folder's open run, which may be alive, or
    a second reply for a run that died between its reply and its close."""
    out = tmp_path / "out"
    out.mkdir()
    assert rp.admit_run(str(out.resolve()), "mine-answered", 0.5, 40) == 1
    assert rp.admit_run(str(out.resolve()), "mine-silent", 0.5, 40) == 2
    assert rp.admit_run("another-folder", "theirs", 0.5, 40) == 3
    replies = out / rp.REPLIES_FILE
    replies.write_text(json.dumps({"request_id": "mine-answered", "stdout": "", "returncode": 0}) + "\n",
                       encoding="utf-8")
    assert rp.run_requests([], runner="widget", out=out, budget=0.5, seed=1, max_calls=40) == (0, 0)
    rows = [json.loads(x) for x in replies.read_text(encoding="utf-8").splitlines()]
    assert sorted(r["request_id"] for r in rows) == ["mine-answered", "mine-silent"]
    ledger = rp.read_ledger(rp.spend_ledger_path())
    assert not [r for r in ledger if r["out"] == "another-folder" and r["state"] == rp.STATE_DONE]
    assert rp.total_spend() == pytest.approx(3 * rp.RUN_RESERVE_USD)


@pytest.mark.parametrize("cost", [float("nan"), float("-inf"), -1.0, True, None, "0.01"])
def test_the_ledger_charges_a_bad_row_at_the_cap(cost: object) -> None:
    """Killed by: reading a bad cost in the ledger as free, or as a refund."""
    done, open_calls = rp.ledger_state([{"request_id": "r", "state": rp.STATE_DONE, "cost_usd": cost}])
    assert done == {"r": float(wb.RUNNER_BUDGET_USD)}
    assert open_calls == {}


@pytest.mark.parametrize("cost", [float("nan"), float("inf"), -0.5, "0.01"])
def test_the_kill_rule_counts_a_cost_no_one_can_read(cost: object) -> None:
    """NaN compares false, so it passed the over-cap check. Killed by: taking
    it or a negative cost as a run within its cap."""
    h7 = rp.h7_cost([_store_report(costs=[0.03, cost])])
    assert h7["over_cap"] == 1
    assert h7["total_usd"] == pytest.approx(0.03 + float(wb.RUNNER_BUDGET_USD))


@pytest.mark.timeout(120)
def test_a_smaller_budget_stops_sooner(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fake_runner: Path,
) -> None:
    """Killed by: reading the ceiling instead of `--budget-usd`, or a run
    with no reported cost counted as free."""
    store = _many_sessions(tmp_path, 4)
    _configure(fake_runner)
    code, text = _replay(tmp_path, capsys, fake_runner, [store], "--sessions", "4", "--max-calls", "5", "--budget-usd", "0.25")
    assert code == 0, text
    assert json.loads(text)["ran_now"] == 2
    assert rp.total_spend() == pytest.approx(0.20)


@pytest.mark.timeout(120)
def test_a_killed_harness_resumes_without_paying_twice(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fake_runner: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Killed by: rerunning a finished request, rerunning the request the
    harness died inside, or counting that one as free."""
    store = _many_sessions(tmp_path, 2)
    out = tmp_path / "scratch" / "replay" / "one"
    real = rp.wb.execute_run
    seen = [0]

    def dies_on_second(*a: Any, **kw: Any) -> Any:
        seen[0] += 1
        if seen[0] == 2:
            raise KeyboardInterrupt
        return real(*a, **kw)

    monkeypatch.setattr(rp.wb, "execute_run", dies_on_second)
    with pytest.raises(KeyboardInterrupt):
        _replay(tmp_path, capsys, fake_runner, [store], "--sessions", "2", "--max-calls", "5")
    monkeypatch.setattr(rp.wb, "execute_run", real)
    assert len(_calls(fake_runner)) == 1
    assert rp.total_spend() == pytest.approx(0.03 + rp.RUN_RESERVE_USD)
    code, text = _replay(tmp_path, capsys, fake_runner, [store], "--sessions", "2", "--max-calls", "5")
    assert code == 0, text
    assert json.loads(text)["ran_now"] == 0
    assert len(_calls(fake_runner)) == 1
    assert rp.total_spend() == pytest.approx(0.03 + rp.RUN_RESERVE_USD)
    records = json.loads((out / rp.RUN_DIR / rp.REPORT_FILE).read_text(encoding="utf-8"))
    assert sum(records["stores"][0]["run_outcomes"].values()) == 2
    assert records["stores"][0]["run_outcomes"][wb.OUTCOME_RUN_FAILED] == 1


@pytest.mark.timeout(120)
def test_a_failing_runner_is_a_recorded_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fake_runner: Path,
) -> None:
    """Killed by: aborting the replay on one failed run, or charging it nothing."""
    store = _many_sessions(tmp_path, 2)
    _configure(fake_runner, cost=0.03, mode="exit")
    code, text = _replay(tmp_path, capsys, fake_runner, [store], "--sessions", "2", "--max-calls", "5")
    assert code == 0, text
    summary = json.loads(text)
    assert summary["H1"]["run_outcomes"][wb.OUTCOME_RUN_FAILED] == 1
    assert summary["spent_usd"] == pytest.approx(0.10 + 0.03)


@pytest.mark.timeout(120)
def test_a_store_whose_every_prompt_the_pass_read_still_reports_its_claims(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fake_runner: Path,
) -> None:
    """Killed by: aborting after the money is spent because a small store has
    no prompt left to sample, or a key `score` then refuses."""
    store = _many_sessions(tmp_path, 2)
    code, text = _replay(tmp_path, capsys, fake_runner, [store], "--sessions", "2")
    assert code == 0, text
    summary = json.loads(text)
    assert summary["retrieval_rows"] == 0
    assert summary["claim_rows"] == 2
    run = tmp_path / "scratch" / "replay" / "one" / rp.RUN_DIR
    key = rp.read_key(run / rp.KEY_FILE)
    assert len(key.claims) == 2
    assert key.retrieval.items == ()


def test_replay_refuses_a_missing_runner_and_an_output_in_the_repo(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fake_runner: Path,
) -> None:
    """Killed by: starting with no runner, or writing results into the repository."""
    store = _store(tmp_path)
    code, text = _main(capsys, "replay", "--store", str(store), "--runner", str(tmp_path / "gone"),
                       "--out", str(tmp_path / "scratch" / "r"))
    assert code == 1
    assert "no runner" in text
    inside = rp.REPO_ROOT.joinpath(f"replay-results-{tmp_path.name}")  # path-filter: none
    code, text = _main(capsys, "replay", "--store", str(store), "--runner", str(fake_runner),
                       "--out", str(inside))
    assert code == 1
    assert "inside the repository" in text
    assert not inside.exists()


def test_prepare_refuses_an_output_in_the_repo(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Killed by: copying real stores into the repository."""
    root = tmp_path / "projects"
    _real_project(root, "widget-one")
    inside = rp.REPO_ROOT.joinpath(f"replay-stores-{tmp_path.name}")  # path-filter: none
    code, text = _main(capsys, "prepare", "--projects-root", str(root), "--out", str(inside),
                       "--exclude", "widget-no")
    assert code == 1
    assert "inside the repository" in text
    assert not inside.exists()


def test_the_budget_order_shuffles_the_stores_with_the_seed() -> None:
    """Killed by: spending in snapshot order, which ties who gets measured to a hash's sort order."""
    rows = [{"snapshot": s, "request_id": s} for s in "abcdefgh"]
    order = [r["snapshot"] for r in rp.budget_order(rows, 1811)]
    assert order != sorted(order)
    assert order == [r["snapshot"] for r in rp.budget_order(rows, 1811)]
    assert order != [r["snapshot"] for r in rp.budget_order(rows, 1812)]


def test_a_store_the_budget_left_with_nothing_drops_out() -> None:
    """Killed by: keeping a store no run reached, whose retrieval would count
    only zeros against a pass that never ran there. A store with no request
    of its own stays: there the pass ran and found nothing to ask."""
    stores = [
        {"kind": rp.ROW_STORE, "snapshot": s, "considered": n, "outcomes": {rp.S_REQUEST: n}}
        for s, n in (("kept", 2), ("stopped", 1), ("empty", 0))
    ]
    rows = [
        {"kind": rp.ROW_REQUEST, "snapshot": "kept", "request_id": "k1"},
        {"kind": rp.ROW_REQUEST, "snapshot": "kept", "request_id": "k2"},
        {"kind": rp.ROW_REQUEST, "snapshot": "stopped", "request_id": "s1"},
    ]
    out = rp.admitted_rows(stores, rows, {"k1"})
    assert [r["snapshot"] for r in out if r["kind"] == rp.ROW_STORE] == ["kept", "empty"]
    assert [r["request_id"] for r in out if r["kind"] == rp.ROW_REQUEST] == ["k1"]
    kept = next(r for r in out if r["snapshot"] == "kept" and r["kind"] == rp.ROW_STORE)
    assert kept["considered"] == 1
    assert kept["outcomes"] == {rp.S_BUDGET_STOPPED: 1, rp.S_REQUEST: 1}


def test_the_pinned_environment_covers_the_user_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Killed by: leaving `AELFRICE_USER_DB` on the machine's real user store
    while the writer's code runs on a copy, or not restoring it."""
    monkeypatch.setenv("AELFRICE_USER_DB", "/widget/real/user.db")
    db = tmp_path / "copy" / "memory.db"
    with rp.pinned_store_env(db):
        assert os.environ["AELFRICE_DB"] == str(db)
        assert Path(os.environ["AELFRICE_USER_DB"]).is_relative_to(db.parent)
        assert os.environ["AELFRICE_FEED_LOG"] == "0"
    assert os.environ["AELFRICE_USER_DB"] == "/widget/real/user.db"


def test_budget_order_goes_round_the_stores_newest_first() -> None:
    """Killed by: spending the budget on one store, or on its oldest sessions."""
    rows = [{"snapshot": s, "request_id": f"{s}{i}"} for s in ("a", "b") for i in range(3)]
    order = [r["request_id"] for r in rp.budget_order(rows, 1811)]
    assert {order[0][0], order[1][0]} == {"a", "b"}
    assert order[0][1] == "2" and order[1][1] == "2"
    assert sorted(order) == sorted(r["request_id"] for r in rows)


def test_replay_skips_a_store_with_no_transcript(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fake_runner: Path,
) -> None:
    """Killed by: one store with no transcript aborting a whole replay."""
    store = _store(tmp_path)
    bare = tmp_path / "bare" / "memory.db"
    bare.parent.mkdir()
    MemoryStore(str(bare)).close()
    _mark_copy(bare)
    code, text = _replay(tmp_path, capsys, fake_runner, [store, bare], "--dry-run")
    assert code == 0, text
    assert json.loads(text)["stores"] == 1


@pytest.mark.timeout(120)
def test_replay_output_goes_through_blind_grading_to_a_score(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fake_runner: Path,
) -> None:
    """The whole path a paid run takes, with a fake runner and scripted labels.

    Killed by: a grader packet that shows an arm, an axis, a session, a
    belief id, or a cost; a claims packet out before the retrieval labels;
    or a replay output `claims` and `score` can't read."""
    store = _store(tmp_path)
    assert _replay(tmp_path, capsys, fake_runner, [store])[0] == 0
    run = tmp_path / "scratch" / "replay" / "one" / rp.RUN_DIR
    assert not (run / rp.CLAIMS_DIR).exists()
    retrieval = _rows(run / rp.RETRIEVAL_DIR / rp.PACKET_FILE)
    assert {frozenset(r) for r in retrieval} == {frozenset({"id", "prompt", "belief"})}
    assert [r["id"] for r in retrieval] == sorted(r["id"] for r in retrieval)
    a, c = _release(tmp_path, capsys, run)
    claims = _rows(run / rp.CLAIMS_DIR / rp.PACKET_FILE)
    assert {frozenset(r) for r in claims} == {frozenset({"id", "claim", "notes"})}
    assert sorted(p.name for p in (run / rp.CLAIMS_DIR).iterdir()) == [rp.INSTRUCTIONS_FILE, rp.PACKET_FILE]
    assert "phantom" not in (run / rp.CLAIMS_DIR / rp.INSTRUCTIONS_FILE).read_text(encoding="utf-8").lower()
    ids = [r["id"] for r in claims]
    ca, cc = _scores(tmp_path / "ca.jsonl", ids), _scores(tmp_path / "cc.jsonl", ids)
    code, text = _main(capsys, "score", "--run", str(run), "--labels", str(a), str(c),
                       "--claim-labels", str(ca), str(cc))
    assert code == 0, text
    report = json.loads(text)
    assert report["H7"]["runs"] == 2
    assert report["H1"]["runs_with_claim"] == 2
    assert report["decision"] in (rp.DECISION_KILL, rp.DECISION_OPEN, rp.DECISION_PASS)
    assert MARK not in text
