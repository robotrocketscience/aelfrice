"""Tests for the #288 phase-1a rebuild_log on the UserPromptSubmit path.

Phase-1a originally instrumented only `rebuild_v14` (the PreCompact
call site). The UserPromptSubmit hook calls `search_for_prompt`
directly, so the high-frequency rebuild path produced no log rows
and phase-1b operator-week data could not accumulate. These tests
cover the UPS-side wiring: schema parity with the PreCompact log,
dedup-drop visibility, env opt-out, TOML opt-out, and fail-soft
behaviour when the log path can't be derived.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from aelfrice.context_rebuilder import (
    REBUILD_LOG_ENV,
    record_user_prompt_submit_log,
)
from aelfrice.hook import user_prompt_submit
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
from aelfrice.store import MemoryStore


# ---- helpers -----------------------------------------------------------


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
        created_at="2026-04-28T00:00:00Z",
        last_retrieved_at=None,
    )


def _seed_db(db_path: Path, beliefs: list[Belief]) -> None:
    store = MemoryStore(str(db_path))
    try:
        for b in beliefs:
            store.insert_belief(b)
    finally:
        store.close()


def _payload(prompt: str, session_id: str = "ups-sess-1") -> str:
    return json.dumps(
        {
            "session_id": session_id,
            "transcript_path": "/dev/null",
            "cwd": "/tmp",
            "hook_event_name": "UserPromptSubmit",
            "prompt": prompt,
        }
    )


def _read_log(log_path: Path) -> list[dict[str, object]]:
    return [
        json.loads(ln)
        for ln in log_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]


# ---- end-to-end through the hook entry point ---------------------------


def test_user_prompt_submit_writes_rebuild_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "the kitchen is full of bananas")])
    monkeypatch.setenv("AELFRICE_DB", str(db))

    sout = io.StringIO()
    rc = user_prompt_submit(
        stdin=io.StringIO(_payload("are there bananas in the kitchen", "ups-sess-1")),
        stdout=sout,
        stderr=io.StringIO(),
    )
    assert rc == 0
    assert sout.getvalue() != ""

    log_path = tmp_path / "rebuild_logs" / "ups-sess-1.jsonl"
    assert log_path.exists(), (
        "UPS hook must emit a rebuild_log row when retrieval returns "
        "any candidate"
    )
    rows = _read_log(log_path)
    assert len(rows) == 1
    rec = rows[0]
    assert rec["session_id"] == "ups-sess-1"
    assert isinstance(rec["ts"], str) and rec["ts"].endswith("Z")
    # Schema parity with the PreCompact rebuild_log: same input/
    # candidates/pack_summary keys with the same `_empty_scores`
    # block per candidate.
    #
    # `scored_query` joined the set in #1405. Kept as an exact-equality
    # pin rather than a subset check: this is the row both replay
    # consumers read, and a key appearing or vanishing unnoticed is the
    # class of defect #1405 documents — `extracted_query` matched neither
    # production path for the whole life of the field (introduced
    # 2026-04-28, UPS path instrumented 2026-05-02 in v1.6.0).
    assert set(rec["input"]) == {
        "recent_turns_hash", "n_recent_turns",
        "extracted_query", "extracted_entities", "extracted_intent",
        "scored_query",
    }
    assert rec["input"]["n_recent_turns"] == 1
    assert isinstance(rec["candidates"], list)
    assert len(rec["candidates"]) >= 1
    cand = rec["candidates"][0]
    assert set(cand["scores"]) == {
        "bm25", "posterior_mean", "reranker", "final",
    }
    assert cand["decision"] == "packed"
    assert cand["reason"] is None
    assert rec["pack_summary"]["n_candidates"] >= 1
    assert rec["pack_summary"]["n_packed"] >= 1


def test_user_prompt_submit_no_log_when_no_hits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No retrieval hits means no candidate set, hence no row —
    same contract as `rebuild_v14` on an empty store."""
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "elephants are large")])
    monkeypatch.setenv("AELFRICE_DB", str(db))

    rc = user_prompt_submit(
        stdin=io.StringIO(_payload("dogs", "no-hit-sess")),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    assert rc == 0
    log_path = tmp_path / "rebuild_logs" / "no-hit-sess.jsonl"
    assert not log_path.exists()


def test_user_prompt_submit_env_opt_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "the kitchen is full of bananas")])
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.setenv(REBUILD_LOG_ENV, "0")

    rc = user_prompt_submit(
        stdin=io.StringIO(_payload("bananas", "opt-out")),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    assert rc == 0
    log_path = tmp_path / "rebuild_logs" / "opt-out.jsonl"
    assert not log_path.exists()


def test_user_prompt_submit_toml_opt_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "the kitchen is full of bananas")])
    monkeypatch.setenv("AELFRICE_DB", str(db))
    cfg = tmp_path / ".aelfrice.toml"
    cfg.write_text(
        "[rebuild_log]\nenabled = false\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    rc = user_prompt_submit(
        stdin=io.StringIO(_payload("bananas", "toml-off")),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    assert rc == 0
    log_path = tmp_path / "rebuild_logs" / "toml-off.jsonl"
    assert not log_path.exists()


def test_user_prompt_submit_no_log_when_session_id_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The on-disk JSONL is keyed by session_id; without it there's
    no file path to write to. Drop silently rather than fabricating
    a session id."""
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "the kitchen is full of bananas")])
    monkeypatch.setenv("AELFRICE_DB", str(db))
    payload = json.dumps(
        {
            # no session_id field
            "transcript_path": "/dev/null",
            "cwd": "/tmp",
            "hook_event_name": "UserPromptSubmit",
            "prompt": "bananas",
        }
    )

    rc = user_prompt_submit(
        stdin=io.StringIO(payload),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    assert rc == 0
    log_dir = tmp_path / "rebuild_logs"
    assert not log_dir.exists() or not any(log_dir.iterdir())


# ---- direct unit coverage of the helper --------------------------------


def test_record_user_prompt_submit_log_marks_dedup_drops(
    tmp_path: Path,
) -> None:
    """Pre-dedup hits with duplicate content collapse to one
    `packed` survivor; the dropped duplicates carry
    `content_hash_collision_with:<survivor>` as their reason."""
    log_path = tmp_path / "rebuild_logs" / "dedup.jsonl"
    pre = [
        _mk("A", "same content"),
        _mk("B", "same content"),  # collides with A
        _mk("C", "different content"),
    ]
    post = [pre[0], pre[2]]  # B dropped by dedup
    record_user_prompt_submit_log(
        prompt="anything",
        session_id="dedup",
        hits_pre_dedup=pre,
        hits_post_dedup=post,
        log_path=log_path,
        enabled=True,
        stderr=io.StringIO(),
    )
    rows = _read_log(log_path)
    assert len(rows) == 1
    rec = rows[0]
    assert rec["pack_summary"]["n_candidates"] == 3
    assert rec["pack_summary"]["n_packed"] == 2
    assert rec["pack_summary"]["n_dropped_by_dedup"] == 1
    by_id = {c["belief_id"]: c for c in rec["candidates"]}
    assert by_id["A"]["decision"] == "packed"
    assert by_id["B"]["decision"] == "dropped"
    assert by_id["B"]["reason"] == "content_hash_collision_with:A"
    assert by_id["C"]["decision"] == "packed"


def test_record_user_prompt_submit_log_lock_level_passthrough(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "rebuild_logs" / "locks.jsonl"
    pre = [
        _mk("L0", "x", lock_level=LOCK_USER, locked_at="2026-04-28T00:00:00Z"),
        _mk("L1", "y"),
    ]
    record_user_prompt_submit_log(
        prompt="q",
        session_id="locks",
        hits_pre_dedup=pre,
        hits_post_dedup=pre,
        log_path=log_path,
        enabled=True,
        stderr=io.StringIO(),
    )
    rec = _read_log(log_path)[0]
    by_id = {c["belief_id"]: c for c in rec["candidates"]}
    assert by_id["L0"]["lock_level"] == "user"
    assert by_id["L1"]["lock_level"] == "none"


def test_record_user_prompt_submit_log_disabled_writes_nothing(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "rebuild_logs" / "off.jsonl"
    record_user_prompt_submit_log(
        prompt="q",
        session_id="off",
        hits_pre_dedup=[_mk("A", "x")],
        hits_post_dedup=[_mk("A", "x")],
        log_path=log_path,
        enabled=False,
        stderr=io.StringIO(),
    )
    assert not log_path.exists()


def test_record_user_prompt_submit_log_no_path_is_noop(
    tmp_path: Path,
) -> None:
    """`log_path=None` means we couldn't derive a per-session file
    (e.g. the brain-graph DB is in-memory). Must be a silent no-op."""
    record_user_prompt_submit_log(
        prompt="q",
        session_id="x",
        hits_pre_dedup=[_mk("A", "x")],
        hits_post_dedup=[_mk("A", "x")],
        log_path=None,
        enabled=True,
        stderr=io.StringIO(),
    )


def test_ups_row_carries_the_string_retrieve_was_handed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The forwarding hop at `hook.py:1910`, which nothing else covers.

    `scored_query=scored_query` in the `record_user_prompt_submit_log(...)`
    call carries this fix for 97.7% of rows, and it is a defaulted
    parameter on all three signatures — so deleting it raises nothing and
    every row silently reverts to `null`, the pre-fix state.

    Neither existing test can see that. The source grep still matches two
    frames up at `hook.py:1164`; the schema test pins the *key* set, and
    the builder always emits the key with a `None` default.

    So this spies on `_retrieve` — the function the hook actually hands
    the query to — and asserts the logged value is that same string, in
    the same fire. Both composition regimes are covered, because the
    conversation-aware branch is guarded by `if recent_turns:`
    (`hook.py:1083`) and is therefore *not* taken on a session whose
    transcript yields no turns.
    """
    from aelfrice import hook as hook_mod

    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "the kitchen is full of bananas")])
    monkeypatch.setenv("AELFRICE_DB", str(db))

    seen: list[str] = []
    real_retrieve = hook_mod._retrieve

    def spy(query: str, *args: object, **kwargs: object) -> object:
        seen.append(query)
        return real_retrieve(query, *args, **kwargs)

    monkeypatch.setattr(hook_mod, "_retrieve", spy)

    prompt = "are there bananas in the kitchen"
    rc = user_prompt_submit(
        stdin=io.StringIO(_payload(prompt, "ups-sq-1")),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    assert rc == 0
    assert len(seen) == 1, f"expected one _retrieve call, got {len(seen)}"

    rows = _read_log(tmp_path / "rebuild_logs" / "ups-sq-1.jsonl")
    assert len(rows) == 1
    scored = rows[0]["input"]["scored_query"]

    assert scored is not None, (
        "the UPS row must carry the string retrieve() was handed; None "
        "means the forwarding hop in hook.py was lost and the row is back "
        "to 'unknown'"
    )
    assert scored == seen[0], (
        f"_retrieve got {seen[0]!r}, the row recorded {scored!r}"
    )
    # With no readable transcript the composition branch is skipped, so
    # the composed and raw forms coincide here; the divergent regime is
    # covered below.


def test_ups_row_carries_the_composition_when_the_branch_is_taken(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The regime where the recorded string is not the raw prompt.

    `_build_conversation_aware_query` is reached only when
    `_read_recent_for_pre_compact` yields turns (`hook.py:1083`). With a
    real turns log present the scored string becomes the prompt repeated
    `conversation_aware_prompt_weight` times plus the turn window — a
    string `_query_for_recent_turns` could not produce — so this arm is
    what distinguishes recording the caller's value from re-deriving one.
    """
    from aelfrice import hook as hook_mod

    cwd = tmp_path / "proj"
    turns_log = cwd / ".git" / "aelfrice" / "transcripts" / "turns.jsonl"
    turns_log.parent.mkdir(parents=True)
    turns_log.write_text(
        "\n".join(
            json.dumps({"role": r, "text": t, "session_id": "ups-sq-2"})
            for r, t in [
                ("user", "we were discussing fruit storage"),
                ("assistant", "yes, the pantry inventory"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    db = tmp_path / "memory.db"
    _seed_db(db, [_mk("F1", "the kitchen is full of bananas")])
    monkeypatch.setenv("AELFRICE_DB", str(db))

    seen: list[str] = []
    real_retrieve = hook_mod._retrieve

    def spy(query: str, *args: object, **kwargs: object) -> object:
        seen.append(query)
        return real_retrieve(query, *args, **kwargs)

    monkeypatch.setattr(hook_mod, "_retrieve", spy)

    prompt = "are there bananas in the kitchen"
    payload = json.dumps({
        "session_id": "ups-sq-2",
        "transcript_path": "/dev/null",
        "cwd": str(cwd),
        "hook_event_name": "UserPromptSubmit",
        "prompt": prompt,
    })
    rc = user_prompt_submit(
        stdin=io.StringIO(payload),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    assert rc == 0
    assert len(seen) == 1

    rows = _read_log(tmp_path / "rebuild_logs" / "ups-sq-2.jsonl")
    assert len(rows) == 1
    scored = rows[0]["input"]["scored_query"]

    assert scored == seen[0], (
        f"_retrieve got {seen[0]!r}, the row recorded {scored!r}"
    )
    assert scored.count(prompt) > 1, (
        "expected the conversation-aware composition (prompt repeated "
        f"{hook_mod.DEFAULT_CONV_AWARE_WEIGHT}x), got {scored!r}"
    )
    assert "fruit storage" in scored, "the turn window must be appended"
    assert scored != rows[0]["input"]["extracted_query"], (
        "the row must record the composed string, not the extraction"
    )
