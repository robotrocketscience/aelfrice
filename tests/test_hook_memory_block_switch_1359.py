"""#1359: the `<aelfrice-memory>` off-switch and the one-line block hint.

Acceptance, from the governing 2026-08-06 ruling ("ship the cheap half"):

- default output is byte-identical to today except the one hint line;
- `AELFRICE_MEMORY_BLOCK=0` emits no per-prompt `<aelfrice-memory>` block;
- the equivalent `[memory_block] enabled = false` TOML key does the same;
- the env var wins over the TOML key, in both directions;
- suppression does not disable the correction lane or the relevance
  sweeper, and does not touch `aelf rebuild`.

Both `<aelfrice-memory>` emit paths are driven, not just the retrieval
one: the `elif gate_skip:` path that ships a session's first prompt when
the #674 shape gate refuses BM25 carries its own copy of the hint and of
the switch.

Suppression also records no exposure evidence — no `injection_events`
row, no `belief_touches` row, no injected-id ring entry. An off-switch
that logged an injection that never happened would hand the #779 Layer-3
sweeper a set of beliefs it can only ever score `referenced=0`. The
ring's `next_fire_idx` is the exception and is not exposure evidence: it
counts fires, and a suppressed fire is still a fire.

The hint's cost is pinned as an explicit number rather than a range: it is
paid on every fire that emits a block, so a silent growth in it is a
silent tax on the retrieval budget.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

import aelfrice.hook as hook_mod
from aelfrice.hook import (
    CLOSE_TAG,
    MEMORY_BLOCK_HINT,
    OPEN_TAG,
    SESSION_START_SUBBLOCK_OPEN,
    _audit_path_for_db,
    memory_block_enabled,
    read_hook_audit,
    user_prompt_submit,
)
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
from aelfrice.store import MemoryStore

_PROMPT = "tell me about bananas"

# Length 2 → below `_MIN_PROMPT_LEN` (12), so `_should_skip_bm25` returns
# ("trivial:short") and the hook takes the `elif gate_skip:` emit path.
_GATED_PROMPT = "ok"


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
        created_at="2026-04-26T00:00:00Z",
        last_retrieved_at=None,
    )


def _seed(db: Path) -> None:
    store = MemoryStore(str(db))
    try:
        store.insert_belief(_mk("F1", "banana is a yellow fruit"))
    finally:
        store.close()


def _payload(cwd: str, prompt: str = _PROMPT, session_id: str = "s1") -> str:
    return json.dumps(
        {
            "session_id": session_id,
            "transcript_path": "/dev/null",
            "cwd": cwd,
            "hook_event_name": "UserPromptSubmit",
            "prompt": prompt,
        }
    )


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    db = tmp_path / "memory.db"
    if not db.exists():
        _seed(db)
    monkeypatch.setenv("AELFRICE_DB", str(db))
    sout = io.StringIO()
    rc = user_prompt_submit(
        stdin=io.StringIO(_payload(str(tmp_path))),
        stdout=sout,
        stderr=io.StringIO(),
    )
    assert rc == 0
    return sout.getvalue()


def _write_toml(tmp_path: Path, enabled: bool) -> None:
    (tmp_path / ".aelfrice.toml").write_text(
        f"[memory_block]\nenabled = {str(enabled).lower()}\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# The hint line
# ---------------------------------------------------------------------------

def test_hint_cost_is_pinned() -> None:
    """The per-fire cost of the hint, as an explicit number.

    97 characters / 99 UTF-8 bytes (the em dash is three) / 25 estimated
    tokens at the project's `_CHARS_PER_TOKEN = 4`. Update deliberately,
    never incidentally: this is spent on every block-emitting turn.
    """
    assert len(MEMORY_BLOCK_HINT) == 97
    assert len(MEMORY_BLOCK_HINT.encode("utf-8")) == 99
    assert -(-len(MEMORY_BLOCK_HINT) // 4) == 25


def test_hint_is_inside_the_audited_token_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hint is audited, and this is what it costs the audit.

    `_write_hook_audit_record` derives `tokens` from the whole rendered
    block, so an emitting fire's audited token count now includes the
    hint. That is correct — the audit records what was injected and this
    line is injected — but it moves the per-turn injected-token baseline
    #1382 is measured against, so the size of the move is pinned here.
    +24 tokens, +25 when the pre-hint block length is a multiple of 4
    (the estimator ceil-divides by 4 and the hint is 97 = 24*4 + 1).
    """
    db = tmp_path / "memory.db"
    _seed(db)
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.setenv("AELFRICE_HOOK_AUDIT", "1")
    monkeypatch.delenv("AELFRICE_MEMORY_BLOCK", raising=False)
    sout = io.StringIO()
    assert user_prompt_submit(
        stdin=io.StringIO(_payload(str(tmp_path))),
        stdout=sout,
        stderr=io.StringIO(),
    ) == 0
    out = sout.getvalue()
    assert out.endswith(MEMORY_BLOCK_HINT)
    rec = read_hook_audit(_audit_path_for_db(db))[0]
    assert rec["tokens"] == hook_mod._audit_tokens_from_block(out)
    without = hook_mod._audit_tokens_from_block(out[: -len(MEMORY_BLOCK_HINT)])
    delta = int(rec["tokens"]) - without  # type: ignore[call-overload]
    expected = 25 if (len(out) - len(MEMORY_BLOCK_HINT)) % 4 == 0 else 24
    assert delta == expected


def test_hint_token_delta_rule_is_exact() -> None:
    """The +24/+25 rule holds for every block length, not just sampled ones.

    The docstring on `MEMORY_BLOCK_HINT` and the CHANGELOG publish the
    delta as a rule rather than as a measured pair, so the rule is what
    gets pinned: over the audited estimator itself, for a pre-hint block
    of L characters the delta is 25 iff `L % 4 == 0` and 24 otherwise.
    Swept over every L a real block can plausibly take.
    """
    seen: set[int] = set()
    for length in range(4000):
        delta = hook_mod._audit_tokens_from_block(
            "x" * length + MEMORY_BLOCK_HINT
        ) - hook_mod._audit_tokens_from_block("x" * length)
        assert delta == (25 if length % 4 == 0 else 24), length
        seen.add(delta)
    assert seen == {24, 25}


def test_hint_is_one_line_naming_switch_and_inspect_command() -> None:
    """One line, and it names both halves of the ask."""
    assert MEMORY_BLOCK_HINT.endswith("\n")
    assert MEMORY_BLOCK_HINT.count("\n") == 1
    assert "AELFRICE_MEMORY_BLOCK=0" in MEMORY_BLOCK_HINT
    assert "aelf tail" in MEMORY_BLOCK_HINT


def test_enabled_output_is_block_plus_exactly_the_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default output = today's bytes + the hint, and nothing else.

    Removing the hint suffix must leave a well-formed block that ends at
    `CLOSE_TAG` — i.e. the hint is appended *outside* the envelope, as the
    #857 coverage line is, so the block's own bytes are untouched.
    """
    out = _run(tmp_path, monkeypatch)
    assert OPEN_TAG in out
    assert out.endswith(MEMORY_BLOCK_HINT)
    without_hint = out[: -len(MEMORY_BLOCK_HINT)]
    assert without_hint.rstrip("\n").endswith(CLOSE_TAG)
    assert MEMORY_BLOCK_HINT not in without_hint


# ---------------------------------------------------------------------------
# The off-switch
# ---------------------------------------------------------------------------

def test_env_zero_suppresses_the_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AELFRICE_MEMORY_BLOCK", "0")
    out = _run(tmp_path, monkeypatch)
    assert out == ""
    assert OPEN_TAG not in out
    assert MEMORY_BLOCK_HINT not in out


def test_toml_false_suppresses_the_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AELFRICE_MEMORY_BLOCK", raising=False)
    _write_toml(tmp_path, enabled=False)
    out = _run(tmp_path, monkeypatch)
    assert out == ""


def test_env_beats_toml_in_both_directions(tmp_path: Path) -> None:
    """Documented precedence: env > TOML > default."""
    _write_toml(tmp_path, enabled=True)
    assert memory_block_enabled(
        start=tmp_path, env={"AELFRICE_MEMORY_BLOCK": "0"},
    ) is False
    _write_toml(tmp_path, enabled=False)
    assert memory_block_enabled(
        start=tmp_path, env={"AELFRICE_MEMORY_BLOCK": "1"},
    ) is True
    # Unset env falls through to TOML rather than forcing a value.
    assert memory_block_enabled(start=tmp_path, env={}) is False


def test_env_beats_toml_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The precedence holds through the live hook, not just the resolver."""
    _write_toml(tmp_path, enabled=False)
    monkeypatch.setenv("AELFRICE_MEMORY_BLOCK", "1")
    out = _run(tmp_path, monkeypatch)
    assert OPEN_TAG in out
    assert out.endswith(MEMORY_BLOCK_HINT)


def test_unparseable_values_degrade_to_enabled(tmp_path: Path) -> None:
    """Junk env value falls through; junk TOML value degrades to on."""
    assert memory_block_enabled(
        start=tmp_path, env={"AELFRICE_MEMORY_BLOCK": "maybe"},
    ) is True
    (tmp_path / ".aelfrice.toml").write_text(
        '[memory_block]\nenabled = "no"\n', encoding="utf-8",
    )
    serr = io.StringIO()
    assert memory_block_enabled(start=tmp_path, env={}, stderr=serr) is True
    assert "expected bool" in serr.getvalue()


def test_no_config_and_no_env_is_enabled(tmp_path: Path) -> None:
    assert memory_block_enabled(start=tmp_path, env={}) is True


# ---------------------------------------------------------------------------
# Suppression is narrow: other lanes keep firing
# ---------------------------------------------------------------------------

def test_suppression_keeps_correction_and_sweeper_lanes_firing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The switch silences stdout, not the rest of the turn.

    `apply_sentiment_feedback` is the correction lane; `_sweep_relevance_signal`
    reads the prior turn's transcript to push relevance evidence. Both run
    before retrieval and must be unaffected by the off-switch.
    """
    calls: list[str] = []
    real_sentiment = hook_mod.apply_sentiment_feedback
    real_sweep = hook_mod._sweep_relevance_signal

    def spy_sentiment(*a: object, **kw: object) -> int:
        calls.append("sentiment")
        return real_sentiment(*a, **kw)  # type: ignore[arg-type]

    def spy_sweep(*a: object, **kw: object) -> object:
        calls.append("sweep")
        return real_sweep(*a, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(hook_mod, "apply_sentiment_feedback", spy_sentiment)
    monkeypatch.setattr(hook_mod, "_sweep_relevance_signal", spy_sweep)
    monkeypatch.setenv("AELFRICE_MEMORY_BLOCK", "0")
    out = _run(tmp_path, monkeypatch)
    assert out == ""
    assert calls == ["sentiment", "sweep"]


def test_suppression_leaves_rebuild_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`aelf rebuild` is the documented inspection path; it is not gated."""
    from aelfrice.context_rebuilder import RecentTurn, rebuild_v14

    db = tmp_path / "memory.db"
    _seed(db)
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.setenv("AELFRICE_MEMORY_BLOCK", "0")
    turns = [RecentTurn(role="user", text=_PROMPT)]
    store = MemoryStore(str(db))
    try:
        block = rebuild_v14(turns, store, rebuild_log_enabled=False)
    finally:
        store.close()
    assert "banana" in block


# ---------------------------------------------------------------------------
# The second emit path: shape-gate skip on a session's first prompt
# ---------------------------------------------------------------------------
#
# `user_prompt_submit` writes an `<aelfrice-memory>` envelope from two
# sites. The `if hits:` site above is the retrieval one; this is the
# `elif gate_skip:` one, which fires when the prompt-shape gate (#674)
# refuses BM25 but the turn is still a session's first prompt, so the
# #578 session-start sub-block must not be silently dropped. It is a
# live path — a bare "ok" on turn one reaches it — and it carries its own
# copy of both the hint and the switch.


def _seed_locked(db: Path) -> None:
    store = MemoryStore(str(db))
    try:
        store.insert_belief(
            _mk(
                "L1",
                "always run the suite before pushing",
                lock_level=LOCK_USER,
                locked_at="2026-04-26T00:00:00Z",
            ),
        )
    finally:
        store.close()


def _run_gated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    session_id: str,
) -> tuple[str, dict[str, object]]:
    """Fire a shape-gated prompt as a session's first prompt.

    Returns (stdout, the hook-audit record). The record is what proves
    the fire took the gate-skip branch: `prompt_shape_gate_skip` is set
    at that call site and nowhere else.
    """
    db = tmp_path / "memory.db"
    _seed_locked(db)
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.setenv("AELFRICE_HOOK_AUDIT", "1")
    sout = io.StringIO()
    rc = user_prompt_submit(
        stdin=io.StringIO(
            _payload(str(tmp_path), _GATED_PROMPT, session_id),
        ),
        stdout=sout,
        stderr=io.StringIO(),
    )
    assert rc == 0
    records = read_hook_audit(_audit_path_for_db(db))
    assert len(records) == 1
    return sout.getvalue(), records[0]


def test_gate_skip_first_prompt_emits_the_block_and_the_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Switch on: the gate-skip envelope ships, and carries the hint."""
    monkeypatch.delenv("AELFRICE_MEMORY_BLOCK", raising=False)
    out, rec = _run_gated(tmp_path, monkeypatch, "s-gate-on")
    assert rec["prompt_shape_gate_skip"] == "trivial:short"
    assert rec["n_beliefs"] == 0
    assert OPEN_TAG in out
    assert SESSION_START_SUBBLOCK_OPEN in out
    assert "always run the suite before pushing" in out
    assert out.endswith(MEMORY_BLOCK_HINT)
    without_hint = out[: -len(MEMORY_BLOCK_HINT)]
    assert without_hint.rstrip("\n").endswith(CLOSE_TAG)


def test_gate_skip_first_prompt_is_suppressed_by_the_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Switch off: the same fire writes nothing, sub-block included.

    The #578 sub-block and #871's `<cadence-resume>` ride inside this
    envelope, so they go with it. That is the documented behaviour, not
    an accident — see `docs/user/CONFIG.md`.
    """
    monkeypatch.setenv("AELFRICE_MEMORY_BLOCK", "0")
    out, rec = _run_gated(tmp_path, monkeypatch, "s-gate-off")
    assert rec["prompt_shape_gate_skip"] == "trivial:short"
    assert out == ""
    assert rec["tokens"] == 0


# ---------------------------------------------------------------------------
# Suppression records no exposure evidence
# ---------------------------------------------------------------------------


def _read_exposure(db: Path, session_id: str) -> tuple[int, int]:
    """Return (injection_events rows, belief_touches rows) for a session."""
    store = MemoryStore(str(db))
    try:
        events = store._conn.execute(
            "SELECT COUNT(*) AS n FROM injection_events WHERE session_id = ?",
            (session_id,),
        ).fetchone()["n"]
        touches = store._conn.execute(
            "SELECT COUNT(*) AS n FROM belief_touches WHERE session_id = ?",
            (session_id,),
        ).fetchone()["n"]
        return int(events), int(touches)
    finally:
        store.close()


def test_suppressed_fire_records_no_exposure_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An off-switch must not manufacture negative evidence.

    `injection_events`, `belief_touches` and the session ring all assert
    that the model *saw* a belief. The Layer-3 sweeper (#779) resolves
    pending injection events against the next assistant turn, so a row
    written for a block that never reached the prompt can only ever
    resolve `referenced=0`. Both halves run here: the enabled fire is
    the control that shows the writes are reachable at all, so the
    disabled half's zeroes mean suppression and not a dead fixture.
    """
    from aelfrice.session_ring import read_ring_state

    on_dir = tmp_path / "on"
    on_dir.mkdir()
    on_db = on_dir / "memory.db"
    _seed(on_db)
    monkeypatch.delenv("AELFRICE_MEMORY_BLOCK", raising=False)
    monkeypatch.setenv("AELFRICE_DB", str(on_db))
    sout = io.StringIO()
    assert user_prompt_submit(
        stdin=io.StringIO(_payload(str(on_dir), _PROMPT, "s-eve-on")),
        stdout=sout,
        stderr=io.StringIO(),
    ) == 0
    assert OPEN_TAG in sout.getvalue()
    on_events, on_touches = _read_exposure(on_db, "s-eve-on")
    assert on_events == 1
    assert on_touches == 1
    assert [e["id"] for e in read_ring_state("s-eve-on")["ring"]] == ["F1"]

    off_dir = tmp_path / "off"
    off_dir.mkdir()
    off_db = off_dir / "memory.db"
    _seed(off_db)
    monkeypatch.setenv("AELFRICE_DB", str(off_db))
    monkeypatch.setenv("AELFRICE_MEMORY_BLOCK", "0")
    sout_off = io.StringIO()
    assert user_prompt_submit(
        stdin=io.StringIO(_payload(str(off_dir), _PROMPT, "s-eve-off")),
        stdout=sout_off,
        stderr=io.StringIO(),
    ) == 0
    assert sout_off.getvalue() == ""
    off_events, off_touches = _read_exposure(off_db, "s-eve-off")
    assert off_events == 0
    assert off_touches == 0
    assert read_ring_state("s-eve-off").get("ring", []) == []


def _read_hook_exposure_rows(db: Path) -> list[tuple[str, str, float]]:
    """(belief_id, source, valence) for every feedback_history row."""
    store = MemoryStore(str(db))
    try:
        return [
            (str(r["belief_id"]), str(r["source"]), float(r["valence"]))
            for r in store._conn.execute(
                "SELECT belief_id, source, valence FROM feedback_history "
                "ORDER BY belief_id"
            ).fetchall()
        ]
    finally:
        store.close()


def _read_pool_and_stamp(db: Path) -> tuple[list[str], str | None, float]:
    """(exploration_pool('banana'), F1.last_retrieved_at, F1.alpha)."""
    store = MemoryStore(str(db))
    try:
        row = store._conn.execute(
            "SELECT last_retrieved_at, alpha FROM beliefs WHERE id = 'F1'"
        ).fetchone()
        stamp = row["last_retrieved_at"]
        return (
            store.exploration_pool("banana"),
            None if stamp is None else str(stamp),
            float(row["alpha"]),
        )
    finally:
        store.close()


def test_suppressed_fire_does_not_evict_from_the_exploration_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fourth exposure writer, and the consequence that makes it a defect.

    `search_for_prompt` writes one `feedback_history` row per hit tagged
    `source='hook'`, and `models.EXPOSURE_ONLY_FEEDBACK_SOURCES` is
    exactly `{'hook'}` — that row *is* this codebase's exposure record.
    Its live consumer is `store.exploration_pool` (#1176), which selects
    beliefs with no `feedback_history` and no `injection_events` row:
    "never been shown". Written on a suppressed fire it evicts a belief
    from that pool permanently, having never shown it, and gating
    `injection_events` alone does not save it.

    The enabled fire is the in-test control: it pins that one fire is
    what empties the pool, so the disabled half's surviving `['F1']`
    means suppression and not an inert pool query. `last_retrieved_at`
    is asserted alongside because `record_retrieval` writes the row and
    the stamp in one transaction — the fix skips the call, so the two
    stay in agreement rather than splitting apart.
    """
    on_dir = tmp_path / "on"
    on_dir.mkdir()
    on_db = on_dir / "memory.db"
    _seed(on_db)
    monkeypatch.delenv("AELFRICE_MEMORY_BLOCK", raising=False)
    monkeypatch.setenv("AELFRICE_DB", str(on_db))
    assert _read_pool_and_stamp(on_db)[0] == ["F1"]
    sout = io.StringIO()
    assert user_prompt_submit(
        stdin=io.StringIO(_payload(str(on_dir), _PROMPT, "s-pool-on")),
        stdout=sout,
        stderr=io.StringIO(),
    ) == 0
    assert OPEN_TAG in sout.getvalue()
    assert _read_hook_exposure_rows(on_db) == [("F1", "hook", 0.1)]
    on_pool, on_stamp, _ = _read_pool_and_stamp(on_db)
    assert on_pool == []
    assert on_stamp is not None

    off_dir = tmp_path / "off"
    off_dir.mkdir()
    off_db = off_dir / "memory.db"
    _seed(off_db)
    monkeypatch.setenv("AELFRICE_DB", str(off_db))
    monkeypatch.setenv("AELFRICE_MEMORY_BLOCK", "0")
    sout_off = io.StringIO()
    assert user_prompt_submit(
        stdin=io.StringIO(_payload(str(off_dir), _PROMPT, "s-pool-off")),
        stdout=sout_off,
        stderr=io.StringIO(),
    ) == 0
    assert sout_off.getvalue() == ""
    assert _read_hook_exposure_rows(off_db) == []
    off_pool, off_stamp, _ = _read_pool_and_stamp(off_db)
    assert off_pool == ["F1"]
    assert off_stamp is None


def test_suppressed_fire_leaves_the_posterior_alone_under_legacy_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`AELFRICE_EXPOSURE_UPDATES_POSTERIOR=1` must not reach a suppressed fire.

    The documented rollback flag restores the pre-#1086 behaviour where a
    hook retrieval moves α. That is defensible for a belief the model
    saw; on a suppressed fire it moves the ranking math for a belief that
    never reached the prompt. The enabled fire is the control that shows
    the flag is live in this environment at all.
    """
    monkeypatch.setenv("AELFRICE_EXPOSURE_UPDATES_POSTERIOR", "1")

    on_dir = tmp_path / "on"
    on_dir.mkdir()
    on_db = on_dir / "memory.db"
    _seed(on_db)
    monkeypatch.delenv("AELFRICE_MEMORY_BLOCK", raising=False)
    monkeypatch.setenv("AELFRICE_DB", str(on_db))
    assert user_prompt_submit(
        stdin=io.StringIO(_payload(str(on_dir), _PROMPT, "s-post-on")),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    ) == 0
    assert _read_pool_and_stamp(on_db)[2] == pytest.approx(1.1)

    off_dir = tmp_path / "off"
    off_dir.mkdir()
    off_db = off_dir / "memory.db"
    _seed(off_db)
    monkeypatch.setenv("AELFRICE_DB", str(off_db))
    monkeypatch.setenv("AELFRICE_MEMORY_BLOCK", "0")
    assert user_prompt_submit(
        stdin=io.StringIO(_payload(str(off_dir), _PROMPT, "s-post-off")),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    ) == 0
    assert _read_pool_and_stamp(off_db)[2] == pytest.approx(1.0)


def test_suppressed_fire_reports_zero_injected_chars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A suppressed fire's telemetry must not claim an injection size.

    `_write_telemetry`'s `total_chars` is the field `aelf doctor` renders
    as `injection size p50/p95: N chars`. The fire is still recorded —
    `n_returned` keeps saying what retrieval found — but with the block
    suppressed nothing reached the prompt, so the injected size is zero.
    The enabled fire is the in-test control: it pins that `total_chars`
    is otherwise the sum of the injected hits' content lengths, so the
    disabled half's zero means suppression and not an empty retrieval.
    """
    from aelfrice.hook import (
        _telemetry_path_for_db,
        read_user_prompt_submit_telemetry,
    )

    on_dir = tmp_path / "on"
    on_dir.mkdir()
    on_db = on_dir / "memory.db"
    _seed(on_db)
    monkeypatch.delenv("AELFRICE_MEMORY_BLOCK", raising=False)
    monkeypatch.setenv("AELFRICE_DB", str(on_db))
    sout = io.StringIO()
    assert user_prompt_submit(
        stdin=io.StringIO(_payload(str(on_dir), _PROMPT, "s-tel-on")),
        stdout=sout,
        stderr=io.StringIO(),
    ) == 0
    assert OPEN_TAG in sout.getvalue()
    on_rec = read_user_prompt_submit_telemetry(
        _telemetry_path_for_db(on_db),
    )[-1]
    assert on_rec["n_returned"] == 1
    assert on_rec["total_chars"] == len("banana is a yellow fruit")

    off_dir = tmp_path / "off"
    off_dir.mkdir()
    off_db = off_dir / "memory.db"
    _seed(off_db)
    monkeypatch.setenv("AELFRICE_DB", str(off_db))
    monkeypatch.setenv("AELFRICE_MEMORY_BLOCK", "0")
    sout_off = io.StringIO()
    assert user_prompt_submit(
        stdin=io.StringIO(_payload(str(off_dir), _PROMPT, "s-tel-off")),
        stdout=sout_off,
        stderr=io.StringIO(),
    ) == 0
    assert sout_off.getvalue() == ""
    off_tel = _telemetry_path_for_db(off_db)
    off_rec = read_user_prompt_submit_telemetry(off_tel)[-1]
    # The fire is still on the record — only the injected size is zeroed.
    assert off_rec["n_returned"] == 1
    assert off_rec["total_chars"] == 0

    # End-to-end: one `aelf doctor` report must not print an injection
    # size beside "injection: disabled".
    from aelfrice.doctor import diagnose, format_report

    out = format_report(
        diagnose(
            user_settings=tmp_path / "none.json",
            project_root=off_dir,
            user_prompt_submit_telemetry_path=off_tel,
        )
    )
    assert "injection:            disabled" in out
    assert "injection size p50: 0 chars" in out
    assert "injection size p95: 0 chars" in out


# ---------------------------------------------------------------------------
# `aelf doctor` reports the state
# ---------------------------------------------------------------------------

def test_doctor_reports_enabled_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aelfrice.doctor import diagnose, format_report

    monkeypatch.delenv("AELFRICE_MEMORY_BLOCK", raising=False)
    report = diagnose(user_settings=tmp_path / "none.json", project_root=tmp_path)
    assert report.memory_block_enabled is True
    out = format_report(report)
    assert "Memory block" in out
    assert "injection:            enabled" in out


def test_doctor_reports_disabled_state_and_names_both_switches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The off state must be legible in doctor, and say how it got there."""
    from aelfrice.doctor import diagnose, format_report

    monkeypatch.setenv("AELFRICE_MEMORY_BLOCK", "0")
    report = diagnose(user_settings=tmp_path / "none.json", project_root=tmp_path)
    assert report.memory_block_enabled is False
    out = format_report(report)
    assert "injection:            disabled" in out
    assert "AELFRICE_MEMORY_BLOCK=0" in out
    assert "[memory_block] enabled = false" in out


def test_doctor_reads_the_project_toml_not_the_process_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`project_root` drives the resolution, not `Path.cwd()`."""
    from aelfrice.doctor import diagnose

    monkeypatch.delenv("AELFRICE_MEMORY_BLOCK", raising=False)
    _write_toml(tmp_path, enabled=False)
    report = diagnose(user_settings=tmp_path / "none.json", project_root=tmp_path)
    assert report.memory_block_enabled is False


# ---------------------------------------------------------------------------
# The fire counter is not exposure evidence
# ---------------------------------------------------------------------------


def _fire_five(
    work_dir: Path, monkeypatch: pytest.MonkeyPatch, session_id: str
) -> list[int | None]:
    """Fire UPS five times in `work_dir`; return next_fire_idx after each."""
    from aelfrice.session_ring import read_ring_state

    db = work_dir / "memory.db"
    _seed(db)
    monkeypatch.setenv("AELFRICE_DB", str(db))
    seen: list[int | None] = []
    for _ in range(5):
        assert user_prompt_submit(
            stdin=io.StringIO(_payload(str(work_dir), _PROMPT, session_id)),
            stdout=io.StringIO(),
            stderr=io.StringIO(),
        ) == 0
        state = read_ring_state(session_id)
        raw = state.get("next_fire_idx")
        seen.append(raw if isinstance(raw, int) else None)
    return seen


def test_suppressed_fires_advance_the_counter_and_record_no_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`session_ring.append_ids` does two jobs; only one of them is exposure.

    The ring's id list is the dedup record of *this fire's injection set*
    — false of a suppressed fire, and honouring it would make the next
    PreToolUse fire dedup against beliefs the model never saw. But the
    same call bumps `next_fire_idx`, which counts *fires*, and a
    suppressed fire is still a fire. Guarding the whole call froze the
    counter, so `should_fire` could never reach a firing multiple and the
    in-session `<cadence-checkpoint>` this switch documents as surviving
    was silently dead under `p1_every_k_turns` and `p3_velocity` (the
    p3_velocity branch requires `next_fire_idx - fire_idx_at_last_fire`
    to be positive off the same counter).

    Both halves are asserted together, because each alone admits the
    wrong fix: the counter alone passes if the ids come back too, and the
    empty ring alone passes on the frozen counter this replaces.
    """
    from aelfrice.cadence import POLICY_P1_EVERY_K_TURNS, CadenceConfig, should_fire
    from aelfrice.session_ring import read_ring_state

    on_dir = tmp_path / "on"
    on_dir.mkdir()
    monkeypatch.delenv("AELFRICE_MEMORY_BLOCK", raising=False)
    assert _fire_five(on_dir, monkeypatch, "s-ring-on") == [1, 2, 3, 4, 5]
    assert [
        e["id"] for e in read_ring_state("s-ring-on")["ring"]
    ] == ["F1"]

    off_dir = tmp_path / "off"
    off_dir.mkdir()
    monkeypatch.setenv("AELFRICE_MEMORY_BLOCK", "0")
    assert _fire_five(off_dir, monkeypatch, "s-ring-off") == [1, 2, 3, 4, 5]
    # The counter advanced; the dedup set stayed empty.
    assert read_ring_state("s-ring-off")["ring"] == []

    # The consumer, not just the field: at k=2 the P1 predicate fires on
    # the turns it fires on with the block enabled.
    cfg = CadenceConfig(enabled=True, policy=POLICY_P1_EVERY_K_TURNS, k=2)
    assert [should_fire(i, cfg) for i in (1, 2, 3, 4, 5)] == [
        False, True, False, True, False,
    ]

