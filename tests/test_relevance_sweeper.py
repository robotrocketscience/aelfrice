"""End-to-end sweeper tests for #779 Layer 3.

UPS turn N injects beliefs → assistant transcript writes a response
that references them → UPS turn N+1 fires sweeper → injection_events
get scored → meta_belief `relevance` sub-posterior shifts.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from aelfrice.hook import (
    _read_assistant_text_since,
    _sweep_relevance_signal,
    user_prompt_submit,
)
from aelfrice.meta_beliefs import SIGNAL_RELEVANCE
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief
from aelfrice.retrieval import (
    ENV_META_BELIEF_HALF_LIFE,
    META_HALF_LIFE_KEY,
)
from aelfrice.store import MemoryStore

TEST_META_KEY = "meta:retrieval.test_relevance_consumer"


def _mk(bid: str, content: str) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        demotion_pressure=0,
        created_at="2026-05-14T00:00:00+00:00",
        last_retrieved_at=None,
    )


def _seed(db: Path, beliefs: list[Belief]) -> None:
    s = MemoryStore(str(db))
    try:
        for b in beliefs:
            s.insert_belief(b)
    finally:
        s.close()


def _payload(prompt: str, session_id: str = "sess-779") -> str:
    return json.dumps(
        {
            "session_id": session_id,
            "transcript_path": "/dev/null",
            "cwd": "/tmp",
            "hook_event_name": "UserPromptSubmit",
            "prompt": prompt,
        }
    )


def _fire_ups(prompt: str, session_id: str = "sess-779") -> str:
    out = io.StringIO()
    rc = user_prompt_submit(
        stdin=io.StringIO(_payload(prompt, session_id)),
        stdout=out,
        stderr=io.StringIO(),
    )
    assert rc == 0
    return out.getvalue()


def _write_assistant_turn(
    transcripts_dir: Path, session_id: str, text: str, ts: str,
) -> None:
    """Mimic transcript_logger's Stop-hook append."""
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    line = {
        "schema_version": 1,
        "ts": ts,
        "role": "assistant",
        "text": text,
        "session_id": session_id,
        "turn_id": ts + "-test",
        "context": {"cwd": "/tmp"},
    }
    with (transcripts_dir / "turns.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False, separators=(",", ":")))
        f.write("\n")


# --- _read_assistant_text_since unit tests ---------------------------

def test_read_assistant_text_missing_file_returns_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AELFRICE_TRANSCRIPTS_DIR", str(tmp_path))
    assert _read_assistant_text_since("s", "2026-01-01T00:00:00Z") == ""


def test_read_assistant_text_filters_by_session_and_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AELFRICE_TRANSCRIPTS_DIR", str(tmp_path))
    # session-A user line — must be skipped (role).
    _write_assistant_turn(tmp_path, "s-A", "user text", "2026-05-14T00:00:01Z")
    # We can't easily inject role=user via the helper; simulate by writing raw.
    raw_user = {
        "schema_version": 1,
        "ts": "2026-05-14T00:00:02Z",
        "role": "user",
        "text": "user not assistant",
        "session_id": "s-A",
        "turn_id": "u1",
        "context": {},
    }
    with (tmp_path / "turns.jsonl").open("a") as f:
        f.write(json.dumps(raw_user))
        f.write("\n")
    # session-B assistant line — must be skipped (session_id).
    _write_assistant_turn(tmp_path, "s-B", "wrong session", "2026-05-14T00:00:03Z")
    # session-A assistant line — kept.
    _write_assistant_turn(
        tmp_path, "s-A", "the right assistant text", "2026-05-14T00:00:04Z",
    )
    out = _read_assistant_text_since("s-A", "2026-05-14T00:00:03Z")
    # Only the s-A assistant line newer than the cutoff survives.
    assert "the right assistant text" in out
    assert "user not assistant" not in out
    assert "wrong session" not in out
    assert "user text" not in out  # was assistant=False-shaped via helper


def test_read_assistant_text_concatenates_in_file_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AELFRICE_TRANSCRIPTS_DIR", str(tmp_path))
    _write_assistant_turn(tmp_path, "s", "first", "2026-05-14T00:00:01Z")
    _write_assistant_turn(tmp_path, "s", "second", "2026-05-14T00:00:02Z")
    _write_assistant_turn(tmp_path, "s", "third", "2026-05-14T00:00:03Z")
    out = _read_assistant_text_since("s", "2026-01-01T00:00:00Z")
    assert "first" in out and "second" in out and "third" in out
    assert out.index("first") < out.index("second") < out.index("third")


def test_read_assistant_text_skips_malformed_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AELFRICE_TRANSCRIPTS_DIR", str(tmp_path))
    p = tmp_path / "turns.jsonl"
    with p.open("w") as f:
        f.write("not valid json\n")
        f.write("\n")  # empty
        f.write(json.dumps({
            "schema_version": 1, "ts": "2026-05-14T00:00:01Z",
            "role": "assistant", "text": "good line",
            "session_id": "s", "turn_id": "x",
        }) + "\n")
    assert "good line" in _read_assistant_text_since("s", "2026-01-01T00:00:00Z")


# --- _sweep_relevance_signal — pure cases ----------------------------

def test_sweep_no_session_returns_silently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "m.db"))
    _sweep_relevance_signal(session_id=None)  # no exception


def test_sweep_no_pending_events_no_op(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "m.db"
    _seed(db, [_mk("B1", "x")])
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.setenv("AELFRICE_TRANSCRIPTS_DIR", str(tmp_path / "tr"))
    _sweep_relevance_signal(session_id="empty")
    # No rows scored because there are no pending events.
    s = MemoryStore(str(db))
    try:
        assert s._conn.execute(
            "SELECT COUNT(*) FROM injection_events WHERE referenced IS NOT NULL"
        ).fetchone()[0] == 0
    finally:
        s.close()


def test_sweep_no_transcript_text_leaves_events_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If transcripts file is missing, sweeper returns; events stay
    pending for a later sweep when transcripts catch up."""
    db = tmp_path / "m.db"
    _seed(db, [_mk("B1", "the load-bearing claim")])
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.setenv("AELFRICE_TRANSCRIPTS_DIR", str(tmp_path / "nonexistent"))
    s = MemoryStore(str(db))
    try:
        s.record_injection_event(
            session_id="s", turn_id="t", belief_id="B1",
            injected_at="2026-05-14T00:00:00+00:00",
            source="ups", active_consumers=[],
        )
    finally:
        s.close()
    _sweep_relevance_signal(session_id="s")
    s = MemoryStore(str(db))
    try:
        n = s._conn.execute(
            "SELECT COUNT(*) FROM injection_events WHERE referenced IS NULL"
        ).fetchone()[0]
        assert n == 1
    finally:
        s.close()


# --- _sweep_relevance_signal — end-to-end ----------------------------

def _install_test_meta_belief(db: Path) -> None:
    """Install a test-only meta-belief subscribed to relevance — the
    half-life consumer (#756) only subscribes to latency, so without
    this we can't observe relevance evidence landing."""
    s = MemoryStore(str(db))
    try:
        s.install_meta_belief(
            TEST_META_KEY,
            static_default=0.5,
            half_life_seconds=30 * 24 * 3600,
            signal_weights={SIGNAL_RELEVANCE: 1.0},
            now_ts=1700000000,
        )
    finally:
        s.close()


def test_sweep_scores_referenced_belief_and_updates_posterior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "m.db"
    transcripts = tmp_path / "tr"
    _seed(db, [_mk("HIT1", "the dedup_key migration ratification statement")])
    _install_test_meta_belief(db)
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.setenv("AELFRICE_TRANSCRIPTS_DIR", str(transcripts))

    # Inject one event referencing HIT1 with the test consumer.
    s = MemoryStore(str(db))
    try:
        s.record_injection_event(
            session_id="s-end-to-end", turn_id="t01", belief_id="HIT1",
            injected_at="2026-05-14T00:00:00+00:00",
            source="ups",
            active_consumers=[TEST_META_KEY],
        )
    finally:
        s.close()

    # Assistant response references the belief content verbatim.
    _write_assistant_turn(
        transcripts, "s-end-to-end",
        "yes the dedup_key migration ratification statement landed",
        "2026-05-14T00:00:30+00:00",
    )

    _sweep_relevance_signal(session_id="s-end-to-end")

    # The event is now scored.
    s = MemoryStore(str(db))
    try:
        row = s._conn.execute(
            "SELECT referenced, referenced_at FROM injection_events"
        ).fetchone()
        assert row["referenced"] == 1
        assert row["referenced_at"] is not None
        # Meta-belief posterior shifted: alpha or beta grew off prior.
        state = s.read_meta_belief_state(TEST_META_KEY)
        assert state is not None
        assert SIGNAL_RELEVANCE in state.posteriors
        p = state.posteriors[SIGNAL_RELEVANCE]
        assert (p.alpha + p.beta) > 1.0
    finally:
        s.close()


def test_sweep_scores_unreferenced_belief_as_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "m.db"
    transcripts = tmp_path / "tr"
    _seed(db, [_mk("MISS1", "the completely-unrelated content xyz123")])
    _install_test_meta_belief(db)
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.setenv("AELFRICE_TRANSCRIPTS_DIR", str(transcripts))

    s = MemoryStore(str(db))
    try:
        s.record_injection_event(
            session_id="s-miss", turn_id="t01", belief_id="MISS1",
            injected_at="2026-05-14T00:00:00+00:00",
            source="ups", active_consumers=[TEST_META_KEY],
        )
    finally:
        s.close()
    _write_assistant_turn(
        transcripts, "s-miss",
        "the response is about something else entirely",
        "2026-05-14T00:00:30+00:00",
    )

    _sweep_relevance_signal(session_id="s-miss")

    s = MemoryStore(str(db))
    try:
        row = s._conn.execute(
            "SELECT referenced FROM injection_events"
        ).fetchone()
        assert row["referenced"] == 0
    finally:
        s.close()


def test_sweep_idempotent_second_run_no_double_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-running the sweeper with no new events leaves posteriors
    unchanged — already-scored rows are skipped (referenced IS NULL
    filter)."""
    db = tmp_path / "m.db"
    transcripts = tmp_path / "tr"
    _seed(db, [_mk("HIT2", "the persistent ratified value here")])
    _install_test_meta_belief(db)
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.setenv("AELFRICE_TRANSCRIPTS_DIR", str(transcripts))

    s = MemoryStore(str(db))
    try:
        s.record_injection_event(
            session_id="s-idem", turn_id="t01", belief_id="HIT2",
            injected_at="2026-05-14T00:00:00+00:00",
            source="ups", active_consumers=[TEST_META_KEY],
        )
    finally:
        s.close()
    _write_assistant_turn(
        transcripts, "s-idem",
        "the persistent ratified value here appears here",
        "2026-05-14T00:00:30+00:00",
    )

    _sweep_relevance_signal(session_id="s-idem")
    s = MemoryStore(str(db))
    try:
        state1 = s.read_meta_belief_state(TEST_META_KEY)
    finally:
        s.close()
    _sweep_relevance_signal(session_id="s-idem")  # second run
    s = MemoryStore(str(db))
    try:
        state2 = s.read_meta_belief_state(TEST_META_KEY)
    finally:
        s.close()
    p1 = state1.posteriors[SIGNAL_RELEVANCE]
    p2 = state2.posteriors[SIGNAL_RELEVANCE]
    # second sweep was a no-op on the relevance posterior; only the
    # passive decay between calls can shift it, and we're far inside
    # the 30d half-life.
    assert abs((p2.alpha + p2.beta) - (p1.alpha + p1.beta)) < 1e-6


def test_sweep_skips_subscribers_that_dont_listen_to_relevance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If active_consumers includes a meta-belief that only subscribes
    to latency (e.g. the v1 half-life consumer), update_meta_belief
    silently no-ops on the relevance signal. The injection_events row
    still gets stamped as scored."""
    db = tmp_path / "m.db"
    transcripts = tmp_path / "tr"
    _seed(db, [_mk("H3", "the verbatim cellar storage capacity")])
    # Install the v1 half-life consumer (latency-only).
    s = MemoryStore(str(db))
    try:
        from aelfrice.meta_beliefs import SIGNAL_LATENCY
        s.install_meta_belief(
            META_HALF_LIFE_KEY,
            static_default=0.5,
            half_life_seconds=30 * 24 * 3600,
            signal_weights={SIGNAL_LATENCY: 1.0},
            now_ts=1700000000,
        )
        s.record_injection_event(
            session_id="s-no-sub", turn_id="t01", belief_id="H3",
            injected_at="2026-05-14T00:00:00+00:00",
            source="ups",
            active_consumers=[META_HALF_LIFE_KEY],
        )
    finally:
        s.close()
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.setenv("AELFRICE_TRANSCRIPTS_DIR", str(transcripts))
    _write_assistant_turn(
        transcripts, "s-no-sub",
        "the verbatim cellar storage capacity is sufficient",
        "2026-05-14T00:00:30+00:00",
    )

    _sweep_relevance_signal(session_id="s-no-sub")

    s = MemoryStore(str(db))
    try:
        row = s._conn.execute(
            "SELECT referenced FROM injection_events"
        ).fetchone()
        assert row["referenced"] == 1  # event stamped
        state = s.read_meta_belief_state(META_HALF_LIFE_KEY)
        # Half-life consumer doesn't subscribe to relevance, so no
        # relevance posterior row materialised.
        assert state is not None
        assert SIGNAL_RELEVANCE not in state.posteriors
    finally:
        s.close()
