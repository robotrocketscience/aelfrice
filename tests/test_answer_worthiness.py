"""Per-belief answer-worthiness accumulator (`belief_relevance`).

Covers the store math (`record_reference_observation` /
`read_answer_worthiness`) and the #779 Layer-3 sweeper wiring that
populates it end-to-end. Basis: relevance-corpus campaign (R5/R6/R7).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aelfrice.hook import _sweep_relevance_signal
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief
from aelfrice.retrieval import ENV_USE_ANSWER_WORTHINESS, retrieve
from aelfrice.store import MemoryStore

_NOW = "2026-07-02T00:00:00+00:00"


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
        created_at=_NOW,
        last_retrieved_at=None,
    )


def _seed(db: Path, beliefs: list[Belief]) -> None:
    s = MemoryStore(str(db))
    try:
        for b in beliefs:
            s.insert_belief(b)
    finally:
        s.close()


def _write_assistant_turn(
    transcripts_dir: Path, session_id: str, text: str, ts: str,
) -> None:
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


# --- store math ------------------------------------------------------

def test_record_accumulates_beta(tmp_path: Path) -> None:
    db = tmp_path / "m.db"
    _seed(db, [_mk("b1", "x")])
    s = MemoryStore(str(db))
    try:
        for r in (1, 0, 1):
            s.record_reference_observation(
                belief_id="b1", referenced=r, now_iso=_NOW,
            )
        # (1,1) prior + 2 referenced / 1 not → (3, 2, inj=3).
        assert s.read_answer_worthiness(["b1"])["b1"] == (3.0, 2.0, 3)
    finally:
        s.close()


def test_record_single_unreferenced(tmp_path: Path) -> None:
    db = tmp_path / "m.db"
    _seed(db, [_mk("b2", "x")])
    s = MemoryStore(str(db))
    try:
        s.record_reference_observation(
            belief_id="b2", referenced=0, now_iso=_NOW,
        )
        assert s.read_answer_worthiness(["b2"])["b2"] == (1.0, 2.0, 1)
    finally:
        s.close()


def test_read_omits_cold_start_and_empty(tmp_path: Path) -> None:
    db = tmp_path / "m.db"
    _seed(db, [_mk("b1", "x")])
    s = MemoryStore(str(db))
    try:
        s.record_reference_observation(
            belief_id="b1", referenced=1, now_iso=_NOW,
        )
        out = s.read_answer_worthiness(["b1", "never_observed"])
        assert "b1" in out and "never_observed" not in out
        assert s.read_answer_worthiness([]) == {}
    finally:
        s.close()


@pytest.mark.parametrize("bad", [2, -1, 5])
def test_record_rejects_bad_referenced(tmp_path: Path, bad: int) -> None:
    db = tmp_path / "m.db"
    _seed(db, [_mk("b1", "x")])
    s = MemoryStore(str(db))
    try:
        with pytest.raises(ValueError):
            s.record_reference_observation(
                belief_id="b1", referenced=bad, now_iso=_NOW,
            )
    finally:
        s.close()


def test_record_rejects_empty_belief_id(tmp_path: Path) -> None:
    db = tmp_path / "m.db"
    _seed(db, [_mk("b1", "x")])
    s = MemoryStore(str(db))
    try:
        with pytest.raises(ValueError):
            s.record_reference_observation(
                belief_id="", referenced=1, now_iso=_NOW,
            )
    finally:
        s.close()


def test_accumulator_is_deterministic(tmp_path: Path) -> None:
    """Same observation sequence → identical posterior across stores."""
    seq = (1, 1, 0, 1, 0)
    outs = []
    for name in ("a.db", "b.db"):
        db = tmp_path / name
        _seed(db, [_mk("b1", "x")])
        s = MemoryStore(str(db))
        try:
            for r in seq:
                s.record_reference_observation(
                    belief_id="b1", referenced=r, now_iso=_NOW,
                )
            outs.append(s.read_answer_worthiness(["b1"])["b1"])
        finally:
            s.close()
    assert outs[0] == outs[1] == (4.0, 3.0, 5)


# --- sweeper wiring (end-to-end) -------------------------------------

def test_sweeper_records_referenced_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "m.db"
    transcripts = tmp_path / "tr"
    _seed(db, [_mk("HIT1", "the dedup_key migration ratification statement")])
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.setenv("AELFRICE_TRANSCRIPTS_DIR", str(transcripts))

    s = MemoryStore(str(db))
    try:
        s.record_injection_event(
            session_id="s-hit", turn_id="t01", belief_id="HIT1",
            injected_at="2026-05-14T00:00:00+00:00",
            source="ups", active_consumers=[],
        )
    finally:
        s.close()
    _write_assistant_turn(
        transcripts, "s-hit",
        "yes the dedup_key migration ratification statement landed",
        "2026-05-14T00:00:30+00:00",
    )

    _sweep_relevance_signal(session_id="s-hit")

    s = MemoryStore(str(db))
    try:
        # Referenced → ref_alpha grew (2,1), inj_count=1.
        assert s.read_answer_worthiness(["HIT1"])["HIT1"] == (2.0, 1.0, 1)
    finally:
        s.close()


def test_sweeper_records_unreferenced_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "m.db"
    transcripts = tmp_path / "tr"
    _seed(db, [_mk("MISS1", "the completely-unrelated content xyz123")])
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.setenv("AELFRICE_TRANSCRIPTS_DIR", str(transcripts))

    s = MemoryStore(str(db))
    try:
        s.record_injection_event(
            session_id="s-miss", turn_id="t01", belief_id="MISS1",
            injected_at="2026-05-14T00:00:00+00:00",
            source="ups", active_consumers=[],
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
        # Not referenced → ref_beta grew (1,2), inj_count=1.
        assert s.read_answer_worthiness(["MISS1"])["MISS1"] == (1.0, 2.0, 1)
    finally:
        s.close()


# --- read-time consumer (Phase C) ------------------------------------

def _mk_content(bid: str, content: str) -> Belief:
    return Belief(
        id=bid, content=content, content_hash=f"h_{bid}",
        alpha=1.0, beta=1.0, type=BELIEF_FACTUAL, lock_level=LOCK_NONE,
        locked_at=None, created_at=_NOW, last_retrieved_at=None,
    )


_Q = "alpha report cellar storage capacity"
_TEXT = "the alpha report on cellar storage capacity"


def _seed_tied(db: Path) -> None:
    # Identical content → identical BM25 → baseline order breaks on id ASC.
    _seed(db, [_mk_content("b_aaa", _TEXT), _mk_content("b_zzz", _TEXT)])


def test_consumer_flag_off_is_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "m.db"
    _seed_tied(db)
    s = MemoryStore(str(db))
    try:
        # Seed a strong answer-worthiness signal that FLAG-OFF must ignore.
        for r in (1, 1, 1, 1):
            s.record_reference_observation(
                belief_id="b_zzz", referenced=r, now_iso=_NOW,
            )
        monkeypatch.delenv(ENV_USE_ANSWER_WORTHINESS, raising=False)
        order = [b.id for b in retrieve(s, _Q, token_budget=2000)]
        # Untouched → id-ascending tiebreak.
        assert order == ["b_aaa", "b_zzz"], order
    finally:
        s.close()


def test_consumer_flag_on_reranks_by_answer_worthiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "m.db"
    _seed_tied(db)
    s = MemoryStore(str(db))
    try:
        # b_zzz highly answer-worthy, b_aaa not — enough obs to clear the gate.
        for r in (1, 1, 1, 1):
            s.record_reference_observation(
                belief_id="b_zzz", referenced=r, now_iso=_NOW,
            )
        for r in (0, 0, 0, 0):
            s.record_reference_observation(
                belief_id="b_aaa", referenced=r, now_iso=_NOW,
            )
        monkeypatch.setenv(ENV_USE_ANSWER_WORTHINESS, "1")
        order = [b.id for b in retrieve(s, _Q, token_budget=2000)]
        # AW posterior overrides the id tiebreak → b_zzz first.
        assert order == ["b_zzz", "b_aaa"], order
    finally:
        s.close()


def test_consumer_cold_start_below_min_obs_no_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "m.db"
    _seed_tied(db)
    s = MemoryStore(str(db))
    try:
        # Only 2 observations (< ANSWER_WORTHINESS_MIN_OBS=3) → cold-start,
        # so even flag-ON must fall back to the type prior (no rerank).
        for r in (1, 1):
            s.record_reference_observation(
                belief_id="b_zzz", referenced=r, now_iso=_NOW,
            )
        monkeypatch.setenv(ENV_USE_ANSWER_WORTHINESS, "1")
        order = [b.id for b in retrieve(s, _Q, token_budget=2000)]
        assert order == ["b_aaa", "b_zzz"], order
    finally:
        s.close()
