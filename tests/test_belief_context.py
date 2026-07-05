"""Tests for #1081 best-effort source-turn recovery (`aelf context`).

All fixtures are synthetic: a `:memory:` store and a tmp-path turns.jsonl
injected via monkeypatching `belief_context.transcripts_dir`.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

import aelfrice.belief_context as bc
from aelfrice.belief_context import recover_context
from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_DERIVED_FROM,
    LOCK_NONE,
    Belief,
    Edge,
)
from aelfrice.store import MemoryStore


def _mk(bid: str, content: str, *, session_id: str | None = None) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-06-01T00:00:00Z",
        last_retrieved_at=None,
        session_id=session_id,
    )


def _write_turns(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def _point_transcripts_at(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(bc, "transcripts_dir", lambda: tmp_path)


def test_recover_missing_belief_returns_none(tmp_path: Path) -> None:
    s = MemoryStore(":memory:")
    try:
        assert recover_context(s, "nope") is None
    finally:
        s.close()


def test_recover_join_surfaces_full_source_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The core context-loss case: a bald belief recovers the full turn
    whose other sentences carry the scope."""
    _point_transcripts_at(monkeypatch, tmp_path)
    _write_turns(
        tmp_path / "turns.jsonl",
        [
            {
                "session_id": "s1",
                "role": "user",
                "text": (
                    "In the coaching API, time averages mislead. "
                    "So all averages are not useful there."
                ),
            }
        ],
    )
    s = MemoryStore(":memory:")
    try:
        s.insert_belief(
            _mk("b1", "all averages are not useful", session_id="s1")
        )
        res = recover_context(s, "b1")
    finally:
        s.close()
    assert res is not None
    assert res.recovered is True
    assert res.turn_match_total == 1
    assert len(res.turn_matches) == 1
    assert "coaching API" in res.turn_matches[0]
    assert "time averages" in res.turn_matches[0]


def test_recover_no_session_reports_unrecoverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An onboard/scanner belief has no session id — the join can never
    fire, and recovery honestly reports nothing."""
    _point_transcripts_at(monkeypatch, tmp_path)
    _write_turns(tmp_path / "turns.jsonl", [])
    s = MemoryStore(":memory:")
    try:
        s.insert_belief(_mk("b1", "the store uses SQLite", session_id=None))
        res = recover_context(s, "b1")
    finally:
        s.close()
    assert res is not None
    assert res.has_session is False
    assert res.recovered is False


def test_recover_session_mismatch_yields_no_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A turn from a different session must not be matched even if the
    text contains the belief content."""
    _point_transcripts_at(monkeypatch, tmp_path)
    _write_turns(
        tmp_path / "turns.jsonl",
        [{"session_id": "OTHER", "role": "user",
          "text": "all averages are not useful in some places."}],
    )
    s = MemoryStore(":memory:")
    try:
        s.insert_belief(
            _mk("b1", "all averages are not useful", session_id="s1")
        )
        res = recover_context(s, "b1")
    finally:
        s.close()
    assert res is not None
    assert res.turn_matches == []
    assert res.recovered is False


def test_recover_ambiguous_counts_all_caps_display(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the sentence recurs across many turns, all are counted but
    the displayed list is capped."""
    _point_transcripts_at(monkeypatch, tmp_path)
    _write_turns(
        tmp_path / "turns.jsonl",
        [
            {"session_id": "s1", "role": "user",
             "text": f"reason {i}: keep going now."}
            for i in range(7)
        ],
    )
    s = MemoryStore(":memory:")
    try:
        s.insert_belief(_mk("b1", "keep going now", session_id="s1"))
        res = recover_context(s, "b1", max_turn_matches=3)
    finally:
        s.close()
    assert res is not None
    assert res.turn_match_total == 7
    assert len(res.turn_matches) == 3  # capped


def test_recover_anchor_context_from_derived_from_edge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The second substrate: outbound DERIVED_FROM anchor_text is surfaced
    as adjacent context even with no turns.jsonl match."""
    _point_transcripts_at(monkeypatch, tmp_path)
    _write_turns(tmp_path / "turns.jsonl", [])
    s = MemoryStore(":memory:")
    try:
        s.insert_belief(_mk("b1", "so drop the averaging", session_id="s1"))
        s.insert_belief(_mk("b0", "we were discussing the coaching API"))
        s.insert_edge(
            Edge(
                src="b1", dst="b0", type=EDGE_DERIVED_FROM, weight=1.0,
                anchor_text="earlier: only time averages, in the coach API",
            )
        )
        res = recover_context(s, "b1")
    finally:
        s.close()
    assert res is not None
    assert res.recovered is True
    assert len(res.anchor_contexts) == 1
    anchor, linked = res.anchor_contexts[0]
    assert "coach API" in anchor
    assert linked == "we were discussing the coaching API"


def test_recover_ignores_non_derived_from_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _point_transcripts_at(monkeypatch, tmp_path)
    _write_turns(tmp_path / "turns.jsonl", [])
    s = MemoryStore(":memory:")
    try:
        s.insert_belief(_mk("b1", "a claim", session_id="s1"))
        s.insert_belief(_mk("b2", "unrelated"))
        s.insert_edge(
            Edge(src="b1", dst="b2", type="RELATES_TO", weight=1.0,
                 anchor_text="not a provenance anchor")
        )
        res = recover_context(s, "b1")
    finally:
        s.close()
    assert res is not None
    assert res.anchor_contexts == []
    assert res.recovered is False


# --- CLI surface ----------------------------------------------------------


def test_cli_context_missing_belief_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import aelfrice.cli as cli_module

    db = tmp_path / "memory.db"
    MemoryStore(str(db)).close()
    monkeypatch.setenv("AELFRICE_DB", str(db))
    buf = io.StringIO()
    code = cli_module.main(argv=["context", "nope"], out=buf)
    assert code == 1
    assert "belief not found" in buf.getvalue()


def test_cli_context_recovers_and_exits_0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import aelfrice.cli as cli_module

    _point_transcripts_at(monkeypatch, tmp_path)
    _write_turns(
        tmp_path / "turns.jsonl",
        [{"session_id": "s1", "role": "user",
          "text": "In the coach API, all averages are not useful there."}],
    )
    db = tmp_path / "memory.db"
    s = MemoryStore(str(db))
    s.insert_belief(_mk("b1", "all averages are not useful", session_id="s1"))
    s.close()
    monkeypatch.setenv("AELFRICE_DB", str(db))
    buf = io.StringIO()
    code = cli_module.main(argv=["context", "b1"], out=buf)
    assert code == 0
    out = buf.getvalue()
    assert "source turn" in out
    assert "coach API" in out
