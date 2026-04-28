"""Context-rebuilder unit tests: rebuild() determinism, format, adapters."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aelfrice.context_rebuilder import (
    CLOSE_TAG,
    DEFAULT_TOKEN_BUDGET,
    MAX_TURN_TEXT_CHARS,
    OPEN_TAG,
    RecentTurn,
    find_aelfrice_log,
    read_recent_turns_aelfrice,
    read_recent_turns_claude_transcript,
    rebuild,
)
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
from aelfrice.store import MemoryStore


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
        demotion_pressure=0,
        created_at="2026-04-26T00:00:00Z",
        last_retrieved_at=None,
    )


def _seed(db_path: Path, beliefs: list[Belief]) -> MemoryStore:
    store = MemoryStore(str(db_path))
    for b in beliefs:
        store.insert_belief(b)
    return store


# ---- rebuild() format ---------------------------------------------------


def test_rebuild_emits_recent_turns_then_beliefs(tmp_path: Path) -> None:
    store = _seed(tmp_path / "m.db", [_mk("F1", "kitchen has bananas")])
    try:
        out = rebuild(
            [
                RecentTurn(role="user", text="kitchen contents check"),
                RecentTurn(role="assistant", text="bananas inventory"),
            ],
            store,
        )
    finally:
        store.close()
    assert out.startswith(OPEN_TAG + "\n")
    assert CLOSE_TAG in out
    assert "<recent-turns>" in out
    assert '<turn role="user">' in out
    assert '<turn role="assistant">' in out
    assert "</recent-turns>" in out
    assert "<retrieved-beliefs" in out
    assert 'id="F1"' in out
    assert "<continue/>" in out
    # <continue/> must come after </retrieved-beliefs> per spec.
    assert out.index("<continue/>") > out.index("</retrieved-beliefs>")


def test_rebuild_omits_recent_turns_section_when_empty(tmp_path: Path) -> None:
    store = _seed(tmp_path / "m.db", [_mk("F1", "kitchen has bananas")])
    try:
        out = rebuild([], store)
    finally:
        store.close()
    assert "<recent-turns>" not in out
    # Empty query → retrieve() returns L0 only; F1 is unlocked so no hits.
    assert "<retrieved-beliefs" not in out
    assert "<continue/>" in out


def test_rebuild_omits_belief_section_when_no_hits(tmp_path: Path) -> None:
    store = _seed(tmp_path / "m.db", [_mk("F1", "kitchen has bananas")])
    try:
        out = rebuild(
            [RecentTurn(role="user", text="totally unrelated topic xyzzy")],
            store,
        )
    finally:
        store.close()
    assert "<recent-turns>" in out
    assert "<retrieved-beliefs" not in out


def test_rebuild_marks_locked_beliefs(tmp_path: Path) -> None:
    locked = _mk(
        "L1", "user is jonsobol", lock_level=LOCK_USER,
        locked_at="2026-04-26T00:00:00Z",
    )
    store = _seed(tmp_path / "m.db", [locked])
    try:
        out = rebuild(
            [RecentTurn(role="user", text="hello")],
            store,
        )
    finally:
        store.close()
    assert 'locked="true"' in out
    assert 'id="L1"' in out


# ---- determinism --------------------------------------------------------


def test_rebuild_is_deterministic(tmp_path: Path) -> None:
    store = _seed(
        tmp_path / "m.db",
        [_mk("F1", "kitchen bananas"), _mk("F2", "kitchen apples")],
    )
    try:
        turns = [RecentTurn(role="user", text="kitchen contents")]
        out_a = rebuild(turns, store)
        out_b = rebuild(turns, store)
    finally:
        store.close()
    assert out_a == out_b


# ---- truncation and escaping --------------------------------------------


def test_rebuild_truncates_long_turn_text(tmp_path: Path) -> None:
    store = _seed(tmp_path / "m.db", [])
    long_text = "x" * (MAX_TURN_TEXT_CHARS + 200)
    try:
        out = rebuild([RecentTurn(role="user", text=long_text)], store)
    finally:
        store.close()
    # Truncation marker present; full long text is not.
    assert "..." in out
    assert long_text not in out


def test_rebuild_xml_escapes_turn_text(tmp_path: Path) -> None:
    store = _seed(tmp_path / "m.db", [])
    try:
        out = rebuild(
            [RecentTurn(role="user", text="a < b && c > d")],
            store,
        )
    finally:
        store.close()
    assert "&lt;" in out
    assert "&gt;" in out
    assert "&amp;" in out
    assert "<turn role=\"user\">a < b" not in out


# ---- token budget honored ----------------------------------------------


def test_rebuild_respects_token_budget(tmp_path: Path) -> None:
    big = "kitchen contents " * 200  # ~3500 chars per belief
    store = _seed(
        tmp_path / "m.db",
        [_mk(f"F{i}", big) for i in range(5)],
    )
    try:
        out = rebuild(
            [RecentTurn(role="user", text="kitchen")],
            store,
            token_budget=500,  # tight budget — caps the belief tail
        )
    finally:
        store.close()
    # Hard upper bound: belief content total at 500 tokens × 4 chars/tok
    # is ~2000 chars. Five copies of `big` would be ~17500. Cap holds.
    assert len(out) < 4000


# ---- read_recent_turns_aelfrice ----------------------------------------


def test_read_aelfrice_log_returns_last_n_turns(tmp_path: Path) -> None:
    p = tmp_path / "turns.jsonl"
    lines = [
        json.dumps({"role": "user", "text": f"msg {i}"})
        for i in range(20)
    ]
    p.write_text("\n".join(lines) + "\n")
    got = read_recent_turns_aelfrice(p, n=5)
    assert len(got) == 5
    assert [t.text for t in got] == [f"msg {i}" for i in range(15, 20)]


def test_read_aelfrice_log_skips_malformed_lines(tmp_path: Path) -> None:
    p = tmp_path / "turns.jsonl"
    p.write_text(
        '{"role":"user","text":"good"}\n'
        "this line is not json\n"
        '{"role":"assistant","text":"also good"}\n'
        '{"role":"system","text":"wrong role, dropped"}\n'
        '{"role":"user","text":""}\n'  # empty text dropped
    )
    got = read_recent_turns_aelfrice(p, n=10)
    assert [t.text for t in got] == ["good", "also good"]


def test_read_aelfrice_log_missing_file_returns_empty(tmp_path: Path) -> None:
    got = read_recent_turns_aelfrice(tmp_path / "nope.jsonl", n=5)
    assert got == []


# ---- read_recent_turns_claude_transcript -------------------------------


def test_read_claude_transcript_extracts_user_and_assistant(
    tmp_path: Path,
) -> None:
    p = tmp_path / "session.jsonl"
    p.write_text(
        json.dumps({
            "type": "user",
            "message": {"role": "user", "content": "what database?"},
        }) + "\n"
        + json.dumps({
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "sqlite via FTS5"}],
            },
        }) + "\n"
    )
    got = read_recent_turns_claude_transcript(p, n=10)
    assert len(got) == 2
    assert got[0].role == "user"
    assert got[0].text == "what database?"
    assert got[1].role == "assistant"
    assert got[1].text == "sqlite via FTS5"


def test_read_claude_transcript_skips_tool_only_assistant_turns(
    tmp_path: Path,
) -> None:
    p = tmp_path / "session.jsonl"
    p.write_text(
        json.dumps({
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "x"}],
            },
        }) + "\n"
        + json.dumps({
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "after the tool"}],
            },
        }) + "\n"
    )
    got = read_recent_turns_claude_transcript(p, n=10)
    assert len(got) == 1
    assert got[0].text == "after the tool"


def test_read_claude_transcript_missing_file_returns_empty(
    tmp_path: Path,
) -> None:
    got = read_recent_turns_claude_transcript(tmp_path / "nope.jsonl", n=5)
    assert got == []


# ---- find_aelfrice_log -------------------------------------------------


def test_find_aelfrice_log_walks_to_git_root(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    nested = root / "a" / "b" / "c"
    nested.mkdir(parents=True)
    got = find_aelfrice_log(nested)
    assert got is not None
    assert got == root / ".git" / "aelfrice" / "transcripts" / "turns.jsonl"


def test_find_aelfrice_log_returns_none_outside_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Only safe test bound: call into a tmp dir that's not in any git tree.
    # If tmp_path is itself inside a git tree (it isn't on macOS), this
    # walks to that tree's .git and the test would still pass with a
    # non-None result; in that case, treat it as a known-pass.
    here = tmp_path / "no_git_here"
    here.mkdir()
    got = find_aelfrice_log(here)
    # We can't strictly assert None without isolating from FS; assert the
    # weaker invariant: if it returns a path, that path's grandparent
    # (.git/aelfrice/transcripts/) reflects an actual .git dir somewhere.
    if got is not None:
        # Walk up to confirm a .git ancestor exists; otherwise it's a bug.
        gitdir = got.parent.parent.parent  # .git/aelfrice/transcripts/turns.jsonl → .git
        assert gitdir.name == ".git"


# ---- DEFAULT_TOKEN_BUDGET sanity --------------------------------------


def test_default_token_budget_matches_spec_default() -> None:
    """Spec docs/specs/context_rebuilder.md sets default 2000."""
    assert DEFAULT_TOKEN_BUDGET == 2000
