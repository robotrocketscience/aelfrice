"""End-to-end: hook writes turns -> PreCompact rotates -> ingest_jsonl produces beliefs.

Mirrors the manual round-trip described in docs/design/transcript_ingest.md
acceptance criterion 2 but without a live Claude Code instance:
drive the hook entry point with synthetic JSON payloads, trigger
PreCompact, then call ingest_jsonl on the rotated archive.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from aelfrice import transcript_logger as tl
from aelfrice.ingest import ingest_jsonl
from aelfrice.models import EDGE_DERIVED_FROM
from aelfrice.store import MemoryStore


@pytest.fixture
def tdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("AELFRICE_TRANSCRIPTS_DIR", str(tmp_path))
    return tmp_path


def _drive(payload: dict[str, object]) -> int:
    sin = io.StringIO(json.dumps(payload))
    serr = io.StringIO()
    return tl.main(stdin=sin, stderr=serr)


def test_full_round_trip(tdir: Path, tmp_path: Path) -> None:
    sess = "round-trip-session"

    # Five user prompts, each followed by an assistant reply read from
    # a fixture transcript file we hand-build.
    transcript = tmp_path / "claude_code_transcript.jsonl"
    transcript_lines: list[dict[str, object]] = []

    # #785 §1: only user-role rows produce beliefs. User prompts here
    # are assertable statements so the round-trip exercises the full
    # ingest → search path; assistant replies remain for boundary
    # tracking but no longer feed belief creation.
    user_prompts = [
        "The project uses SQLite for storage.",
        "The brain-graph DB lives under .git/aelfrice/memory.db.",
        "Two worktrees of one repo share one DB via git-common-dir.",
        "The store sets busy_timeout to five thousand milliseconds.",
        "Aelfrice uses FTS5 with the porter unicode61 tokenizer.",
    ]
    assistant_replies = [
        "Acknowledged.",
        "Got it.",
        "Right.",
        "Understood.",
        "Confirmed.",
    ]

    for i, (prompt, reply) in enumerate(zip(user_prompts, assistant_replies)):
        # Write user prompt via the hook.
        rc = _drive({
            "hook_event_name": "UserPromptSubmit",
            "prompt": prompt,
            "session_id": sess,
        })
        assert rc == 0
        # Append the assistant message to the fixture transcript.
        transcript_lines.append({
            "role": "user",
            "message": {"content": prompt},
        })
        transcript_lines.append({
            "role": "assistant",
            "message": {"content": reply},
        })
        transcript.write_text(
            "\n".join(json.dumps(x) for x in transcript_lines) + "\n",
            encoding="utf-8",
        )
        # Drive Stop with the transcript_path; logger pulls last assistant message.
        rc = _drive({
            "hook_event_name": "Stop",
            "transcript_path": str(transcript),
            "session_id": sess,
        })
        assert rc == 0

    # 10 turns recorded.
    turns = (tdir / "turns.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(turns) == 10

    # Trigger PreCompact: rotates turns.jsonl + appends marker.
    _drive({"hook_event_name": "PreCompact"})
    assert not (tdir / "turns.jsonl").is_file()
    archive_dir = tdir / "archive"
    archived_files = list(archive_dir.glob("turns-*.jsonl"))
    assert len(archived_files) == 1
    archived = archived_files[0]
    archived_lines = archived.read_text(encoding="utf-8").splitlines()
    # 10 turns + compaction_start marker.
    assert len(archived_lines) == 11

    # Now ingest the archived file ourselves (in production this would
    # run as a detached subprocess spawned by the PreCompact hook).
    store = MemoryStore(":memory:")
    try:
        result = ingest_jsonl(store, archived)
        assert result.lines_read == 11
        # #785 §1: assistant-role rows are excluded from belief creation;
        # only the 5 user turns produce beliefs. Assistant rows count
        # under skipped_lines alongside the compaction marker.
        assert result.turns_ingested == 5
        assert result.skipped_lines == 6  # 5 assistant rows + 1 compaction marker
        assert result.beliefs_inserted >= 1  # depends on classifier persist rate

        # Every inserted belief should carry the round-trip session id.
        rows = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
            "SELECT session_id FROM beliefs"
        ).fetchall()
        assert rows
        for row in rows:
            assert row["session_id"] == sess

        # DERIVED_FROM edges link consecutive turns.
        edges = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
            "SELECT * FROM edges WHERE type = ?", (EDGE_DERIVED_FROM,)
        ).fetchall()
        assert len(edges) >= 1

        # A search against one of the prompt topics should surface a belief.
        hits = store.search_beliefs("SQLite storage", limit=5)
        assert any("SQLite" in h.content for h in hits)
    finally:
        store.close()


def test_idempotent_round_trip(tdir: Path) -> None:
    """Running ingest_jsonl twice on the same archive produces no new data."""
    archived = tdir / "archive" / "turns-fixed.jsonl"
    archived.parent.mkdir(parents=True, exist_ok=True)
    archived.write_text(
        json.dumps({
            "role": "user", "text": "Pi equals 3.14.",
            "session_id": "S", "turn_id": "t1",
        }) + "\n" +
        json.dumps({
            "role": "assistant", "text": "Tau equals 6.28.",
            "session_id": "S", "turn_id": "t2",
        }) + "\n",
        encoding="utf-8",
    )
    store = MemoryStore(":memory:")
    try:
        r1 = ingest_jsonl(store, archived)
        r2 = ingest_jsonl(store, archived)
        assert r2.beliefs_inserted == 0
        assert r2.edges_inserted == 0
        # turns_ingested still counts the lines read, but beliefs/edges should be 0.
        assert r1.turns_ingested == r2.turns_ingested
    finally:
        store.close()
