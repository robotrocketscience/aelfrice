"""Per-turn transcript logger: append, dispatch, non-blocking contract."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pytest

from aelfrice import transcript_logger as tl


@pytest.fixture
def tdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Override transcripts dir to a tmp path."""
    monkeypatch.setenv("AELFRICE_TRANSCRIPTS_DIR", str(tmp_path))
    return tmp_path


def _run_main(payload: dict[str, object]) -> int:
    sin = io.StringIO(json.dumps(payload))
    serr = io.StringIO()
    return tl.main(stdin=sin, stderr=serr)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    out: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def test_user_prompt_submit_appends_user_line(tdir: Path) -> None:
    rc = _run_main({
        "hook_event_name": "UserPromptSubmit",
        "prompt": "What does the project use for storage?",
        "session_id": "sess-abc",
    })
    assert rc == 0
    lines = _read_jsonl(tdir / "turns.jsonl")
    assert len(lines) == 1
    assert lines[0]["role"] == "user"
    assert lines[0]["text"] == "What does the project use for storage?"
    assert lines[0]["session_id"] == "sess-abc"
    assert lines[0]["schema_version"] == 1
    assert "ts" in lines[0]
    assert "turn_id" in lines[0]
    assert "context" in lines[0]


def test_user_prompt_submit_skips_empty_prompt(tdir: Path) -> None:
    rc = _run_main({
        "hook_event_name": "UserPromptSubmit",
        "prompt": "   ",
        "session_id": "sess-abc",
    })
    assert rc == 0
    assert not (tdir / "turns.jsonl").is_file()


def test_user_prompt_submit_skips_transcript_noise_prompt(tdir: Path) -> None:
    """#747: harness-wrapper prompts must not be appended to turns.jsonl.

    `<task-notification>` and `<summary>Monitor` shapes are scaffolding,
    not user intent — they crowd the rebuilder's recent-turns window and
    pollute downstream ingest. The logger now consults
    `noise_filter.is_transcript_noise` before append.
    """
    rc1 = _run_main({
        "hook_event_name": "UserPromptSubmit",
        "prompt": "<task-notification>worker idle</task-notification>",
        "session_id": "sess-noise-1",
    })
    rc2 = _run_main({
        "hook_event_name": "UserPromptSubmit",
        "prompt": '<summary>Monitor "PR 743" stream ended</summary>',
        "session_id": "sess-noise-2",
    })
    assert rc1 == 0
    assert rc2 == 0
    assert not (tdir / "turns.jsonl").is_file()


def test_user_prompt_submit_no_session_id_writes_null(tdir: Path) -> None:
    rc = _run_main({
        "hook_event_name": "UserPromptSubmit",
        "prompt": "hi",
    })
    assert rc == 0
    lines = _read_jsonl(tdir / "turns.jsonl")
    assert len(lines) == 1
    assert lines[0]["session_id"] is None


def test_stop_writes_assistant_line_from_transcript(tdir: Path, tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        json.dumps({"role": "user", "message": {"content": "hi"}}) + "\n" +
        json.dumps({"role": "assistant", "message": {"content": "hello back"}}) + "\n",
        encoding="utf-8",
    )
    rc = _run_main({
        "hook_event_name": "Stop",
        "transcript_path": str(transcript),
        "session_id": "sess-z",
    })
    assert rc == 0
    lines = _read_jsonl(tdir / "turns.jsonl")
    assert len(lines) == 1
    assert lines[0]["role"] == "assistant"
    assert lines[0]["text"] == "hello back"
    assert lines[0]["session_id"] == "sess-z"


def test_stop_handles_segmented_content(tdir: Path, tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.jsonl"
    msg = {
        "role": "assistant",
        "message": {
            "content": [
                {"type": "text", "text": "part1 "},
                {"type": "text", "text": "part2"},
            ],
        },
    }
    transcript.write_text(json.dumps(msg) + "\n", encoding="utf-8")
    _run_main({
        "hook_event_name": "Stop",
        "transcript_path": str(transcript),
    })
    lines = _read_jsonl(tdir / "turns.jsonl")
    assert lines[0]["text"] == "part1 part2"


def test_stop_writes_empty_text_when_no_transcript(tdir: Path) -> None:
    rc = _run_main({"hook_event_name": "Stop"})
    assert rc == 0
    lines = _read_jsonl(tdir / "turns.jsonl")
    assert len(lines) == 1
    assert lines[0]["role"] == "assistant"
    assert lines[0]["text"] == ""


def test_pre_compact_rotates_and_marks(tdir: Path) -> None:
    turns = tdir / "turns.jsonl"
    turns.write_text(json.dumps({"role": "user", "text": "x"}) + "\n", encoding="utf-8")
    rc = _run_main({"hook_event_name": "PreCompact"})
    assert rc == 0
    # Original file should be moved.
    assert not turns.is_file()
    archive_dir = tdir / "archive"
    assert archive_dir.is_dir()
    archived = list(archive_dir.glob("turns-*.jsonl"))
    assert len(archived) == 1
    archived_lines = _read_jsonl(archived[0])
    # Original line + compaction_start marker, in that order.
    assert len(archived_lines) == 2
    assert archived_lines[0]["text"] == "x"
    assert archived_lines[1]["event"] == "compaction_start"


def test_pre_compact_no_op_when_no_turns_file(tdir: Path) -> None:
    rc = _run_main({"hook_event_name": "PreCompact"})
    assert rc == 0
    archive_dir = tdir / "archive"
    if archive_dir.is_dir():
        assert not list(archive_dir.glob("turns-*.jsonl"))


def test_post_compact_writes_marker(tdir: Path) -> None:
    rc = _run_main({"hook_event_name": "PostCompact"})
    assert rc == 0
    lines = _read_jsonl(tdir / "turns.jsonl")
    assert len(lines) == 1
    assert lines[0]["event"] == "compaction_complete"


def test_unknown_event_is_no_op(tdir: Path) -> None:
    rc = _run_main({"hook_event_name": "WeirdEvent"})
    assert rc == 0
    assert not (tdir / "turns.jsonl").is_file()


def test_malformed_json_returns_zero(tdir: Path) -> None:
    sin = io.StringIO("{not json")
    serr = io.StringIO()
    rc = tl.main(stdin=sin, stderr=serr)
    assert rc == 0
    assert not (tdir / "turns.jsonl").is_file()


def test_empty_stdin_returns_zero(tdir: Path) -> None:
    sin = io.StringIO("")
    serr = io.StringIO()
    rc = tl.main(stdin=sin, stderr=serr)
    assert rc == 0


def test_filesystem_error_is_swallowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Point at a path that cannot be created (a file blocking the dir).
    blocker = tmp_path / "blocker"
    blocker.write_text("im a file", encoding="utf-8")
    monkeypatch.setenv("AELFRICE_TRANSCRIPTS_DIR", str(blocker / "subdir"))
    serr = io.StringIO()
    sin = io.StringIO(json.dumps({
        "hook_event_name": "UserPromptSubmit",
        "prompt": "hi",
    }))
    rc = tl.main(stdin=sin, stderr=serr)
    # Non-blocking contract: must still return 0.
    assert rc == 0
    # Stack trace surfaced on stderr.
    assert "Traceback" in serr.getvalue() or serr.getvalue() == ""
    assert blocker.is_file()


def test_turn_ids_unique_across_sequential_writes(tdir: Path) -> None:
    for i in range(5):
        _run_main({
            "hook_event_name": "UserPromptSubmit",
            "prompt": f"msg {i}",
        })
    lines = _read_jsonl(tdir / "turns.jsonl")
    assert len(lines) == 5
    ids = {line["turn_id"] for line in lines}
    assert len(ids) == 5


def test_duplicate_burst_collapses_to_one_line(tdir: Path) -> None:
    """#968: N identical hook fires (duplicated registration) -> one line.

    Same session/role/text within the dedup window; distinct turn_ids and
    sub-second spacing — the burst shape the issue describes.
    """
    for _ in range(4):
        rc = _run_main({
            "hook_event_name": "UserPromptSubmit",
            "prompt": "what storage does this use?",
            "session_id": "sess-burst",
        })
        assert rc == 0
    lines = _read_jsonl(tdir / "turns.jsonl")
    assert len(lines) == 1
    assert lines[0]["text"] == "what storage does this use?"


def test_assistant_stub_burst_collapses(tdir: Path) -> None:
    """A repeated Stop with no accessible text writes one empty stub, not N."""
    for _ in range(3):
        _run_main({"hook_event_name": "Stop", "session_id": "sess-stop"})
    lines = _read_jsonl(tdir / "turns.jsonl")
    assert len(lines) == 1
    assert lines[0]["role"] == "assistant"
    assert lines[0]["text"] == ""


def test_distinct_text_within_window_not_deduped(tdir: Path) -> None:
    """Acceptance: distinct-text appends within the window are unaffected."""
    _run_main({
        "hook_event_name": "UserPromptSubmit",
        "prompt": "first", "session_id": "s",
    })
    _run_main({
        "hook_event_name": "UserPromptSubmit",
        "prompt": "second", "session_id": "s",
    })
    lines = _read_jsonl(tdir / "turns.jsonl")
    assert len(lines) == 2
    assert [line["text"] for line in lines] == ["first", "second"]


def test_same_text_different_session_not_deduped(tdir: Path) -> None:
    for sid in ("sess-a", "sess-b"):
        _run_main({
            "hook_event_name": "UserPromptSubmit",
            "prompt": "identical", "session_id": sid,
        })
    lines = _read_jsonl(tdir / "turns.jsonl")
    assert len(lines) == 2


def test_duplicate_inside_window_is_dropped(
    tdir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    stamps = iter(["2026-06-18T00:00:00+00:00", "2026-06-18T00:00:01+00:00"])
    monkeypatch.setattr(tl, "_now_iso", lambda: next(stamps))
    for _ in range(2):
        _run_main({
            "hook_event_name": "UserPromptSubmit",
            "prompt": "dup", "session_id": "s",
        })
    lines = _read_jsonl(tdir / "turns.jsonl")
    assert len(lines) == 1


def test_duplicate_outside_window_is_appended(
    tdir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same text deliberately resent past the window still logs (not a burst)."""
    stamps = iter(["2026-06-18T00:00:00+00:00", "2026-06-18T00:00:05+00:00"])
    monkeypatch.setattr(tl, "_now_iso", lambda: next(stamps))
    for _ in range(2):
        _run_main({
            "hook_event_name": "UserPromptSubmit",
            "prompt": "dup", "session_id": "s",
        })
    lines = _read_jsonl(tdir / "turns.jsonl")
    assert len(lines) == 2


def test_compaction_markers_never_deduped(tdir: Path) -> None:
    """Markers append directly and are exempt from the turn-dedup guard."""
    _run_main({"hook_event_name": "PostCompact"})
    _run_main({"hook_event_name": "PostCompact"})
    lines = _read_jsonl(tdir / "turns.jsonl")
    assert len(lines) == 2
    assert all(line["event"] == "compaction_complete" for line in lines)


def test_turn_after_marker_is_appended(tdir: Path) -> None:
    """A turn whose previous line is a compaction marker is never dropped."""
    _run_main({"hook_event_name": "PostCompact"})
    _run_main({
        "hook_event_name": "UserPromptSubmit",
        "prompt": "after marker", "session_id": "s",
    })
    lines = _read_jsonl(tdir / "turns.jsonl")
    assert len(lines) == 2
    assert lines[0]["event"] == "compaction_complete"
    assert lines[1]["text"] == "after marker"


def test_skipped_duplicate_recorded_in_hook_audit(
    tdir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#968 acceptance: skip count is observable in hook_audit.jsonl."""
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "memory.db"))
    monkeypatch.delenv("AELFRICE_HOOK_AUDIT", raising=False)
    for _ in range(4):
        _run_main({
            "hook_event_name": "UserPromptSubmit",
            "prompt": "dup", "session_id": "sess-audit",
        })
    assert len(_read_jsonl(tdir / "turns.jsonl")) == 1
    audit = _read_jsonl(tmp_path / "hook_audit.jsonl")
    skips = [r for r in audit if r.get("event") == "skipped_duplicate"]
    assert len(skips) == 3
    assert skips[0]["hook"] == "transcript_logger"
    assert skips[0]["role"] == "user"
    assert skips[0]["session_id"] == "sess-audit"


def test_per_turn_latency_under_budget(tdir: Path) -> None:
    """Sub-10ms p99 is the spec target; we run 50 turns and assert p95
    stays under 50ms locally as a soft guard. p99 timing on dev
    machines is too noisy for a CI assertion; the budget assertion
    here is generous enough to flag a 10x regression without flaking
    on shared runners."""
    import time

    timings: list[float] = []
    for i in range(50):
        sin = io.StringIO(json.dumps({
            "hook_event_name": "UserPromptSubmit",
            "prompt": f"perf-msg-{i}",
        }))
        serr = io.StringIO()
        t0 = time.perf_counter()
        tl.main(stdin=sin, stderr=serr)
        timings.append((time.perf_counter() - t0) * 1000.0)
    timings.sort()
    p95 = timings[int(len(timings) * 0.95)]
    assert p95 < 50.0, f"p95={p95:.2f}ms exceeds 50ms regression guard"
    assert (tdir / "turns.jsonl").is_file()
    assert os.path.getsize(tdir / "turns.jsonl") > 0


# ---------------------------------------------------------------------------
# #1011: Stop-cadence ingestion flush (ingest live turns.jsonl, no rotation).
# ---------------------------------------------------------------------------


def _write_turns(tdir: Path, n: int) -> None:
    """Pre-populate turns.jsonl with n distinct role-bearing turn lines."""
    lines = [
        json.dumps({"role": "user", "text": f"fact number {i}", "session_id": "s"})
        for i in range(n)
    ]
    (tdir / "turns.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def captured_ingest(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Record _spawn_background_ingest targets instead of forking `aelf`.

    Returns True like a successful spawn so the cursor advances; the
    spawn-failure path is covered by test_stop_flush_failed_spawn_*.
    """
    calls: list[Path] = []
    monkeypatch.setattr(
        tl, "_spawn_background_ingest", lambda p: bool(calls.append(p)) or True
    )
    return calls


def test_count_turn_lines_excludes_markers(tdir: Path) -> None:
    src = tdir / "turns.jsonl"
    src.write_text(
        json.dumps({"role": "user", "text": "a"}) + "\n"
        + json.dumps({"event": "compaction_start"}) + "\n"
        + json.dumps({"role": "assistant", "text": "b"}) + "\n",
        encoding="utf-8",
    )
    assert tl._count_turn_lines(src) == 2


def test_stop_flush_fires_at_threshold(
    tdir: Path, captured_ingest: list[Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AELFRICE_INGEST_STOP_FLUSH_TURNS", "3")
    _write_turns(tdir, 2)
    rc = _run_main({"hook_event_name": "Stop"})  # +1 assistant stub -> 3
    assert rc == 0
    # Ingest the LIVE turns.jsonl in place; no rotation/archive.
    assert captured_ingest == [tdir / "turns.jsonl"]
    assert (tdir / "turns.jsonl").is_file()
    archive = tdir / "archive"
    assert not archive.exists() or not list(archive.glob("*"))
    assert (tdir / tl.STOP_FLUSH_CURSOR_FILENAME).read_text().strip() == "3"


def test_stop_flush_below_threshold_no_fire(
    tdir: Path, captured_ingest: list[Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AELFRICE_INGEST_STOP_FLUSH_TURNS", "12")
    _write_turns(tdir, 2)
    rc = _run_main({"hook_event_name": "Stop"})  # -> 3 turns, < 12
    assert rc == 0
    assert captured_ingest == []


def test_stop_flush_disabled_when_zero(
    tdir: Path, captured_ingest: list[Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AELFRICE_INGEST_STOP_FLUSH_TURNS", "0")
    _write_turns(tdir, 100)
    rc = _run_main({"hook_event_name": "Stop"})
    assert rc == 0
    assert captured_ingest == []


def test_stop_flush_does_not_refire_until_next_threshold(
    tdir: Path, captured_ingest: list[Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AELFRICE_INGEST_STOP_FLUSH_TURNS", "3")
    _write_turns(tdir, 3)
    tl._write_flush_cursor(tdir, 3)  # already flushed at 3
    rc = _run_main({"hook_event_name": "Stop"})  # -> 4 turns, 4-3 < 3
    assert rc == 0
    assert captured_ingest == []


def test_stop_flush_resets_cursor_after_rotation(
    tdir: Path, captured_ingest: list[Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AELFRICE_INGEST_STOP_FLUSH_TURNS", "3")
    tl._write_flush_cursor(tdir, 500)  # stale cursor from a rotated session
    _write_turns(tdir, 2)
    rc = _run_main({"hook_event_name": "Stop"})  # fresh count 3 < cursor -> reset to 0
    assert rc == 0
    assert captured_ingest == [tdir / "turns.jsonl"]


def test_stop_flush_failed_spawn_does_not_advance_cursor(
    tdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # #1012 review: if the ingest can't be spawned, the cursor must NOT
    # advance, so the next Stop retries rather than silently dropping the
    # turns and reopening the recall gap.
    monkeypatch.setenv("AELFRICE_INGEST_STOP_FLUSH_TURNS", "3")
    monkeypatch.setattr(tl, "_spawn_background_ingest", lambda p: False)
    _write_turns(tdir, 3)
    assert tl._maybe_stop_flush(tdir) is False
    assert not (tdir / tl.STOP_FLUSH_CURSOR_FILENAME).exists()

    # Next Stop with a working spawn flushes and advances the cursor.
    calls: list[Path] = []
    monkeypatch.setattr(
        tl, "_spawn_background_ingest", lambda p: bool(calls.append(p)) or True
    )
    assert tl._maybe_stop_flush(tdir) is True
    assert calls == [tdir / "turns.jsonl"]
    assert (tdir / tl.STOP_FLUSH_CURSOR_FILENAME).read_text().strip() == "3"
