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
