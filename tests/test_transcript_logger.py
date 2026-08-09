"""Per-turn transcript logger: append, dispatch, non-blocking contract."""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
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


def _codex_rollout_line(role: str, *texts: str, item_type: str = "message") -> str:
    return json.dumps({
        "type": "response_item",
        "payload": {
            "type": item_type,
            "role": role,
            "content": [{"type": "output_text", "text": t} for t in texts],
        },
    })


def test_stop_codex_rollout_writes_assistant_text(tdir: Path, tmp_path: Path) -> None:
    """#1051: Codex rollout JSONL yields real assistant text, not a stub."""
    transcript = tmp_path / "rollout.jsonl"
    transcript.write_text(
        _codex_rollout_line("user", "hi") + "\n" +
        _codex_rollout_line("assistant", "hello ", "from codex") + "\n",
        encoding="utf-8",
    )
    rc = _run_main({
        "hook_event_name": "Stop",
        "transcript_path": str(transcript),
        "session_id": "01JCODEX0000000000000000",
    })
    assert rc == 0
    lines = _read_jsonl(tdir / "turns.jsonl")
    assert len(lines) == 1
    assert lines[0]["role"] == "assistant"
    assert lines[0]["text"] == "hello from codex"
    assert lines[0]["session_id"] == "01JCODEX0000000000000000"


def test_stop_codex_rollout_picks_last_assistant_message(
    tdir: Path, tmp_path: Path,
) -> None:
    transcript = tmp_path / "rollout.jsonl"
    transcript.write_text(
        _codex_rollout_line("assistant", "first answer") + "\n" +
        _codex_rollout_line("user", "follow-up") + "\n" +
        _codex_rollout_line("assistant", "second answer") + "\n",
        encoding="utf-8",
    )
    _run_main({
        "hook_event_name": "Stop",
        "transcript_path": str(transcript),
    })
    lines = _read_jsonl(tdir / "turns.jsonl")
    assert lines[0]["text"] == "second answer"


def test_stop_codex_rollout_skips_non_message_items(
    tdir: Path, tmp_path: Path,
) -> None:
    """Reasoning / tool-call response_items must not shadow the answer."""
    transcript = tmp_path / "rollout.jsonl"
    transcript.write_text(
        _codex_rollout_line("assistant", "real answer") + "\n" +
        _codex_rollout_line("assistant", "chain of thought", item_type="reasoning")
        + "\n" +
        json.dumps({"type": "response_item",
                    "payload": {"type": "function_call", "name": "shell"}}) + "\n",
        encoding="utf-8",
    )
    _run_main({
        "hook_event_name": "Stop",
        "transcript_path": str(transcript),
    })
    lines = _read_jsonl(tdir / "turns.jsonl")
    assert lines[0]["text"] == "real answer"


# --- #1439: the tail scan must not cross the current turn's boundary ------
#
# Every case below puts turn N-1's answer in the file and gives turn N no
# assistant message. The assertion is always that turn N-1's text is NOT
# returned; on unmodified main each of these returns "TURN-1 ANSWER".


def _claude_user_line(content: object) -> str:
    return json.dumps({"type": "user", "message": {
        "role": "user", "content": content}})


def _claude_assistant_line(*segments: object) -> str:
    return json.dumps({"type": "assistant", "message": {
        "role": "assistant", "content": list(segments)}})


def test_codex_scan_stops_at_turn_boundary(tmp_path: Path) -> None:
    """#1439: a Codex turn ending in a tool call yields no text."""
    transcript = tmp_path / "rollout.jsonl"
    transcript.write_text(
        _codex_rollout_line("user", "question one") + "\n" +
        _codex_rollout_line("assistant", "TURN-1 ANSWER") + "\n" +
        _codex_rollout_line("user", "question two") + "\n" +
        json.dumps({"type": "response_item",
                    "payload": {"type": "function_call", "name": "shell"}})
        + "\n" +
        json.dumps({"type": "response_item",
                    "payload": {"type": "function_call_output",
                                "output": "ok"}}) + "\n",
        encoding="utf-8",
    )
    assert tl._last_assistant_text(str(transcript)) is None


def test_claude_scan_stops_at_turn_boundary(tmp_path: Path) -> None:
    """#1439: the Claude-host shape has the same exposure, not just Codex."""
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        _claude_user_line("question one") + "\n" +
        _claude_assistant_line({"type": "text", "text": "TURN-1 ANSWER"})
        + "\n" +
        _claude_user_line("question two") + "\n" +
        _claude_assistant_line(
            {"type": "tool_use", "name": "Bash", "input": {}}) + "\n",
        encoding="utf-8",
    )
    assert tl._last_assistant_text(str(transcript)) is None


def test_stop_writes_stub_when_current_turn_has_no_answer(
    tdir: Path, tmp_path: Path,
) -> None:
    """The Stop row is an empty stub, not the previous turn's answer."""
    transcript = tmp_path / "rollout.jsonl"
    transcript.write_text(
        _codex_rollout_line("user", "question one") + "\n" +
        _codex_rollout_line("assistant", "TURN-1 ANSWER") + "\n" +
        _codex_rollout_line("user", "question two") + "\n" +
        json.dumps({"type": "response_item",
                    "payload": {"type": "function_call", "name": "shell"}})
        + "\n",
        encoding="utf-8",
    )
    rc = _run_main({
        "hook_event_name": "Stop",
        "transcript_path": str(transcript),
    })
    assert rc == 0
    lines = _read_jsonl(tdir / "turns.jsonl")
    assert len(lines) == 1
    assert lines[0]["role"] == "assistant"
    assert lines[0]["text"] == ""


def test_scan_still_returns_current_turn_answer(tmp_path: Path) -> None:
    """#1439 must not suppress a turn that *did* answer (both shapes)."""
    codex = tmp_path / "rollout.jsonl"
    codex.write_text(
        _codex_rollout_line("user", "question one") + "\n" +
        _codex_rollout_line("assistant", "TURN-1 ANSWER") + "\n" +
        _codex_rollout_line("user", "question two") + "\n" +
        _codex_rollout_line("assistant", "TURN-2 ANSWER") + "\n",
        encoding="utf-8",
    )
    assert tl._last_assistant_text(str(codex)) == "TURN-2 ANSWER"
    claude_path = tmp_path / "transcript.jsonl"
    claude_path.write_text(
        _claude_user_line("question one") + "\n" +
        _claude_assistant_line({"type": "text", "text": "TURN-1 ANSWER"})
        + "\n" +
        _claude_user_line("question two") + "\n" +
        _claude_assistant_line({"type": "text", "text": "TURN-2 ANSWER"})
        + "\n",
        encoding="utf-8",
    )
    assert tl._last_assistant_text(str(claude_path)) == "TURN-2 ANSWER"


def test_claude_tool_result_is_not_a_turn_boundary(tmp_path: Path) -> None:
    """Tool results arrive as user records on the Claude-host shape.

    They are mid-turn, not turn starts.

    The current turn's own text sits behind one, so treating a
    tool-result record as a boundary would discard a real answer.
    """
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        _claude_user_line("question one") + "\n" +
        _claude_assistant_line({"type": "text", "text": "TURN-1 ANSWER"})
        + "\n" +
        _claude_user_line("question two") + "\n" +
        _claude_assistant_line({"type": "text", "text": "TURN-2 ANSWER"})
        + "\n" +
        _claude_assistant_line(
            {"type": "tool_use", "name": "Bash", "input": {}}) + "\n" +
        _claude_user_line([{"type": "tool_result", "content": "ok"}]) + "\n",
        encoding="utf-8",
    )
    assert tl._last_assistant_text(str(transcript)) == "TURN-2 ANSWER"


def test_claude_tool_result_with_sibling_text_is_not_a_boundary(
    tmp_path: Path,
) -> None:
    """A tool result plus an appended text segment is still mid-turn.

    A <system-reminder> rides along with the tool result as a sibling
    text segment on the Claude-host shape, so the record is a mixed
    list. Requiring *every* segment to be a tool_result made that shape
    a boundary and suppressed the answer the turn really gave.
    """
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        _claude_user_line("question one") + "\n" +
        _claude_assistant_line({"type": "text", "text": "TURN-1 ANSWER"})
        + "\n" +
        _claude_user_line("question two") + "\n" +
        _claude_assistant_line(
            {"type": "text", "text": "TURN-2 REAL ANSWER"}) + "\n" +
        _claude_assistant_line(
            {"type": "tool_use", "name": "Bash", "input": {}}) + "\n" +
        _claude_user_line([
            {"type": "tool_result", "content": "ok"},
            {"type": "text", "text": "<system-reminder>note</system-reminder>"},
        ]) + "\n",
        encoding="utf-8",
    )
    assert tl._last_assistant_text(str(transcript)) == "TURN-2 REAL ANSWER"


def test_claude_interrupt_marker_is_not_a_boundary(tmp_path: Path) -> None:
    """An interrupt is a harness-written user record, not a new prompt.

    The turn's partial text is the right answer for it; treating the
    marker as a turn start returned None instead.
    """
    transcript = tmp_path / "transcript.jsonl"
    head = (
        _claude_user_line("question one") + "\n" +
        _claude_assistant_line({"type": "text", "text": "TURN-1 ANSWER"})
        + "\n" +
        _claude_user_line("question two") + "\n" +
        _claude_assistant_line(
            {"type": "text", "text": "TURN-2 REAL ANSWER"}) + "\n"
    )
    # Both marker texts, under both content encodings user records use
    # (a one-segment text list, and a bare string).
    for marker in (
        "[Request interrupted by user]",
        "[Request interrupted by user for tool use]",
    ):
        for content in ([{"type": "text", "text": marker}], marker):
            transcript.write_text(
                head + _claude_user_line(content) + "\n", encoding="utf-8")
            assert (
                tl._last_assistant_text(str(transcript))
                == "TURN-2 REAL ANSWER"
            )


def test_claude_meta_record_is_not_a_boundary(tmp_path: Path) -> None:
    """`isMeta` user records are harness text, not the turn's prompt."""
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        _claude_user_line("question one") + "\n" +
        _claude_assistant_line({"type": "text", "text": "TURN-1 ANSWER"})
        + "\n" +
        _claude_user_line("question two") + "\n" +
        _claude_assistant_line(
            {"type": "text", "text": "TURN-2 REAL ANSWER"}) + "\n" +
        json.dumps({"type": "user", "isMeta": True, "message": {
            "role": "user",
            "content": "Stop hook feedback: run the gate"}}) + "\n",
        encoding="utf-8",
    )
    assert tl._last_assistant_text(str(transcript)) == "TURN-2 REAL ANSWER"


def test_interrupt_tolerance_does_not_cross_the_prompt(
    tmp_path: Path,
) -> None:
    """Skipping the interrupt marker must not re-open the #1439 leak.

    Turn N is interrupted before it says anything, so the answer is
    None — not turn N-1's text sitting one record further back.
    """
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        _claude_user_line("question one") + "\n" +
        _claude_assistant_line({"type": "text", "text": "TURN-1 ANSWER"})
        + "\n" +
        _claude_user_line("question two") + "\n" +
        _claude_assistant_line(
            {"type": "tool_use", "name": "Bash", "input": {}}) + "\n" +
        _claude_user_line(
            [{"type": "text", "text": "[Request interrupted by user]"}])
        + "\n",
        encoding="utf-8",
    )
    assert tl._last_assistant_text(str(transcript)) is None


# The body Codex writes verbatim when a turn is aborted; all 30
# occurrences across the 76 local rollouts are byte-identical to this and
# are the record's only content segment.
_CODEX_ABORT_MARKER = (
    "<turn_aborted>\n"
    "The user interrupted the previous turn on purpose. Any running "
    "unified exec processes may still be running in the background. If "
    "any tools/commands were aborted, they may have partially executed.\n"
    "</turn_aborted>"
)


def _codex_user_input_line(*texts: str) -> str:
    """A Codex user record in the shape rollouts actually write.

    User content segments are `input_text`; only assistant records use
    the `output_text` type `_codex_rollout_line` emits.
    """
    return json.dumps({
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": t} for t in texts],
        },
    })


def test_codex_abort_marker_is_not_a_boundary(tmp_path: Path) -> None:
    """#1439: the Codex abort marker is harness text, not a new prompt.

    The three exclusions were reachable on the Claude-host arm only —
    the `response_item` arm returned before any of them — so this
    record ended the scan and the aborted turn's own partial answer
    was discarded. A turn that *did* answer returned None.
    """
    transcript = tmp_path / "rollout.jsonl"
    transcript.write_text(
        _codex_user_input_line("question one") + "\n" +
        _codex_rollout_line("assistant", "TURN-1 ANSWER") + "\n" +
        _codex_user_input_line("question two") + "\n" +
        _codex_rollout_line("assistant", "TURN-2 PARTIAL") + "\n" +
        _codex_user_input_line(_CODEX_ABORT_MARKER) + "\n",
        encoding="utf-8",
    )
    assert tl._last_assistant_text(str(transcript)) == "TURN-2 PARTIAL"


def test_codex_abort_tolerance_does_not_cross_the_prompt(
    tmp_path: Path,
) -> None:
    """Skipping the Codex abort marker must not re-open the #1439 leak.

    The aborted turn said nothing before it was interrupted, so None is
    the answer — not turn N-1's text two records further back.
    """
    transcript = tmp_path / "rollout.jsonl"
    transcript.write_text(
        _codex_user_input_line("question one") + "\n" +
        _codex_rollout_line("assistant", "TURN-1 ANSWER") + "\n" +
        _codex_user_input_line("question two") + "\n" +
        json.dumps({"type": "response_item",
                    "payload": {"type": "function_call", "name": "shell"}})
        + "\n" +
        _codex_user_input_line(_CODEX_ABORT_MARKER) + "\n",
        encoding="utf-8",
    )
    assert tl._last_assistant_text(str(transcript)) is None


def test_stop_writes_empty_text_when_no_transcript(tdir: Path) -> None:
    rc = _run_main({"hook_event_name": "Stop"})
    assert rc == 0
    lines = _read_jsonl(tdir / "turns.jsonl")
    assert len(lines) == 1
    assert lines[0]["role"] == "assistant"
    assert lines[0]["text"] == ""


# --- #1051: assistant text carried on the Stop payload itself -------------
#
# Codex CLI 0.146.1 sends `transcript_path: null` and puts the answer in
# `last_assistant_message`. Before the ordered adapter the rollout parser
# had no file to read, so every Codex turn degraded to an empty stub.


def test_stop_payload_message_used_when_transcript_path_null(
    tdir: Path,
) -> None:
    """#1051: the reopen repro — null transcript_path, text on the payload."""
    rc = _run_main({
        "hook_event_name": "Stop",
        "transcript_path": None,
        "last_assistant_message": "codex-stop-sentinel",
        "session_id": "01JCODEX0000000000000000",
    })
    assert rc == 0
    lines = _read_jsonl(tdir / "turns.jsonl")
    assert len(lines) == 1
    assert lines[0]["role"] == "assistant"
    assert lines[0]["text"] == "codex-stop-sentinel"


def test_stop_payload_message_absent_falls_back_to_rollout(
    tdir: Path, tmp_path: Path,
) -> None:
    """No payload field -> the Codex rollout parser still runs."""
    transcript = tmp_path / "rollout.jsonl"
    transcript.write_text(
        _codex_rollout_line("assistant", "from the rollout") + "\n",
        encoding="utf-8",
    )
    _run_main({
        "hook_event_name": "Stop",
        "transcript_path": str(transcript),
    })
    lines = _read_jsonl(tdir / "turns.jsonl")
    assert len(lines) == 1
    assert lines[0]["text"] == "from the rollout"


def test_stop_payload_message_wins_over_rollout(
    tdir: Path, tmp_path: Path,
) -> None:
    """Both sources readable -> exactly one row, and the payload wins.

    The two sources carry *different* text on purpose: identical text
    would leave the precedence unobservable, so the row count alone
    could not tell which branch produced it.
    """
    transcript = tmp_path / "rollout.jsonl"
    transcript.write_text(
        _codex_rollout_line("assistant", "from the rollout") + "\n",
        encoding="utf-8",
    )
    _run_main({
        "hook_event_name": "Stop",
        "transcript_path": str(transcript),
        "last_assistant_message": "from the payload",
    })
    lines = _read_jsonl(tdir / "turns.jsonl")
    assert len(lines) == 1
    assert lines[0]["text"] == "from the payload"


def test_stop_payload_whitespace_message_falls_back_to_rollout(
    tdir: Path, tmp_path: Path,
) -> None:
    """Whitespace-only is absent, not an answer — it must not suppress."""
    transcript = tmp_path / "rollout.jsonl"
    transcript.write_text(
        _codex_rollout_line("assistant", "from the rollout") + "\n",
        encoding="utf-8",
    )
    _run_main({
        "hook_event_name": "Stop",
        "transcript_path": str(transcript),
        "last_assistant_message": "   \n\t ",
    })
    lines = _read_jsonl(tdir / "turns.jsonl")
    assert len(lines) == 1
    assert lines[0]["text"] == "from the rollout"


def test_stop_payload_whitespace_message_no_transcript_writes_stub(
    tdir: Path,
) -> None:
    """Neither source has text -> the explicit stub behaviour is kept."""
    rc = _run_main({
        "hook_event_name": "Stop",
        "transcript_path": None,
        "last_assistant_message": "   ",
    })
    assert rc == 0
    lines = _read_jsonl(tdir / "turns.jsonl")
    assert len(lines) == 1
    assert lines[0]["role"] == "assistant"
    assert lines[0]["text"] == ""


def test_stop_payload_message_preserves_non_ascii(tdir: Path) -> None:
    """Every Unicode scalar survives verbatim, astral planes included."""
    answer = "café — 東京 \U0001f642"
    _run_main({
        "hook_event_name": "Stop",
        "transcript_path": None,
        "last_assistant_message": answer,
    })
    lines = _read_jsonl(tdir / "turns.jsonl")
    assert lines[0]["text"] == answer


@pytest.mark.timeout(60)
def test_stop_payload_message_survives_redirected_stdin(
    tdir: Path, tmp_path: Path,
) -> None:
    """End-to-end over a real redirected pipe, not an in-process StringIO.

    Drives the module entry point in a subprocess with UTF-8 bytes on
    stdin, which is how the hook is actually invoked. This covers the
    adapter on the wire; the Windows locale-decode half of the same
    surface is #1426 and is not addressed here.
    """
    answer = "café — 東京 \U0001f642"
    wire = json.dumps(
        {
            "hook_event_name": "Stop",
            "transcript_path": None,
            "last_assistant_message": answer,
            "session_id": "01JCODEXSTDIN00000000000",
        },
        ensure_ascii=False,
    ).encode("utf-8")

    env = dict(os.environ)
    env["AELFRICE_TRANSCRIPTS_DIR"] = str(tdir)
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run(
        [sys.executable, "-m", "aelfrice.transcript_logger"],
        input=wire,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    lines = _read_jsonl(tdir / "turns.jsonl")
    assert len(lines) == 1
    assert lines[0]["text"] == answer


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
