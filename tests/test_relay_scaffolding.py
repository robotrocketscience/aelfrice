"""Relay text from other agents is not the user's words.

A teammate's brief (`<teammate-message>`), a background agent's report
(`<agent-message>`) and the host's cross-session notices
(`[Cross-session delivery notice]`, `[Cross-session idle notice]`) reach the
transcript as user records. Before this change the logger wrote them as user
turns and ingest stored their sentences with the user prior. The fixture
shapes are copied from real session files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aelfrice.ingest import _user_speech
from aelfrice.noise_filter import is_transcript_scaffolding

BODY = (
    "Ruling: option 1. Use the ECS wrapper with the runner untouched, "
    "timing 3 rows per block across a 60-process pool."
)

RELAY_PROMPTS = [
    f'<teammate-message teammate_id="team-lead">\n{BODY}\n</teammate-message>',
    f'<teammate-message teammate_id="team-lead" summary="ruling: option 1">'
    f"\n{BODY}\n</teammate-message>",
    f"<teammate-message>\n{BODY}\n</teammate-message>",
    f'<agent-message from="site3d-739">\n{BODY}\n</agent-message>',
    f'<cross-session-message from="uds:/tmp/s.sock" from-name="sandbox-dc">'
    f"\n{BODY}\n</cross-session-message>",
    "[Cross-session delivery notice] Your message to another session was "
    "held by that session. NOT delivered: its Claude has not seen it.",
    "[Cross-session idle notice] The session you subscribed to is idle.",
]


def _log(tmp_path: Path, prompt: str, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """Drive the real UserPromptSubmit handler; return the rows it wrote."""
    monkeypatch.setenv("AELFRICE_DOTDIR", str(tmp_path / "dot"))
    from aelfrice import transcript_logger

    tdir = tmp_path / "transcripts"
    tdir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(transcript_logger, "transcripts_dir", lambda: tdir)
    transcript_logger._handle_user_prompt_submit(
        {"session_id": "s1", "prompt": prompt, "cwd": "/tmp"}
    )
    out = tdir / transcript_logger.TURNS_FILENAME
    if not out.exists():
        return []
    return [json.loads(x) for x in out.read_text().splitlines() if x.strip()]


@pytest.mark.parametrize("prompt", RELAY_PROMPTS)
def test_the_logger_writes_no_row_for_a_relay_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prompt: str,
) -> None:
    assert _log(tmp_path, prompt, monkeypatch) == []


def test_the_logger_still_writes_a_plain_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _log(tmp_path, BODY, monkeypatch)
    assert [r["text"] for r in rows] == [BODY]


@pytest.mark.parametrize("prompt", RELAY_PROMPTS)
def test_a_relay_record_holds_no_user_speech(prompt: str) -> None:
    assert _user_speech(prompt) is None


@pytest.mark.parametrize("prompt", RELAY_PROMPTS)
def test_the_users_words_before_a_relay_block_are_kept(prompt: str) -> None:
    mine = "Check the wrapper timing first."
    assert _user_speech(f"{mine}\n{prompt}") == mine


def test_the_users_words_after_a_closed_relay_block_are_kept() -> None:
    mine = "Check the wrapper timing first."
    text = f'<agent-message from="x">\n{BODY}\n</agent-message>\n{mine}'
    assert _user_speech(text) == mine


@pytest.mark.parametrize("sentence", [
    "teammate-message turns never fire UserPromptSubmit.",
    "The agent-message tag carries a report.",
    "Cross-session notices are host text.",
])
def test_a_sentence_that_names_a_relay_tag_is_not_scaffolding(sentence: str) -> None:
    assert not is_transcript_scaffolding(sentence)
