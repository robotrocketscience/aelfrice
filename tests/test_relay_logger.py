"""The transcript logger writes a prompt without the relay text it opens with."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aelfrice.ingest import _user_speech
from aelfrice.noise_filter import cut_relay
from tests.test_cross_session_notice import IDLE_NOW, LOOKALIKE_LINES, REAL_NOTICES
from tests.test_relay_scaffolding import BODY, LAST, MINE, RELAY_BLOCKS, _paste


def _log(tmp_path: Path, prompt: str, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Drive the real UserPromptSubmit handler and return the text it wrote."""
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
    return [json.loads(x)["text"] for x in out.read_text().splitlines() if x.strip()]


RELAY_PROMPTS = RELAY_BLOCKS + REAL_NOTICES


@pytest.mark.parametrize("prompt", RELAY_PROMPTS)
def test_the_logger_writes_no_row_for_a_relay_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prompt: str,
) -> None:
    assert _log(tmp_path, prompt, monkeypatch) == []


@pytest.mark.parametrize("lead", ["  ", "\t", "\n", " \n  "])
@pytest.mark.parametrize("prompt", [RELAY_BLOCKS[0], IDLE_NOW])
def test_leading_whitespace_does_not_hide_a_relay_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lead: str, prompt: str,
) -> None:
    assert _log(tmp_path, f"{lead}{prompt}", monkeypatch) == []


def test_an_uppercase_relay_prompt_writes_no_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt = f"<TEAMMATE-MESSAGE>\n{BODY}\n</TEAMMATE-MESSAGE>"
    assert _log(tmp_path, prompt, monkeypatch) == []


@pytest.mark.parametrize("prompt", RELAY_PROMPTS)
def test_the_logger_keeps_the_words_after_a_relay_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prompt: str,
) -> None:
    assert _log(tmp_path, f"{prompt}\n{MINE}\n{LAST}", monkeypatch) == [f"{MINE}\n{LAST}"]


@pytest.mark.parametrize("prompt", RELAY_PROMPTS)
def test_the_logged_row_and_the_record_give_the_same_speech(prompt: str) -> None:
    text = f"{prompt}\n{MINE}\n{LAST}"
    assert _user_speech(cut_relay(text)) == _user_speech(text)


@pytest.mark.parametrize("prompt", [
    f"  {BODY}\n",
    "  <event> handling in the parser is broken today.",
    f"{MINE}\n{RELAY_BLOCKS[0]}\n{LAST}",
    f"{MINE}\n<agent-message>\n{BODY}\n{LAST}",
    f"{MINE}\n{IDLE_NOW}\n{LAST}",
    f"{MINE}\n{_paste(RELAY_BLOCKS[3])}\n{LAST}",
    f"{_paste(RELAY_BLOCKS[3])}\n{LAST}",
    "<agent-messages> is a tag I invented myself.",
    "<agent-message> is the tag I keep seeing here.",
    *LOOKALIKE_LINES,
])
def test_the_logger_writes_the_users_prompt_verbatim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prompt: str,
) -> None:
    assert _log(tmp_path, prompt, monkeypatch) == [prompt]


@pytest.mark.parametrize("rest", [
    f"<task-notification>\n<status>done</status>\n</task-notification>\n{MINE}",
    f"<summary>Monitor done</summary>\n{MINE}",
    "Yes.",
    "<system-reminder>r</system-reminder>",
    "<local-command-caveat>Caveat</local-command-caveat>",
    f"</result>\n{MINE}",
])
def test_a_rest_the_gates_drop_alone_logs_the_whole_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rest: str,
) -> None:
    # On its own, each rest is a prompt the gates drop, so the row is the
    # whole prompt, as main writes it.
    assert _log(tmp_path / "alone", rest, monkeypatch) == []
    prompt = f"{RELAY_BLOCKS[3]}\n{rest}"
    assert _log(tmp_path / "cut", prompt, monkeypatch) == [prompt]


@pytest.mark.parametrize("rest", ["", "   \n"])
def test_no_row_for_a_blank_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rest: str,
) -> None:
    # Main writes the relay as a row here; the cut leaves nothing to write.
    assert _log(tmp_path, f"{RELAY_BLOCKS[3]}\n{rest}", monkeypatch) == []


def test_a_rest_the_gates_keep_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rest = f"<status>done</status> {MINE}"
    assert _log(tmp_path / "alone", rest, monkeypatch) == [rest]
    assert _log(tmp_path / "cut", f"{RELAY_BLOCKS[3]}\n{rest}", monkeypatch) == [rest]
