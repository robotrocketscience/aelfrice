"""Relay text that opens a user record is not the user's words, and every word
from the user's first one on stays.

A teammate's brief (`<teammate-message>`), a background agent's report
(`<agent-message>`) and another session's message reach the transcript as user
records. The block fixtures follow the host's wrapper, an opening tag and a
closing tag each on their own line.
"""

from __future__ import annotations

import json
import tracemalloc
from pathlib import Path

import pytest

from aelfrice.ingest import _user_speech, ingest_jsonl
from aelfrice.noise_filter import cut_relay
from aelfrice.store import MemoryStore

BODY = "Ruling: option 1. Keep the retry wrapper and leave the runner untouched."
MINE = "Check the wrapper timing first."
LAST = "And then read the logs after that."
RELAY_TAGS = ["teammate-message", "agent-message", "cross-session-message"]

RELAY_BLOCKS = [
    f'<teammate-message teammate_id="team-lead">\n{BODY}\n</teammate-message>',
    f'<teammate-message teammate_id="team-lead" summary="ruling: option 1">\n{BODY}\n</teammate-message>',
    f"<teammate-message>\n{BODY}\n</teammate-message>",
    f'<agent-message from="worker-7">\n{BODY}\n</agent-message>',
    f'<cross-session-message from="uds:/tmp/s.sock" from-name="peer">\n{BODY}\n</cross-session-message>',
]


def _stored(tmp_path: Path, text: str) -> list[str]:
    """Every belief that one user record gives through the real ingest path."""
    rec = {"schema_version": 1, "role": "user", "session_id": "s1",
           "ts": "2026-11-06T10:00:00Z", "text": text}
    log = tmp_path / "turns.jsonl"
    log.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    store = MemoryStore(str(tmp_path / "memory.db"))
    try:
        ingest_jsonl(store, log)
        return [r[0] for r in store._conn.execute(  # noqa: SLF001
            "SELECT content FROM beliefs").fetchall()]
    finally:
        store.close()


def _paste(body: str) -> str:
    return f"<pasted_content>\n{body}\n</pasted_content>"


@pytest.mark.parametrize("block", RELAY_BLOCKS)
def test_a_relay_record_holds_no_user_speech(block: str) -> None:
    assert cut_relay(block) == ""
    assert _user_speech(block) is None


@pytest.mark.parametrize("block", RELAY_BLOCKS)
def test_the_words_after_a_relay_block_stay(block: str) -> None:
    assert cut_relay(f"{block}\n{MINE}") == MINE
    assert _user_speech(f"{block}\n{MINE}") == MINE


@pytest.mark.parametrize("block", RELAY_BLOCKS[:4])
def test_a_relay_block_after_the_users_words_stays(block: str) -> None:
    text = f"{MINE}\n{block}\n{LAST}"
    assert cut_relay(text) == text
    assert _user_speech(text) is not None
    assert BODY in (_user_speech(text) or "")


@pytest.mark.parametrize("block", RELAY_BLOCKS[:4])
def test_a_relay_block_after_the_users_words_is_stored_as_typed(tmp_path: Path, block: str) -> None:
    rows = _stored(tmp_path, f"{MINE}\n{block}\n{LAST}")
    assert any("retry wrapper" in row for row in rows), rows


def test_a_cross_session_block_after_the_users_words_is_cut_as_on_main() -> None:
    block = RELAY_BLOCKS[4]
    assert _user_speech(f"{MINE}\n{block}\n{LAST}") == f"{MINE}\n\n{LAST}"


@pytest.mark.parametrize("lead", ["  ", "\t", "\n\n", " \n  ", "\ufeff", "\r\n"])
def test_whitespace_before_the_opener_does_not_hide_it(lead: str) -> None:
    assert cut_relay(f"{lead}{RELAY_BLOCKS[0]}\n{MINE}") == MINE


@pytest.mark.parametrize("tag", RELAY_TAGS)
@pytest.mark.parametrize("case", [str.lower, str.upper])
def test_every_relay_tag_counts_in_any_case(tag: str, case: object) -> None:
    name = case(tag)  # type: ignore[operator]
    assert cut_relay(f"<{name}>\n{BODY}\n</{name}>\n{MINE}") == MINE


def test_a_block_with_crlf_line_endings_is_cut() -> None:
    text = "\r\n".join(["<agent-message>", BODY, "</agent-message>", MINE])
    assert cut_relay(text) == MINE


def test_a_closer_ending_in_a_lone_cr_keeps_the_record() -> None:
    text = "\r".join(["<agent-message>", BODY, "</agent-message>", MINE])
    assert cut_relay(text) == text


def test_blocks_in_a_row_are_all_cut() -> None:
    text = f"{RELAY_BLOCKS[0]}\n\n{RELAY_BLOCKS[3]}\n  {RELAY_BLOCKS[4]}\n{MINE}"
    assert cut_relay(text) == MINE


def test_the_cut_stops_at_the_users_first_word() -> None:
    text = f"{RELAY_BLOCKS[0]}\n{MINE}\n{RELAY_BLOCKS[3]}"
    assert cut_relay(text) == f"{MINE}\n{RELAY_BLOCKS[3]}"


@pytest.mark.parametrize("after", [f" {MINE}", " <result>", " <system-reminder>", "<pasted_content>"])
def test_a_block_whose_closer_shares_its_line_stays(after: str) -> None:
    text = f"<agent-message>\n{BODY}\n</agent-message>{after}\n{MINE}\n{LAST}"
    assert cut_relay(text) == text


def test_the_cut_stops_at_a_block_whose_closer_shares_its_line() -> None:
    second = f"<teammate-message>\n{BODY}\n</teammate-message> {MINE}"
    assert cut_relay(f"{RELAY_BLOCKS[3]}\n{second}") == second


def test_the_rest_keeps_its_leading_whitespace() -> None:
    assert cut_relay(f"{RELAY_BLOCKS[0]}\n\n  {MINE}") == f"\n  {MINE}"


def test_a_block_ends_at_the_first_closer_of_its_tag() -> None:
    text = (
        "<agent-message>\nquote </teammate-message> here\n</agent-message>\n"
        f"{MINE}\n</agent-message>"
    )
    assert cut_relay(text) == f"{MINE}\n</agent-message>"


def test_a_quoted_closer_ends_the_block_and_the_rest_stays() -> None:
    text = f"<agent-message>\nquote\n</agent-message>\n{BODY}\n</agent-message>\n{MINE}"
    assert cut_relay(text) == f"{BODY}\n</agent-message>\n{MINE}"


def test_an_unclosed_opener_takes_the_record() -> None:
    assert cut_relay(f"<agent-message>\n{BODY}\n{MINE}") == ""
    assert _user_speech(f"<agent-message>\n{BODY}\n{MINE}") is None


def test_an_unclosed_opener_after_a_block_takes_the_record() -> None:
    assert cut_relay(f"{RELAY_BLOCKS[0]}\n<agent-message>\n{BODY}") == ""


def test_an_unclosed_opener_stops_at_a_paste() -> None:
    paste = _paste(MINE)
    text = f"<agent-message>\n{BODY}\n{paste}\n</agent-message>\n{LAST}"
    assert cut_relay(text) == f"{paste}\n</agent-message>\n{LAST}"
    assert cut_relay(f"<agent-message>\n{BODY} {paste}") == paste
    speech = _user_speech(text) or ""
    assert MINE in speech
    assert LAST in speech
    assert BODY not in speech


@pytest.mark.parametrize("opener", [
    "<agent-message> is the tag I keep seeing",
    "<agent-messages>",
    "<agent-message-log>",
    "<agent-message/>",
    "<agent-message />",
    '<agent-message from="a/>',
    "<agent-message from=\n\"a\">",
    "<agent-message\nfrom=\"a\">",
    "<agent-message from=\"a\nb\">",
    "<agent-message from=a/b>",
    '<agent-message summary="Fix the "foo" bug">',
    "<agent-message>x</agent-message>",
    "`<agent-message>`",
])
def test_an_opener_that_is_not_alone_and_whole_keeps_the_record(opener: str) -> None:
    text = f"{opener}\n{MINE}\n</agent-message>\n{LAST}"
    assert cut_relay(text) == text


def test_a_one_line_block_at_the_start_stays() -> None:
    text = f"<agent-message>x</agent-message> {MINE}"
    assert cut_relay(text) == text
    assert _user_speech(text) == text


@pytest.mark.parametrize("opener", [
    '<teammate-message teammate_id="lead" color="blue" summary="a > b">',
    "<teammate-message from='x' to=y>",
    "<teammate-message flag>",
    "<teammate-message   >",
    "<teammate-message>  ",
])
def test_an_opener_with_attributes_counts(opener: str) -> None:
    assert cut_relay(f"{opener}\n{BODY}\n</teammate-message>\n{MINE}") == MINE


def test_a_reminder_inside_a_block_goes_with_it() -> None:
    text = f"<agent-message>\n<system-reminder>\nx\n</system-reminder>\n{BODY}\n</agent-message>\n{MINE}"
    assert cut_relay(text) == MINE
    assert _user_speech(text) == MINE


@pytest.mark.parametrize("reminder", [
    "<system-reminder>\n</agent-message>\n</system-reminder>",
    '<system-reminder a="1">\n</agent-message>\n</system-reminder>',
    "<system-reminder>\n</agent-message>\n</system-reminder >",
])
def test_a_closer_inside_a_reminder_does_not_close_the_block(reminder: str) -> None:
    text = f"<agent-message>\n{reminder}\n{BODY}\n</agent-message>\n{MINE}"
    assert cut_relay(text) == MINE
    assert _user_speech(text) == MINE


def test_a_nested_reminder_opener_does_not_extend_the_reminder() -> None:
    reminder = "<system-reminder>\n<system-reminder>\nx\n</system-reminder>"
    text = f"<agent-message>\n{reminder}\n</agent-message>\n{MINE}"
    assert cut_relay(text) == MINE


def test_a_closer_after_a_reminder_closes_the_block() -> None:
    text = f"<agent-message>\n{BODY}\n<system-reminder>\nx\n</system-reminder></agent-message>\n{MINE}"
    assert cut_relay(text) == MINE


def test_a_closer_after_a_paste_does_not_close_the_block() -> None:
    text = f"<agent-message>\n{BODY}\n{_paste('x')}\n</agent-message>\n{MINE}"
    assert cut_relay(text) == f"{_paste('x')}\n</agent-message>\n{MINE}"


def test_a_closer_right_before_a_paste_keeps_the_record() -> None:
    text = f"<agent-message>\n{BODY}\n</agent-message>{_paste('x')}\n{MINE}"
    assert cut_relay(text) == text


def test_a_paste_tag_inside_a_reminder_is_not_a_paste() -> None:
    text = f"<agent-message>\n<system-reminder>\nsee <pasted_content> here\n</system-reminder>\n{BODY}\n</agent-message>\n{MINE}"
    assert cut_relay(text) == MINE
    assert _user_speech(text) == MINE


def test_a_stray_paste_closer_is_not_a_paste() -> None:
    text = f"<agent-message>\n</pasted_content>\n{BODY}\n</agent-message>\n{MINE}"
    assert cut_relay(text) == MINE


def test_a_paste_ahead_of_a_relay_keeps_the_record() -> None:
    text = f"{_paste(MINE)}\n{RELAY_BLOCKS[0]}"
    assert cut_relay(text) == text


@pytest.mark.parametrize("lead", [
    "<system-reminder>\nx\n</system-reminder>\n",
    "[Request interrupted by user]\n",
])
def test_a_relay_behind_a_reminder_or_a_marker_stays(lead: str) -> None:
    text = f"{lead}{RELAY_BLOCKS[0]}\n{MINE}"
    assert cut_relay(text) == text
    assert BODY in (_user_speech(text) or "")


def test_a_slash_command_behind_a_relay_drops_the_record() -> None:
    text = f"{RELAY_BLOCKS[3]}\n<command-name>/foo</command-name>\n{MINE}"
    assert _user_speech(text) is None


@pytest.mark.parametrize("text", [
    "",
    "   ",
    f"  {MINE}  \n",
    "The <agent-message> tag carries a report.",
    "teammate-message turns never fire on their own here.",
])
def test_text_with_no_opening_relay_comes_back_unchanged(text: str) -> None:
    assert cut_relay(text) == text


@pytest.mark.parametrize("text", [
    f"{RELAY_BLOCKS[0]}\n{MINE}",
    f"{RELAY_BLOCKS[0]}\n<system-reminder>\nx\n</system-reminder>\n{RELAY_BLOCKS[3]}\n{MINE}",
    f"{RELAY_BLOCKS[0]}\n[Request interrupted by user]\n{RELAY_BLOCKS[3]}\n{MINE}",
    f"<system-reminder>\nx\n</system-reminder>\n{RELAY_BLOCKS[0]}\n{MINE}",
    f"<agent-message>\n<system-reminder>\n</agent-message>\n</system-reminder>\n{BODY}\n</agent-message>\n{MINE}",
    f"{RELAY_BLOCKS[3]}\n<command-name>/foo</command-name>\n{MINE}",
    f"<agent-message>\n{BODY}\n{_paste(MINE)}\n{LAST}",
])
def test_the_logged_row_and_the_record_give_the_same_speech(text: str) -> None:
    assert _user_speech(cut_relay(text)) == _user_speech(text)
    assert cut_relay(cut_relay(text)) == cut_relay(text)


def test_a_relay_record_stores_nothing(tmp_path: Path) -> None:
    assert _stored(tmp_path, f"{RELAY_BLOCKS[0]}\n{RELAY_BLOCKS[3]}") == []


def test_the_words_after_a_relay_reach_the_store(tmp_path: Path) -> None:
    rows = _stored(tmp_path, f"{RELAY_BLOCKS[0]}\n{MINE} {LAST}")
    assert any("wrapper timing" in row for row in rows), rows
    assert not any("retry wrapper" in row for row in rows), rows


def test_a_long_run_of_attributes_with_no_end_holds_memory_flat() -> None:
    text = "<agent-message" + " a=b" * 250_000
    tracemalloc.start()
    try:
        assert cut_relay(text) == text
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 50 * 1024 * 1024


def test_a_closer_with_spaces_inside_or_after_it_closes_the_block() -> None:
    text = f"<agent-message>\n{BODY}\n</agent-message >  \n{MINE}"
    assert cut_relay(text) == MINE


def test_an_unclosed_opener_stops_at_the_first_paste() -> None:
    first, second = _paste("x"), _paste("y")
    text = f"<agent-message>\n{BODY}\n{first}\n{MINE}\n{second}"
    assert cut_relay(text) == f"{first}\n{MINE}\n{second}"


def test_a_reminder_closer_inside_its_opener_does_not_end_the_reminder() -> None:
    text = (
        "<agent-message>\n<system-reminder </system-reminder>\n</agent-message>\n"
        f"</system-reminder>\n{BODY}\n</agent-message>\n{MINE}"
    )
    assert cut_relay(text) == MINE


@pytest.mark.parametrize("space", ["\r", "\x85", "\x0b", "\u2028", " \r "])
def test_nothing_is_cut_when_stripping_would_move_a_tag_to_a_line_start(space: str) -> None:
    text = f"{RELAY_BLOCKS[3]}\n{space}<result>\n{MINE}"
    assert cut_relay(text) == text
    assert MINE in (_user_speech(text) or "")


@pytest.mark.parametrize("space", ["", "  ", "\n", "\n\t", "\r\n  "])
def test_blank_lines_and_spaces_after_the_cut_are_fine(space: str) -> None:
    assert cut_relay(f"{RELAY_BLOCKS[3]}\n{space}{MINE}") == f"{space}{MINE}"


@pytest.mark.parametrize("between", [
    "<system-reminder>x</system-reminder>\r",
    "<system-reminder>x</system-reminder>\n<system-reminder>y</system-reminder>\r",
])
def test_the_strip_guard_looks_past_a_reminder(between: str) -> None:
    text = f"{RELAY_BLOCKS[3]}\n{between}<result>\n{MINE}"
    assert cut_relay(text) == text
    assert MINE in (_user_speech(text) or "")


def test_a_reminder_after_the_cut_is_fine_when_a_line_follows() -> None:
    reminder = "<system-reminder>x</system-reminder>"
    assert cut_relay(f"{RELAY_BLOCKS[3]}\n{reminder}\n{MINE}") == f"{reminder}\n{MINE}"
