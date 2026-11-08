"""The host's cross-session notices that open a user record are not the user's
words, and a line that only resembles one is. The notice wording follows the
host's templates.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aelfrice.ingest import _user_speech
from aelfrice.noise_filter import cut_relay
from tests.test_relay_scaffolding import LAST, MINE, RELAY_BLOCKS, _paste, _stored

DELIVERY = "[Cross-session delivery notice]"
IDLE = "[Cross-session idle notice]"
HELD = f"{DELIVERY} Your message to another session was held by that session. NOT delivered."
IDLE_NOW = f'{IDLE} "peer", which you asked to be notified about, is idle now.'

REAL_NOTICES = [
    HELD,
    f"{DELIVERY} Your message to another session was denied by the recipient user.",
    f"{DELIVERY} Your message to another session was not approved before expiry.",
    f"{DELIVERY} Your message to another session was approved and released to that session.",
    f"{DELIVERY} Your message to another session was refused: that session is not accepting them.",
    f"{DELIVERY} 3 of your messages to another session were held by that session.",
    f"{DELIVERY} 12 of your messages to another session were denied by the recipient user.",
    f"{DELIVERY} Do not resend now: one of your messages to another session was dropped at that session's inbox.",
    f"{DELIVERY} Do not resend now: 12 of your messages to another session were dropped at that session's inbox.",
    IDLE_NOW,
    f'{IDLE} "peer", which you asked to be notified about, has exited before going idle.',
    f'{IDLE} "peer" is not holding your idle subscription (refused), so no idle notice will arrive.',
    f'{IDLE} No idle signal arrived from "peer" within 24 hours; the subscription has expired.',
]

LOOKALIKE_LINES = [
    "[Cross-session compatibility notes] keep the old API stable.",
    "[Cross-session retry notice] is what I want to call this feature.",
    "[Cross-session retry notice] Your message to another session was held by that session.",
    '[Cross-session retry notice] "peer", which you asked to be notified about, is idle now.',
    f"{DELIVERY} Do not resend now: one of your messages to another session was dropped by me.",
    f"{IDLE} is the line I look for in the logs.",
    f"{DELIVERY}: this header means nothing to me.",
    f'{IDLE} "peer" is a confusing label to show.',
    f'{IDLE} "peer" is not holding the door for anyone.',
    f'{IDLE} "peer" is not holding your idle subscription is a bad phrase.',
    f'{IDLE} "peer", which you asked to be notified about, is a bad phrase.',
    f'{IDLE} No idle signal arrived from me today.',
    "[cross-session idle notice] No idle signal arrived from \"peer\" within a day.",
    f"{DELIVERY} Your messages need a better header.",
    f"{DELIVERY} Your message to another session is a confusing phrase to me.",
    f"{DELIVERY} Do not resend now: is the phrase I would drop.",
    f"{DELIVERY} 1 of your messages to another session was held by that session.",
    f"{DELIVERY} x of your messages to another session were held by that session.",
]


@pytest.mark.parametrize("notice", REAL_NOTICES)
def test_a_real_notice_holds_no_speech(notice: str) -> None:
    assert cut_relay(notice) == ""
    assert _user_speech(notice) is None


@pytest.mark.parametrize("notice", REAL_NOTICES)
def test_the_words_after_a_notice_stay(notice: str) -> None:
    assert cut_relay(f"{notice}\n{MINE}") == MINE
    assert _user_speech(f"{notice}\r\n{MINE}") == MINE


@pytest.mark.parametrize("notice", [HELD, IDLE_NOW])
def test_a_notice_after_the_users_words_stays(notice: str) -> None:
    text = f"{MINE}\n{notice}\n{LAST}"
    assert cut_relay(text) == text
    assert _user_speech(text) == text


def test_notices_and_relay_blocks_in_a_row_are_all_cut() -> None:
    text = f"  {IDLE_NOW}\n{RELAY_BLOCKS[0]}\n{HELD}\n{RELAY_BLOCKS[3]}\n{MINE}"
    assert cut_relay(text) == MINE


def test_a_notice_ending_in_a_lone_cr_stays() -> None:
    assert cut_relay(f"{IDLE_NOW}\r{MINE}") == f"{IDLE_NOW}\r{MINE}"


def test_a_notice_line_goes_whole() -> None:
    assert cut_relay(f"{IDLE_NOW} {MINE}\n{LAST}") == LAST


@pytest.mark.parametrize("between", [" ", f" and {LAST} "])
def test_a_notice_line_that_holds_a_paste_stays(between: str) -> None:
    text = f"{IDLE_NOW}{between}{_paste(MINE)}"
    assert cut_relay(text) == text
    assert IDLE_NOW in (_user_speech(text) or "")


def test_a_notice_before_a_paste_line_goes() -> None:
    paste = _paste(MINE)
    assert cut_relay(f"{IDLE_NOW}\n{paste}") == paste


@pytest.mark.parametrize("line", LOOKALIKE_LINES)
def test_a_lookalike_line_is_the_users_speech(line: str) -> None:
    assert cut_relay(line) == line
    assert _user_speech(line) == line


@pytest.mark.parametrize("line", LOOKALIKE_LINES)
def test_a_lookalike_line_keeps_a_relay_after_it(line: str) -> None:
    text = f"{line}\n{RELAY_BLOCKS[0]}"
    assert cut_relay(text) == text


@pytest.mark.parametrize("line", LOOKALIKE_LINES)
def test_a_lookalike_line_reaches_the_store(tmp_path: Path, line: str) -> None:
    assert any(line[:40] in r for r in _stored(tmp_path, line + " And more words of mine.")), line
