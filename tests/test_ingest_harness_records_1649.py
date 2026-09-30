"""#1649: a harness record in a turn log is not user speech.

A background task's completion notice reaches the transcript as a *user*
turn, and its `<result>` body is a sub-agent's report. The #785 speaker
gate keeps the model's replies out of belief creation, and the #747 logger
gate drops such a prompt before it is written. But `ingest_jsonl` reads
turn logs that other loggers wrote, and the host's own session files,
and it applied only the per-sentence filter. A report's plain sentences
pass that filter, so they were stored as the user's beliefs: on one store,
169 ingested rows carried a report's closing tag, and those beliefs were
retrieved into later prompts.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aelfrice.ingest import ingest_jsonl
from aelfrice.store import MemoryStore

REPORT = (
    "The review found no blocking defect in the parser change. "
    "Every fuzzed body stayed inside its room and no cut lock would fit. "
    "The mutation harness killed all three mutants against the new tests."
)
NOTICE = (
    "<task-notification>\n"
    "<task-id>a70e90684d0f87ce2</task-id>\n"
    "<tool-use-id>toolu_01PBsgqqLAjY7R3oThkewRk9</tool-use-id>\n"
    "<output-file>/tmp/tasks/a70e90684d0f87ce2.output</output-file>\n"
    "<status>completed</status>\n"
    '<summary>Agent "Blind review" finished</summary>\n'
    f"<result>{REPORT}</result>\n"
    "<usage><total_tokens>94483</total_tokens></usage>\n"
    "</task-notification>"
)
USER = "The release checks must include pyright before every tag."


def _write(path: Path, records: list[dict[str, object]]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n",
                    encoding="utf-8")
    return path


def _contents(store: MemoryStore) -> list[str]:
    rows = store._conn.execute("SELECT content FROM beliefs").fetchall()  # noqa: SLF001
    return [r[0] for r in rows]


def test_a_task_notice_logged_as_a_user_turn_is_not_ingested(
    tmp_path: Path,
) -> None:
    log = _write(tmp_path / "turns.jsonl", [
        {"schema_version": 1, "role": "user", "text": NOTICE,
         "session_id": "s1", "ts": "2026-09-30T10:00:00Z"},
        {"schema_version": 1, "role": "user", "text": USER,
         "session_id": "s1", "ts": "2026-09-30T10:01:00Z"},
    ])
    store = MemoryStore(str(tmp_path / "memory.db"))
    try:
        ingest_jsonl(store, log)
        contents = _contents(store)
    finally:
        store.close()
    joined = "\n".join(contents)
    for sentence in REPORT.split(". "):
        assert sentence.rstrip(".") not in joined, sentence
    assert any("pyright" in c for c in contents), contents


def test_the_local_logger_shape_is_gated_too(tmp_path: Path) -> None:
    """A second logger's shape: it writes `event` and `timestamp` beside
    `role` and `ts`. That log is where the measured leak came from, so the
    gate has to hold for it, not only for aelfrice's own logger."""
    def rec(ts: str, text: str) -> dict[str, object]:
        return {"timestamp": ts, "ts": ts, "event": "user", "role": "user",
                "session_id": "s1", "cwd": str(tmp_path), "text": text}

    log = _write(tmp_path / "turns.jsonl", [
        rec("2026-09-30T10:00:00Z", NOTICE),
        rec("2026-09-30T10:01:00Z", USER),
    ])
    store = MemoryStore(str(tmp_path / "memory.db"))
    try:
        ingest_jsonl(store, log)
        contents = _contents(store)
    finally:
        store.close()
    joined = "\n".join(contents)
    assert "mutation harness" not in joined, contents
    assert any("pyright" in c for c in contents), contents


BANNER = (
    "[SYSTEM NOTIFICATION - NOT USER INPUT]\n"
    "This is an automated background-task event, NOT a message from the user.\n"
    "Do NOT interpret this as user acknowledgement, confirmation, or response "
    "to any pending question.\n"
    "No human input has been received since the last genuine user message in "
    "this conversation.\n\n"
)


def _ingest_one(tmp_path: Path, record: dict[str, object]) -> list[str]:
    log = _write(tmp_path / "turns.jsonl", [record])
    store = MemoryStore(str(tmp_path / "memory.db"))
    try:
        ingest_jsonl(store, log)
        return _contents(store)
    finally:
        store.close()


def _tl(text: str) -> dict[str, object]:
    return {"schema_version": 1, "role": "user", "text": text,
            "session_id": "s1", "ts": "2026-09-30T10:00:00Z"}


def _cc(*blocks: str) -> dict[str, object]:
    """The host's own session-file shape: content blocks on a user message."""
    return {"type": "user", "sessionId": "s1",
            "timestamp": "2026-09-30T10:00:00Z",
            "message": {"role": "user",
                        "content": [{"type": "text", "text": b} for b in blocks]}}


def _wrap(tag: str, body: str = REPORT) -> str:
    return f"<{tag}>\n{body}\n</{tag}>"


@pytest.mark.parametrize("record", [
    _tl(" " + NOTICE),
    _tl("\n" + NOTICE),
    _tl(BANNER + NOTICE),
    _tl(BANNER + BANNER + NOTICE),
    _cc(BANNER + NOTICE),
    _cc(NOTICE),
    _tl(_wrap("bash-stdout")),
    _tl(_wrap("tool-result")),
    _tl(_wrap("cross-session-message")),
    _tl("<command-message>aelf:onboard</command-message>\n"
        "<command-name>/aelf:onboard</command-name>\n" + REPORT),
    _cc("<system-reminder>Background context.</system-reminder>",
        "<command-name>/aelf:onboard</command-name>\n" + REPORT),
], ids=["lead-space", "lead-newline", "banner", "two-banners",
        "session-file-banner", "session-file", "bash-stdout", "tool-result",
        "cross-session", "command-wrapper", "reminder-then-command"])
def test_no_harness_shape_leaks_its_body(
    tmp_path: Path, record: dict[str, object],
) -> None:
    """Found by review: the first gate tested only how the text began, so a
    notice after one space, a newline or the host's banner still leaked
    (one real record: 120 beliefs), and machine-output wrappers the logger
    never covered leaked too. A slash command's body is dropped with its
    wrapper, because it is the command's text, not the user's."""
    joined = "\n".join(_ingest_one(tmp_path, record))
    assert "mutation harness" not in joined
    assert "fuzzed body" not in joined
    assert "background-task event" not in joined, joined


@pytest.mark.parametrize("record", [
    _tl(USER + "\n" + NOTICE),
    _tl(NOTICE + "\n\n" + USER),
    _tl(BANNER.rstrip("\n") + "\n" + USER),
    _tl(USER + "\n" + NOTICE.replace("</task-notification>", "")),
    _tl(USER + "\n<bash-stdout>\n" + REPORT),
    _tl(_wrap("BASH-STDOUT") + "\n" + USER),
    _tl(USER + "\n<result>" + REPORT + "</result>"),
    _tl(USER + "\n" + _wrap("aelfrice-worker-context")),
    _tl("[Request interrupted by user]\n" + USER),
    _cc("<system-reminder>Background context.</system-reminder>", USER),
], ids=["user-then-notice", "notice-then-user", "banner-no-blank-line",
        "unclosed-notice", "unclosed-block", "uppercase-block", "bare-result",
        "worker-context", "interrupt-marker", "reminder-then-user"])
def test_the_users_own_words_beside_a_harness_block_are_kept(
    tmp_path: Path, record: dict[str, object],
) -> None:
    """Found by the second review: a notice before the user's words dropped
    the whole record, a banner with no blank line after it took the user's
    paragraph, and an unclosed, uppercase or unwrapped block leaked."""
    contents = _ingest_one(tmp_path, record)
    joined = "\n".join(contents)
    assert any("pyright" in c for c in contents), contents
    assert "mutation harness" not in joined.lower(), contents
    assert "background-task event" not in joined, contents
    assert "Request interrupted" not in joined, contents


def test_a_reminder_the_user_quotes_mid_sentence_is_their_text() -> None:
    """Found in PR review: the reminder pass matched anywhere, so a user
    quoting one inline lost the quoted words, unlike every other block."""
    text = ("The hook adds <system-reminder>stay terse</system-reminder> "
            "to my prompt. " + USER)
    speech = _user_speech(text) or ""
    assert "stay terse" in speech, speech
    assert "pyright" in speech, speech


def test_a_tag_the_user_mentions_mid_sentence_is_their_text(
    tmp_path: Path,
) -> None:
    """Only a block that opens on its own line is harness output. A user
    asking about a tag keeps every word, including text after a later real
    block's closer, which a non-anchored pattern would have swallowed."""
    text = ("The <bash-stdout> tag gets logged for every shell call. " + USER
            + "\n<bash-stdout>ok</bash-stdout>")
    joined = "\n".join(_ingest_one(tmp_path, _tl(text)))
    assert "gets logged for every shell call" in joined, joined
    assert "pyright" in joined, joined


def test_a_shell_line_does_not_take_the_users_other_sentences(
    tmp_path: Path,
) -> None:
    """Found by the second review: a record that opened like a shell
    command was dropped whole, losing the user's sentences after it. The
    per-sentence filter drops the shell line alone."""
    text = "git status\n" + USER
    contents = _ingest_one(tmp_path, _tl(text))
    assert any("pyright" in c for c in contents), contents


@pytest.mark.parametrize("flag", ["isSidechain", "isMeta", "isCompactSummary"])
def test_a_host_record_that_is_not_a_person_typing_is_skipped(
    tmp_path: Path, flag: str,
) -> None:
    """Found by the second review: a sub-agent's prompt (sidechain), harness
    text (meta) and the model's own session summary reached belief creation
    as user turns. On one machine 2,209 sub-agent prompts and 15 summaries
    were in the files ingest reads."""
    record = _cc(USER)
    record[flag] = True
    assert _ingest_one(tmp_path, record) == []


def test_the_chain_anchor_is_the_cleaned_text(tmp_path: Path) -> None:
    """The DERIVED_FROM anchor carries the previous turn's text. It must be
    the user's words, not the harness block that sat beside them."""
    log = _write(tmp_path / "turns.jsonl", [
        _tl(USER + "\n" + NOTICE),
        {**_tl("We ship the release from the publish script."),
         "ts": "2026-09-30T10:01:00Z"},
    ])
    store = MemoryStore(str(tmp_path / "memory.db"))
    try:
        ingest_jsonl(store, log)
        anchors = [r[0] for r in store._conn.execute(  # noqa: SLF001
            "SELECT anchor_text FROM edges WHERE type = 'DERIVED_FROM'")]
    finally:
        store.close()
    assert anchors, "expected a DERIVED_FROM edge between the two turns"
    assert all("mutation harness" not in (a or "") for a in anchors), anchors
    assert any("pyright" in (a or "") for a in anchors), anchors


from aelfrice.ingest import _user_speech  # noqa: E402

# Spelled out here, not read from `ingest._HARNESS_BLOCK_TAGS`: a test
# parametrized over the module's own list loses the case for any tag the
# list loses, so it could never catch the removal it exists to catch.
AUDITED_TAGS = (
    "task-notification", "system-reminder", "tool-result",
    "cross-session-message", "aelfrice-worker-context",
    "task-id", "tool-use-id", "output-file", "status", "summary", "result",
    "usage", "bash-input", "bash-stdout", "bash-stderr",
    "local-command-stdout", "local-command-stderr", "local-command-caveat",
    "user-prompt-submit-hook", "command-name", "command-message",
    "command-args",
)


@pytest.mark.parametrize("tag", AUDITED_TAGS)
def test_every_harness_tag_is_cut_after_the_users_words(tag: str) -> None:
    """Found by the third review: dropping any one tag from the list left
    every test green while real records leaked (five `bash-input`, ten
    `local-command-stdout`). One case per tag pins the whole list."""
    speech = _user_speech(USER + "\n" + _wrap(tag))
    assert speech == USER, (tag, speech)


@pytest.mark.parametrize("line", [ln for ln in BANNER.splitlines() if ln])
def test_each_banner_line_is_cut_on_its_own(line: str) -> None:
    """Each of the host's four banner lines goes, whichever of them a
    record carries, and the user's words beside it stay."""
    assert _user_speech(line + "\n" + USER) == USER


@pytest.mark.parametrize("text", [
    "<system-reminder>\nfirst line\nsecond line\n</system-reminder>\n"
    "<command-name>/aelf:onboard</command-name>\n" + REPORT,
    "<SYSTEM-REMINDER>x</SYSTEM-REMINDER>\n"
    "<command-name>/aelf:onboard</command-name>\n" + REPORT,
    "<command-args>.</command-args>\n" + REPORT,
    "<bash-stdout>ok</bash-stdout>",
    "\n<system-reminder>x</system-reminder>\n"
    "<command-name>/aelf:onboard</command-name>\n" + REPORT,
], ids=["multiline-reminder-then-command", "uppercase-reminder-then-command",
        "command-args-opens", "only-a-block", "blank-line-reminder-then-command"])
def test_nothing_of_these_records_is_the_users(text: str) -> None:
    """Found by the fourth review: real reminders span lines, and one in
    front of a slash command let the command's body through when the
    reminder pattern could not cross a newline. A record that is only a
    block yields nothing at all, not an empty string."""
    assert _user_speech(text) is None


@pytest.mark.parametrize("text", [
    "<bash-stdout>x</bash-stdout> " + USER,
    USER + "\n<BASH-STDOUT>\n" + REPORT,
    "<resultant> forces were measured twice.\n" + USER,
    "<summaryx> pyright notes </summary> stay mine.",
], ids=["words-after-closer", "unclosed-uppercase", "word-that-starts-like-a-tag",
        "tag-prefix-with-a-closer"])
def test_the_users_words_survive_the_block_boundaries(text: str) -> None:
    """Words after a closer on its line are the user's; an unclosed block
    is matched in any case; a word that merely starts like a tag is not
    a tag."""
    speech = _user_speech(text) or ""
    assert "pyright" in speech, speech
    assert "mutation harness" not in speech, speech


def test_a_dropped_record_is_counted_as_skipped(tmp_path: Path) -> None:
    log = _write(tmp_path / "turns.jsonl", [_tl(NOTICE), _tl(USER)])
    store = MemoryStore(str(tmp_path / "memory.db"))
    try:
        result = ingest_jsonl(store, log)
    finally:
        store.close()
    assert result.skipped_lines == 1, result
    assert result.turns_ingested == 1, result


def test_an_unclosed_reminder_is_cut_to_the_end() -> None:
    """The reminder pass only matches a closed block; an unclosed one is
    left to the tag list's unclosed-block pass."""
    assert _user_speech(USER + "\n<system-reminder>\n" + REPORT) == USER


@pytest.mark.parametrize("tag", ["bash-stdout", "system-reminder"])
def test_the_users_words_between_two_blocks_are_kept(tag: str) -> None:
    """A greedy match would run from the first block's opener to the last
    block's closer and take the user's words between them."""
    text = f"{_wrap(tag, 'first')}\n{USER}\n{_wrap(tag, 'second')}"
    assert _user_speech(text) == USER
