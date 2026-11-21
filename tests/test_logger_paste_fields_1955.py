"""The logger's user turn line counts its `<pasted_content>` pastes (#1955).

Each user line carries `paste_blocks`, the number of pastes, and
`paste_bytes`, the UTF-8 bytes of their text. Both are integers; no text
is added. The counts read the text ingest segments, cut by the same
code, and segment it with the same scanner. A line without tags carries
zeros, and the readers of the log accept the new fields.
"""
from __future__ import annotations

import io
import itertools
import json
import random
import re
import sys
import time
from pathlib import Path

import pytest

from aelfrice import ingest, noise_filter
from aelfrice import transcript_logger as tl
from aelfrice.context_rebuilder import read_recent_turns_aelfrice
from aelfrice.ingest import ingest_jsonl, user_turn_key
from aelfrice.noise_filter import (
    HARNESS_TAG_ALT,
    paste_fields,
    paste_segments,
    speech_before_pastes,
)
from aelfrice.store import MemoryStore


def _block(pid: str, body: str) -> str:
    return f'<pasted_content id="{pid}">\n{body}\n</pasted_content id="{pid}">'


def _nbytes(s: str) -> int:
    return len(s.encode("utf-8"))


@pytest.fixture
def tdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("AELFRICE_TRANSCRIPTS_DIR", str(tmp_path))
    return tmp_path


def _log_prompt_err(prompt: str) -> tuple[dict[str, object], list[str]]:
    """Log `prompt`; return its line and the `aelf-logger:` lines main's stderr got."""
    serr = io.StringIO()
    rc = tl.main(
        stdin=io.StringIO(json.dumps({
            "hook_event_name": "UserPromptSubmit",
            "session_id": "sess-1955",
            "prompt": prompt,
        })),
        stderr=serr,
    )
    assert rc == 0
    lines = tl.turns_path().read_text(encoding="utf-8").splitlines()
    notes = [ln for ln in serr.getvalue().splitlines() if ln.startswith("aelf-logger:")]
    return json.loads(lines[-1]), notes


def _log_prompt(prompt: str) -> dict[str, object]:
    return _log_prompt_err(prompt)[0]


# --- the counter --------------------------------------------------------

def test_no_tags_give_zeros() -> None:
    assert paste_fields("Please check the storage layer.") == (0, 0)


def test_one_block_counts_its_text_with_line_breaks() -> None:
    assert paste_fields("see " + _block("1", "abc") + " thanks") == (1, 5)


def test_bytes_are_utf8_not_characters() -> None:
    # "é" is 2 bytes and "€" is 3: 2 characters, 5 bytes, plus 2 line breaks.
    assert paste_fields(_block("1", "é€")) == (1, 7)


def test_two_blocks_sum() -> None:
    text = _block("1", "ab") + "\nand\n" + _block("2", "cde")
    assert paste_fields(text) == (2, 4 + 5)


def test_unclosed_opener_runs_to_the_end() -> None:
    assert paste_fields('<pasted_content id="1">\nabc') == (1, 4)


def test_closer_of_another_id_closes() -> None:
    text = '<pasted_content id="1">\nab\n</pasted_content id="2">tail'
    assert paste_fields(text) == (1, 4)


def test_bare_closer_and_extra_attributes_close() -> None:
    text = '<pasted_content id="1" kind="x">\nab\n</pasted_content>tail'
    assert paste_fields(text) == (1, 4)


def test_stray_closer_is_dropped() -> None:
    text = '</pasted_content id="1">\n' + _block("2", "ab")
    assert paste_fields(text) == (1, 4)


def test_inner_opener_is_dropped_and_the_paste_goes_on() -> None:
    """A nested-looking opener neither starts a block nor counts as text."""
    text = '<pasted_content id="2">\nab<pasted_content id="1">cd\n</pasted_content id="2">'
    assert paste_fields(text) == (1, _nbytes("\nab") + _nbytes("cd\n"))


def test_nested_looking_block_ends_at_the_inner_closer() -> None:
    inner = _block("1", "in")
    text = _block("2", inner)
    # The paste runs from the outer opener to the inner closer; the outer
    # closer is then a stray.
    assert paste_fields(text) == (1, _nbytes("\n") + _nbytes("\nin\n"))


def test_tags_match_case_insensitively() -> None:
    text = '<PASTED_CONTENT id="1">\nab\n</Pasted_Content id="1">'
    assert paste_fields(text) == (1, 4)


def test_empty_block_counts_with_zero_bytes() -> None:
    assert paste_fields('<pasted_content id="1"></pasted_content id="1">') == (1, 0)


class _CountingText(str):
    """A str that counts the characters its searches and slices cover."""

    scanned = 0

    def _span(self, start: int | None, end: int | None) -> int:
        lo = 0 if start is None else start
        hi = len(self) if end is None else end
        return max(0, hi - lo)

    def find(self, sub, start=None, end=None):  # type: ignore[override]
        _CountingText.scanned += self._span(start, end)
        return super().find(sub, start, end)

    def index(self, sub, start=None, end=None):  # type: ignore[override]
        _CountingText.scanned += self._span(start, end)
        return super().index(sub, start, end)

    def __getitem__(self, key):  # type: ignore[override]
        out = super().__getitem__(key)
        _CountingText.scanned += len(out)
        return out


class _CountingPattern:
    """`PASTE_TAG_RE` with every scan's covered length added to the count."""

    def __init__(self, pattern: "re.Pattern[str]") -> None:
        self._p = pattern

    def _count(self, text: str, pos: int, endpos: int | None) -> None:
        _CountingText.scanned += max(0, (len(text) if endpos is None else endpos) - pos)

    def finditer(self, text: str, pos: int = 0, endpos: int | None = None):
        self._count(text, pos, endpos)
        return self._p.finditer(text, pos, len(text) if endpos is None else endpos)

    def search(self, text: str, pos: int = 0, endpos: int | None = None):
        self._count(text, pos, endpos)
        return self._p.search(text, pos, len(text) if endpos is None else endpos)

    def match(self, text: str, pos: int = 0, endpos: int | None = None):
        self._count(text, pos, endpos)
        return self._p.match(text, pos, len(text) if endpos is None else endpos)


def test_counter_is_linear_on_many_unclosed_openers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """5,000 openers and no closer: one paste, in a linear number of steps.

    Counts the characters every tag search and every slice covers, so a scan
    that looks ahead for a closer from each opener (about 5,000 x 60 KB here)
    fails the bound without reading a clock.
    """
    monkeypatch.setattr(noise_filter, "PASTE_TAG_RE", _CountingPattern(noise_filter.PASTE_TAG_RE))
    _CountingText.scanned = 0
    text = _CountingText('<pasted_content id="u">\n' * 5_000)
    segments = paste_segments(text)
    assert {b for b, _, _ in segments} == {-1, 0}
    assert sum(e - s for b, s, e in segments if b == 0) == 5_000
    # Ingest strips the record first, which takes the final line break.
    assert paste_fields(str(text)) == (1, 4_999)
    assert _CountingText.scanned <= 4 * len(text), (
        f"scanned {_CountingText.scanned} characters for a {len(text)}-character prompt"
    )


# --- the counter agrees with ingest --------------------------------------

# The review's cases: what ingest cuts before it reads pastes.
_IN_REMINDER = (
    "<system-reminder>\n" + _block("1", "abc") + "\n</system-reminder>\nhello"
)
_UNCLOSED_HARNESS_IN_PASTE = (
    _block("1", "x\n<task-notification>\nhalf") + "\nthen " + _block("2", "second paste")
)
_RESULT_LINE_IN_PASTE = _block("1", "<result>\nexit code 0 here\n</result>\nk")
# An unclosed harness opener inside an opening relay: the relay cut takes
# it, so it can't cut the user's paste after the relay.
_RELAYED_PASTE = (
    '<teammate-message teammate_id="x">\n<task-notification>\nhalf\n'
    "</teammate-message>\nmine " + _block("1", "kept")
)
_REMINDER_THEN_COMMAND = (
    "<system-reminder>\nnote\n</system-reminder>\n"
    "<command-name>/x</command-name>\n" + _block("1", "body")
)

_AGREEMENT_INPUTS = [
    "plain typed text",
    _block("1", "one line"),
    _block("1", "a") + " mid " + _block("2", "b"),
    '<pasted_content id="1">\nunclosed to the end',
    '<pasted_content id="1">\nab\n</pasted_content id="2">tail',
    '<pasted_content id="1" kind="x">\nab\n</pasted_content>tail',
    '</pasted_content id="9">typed' + _block("2", "p"),
    '<pasted_content id="2">\nab<pasted_content id="1">cd\n</pasted_content id="2">',
    _block("2", _block("1", "nested")),
    '<PASTED_CONTENT id="1">\nab\n</Pasted_Content id="1">',
    "typed <pasted_content>x</pasted_content> typed",
    _IN_REMINDER,
    _UNCLOSED_HARNESS_IN_PASTE,
    _RESULT_LINE_IN_PASTE,
    '<command-name>/x</command-name>\n' + _block("1", "wrapped"),
    _REMINDER_THEN_COMMAND,
    _RELAYED_PASTE,
]


def _ingest_view(
    text: str, monkeypatch: pytest.MonkeyPatch,
) -> tuple[int, int, int]:
    """What `ingest._user_speech` reads as pasted in `text`.

    Returns (pastes and bytes in the text it hands `paste_segments`, bytes
    `_filter_pastes` then hands `_pasted_prose`).
    """
    seen: list[tuple[str, list[tuple[int, int, int]]]] = []
    prose: list[str] = []
    real = ingest.paste_segments

    def _spy(t: str) -> list[tuple[int, int, int]]:
        segs = real(t)
        seen.append((t, segs))
        return segs

    def _record(body: str) -> str:
        prose.append(body)
        return ""

    monkeypatch.setattr(ingest, "paste_segments", _spy)
    monkeypatch.setattr(ingest, "_pasted_prose", _record)
    ingest._user_speech(text)  # pyright: ignore[reportPrivateUsage]
    prose_bytes = sum(_nbytes(b) for b in prose)
    if not seen:
        return 0, 0, prose_bytes
    ((t, segs),) = seen
    blocks = 1 + max((b for b, _, _ in segs), default=-1)
    seg_bytes = sum(_nbytes(t[s:e]) for b, s, e in segs if b >= 0)
    return blocks, seg_bytes, prose_bytes


@pytest.mark.parametrize("text", _AGREEMENT_INPUTS)
def test_counts_equal_what_ingest_reads_as_pasted(
    text: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocks, seg_bytes, prose_bytes = _ingest_view(text, monkeypatch)
    assert paste_fields(text) == (blocks, seg_bytes)
    assert paste_fields(text)[1] == prose_bytes


def test_paste_inside_a_reminder_does_not_count() -> None:
    assert paste_fields(_IN_REMINDER) == (0, 0)


def test_unclosed_harness_opener_cuts_the_rest() -> None:
    """Ingest cuts from the opener to the end, second paste included."""
    assert paste_fields(_UNCLOSED_HARNESS_IN_PASTE) == (1, _nbytes("\nx\n"))


def test_harness_block_inside_a_paste_is_cut() -> None:
    assert paste_fields(_RESULT_LINE_IN_PASTE) == (1, _nbytes("\n\nk\n"))


def test_relay_cut_runs_before_the_harness_cut() -> None:
    assert paste_fields(_RELAYED_PASTE) == (1, len("\nkept\n"))


def test_reminder_before_a_slash_command_counts_nothing() -> None:
    """The reminder is cut first, so the record opens with the wrapper."""
    assert paste_fields(_REMINDER_THEN_COMMAND) == (0, 0)


def test_slash_command_record_counts_nothing() -> None:
    assert paste_fields('<command-name>/x</command-name>\n' + _block("1", "w")) == (0, 0)


# --- the linear cut equals the regex cut ---------------------------------

# The regex cut as it shipped before #1955 made it linear, kept here as the
# reference. Patterns copied, not imported, so a change to the module can't
# change the reference too.
_OLD_REMINDER = re.compile(
    r"^[ \t]*<system-reminder\b[^>]*>.*?</system-reminder\s*>",
    re.DOTALL | re.IGNORECASE | re.MULTILINE,
)
_OLD_HARNESS = re.compile(
    r"^[ \t]*<(" + HARNESS_TAG_ALT + r")\b[^>]*>.*?</\1\s*>",
    re.DOTALL | re.IGNORECASE | re.MULTILINE,
)
_OLD_UNCLOSED = re.compile(
    r"^[ \t]*<(?:" + HARNESS_TAG_ALT + r")\b[^>]*>.*\Z",
    re.DOTALL | re.IGNORECASE | re.MULTILINE,
)


def _old_speech_before_pastes(cut: str) -> str | None:
    rest = noise_filter.HOST_MARKER_LINE_RE.sub("", cut)
    rest = _OLD_REMINDER.sub("", rest).strip()
    if not rest or noise_filter.COMMAND_WRAPPER_RE.match(rest):
        return None
    rest = _OLD_HARNESS.sub("", rest)
    return _OLD_UNCLOSED.sub("", rest)


_CUT_PIECES = [
    "<system-reminder>", "<SYSTEM-Reminder x='1'>", "<system-reminder", "</system-reminder>",
    "</system-reminder \n>", "</System-Reminder>", "<result>", "</result>", "<RESULT x>",
    "</Result >", "<status>", "</status>", "<result-x>", "<task-id>", "</task-id>",
    "<task-notification>", "</task-notification>", "<tool-result id=2>", "</tool-result>",
    "<local-command-stdout>", "</local-command-stdout>", "<command-name>/x</command-name>",
    '<pasted_content id="1">', '</pasted_content id="1">', "<pasted_content>",
    "[Request interrupted by user]", ">", "<", "\n", "\n", "\n", " ", "\t", "  ",
    "words", "more text", "\u00e9t\u00e9",
    # Spellings the engine's case-blind match accepts and str.lower doesn't
    # map to ASCII: long s, Kelvin sign, dotless and dotted i.
    "<\u017ftatus>", "</\u017ftatus>", "</\u017fTATUS>", "<\u017fTATUS x>",
    "<tas\u212a-id>", "</tas\u212a-id>", "</TASK-ID>", "<\u017fystem-reminder>",
    "</\u017fystem-reminder>", "<task-notificat\u0131on>", "</task-notificat\u0131on>",
    "</TASK-NOTIFICATION>", "<task-notificat\u0130on>", "</task-notificat\u0130on>",
    # A long s as the last character: names that differ only there.
    "<statuſ>", "</statuſ>", "<command-argſ>", "</command-argſ>",
    "<command-args>", "</command-args>",
]


@pytest.mark.parametrize("seed", range(8))
def test_linear_cut_equals_the_regex_cut(seed: int) -> None:
    rng = random.Random(1955 + seed)
    for _ in range(600):
        text = "".join(rng.choice(_CUT_PIECES) for _ in range(rng.randrange(1, 30)))
        assert speech_before_pastes(text) == _old_speech_before_pastes(text), repr(text)


@pytest.mark.parametrize("text", _AGREEMENT_INPUTS)
def test_linear_cut_equals_the_regex_cut_on_the_agreement_inputs(text: str) -> None:
    assert speech_before_pastes(text) == _old_speech_before_pastes(text)


@pytest.mark.parametrize("text", [
    # The engine's backreference refuses long s and dotless i against ASCII,
    # so these openers find no closer and the next opener of that tag must.
    "<\u017ftatus>\n<status>\n</status>\nafter",
    "<status>\n<\u017ftatus>\nx\n</\u017ftatus>\n</status>\nafter",
    "<task-notificat\u0131on>\nx\n</task-notification>\n<task-notification>\ny\n"
    "</task-notification>\nafter",
    "<tas\u212a-id>\nx\n</task-id>\nafter",
    "<\u017fystem-reminder>\nx\n</system-reminder>\nafter",
])
def test_non_ascii_tag_names_cut_as_the_regex_does(text: str) -> None:
    assert speech_before_pastes(text) == _old_speech_before_pastes(text)
    assert _exact_spans(text) == (
        [m.span() for m in _OLD_REMINDER.finditer(text)],
        [m.span() for m in _OLD_HARNESS.finditer(text)],
    )


def _exact_spans(text: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """The exact span finder alone, for both patterns.

    In `speech_before_pastes` the unclosed cut removes everything from an
    opener that found no closer, which can hide a wrong span after it, so
    the spans are compared directly too.
    """
    nf = noise_filter
    return (
        nf._exact_block_spans(  # pyright: ignore[reportPrivateUsage]
            text, nf._REMINDER_OPENER_IDX_RE, nf._REMINDER_CLOSER_IDX_RE,  # pyright: ignore[reportPrivateUsage]
            backref=False,
        ),
        nf._exact_block_spans(  # pyright: ignore[reportPrivateUsage]
            text, nf._HARNESS_OPENER_IDX_RE, nf._HARNESS_CLOSER_IDX_RE,  # pyright: ignore[reportPrivateUsage]
            backref=True,
        ),
    )


@pytest.mark.parametrize("seed", range(4))
def test_exact_spans_equal_the_regex_spans(seed: int) -> None:
    rng = random.Random(7_1955 + seed)
    for _ in range(600):
        text = "".join(rng.choice(_CUT_PIECES) for _ in range(rng.randrange(1, 30)))
        assert _exact_spans(text) == (
            [m.span() for m in _OLD_REMINDER.finditer(text)],
            [m.span() for m in _OLD_HARNESS.finditer(text)],
        ), repr(text)


class _NoRegexCut:
    """Stands in for a lazy block pattern; any `sub` is the quadratic path."""

    def sub(self, *_a: object, **_k: object) -> str:
        raise AssertionError("the lazy regex cut ran")


@pytest.mark.parametrize("tail", [
    "\n</\u017ftatus>",
    "\n<\u017ftatus>\nx\n</status>",
    "\n</\u017fystem-reminder>",
    "\n<tas\u212a-id>\n</TAS\u212a-ID>",
])
def test_non_ascii_tag_names_never_reach_the_lazy_regex(
    tail: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One non-ASCII tag in a flood used to send the whole cut to the regex.

    Ingest then took quadratic time on the flood. The cut now reads any
    spelling in one scan, so the regex never runs, and the output still
    equals the regex reference.
    """
    text = "hi\n<pasted_content>" + "\n<result>" * 2_000 + tail
    expected = _old_speech_before_pastes(text)
    monkeypatch.setattr(noise_filter, "SYSTEM_REMINDER_RE", _NoRegexCut())
    monkeypatch.setattr(noise_filter, "HARNESS_BLOCK_RE", _NoRegexCut())
    monkeypatch.setattr(noise_filter, "UNCLOSED_HARNESS_RE", _NoRegexCut())
    assert speech_before_pastes(text) == expected
    speech = None if expected is None else (
        ingest._filter_pastes(expected).strip() or None  # pyright: ignore[reportPrivateUsage]
    )
    assert ingest._user_speech(text) == speech  # pyright: ignore[reportPrivateUsage]


def _long_s_spellings(tag: str) -> list[str]:
    """Every spelling of `tag` the engine reads as it, with at least one long s."""
    opts = [
        ["s", "\u017f"] if c == "s" else ["i", "\u0130", "\u0131"] if c == "i"
        else ["k", "\u212a"] if c == "k" else [c]
        for c in tag
    ]
    return [w for w in map("".join, itertools.product(*opts)) if "\u017f" in w]


def test_handler_is_fast_on_many_opener_spellings_and_closers(tdir: Path) -> None:
    """Hundreds of opener spellings the ASCII closer refuses, then its closers.

    A cursor per opener spelling walked every closer once per spelling, 31.6M
    backreference checks at the size bound. Closers are now grouped by name
    class, one cursor per group, so this takes milliseconds. A wall-clock
    budget, listed in test_wallclock_budget_census_1473.py with its reason.
    """
    tag = "cross-session-message"
    spellings = _long_s_spellings(tag)
    assert len(spellings) > 300
    head = "".join("<" + w + ">\n" for w in spellings)
    closer = "</" + tag + ">"
    prompt = "see\n<pasted_content>\n" + head + closer * ((360_000 - len(head)) // len(closer))
    assert len(prompt) > 350_000
    t0 = time.perf_counter()
    line = _log_prompt(prompt)
    elapsed = time.perf_counter() - t0
    assert line["text"] == prompt
    assert "paste_blocks" in line
    assert elapsed < 0.5, f"the handler took {elapsed:.2f} s"


@pytest.mark.parametrize("opener", ["<result>", "<system-reminder>", "<result"])
def test_handler_is_fast_on_a_flood_of_unclosed_openers(
    tdir: Path, opener: str,
) -> None:
    """360,000 characters of unclosed openers at line starts inside a paste.

    The regex cut rescanned the rest of the prompt from every opener and
    took about 51 s here; the linear cut takes milliseconds. A wall-clock
    budget, listed in test_wallclock_budget_census_1473.py with its reason.
    """
    body = (opener + "\n") * (360_000 // (len(opener) + 1) + 1)
    prompt = "Look at this log.\n<pasted_content>\n" + body
    assert len(prompt) > 360_000
    t0 = time.perf_counter()
    line = _log_prompt(prompt)
    elapsed = time.perf_counter() - t0
    assert line["text"] == prompt
    assert "paste_blocks" in line
    assert elapsed < 0.5, f"the handler took {elapsed:.2f} s"


def test_a_prompt_over_the_size_bound_keeps_the_line_without_fields(
    tdir: Path,
) -> None:
    prompt = "x " * (tl.PASTE_COUNT_MAX_CHARS // 2) + _block("1", "p")
    assert len(prompt) > tl.PASTE_COUNT_MAX_CHARS
    line, err = _log_prompt_err(prompt)
    assert line["text"] == prompt
    assert "paste_blocks" not in line and "paste_bytes" not in line
    assert err == [
        f"aelf-logger: paste counts skipped (prompt over {tl.PASTE_COUNT_MAX_CHARS} characters)"
    ]


def test_a_prompt_at_the_size_bound_is_counted(tdir: Path) -> None:
    block = _block("1", "p")
    prompt = "x" * (tl.PASTE_COUNT_MAX_CHARS - len(block)) + block
    assert len(prompt) == tl.PASTE_COUNT_MAX_CHARS
    line = _log_prompt(prompt)
    assert (line["paste_blocks"], line["paste_bytes"]) == (1, 3)


def test_the_size_bound_is_two_million_characters() -> None:
    assert tl.PASTE_COUNT_MAX_CHARS == 2_000_000


def test_a_non_ascii_tag_is_counted_like_any_other(tdir: Path) -> None:
    prompt = "see\n<\u017ftatus>\nx\n</status>\n" + _block("1", "p")
    line, err = _log_prompt_err(prompt)
    assert (line["paste_blocks"], line["paste_bytes"]) == paste_fields(prompt)
    assert err == []


class _BrokenStream(io.TextIOBase):
    """A stderr whose every write fails, as a closed pipe does."""

    def writable(self) -> bool:
        return True

    def write(self, _s: str) -> int:
        raise BrokenPipeError(32, "Broken pipe")


@pytest.mark.parametrize("case", ["over_bound", "counter_error"])
def test_a_failing_stderr_never_costs_the_line(
    tdir: Path, monkeypatch: pytest.MonkeyPatch, case: str,
) -> None:
    """Both stderr streams raise on write; the user line is still appended."""
    if case == "over_bound":
        prompt = "x " * (tl.PASTE_COUNT_MAX_CHARS // 2 + 1)
    else:
        prompt = _FALLBACK_PROMPT

        def _boom(_text: str) -> tuple[int, int]:
            raise RuntimeError("counter bug")

        monkeypatch.setattr(noise_filter, "paste_fields", _boom)
    broken = _BrokenStream()
    monkeypatch.setattr(sys, "stderr", broken)
    rc = tl.main(
        stdin=io.StringIO(json.dumps({
            "hook_event_name": "UserPromptSubmit",
            "session_id": "sess-1955",
            "prompt": prompt,
        })),
        stderr=broken,
    )
    assert rc == 0
    (raw,) = tl.turns_path().read_text(encoding="utf-8").splitlines()
    line = json.loads(raw)
    assert line["text"] == prompt
    assert "paste_blocks" not in line


# --- the logged line ---------------------------------------------------

def test_user_line_carries_zero_fields_without_tags(tdir: Path) -> None:
    line = _log_prompt("What does the project use for storage?")
    assert (line["paste_blocks"], line["paste_bytes"]) == (0, 0)


def test_user_line_carries_counts_and_no_extra_text(tdir: Path) -> None:
    prompt = "Look at this log:\n" + _block("1", "error at line four")
    line = _log_prompt(prompt)
    assert (line["paste_blocks"], line["paste_bytes"]) == (1, 20)
    assert type(line["paste_blocks"]) is int
    assert type(line["paste_bytes"]) is int
    assert line["text"] == prompt
    assert set(line) == {
        "schema_version", "ts", "role", "text", "session_id", "turn_id",
        "context", "paste_blocks", "paste_bytes",
    }


def test_assistant_line_has_no_paste_fields(
    tdir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tl, "_maybe_stop_flush", lambda _d: None)
    rc = tl.main(
        stdin=io.StringIO(json.dumps({
            "hook_event_name": "Stop",
            "session_id": "sess-1955",
            "transcript_path": None,
            "last_assistant_message": _block("1", "quoted"),
        })),
        stderr=io.StringIO(),
    )
    assert rc == 0
    (line,) = [
        json.loads(ln)
        for ln in tl.turns_path().read_text(encoding="utf-8").splitlines()
    ]
    assert line["role"] == "assistant"
    assert "paste_blocks" not in line and "paste_bytes" not in line


_FALLBACK_PROMPT = "Look at this:\n" + _block("1", "pasted")


def _assert_line_without_fields(err: list[str], exc_name: str) -> None:
    (raw,) = tl.turns_path().read_text(encoding="utf-8").splitlines()
    line = json.loads(raw)
    assert line["role"] == "user"
    assert line["text"] == _FALLBACK_PROMPT
    assert "paste_blocks" not in line and "paste_bytes" not in line
    assert err == [f"aelf-logger: paste counts skipped ({exc_name})"]


def _block_import(monkeypatch: pytest.MonkeyPatch, exc: Exception) -> None:
    import builtins

    real_import = builtins.__import__

    def _guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "aelfrice.noise_filter" and "paste_fields" in (fromlist or ()):
            raise exc
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _guarded)


def test_a_failed_import_keeps_the_line_without_fields(
    tdir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_import(monkeypatch, ImportError("blocked by test"))
    _, err = _log_prompt_err(_FALLBACK_PROMPT)
    _assert_line_without_fields(err, "ImportError")


def test_an_import_raising_something_else_keeps_the_line(
    tdir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_import(monkeypatch, RuntimeError("module body raised"))
    _, err = _log_prompt_err(_FALLBACK_PROMPT)
    _assert_line_without_fields(err, "RuntimeError")


def test_a_cut_relay_error_keeps_the_line_without_fields(
    tdir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The count's own `cut_relay` raising costs the counts, not the line.

    The noise gate imports `cut_relay` by name before the count does, so
    patching the module attribute reaches both; the gate swallows its own
    error and the line is still the whole prompt.
    """
    def _boom(_text: str) -> str:
        raise ValueError("cut_relay bug")

    monkeypatch.setattr(noise_filter, "cut_relay", _boom)
    _, err = _log_prompt_err(_FALLBACK_PROMPT)
    _assert_line_without_fields(err, "ValueError")


def test_a_counter_error_keeps_the_line_without_fields(
    tdir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(_text: str) -> tuple[int, int]:
        raise RuntimeError("counter bug")

    monkeypatch.setattr(noise_filter, "paste_fields", _boom)
    _, err = _log_prompt_err(_FALLBACK_PROMPT)
    _assert_line_without_fields(err, "RuntimeError")


def test_logged_counts_equal_what_ingest_reads_from_the_logged_line(
    tdir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Round trip: a relay-opened prompt through the logger, then ingest_jsonl.

    The logger writes the relay cut, and ingest cuts the line again; the
    counts on the line must equal the pastes in the text ingest segments.
    """
    prompt = (
        '<teammate-message teammate_id="lead">\nrelayed report\n'
        "</teammate-message>\nThe cellar door sticks, see " + _block("1", "the cellar log")
        + "\nand " + _block("2", "a second cellar note")
    )
    line = _log_prompt(prompt)
    assert line["text"] != prompt, "the relay cut must reach the logged line"
    seen: list[tuple[str, list[tuple[int, int, int]]]] = []
    real = ingest.paste_segments

    def _spy(t: str) -> list[tuple[int, int, int]]:
        segs = real(t)
        seen.append((t, segs))
        return segs

    monkeypatch.setattr(ingest, "paste_segments", _spy)
    store = MemoryStore(str(tmp_path / "memory.db"))
    try:
        ingest_jsonl(store, tl.turns_path())
    finally:
        store.close()
    assert seen, "ingest must read the logged line's pastes"
    ((t, segs),) = seen[:1]
    blocks = 1 + max((b for b, _, _ in segs), default=-1)
    nbytes = sum(_nbytes(t[s:e]) for b, s, e in segs if b >= 0)
    assert (line["paste_blocks"], line["paste_bytes"]) == (blocks, nbytes) == (2, 38)


# --- readers -------------------------------------------------------------

def test_readers_accept_lines_with_paste_fields(
    tdir: Path, tmp_path: Path,
) -> None:
    prompt = "The cellar stores twelve barrels of cider.\n" + _block(
        "1", "The cellar door sticks in winter.",
    )
    _log_prompt(prompt)
    (raw,) = tl.turns_path().read_text(encoding="utf-8").splitlines()
    record = json.loads(raw)
    # The line under test must carry the fields, or this checks nothing new.
    assert record["paste_blocks"] == 1
    turns = read_recent_turns_aelfrice(tl.turns_path(), 10)
    assert [t.text for t in turns] == [prompt]
    assert user_turn_key(record) is not None
    store = MemoryStore(str(tmp_path / "memory.db"))
    try:
        result = ingest_jsonl(store, tl.turns_path())
        assert result.lines_read == 1
        assert result.skipped_lines == 0
    finally:
        store.close()
