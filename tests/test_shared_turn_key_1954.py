"""The host's `prompt_id` is the turn id both writers record (#1954).

`aelf-hook` and `aelf-transcript-logger` run as separate processes on one
UserPromptSubmit. When the payload carries a valid `prompt_id`, the
hook's `injection_events.turn_id` and the logger's user turn line
`turn_id` are that value, so they join exactly. When it is absent or
invalid, each writer mints its own id, as before.
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

import pytest

from aelfrice import transcript_logger as tl
from aelfrice.hook import _extract_prompt_id, user_prompt_submit
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief
from aelfrice.store import MemoryStore
from aelfrice.hook_payload import PROMPT_ID_MAX_CHARS, payload_prompt_id

_PROMPT_ID = "3f2b8c1e-6a4d-4f0b-9c7e-2d5a1b8e4f60"
_SESSION = "sess-1954"
_PROMPT = "how many barrels are in the cellar door storage"
_MINTED = re.compile(r"^\d{8}T\d{12}Z-[0-9a-f]{8}$")


def _payload(**extra: object) -> dict[str, object]:
    base: dict[str, object] = {
        "session_id": _SESSION,
        "transcript_path": "/dev/null",
        "cwd": "/tmp",
        "hook_event_name": "UserPromptSubmit",
        "prompt": _PROMPT,
    }
    base.update(extra)
    return base


def _seed(db: Path) -> None:
    s = MemoryStore(str(db))
    try:
        s.insert_belief(Belief(
            id="HIT1954",
            content="the cellar door is full of barrels and casks",
            content_hash="h_HIT1954",
            alpha=1.0,
            beta=1.0,
            type=BELIEF_FACTUAL,
            lock_level=LOCK_NONE,
            locked_at=None,
            created_at="2026-05-14T00:00:00+00:00",
            last_retrieved_at=None,
        ))
    finally:
        s.close()


def _injection_turn_ids(db: Path) -> set[str]:
    s = MemoryStore(str(db))
    try:
        rows = s._conn.execute(  # pyright: ignore[reportPrivateUsage]
            "SELECT turn_id FROM injection_events WHERE session_id = ?",
            (_SESSION,),
        ).fetchall()
        return {str(r[0]) for r in rows}
    finally:
        s.close()


def _logged_lines(tdir: Path) -> list[dict[str, object]]:
    path = tdir / "turns.jsonl"
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _fire_hook(payload: dict[str, object]) -> None:
    rc = user_prompt_submit(
        stdin=io.StringIO(json.dumps(payload)),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    assert rc == 0


def _fire_logger(payload: dict[str, object]) -> None:
    rc = tl.main(stdin=io.StringIO(json.dumps(payload)), stderr=io.StringIO())
    assert rc == 0


@pytest.fixture
def roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    db = tmp_path / "memory.db"
    tdir = tmp_path / "transcripts"
    _seed(db)
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.setenv("AELFRICE_TRANSCRIPTS_DIR", str(tdir))
    return db, tdir


# --- validation --------------------------------------------------------

def test_valid_prompt_id_is_returned() -> None:
    assert payload_prompt_id({"prompt_id": _PROMPT_ID}) == _PROMPT_ID


def test_prompt_id_at_the_length_bound_is_accepted() -> None:
    # Literal, not derived from the constant, so moving the bound fails here.
    assert PROMPT_ID_MAX_CHARS == 128
    value = "a" * 128
    assert payload_prompt_id({"prompt_id": value}) == value


def test_prompt_id_uses_the_whole_allowlist() -> None:
    value = "AZaz09_-"
    assert payload_prompt_id({"prompt_id": value}) == value


_SURROGATE = "ab\ud800"
_PADDED = "  P1  "
_NON_ASCII = "P\u00e91"

# Values the writers must refuse: a lone surrogate can't be encoded to
# UTF-8, so accepting it loses the logger's line and the hook's batch.
_REFUSED_ON_BOTH_WRITERS = [
    pytest.param(_SURROGATE, id="lone-surrogate"),
    pytest.param(_PADDED, id="padded"),
    pytest.param(_NON_ASCII, id="non-ascii-letter"),
]


@pytest.mark.parametrize("payload", [
    pytest.param({}, id="missing"),
    pytest.param({"prompt_id": None}, id="null"),
    pytest.param({"prompt_id": 1954}, id="int"),
    pytest.param({"prompt_id": [_PROMPT_ID]}, id="list"),
    pytest.param({"prompt_id": ""}, id="empty"),
    pytest.param({"prompt_id": "   "}, id="whitespace"),
    pytest.param({"prompt_id": "a" * 129}, id="too-long"),
    pytest.param({"prompt_id": "abc\ndef"}, id="newline"),
    pytest.param({"prompt_id": "abc\x00"}, id="nul"),
    pytest.param({"prompt_id": "\x1fabc"}, id="unit-separator"),
    pytest.param({"prompt_id": "abc\x7f"}, id="del"),
    pytest.param({"prompt_id": "abc\x9f"}, id="c1-last"),
    pytest.param({"prompt_id": "abc\x80"}, id="c1-first"),
    pytest.param({"prompt_id": "a b"}, id="inner-space"),
    pytest.param({"prompt_id": "a~b"}, id="tilde"),
    pytest.param({"prompt_id": "a.b"}, id="dot"),
    pytest.param({"prompt_id": _PADDED}, id="padded"),
    pytest.param({"prompt_id": "P1\n"}, id="trailing-newline"),
    pytest.param({"prompt_id": _SURROGATE}, id="lone-surrogate"),
    pytest.param({"prompt_id": _NON_ASCII}, id="latin-letter"),
    pytest.param({"prompt_id": "\u043f\u0440\u043e"}, id="cyrillic"),
    pytest.param({"prompt_id": "\uff11\uff12"}, id="fullwidth-digits"),
])
def test_invalid_prompt_id_reads_as_absent(payload: dict[str, object]) -> None:
    assert payload_prompt_id(payload) is None


def test_hook_extractor_reads_raw_payload() -> None:
    assert _extract_prompt_id(json.dumps({"prompt_id": _PROMPT_ID})) == _PROMPT_ID
    assert _extract_prompt_id(json.dumps({"prompt_id": "x\ny"})) is None
    assert _extract_prompt_id("not json") is None
    assert _extract_prompt_id("[1, 2]") is None
    assert _extract_prompt_id("") is None


# --- each writer on its own ---------------------------------------------

def test_logger_user_line_uses_prompt_id(roots: tuple[Path, Path]) -> None:
    _, tdir = roots
    _fire_logger(_payload(prompt_id=_PROMPT_ID))
    lines = _logged_lines(tdir)
    assert [ln["turn_id"] for ln in lines] == [_PROMPT_ID]


def test_logger_user_line_mints_without_prompt_id(
    roots: tuple[Path, Path],
) -> None:
    _, tdir = roots
    _fire_logger(_payload())
    (line,) = _logged_lines(tdir)
    assert _MINTED.match(str(line["turn_id"]))


def test_logger_user_line_mints_on_invalid_prompt_id(
    roots: tuple[Path, Path],
) -> None:
    _, tdir = roots
    _fire_logger(_payload(prompt_id="bad\x00id"))
    (line,) = _logged_lines(tdir)
    assert _MINTED.match(str(line["turn_id"]))


def test_hook_injection_rows_use_prompt_id(roots: tuple[Path, Path]) -> None:
    db, _ = roots
    _fire_hook(_payload(prompt_id=_PROMPT_ID))
    assert _injection_turn_ids(db) == {_PROMPT_ID}


def test_hook_injection_rows_mint_without_prompt_id(
    roots: tuple[Path, Path],
) -> None:
    db, _ = roots
    _fire_hook(_payload())
    (turn_id,) = _injection_turn_ids(db)
    assert _MINTED.match(turn_id)


def test_hook_injection_rows_mint_on_invalid_prompt_id(
    roots: tuple[Path, Path],
) -> None:
    db, _ = roots
    _fire_hook(_payload(prompt_id="a" * 129))
    (turn_id,) = _injection_turn_ids(db)
    assert _MINTED.match(turn_id)


@pytest.mark.parametrize("value", _REFUSED_ON_BOTH_WRITERS)
def test_logger_user_line_survives_a_refused_prompt_id(
    roots: tuple[Path, Path], value: str,
) -> None:
    _, tdir = roots
    _fire_logger(_payload(prompt_id=value))
    (line,) = _logged_lines(tdir)
    assert _MINTED.match(str(line["turn_id"]))


@pytest.mark.parametrize("value", _REFUSED_ON_BOTH_WRITERS)
def test_hook_batch_survives_a_refused_prompt_id(
    roots: tuple[Path, Path], value: str,
) -> None:
    db, _ = roots
    _fire_hook(_payload(prompt_id=value))
    (turn_id,) = _injection_turn_ids(db)
    assert _MINTED.match(turn_id)


# --- both writers on one payload ----------------------------------------

def test_both_writers_join_on_prompt_id(roots: tuple[Path, Path]) -> None:
    """One payload through the hook and the logger: the ids are equal."""
    db, tdir = roots
    payload = _payload(prompt_id=_PROMPT_ID)
    _fire_hook(payload)
    _fire_logger(payload)
    injected = _injection_turn_ids(db)
    (line,) = [ln for ln in _logged_lines(tdir) if ln["role"] == "user"]
    assert injected, "the seeded belief must be injected for the join to mean anything"
    assert injected == {line["turn_id"]}


def test_both_writers_stay_independent_without_prompt_id(
    roots: tuple[Path, Path],
) -> None:
    """Absent `prompt_id`: today's behavior, two unrelated minted ids."""
    db, tdir = roots
    payload = _payload()
    _fire_hook(payload)
    _fire_logger(payload)
    (injected,) = _injection_turn_ids(db)
    (line,) = [ln for ln in _logged_lines(tdir) if ln["role"] == "user"]
    assert _MINTED.match(injected)
    assert _MINTED.match(str(line["turn_id"]))
    assert injected != line["turn_id"]


# --- Stop -----------------------------------------------------------------

def _stop_payload(**extra: object) -> dict[str, object]:
    base: dict[str, object] = {
        "session_id": _SESSION,
        "transcript_path": None,
        "cwd": "/tmp",
        "hook_event_name": "Stop",
        "stop_hook_active": False,
        "last_assistant_message": "There are twelve barrels in the cellar.",
    }
    base.update(extra)
    return base


def test_assistant_line_carries_its_prompt_id(
    roots: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A turn's user and assistant lines share the host's prompt id."""
    _, tdir = roots
    monkeypatch.setattr(tl, "_maybe_stop_flush", lambda _d: None)
    _fire_logger(_payload(prompt_id=_PROMPT_ID))
    _fire_logger(_stop_payload(prompt_id=_PROMPT_ID))
    lines = _logged_lines(tdir)
    assert [(ln["role"], ln["turn_id"]) for ln in lines] == [
        ("user", _PROMPT_ID), ("assistant", _PROMPT_ID),
    ]


@pytest.mark.parametrize("extra", [
    pytest.param({}, id="missing"),
    pytest.param({"prompt_id": "bad\nid"}, id="invalid"),
    pytest.param({"prompt_id": _SURROGATE}, id="lone-surrogate"),
    pytest.param({"prompt_id": _PADDED}, id="padded"),
    pytest.param({"prompt_id": _NON_ASCII}, id="non-ascii-letter"),
])
def test_assistant_line_mints_without_a_valid_prompt_id(
    roots: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    extra: dict[str, object],
) -> None:
    _, tdir = roots
    monkeypatch.setattr(tl, "_maybe_stop_flush", lambda _d: None)
    _fire_logger(_stop_payload(**extra))
    (line,) = _logged_lines(tdir)
    assert line["role"] == "assistant"
    assert _MINTED.match(str(line["turn_id"]))
