"""A paste the user rejects is not stored as the user's words.

Operator ruling rd24888de (2026-11-06): when the user's own words outside
a `<pasted_content>` block carry a rejection cue, the pasted prose is
dropped and only those words are kept. A bare paste, or one with neutral
words around it, is still kept (ruling of 2026-09-30).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aelfrice.ingest import _user_speech, ingest_jsonl
from aelfrice.store import MemoryStore

MODEL = "Let me know how you would like to proceed with the batching plan."


def _paste(body: str = MODEL, pid: str = "700c") -> str:
    return f'<pasted_content id="{pid}">\n{body}\n</pasted_content id="{pid}">'


# One user sentence per cue in `_REJECTION_CUE_RE`, in list order.
CUE_WORDS = [
    "this is wrong",
    "Incorrect, the onboard never ran",
    "that is not right",
    "that's not true at all",
    "this is nonsense",
    "complete gibberish",
    "such ai-slop",
    "sloppy summary",
    "you made up that number",
    "the made-up term again",
    "I disagree with this",
    "reject this plan",
    "I rejected this already",
]


@pytest.mark.parametrize("mine", CUE_WORDS)
def test_a_paste_with_a_rejection_cue_is_dropped(mine: str) -> None:
    assert _user_speech(f"{mine}\n{_paste()}") == mine


@pytest.mark.parametrize("mine", CUE_WORDS)
def test_a_cue_after_the_paste_drops_it_too(mine: str) -> None:
    assert _user_speech(f"{_paste()}\n{mine}") == mine


def test_a_bare_paste_is_kept() -> None:
    assert _user_speech(_paste()) == MODEL


def test_a_paste_with_neutral_words_is_kept() -> None:
    mine = "here is what gemini said about the onboard"
    assert _user_speech(f"{mine}\n{_paste()}") == f"{mine}\n\n{MODEL}"


def test_a_cue_inside_the_paste_does_not_drop_it() -> None:
    body = "The earlier estimate was wrong, so the rate is now 2 Gbps."
    assert _user_speech(f"see below\n{_paste(body)}") == f"see below\n\n{body}"


def test_made_up_of_is_not_a_cue() -> None:
    mine = "the pool is made up of four stations"
    assert _user_speech(f"{mine}\n{_paste()}") == f"{mine}\n\n{MODEL}"


def test_every_paste_in_a_rejecting_record_is_dropped() -> None:
    text = f"{_paste(pid='a1')}\nboth of these are wrong\n{_paste(pid='b2')}"
    assert _user_speech(text) == "both of these are wrong"


def test_a_rejected_paste_never_reaches_the_store(tmp_path: Path) -> None:
    rec = {"schema_version": 1, "role": "user", "session_id": "s1",
           "ts": "2026-11-06T10:00:00Z",
           "text": f"this is ai slop, redo it\n{_paste()}"}
    log = tmp_path / "turns.jsonl"
    log.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    store = MemoryStore(str(tmp_path / "memory.db"))
    try:
        ingest_jsonl(store, log)
        rows = [r[0] for r in store._conn.execute(  # noqa: SLF001
            "SELECT content FROM beliefs").fetchall()]
    finally:
        store.close()
    assert any("redo it" in r for r in rows)
    assert not any("proceed" in r for r in rows)
