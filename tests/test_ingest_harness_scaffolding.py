"""#1371 §9: harness scaffolding must not be stored as user-authored belief.

The host harness injects `<system-reminder>` blocks, slash-command
scaffolding and hook output as text chunks inside turns whose `type` is
`"user"`. `ingest._normalize_jsonl_turn` cannot tell them apart from
typed prose, so `derivation` stamped every interior sentence
`origin=user_transcript` with the *undeflated* user prior (factual
alpha=3.0 rather than 0.6) — provenance claiming the user said what the
harness said.

Two layers are asserted:

* the open tag is recognised at sentence start (`is_transcript_noise`), and
* the balanced region is stripped, so the sentences *inside* a
  multi-sentence block never reach ingest either.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aelfrice.extraction import extract_sentences
from aelfrice.ingest import ingest_jsonl
from aelfrice.models import ORIGIN_USER_TRANSCRIPT
from aelfrice.noise_filter import HARNESS_TAG_NAMES, is_transcript_noise
from aelfrice.store import MemoryStore

_SCAFFOLDING_LINES: tuple[str, ...] = (
    "<system-reminder>Read the codebase instructions first.</system-reminder>",
    "<command-name>/aelf:search</command-name>",
    "<command-message>search is running in the background</command-message>",
    "<command-args>retrieval budget</command-args>",
    "<local-command-stdout>3 beliefs matched</local-command-stdout>",
    "<local-command-stderr>index rebuild skipped</local-command-stderr>",
    "<user-prompt-submit-hook>injected 4 locked beliefs</user-prompt-submit-hook>",
)


def _write_jsonl(path: Path, lines: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")


def test_every_harness_tag_name_has_a_covering_case() -> None:
    """Guard the guard: a tag added to the module constant without a
    fixture line here would be silently untested."""
    covered = {
        name for name in HARNESS_TAG_NAMES
        if any(f"<{name}" in line for line in _SCAFFOLDING_LINES)
    }
    assert covered == set(HARNESS_TAG_NAMES)


@pytest.mark.parametrize("line", _SCAFFOLDING_LINES)
def test_scaffolding_open_tag_is_transcript_noise(line: str) -> None:
    assert is_transcript_noise(line) is True, line


def test_multi_sentence_reminder_body_is_stripped_before_extraction() -> None:
    """The interior is the part the prefix check cannot reach.

    Sentence-splitting puts the tag on the first sentence only, so
    without the balanced-region strip every later sentence of a reminder
    block became a user-attributed belief.
    """
    text = (
        "<system-reminder>The scratchpad directory is session specific. "
        "Never write results files into the user project.</system-reminder>"
    )
    assert extract_sentences(text) == []


def test_unpaired_open_tag_in_prose_is_left_alone() -> None:
    """High precision: only *balanced* regions are stripped."""
    text = "The rebuilder emits a <system-reminder> wrapper around locks."
    assert extract_sentences(text) == [text]


def test_reminder_block_produces_no_user_transcript_belief(
    tmp_path: Path,
) -> None:
    """End to end: the block is a `role='user'` turn, as the harness emits it.

    Falsifiable in the exact way the audit described — if any belief row
    lands with `origin=user_transcript` carrying the reminder's prose,
    the harness has been laundered into user provenance.
    """
    p = tmp_path / "turns.jsonl"
    _write_jsonl(p, [
        {
            "schema_version": 1, "ts": "2026-08-05T00:00:00Z",
            "role": "user",
            "text": (
                "<system-reminder>The scratchpad directory is session "
                "specific. Never write results files into the user "
                "project.</system-reminder>"
            ),
            "session_id": "S1", "turn_id": "t1",
        },
    ])
    store = MemoryStore(":memory:")
    try:
        result = ingest_jsonl(store, p)
        assert result.turns_ingested == 1
        rows = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
            "SELECT content, origin FROM beliefs"
        ).fetchall()
        assert rows == [], [dict(r) for r in rows]
    finally:
        store.close()


def test_real_user_prose_in_the_same_turn_still_lands(tmp_path: Path) -> None:
    """The strip must remove the block, not the turn.

    Without this the fix would be indistinguishable from dropping any
    turn that mentions a harness tag.
    """
    p = tmp_path / "turns.jsonl"
    _write_jsonl(p, [
        {
            "schema_version": 1, "ts": "2026-08-05T00:00:00Z",
            "role": "user",
            "text": (
                "<system-reminder>Ignore this scaffolding entirely."
                "</system-reminder>"
                "The retrieval budget is fifty beliefs per turn."
            ),
            "session_id": "S1", "turn_id": "t1",
        },
    ])
    store = MemoryStore(":memory:")
    try:
        ingest_jsonl(store, p)
        rows = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
            "SELECT content, origin FROM beliefs"
        ).fetchall()
        contents = [str(r["content"]) for r in rows]
        assert any("retrieval budget" in c for c in contents), contents
        assert not any("scaffolding" in c for c in contents), contents
        for row in rows:
            assert row["origin"] == ORIGIN_USER_TRANSCRIPT
    finally:
        store.close()
