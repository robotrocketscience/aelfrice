"""E2E scenario #2 (#334): hook -> ingest -> rebuild -> inject roundtrip.

Exercises the seam the UserPromptSubmit / PreCompact hooks travel:

    pre-seeded store + recent-turn transcript
        -> rebuild_v14 (the same code path the hook calls)
        -> stdout context block
        -> what the next turn would see injected.

`aelf rebuild --transcript <jsonl>` is the manual entry point onto
the hook code path (CLI docstring: "same code path as the PreCompact
hook"), so a passing test here means the hook would also produce the
expected block on real input. A regression on any of the three
moving parts (transcript reader, retrieval, block renderer) drops a
locked belief out of the rebuild output and fails this test.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Callable

import pytest


pytestmark = pytest.mark.timeout(120)


def test_locked_belief_reaches_rebuild_block(
    aelf_run: Callable[..., subprocess.CompletedProcess[str]],
    tmp_path: Path,
) -> None:
    """A locked belief related to a recent transcript turn must appear
    in the rebuild block stdout. Failure here means the hook would
    inject an empty / stale memory block to the next turn.
    """
    distinctive = (
        "Wibble pickling requires the canonical protocol header bytes."
    )
    aelf_run("lock", distinctive)

    # the host harness internal transcript shape: type/message/sessionId/
    # timestamp/cwd. The rebuilder accepts this format via
    # `--transcript` (read_recent_turns_claude_transcript).
    transcript = tmp_path / "claude-session.jsonl"
    turn = {
        "type": "user",
        "message": {
            "role": "user",
            "content": "How does wibble pickling handle the header?",
        },
        "sessionId": "e2e-hook-roundtrip",
        "timestamp": "2026-05-02T22:00:00Z",
        "cwd": str(tmp_path),
    }
    transcript.write_text(json.dumps(turn) + "\n")

    result = aelf_run("rebuild", "--transcript", str(transcript))
    assert result.returncode == 0, (
        f"rebuild exited {result.returncode}; stderr: {result.stderr!r}"
    )

    block = result.stdout
    # The locked belief carries the distinctive token. If retrieval +
    # pack-to-budget are wired, "wibble" lands in the block. The exact
    # rendering format is not coupled to here — only the content.
    assert "wibble" in block.lower(), (
        f"expected locked belief in rebuild block; got:\n{block!r}"
    )


def test_empty_store_rebuild_completes_cleanly(
    aelf_run: Callable[..., subprocess.CompletedProcess[str]],
    tmp_path: Path,
) -> None:
    """Rebuild against an empty store + empty transcript must exit 0.

    Guards the no-op contract: the hook fires every turn and must
    never wedge an empty session. If the rebuilder raised when the
    transcript / store had no usable input, every fresh project
    would crash on first prompt.
    """
    transcript = tmp_path / "empty-session.jsonl"
    transcript.write_text("")
    result = aelf_run(
        "rebuild", "--transcript", str(transcript), check=False
    )
    assert result.returncode == 0, (
        f"empty-input rebuild exited {result.returncode}; "
        f"stderr: {result.stderr!r}"
    )
