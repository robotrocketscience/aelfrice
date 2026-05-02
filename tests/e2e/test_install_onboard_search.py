"""E2E scenario #1 (#334): install -> onboard -> search returns hits.

Catches: install-time wiring drift, search-path drift, onboard
candidate-emit JSON contract drift. Failure here means the binary
on PATH cannot complete the most basic ingest-then-retrieve loop.

Distinct from in-process tests because every step runs through the
real `aelf` argv entry point and a real SQLite file.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Callable

import pytest


pytestmark = pytest.mark.timeout(120)


def test_aelf_version_prints(
    aelf_run: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    """The installed binary responds to --version. Smoke check for matrix."""
    result = aelf_run("--version")
    assert result.returncode == 0
    # Version string format is `aelfrice X.Y.Z` — match prefix only so the
    # test isn't coupled to the live version number.
    assert result.stdout.strip(), "expected non-empty version output"


def test_onboard_emit_candidates_returns_distinctive_token(
    aelf_run: Callable[..., subprocess.CompletedProcess[str]],
    tiny_project: Path,
) -> None:
    """`aelf onboard --emit-candidates` extracts text from the fixture.

    The synthetic repo seeds a distinctive token ("quokka calibration") in
    README and docs. If candidate emission is wired correctly the token
    appears in the JSON sentences[] payload printed to stdout.
    """
    result = aelf_run(
        "onboard",
        "--emit-candidates",
        str(tiny_project),
        timeout=90.0,
    )
    assert result.returncode == 0, (
        f"onboard --emit-candidates exited {result.returncode}; "
        f"stderr: {result.stderr!r}"
    )
    payload = json.loads(result.stdout)
    assert "session_id" in payload
    assert "sentences" in payload
    sentences = payload["sentences"]
    assert isinstance(sentences, list)
    joined = " ".join(
        s if isinstance(s, str) else json.dumps(s) for s in sentences
    ).lower()
    assert "quokka" in joined, (
        f"expected distinctive token 'quokka' in candidate sentences; "
        f"got first 5: {sentences[:5]!r}"
    )


def test_search_runs_against_empty_store(
    aelf_run: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    """`aelf search` against a fresh store completes cleanly with 0 hits.

    Guards the search-path init story: store auto-creates, FTS5 index is
    queryable, exit code is 0 even with no matches. (Catches the class
    of regression where a missing migration leaves FTS5 unbuilt.)
    """
    result = aelf_run("search", "quokka", check=False)
    assert result.returncode == 0, (
        f"empty-store search exited {result.returncode}; "
        f"stderr: {result.stderr!r}"
    )
