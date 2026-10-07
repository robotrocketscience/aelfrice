"""#1706: every turn-log reader reads where the transcript logger writes,
whatever the payload says about `cwd`.

The readers used to walk up from the payload `cwd` to the first `.git`
entry, and read nothing when the payload carried no `cwd`. They now use the
logger's own resolver (`transcript_logger.turns_path`).
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

import aelfrice.cli as cli_module
import aelfrice.context_rebuilder as rebuilder
from aelfrice import hook
from aelfrice.transcript_logger import turns_path

# The logger stamps each line with the payload's session, and the hook
# readers keep only that session's lines (#1744).
SESSION = "s-pelican"


@pytest.fixture
def log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A turn log at the logger's own location, with one recognizable turn."""
    monkeypatch.setenv("AELFRICE_TRANSCRIPTS_DIR", str(tmp_path / "t"))
    p = turns_path()
    p.parent.mkdir(parents=True)
    p.write_text(
        json.dumps({"role": "user", "text": "the pelican ledger",
                    "session_id": SESSION}) + "\n",
        encoding="utf-8",
    )
    return p


def _texts(turns: list[object]) -> list[str]:
    return [str(getattr(t, "text", "")) for t in turns]


@pytest.mark.parametrize("payload_cwd", [None, "/elsewhere/entirely"])
def test_the_hook_reader_reads_the_logger_location(
    log: Path, payload_cwd: str | None,
) -> None:
    payload: dict[str, object] = {"session_id": SESSION}
    if payload_cwd is not None:
        payload["cwd"] = payload_cwd
    turns = hook._read_recent_for_pre_compact(payload, 5)  # pyright: ignore[reportPrivateUsage]
    assert _texts(turns) == ["the pelican ledger"]


@pytest.mark.parametrize("payload_cwd", [None, "/elsewhere/entirely"])
def test_the_rebuilder_reader_reads_the_logger_location(
    log: Path, payload_cwd: str | None,
) -> None:
    payload: dict[str, object] = {"session_id": SESSION}
    if payload_cwd is not None:
        payload["cwd"] = payload_cwd
    turns = rebuilder._read_recent_for_pre_compact(payload, 5)  # pyright: ignore[reportPrivateUsage]
    assert _texts(turns) == ["the pelican ledger"]


def test_aelf_rebuild_reads_the_logger_location(
    log: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Path] = []
    real = rebuilder.read_recent_turns_latest_session

    def recording(path: Path, n: int) -> list[object]:
        seen.append(path)
        return real(path, n)

    monkeypatch.setattr(rebuilder, "read_recent_turns_latest_session", recording)
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "memory.db"))
    assert cli_module.main(argv=["rebuild"], out=io.StringIO()) == 0
    assert seen == [log]
