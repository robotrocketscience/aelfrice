"""The shared turn log is read one session at a time (#1744).

The transcript logger writes one `turns.jsonl` per repository, and every
session and linked worktree of that repository appends to it. Each reader
must keep only the current session's turns, or a concurrent session's
conversation and its session-scoped beliefs enter this session's rebuilt
context. Two sessions interleave in one log in every test here; the turn
text is synthetic. The suite pins `AELFRICE_TRANSCRIPTS_DIR` to a sandbox
(#1706), so `find_aelfrice_log()` resolves under `tmp_path`.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

import aelfrice.hook as hook
from aelfrice import context_rebuilder as cr
from aelfrice.cli import main as cli_main
from aelfrice.context_rebuilder import RecentTurn, read_recent_turns_aelfrice
from aelfrice.models import BELIEF_FACTUAL, LOCK_USER, Belief
from aelfrice.store import MemoryStore

pytestmark = pytest.mark.timeout(60)

MINE = "widgetsession-mine"
THEIRS = "widgetsession-theirs"


@pytest.fixture
def log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    tdir = tmp_path / "transcripts"
    tdir.mkdir()
    monkeypatch.setenv("AELFRICE_TRANSCRIPTS_DIR", str(tdir))
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "memory.db"))
    monkeypatch.setenv("AELF_NO_UPDATE_CHECK", "1")
    monkeypatch.setenv("AELFRICE_NO_AUTO_INSTALL", "1")
    path = cr.find_aelfrice_log()
    assert path.parent == tdir
    return path


def _write(log: Path, rows: list[tuple[str | None, str, str]]) -> None:
    lines = []
    for sid, role, text in rows:
        rec: dict[str, object] = {"role": role, "text": text}
        if sid is not None:
            rec["session_id"] = sid
        lines.append(json.dumps(rec))
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _interleaved(log: Path) -> None:
    """Mine first, then theirs, so a plain tail is all theirs."""
    _write(log, [
        (MINE, "user", "mine widgetalpha one"),
        (MINE, "assistant", "mine widgetalpha two"),
        (THEIRS, "user", "theirs widgetomega one"),
        (MINE, "user", "mine widgetalpha three"),
        (THEIRS, "assistant", "theirs widgetomega two"),
        (THEIRS, "user", "theirs widgetomega three"),
        (None, "user", "unattributed widgetnull"),
    ])


def _texts(turns: list[RecentTurn]) -> list[str]:
    return [t.text for t in turns]


def _payload(tmp_path: Path, **extra: object) -> dict[str, object]:
    p: dict[str, object] = {"session_id": MINE, "cwd": str(tmp_path)}
    p.update(extra)
    return p


def _host_transcript(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "host.jsonl"
    p.write_text(json.dumps({
        "type": "user", "sessionId": MINE,
        "message": {"role": "user", "content": text},
    }) + "\n", encoding="utf-8")
    return p


# --- the reader ----------------------------------------------------------


def test_reader_keeps_only_the_session_then_tails(log: Path) -> None:
    """Killed by: tailing before filtering (the last 2 lines are theirs),
    or keeping lines with no `session_id`."""
    _interleaved(log)
    got = read_recent_turns_aelfrice(log, n=2, session_id=MINE)
    assert _texts(got) == ["mine widgetalpha two", "mine widgetalpha three"]


def test_reader_without_session_is_unchanged(log: Path) -> None:
    """Killed by: filtering when no session is given."""
    _interleaved(log)
    got = read_recent_turns_aelfrice(log, n=3)
    assert _texts(got) == [
        "theirs widgetomega two", "theirs widgetomega three", "unattributed widgetnull",
    ]


# --- the hook readers ----------------------------------------------------


@pytest.mark.parametrize("reader", [
    hook._read_recent_for_pre_compact,  # pyright: ignore[reportPrivateUsage]
    cr._read_recent_for_pre_compact,  # pyright: ignore[reportPrivateUsage]
], ids=["hook", "context_rebuilder"])
def test_hook_reader_reads_only_this_session(
    log: Path, tmp_path: Path, reader: object,
) -> None:
    """Killed by: not passing the payload's `session_id` to the reader."""
    _interleaved(log)
    got = reader(_payload(tmp_path), 10)  # type: ignore[operator]
    assert _texts(got) == [
        "mine widgetalpha one", "mine widgetalpha two", "mine widgetalpha three",
    ]


@pytest.mark.parametrize("reader", [
    hook._read_recent_for_pre_compact,  # pyright: ignore[reportPrivateUsage]
    cr._read_recent_for_pre_compact,  # pyright: ignore[reportPrivateUsage]
], ids=["hook", "context_rebuilder"])
def test_hook_reader_falls_back_when_log_has_no_turns_for_session(
    log: Path, tmp_path: Path, reader: object,
) -> None:
    """Killed by: returning the empty filtered read (or the other
    session's turns) instead of the host's per-session transcript."""
    _write(log, [(THEIRS, "user", "theirs widgetomega one")])
    host = _host_transcript(tmp_path, "mine widgethost")
    got = reader(_payload(tmp_path, transcript_path=str(host)), 10)  # type: ignore[operator]
    assert _texts(got) == ["mine widgethost"]


@pytest.mark.parametrize("reader", [
    hook._read_recent_for_pre_compact,  # pyright: ignore[reportPrivateUsage]
    cr._read_recent_for_pre_compact,  # pyright: ignore[reportPrivateUsage]
], ids=["hook", "context_rebuilder"])
@pytest.mark.parametrize("sid", [None, "", "   "], ids=["absent", "empty", "blank"])
def test_payload_without_a_session_skips_the_log(
    log: Path, tmp_path: Path, reader: object, sid: str | None,
) -> None:
    """Killed by: reading the whole shared log when the payload names no
    session, or treating a blank id as a session."""
    _interleaved(log)
    host = _host_transcript(tmp_path, "mine widgethost")
    payload: dict[str, object] = {"cwd": str(tmp_path), "transcript_path": str(host)}
    if sid is not None:
        payload["session_id"] = sid
    assert _texts(reader(payload, 10)) == ["mine widgethost"]  # type: ignore[operator]
    del payload["transcript_path"]
    assert reader(payload, 10) == []  # type: ignore[operator]


@pytest.mark.parametrize("reader", [
    hook._read_recent_for_pre_compact,  # pyright: ignore[reportPrivateUsage]
    cr._read_recent_for_pre_compact,  # pyright: ignore[reportPrivateUsage]
], ids=["hook", "context_rebuilder"])
def test_blank_session_is_no_session(
    log: Path, tmp_path: Path, reader: object,
) -> None:
    """Killed by: treating a blank `session_id` as a session. The logger
    stores whatever the payload sent, so another blank-id payload's lines
    carry the same blank id and must not be read as this session's."""
    _write(log, [("   ", "user", "blank widgetblank")])
    host = _host_transcript(tmp_path, "mine widgethost")
    payload = {"session_id": "   ", "cwd": str(tmp_path), "transcript_path": str(host)}
    assert _texts(reader(payload, 10)) == ["mine widgethost"]  # type: ignore[operator]


def _seed_lock(tmp_path: Path) -> None:
    """One locked belief, so a rebuild emits a block rather than its
    silent empty-store path."""
    store = MemoryStore(str(tmp_path / "memory.db"))
    try:
        store.insert_belief(Belief(
            id="w-lock", content="The widget rebuild check is synthetic.",
            content_hash="hash-w-lock", alpha=1.0, beta=1.0, type=BELIEF_FACTUAL,
            lock_level=LOCK_USER, locked_at="2026-10-01T00:00:00Z",
            created_at="2026-10-01T00:00:00Z", last_retrieved_at=None,
        ))
    finally:
        store.close()


def _threshold_config(tmp_path: Path) -> None:
    (tmp_path / ".aelfrice.toml").write_text(
        '[rebuilder]\ntrigger_mode = "threshold"\n'
        "[rebuild_floor]\nsession = 0.0\nl1 = 0.0\n",
        encoding="utf-8",
    )


def test_compact_rebuild_block_reads_only_this_session(
    log: Path, tmp_path: Path,
) -> None:
    """Killed by: the SessionStart(compact) builder dropping the payload's
    session on the way to the reader."""
    _interleaved(log)
    _seed_lock(tmp_path)
    _threshold_config(tmp_path)
    block = hook._build_rebuild_block_from_payload(_payload(tmp_path))  # pyright: ignore[reportPrivateUsage]
    assert "widgetalpha" in block
    assert "widgetomega" not in block


def test_cadence_rebuild_reads_only_this_session(
    log: Path, tmp_path: Path,
) -> None:
    """Killed by: the cadence rebuild dropping the payload's session on
    the way to the reader."""
    _interleaved(log)
    _seed_lock(tmp_path)
    _threshold_config(tmp_path)
    body = hook._run_cadence_rebuild(_payload(tmp_path), tmp_path)  # pyright: ignore[reportPrivateUsage]
    assert body is not None
    assert "widgetalpha" in body
    assert "widgetomega" not in body


def test_conversation_aware_query_reads_only_this_session(
    log: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Killed by: the UserPromptSubmit query reading the shared log
    without the session filter."""
    _interleaved(log)
    seen: list[list[str]] = []
    real = hook._build_conversation_aware_query  # pyright: ignore[reportPrivateUsage]

    def spy(prompt: str, turns: list[RecentTurn], **kw: object) -> str:
        seen.append(_texts(turns))
        return real(prompt, turns, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(hook, "_build_conversation_aware_query", spy)
    payload = _payload(tmp_path, hook_event_name="UserPromptSubmit",
                       prompt="what did we settle on for the widget thing")
    assert hook.user_prompt_submit(
        stdin=io.StringIO(json.dumps(payload)), stdout=io.StringIO(),
        stderr=io.StringIO(),
    ) == 0
    assert seen, "the conversation-aware query did not run"
    assert all(t.startswith("mine ") for turns in seen for t in turns), seen


# --- aelf rebuild --------------------------------------------------------


def test_aelf_rebuild_reads_only_the_latest_session(
    log: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Killed by: `aelf rebuild` reading every session in the log. With no
    payload, it keeps the session of the log's last attributed turn."""
    _write(log, [
        (THEIRS, "user", "theirs widgetomega one"),
        (MINE, "user", "mine widgetalpha one"),
        (THEIRS, "assistant", "theirs widgetomega two"),
        (MINE, "assistant", "mine widgetalpha two"),
        (None, "user", "unattributed widgetnull"),
    ])
    _seed_lock(tmp_path)
    assert cli_main(["rebuild"]) == 0
    out = capsys.readouterr().out
    assert "widgetalpha" in out
    assert "widgetomega" not in out
    assert "widgetnull" not in out
