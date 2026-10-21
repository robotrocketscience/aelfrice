"""#1780: the logger drops a prompt the host transcript marks as scheduled.

The `UserPromptSubmit` payload carries the prompt text and no marker. The
host transcript record for a scheduled continuation prompt carries
`isMeta: true` and `turnOrigin: "scheduled"`. The logger reads a bounded
tail of that file and skips the append when the newest matching user
record carries either flag. Every fixture here is synthetic.
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import pytest

from aelfrice import transcript_logger as tl

_PROMPT = "Continue the release checklist from the saved plan."
_SESSION = "sess-1780"


@pytest.fixture
def tdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    out = tmp_path / "transcripts"
    monkeypatch.setenv("AELFRICE_TRANSCRIPTS_DIR", str(out))
    return out


def _user(text: object, **extra: object) -> dict[str, object]:
    rec: dict[str, object] = {
        "type": "user",
        "sessionId": _SESSION,
        "timestamp": "2026-10-11T15:05:00.000Z",
        "message": {"role": "user", "content": text},
    }
    rec.update(extra)
    return rec


def _scheduled(text: str) -> dict[str, object]:
    return _user(
        text, isMeta=True, promptSource="system", turnOrigin="scheduled",
        queuePriority="later",
    )


def _typed(text: str) -> dict[str, object]:
    return _user(text, promptSource="typed", turnOrigin="human")


def _assistant(text: str) -> dict[str, object]:
    return {
        "type": "assistant",
        "sessionId": _SESSION,
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
    }


def _write(path: Path, records: list[dict[str, object]]) -> Path:
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8",
    )
    return path


def _submit(prompt: str, transcript_path: object) -> None:
    payload: dict[str, object] = {
        "hook_event_name": "UserPromptSubmit",
        "prompt": prompt,
        "session_id": _SESSION,
    }
    if transcript_path is not _MISSING:
        payload["transcript_path"] = transcript_path
    rc = tl.main(stdin=io.StringIO(json.dumps(payload)), stderr=io.StringIO())
    assert rc == 0


_MISSING = object()


def _logged(tdir: Path) -> list[str]:
    path = tdir / "turns.jsonl"
    if not path.is_file():
        return []
    return [
        json.loads(line)["text"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_scheduled_record_is_not_logged(tdir: Path, tmp_path: Path) -> None:
    host = _write(tmp_path / "host.jsonl", [
        _typed("Set up the nightly check."),
        _assistant("Scheduled."),
        _scheduled(_PROMPT),
        {"type": "attachment", "hookEvent": "UserPromptSubmit"},
    ])
    _submit(_PROMPT, str(host))
    assert _logged(tdir) == []


def test_typed_record_is_logged(tdir: Path, tmp_path: Path) -> None:
    host = _write(tmp_path / "host.jsonl", [_assistant("Done."), _typed(_PROMPT)])
    _submit(_PROMPT, str(host))
    assert _logged(tdir) == [_PROMPT]


@pytest.mark.parametrize(
    "extra",
    [{"isMeta": True}, {"turnOrigin": "scheduled"}],
    ids=["isMeta-only", "turnOrigin-only"],
)
def test_either_flag_alone_skips(
    tdir: Path, tmp_path: Path, extra: dict[str, object],
) -> None:
    host = _write(tmp_path / "host.jsonl", [_user(_PROMPT, **extra)])
    _submit(_PROMPT, str(host))
    assert _logged(tdir) == []


def test_missing_transcript_path_field_logs(tdir: Path) -> None:
    _submit(_PROMPT, _MISSING)
    assert _logged(tdir) == [_PROMPT]


@pytest.mark.parametrize("value", [None, "", 7], ids=["none", "empty", "int"])
def test_unusable_transcript_path_value_logs(tdir: Path, value: object) -> None:
    _submit(_PROMPT, value)
    assert _logged(tdir) == [_PROMPT]


def test_missing_file_logs(tdir: Path, tmp_path: Path) -> None:
    _submit(_PROMPT, str(tmp_path / "absent.jsonl"))
    assert _logged(tdir) == [_PROMPT]


def test_directory_path_logs(tdir: Path, tmp_path: Path) -> None:
    _submit(_PROMPT, str(tmp_path))
    assert _logged(tdir) == [_PROMPT]


@pytest.mark.skipif(
    sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="chmod 000 does not block reads on Windows or as root",
)
def test_unreadable_file_logs(tdir: Path, tmp_path: Path) -> None:
    host = _write(tmp_path / "host.jsonl", [_scheduled(_PROMPT)])
    host.chmod(0)
    try:
        _submit(_PROMPT, str(host))
    finally:
        host.chmod(0o600)
    assert _logged(tdir) == [_PROMPT]


def test_no_matching_record_logs(tdir: Path, tmp_path: Path) -> None:
    host = _write(tmp_path / "host.jsonl", [_scheduled("Some other prompt.")])
    _submit(_PROMPT, str(host))
    assert _logged(tdir) == [_PROMPT]


def test_empty_file_logs(tdir: Path, tmp_path: Path) -> None:
    host = tmp_path / "host.jsonl"
    host.write_bytes(b"")
    _submit(_PROMPT, str(host))
    assert _logged(tdir) == [_PROMPT]


def test_list_content_is_not_matched(tdir: Path, tmp_path: Path) -> None:
    rec = _user([{"type": "text", "text": _PROMPT}], isMeta=True)
    host = _write(tmp_path / "host.jsonl", [rec])
    _submit(_PROMPT, str(host))
    assert _logged(tdir) == [_PROMPT]


def test_match_behind_an_assistant_record_is_an_earlier_turn(
    tdir: Path, tmp_path: Path,
) -> None:
    # The same text fired as a scheduled prompt last turn; now it is typed
    # and the host has not flushed the typed record yet.
    host = _write(tmp_path / "host.jsonl", [_scheduled(_PROMPT), _assistant("Ok.")])
    _submit(_PROMPT, str(host))
    assert _logged(tdir) == [_PROMPT]


def test_newest_matching_record_decides(tdir: Path, tmp_path: Path) -> None:
    host = _write(tmp_path / "host.jsonl", [_scheduled(_PROMPT), _typed(_PROMPT)])
    _submit(_PROMPT, str(host))
    assert _logged(tdir) == [_PROMPT]


def test_torn_last_line_does_not_hide_the_record(
    tdir: Path, tmp_path: Path,
) -> None:
    host = _write(tmp_path / "host.jsonl", [_scheduled(_PROMPT)])
    with host.open("ab") as f:
        f.write(b'{"type": "attachment", "hookEv')
    _submit(_PROMPT, str(host))
    assert _logged(tdir) == []


def test_record_past_the_window_is_not_read(
    tdir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The scan reads a bounded tail; a record before it is never seen.
    monkeypatch.setattr(tl, "_HOST_TAIL_READ_BYTES", 4096)
    filler = [_user(f"tool output {i}", isMeta=True) for i in range(200)]
    host = _write(tmp_path / "host.jsonl", [_scheduled(_PROMPT), *filler])
    _submit(_PROMPT, str(host))
    assert _logged(tdir) == [_PROMPT]


def test_window_grows_with_the_prompt(
    tdir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tl, "_HOST_TAIL_READ_BYTES", 1024)
    big = "Resume step " + "é" * 20000 + " now."
    host = _write(tmp_path / "host.jsonl", [_assistant("Ok."), _scheduled(big)])
    assert host.stat().st_size > 1024
    _submit(big, str(host))
    assert _logged(tdir) == []


def test_codex_rollout_is_never_matched(tdir: Path, tmp_path: Path) -> None:
    rec = {
        "type": "response_item",
        "isMeta": True,
        "payload": {
            "type": "message", "role": "user",
            "content": [{"type": "input_text", "text": _PROMPT}],
        },
    }
    host = _write(tmp_path / "host.jsonl", [rec])
    _submit(_PROMPT, str(host))
    assert _logged(tdir) == [_PROMPT]
