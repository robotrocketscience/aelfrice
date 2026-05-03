"""Tests for the #218 UserPromptSubmit telemetry ring buffer (AC1-3, AC5).

Coverage:
- Telemetry record is written after a fire that produces a block.
- Ring-buffer cap: oldest entries evicted when > 1000.
- Fail-soft: unwritable path prints one stderr line and does not raise.
- read_user_prompt_submit_telemetry returns [] for missing file.
- read_user_prompt_submit_telemetry raises ValueError on corrupt JSON.
- Blank lines in the file are silently skipped.
- End-to-end: user_prompt_submit() writes a record when hits are non-empty.
- No record written when hits is empty.
"""
from __future__ import annotations

import json
import os
import stat
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aelfrice.hook import (
    TELEMETRY_RING_CAP,
    UserPromptSubmitConfig,
    _append_telemetry,
    read_user_prompt_submit_telemetry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(**kwargs: object) -> dict[str, object]:
    defaults: dict[str, object] = {
        "timestamp": "2026-01-01T00:00:00Z",
        "query": "test prompt",
        "n_returned": 3,
        "n_unique_content_hashes": 3,
        "n_l0": 1,
        "n_l1": 2,
        "total_chars": 300,
    }
    defaults.update(kwargs)
    return defaults


def _write_prompt_payload(prompt: str = "hello world") -> str:
    return json.dumps({"prompt": prompt})


# ---------------------------------------------------------------------------
# _append_telemetry
# ---------------------------------------------------------------------------


def test_append_telemetry_creates_file(tmp_path: Path) -> None:
    tel = tmp_path / "user_prompt_submit.jsonl"
    _append_telemetry(tel, _make_record())
    assert tel.exists()
    records = read_user_prompt_submit_telemetry(tel)
    assert len(records) == 1
    assert records[0]["n_returned"] == 3
    assert records[0]["n_l0"] == 1
    assert records[0]["total_chars"] == 300


def test_append_telemetry_accumulates(tmp_path: Path) -> None:
    tel = tmp_path / "user_prompt_submit.jsonl"
    for i in range(5):
        _append_telemetry(tel, _make_record(query=f"prompt_{i}"))
    records = read_user_prompt_submit_telemetry(tel)
    assert len(records) == 5
    assert [r["query"] for r in records] == [f"prompt_{i}" for i in range(5)]


def test_append_telemetry_ring_cap_evicts_oldest(tmp_path: Path) -> None:
    tel = tmp_path / "ring.jsonl"
    total = TELEMETRY_RING_CAP + 10
    for i in range(total):
        _append_telemetry(tel, _make_record(query=f"q{i}"))
    records = read_user_prompt_submit_telemetry(tel)
    assert len(records) == TELEMETRY_RING_CAP
    # Oldest 10 evicted; surviving records start at index 10.
    assert records[0]["query"] == "q10"
    assert records[-1]["query"] == f"q{total - 1}"


def test_append_telemetry_failsoft_on_unwritable_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unwritable dir → one stderr trace; no exception."""
    if sys.platform == "win32":
        pytest.skip("permission model differs on Windows")
    locked_dir = tmp_path / "locked"
    locked_dir.mkdir()
    os.chmod(locked_dir, stat.S_IRUSR | stat.S_IXUSR)  # r-x, no write
    tel = locked_dir / "sub" / "user_prompt_submit.jsonl"
    serr = StringIO()
    # Should not raise.
    _append_telemetry(tel, _make_record(), stderr=serr)
    assert "non-fatal" in serr.getvalue()


def test_append_telemetry_creates_parent_dirs(tmp_path: Path) -> None:
    tel = tmp_path / "a" / "b" / "c" / "user_prompt_submit.jsonl"
    _append_telemetry(tel, _make_record())
    assert tel.exists()


def test_append_telemetry_record_has_timestamp(tmp_path: Path) -> None:
    tel = tmp_path / "user_prompt_submit.jsonl"
    _append_telemetry(tel, _make_record())
    records = read_user_prompt_submit_telemetry(tel)
    ts = records[0].get("timestamp")
    assert isinstance(ts, str) and len(ts) > 0


# ---------------------------------------------------------------------------
# read_user_prompt_submit_telemetry
# ---------------------------------------------------------------------------


def test_read_telemetry_missing_file_returns_empty(tmp_path: Path) -> None:
    result = read_user_prompt_submit_telemetry(tmp_path / "no-such-file.jsonl")
    assert result == []


def test_read_telemetry_empty_file_returns_empty(tmp_path: Path) -> None:
    tel = tmp_path / "empty.jsonl"
    tel.write_text("", encoding="utf-8")
    assert read_user_prompt_submit_telemetry(tel) == []


def test_read_telemetry_corrupt_json_raises(tmp_path: Path) -> None:
    tel = tmp_path / "corrupt.jsonl"
    tel.write_text('{"ok": true}\nnot valid json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        read_user_prompt_submit_telemetry(tel)


def test_read_telemetry_skips_blank_lines(tmp_path: Path) -> None:
    tel = tmp_path / "user_prompt_submit.jsonl"
    tel.write_text(
        '\n{"n_returned": 1}\n\n{"n_returned": 2}\n',
        encoding="utf-8",
    )
    records = read_user_prompt_submit_telemetry(tel)
    assert len(records) == 2
    assert records[0]["n_returned"] == 1
    assert records[1]["n_returned"] == 2


def test_read_telemetry_skips_non_dict_json(tmp_path: Path) -> None:
    tel = tmp_path / "user_prompt_submit.jsonl"
    tel.write_text(
        '["not", "a", "dict"]\n{"n_returned": 5}\n',
        encoding="utf-8",
    )
    records = read_user_prompt_submit_telemetry(tel)
    assert len(records) == 1
    assert records[0]["n_returned"] == 5


# ---------------------------------------------------------------------------
# Integration: user_prompt_submit() writes telemetry
# ---------------------------------------------------------------------------


def _make_belief(content: str, lock_level: int = 0) -> object:
    """Return a minimal Belief-like mock."""
    b = MagicMock()
    b.content = content
    b.lock_level = lock_level
    b.id = "beef" + content[:4].encode().hex()
    return b


def test_user_prompt_submit_writes_telemetry_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """user_prompt_submit() writes a telemetry record when hits are non-empty."""
    from aelfrice.hook import user_prompt_submit
    from aelfrice.models import LOCK_USER

    tel = tmp_path / "user_prompt_submit.jsonl"
    hits = [_make_belief("alpha content"), _make_belief("beta content", LOCK_USER)]

    monkeypatch.setattr("aelfrice.hook._retrieve", lambda prompt, budget, **_: hits)
    monkeypatch.setattr(
        "aelfrice.hook._telemetry_path_for_db", lambda p: tel
    )
    monkeypatch.setattr("aelfrice.hook.db_path", lambda: tmp_path / "memory.db")
    monkeypatch.setattr(
        "aelfrice.hook.load_user_prompt_submit_config",
        lambda **_kw: UserPromptSubmitConfig(),
    )

    sin = StringIO(_write_prompt_payload("explain telemetry"))
    sout = StringIO()
    serr = StringIO()
    rc = user_prompt_submit(stdin=sin, stdout=sout, stderr=serr)
    assert rc == 0

    records = read_user_prompt_submit_telemetry(tel)
    assert len(records) == 1
    r = records[0]
    assert r["n_returned"] == 2
    assert r["n_l0"] == 1
    assert r["n_l1"] == 1
    assert r["total_chars"] == len("alpha content") + len("beta content")
    assert r["n_unique_content_hashes"] == 2
    assert isinstance(r["timestamp"], str)


def test_user_prompt_submit_no_telemetry_when_empty_hits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No telemetry record written when retrieval returns no hits."""
    from aelfrice.hook import user_prompt_submit

    tel = tmp_path / "user_prompt_submit.jsonl"

    monkeypatch.setattr("aelfrice.hook._retrieve", lambda prompt, budget, **_: [])
    monkeypatch.setattr(
        "aelfrice.hook._telemetry_path_for_db", lambda p: tel
    )
    monkeypatch.setattr("aelfrice.hook.db_path", lambda: tmp_path / "memory.db")
    monkeypatch.setattr(
        "aelfrice.hook.load_user_prompt_submit_config",
        lambda **_kw: UserPromptSubmitConfig(),
    )

    sin = StringIO(_write_prompt_payload("no results prompt"))
    sout = StringIO()
    serr = StringIO()
    rc = user_prompt_submit(stdin=sin, stdout=sout, stderr=serr)
    assert rc == 0
    assert not tel.exists()


def test_telemetry_query_capped_at_500_chars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prompt longer than 500 chars is truncated in the telemetry record."""
    from aelfrice.hook import user_prompt_submit

    tel = tmp_path / "user_prompt_submit.jsonl"
    long_prompt = "x" * 1000
    hits = [_make_belief("some content")]

    monkeypatch.setattr("aelfrice.hook._retrieve", lambda prompt, budget, **_: hits)
    monkeypatch.setattr("aelfrice.hook._telemetry_path_for_db", lambda p: tel)
    monkeypatch.setattr("aelfrice.hook.db_path", lambda: tmp_path / "memory.db")
    monkeypatch.setattr(
        "aelfrice.hook.load_user_prompt_submit_config",
        lambda **_kw: UserPromptSubmitConfig(),
    )

    sin = StringIO(json.dumps({"prompt": long_prompt}))
    sout = StringIO()
    serr = StringIO()
    user_prompt_submit(stdin=sin, stdout=sout, stderr=serr)

    records = read_user_prompt_submit_telemetry(tel)
    assert len(records[0]["query"]) == 500  # type: ignore[arg-type]
