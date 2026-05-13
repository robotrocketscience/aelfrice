"""`aelf doctor` surfaces injection-dedup ring state (#740)."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from aelfrice.cli import _print_doctor_session_ring
from aelfrice.session_ring import append_ids


def test_doctor_silent_when_ring_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "memory.db"))
    buf = io.StringIO()
    _print_doctor_session_ring(buf)
    assert buf.getvalue() == ""


def test_doctor_shows_ring_after_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "memory.db"))
    append_ids("sess-A", ["b1", "b2", "b3"])
    buf = io.StringIO()
    _print_doctor_session_ring(buf)
    line = buf.getvalue().strip()
    assert "injection ring:" in line
    assert "3/200 ids" in line
    assert "evicted 0" in line


def test_doctor_reflects_eviction_total(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "memory.db"))
    monkeypatch.setenv("AELFRICE_INJECTION_RING_MAX", "2")
    append_ids("sess-A", ["b1", "b2"])
    append_ids("sess-A", ["b3"])  # forces 1 eviction
    buf = io.StringIO()
    _print_doctor_session_ring(buf)
    line = buf.getvalue().strip()
    assert "2/2 ids" in line
    assert "evicted 1" in line
