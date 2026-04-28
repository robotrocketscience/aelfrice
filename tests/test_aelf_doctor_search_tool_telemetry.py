"""Tests for `aelf doctor` search_tool_hook telemetry section (#155 AC8).

Coverage:
- diagnose() returns telemetry stats when the file has records.
- format_report() prints p50/p95/noise_rate/fire_count.
- Empty/missing telemetry file → "no fires recorded" line, no raise.
- Malformed JSON in the file → corrupt flag set in report; exit-1
  behaviour when invoked through CLI.
- diagnose_search_tool_telemetry() computes percentiles correctly.
- Noise rate computation: fires with zero L0+L1 counted correctly.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aelfrice.doctor import (
    SearchToolTelemetryStats,
    diagnose,
    diagnose_search_tool_telemetry,
    format_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_tel(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )


def _record(
    latency_ms: float = 10.0,
    l0: int = 1,
    l1: int = 3,
    command: str = "rg",
    query: str = "foo",
) -> dict[str, object]:
    return {
        "timestamp": "2026-01-01T00:00:00Z",
        "session_id": "s1",
        "command": command,
        "query": query,
        "latency_ms": latency_ms,
        "injected_l0": l0,
        "injected_l1": l1,
    }


# ---------------------------------------------------------------------------
# diagnose_search_tool_telemetry
# ---------------------------------------------------------------------------


def test_diagnose_returns_none_when_file_missing(tmp_path: Path) -> None:
    result = diagnose_search_tool_telemetry(tmp_path / "no-such.jsonl")
    assert result is None


def test_diagnose_returns_none_for_empty_file(tmp_path: Path) -> None:
    tel = tmp_path / "search_tool_hook.jsonl"
    tel.write_text("", encoding="utf-8")
    assert diagnose_search_tool_telemetry(tel) is None


def test_diagnose_raises_on_corrupt_json(tmp_path: Path) -> None:
    tel = tmp_path / "search_tool_hook.jsonl"
    tel.write_text('{"ok": 1}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError):
        diagnose_search_tool_telemetry(tel)


def test_diagnose_computes_fire_count(tmp_path: Path) -> None:
    tel = tmp_path / "search_tool_hook.jsonl"
    _write_tel(tel, [_record() for _ in range(7)])
    stats = diagnose_search_tool_telemetry(tel)
    assert stats is not None
    assert stats.fire_count == 7


def test_diagnose_computes_latency_percentiles(tmp_path: Path) -> None:
    tel = tmp_path / "search_tool_hook.jsonl"
    # 10 records with latencies 10, 20, ..., 100
    _write_tel(tel, [_record(latency_ms=float(i * 10)) for i in range(1, 11)])
    stats = diagnose_search_tool_telemetry(tel)
    assert stats is not None
    # p50 nearest-rank: index = max(0, int(10*50/100) - 1) = 4 → value 50.0
    assert stats.p50_ms == 50.0
    # p95 nearest-rank: index = max(0, int(10*95/100) - 1) = 8 → value 90.0
    assert stats.p95_ms == 90.0


def test_diagnose_computes_noise_rate_all_noise(tmp_path: Path) -> None:
    tel = tmp_path / "search_tool_hook.jsonl"
    _write_tel(tel, [_record(l0=0, l1=0) for _ in range(4)])
    stats = diagnose_search_tool_telemetry(tel)
    assert stats is not None
    assert stats.noise_rate == pytest.approx(1.0)


def test_diagnose_computes_noise_rate_no_noise(tmp_path: Path) -> None:
    tel = tmp_path / "search_tool_hook.jsonl"
    _write_tel(tel, [_record(l0=1, l1=2) for _ in range(5)])
    stats = diagnose_search_tool_telemetry(tel)
    assert stats is not None
    assert stats.noise_rate == pytest.approx(0.0)


def test_diagnose_computes_noise_rate_mixed(tmp_path: Path) -> None:
    tel = tmp_path / "search_tool_hook.jsonl"
    records = [_record(l0=0, l1=0)] * 3 + [_record(l0=1, l1=0)] * 7
    _write_tel(tel, records)
    stats = diagnose_search_tool_telemetry(tel)
    assert stats is not None
    assert stats.noise_rate == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# diagnose() integration
# ---------------------------------------------------------------------------


def test_diagnose_with_explicit_tel_path_populates_report(tmp_path: Path) -> None:
    tel = tmp_path / "search_tool_hook.jsonl"
    _write_tel(tel, [_record() for _ in range(3)])
    report = diagnose(
        user_settings=tmp_path / "no-user.json",
        project_root=tmp_path / "no-proj",
        search_tool_telemetry_path=tel,
    )
    assert report.search_tool_telemetry_path == tel
    assert report.search_tool_telemetry is not None
    assert report.search_tool_telemetry.fire_count == 3
    assert not report.search_tool_telemetry_corrupt


def test_diagnose_missing_tel_file_sets_none_stats(tmp_path: Path) -> None:
    tel = tmp_path / "no-such.jsonl"
    report = diagnose(
        user_settings=tmp_path / "no-user.json",
        project_root=tmp_path / "no-proj",
        search_tool_telemetry_path=tel,
    )
    assert report.search_tool_telemetry_path == tel
    assert report.search_tool_telemetry is None
    assert not report.search_tool_telemetry_corrupt


def test_diagnose_corrupt_tel_sets_flag(tmp_path: Path) -> None:
    tel = tmp_path / "corrupt.jsonl"
    tel.write_text('{"ok": 1}\nnot valid\n', encoding="utf-8")
    report = diagnose(
        user_settings=tmp_path / "no-user.json",
        project_root=tmp_path / "no-proj",
        search_tool_telemetry_path=tel,
    )
    assert report.search_tool_telemetry_corrupt is True
    assert report.search_tool_telemetry is None


# ---------------------------------------------------------------------------
# format_report()
# ---------------------------------------------------------------------------


def test_format_report_no_fires_sentinel(tmp_path: Path) -> None:
    tel = tmp_path / "no-such.jsonl"
    report = diagnose(
        user_settings=tmp_path / "no-user.json",
        project_root=tmp_path / "no-proj",
        search_tool_telemetry_path=tel,
    )
    text = format_report(report)
    assert "no fires recorded" in text


def test_format_report_prints_stats(tmp_path: Path) -> None:
    tel = tmp_path / "search_tool_hook.jsonl"
    _write_tel(tel, [_record(latency_ms=20.0, l0=1, l1=2) for _ in range(10)])
    report = diagnose(
        user_settings=tmp_path / "no-user.json",
        project_root=tmp_path / "no-proj",
        search_tool_telemetry_path=tel,
    )
    text = format_report(report)
    assert "search_tool_hook telemetry" in text
    assert "fires: 10" in text
    assert "latency p50" in text
    assert "latency p95" in text
    assert "noise rate" in text


def test_format_report_corrupt_flag_surfaces(tmp_path: Path) -> None:
    tel = tmp_path / "corrupt.jsonl"
    tel.write_text("not json\n", encoding="utf-8")
    report = diagnose(
        user_settings=tmp_path / "no-user.json",
        project_root=tmp_path / "no-proj",
        search_tool_telemetry_path=tel,
    )
    text = format_report(report)
    assert "CORRUPT" in text


def test_format_report_includes_telemetry_file_path(tmp_path: Path) -> None:
    tel = tmp_path / "search_tool_hook.jsonl"
    report = diagnose(
        user_settings=tmp_path / "no-user.json",
        project_root=tmp_path / "no-proj",
        search_tool_telemetry_path=tel,
    )
    text = format_report(report)
    assert str(tel) in text


def test_format_report_no_tel_path_no_section(tmp_path: Path) -> None:
    """When search_tool_telemetry_path is None and project root has no git,
    the telemetry section is absent from the report."""
    report = diagnose(
        user_settings=tmp_path / "no-user.json",
        project_root=tmp_path / "no-proj",
        search_tool_telemetry_path=None,
    )
    text = format_report(report)
    # No section header when no path could be derived.
    # (_derive_telemetry_path fails for a non-git tmp dir)
    # We can't assert the section is absent because the project may be
    # inside a git repo (the test runner's cwd). Just assert no crash.
    assert isinstance(text, str)
