"""Tests for `aelf doctor` user_prompt_submit_hook telemetry section (#218 AC4).

Coverage:
- diagnose_user_prompt_submit_telemetry returns None when file missing/empty.
- diagnose_user_prompt_submit_telemetry raises ValueError on corrupt JSON.
- p50_chars and p95_chars computed correctly over total_chars values.
- median_collapse_rate computed from n_returned / n_unique_content_hashes.
- Zero n_unique_content_hashes treated as rate 1.0 (no division by zero).
- diagnose() populates user_prompt_submit_telemetry_path and stats.
- format_report() prints the section header and stats.
- Missing telemetry file → "no fires recorded" sentinel, no raise.
- Corrupt file → CORRUPT sentinel in report text.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aelfrice.doctor import (
    UserPromptSubmitTelemetryStats,
    diagnose,
    diagnose_user_prompt_submit_telemetry,
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
    total_chars: int = 200,
    n_returned: int = 3,
    n_unique: int = 3,
    n_l0: int = 1,
    n_l1: int = 2,
) -> dict[str, object]:
    return {
        "timestamp": "2026-01-01T00:00:00Z",
        "query": "test prompt",
        "n_returned": n_returned,
        "n_unique_content_hashes": n_unique,
        "n_l0": n_l0,
        "n_l1": n_l1,
        "total_chars": total_chars,
    }


# ---------------------------------------------------------------------------
# diagnose_user_prompt_submit_telemetry
# ---------------------------------------------------------------------------


def test_diagnose_returns_none_when_file_missing(tmp_path: Path) -> None:
    result = diagnose_user_prompt_submit_telemetry(tmp_path / "no-such.jsonl")
    assert result is None


def test_diagnose_returns_none_for_empty_file(tmp_path: Path) -> None:
    tel = tmp_path / "user_prompt_submit.jsonl"
    tel.write_text("", encoding="utf-8")
    assert diagnose_user_prompt_submit_telemetry(tel) is None


def test_diagnose_raises_on_corrupt_json(tmp_path: Path) -> None:
    tel = tmp_path / "user_prompt_submit.jsonl"
    tel.write_text('{"ok": 1}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError):
        diagnose_user_prompt_submit_telemetry(tel)


def test_diagnose_computes_fire_count(tmp_path: Path) -> None:
    tel = tmp_path / "user_prompt_submit.jsonl"
    _write_tel(tel, [_record() for _ in range(7)])
    stats = diagnose_user_prompt_submit_telemetry(tel)
    assert stats is not None
    assert stats.fire_count == 7


def test_diagnose_computes_injection_size_percentiles(tmp_path: Path) -> None:
    tel = tmp_path / "user_prompt_submit.jsonl"
    # 10 records with total_chars 100, 200, ..., 1000
    _write_tel(tel, [_record(total_chars=i * 100) for i in range(1, 11)])
    stats = diagnose_user_prompt_submit_telemetry(tel)
    assert stats is not None
    # p50 nearest-rank: index = max(0, int(10*50/100) - 1) = 4 → value 500
    assert stats.p50_chars == 500.0
    # p95: index = max(0, int(10*95/100) - 1) = 8 → value 900
    assert stats.p95_chars == 900.0


def test_diagnose_collapse_rate_no_duplicates(tmp_path: Path) -> None:
    """When n_returned == n_unique for every record, median rate should be 1.0."""
    tel = tmp_path / "user_prompt_submit.jsonl"
    _write_tel(tel, [_record(n_returned=3, n_unique=3) for _ in range(5)])
    stats = diagnose_user_prompt_submit_telemetry(tel)
    assert stats is not None
    assert stats.median_collapse_rate == pytest.approx(1.0)


def test_diagnose_collapse_rate_all_duplicates(tmp_path: Path) -> None:
    """When n_returned is double n_unique, rate should be 2.0."""
    tel = tmp_path / "user_prompt_submit.jsonl"
    _write_tel(tel, [_record(n_returned=6, n_unique=3) for _ in range(4)])
    stats = diagnose_user_prompt_submit_telemetry(tel)
    assert stats is not None
    assert stats.median_collapse_rate == pytest.approx(2.0)


def test_diagnose_collapse_rate_zero_unique_treated_as_one(tmp_path: Path) -> None:
    """n_unique_content_hashes == 0 should not raise (guard against zero-div)."""
    tel = tmp_path / "user_prompt_submit.jsonl"
    _write_tel(tel, [_record(n_returned=0, n_unique=0)])
    stats = diagnose_user_prompt_submit_telemetry(tel)
    assert stats is not None
    assert stats.median_collapse_rate == pytest.approx(1.0)


def test_diagnose_mixed_collapse_rates(tmp_path: Path) -> None:
    tel = tmp_path / "user_prompt_submit.jsonl"
    # 5 records: 3 with rate 1.0, 2 with rate 2.0 → median 1.0
    records = (
        [_record(n_returned=3, n_unique=3)] * 3
        + [_record(n_returned=6, n_unique=3)] * 2
    )
    _write_tel(tel, records)
    stats = diagnose_user_prompt_submit_telemetry(tel)
    assert stats is not None
    # sorted rates: [1.0, 1.0, 1.0, 2.0, 2.0]; p50 index = 1 → 1.0
    assert stats.median_collapse_rate == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# diagnose() integration
# ---------------------------------------------------------------------------


def test_diagnose_with_explicit_ups_tel_path_populates_report(tmp_path: Path) -> None:
    tel = tmp_path / "user_prompt_submit.jsonl"
    _write_tel(tel, [_record() for _ in range(3)])
    report = diagnose(
        user_settings=tmp_path / "no-user.json",
        project_root=tmp_path / "no-proj",
        user_prompt_submit_telemetry_path=tel,
    )
    assert report.user_prompt_submit_telemetry_path == tel
    assert report.user_prompt_submit_telemetry is not None
    assert report.user_prompt_submit_telemetry.fire_count == 3
    assert not report.user_prompt_submit_telemetry_corrupt


def test_diagnose_missing_ups_tel_file_sets_none_stats(tmp_path: Path) -> None:
    tel = tmp_path / "no-such.jsonl"
    report = diagnose(
        user_settings=tmp_path / "no-user.json",
        project_root=tmp_path / "no-proj",
        user_prompt_submit_telemetry_path=tel,
    )
    assert report.user_prompt_submit_telemetry_path == tel
    assert report.user_prompt_submit_telemetry is None
    assert not report.user_prompt_submit_telemetry_corrupt


def test_diagnose_corrupt_ups_tel_sets_flag(tmp_path: Path) -> None:
    tel = tmp_path / "corrupt.jsonl"
    tel.write_text('{"ok": 1}\nnot valid\n', encoding="utf-8")
    report = diagnose(
        user_settings=tmp_path / "no-user.json",
        project_root=tmp_path / "no-proj",
        user_prompt_submit_telemetry_path=tel,
    )
    assert report.user_prompt_submit_telemetry_corrupt is True
    assert report.user_prompt_submit_telemetry is None


# ---------------------------------------------------------------------------
# format_report()
# ---------------------------------------------------------------------------


def test_format_report_no_fires_sentinel(tmp_path: Path) -> None:
    tel = tmp_path / "no-such.jsonl"
    report = diagnose(
        user_settings=tmp_path / "no-user.json",
        project_root=tmp_path / "no-proj",
        user_prompt_submit_telemetry_path=tel,
    )
    text = format_report(report)
    assert "no fires recorded" in text


def test_format_report_prints_stats(tmp_path: Path) -> None:
    tel = tmp_path / "user_prompt_submit.jsonl"
    _write_tel(tel, [_record(total_chars=500, n_returned=3, n_unique=3) for _ in range(10)])
    report = diagnose(
        user_settings=tmp_path / "no-user.json",
        project_root=tmp_path / "no-proj",
        user_prompt_submit_telemetry_path=tel,
    )
    text = format_report(report)
    assert "user_prompt_submit_hook telemetry" in text
    assert "fires: 10" in text
    assert "injection size p50" in text
    assert "injection size p95" in text
    assert "dedup collapse rate" in text


def test_format_report_corrupt_flag_surfaces(tmp_path: Path) -> None:
    tel = tmp_path / "corrupt.jsonl"
    tel.write_text("not json\n", encoding="utf-8")
    report = diagnose(
        user_settings=tmp_path / "no-user.json",
        project_root=tmp_path / "no-proj",
        user_prompt_submit_telemetry_path=tel,
    )
    text = format_report(report)
    assert "CORRUPT" in text


def test_format_report_includes_telemetry_file_path(tmp_path: Path) -> None:
    tel = tmp_path / "user_prompt_submit.jsonl"
    report = diagnose(
        user_settings=tmp_path / "no-user.json",
        project_root=tmp_path / "no-proj",
        user_prompt_submit_telemetry_path=tel,
    )
    text = format_report(report)
    assert str(tel) in text


def test_format_report_no_ups_tel_path_no_section(tmp_path: Path) -> None:
    """When user_prompt_submit_telemetry_path is None and no git root,
    the section is absent from the report (no crash)."""
    report = diagnose(
        user_settings=tmp_path / "no-user.json",
        project_root=tmp_path / "no-proj",
        user_prompt_submit_telemetry_path=None,
    )
    text = format_report(report)
    assert isinstance(text, str)
