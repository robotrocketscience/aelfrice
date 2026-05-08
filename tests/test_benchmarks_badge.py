"""Tests for benchmarks.badge — README reproducibility-badge text formatter.

Issue: #477.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks import badge


def _write_report(
    tmp_path: Path,
    *,
    headline_cut: dict,
    results: dict,
) -> Path:
    p = tmp_path / "report.json"
    p.write_text(
        json.dumps(
            {
                "label": "test",
                "headline_cut": headline_cut,
                "metric_overrides": {},
                "results": results,
            }
        )
    )
    return p


def test_all_ok_renders_check_icon(tmp_path):
    report = _write_report(
        tmp_path,
        headline_cut={"a": [{"sub_key": None}], "b": [{"sub_key": None}]},
        results={
            "a": {"_": {"_status": "ok"}},
            "b": {"_": {"_status": "ok"}},
        },
    )
    text = badge.compute_badge_text(report, today="2026-05-08")
    assert text == "reproducibility: ✅ 2/2 ok · last run 2026-05-08"


def test_partial_renders_warn_icon(tmp_path):
    report = _write_report(
        tmp_path,
        headline_cut={"a": [{"sub_key": None}], "b": [{"sub_key": None}]},
        results={
            "a": {"_": {"_status": "ok"}},
            "b": {"_": {"_status": "error"}},
        },
    )
    text = badge.compute_badge_text(report, today="2026-05-08")
    assert text == "reproducibility: ⚠️ 1/2 ok · last run 2026-05-08"


def test_total_counts_subkeys_not_adapters(tmp_path):
    """An adapter with N parametrised invocations contributes N to total."""
    report = _write_report(
        tmp_path,
        headline_cut={
            "mab": [{"sub_key": "x"}, {"sub_key": "y"}, {"sub_key": "z"}],
            "amabench": [{"sub_key": None}],
        },
        results={
            "mab": {"x": {"_status": "ok"}, "y": {"_status": "ok"}, "z": {"_status": "ok"}},
            "amabench": {"_": {"_status": "ok"}},
        },
    )
    text = badge.compute_badge_text(report, today="2026-05-08")
    assert "4/4" in text


def test_skipped_counts_as_not_ok(tmp_path):
    """Per #479, skipped_data_missing is distinct from ok; it does not count."""
    report = _write_report(
        tmp_path,
        headline_cut={"a": [{"sub_key": None}], "b": [{"sub_key": None}]},
        results={
            "a": {"_": {"_status": "ok"}},
            "b": {"_": {"_status": "skipped_data_missing"}},
        },
    )
    text = badge.compute_badge_text(report, today="2026-05-08")
    assert "1/2 ok" in text
    assert text.startswith("reproducibility: ⚠️")


def test_canonical_v200_partial(tmp_path):
    """Sanity: today's checked-in canonical reports 6/11."""
    canonical = Path(__file__).parent.parent / "benchmarks" / "results" / "v2.0.0.json"
    if not canonical.exists():
        pytest.skip("canonical baseline not present")
    text = badge.compute_badge_text(canonical, today="2026-05-08")
    assert "6/11 ok" in text


def test_zero_total_does_not_render_check(tmp_path):
    """Empty headline_cut shouldn't produce the all-green icon."""
    report = _write_report(tmp_path, headline_cut={}, results={})
    text = badge.compute_badge_text(report, today="2026-05-08")
    assert text.startswith("reproducibility: ⚠️ 0/0 ok")


def test_today_defaults_to_utc(tmp_path):
    report = _write_report(
        tmp_path,
        headline_cut={"a": [{"sub_key": None}]},
        results={"a": {"_": {"_status": "ok"}}},
    )
    text = badge.compute_badge_text(report)
    # YYYY-MM-DD shape; don't pin the actual day.
    suffix = text.rsplit("last run ", 1)[1]
    assert len(suffix) == 10 and suffix[4] == "-" and suffix[7] == "-"
