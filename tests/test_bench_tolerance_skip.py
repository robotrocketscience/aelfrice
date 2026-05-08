"""Tolerance SKIP-verdict tests for #479.

Verifies that when an observed sub-result carries
`_status: skipped_data_missing`, the band-check emits Verdict.SKIP for
each canonical leaf under that sub-result and `summarize()` does not
promote SKIP to FAIL or WARN.
"""
from __future__ import annotations

from benchmarks import tolerance
from benchmarks.tolerance import Verdict


def _canonical(results: dict) -> dict:
    return {
        "schema_version": 2,
        "label": "test canonical",
        "captured_at_utc": "2026-05-08T00:00:00Z",
        "git_commit": "deadbeef",
        "aelfrice_version": "2.1.0",
        "harness_version": "1",
        "headline_cut": {},
        "results": results,
    }


def test_skipped_observed_emits_skip_not_fail() -> None:
    cano = _canonical({"structmemeval": {"location": {"em": 0.5, "f1": 0.6}}})
    obs = _canonical({
        "structmemeval": {
            "location": {"_status": "skipped_data_missing"},
        },
    })
    checks = tolerance.check_report(cano, obs)
    assert len(checks) == 2
    assert all(c.verdict == Verdict.SKIP for c in checks)
    assert all(c.band_kind == "skipped" for c in checks)


def test_skipped_at_adapter_root_propagates_to_all_descendants() -> None:
    """If `_status: skipped_data_missing` is at the adapter level,
    every nested canonical leaf becomes SKIP."""
    cano = _canonical({
        "structmemeval": {
            "location": {"em": 0.5, "f1": 0.6, "latency_ms": 100},
        },
    })
    obs = _canonical({
        "structmemeval": {"_status": "skipped_data_missing"},
    })
    checks = tolerance.check_report(cano, obs)
    assert len(checks) == 3
    assert all(c.verdict == Verdict.SKIP for c in checks)


def test_summarize_skip_does_not_block_pass() -> None:
    """A run with all-PASS leaves and one SKIP sub-result still rolls
    up to PASS (not WARN, not FAIL)."""
    cano = _canonical({
        "mab": {"split_a": {"f1": 0.5}},
        "structmemeval": {"location": {"em": 0.5}},
    })
    obs = _canonical({
        "mab": {"split_a": {"f1": 0.51}},
        "structmemeval": {"location": {"_status": "skipped_data_missing"}},
    })
    checks = tolerance.check_report(cano, obs)
    overall, counts = tolerance.summarize(checks)
    assert overall == Verdict.PASS
    assert counts[Verdict.SKIP.value] == 1
    assert counts[Verdict.PASS.value] == 1


def test_summarize_fail_still_dominates_skip() -> None:
    """Skip alongside an actual FAIL → overall FAIL (FAIL wins)."""
    cano = _canonical({
        "mab": {"split_a": {"f1": 0.5}},
        "structmemeval": {"location": {"em": 0.5}},
    })
    obs = _canonical({
        "mab": {"split_a": {"f1": 0.99}},  # huge regression → FAIL
        "structmemeval": {"location": {"_status": "skipped_data_missing"}},
    })
    checks = tolerance.check_report(cano, obs)
    overall, counts = tolerance.summarize(checks)
    assert overall == Verdict.FAIL
    assert counts[Verdict.SKIP.value] == 1
    assert counts[Verdict.FAIL.value] == 1


def test_skipped_does_not_short_circuit_other_adapters() -> None:
    """Ensure the SKIP check on one adapter does not bleed across
    sibling adapters: a sibling adapter with no `_status` skip should
    still get its leaves checked normally."""
    cano = _canonical({
        "skipped_one": {"x": {"f1": 0.5}},
        "ok_one": {"y": {"f1": 0.5}},
    })
    obs = _canonical({
        "skipped_one": {"x": {"_status": "skipped_data_missing"}},
        "ok_one": {"y": {"f1": 0.51}},
    })
    checks = tolerance.check_report(cano, obs)
    by_path = {c.path: c for c in checks}
    assert by_path[("skipped_one", "x", "f1")].verdict == Verdict.SKIP
    assert by_path[("ok_one", "y", "f1")].verdict == Verdict.PASS


def test_skipped_status_only_triggers_on_skipped_data_missing() -> None:
    """Other `_status` values (e.g. `error`, `ok`) do not trigger SKIP.
    `error` collapses through to the existing missing-leaf FAIL path.
    """
    cano = _canonical({"mab": {"split_a": {"f1": 0.5}}})
    obs = _canonical({"mab": {"split_a": {"_status": "error"}}})
    checks = tolerance.check_report(cano, obs)
    assert len(checks) == 1
    assert checks[0].verdict == Verdict.FAIL
