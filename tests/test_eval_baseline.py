"""Drift guard for the synthetic-corpus calibration baseline (#365 R5).

Pins ``benchmarks/posterior_ranking/baseline.json`` to the metric block
that ``aelf eval --json`` produces against the bundled public corpus
under default flags. Any change to the corpus, harness, or scorer that
moves a metric must be a deliberate baseline update — this test is the
PR-time tripwire that fires before the change reaches main, paired
with the push-to-main status check workflow that re-asserts it.

The ``corpus`` key is path-dependent (worktree absolute path) so it is
stripped before comparison; what we pin is the metric subset.
"""
from __future__ import annotations

import json
from pathlib import Path

from aelfrice import eval_harness as eh

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = REPO_ROOT / "benchmarks" / "posterior_ranking" / "baseline.json"


def _metrics_only(payload: dict) -> dict:
    return {k: v for k, v in payload.items() if k != "corpus"}


def test_baseline_file_exists_and_parses():
    assert BASELINE_PATH.is_file(), f"baseline missing at {BASELINE_PATH}"
    data = json.loads(BASELINE_PATH.read_text())
    assert "corpus" not in data, "baseline must not pin path-dependent corpus key"
    assert set(data) == {
        "k",
        "n_observations",
        "n_queries",
        "n_truncated_queries",
        "p_at_k",
        "roc_auc",
        "seed",
        "spearman_rho",
    }


def test_baseline_matches_default_eval_output():
    fixtures = eh.load_calibration_fixtures(eh.DEFAULT_CALIBRATION_CORPUS)
    report = eh.run_calibration_on_fixtures(
        fixtures, k=eh.DEFAULT_K, seed=eh.DEFAULT_SEED
    )
    observed = {
        "k": eh.DEFAULT_K,
        "n_observations": report.n_observations,
        "n_queries": report.n_queries,
        "n_truncated_queries": report.n_truncated_queries,
        "p_at_k": report.p_at_k,
        "roc_auc": report.roc_auc,
        "seed": eh.DEFAULT_SEED,
        "spearman_rho": report.spearman_rho,
    }
    expected = _metrics_only(json.loads(BASELINE_PATH.read_text()))
    assert observed == expected, (
        "synthetic-corpus calibration metrics drifted from pinned baseline. "
        "If intentional, regenerate baseline.json from `aelf eval --json` "
        "output (with `corpus` key stripped) in the same commit. "
        f"observed={observed!r} expected={expected!r}"
    )


def test_baseline_is_canonical_form():
    """Baseline must be sorted-keys, compact-separators, single line + \\n.

    This makes diffs against future regenerations one-line, and matches
    the canonical form `aelf eval --json` emits (sort_keys=True,
    separators=(',', ':')).
    """
    raw = BASELINE_PATH.read_text()
    assert raw.endswith("\n") and raw.count("\n") == 1, (
        "baseline must be exactly one line followed by a single trailing newline"
    )
    parsed = json.loads(raw)
    canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":")) + "\n"
    assert raw == canonical, (
        "baseline.json is not in canonical form (sort_keys, compact separators)"
    )
