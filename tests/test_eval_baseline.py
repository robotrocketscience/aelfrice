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
import os
from pathlib import Path

import pytest

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


def _isolate_from_ambient_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run with default flags: no project config and no `AELFRICE_*` env.

    The harness's `retrieve()` reads 23 TOML keys, found by walking up
    from the working directory, and more than 30 `AELFRICE_*` variables.
    Some of the variables override the harness's own arguments, because
    each resolver checks the environment first. The pin is for default
    flags, so neither a developer's project config nor their shell may
    reach it (#1659, the #1295 class). The `.git` marker stops the walk
    here instead of letting it climb out of `tmp_path`.
    """
    clean = tmp_path / "clean"
    (clean / ".git").mkdir(parents=True)
    monkeypatch.chdir(clean)
    for name in [n for n in os.environ if n.startswith("AELFRICE_")]:
        monkeypatch.delenv(name)


def _assert_matches_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolate_from_ambient_config(tmp_path, monkeypatch)
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


def test_baseline_matches_default_eval_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_matches_baseline(tmp_path, monkeypatch)


def test_the_baseline_ignores_an_ambient_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Started from a project whose config raises the posterior weight,
    under a shell that sets knobs the metrics respond to, the pinned
    metrics still match (#1659). Each arm fails alone without its guard:
    the working-directory config without the `chdir`, the config above
    `tmp_path` without the `.git` marker, and the environment without
    the `delenv` loop."""
    config = "[retrieval]\nposterior_weight = 1.5\n"
    (tmp_path / ".aelfrice.toml").write_text(config, encoding="utf-8")
    dirty = tmp_path / "dirty"
    (dirty / ".git").mkdir(parents=True)
    (dirty / ".aelfrice.toml").write_text(config, encoding="utf-8")
    monkeypatch.chdir(dirty)
    monkeypatch.setenv("AELFRICE_BM25F", "0")
    monkeypatch.setenv("AELFRICE_USE_GAMMA_POSTERIOR_TEMPERATURE", "1")
    # The arms are live before isolation: the harness would read 1.5 here,
    # from `dirty` and from above `tmp_path`, so a pass below is the
    # isolation's doing and not a mis-spelled key's.
    from aelfrice.retrieval import resolve_posterior_weight

    assert resolve_posterior_weight() == 1.5
    assert resolve_posterior_weight(start=tmp_path) == 1.5
    _assert_matches_baseline(tmp_path, monkeypatch)


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
