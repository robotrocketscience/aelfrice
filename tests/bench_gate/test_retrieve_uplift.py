"""Bench gate for #154 default-on flip — per-flag retrieve() NDCG@k uplift.

Loads the lab-mounted `retrieve_uplift` corpus and asserts that no
v1.7 flag regresses mean NDCG@k against the all-flags-off baseline.
A regression is the trigger to leave that flag default-off; a
positive uplift is the trigger to flip it default-on.

This test DOES NOT assert a specific positive uplift threshold.
That's an operator decision per flag, made on the resulting evidence
table — see the runner CLI:

    AELFRICE_CORPUS_ROOT=~/projects/aelfrice-lab/tests/corpus/v2_0 \\
        python -m tests.retrieve_uplift_runner

The bench-gate test only enforces the no-regression invariant; it
fails loudly if any flag is net-negative, succeeds quietly otherwise.

Skips on public CI (corpus absent) per the directory-of-origin rule.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import load_corpus_module
from tests.retrieve_uplift_runner import run_per_flag_uplift


@pytest.mark.bench_gated
def test_retrieve_per_flag_no_regression(aelfrice_corpus_root: Path) -> None:
    rows = load_corpus_module(aelfrice_corpus_root, "retrieve_uplift")
    assert rows, "retrieve_uplift corpus produced zero rows"

    results = run_per_flag_uplift(rows)
    regressions = [r for r in results if r.uplift < 0]
    detail = "\n".join(
        f"  {r.flag}: NDCG_off={r.mean_ndcg_off:.4f} "
        f"NDCG_on={r.mean_ndcg_on:.4f} uplift={r.uplift:+.4f}"
        for r in results
    )
    assert not regressions, (
        f"v1.7 flags regress NDCG@k against baseline:\n{detail}\n"
        f"regressing flags: {[r.flag for r in regressions]}"
    )
