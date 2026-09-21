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

The null-model precondition (#1581) is the baseline arm — this gate's
declared null model — plus the requirement that the baseline reaches
the gold on a real share of rows. A corpus where nothing is retrievable
without a flag reports +1.000 uplift for that flag and proves nothing
about it. The recorded line also carries `null_clears_bar=True`,
because a zero uplift clears a no-regression bar by construction: this
gate catches a regression, and its green is not positive evidence.

Skips on public CI (corpus absent) per the directory-of-origin rule.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from tests.bench_gate.null_model import (
    AblationArms,
    bar_at_least,
    guard_ablation_gate,
)
from tests.conftest import load_corpus_module
from tests.retrieve_uplift_runner import run_per_flag_uplift


@pytest.mark.bench_gated
def test_retrieve_per_flag_no_regression(
    aelfrice_corpus_root: Path,
    record_property: Callable[[str, object], None],
) -> None:
    rows = load_corpus_module(aelfrice_corpus_root, "retrieve_uplift")
    assert rows, "retrieve_uplift corpus produced zero rows"

    measured: dict[str, object] = {}

    def arms() -> AblationArms:
        results = run_per_flag_uplift(rows)
        measured["results"] = results
        # The baseline arm is shared by every flag, so the weakest
        # treatment arm is the one the no-regression bar is read
        # against, and the baseline's per-row scores are identical
        # across flags.
        worst = min(results, key=lambda r: r.uplift)
        return AblationArms(
            shipped=worst.mean_ndcg_on,
            ablated=worst.mean_ndcg_off,
            without_row_scores=worst.off_row_scores,
        )

    guard_ablation_gate(
        module="retrieve_uplift",
        rows=rows,
        arms=arms,
        bar=bar_at_least(0.0),
        record_property=record_property,
        gold_key="expected_top_k",
        pool_key="beliefs",
    )

    results = measured["results"]
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
