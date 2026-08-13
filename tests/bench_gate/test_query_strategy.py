"""Bench gate for #291 query-strategy uplift (sub-issue #527).

#291 § Bench gates ratifies a real-corpus uplift contract:

    NDCG@k(stack-r1-r3) > NDCG@k(legacy-bm25)

The OFF arm runs `transform_query(raw, store, "legacy-bm25")` (raw
passthrough) into `retrieve()`. The ON arm runs `transform_query(raw,
store, "stack-r1-r3")` (R1 capitalised-token entity expand + R3 per-
store IDF-quantile clip) into the same `retrieve()`. Strictly positive
uplift is the ship trigger; zero or negative uplift fails the gate.

The +0.05 absolute P@10 floor from #291 body is the *flip-default*
trigger evaluated lab-side once the labelled corpus exists; the gate
asserted here is the simpler `> 0` shape so PR-2.6 (corpus seeding) and
PR-3 (default flip) can land independently.

Public CI skips when ``AELFRICE_CORPUS_ROOT`` is unset, per the
directory-of-origin rule (labelled corpus lives only in
``~/projects/aelfrice-lab/tests/corpus/v2_0/query_strategy/``).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from aelfrice.query_understanding import (
    DEFAULT_STRATEGY,
    LEGACY_STRATEGY,
    STACK_R1_R3_STRATEGY,
)
from tests.conftest import load_corpus_module


def _corpus_digest(root: Path) -> str:
    """Return a short digest over the query_strategy corpus files.

    Stamped into the failure text so a red gate records *which* corpus
    produced the number. Computed rather than pinned: a literal here
    would assert the corpus never changes, which is a different contract
    from the one this gate is for, and it would go red on an intended
    re-label instead of on a retrieval change.
    """
    h = hashlib.sha256()
    for p in sorted((root / "query_strategy").glob("*.jsonl")):
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


@pytest.mark.bench_gated
def test_query_strategy_uplift(aelfrice_corpus_root: Path) -> None:
    rows = load_corpus_module(aelfrice_corpus_root, "query_strategy")
    assert rows, "query_strategy corpus produced zero rows"

    runner_mod = pytest.importorskip(
        "tests.retrieve_uplift_runner",
        reason=(
            "query-strategy uplift runner not yet wired (operator gate; "
            "#291 § Bench gates — pending lab-side corpus)"
        ),
    )

    results = runner_mod.run_query_strategy_uplift(rows)
    scores = {
        LEGACY_STRATEGY: results.mean_ndcg_off,
        STACK_R1_R3_STRATEGY: results.mean_ndcg_on,
    }
    other = next(s for s in scores if s != DEFAULT_STRATEGY)
    detail = (
        f"  NDCG_legacy_bm25={results.mean_ndcg_off:.4f} "
        f"NDCG_stack_r1_r3={results.mean_ndcg_on:.4f} "
        f"uplift={results.uplift:+.4f}\n"
        f"  default={DEFAULT_STRATEGY} rows={len(rows)} "
        f"corpus_sha256={_corpus_digest(aelfrice_corpus_root)}"
    )
    assert scores[DEFAULT_STRATEGY] >= scores[other], (
        f"the shipped default ({DEFAULT_STRATEGY}) is not the winning arm "
        f"on the labelled corpus:\n{detail}\n"
        f"  Two ways to reach this. Either the default was re-flipped "
        f"without a re-measure, or the disjunctive MATCH (#1177, "
        f"4db6744d) regressed and the AND recall cliff is back — that "
        f"cliff is what made stack-r1-r3 win in v3.0 (#718). Find out "
        f"which before changing this assertion."
    )


# Per-rebuild p99 latency budget from #291 § Bench gates.
#
#     p99(stack-r1-r3) <= p99(legacy-bm25) + 5 ms
#
# Scope: timed span is `transform_query → retrieve` per row, the only
# spans that differ between the two arms. Downstream rebuild work
# (compression, packing) is strategy-invariant and intentionally not
# included. Sample count: `n_rows * reps_per_row` per arm with one
# warmup repetition discarded per arm per row.
_LATENCY_BUDGET_NS = 5_000_000


@pytest.mark.bench_gated
def test_query_strategy_latency(aelfrice_corpus_root: Path) -> None:
    rows = load_corpus_module(aelfrice_corpus_root, "query_strategy")
    assert rows, "query_strategy corpus produced zero rows"

    runner_mod = pytest.importorskip(
        "tests.retrieve_uplift_runner",
        reason=(
            "query-strategy latency runner not yet wired (operator gate; "
            "#291 § Bench gates — pending lab-side corpus)"
        ),
    )

    results = runner_mod.run_query_strategy_latency(rows, reps_per_row=20)
    detail = (
        f"  p99_legacy_bm25={results.p99_off_ns/1e6:.3f}ms "
        f"p99_stack_r1_r3={results.p99_on_ns/1e6:.3f}ms "
        f"delta={results.delta_ns/1e6:+.3f}ms "
        f"budget=+{_LATENCY_BUDGET_NS/1e6:.1f}ms "
        f"(n_rows={results.n_rows} reps_per_row={results.reps_per_row})"
    )
    assert results.delta_ns <= _LATENCY_BUDGET_NS, (
        f"query-strategy stack-r1-r3 p99 latency exceeds legacy by "
        f"more than {_LATENCY_BUDGET_NS/1e6:.1f}ms on {len(rows)} "
        f"rows:\n{detail}"
    )
