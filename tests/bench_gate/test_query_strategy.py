"""Bench gate for the shipped query strategy (#291 sub-issue #527, #1501).

Two arms over the same `retrieve()`: `transform_query(raw, store,
"legacy-bm25")` passes the raw query through, and `transform_query(raw,
store, "stack-r1-r3")` runs the R1 capitalised-token entity expand plus
the R3 per-store IDF-quantile clip.

**The contract is that the shipped default is the winning arm**, not that
the uplift is positive. #291 § Bench gates ratified `NDCG@k(stack-r1-r3)
> NDCG@k(legacy-bm25)` as the ship trigger for flipping the default, and
that flip shipped in v3.0 (#718). #1501 reverted it: #1177 replaced the
conjunctive FTS5 MATCH with a disjunction over the rarest tokens, which
is the recall cliff R3 existed to work around, and across that one commit
the same 30 rows moved from +0.2851 to −0.1324 — because `legacy-bm25`
gained 0.65 while `stack-r1-r3` gained 0.24.

So a fixed direction is the wrong shape for this gate. Asserted against
`DEFAULT_STRATEGY`, it goes red on a re-flip without a re-measure, and on
a regression of the disjunctive MATCH — which would restore the cliff and
put `stack-r1-r3` back in front.

A comparison alone is not enough, and the floor above it is load-bearing:
a breakage that zeroes both arms is a tie, and a tie satisfies `>=`.

The +0.05 absolute P@10 floor in #291's body was the flip-default trigger,
evaluated lab-side. It is moot: the default is `legacy-bm25` again.

Public CI skips when ``AELFRICE_CORPUS_ROOT`` is unset, per the
directory-of-origin rule (labelled corpus lives only in
``~/projects/aelfrice-lab/tests/corpus/v2_0/query_strategy/``).
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import pytest

from aelfrice.query_understanding import (
    DEFAULT_STRATEGY,
    LEGACY_STRATEGY,
    STACK_R1_R3_STRATEGY,
)
from tests.bench_gate.null_model import (
    bar_above,
    guard_ranking_gate,
    precision_at_k,
    shuffled_ranker_score,
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


def _null_precision(rows: list[dict]) -> float:
    """P@k of the same shuffle, recorded beside the gate's own NDCG@k.

    The gate scores NDCG, so NDCG is what the bar is read against. P@k
    is carried alongside because it is the number #1581 quotes and the
    plainer statement of the same degeneracy: with the gold set equal
    to the pool, nearly every position in the top-k is a hit.
    """
    return shuffled_ranker_score(
        rows,
        metric=precision_at_k,
        gold_key="expected_top_k",
        pool_key="beliefs",
    )


@pytest.mark.bench_gated
def test_query_strategy_uplift(
    aelfrice_corpus_root: Path,
    record_property: Callable[[str, object], None],
) -> None:
    rows = load_corpus_module(aelfrice_corpus_root, "query_strategy")
    assert rows, "query_strategy corpus produced zero rows"

    runner_mod = pytest.importorskip(
        "tests.retrieve_uplift_runner",
        reason=(
            "query-strategy uplift runner not yet wired (operator gate; "
            "#291 § Bench gates — pending lab-side corpus)"
        ),
    )

    measured: dict[str, object] = {}

    def shipped() -> float:
        results = runner_mod.run_query_strategy_uplift(rows)
        measured["results"] = results
        return {
            LEGACY_STRATEGY: results.mean_ndcg_off,
            STACK_R1_R3_STRATEGY: results.mean_ndcg_on,
        }[DEFAULT_STRATEGY]

    # The null-model precondition (#1581). The declared null is the
    # row's candidate pool in deterministic shuffled order, scored with
    # this gate's own NDCG@k against this gate's own `> 0.0` floor.
    # Measured 2026-09-20 on the mounted corpus: the two structural
    # pre-filters reject it before the arms run — the gold set is the
    # whole candidate pool on 30 of 30 rows — and a shuffle scores
    # NDCG@10 0.8261 / P@10 0.9933 against a floor of 0.0. Rebuilding
    # those rows is #1581's out-of-scope follow-on, not a reason to
    # weaken this.
    guard_ranking_gate(
        module="query_strategy",
        rows=rows,
        shipped=shipped,
        bar=bar_above(0.0),
        metric=runner_mod.ndcg_at_k,
        record_property=record_property,
        gold_key="expected_top_k",
        pool_key="beliefs",
        extra=(
            f"null_p_at_k={_null_precision(rows):.4f} "
            f"corpus_sha256={_corpus_digest(aelfrice_corpus_root)}"
        ),
    )

    results = measured["results"]
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
    # The floor comes first, and it is not redundant with the comparison
    # below. A comparison between two arms is silent about a breakage that
    # zeroes *both* — that is a tie, and a tie satisfies `>=`. Emptying the
    # FTS5 MATCH expression does exactly that: every row retrieves nothing,
    # both arms score 0.0, and a gate written only as a comparison reports
    # green at the release cut, which is the only place this tier runs.
    # These rows carry labelled `expected_top_k`, so a non-zero score is a
    # statement that retrieval happened at all.
    assert scores[DEFAULT_STRATEGY] > 0.0, (
        f"the shipped default ({DEFAULT_STRATEGY}) retrieved nothing "
        f"scoreable on any of {len(rows)} labelled rows:\n{detail}\n"
        f"  This is retrieval being broken, not a strategy comparison. "
        f"Look at the FTS5 MATCH builder before anything else — an empty "
        f"match expression produces exactly this."
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
