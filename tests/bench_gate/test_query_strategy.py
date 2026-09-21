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
gained 0.65 while `stack-r1-r3` gained 0.24. Those 30 rows are the
corpus #1581 replaced, so the uplift figures in this paragraph are
history and do not describe what this gate measures today. The direction
is unchanged on the rebuilt rows, and by more: `legacy-bm25` 0.8716
against `stack-r1-r3` 0.6508.

So a fixed direction is the wrong shape for this gate. Asserted against
`DEFAULT_STRATEGY`, it goes red on a re-flip without a re-measure, and on
a regression of the disjunctive MATCH — which would restore the cliff and
put `stack-r1-r3` back in front.

A comparison alone is not enough, and the floor above it is load-bearing:
a breakage that zeroes both arms is a tie, and a tie satisfies `>=`.

The +0.05 absolute P@10 floor in #291's body was the flip-default trigger,
evaluated lab-side. It is moot: the default is `legacy-bm25` again.

**The floor is `_NULL_FLOOR`, and it is the same number the #1581 null
model is held to.** It was `0.0`, which is a tripwire for "retrieval
returned nothing" and nothing else — every null model clears it, so the
#1581 bar could never reject a corpus on its own strength. Raising it into
the window between the null sweep's maximum and the shipped arm is what
turns the bar into a null-model trap. See `_NULL_FLOOR` for the
derivation and for the producer that re-derives it.

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

# The floor this gate reads twice: as the bar the declared null model must
# **fail** (#1581's precondition to counting) and as the assertion the
# shipped default must **clear**. One number, because the two readings are
# the same claim — a score a shuffled pool can reach is not evidence that
# retrieval happened.
#
# Derived rather than picked. `benchmarks/query_strategy_null_distribution.py`
# re-shuffles every row's candidate pool under 200 independent seeds and
# takes the worst case: a null NDCG@k of 0.2307, against a canonical-salt
# null of 0.1499 and a sweep mean of 0.1346. The floor is that maximum
# times 1.5, rounded up to two decimals, which is 0.35 — half again above
# the worst draw, because a sweep maximum is an empirical maximum and not a
# bound.
#
# The margin on the other side is what keeps the gate from being flaky
# rather than strict: the shipped `legacy-bm25` arm measures 0.8716 on
# these rows, so it clears the floor by +0.5216 (2.49x), and the losing
# `stack-r1-r3` arm measures 0.6508 and clears it too — deliberately, since
# a floor that only the current default can pass would pre-judge the
# re-flip this gate's comparison exists to detect.
#
# Re-derive the lower edge with the corpus mounted:
#
#     uv run python -m benchmarks.query_strategy_null_distribution \
#         --corpus "$AELFRICE_CORPUS_ROOT/query_strategy" --shipped 0.8716
#
# <!-- derived: benchmarks/query_strategy_null_distribution.py#null_ndcg_max = 0.2307 corpus=lab-corpus/query_strategy-v1_0@2026-09-21 producer-sha=6ed2633f4bac -->
# <!-- derived: benchmarks/query_strategy_null_distribution.py#null_ndcg_canonical = 0.1499 corpus=lab-corpus/query_strategy-v1_0@2026-09-21 producer-sha=6ed2633f4bac -->
# <!-- derived: benchmarks/query_strategy_null_distribution.py#null_ndcg_mean = 0.1346 corpus=lab-corpus/query_strategy-v1_0@2026-09-21 producer-sha=6ed2633f4bac -->
# <!-- derived: benchmarks/query_strategy_null_distribution.py#floor = 0.35 corpus=lab-corpus/query_strategy-v1_0@2026-09-21 producer-sha=6ed2633f4bac -->
_NULL_FLOOR = 0.35


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
    # this gate's own NDCG@k against this gate's own `_NULL_FLOOR`.
    #
    # The 30-row corpus this replaced failed here twice over, and both
    # failures are why the floor moved. Structurally, the gold set was the
    # whole candidate pool on 30 of 30 rows, so no distractor could be lost
    # and a shuffle scored P@10 0.9933; the `separability` pre-filter
    # rejects that before the arms run. But the rebuilt rows clear both
    # pre-filters and a `> 0.0` bar would still have passed their null
    # model at 0.1499, which is the half a rebuild alone does not fix.
    guard_ranking_gate(
        module="query_strategy",
        rows=rows,
        shipped=shipped,
        bar=bar_above(_NULL_FLOOR),
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
    #
    # At `_NULL_FLOOR` rather than at 0.0 the assertion says more than
    # "retrieval happened at all": it says the shipped arm out-ranks a model
    # that cannot rank. Substituting the declared null model for `shipped`
    # lands at 0.1499 and fails here, which is the property that makes this
    # floor load-bearing — at 0.0 that substitution passed both assertions.
    assert scores[DEFAULT_STRATEGY] > _NULL_FLOOR, (
        f"the shipped default ({DEFAULT_STRATEGY}) scored at or below the "
        f"null-model floor of {_NULL_FLOOR:g} on {len(rows)} labelled "
        f"rows:\n{detail}\n"
        f"  A shuffle of each row's candidate pool reaches this floor's "
        f"worst case over 200 seeds, so this is retrieval being broken or "
        f"degenerate, not a strategy comparison. Look at the FTS5 MATCH "
        f"builder before anything else — an empty match expression scores "
        f"0.0 and produces exactly this. Re-derive the floor with "
        f"benchmarks/query_strategy_null_distribution.py before changing it."
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
