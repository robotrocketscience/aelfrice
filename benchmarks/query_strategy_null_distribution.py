"""#1581 — the query-strategy gate's floor, and the null distribution under it.

`tests/bench_gate/test_query_strategy.py` states a floor twice: once as the
bar the declared null model must **fail** (`bar_above`, the #1581
precondition to counting), and once as the assertion the shipped arm must
**clear**. Both readings use one number, and this script is what re-derives
it. Per the #1469 rule a published figure names the script that re-derives
it, so the floor and the distribution it sits above carry markers naming
this producer.

## Why a floor of exactly 0.0 was not enough

The gate shipped `> 0.0`. That floor is a tripwire for "retrieval returned
nothing at all" and nothing more: any null model scoring a hair above zero
clears it, so the #1581 precondition could never reject a corpus on the
strength of the bar. It rejected the old 30-row corpus on a *structural*
pre-filter instead — gold == pool on 30 of 30 rows — and the rebuilt corpus
passes those filters while still leaving the bar toothless.

Raising the floor into the measured window is what makes the bar itself a
null-model trap. The window has two edges, and this script measures the
lower one.

## The lower edge: what a model that cannot rank actually scores

The declared null is the row's candidate pool in deterministic shuffled
order. `tests.bench_gate.null_model.shuffled_ranker_score` is imported
here rather than reimplemented — it is the function the gate itself calls,
and it applies `shuffled_pool` under `SHUFFLE_SALT` — so the canonical
figure this script emits is computed by the code the gate runs.

One permutation is one sample, so the floor is not set against it:
`--seeds` re-shuffles under `N` independent seeds and the floor is derived
from the **maximum** over that sweep, which is the worst case a non-ranking
model reached.

## The floor rule

    floor = ceil(FLOOR_MULTIPLIER * max(null NDCG@k over the seed sweep), 2dp)

`FLOOR_MULTIPLIER` is 1.5, and it is a margin rather than a measurement:
the sweep's maximum is an empirical maximum over `N` draws, not a bound, so
the floor is set half again above it to leave room for a draw the sweep did
not take. Rounding *up* to two decimals keeps the published number short
enough to read in a failure message without ever rounding back under the
multiplier.

The upper edge — the shipped arm's score — is deliberately **not** measured
here. It is measured by the gate itself on every run, against the same
rows, and recorded through `record_property`, so it is a live number rather
than a published literal needing a marker. What this script reports about
it is the headroom, when `--shipped` is passed: the floor is only useful if
the shipped arm clears it by a wide margin, and a floor that crowds the
shipped score is a flaky gate rather than a strict one.

## This is a store-backed producer

The rows are labelled corpus and live only in the private lab
repository, outside this tree, mounted through `AELFRICE_CORPUS_ROOT`
(the directory-of-origin rule; `tests/corpus/v2_0/README.md` has the
export). A public runner has none, so the markers naming
this script carry `corpus=` and `producer-sha=` and CI holds them to
self-consistency and code staleness rather than re-running them (#1456).

Usage:

    uv run python -m benchmarks.query_strategy_null_distribution \\
        --corpus "$AELFRICE_CORPUS_ROOT/query_strategy"

    # machine-readable, the shape `scripts/check_derived_figures.py` reads
    uv run python -m benchmarks.query_strategy_null_distribution \\
        --corpus "$AELFRICE_CORPUS_ROOT/query_strategy" --emit-figures

Exit status is 1 when `--shipped` is given and the shipped score does not
clear the derived floor, so a caller can gate on the headroom.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any

from tests.bench_gate.null_model import (
    SHUFFLE_SALT,
    precision_at_k,
    shuffled_ranker_score,
)
from tests.retrieve_uplift_runner import ndcg_at_k

GOLD_KEY = "expected_top_k"
POOL_KEY = "beliefs"
K_KEY = "k"
DEFAULT_K = 10

DEFAULT_SEEDS = 200
"""Seeds in the sweep the floor is derived from.

Two hundred rather than a handful because the figure taken off the sweep is
its *maximum*, and a maximum over few draws understates the worst case the
floor has to sit above. Two hundred rather than more because the sweep is
pure arithmetic over id lists — it runs in well under a second — and the
multiplier below, not the draw count, is what carries the safety margin.
"""

FLOOR_MULTIPLIER = 1.5
"""Margin between the sweep maximum and the published floor.

The sweep maximum is an empirical maximum, not a bound. Half again above it
leaves room for a permutation the sweep did not draw, while staying far
enough under the shipped arm that the gate is not flaky. Both directions are
reported by `--shipped` rather than asserted here.
"""


def load_rows(corpus_dir: Path) -> list[dict[str, Any]]:
    """Every row under `corpus_dir/*.jsonl`, in sorted file order.

    Mirrors `tests.conftest.load_corpus_module` so the producer reads the
    same rows the gate loads. Raises rather than skipping on an empty
    directory: a producer that emits figures over zero rows publishes a
    number nobody measured.
    """
    rows: list[dict[str, Any]] = []
    for path in sorted(corpus_dir.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"no rows under {corpus_dir}; nothing to measure")
    return rows


def _row_k(row: dict[str, Any]) -> int:
    return int(row.get(K_KEY, DEFAULT_K))


def seeded_null_score(rows: list[dict[str, Any]], seed: int) -> tuple[float, float]:
    """Mean NDCG@k and P@k of one shuffle of every row's pool.

    The seed is composed with the row id exactly as `shuffled_pool` composes
    `SHUFFLE_SALT` with it, so the sweep and the canonical figure differ only
    in which salt they use — the sweep is the same null model under other
    permutations, not a second null model.
    """
    total_ndcg = 0.0
    total_prec = 0.0
    for row in rows:
        k = _row_k(row)
        gold = list(row[GOLD_KEY])
        ids = [str(b["id"]) for b in row[POOL_KEY]]
        random.Random(f"{seed}:{row.get('id', '')}").shuffle(ids)
        total_ndcg += ndcg_at_k(ids[:k], gold, k)
        total_prec += precision_at_k(ids, gold, k)
    n = len(rows)
    return total_ndcg / n, total_prec / n


def derive_floor(null_max: float) -> float:
    """The published floor: `FLOOR_MULTIPLIER * null_max`, rounded up to 2dp.

    Rounded up rather than to nearest, because rounding to nearest can land
    the published number *below* the multiplier it claims, which is the
    published-figure defect #1469 exists to stop.
    """
    return math.ceil(FLOOR_MULTIPLIER * null_max * 100) / 100


def measure(corpus_dir: Path, seeds: int = DEFAULT_SEEDS) -> dict[str, Any]:
    """Every figure this producer publishes, keyed as the markers name them."""
    rows = load_rows(corpus_dir)

    # The canonical figure runs the gate's own code on the gate's own salt,
    # so it cannot drift from what the gate computes at run time.
    canonical_ndcg = shuffled_ranker_score(
        rows, metric=ndcg_at_k, gold_key=GOLD_KEY, pool_key=POOL_KEY
    )
    canonical_prec = shuffled_ranker_score(
        rows, metric=precision_at_k, gold_key=GOLD_KEY, pool_key=POOL_KEY
    )

    ndcgs: list[float] = []
    precs: list[float] = []
    for seed in range(seeds):
        n, p = seeded_null_score(rows, seed)
        ndcgs.append(n)
        precs.append(p)

    null_max = max(ndcgs)
    # Structural shape, reported beside the scores because the two
    # pre-filters in `null_model.structural_prefilter` reject on it before
    # any of these numbers are reached.
    gold_sizes = [len(row[GOLD_KEY]) for row in rows]
    separable = sum(
        1
        for row in rows
        if len(row[GOLD_KEY]) < len(row[POOL_KEY]) and _row_k(row) < len(row[POOL_KEY])
    )
    median_gold = statistics.median(gold_sizes)
    return {
        "n_rows": len(rows),
        "n_seeds": seeds,
        "salt": SHUFFLE_SALT,
        "null_ndcg_canonical": round(canonical_ndcg, 4),
        "null_p_at_k_canonical": round(canonical_prec, 4),
        "null_ndcg_min": round(min(ndcgs), 4),
        "null_ndcg_mean": round(statistics.fmean(ndcgs), 4),
        "null_ndcg_max": round(null_max, 4),
        "null_p_at_k_max": round(max(precs), 4),
        "floor_multiplier": FLOOR_MULTIPLIER,
        "floor": derive_floor(null_max),
        "gold_eq_pool": sum(
            1 for row in rows if len(row[GOLD_KEY]) == len(row[POOL_KEY])
        ),
        "separable_share": round(separable / len(rows), 4),
        "gold_skew": round(max(gold_sizes) / median_gold, 2) if median_gold else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Re-derive the #1581 query-strategy null floor.",
    )
    parser.add_argument(
        "--corpus",
        required=True,
        type=Path,
        help="the query_strategy corpus directory (lab-side; holds *.jsonl)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=DEFAULT_SEEDS,
        help=f"seeds in the sweep the floor is derived from (default {DEFAULT_SEEDS})",
    )
    parser.add_argument(
        "--shipped",
        type=float,
        default=None,
        help=(
            "the shipped arm's NDCG@k, to report headroom against the derived "
            "floor. Exits 1 when it does not clear the floor."
        ),
    )
    parser.add_argument(
        "--emit-figures",
        action="store_true",
        help="print the figures as a JSON object and exit (the #1469 shape)",
    )
    args = parser.parse_args(argv)

    figures = measure(args.corpus, seeds=args.seeds)
    if args.emit_figures:
        print(json.dumps(figures, indent=2, sort_keys=True))
        return 0

    print(f"corpus={args.corpus} rows={figures['n_rows']} seeds={figures['n_seeds']}")
    print(f"  salt={figures['salt']!r}")
    print(
        f"  null NDCG@k  canonical={figures['null_ndcg_canonical']:.4f}  "
        f"min={figures['null_ndcg_min']:.4f} "
        f"mean={figures['null_ndcg_mean']:.4f} "
        f"max={figures['null_ndcg_max']:.4f}"
    )
    print(f"  null P@k     canonical={figures['null_p_at_k_canonical']:.4f}")
    print(
        f"  structural   gold==pool={figures['gold_eq_pool']}/{figures['n_rows']} "
        f"separable={figures['separable_share']:.1%} "
        f"gold_skew={figures['gold_skew']:.2f}x"
    )
    print(
        f"  floor        {figures['floor_multiplier']:g} x "
        f"{figures['null_ndcg_max']:.4f} -> {figures['floor']:g}"
    )
    if args.shipped is None:
        return 0
    headroom = args.shipped - figures["floor"]
    ratio = args.shipped / figures["floor"] if figures["floor"] else float("inf")
    print(
        f"  shipped      {args.shipped:.4f}  headroom={headroom:+.4f} "
        f"({ratio:.2f}x the floor)"
    )
    if headroom < 0:
        print("FAILS — the shipped arm does not clear the derived floor.")
        return 1
    print("OK — the floor sits between the null sweep's maximum and the shipped arm.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
