"""Bake-off runner for the #228 wonder-consolidation campaign.

Wires the corpus, the three strategies, and the evaluator into a
single ``run_bakeoff`` entry point. Multi-seed sweeping per
Decision D in the planning memo (default N=10): metrics are
reported as mean across seeds, with per-seed detail kept in the
result dict for variance auditing.

CLI usage::

    uv run python -m aelfrice.wonder.runner \\
        --feedback-budget 16 --seeds 10 --output bake_off_R0.json

Output JSON shape::

    {
      "config": {...},
      "per_seed": [{seed, strategy_metrics, jaccard, verdict}, ...],
      "aggregate": {
        "strategy_metrics": {strategy: {metric: mean_value}},
        "jaccard": {"RW|TC": mean, ...},
        "verdict_distribution": {"single": n, ...},
        "majority_verdict": "..."
      }
    }
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from aelfrice.store import MemoryStore

from .evaluator import (
    H0_NULL_RATE,
    H0_PLUS_PP_FLOOR,
    StrategyMetrics,
    adoption_verdict,
    evaluate_strategy,
    pairwise_jaccard,
)
from .models import STRATEGY_RW, STRATEGY_STS, STRATEGY_TC, Phantom
from .simulator import build_corpus, populate_store
from .strategies import (
    random_walk,
    span_topic_sampling,
    triangle_closure,
)


def _run_one_seed(
    seed: int,
    *,
    n_topics: int,
    n_atoms_per_topic: int,
    n_walks: int,
    n_sts_samples: int,
    feedback_budget: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    corpus = build_corpus(
        rng=rng,
        n_topics=n_topics,
        n_atoms_per_topic=n_atoms_per_topic,
    )
    store = MemoryStore(":memory:")
    populate_store(store, corpus, rng=rng)

    rw = random_walk(store, rng=random.Random(seed + 1), n_walks=n_walks)
    tc = triangle_closure(store)
    sts = span_topic_sampling(
        store, rng=random.Random(seed + 2), n_samples=n_sts_samples
    )

    metrics_list: list[StrategyMetrics] = [
        evaluate_strategy(rw, corpus, feedback_budget_per_phantom=feedback_budget),
        evaluate_strategy(tc, corpus, feedback_budget_per_phantom=feedback_budget),
        evaluate_strategy(sts, corpus, feedback_budget_per_phantom=feedback_budget),
    ]
    strategy_to_phantoms: dict[str, list[Phantom]] = {
        STRATEGY_RW: rw,
        STRATEGY_TC: tc,
        STRATEGY_STS: sts,
    }
    j: dict[tuple[str, str], float] = {}
    pairs = [
        (STRATEGY_RW, STRATEGY_TC),
        (STRATEGY_RW, STRATEGY_STS),
        (STRATEGY_STS, STRATEGY_TC),
    ]
    for s1, s2 in pairs:
        key = tuple(sorted((s1, s2)))
        j[key] = pairwise_jaccard(
            strategy_to_phantoms[s1], strategy_to_phantoms[s2]
        )

    metrics_tuple = tuple(metrics_list)
    verdict = adoption_verdict(metrics_tuple, j)

    return {
        "seed": seed,
        "strategy_metrics": [asdict(m) for m in metrics_tuple],
        "jaccard": {f"{a}|{b}": v for (a, b), v in j.items()},
        "verdict": verdict,
    }


def _aggregate(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    strategies = [m["strategy"] for m in per_seed[0]["strategy_metrics"]]
    metric_names = [
        "confirmation_rate",
        "retrieval_freq_per_cost",
        "junk_rate",
        "mean_construction_cost",
        "n_phantoms",
    ]
    agg: dict[str, dict[str, float]] = {s: {} for s in strategies}
    for s_idx, strategy in enumerate(strategies):
        for name in metric_names:
            values = [
                seed["strategy_metrics"][s_idx][name] for seed in per_seed
            ]
            agg[strategy][name] = sum(values) / len(values)

    jaccard_keys = list(per_seed[0]["jaccard"].keys())
    jaccard_agg = {
        k: sum(s["jaccard"][k] for s in per_seed) / len(per_seed)
        for k in jaccard_keys
    }
    verdict_counter = Counter(s["verdict"] for s in per_seed)
    return {
        "strategy_metrics": agg,
        "jaccard": jaccard_agg,
        "verdict_distribution": dict(verdict_counter),
        "majority_verdict": verdict_counter.most_common(1)[0][0],
    }


def run_bakeoff(
    *,
    n_topics: int = 8,
    n_atoms_per_topic: int = 25,
    n_walks: int = 50,
    n_sts_samples: int = 50,
    feedback_budget: int = 16,
    seeds: int = 10,
) -> dict[str, Any]:
    """Run the full bake-off and return the result dict.

    ``seeds`` controls the number of random seeds swept per
    Decision D in the planning memo. Defaults to 10 (variance
    estimate for the public-side R<final> run).
    """
    per_seed = [
        _run_one_seed(
            seed=s,
            n_topics=n_topics,
            n_atoms_per_topic=n_atoms_per_topic,
            n_walks=n_walks,
            n_sts_samples=n_sts_samples,
            feedback_budget=feedback_budget,
        )
        for s in range(seeds)
    ]
    return {
        "config": {
            "n_topics": n_topics,
            "n_atoms_per_topic": n_atoms_per_topic,
            "n_walks": n_walks,
            "n_sts_samples": n_sts_samples,
            "feedback_budget": feedback_budget,
            "seeds": seeds,
            "h0_null_rate": H0_NULL_RATE,
            "h0_floor": H0_NULL_RATE + H0_PLUS_PP_FLOOR,
        },
        "per_seed": per_seed,
        "aggregate": _aggregate(per_seed),
    }


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m aelfrice.wonder.runner",
        description="Run the wonder-consolidation strategy bake-off (#228).",
    )
    p.add_argument("--n-topics", type=int, default=8)
    p.add_argument("--n-atoms-per-topic", type=int, default=25)
    p.add_argument("--n-walks", type=int, default=50)
    p.add_argument("--n-sts-samples", type=int, default=50)
    p.add_argument("--feedback-budget", type=int, default=16)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the result JSON here; default stdout.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    result = run_bakeoff(
        n_topics=args.n_topics,
        n_atoms_per_topic=args.n_atoms_per_topic,
        n_walks=args.n_walks,
        n_sts_samples=args.n_sts_samples,
        feedback_budget=args.feedback_budget,
        seeds=args.seeds,
    )
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output is None:
        sys.stdout.write(payload + "\n")
    else:
        args.output.write_text(payload + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main", "run_bakeoff"]
