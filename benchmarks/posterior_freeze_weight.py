"""Measure where `posterior_weight` stops changing the ranking (#1584).

`posterior_weight` responds as a staircase: two candidates' scores differ
by an expression affine in the weight, so a pair with different posteriors
swaps at most once, and above the largest crossing no finite weight
reorders anything again. That largest crossing is the **freeze weight**.

#1584 measured a freeze weight of about 2.08 on the bundled 7-fixture
calibration corpus and asked whether that is an artefact of a corpus with
at most 4 candidates per query. An algebra-only simulation in that issue
predicted production would sit far higher. This producer answers the
question by measurement instead, through production `retrieve()`.

Why a ladder rather than a bisection: the bracket only needs to be tight
enough to compare against 2.08, and a bisection assumes the per-pair
single-crossing property that `--check-monotone` exists to verify rather
than assume. `retrieve()` under shipped defaults packs with a greedy
clustering fill rather than sorting, which is not obliged to be monotone
in the weight, so run `--check-monotone` before trusting a bracket.

**Never point this at a live store.** Opening a `MemoryStore` performs
schema DDL and migrations, so it is a write. Byte-copy the database
first and pass the copy:

    cp .git/aelfrice/memory.db /tmp/probe/memory.db
    cp .git/aelfrice/memory.db-wal /tmp/probe/ 2>/dev/null || true
    uv run python benchmarks/posterior_freeze_weight.py /tmp/probe/memory.db

Usage:
    posterior_freeze_weight.py STORE [--queries N] [--seed N]
                                     [--check-monotone] [--dry-run]

Exits non-zero if the store carries a single distinct posterior mean, in
which case the blend term is a constant offset and no crossing exists.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

#: Geometric ladder. Dense below 4 because the question is whether the
#: freeze weight sits near the 2.08 measured on the fixture corpus, and
#: wide above it so a genuinely saturating store is still bracketed.
LADDER: tuple[float, ...] = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0, 16.0, 32.0,
    64.0, 128.0, 256.0, 512.0, 1024.0,
)

#: Dense ladder for the monotonicity check only.
MONOTONE_LADDER: tuple[float, ...] = tuple(
    round(0.1 * i, 2) for i in range(0, 121)
)

DEFAULT_QUERIES = 40
DEFAULT_SEED = 0
L1_LIMIT = 50


def _queries_from_store(store, n: int, seed: int) -> list[str]:
    """Synthesise queries from the store's own belief contents.

    Biases toward strong BM25 matches, which widens the intra-query BM25
    spread. Since a crossing weight is a BM25 log-gap divided by a
    posterior log-gap, that bias pushes the measured freeze weight
    **up** — so a low measured value is conservative.
    """
    ids = store.list_belief_ids()
    rng = random.Random(seed)
    out: list[str] = []
    for bid in rng.sample(ids, min(len(ids), n * 10)):
        belief = store.get_belief(bid)
        if belief is None:
            continue
        words = [w for w in belief.content.split() if len(w) > 3][:6]
        if len(words) >= 3:
            out.append(" ".join(words))
        if len(out) >= n:
            break
    return out


def _ranked(store, query: str, weight: float) -> tuple[str, ...]:
    from aelfrice.retrieval import retrieve  # noqa: PLC0415

    os.environ["AELFRICE_POSTERIOR_WEIGHT"] = str(weight)
    hits = retrieve(
        store,
        query,
        l1_limit=L1_LIMIT,
        entity_index_enabled=False,
        bfs_enabled=False,
        posterior_weight=None,
    )
    return tuple(b.id for b in hits)


def _posterior_grid(store) -> dict[tuple[float, float], int]:
    grid: dict[tuple[float, float], int] = {}
    for bid in store.list_belief_ids():
        belief = store.get_belief(bid)
        if belief is None:
            continue
        key = (belief.alpha, belief.beta)
        grid[key] = grid.get(key, 0) + 1
    return grid


def _check_monotone(store, queries: list[str]) -> tuple[int, int]:
    """Return (queries whose ranking reverts, queries with a double cross).

    Both must be zero for a ladder bracket to mean what it says.
    """
    reverting = 0
    multi = 0
    for query in queries:
        seqs = [_ranked(store, query, w) for w in MONOTONE_LADDER]
        seen: dict[tuple[str, ...], int] = {}
        reverted = False
        for i, seq in enumerate(seqs):
            if seq in seen and seen[seq] != i - 1:
                reverted = True
            seen[seq] = i
        if reverted:
            reverting += 1
        common = set(seqs[0])
        for seq in seqs:
            common &= set(seq)
        ordered = sorted(common)
        worst = 0
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                a, b = ordered[i], ordered[j]
                rel = [s.index(a) < s.index(b) for s in seqs]
                flips = sum(
                    1 for k in range(1, len(rel)) if rel[k] != rel[k - 1]
                )
                worst = max(worst, flips)
        if worst > 1:
            multi += 1
    return reverting, multi


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("store", type=Path, help="path to a COPY of a store")
    parser.add_argument("--queries", type=int, default=DEFAULT_QUERIES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--check-monotone",
        action="store_true",
        help="verify no pair crosses twice before trusting the bracket",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report the store's posterior grid and exit without retrieving",
    )
    args = parser.parse_args(argv)

    if not args.store.is_file():
        print(f"error: no store at {args.store}", file=sys.stderr)
        return 2

    from aelfrice.store import MemoryStore  # noqa: PLC0415

    store = MemoryStore(str(args.store))
    try:
        total = len(store.list_belief_ids())
        grid = _posterior_grid(store)
        means = {a / (a + b) for a, b in grid}
        print(f"store: {args.store}")
        print(f"beliefs: {total}")
        print(f"distinct (alpha, beta) pairs: {len(grid)}")
        for (a, b), n in sorted(grid.items(), key=lambda kv: -kv[1]):
            share = 100.0 * n / total if total else 0.0
            print(f"  ({a}, {b}) -> {a / (a + b):.4f}  {n:>7}  {share:5.2f}%")
        if len(means) <= 1:
            print(
                "DEGENERATE: one distinct posterior mean, so the blend term "
                "is a constant offset and no crossing exists at any weight.",
                file=sys.stderr,
            )
            return 1
        if args.dry_run:
            return 0

        queries = _queries_from_store(store, args.queries, args.seed)
        print(f"queries: {len(queries)}")

        if args.check_monotone:
            reverting, multi = _check_monotone(store, queries)
            print(f"rankings that revert to an earlier one: {reverting}")
            print(f"queries with a pair crossing twice:     {multi}")
            if reverting or multi:
                print(
                    "NON-MONOTONE: the ladder bracket below is not a valid "
                    "upper bound for max(w*) on this store.",
                    file=sys.stderr,
                )

        lows: list[float] = []
        frozen_everywhere = 0
        n_candidates: list[int] = []
        for query in queries:
            seqs = [_ranked(store, query, w) for w in LADDER]
            n_candidates.append(len(seqs[0]))
            last = None
            for i in range(len(LADDER) - 1):
                if seqs[i] != seqs[i + 1]:
                    last = LADDER[i]
            if last is None:
                frozen_everywhere += 1
            else:
                lows.append(last)

        if n_candidates:
            ordered = sorted(n_candidates)
            print(
                f"candidates per query: min={ordered[0]} "
                f"median={ordered[len(ordered) // 2]} max={ordered[-1]}"
            )
        print(
            f"queries whose ranking never moves on the ladder: "
            f"{frozen_everywhere}/{len(queries)}"
        )
        if lows:
            lows.sort()
            print(f"freeze-weight bracket, lower bound over {len(lows)}:")
            for pct in (0, 25, 50, 75, 100):
                idx = min(len(lows) - 1, (pct * len(lows)) // 100)
                print(f"  p{pct:<4} {lows[idx]}")
            still = sum(1 for lo in lows if lo >= 1.5)
            print(f"queries still reordering at w >= 1.5: {still}/{len(lows)}")
        return 0
    finally:
        os.environ.pop("AELFRICE_POSTERIOR_WEIGHT", None)
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
