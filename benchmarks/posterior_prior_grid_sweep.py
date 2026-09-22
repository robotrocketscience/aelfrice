"""Measure how many beliefs carry an updated posterior, across many stores (#1592).

#1592 reported that on one 19,875-belief store every belief sits on its
insertion prior, so `posterior_weight` reranks by `(belief_type, source)`
rather than by observed confidence. One store is not a population. This
producer answers AC1 by classifying every belief in every store it can
reach against the **complete set of insertion priors**, and reporting how
many beliefs sit off that set.

Why classification rather than a distinct-pair count: the pair count that
#1592 published cannot distinguish "seven priors" from "one prior plus six
updated posteriors". The classification can, because of an invariant the
tree already pins.

**The invariant.** Every belief-posterior write goes through
`store.bump_posterior`, and `feedback._bayesian_delta` only ever hands it
non-negative deltas: positive valence adds `|valence|` to alpha and nothing
to beta, negative valence the reverse. Zero valence is rejected at the
`apply_feedback` boundary, so every accepted event moves one coordinate
strictly up. A belief that has received any evidence therefore cannot sit on
an insertion prior. The store documents this at `query_wonder_gc_candidates`,
which relies on it to make its epsilon band exact, and it is pinned by
`test_bayesian_update_is_monotone_so_the_prior_band_is_exact`.

Note it is `_bayesian_delta`, not `meta_beliefs.apply_evidence`, that governs
here. The latter splits one observation as `alpha += e, beta += 1 - e`, so it
adds exactly 1.0 of mass per event — but it serves meta-beliefs, which live in
their own tables. Reading the belief grid through it would predict integer
mass steps that the belief path does not produce.

Therefore: **a belief is on its insertion prior iff `(alpha, beta)` equals
some reachable prior pair.** Anything else has been written to.

**Excess mass is not the same as moved confidence,** and the split matters
more than the headline count. `MemoryStore` dedupe collapses a group of
duplicate beliefs by *summing* their alphas and betas onto the canonical row
(`total_alpha = sum(...)`). For a group of `n` rows sharing one prior that
yields `n * (alpha, beta)` — off the grid, carrying `n` times the mass, and
with the posterior *mean exactly unchanged*. Retrieval scores the mean, so
those beliefs moved without learning anything. This producer reports
mean-preserving and mean-moving off-grid beliefs separately; only the second
group is evidence that the feedback loop ran.

Point this at real stores; it copies each one before opening it and opens
the copy `read_only=True`. Opening a store read-write runs schema DDL and
migrations, which is a write, so a live store must never be the argument to
`MemoryStore` here. The copy also stops a live writer moving under the
probe mid-run.

Usage:
    posterior_prior_grid_sweep.py [--root DIR]... [STORE]...
                                  [--limit N] [--dry-run]

    --dry-run  list the stores that would be probed, and the prior grid
               that would be classified against, without copying or
               opening anything.

Exits non-zero if any store fails to open, or if no store was found.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

#: Where stores live on a normal install: one per git work tree under
#: `.git/aelfrice/`, plus the dotdir's own and its per-project mirrors.
DEFAULT_ROOTS: tuple[str, ...] = ("~/projects", "~/.aelfrice", "~/.git")

STORE_NAME = "memory.db"

#: Float comparison tolerance. The deflated alphas are exact IEEE 754
#: products (`3.0 * 0.2 == 0.6000000000000001`), so an exact compare would
#: work for values written by the current code, but a tolerance keeps the
#: classification stable against a store written by an older build that
#: rounded differently. It is far tighter than the smallest possible
#: evidence step, so it cannot absorb a real update.
TOL = 1e-9


def insertion_priors() -> dict[tuple[float, float], str]:
    """Every `(alpha, beta)` a belief can be *created* at, with provenance.

    Derived by calling `get_source_adjusted_prior` over the real
    `TYPE_PRIORS` keys rather than by transcribing its arithmetic, so the
    grid cannot drift from the shipped function. The literal-prior insert
    sites are then added explicitly, each with the site that writes it.
    """
    from aelfrice.classification_core import (  # noqa: PLC0415
        TYPE_PRIORS,
        USER_SOURCE,
        get_source_adjusted_prior,
    )

    grid: dict[tuple[float, float], str] = {}
    for belief_type in TYPE_PRIORS:
        for source, tag in ((USER_SOURCE, "user"), ("agent", "non-user")):
            pair = get_source_adjusted_prior(belief_type, source)
            grid.setdefault(pair, f"{belief_type}/{tag}")

    # Sites that write a literal prior instead of going through
    # get_source_adjusted_prior. Each is a real insert path; keep this
    # list in step with them or a legitimate prior reads as an update.
    for pair, why in (
        ((9.0, 0.5), "derivation.py lock/remember path"),
        ((1.0, 1.0), "derivation.py git triple-extraction path"),
        ((0.6, 1.0), "llm_classifier.py structural-failure fallback route"),
        ((0.3, 1.0), "wonder ingest speculative default"),
    ):
        grid.setdefault(pair, why)
    return grid


def _matches_prior(
    pair: tuple[float, float], priors: dict[tuple[float, float], str],
) -> str | None:
    """Return the provenance of the prior `pair` sits on, else None."""
    alpha, beta = pair
    for (pa, pb), why in priors.items():
        if abs(alpha - pa) <= TOL and abs(beta - pb) <= TOL:
            return why
    return None


def _matches_prior_mean(
    pair: tuple[float, float], priors: dict[tuple[float, float], str],
) -> str | None:
    """Return the provenance of the prior whose *mean* `pair` shares.

    An off-grid pair on a prior's mean has gained mass without gaining
    information — the signature of dedupe summing `n` copies of one prior
    onto a canonical row. Retrieval blends `log(posterior_mean)`, so such
    a belief scores exactly as it did before it moved.
    """
    alpha, beta = pair
    total = alpha + beta
    if total <= 0.0:
        return None
    mean = alpha / total
    for (pa, pb), why in priors.items():
        if abs(mean - pa / (pa + pb)) <= TOL:
            return why
    return None


def discover(roots: list[Path], limit: int | None) -> list[Path]:
    """Find every store under `roots`, skipping worktree and vendor copies.

    Worktrees are excluded because they share the parent repository's
    store, which would double-count it; `node_modules` because a vendored
    fixture is not a real store.
    """
    found: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob(STORE_NAME)):
            parts = set(path.parts)
            if "node_modules" in parts or "worktrees" in parts:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            found.append(path)
    return found[:limit] if limit else found


def probe(
    path: Path, priors: dict[tuple[float, float], str],
) -> dict[str, Any]:
    """Copy `path` to a throwaway, classify its grid, and return the counts.

    Copies the `-wal` and `-shm` sidecars too. Without the write-ahead log
    the copy would silently miss every belief committed since the last
    checkpoint, which on a busy store is the most recent and most likely
    to have been updated — exactly the population this is counting.
    """
    from aelfrice.store import MemoryStore  # noqa: PLC0415

    with tempfile.TemporaryDirectory(prefix="prior-grid-") as tmp:
        copy = Path(tmp) / STORE_NAME
        shutil.copyfile(path, copy)
        for suffix in ("-wal", "-shm"):
            sidecar = path.with_name(path.name + suffix)
            if sidecar.exists():
                shutil.copyfile(sidecar, copy.with_name(copy.name + suffix))

        # read_only=True is load-bearing, not hygiene: a read-write open
        # runs schema DDL and migrations, which is a write.
        store = MemoryStore(str(copy), read_only=True)
        try:
            pairs = store.alpha_beta_pairs()
        finally:
            store.close()

    grid: dict[tuple[float, float], int] = {}
    for pair in pairs:
        grid[pair] = grid.get(pair, 0) + 1

    on_prior = 0
    off: dict[tuple[float, float], int] = {}
    mean_preserving = 0
    for pair, n in grid.items():
        if _matches_prior(pair, priors) is not None:
            on_prior += n
            continue
        off[pair] = n
        if _matches_prior_mean(pair, priors) is not None:
            mean_preserving += n
    off_total = sum(off.values())
    return {
        "store": path,
        "beliefs": len(pairs),
        "distinct": len(grid),
        "on_prior": on_prior,
        "off_prior": off_total,
        "mean_preserving": mean_preserving,
        "mean_moved": off_total - mean_preserving,
        "off_pairs": off,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stores", nargs="*", type=Path, help="explicit stores")
    parser.add_argument(
        "--root", action="append", default=None,
        help=f"directory to search for {STORE_NAME} (repeatable); "
             f"defaults to {', '.join(DEFAULT_ROOTS)}",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="list the stores and the prior grid without opening anything",
    )
    args = parser.parse_args(argv)

    priors = insertion_priors()

    if args.stores:
        stores = list(args.stores)
    else:
        roots = [
            Path(r).expanduser()
            for r in (args.root if args.root else DEFAULT_ROOTS)
        ]
        stores = discover(roots, args.limit)

    if not stores:
        print("error: no stores found", file=sys.stderr)
        return 2

    print(f"insertion priors: {len(priors)}")
    for (a, b), why in sorted(priors.items()):
        print(f"  ({a}, {b}) -> {a / (a + b):.4f}   {why}")
    print(f"stores: {len(stores)}")

    if args.dry_run:
        for path in stores:
            print(f"  {path}")
        return 0

    failures = 0
    total_beliefs = 0
    total_off = 0
    total_moved = 0
    rows: list[dict[str, Any]] = []
    for path in stores:
        try:
            row = probe(path, priors)
        except Exception as exc:  # noqa: BLE001 - one bad store must not
            # abort the sweep; the point is the population, and a store
            # that cannot be opened is reported rather than hidden.
            print(f"  FAILED {path}: {exc}", file=sys.stderr)
            failures += 1
            continue
        rows.append(row)
        total_beliefs += row["beliefs"]
        total_off += row["off_prior"]
        total_moved += row["mean_moved"]

    print()
    print(
        f"{'beliefs':>9} {'pairs':>6} {'off-grid':>9} {'mean-kept':>10} "
        f"{'mean-moved':>11}  store"
    )
    for row in sorted(rows, key=lambda r: -r["beliefs"]):
        print(
            f"{row['beliefs']:>9} {row['distinct']:>6} {row['off_prior']:>9} "
            f"{row['mean_preserving']:>10} {row['mean_moved']:>11}  "
            f"{row['store']}"
        )

    print()
    print(f"stores probed: {len(rows)}   failed: {failures}")
    print(f"beliefs: {total_beliefs}")
    print(f"beliefs off the insertion-prior grid: {total_off}")
    print(f"  of those, posterior mean unchanged (dedupe mass): "
          f"{total_off - total_moved}")
    print(f"  of those, posterior mean actually moved: {total_moved}")
    if total_beliefs:
        print(f"off-grid share:    {100.0 * total_off / total_beliefs:.4f}%")
        print(f"mean-moved share:  {100.0 * total_moved / total_beliefs:.4f}%")
    stores_with_moves = sum(1 for r in rows if r["mean_moved"])
    live = [r for r in rows if r["beliefs"]]
    print(
        f"stores with at least one moved posterior: "
        f"{stores_with_moves} of {len(live)} non-empty"
    )

    off_all: dict[tuple[float, float], int] = {}
    for row in rows:
        for pair, n in row["off_pairs"].items():
            off_all[pair] = off_all.get(pair, 0) + n
    if off_all:
        print()
        print("off-grid pairs. `=prior-mean` marks a pair that gained mass")
        print("without moving its mean, so it scores exactly as before —")
        print("the dedupe signature, not evidence. An unmarked pair is a")
        print("real move, or a prior this script does not know about:")
        for (a, b), n in sorted(off_all.items(), key=lambda kv: -kv[1]):
            kept = _matches_prior_mean((a, b), priors)
            tag = "  =prior-mean" if kept else ""
            print(
                f"  ({a}, {b}) -> {a / (a + b):.4f}  {n:>7}  "
                f"mass {a + b}{tag}"
            )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
