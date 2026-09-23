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

**What sitting on a prior does and does not mean.** `feedback._bayesian_delta`
hands `store.bump_posterior` only non-negative deltas — positive valence adds
`|valence|` to alpha, negative to beta — so an *applied* feedback event always
moves a belief off its prior. Reading that as "on the prior means no evidence
ever arrived" is wrong, and the error is load-bearing enough to name here:

- `apply_feedback(update_posterior=False)` writes the audit row and skips the
  bump entirely (`feedback.py`: "the posterior deliberately not moved — either
  `update_posterior=False` (#1086 exposure lane) or the #1168 lock floor").
  Since #1086 that is the default for retrieval exposure, so the commonest
  feedback event in the system leaves the posterior exactly where it was.
- `clamp_ghosts` writes `UPDATE beliefs SET alpha = ?` with a *lower* alpha,
  so a moved belief can be pushed back onto a prior value.

Measured: **6,220 beliefs across this host sit on an insertion prior while
carrying a non-zero-valence `feedback_history` row**, 283 of them on the
aelfrice store. So the honest reading is narrow:

    on the grid  =>  the posterior was never moved
    on the grid  =/=>  no evidence ever reached the belief

and `off_prior` is a **lower bound** on beliefs that were ever written to.
This producer therefore reports the feedback-event count alongside the grid
classification; the gap between them is the signal that arrived and was
discarded, which is the quantity #1592 is really about.

Note it is `_bayesian_delta`, not `meta_beliefs.apply_evidence`, that governs
here. The latter splits one observation as `alpha += e, beta += 1 - e`, so it
adds exactly 1.0 of mass per event — but it serves meta-beliefs, which live in
their own tables (`meta_belief_signal_posteriors`), and cannot reach `beliefs`.

**Excess mass is not the same as moved confidence,** and the split matters
more than the headline count. `MemoryStore` dedupe collapses a group of
duplicate beliefs by *summing* their alphas and betas onto the canonical row
(`total_alpha = sum(...)`). For a group of `n` rows sharing one prior that
yields `n * (alpha, beta)` — off the grid, carrying `n` times the mass, and
with the posterior *mean exactly unchanged*. Retrieval scores the mean, so
those beliefs moved without learning anything.

A duplicate group is not always uniform, though. `content_hash` is `sha256`
of the text alone, so beliefs with identical text but a different type or
source share a hash and get merged too — and *their* sum moves the mean
while still being no evidence at all (#1598). The three cases are reported
separately: a scalar multiple of one prior, an exact sum of two or more
different priors, and the remainder, which is the only group that is
evidence the loop ran.

Point this at real stores; it copies each one before opening it and opens
the copy `read_only=True`. Opening a store read-write runs schema DDL and
migrations, which is a write, so a live store must never be the argument to
`MemoryStore` here. The copy also stops a live writer moving under the
probe mid-run.

Usage:
    posterior_prior_grid_sweep.py [--root DIR]... [STORE]...
                                  [--limit N] [--dry-run] [--sources]

    --dry-run  list the stores that would be probed, and the prior grid
               that would be classified against, without copying or
               opening anything.
    --sources  also report the feedback-event source histogram (#1592
               AC2): which lanes actually fire, how many of their events
               carry a posterior-moving valence, and how many of those
               land on a user-locked belief where the #1168 lock floor
               refuses the bump regardless.
               `scripts/check_posterior_writers.py` enumerates the
               writers statically; this measures which of them run.

Stores that fail to open are counted and reported on stderr, not hidden:
a legacy schema a read-only handle cannot migrate is excluded from every
figure here, so read `stores probed` rather than assuming the whole host.

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

#: Merge date of 46301160, "fix(feedback): make retrieval-exposure
#: audit-only by default (#1086)". Before it, every retrieval exposure
#: added +0.1 to alpha; #1086 removed that as a defect, having measured
#: that it scored junk above clean content. A posterior whose earliest
#: real feedback event predates this therefore moved under a rule the
#: tree has since deleted, and reading it as evidence that the feedback
#: loop works today is the confound this figure exists to expose.
EXPOSURE_AUDIT_ONLY_SINCE = "2026-07-04"


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
    #
    # This list cannot be complete by construction, and that is a known
    # limit rather than an oversight: `derivation.py` writes `alpha`/`beta`
    # verbatim from `raw_meta` route-overrides, and `migrate.py` copies
    # alpha verbatim from a legacy store. Neither is bounded, so a store
    # carrying either will over-report moved posteriors. `off_pairs` exists
    # so such a pair is visible rather than silently counted as evidence.
    for pair, why in (
        ((9.0, 0.5), "derivation.py lock/remember path"),
        ((1.0, 1.0), "derivation.py git triple-extraction path"),
        ((0.6, 1.0), "llm_classifier.py regex-route alpha (regex.alpha)"),
        ((0.3, 1.0), "wonder/lifecycle.py _INGEST_ALPHA/_INGEST_BETA"),
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


def _matches_prior_multiple(
    pair: tuple[float, float], priors: dict[tuple[float, float], str],
) -> str | None:
    """Return the provenance of the prior `pair` is a scalar multiple of.

    Dedupe collapses a group by summing, so `n` copies of one prior land
    at `n * (alpha, beta)` — off the grid, `n` times the mass, mean
    unchanged. Retrieval blends `log(posterior_mean)`, so such a belief
    scores exactly as it did before it moved.

    The test is **structural**, not a mean comparison, and that matters.
    Means collide across priors: `(3.6000000000000005, 1.0)` is a belief
    that started at the factual non-user prior `(0.6000000000000001, 1.0)`
    and gained 3.0 of alpha with beta untouched — a real move — yet its
    mean 0.7826 is exactly `(1.8, 0.5)`'s. Classifying by mean calls that
    dedupe mass and hides the move. Requiring `alpha/pa == beta/pb`
    rejects it, because beta did not scale with alpha. Fifty-six beliefs
    on this host are in that class.
    """
    alpha, beta = pair
    for (pa, pb), why in priors.items():
        if pa <= 0.0 or pb <= 0.0:
            continue
        n = round(alpha / pa)
        if n < 2:
            continue  # n == 1 is the prior itself, already on-grid
        # Exact float reproduction, not a tolerance. A dedupe group has an
        # integer size, so the canonical row holds precisely `n * pa`
        # as IEEE 754 computed it — and that residue is the only thing
        # separating two readings of the same number. `(3.6000000000000005,
        # 1.0)` is `6 * 0.6000000000000001` exactly, but `2 * 1.8` is
        # `3.6` exactly, a different float. A tolerance wide enough to
        # call the second a match silently relabels a real alpha-only
        # move as dedupe mass; requiring the exact product does not.
        if n * pa == alpha and n * pb == beta:
            return why
    return None


#: Largest duplicate group this producer will recognise as a merge. The
#: search is a breadth-first expansion over reachable sums, so the cost is
#: the number of *distinct* sums (about 103k here, built in well under a
#: second) rather than the number of combinations. A merge of more than
#: this many beliefs is classified as a moved posterior and will show up
#: in `off_pairs`; say so rather than letting the bound go unstated.
MAX_MERGE_GROUP = 16

_prior_sum_cache: dict[tuple[float, float], int] | None = None


def _prior_sums(
    priors: dict[tuple[float, float], str],
) -> dict[tuple[float, float], int]:
    """Every `(α, β)` reachable by summing 2..MAX_MERGE_GROUP priors.

    Maps each reachable sum to the smallest group size that produces it.

    **Exact arithmetic, no tolerance.** Summation order does not matter
    for this prior set: over 120,000 random permutations of 16-term sums
    the spread is exactly 0.0, pinned by
    `test_summing_priors_is_order_independent`. That is a property of
    these particular values, not of floating point in general, which is
    why it is measured and pinned rather than assumed — if `TYPE_PRIORS`
    changes and the property lapses, that test reds and this function
    needs a tolerance with a stated justification.
    """
    global _prior_sum_cache  # noqa: PLW0603 - one-shot build, never varies
    if _prior_sum_cache is not None:
        return _prior_sum_cache
    base = list(priors)
    level: set[tuple[float, float]] = set(base)
    out: dict[tuple[float, float], int] = {}
    for size in range(2, MAX_MERGE_GROUP + 1):
        nxt: set[tuple[float, float]] = set()
        for a, b in level:
            for pa, pb in base:
                nxt.add((a + pa, b + pb))
        level = nxt
        for pair in nxt:
            out.setdefault(pair, size)
    _prior_sum_cache = out
    return out


def _matches_prior_sum(
    pair: tuple[float, float], priors: dict[tuple[float, float], str],
) -> int | None:
    """Group size if `pair` is an exact sum of two or more priors, else None.

    `_matches_prior_multiple` only recognises `n` copies of the *same*
    prior, which assumes a duplicate group is uniform. It often is not:
    `content_hash` is `sha256(text)` alone, so beliefs with identical
    text but different type or source share a hash and get merged. Their
    sum is neither on the grid nor a multiple of one prior, and without
    this test it reads as evidence (#1598).

    A mixed merge differs from a uniform one in a way that matters for
    ranking: the posterior *mean* moves, so it is not rank-neutral. It is
    still not evidence — no feedback event occurred — so it is reported
    as its own category rather than folded into either neighbour.
    """
    return _prior_sums(priors).get(pair)


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


def _feedback_split(
    store: Any, priors: dict[tuple[float, float], str],
) -> tuple[int, int]:
    """Return (stranded, legacy_moved) from the feedback history.

    `stranded` counts beliefs on a prior that nonetheless carry a real
    feedback event — see `_stranded_on_prior` for why that matters.

    `legacy_moved` counts **off-grid** beliefs whose earliest non-zero
    feedback event predates `EXPOSURE_AUDIT_ONLY_SINCE`. Those moved
    under the pre-#1086 rule that added +0.1 per retrieval exposure,
    which the tree removed as a defect. They are residue, not evidence
    that today's feedback path works, and counting them as the latter is
    the confound that makes a store look healthy when it is not.

    Returns `(0, 0)` on a store with no `feedback_history` table rather
    than failing, so an old schema costs the extra columns and not the
    row.
    """
    conn = store._conn  # noqa: SLF001 - no public accessor for this join
    has_table = conn.execute(
        "SELECT count(*) FROM sqlite_master "
        "WHERE type='table' AND name='feedback_history'"
    ).fetchone()
    if not has_table or not has_table[0]:
        return (0, 0)
    stranded = 0
    legacy = 0
    # One row per belief, not per event: a belief with forty exposure
    # events is one belief, not forty.
    for row in conn.execute(
        "SELECT b.id, b.alpha, b.beta, "
        "       (SELECT min(f.created_at) FROM feedback_history f "
        "        WHERE f.belief_id = b.id AND f.valence != 0) AS first_ev "
        "FROM beliefs b"
    ):
        first = row["first_ev"]
        if first is None:
            continue
        pair = (float(row["alpha"]), float(row["beta"]))
        if _matches_prior(pair, priors) is not None:
            stranded += 1
            continue
        # `legacy` is reported as a share of the genuinely-moved
        # population, so it has to be counted over that same population.
        # Counting it over all off-grid beliefs mixes in dedupe mass and
        # can print a share above 100%.
        if _matches_prior_multiple(pair, priors) is not None:
            continue
        if _matches_prior_sum(pair, priors) is not None:
            continue
        if str(first) < EXPOSURE_AUDIT_ONLY_SINCE:
            legacy += 1
    return (stranded, legacy)


def _source_histogram(store: Any) -> dict[str, tuple[int, int, int]]:
    """Per-source `(events, movable, blocked-by-lock)` from feedback_history.

    #1592 AC2 asks not which writers *can* fire but which *do*. Every
    feedback event writes an audit row naming its source, whether or not
    the posterior moved. `movable` counts events whose valence is
    non-zero, which is necessary for a move and not sufficient, and
    `blocked` counts the subset landing on a user-locked belief, which
    the #1168 lock floor refuses at any setting of the exposure flag
    (`posterior_applied = update_posterior and not locked`). So even
    `movable - blocked` is an upper bound, since the flag can still be
    off. Pair it with the stranded count to tell a lane that moved
    posteriors from one that fired and was discarded.

    Lock state is read as it is now, not as it was at event time, so
    `blocked` is an estimate for events on beliefs locked since.
    """
    conn = store._conn  # noqa: SLF001 - no public accessor for this rollup
    present = conn.execute(
        "SELECT count(*) FROM sqlite_master "
        "WHERE type='table' AND name='feedback_history'"
    ).fetchone()
    if not present or not present[0]:
        return {}
    out: dict[str, tuple[int, int, int]] = {}
    for row in conn.execute(
        "SELECT f.source AS source, count(*) AS n, "
        "       sum(CASE WHEN f.valence != 0 THEN 1 ELSE 0 END) AS movable, "
        "       sum(CASE WHEN f.valence != 0 "
        "                 AND b.lock_level = 'user' THEN 1 ELSE 0 END) "
        "           AS locked "
        "FROM feedback_history f "
        "LEFT JOIN beliefs b ON b.id = f.belief_id "
        "GROUP BY f.source"
    ):
        out[str(row["source"])] = (
            int(row["n"]),
            int(row["movable"] or 0),
            int(row["locked"] or 0),
        )
    return out


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
            stranded, legacy = _feedback_split(store, priors)
            sources = _source_histogram(store)
        finally:
            store.close()

    grid: dict[tuple[float, float], int] = {}
    for pair in pairs:
        grid[pair] = grid.get(pair, 0) + 1

    on_prior = 0
    off: dict[tuple[float, float], int] = {}
    mean_preserving = 0
    mixed_merge = 0
    for pair, n in grid.items():
        if _matches_prior(pair, priors) is not None:
            on_prior += n
            continue
        off[pair] = n
        if _matches_prior_multiple(pair, priors) is not None:
            # n copies of ONE prior: mass scales, mean unchanged.
            mean_preserving += n
        elif _matches_prior_sum(pair, priors) is not None:
            # A merge of DIFFERENT priors. The mean moves, so unlike the
            # uniform case it is not rank-neutral — but no feedback event
            # occurred, so it is not evidence either (#1598).
            mixed_merge += n
    off_total = sum(off.values())
    return {
        "store": path,
        "beliefs": len(pairs),
        "distinct": len(grid),
        "on_prior": on_prior,
        "off_prior": off_total,
        "mean_preserving": mean_preserving,
        "mixed_merge": mixed_merge,
        "mean_moved": off_total - mean_preserving - mixed_merge,
        "stranded": stranded,
        "legacy_moved": legacy,
        "sources": sources,
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
        "--sources", action="store_true",
        help="also report the feedback-event source histogram (#1592 AC2): "
             "which writers actually fire, and how often they move a "
             "posterior rather than only auditing",
    )
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
    total_uniform = 0
    total_mixed = 0
    total_stranded = 0
    total_legacy = 0
    all_sources: dict[str, tuple[int, int, int]] = {}
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
        total_uniform += row["mean_preserving"]
        total_mixed += row["mixed_merge"]
        total_stranded += row["stranded"]
        total_legacy += row["legacy_moved"]
        for src, (n, movable, blocked) in row["sources"].items():
            have = all_sources.get(src, (0, 0, 0))
            all_sources[src] = (
                have[0] + n, have[1] + movable, have[2] + blocked,
            )

    print()
    print(
        f"{'beliefs':>9} {'pairs':>6} {'off-grid':>9} {'dedupe':>8} "
        f"{'mixed':>7} {'moved':>7} {'stranded':>9}  store"
    )
    for row in sorted(rows, key=lambda r: -r["beliefs"]):
        print(
            f"{row['beliefs']:>9} {row['distinct']:>6} {row['off_prior']:>9} "
            f"{row['mean_preserving']:>8} {row['mixed_merge']:>7} "
            f"{row['mean_moved']:>7} {row['stranded']:>9}  {row['store']}"
        )

    print()
    print(f"stores probed: {len(rows)}   failed: {failures}")
    print(f"beliefs: {total_beliefs}")
    print(f"beliefs off the insertion-prior grid: {total_off}")
    print(f"  of those, a scalar multiple of one prior (uniform dedupe, "
          f"mean unchanged): {total_uniform}")
    print(f"  of those, a sum of different priors (mixed dedupe, mean "
          f"moved but not evidence): {total_mixed}")
    print(f"  of those, genuinely moved: {total_moved}")
    print(f"beliefs ON a prior that carry a real feedback event: "
          f"{total_stranded}")
    print("  (signal arrived and was discarded — the audit-only lane)")
    print(f"off-grid beliefs whose earliest event predates "
          f"{EXPOSURE_AUDIT_ONLY_SINCE}: {total_legacy}")
    print("  (moved under the pre-#1086 exposure rule the tree deleted)")
    if total_moved:
        print(f"  share of genuinely-moved: "
              f"{100.0 * total_legacy / total_moved:.1f}%")
    if total_beliefs:
        print(f"off-grid share:    {100.0 * total_off / total_beliefs:.4f}%")
        print(f"mean-moved share:  {100.0 * total_moved / total_beliefs:.4f}%")
    stores_with_moves = sum(1 for r in rows if r["mean_moved"])
    live = [r for r in rows if r["beliefs"]]
    print(
        f"stores with at least one moved posterior: "
        f"{stores_with_moves} of {len(live)} non-empty"
    )

    if args.sources and all_sources:
        print()
        print("feedback events by source. `movable` counts non-zero")
        print("valence — necessary for the posterior to move, not")
        print("sufficient: apply_feedback still drops the bump when")
        print("update_posterior=False, which since #1086 is the default")
        print("for retrieval exposure. So a large movable count is not")
        print("evidence that anything moved.")
        print("`blocked` is the subset of movable landing on a")
        print("user-locked belief, which the #1168 lock floor refuses")
        print("regardless of the flag.")
        print(f"  {'events':>9} {'movable':>9} {'blocked':>9}  source")
        for src, (n, movable, blocked) in sorted(
            all_sources.items(), key=lambda kv: -kv[1][0]
        ):
            print(f"  {n:>9} {movable:>9} {blocked:>9}  {src}")

    off_all: dict[tuple[float, float], int] = {}
    for row in rows:
        for pair, n in row["off_pairs"].items():
            off_all[pair] = off_all.get(pair, 0) + n
    if off_all:
        print()
        print("off-grid pairs. `=n x prior` marks an exact scalar multiple")
        print("of a prior: mass gained with the mean unchanged, so it scores")
        print("exactly as before — the dedupe signature, not evidence. An")
        print("unmarked pair is a real move, or a prior not in the grid:")
        for (a, b), n in sorted(off_all.items(), key=lambda kv: -kv[1]):
            kept = _matches_prior_multiple((a, b), priors)
            tag = "  =n x prior" if kept else ""
            print(
                f"  ({a}, {b}) -> {a / (a + b):.4f}  {n:>7}  "
                f"mass {a + b}{tag}"
            )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
