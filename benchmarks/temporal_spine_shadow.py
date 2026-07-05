"""G2 shadow eval — temporal-spine default-flip gate (#1064).

The G2 flip-gate criterion has two halves. The **bench-pool half** (coverage
delta + top-rank invariance at the production 1500-token budget on a labelled
corpus) is measured by ``temporal_spine_ablation.py`` on LoCoMo. This module
is the **shadow-eval half**: the same top-rank-invariance check plus the
lane's operational aggregates, run on a *real hook-ingested backfilled store*
driven by *real turn queries*, at the production operating point.

Why a separate harness. A labelled benchmark corpus (LoCoMo) has clean, short,
topically-coherent per-session chains and no user locks. A real store does
not: hook-ingested beliefs chain within long-running host sessions, so the
chain-length distribution is heavy-tailed and heterogeneous (open question 1
in the #1064 landing PR), and there is a real L0 locked pool. The shadow eval
answers two questions the bench pool structurally cannot:

  1. **Chain-length distribution under production ``session_id`` semantics.**
     What do the chains the writer/backfill actually build on a real store
     look like? (open question 1)
  2. **Does top-rank invariance survive on production-shaped data?** With a
     real locked pool, a heavy-tailed chain distribution, and the retrieval
     reranking stack (compression / clustering) resolved to its real
     production settings — none of which LoCoMo exercises — does turning the
     lane on still never displace or reorder a ``[locked, l25, l1, hrr]``
     core belief at budget 1500?

Coverage-vs-gold is **not** measured here: a real store has no gold evidence
labels, so per-question coverage is undefined. The ``≥ +3pp coverage``
criterion is the bench-pool half's job (DONE on LoCoMo); this half is
top-rank invariance + operational aggregates only, matching the #1064 gate
text ("a shadow eval on a real backfilled store (aggregate-only)").

Aggregate-only, by construction
-------------------------------

This harness reads a real belief store and a real turn log — both private.
It emits **only aggregate statistics**: counts, rates, percentiles, and a
chain-length histogram. It never prints, writes, or returns belief text,
query text, belief ids, or session ids. The JSON report and the stderr
summary are safe to paste into a public PR; the store and turn log are not,
and never leave this process.

Not a null
----------

Top-rank invariance is only meaningful if the lane actually fired. A store
whose queries never reach a chained belief would make the on-arm a no-op and
the invariant hold vacuously (the #981 trivial-null trap). ``main`` refuses
to report unless the lane fired on at least one query.

Usage
-----

    # Against the maintainer's real store + turn log, backfilling the spine
    # onto a scratch copy first (never mutate the live store — copy the DB
    # that db_path() resolves to for the project, e.g.
    # <git-common-dir>/aelfrice/memory.db, to /tmp/shadow.db):
    uv run python -m benchmarks.temporal_spine_shadow \\
        --db /tmp/shadow.db \\
        --turns <git-common-dir>/aelfrice/transcripts/turns.jsonl \\
        --backfill --out /tmp/temporal_spine_shadow.json

    # smoke against any backfilled store:
    uv run python -m benchmarks.temporal_spine_shadow --db /tmp/shadow.db \\
        --turns /tmp/turns.jsonl --subset 10
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass, field
from typing import Final, Sequence

from aelfrice.models import EDGE_TEMPORAL_NEXT
from aelfrice.retrieval import last_lane_telemetry, retrieve_v2
from aelfrice.store import MemoryStore
from aelfrice.temporal_spine import backfill_temporal_spine

# The rank-invariance accumulator + LCP helper are the exact primitives the
# bench-pool half uses; reuse them so the two G2 halves report the same
# top-rank-invariance metric under the same definition.
from benchmarks.temporal_spine_ablation import RankInvarianceAccumulator

HARNESS_NAME: Final[str] = "temporal_spine_shadow"

# Production operating point (the #1064 G2 question). Overridable for
# experimentation, but these are the values the flip gate is defined at.
DEFAULT_BUDGET: Final[int] = 1500
DEFAULT_L1_LIMIT: Final[int] = 50

# Histogram bucket upper bounds (exclusive) + labels for the chain-length
# distribution. Chosen to separate the singleton/short mass from the
# heavy tail that production host-session chains produce.
_HIST_BUCKETS: Final[tuple[tuple[str, int, int], ...]] = (
    ("1", 1, 1),
    ("2", 2, 2),
    ("3-4", 3, 4),
    ("5-9", 5, 9),
    ("10-24", 10, 24),
    ("25-49", 25, 49),
    ("50-99", 50, 99),
    ("100-249", 100, 249),
    ("250-499", 250, 499),
    ("500-999", 500, 999),
    ("1000+", 1000, 2**62),
)


@dataclass
class ChainLengthSummary:
    """Aggregate shape of the per-session TEMPORAL_NEXT chains.

    ``sizes`` are per-session belief counts (chain node counts) for
    sessions with a non-null ``session_id``; a chain of length ``n``
    carries ``n - 1`` TEMPORAL_NEXT edges, and singletons (``n == 1``)
    carry none. ``no_session_beliefs`` are beliefs the writer never
    chains (null ``session_id``). All fields are counts / percentiles —
    no belief content.
    """

    n_sessions: int = 0
    n_beliefs_in_chains: int = 0
    no_session_beliefs: int = 0
    singletons: int = 0
    min_len: int = 0
    max_len: int = 0
    mean_len: float = 0.0
    median_len: float = 0.0
    p25: float = 0.0
    p50: float = 0.0
    p75: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    histogram: dict[str, int] = field(default_factory=dict)
    histogram_belief_share: dict[str, float] = field(default_factory=dict)


def _percentile(sorted_sizes: list[int], p: float) -> float:
    """Linear-interpolated percentile of an already-sorted list."""
    if not sorted_sizes:
        return 0.0
    k = (len(sorted_sizes) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(sorted_sizes) - 1)
    return sorted_sizes[lo] + (sorted_sizes[hi] - sorted_sizes[lo]) * (k - lo)


def summarize_chain_lengths(
    sizes: Sequence[int], *, no_session_beliefs: int = 0,
) -> ChainLengthSummary:
    """Pure aggregate of a list of per-session chain lengths.

    Split out from any store access so it is unit-testable with a
    synthetic list. ``sizes`` need not be sorted.
    """
    s = sorted(int(x) for x in sizes)
    summary = ChainLengthSummary(no_session_beliefs=no_session_beliefs)
    if not s:
        return summary
    total_b = sum(s)
    summary.n_sessions = len(s)
    summary.n_beliefs_in_chains = total_b
    summary.singletons = sum(1 for x in s if x == 1)
    summary.min_len = s[0]
    summary.max_len = s[-1]
    summary.mean_len = statistics.mean(s)
    summary.median_len = statistics.median(s)
    summary.p25 = _percentile(s, 0.25)
    summary.p50 = _percentile(s, 0.50)
    summary.p75 = _percentile(s, 0.75)
    summary.p90 = _percentile(s, 0.90)
    summary.p95 = _percentile(s, 0.95)
    summary.p99 = _percentile(s, 0.99)
    hist: dict[str, int] = {}
    belief_share: dict[str, float] = {}
    for label, lo, hi in _HIST_BUCKETS:
        in_bucket = [x for x in s if lo <= x <= hi]
        hist[label] = len(in_bucket)
        belief_share[label] = (
            sum(in_bucket) / total_b if total_b else 0.0
        )
    summary.histogram = hist
    summary.histogram_belief_share = belief_share
    return summary


def chain_lengths_from_store(store: MemoryStore) -> tuple[list[int], int]:
    """Return (per-session chain lengths, count of null-session beliefs).

    A chain length is the number of beliefs in a session — equal to the
    node count of that session's TEMPORAL_NEXT chain. Reads counts only.
    """
    conn = store._conn  # noqa: SLF001
    sizes = [
        row[0]
        for row in conn.execute(
            "SELECT COUNT(*) FROM beliefs "
            "WHERE session_id IS NOT NULL GROUP BY session_id"
        ).fetchall()
    ]
    no_session = conn.execute(
        "SELECT COUNT(*) FROM beliefs WHERE session_id IS NULL"
    ).fetchone()[0]
    return sizes, no_session


def load_user_queries(turns_path: str) -> list[str]:
    """Real user-role queries from a turn log, order-preserving dedup.

    The turn log is a private artifact; only the query *strings* are
    read, and they are used solely to drive retrieval — never emitted.
    """
    seen: set[str] = set()
    out: list[str] = []
    with open(turns_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                turn = json.loads(line)
            except json.JSONDecodeError:
                continue
            if turn.get("role") != "user":
                continue
            text = (turn.get("text") or "").strip()
            if text and text not in seen:
                seen.add(text)
                out.append(text)
    return out


@dataclass
class LaneAggregates:
    """Operational aggregates of the lane on the shadow query set."""

    n_queries: int = 0
    lane_fired: int = 0            # queries with ≥1 spine candidate
    spine_candidates: int = 0      # pre-trim discovered nodes, summed
    spine_survivors: int = 0       # post-budget-trim packed hits, summed

    @property
    def fire_rate(self) -> float:
        return self.lane_fired / self.n_queries if self.n_queries else 0.0

    @property
    def trim_loss(self) -> int:
        return self.spine_candidates - self.spine_survivors

    @property
    def trim_rate(self) -> float:
        return (
            self.trim_loss / self.spine_candidates
            if self.spine_candidates else 0.0
        )

    @property
    def mean_survivors_per_query(self) -> float:
        return (
            self.spine_survivors / self.n_queries if self.n_queries else 0.0
        )


def run_shadow(
    store: MemoryStore,
    queries: Sequence[str],
    *,
    budget: int,
    l1_limit: int,
    include_locked: bool = True,
) -> tuple[RankInvarianceAccumulator, LaneAggregates]:
    """Paired lane-off/on retrieval over real queries on a real store.

    Ingest is arm-independent (the flag only changes retrieval); the only
    variable is ``use_temporal_spine``. The core boundary is read from
    ``last_lane_telemetry()`` — ``locked + l25 + l1 + hrr_expand`` — but
    **the locked count is dropped when ``include_locked`` is False**,
    because those beliefs are filtered out of the returned list in that
    mode. (This differs from the LoCoMo rank-invariance runner, which can
    add the locked count unconditionally only because LoCoMo has zero
    locks; on a real store with an L0 pool the distinction is load-bearing
    — getting it wrong slides the comparison window into the by-design
    tail and manufactures a false regression.)
    """
    acc = RankInvarianceAccumulator()
    lane = LaneAggregates()
    for query in queries:
        base = retrieve_v2(
            store, query, budget=budget, l1_limit=l1_limit,
            include_locked=include_locked, use_temporal_spine=False,
        )
        base_ids = [b.id for b in base.beliefs]
        tel_b = last_lane_telemetry()
        core_b = (
            (tel_b.locked if include_locked else 0)
            + tel_b.l25 + tel_b.l1 + tel_b.hrr_expand
        )
        spine = retrieve_v2(
            store, query, budget=budget, l1_limit=l1_limit,
            include_locked=include_locked, use_temporal_spine=True,
        )
        spine_ids = [b.id for b in spine.beliefs]
        tel_s = last_lane_telemetry()
        core_s = (
            (tel_s.locked if include_locked else 0)
            + tel_s.l25 + tel_s.l1 + tel_s.hrr_expand
        )
        acc.add(base_ids, spine_ids, core_b, core_s, tel_s.temporal_spine)
        lane.n_queries += 1
        if tel_s.temporal_spine_candidates > 0:
            lane.lane_fired += 1
        lane.spine_candidates += tel_s.temporal_spine_candidates
        lane.spine_survivors += tel_s.temporal_spine
    return acc, lane


def build_report(
    chains: ChainLengthSummary,
    acc: RankInvarianceAccumulator,
    lane: LaneAggregates,
    *,
    budget: int,
    l1_limit: int,
    include_locked: bool,
) -> dict:
    """Assemble the aggregate-only JSON report."""
    return {
        "harness": HARNESS_NAME,
        "gate": "G2 shadow-eval half (#1064)",
        "operating_point": {
            "budget": budget,
            "l1_limit": l1_limit,
            "include_locked": include_locked,
        },
        "chain_length_distribution": asdict(chains),
        "lane_aggregates": {
            "n_queries": lane.n_queries,
            "lane_fired": lane.lane_fired,
            "fire_rate": lane.fire_rate,
            "spine_candidates": lane.spine_candidates,
            "spine_survivors": lane.spine_survivors,
            "trim_loss": lane.trim_loss,
            "trim_rate": lane.trim_rate,
            "mean_survivors_per_query": lane.mean_survivors_per_query,
        },
        "top_rank_invariance": {
            "n_questions": acc.n_questions,
            "head_invariant": acc.head_invariant,
            "head_invariant_rate": acc.head_invariant_rate(),
            "top_rank_displacements": acc.top_rank_displacements,
            "core_mismatch": acc.core_mismatch,
            "mean_lcp": acc.mean_lcp(),
            "min_lcp": acc.min_lcp,
            "tail_eviction_questions": acc.tail_eviction_questions,
            "tail_evicted_total": acc.tail_evicted_total,
            "spine_contributed_questions": acc.spine_contributed_questions,
            "spine_added_sum": acc.spine_added_sum,
            "passed": acc.passed(),
        },
    }


def _print_summary(report: dict) -> None:
    """Human-readable stderr summary — aggregate figures only."""
    c = report["chain_length_distribution"]
    la = report["lane_aggregates"]
    ti = report["top_rank_invariance"]
    op = report["operating_point"]
    p = lambda *a: print(*a, file=sys.stderr)  # noqa: E731
    p(f"[{HARNESS_NAME}] G2 shadow eval — budget={op['budget']} "
      f"l1_limit={op['l1_limit']} include_locked={op['include_locked']}")
    p("  chain-length distribution (open question 1):")
    p(f"    sessions/chains: {c['n_sessions']}   "
      f"beliefs in chains: {c['n_beliefs_in_chains']}   "
      f"null-session beliefs: {c['no_session_beliefs']}")
    p(f"    min/median/mean/max: {c['min_len']} / {c['median_len']:.0f} / "
      f"{c['mean_len']:.1f} / {c['max_len']}   "
      f"p90/p95/p99: {c['p90']:.0f}/{c['p95']:.0f}/{c['p99']:.0f}")
    p(f"    singletons (no edge): {c['singletons']}")
    p("    histogram (sessions | % of chained beliefs):")
    for label, _lo, _hi in _HIST_BUCKETS:
        n = c["histogram"].get(label, 0)
        share = c["histogram_belief_share"].get(label, 0.0)
        if n:
            p(f"      {label:>8}: {n:>5} sessions   "
              f"{100 * share:5.1f}% of beliefs")
    p("  lane aggregates:")
    p(f"    fired: {la['lane_fired']}/{la['n_queries']} "
      f"({100 * la['fire_rate']:.0f}%)   "
      f"candidates={la['spine_candidates']} survivors={la['spine_survivors']} "
      f"trim={la['trim_loss']} ({100 * la['trim_rate']:.0f}%)   "
      f"mean survivors/query={la['mean_survivors_per_query']:.2f}")
    p("  top-rank invariance (G2 pass criterion):")
    p(f"    core-prefix invariant: {ti['head_invariant']}/{ti['n_questions']} "
      f"({100 * ti['head_invariant_rate']:.1f}%)   "
      f"displacements={ti['top_rank_displacements']}   "
      f"core_mismatch={ti['core_mismatch']}")
    p(f"    >>> G2 top-rank invariance PASSED: {ti['passed']}")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="#1064 G2 temporal-spine shadow eval")
    ap.add_argument("--db", required=True, help="path to a backfilled store COPY")
    ap.add_argument("--turns", required=True, help="turn-log jsonl (user queries)")
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    ap.add_argument("--l1-limit", type=int, default=DEFAULT_L1_LIMIT)
    ap.add_argument(
        "--backfill", action="store_true",
        help="backfill the TEMPORAL_NEXT spine onto the store before eval "
             "(idempotent; run on a COPY, never the live store)",
    )
    ap.add_argument(
        "--no-locked", action="store_true",
        help="exclude the L0 locked pool from retrieval (default: include, "
             "matching the production hook)",
    )
    ap.add_argument("--subset", type=int, default=None,
                    help="cap the query set (smoke)")
    ap.add_argument("--out", default="/tmp/temporal_spine_shadow.json")
    args = ap.parse_args(argv)

    store = MemoryStore(args.db)
    if args.backfill:
        report = backfill_temporal_spine(store)
        print(f"[{HARNESS_NAME}] backfill: {report}", file=sys.stderr)
    if not store.has_edge_type(EDGE_TEMPORAL_NEXT):
        print(
            f"[{HARNESS_NAME}] ERROR: store has no TEMPORAL_NEXT spine — "
            "pass --backfill (on a copy) or point --db at a backfilled store.",
            file=sys.stderr,
        )
        return 2

    queries = load_user_queries(args.turns)
    if args.subset:
        queries = queries[: args.subset]
    if not queries:
        print(f"[{HARNESS_NAME}] ERROR: no user queries in {args.turns}",
              file=sys.stderr)
        return 2

    sizes, no_session = chain_lengths_from_store(store)
    chains = summarize_chain_lengths(sizes, no_session_beliefs=no_session)
    acc, lane = run_shadow(
        store, queries,
        budget=args.budget, l1_limit=args.l1_limit,
        include_locked=not args.no_locked,
    )

    # Not-a-null guard: an all-no-op on-arm makes invariance vacuous.
    if lane.lane_fired == 0:
        print(
            f"[{HARNESS_NAME}] ERROR: lane never fired across "
            f"{lane.n_queries} queries — invariance would hold vacuously "
            "(#981 trivial-null). Point --db at a store whose chains the "
            "query set actually reaches.",
            file=sys.stderr,
        )
        return 3

    report = build_report(
        chains, acc, lane,
        budget=args.budget, l1_limit=args.l1_limit,
        include_locked=not args.no_locked,
    )
    from pathlib import Path
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    _print_summary(report)
    print(f"[{HARNESS_NAME}] wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
