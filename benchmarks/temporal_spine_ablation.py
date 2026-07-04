"""#1064 temporal-spine lane ablation — gold-set evidence coverage on LoCoMo.

Runs LoCoMo under three retrieval configurations to isolate the
``use_temporal_spine`` lane:

    baseline         — spine lane off (production-style default)
    +spine           — spine lane on, real chronological chains
    shuffled-control — spine lane on, identical edge count but endpoints
                       deterministically permuted (chronology destroyed)

Scoring is **gold-set coverage**: ``|gold ∩ retrieved| / |gold|`` per
question, where the gold set is the belief ids derived from the QA pair's
evidence dialogue turns. This is the lens the #1064 campaign pre-registered —
single-hit recall stays saturated and misses the aggregation/temporal
questions whose gold is diffuse. Also reported: the all-evidence rate
(fraction of questions whose gold set is fully covered). No LLM reader is
involved; the run is deterministic end-to-end.

The shuffled control isolates the cause: identical density with scrambled
endpoints recovering ~nothing means the value is the chronology, not the
extra connectivity. The permutation is seeded (``--shuffle-seed``) so the
control is reproducible.

Ingest is arm-independent (the flag only changes retrieval), so each
conversation is ingested once and the spine backfilled once; the shuffled
arm rewrites the TEMPORAL_NEXT edge set in place and restores it after.

Usage:
    uv run python -m benchmarks.temporal_spine_ablation \\
        --data /tmp/LoCoMo/data/locomo10.json \\
        --out /tmp/temporal_spine_ablation.json
    # smoke: --subset-convs 1 --subset-qa 20
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

from aelfrice.ingest import _ingest_turn_ids
from aelfrice.models import EDGE_TEMPORAL_NEXT
from aelfrice.retrieval import retrieve_v2
from aelfrice.store import MemoryStore
from aelfrice.temporal_spine import backfill_temporal_spine

# The LoCoMo dataset stack (``locomo_adapter`` → ``nltk``) is imported
# lazily inside the functions that actually run the bench, so this module's
# pure logic (``RankInvarianceAccumulator``, ``_lcp_len``) can be imported —
# and unit-tested — without the benchmark dependency set. Annotations are
# strings under ``from __future__ import annotations``, so the type-only
# import lives under ``TYPE_CHECKING``.
if TYPE_CHECKING:
    from benchmarks.locomo_adapter import LoCoMoConversation

ARM_BASELINE: Final[str] = "baseline"
ARM_SPINE: Final[str] = "+spine"
ARM_SHUFFLED: Final[str] = "shuffled-control"
ARMS: Final[tuple[str, ...]] = (ARM_BASELINE, ARM_SPINE, ARM_SHUFFLED)

# Wide-retrieval operating point of the #1064 campaign (dev ran
# l1_limit=200 / budget=8000; the confirmatory inherited it). The
# production operating point (1500-token hook budget) is the G2
# flip-gate question, run via --budget / --l1-limit overrides.
DEFAULT_BUDGET: Final[int] = 8000
DEFAULT_L1_LIMIT: Final[int] = 200


@dataclass
class CoverageAccumulator:
    """Per-arm coverage aggregates."""

    n_questions: int = 0
    coverage_sum: float = 0.0
    n_all_evidence: int = 0
    per_category: dict[int, list[float]] = field(default_factory=dict)

    def add(self, category: int, coverage: float) -> None:
        self.n_questions += 1
        self.coverage_sum += coverage
        if coverage >= 1.0:
            self.n_all_evidence += 1
        self.per_category.setdefault(category, []).append(coverage)

    def overall(self) -> float:
        return (
            self.coverage_sum / self.n_questions if self.n_questions else 0.0
        )

    def all_evidence_rate(self) -> float:
        return (
            self.n_all_evidence / self.n_questions if self.n_questions else 0.0
        )


def ingest_with_evidence_map(
    store: MemoryStore, conv: LoCoMoConversation,
) -> dict[str, list[str]]:
    """Ingest a conversation, returning ``{dia_id: [belief_ids]}``.

    Mirrors ``locomo_adapter.ingest_conversation`` (one store session per
    LoCoMo session, date-prefixed turn text, parsed created_at) but records
    which belief ids each dialogue turn derived — the mapping the coverage
    scorer needs to turn evidence dia_ids into gold belief sets. Uses the
    internal ``_ingest_turn_ids`` because the public ``ingest_turn``
    returns a count only.
    """
    from benchmarks.locomo_adapter import _parse_locomo_datetime

    evidence_map: dict[str, list[str]] = {}
    for session in conv.sessions:
        am_session = store.create_session(
            model="locomo-benchmark",
            project_context=f"{conv.sample_id} session {session.session_num}",
        )
        created = _parse_locomo_datetime(session.date_time)
        if session.date_time:
            _ingest_turn_ids(
                store=store,
                text=f"[Session {session.session_num}, {session.date_time}]",
                source="locomo",
                session_id=am_session.id,
                created_at=created,
            )
        for turn in session.turns:
            ids = _ingest_turn_ids(
                store=store,
                text=f"[{session.date_time}] {turn.speaker}: {turn.text}",
                source="locomo",
                session_id=am_session.id,
                created_at=created,
            )
            if turn.dia_id:
                evidence_map[turn.dia_id] = list(ids)
        store.complete_session(am_session.id)
    return evidence_map


def shuffle_spine_edges(store: MemoryStore, *, seed: int) -> int:
    """Replace TEMPORAL_NEXT edges with an endpoint-permuted set.

    Same edge count, same node population, chronology destroyed: dst
    endpoints are permuted across edges with a seeded Fisher-Yates, and
    self-loops / duplicate triples are skipped (the tiny count lost to
    collisions is reported by the return value so the arms stay honest).
    Returns the number of edges written.
    """
    conn = store._conn  # noqa: SLF001 — bench-only surgical rewrite
    rows = conn.execute(
        "SELECT src, dst, weight FROM edges WHERE type = ?",
        (EDGE_TEMPORAL_NEXT,),
    ).fetchall()
    srcs = [str(r["src"]) for r in rows]
    dsts = [str(r["dst"]) for r in rows]
    weights = [float(r["weight"]) for r in rows]
    rng = random.Random(seed)
    rng.shuffle(dsts)
    conn.execute("DELETE FROM edges WHERE type = ?", (EDGE_TEMPORAL_NEXT,))
    written = 0
    seen: set[tuple[str, str]] = set()
    for src, dst, weight in zip(srcs, dsts, weights):
        if src == dst or (src, dst) in seen:
            continue
        seen.add((src, dst))
        conn.execute(
            "INSERT INTO edges (src, dst, type, weight) VALUES (?, ?, ?, ?)",
            (src, dst, EDGE_TEMPORAL_NEXT, weight),
        )
        written += 1
    conn.commit()
    return written


def restore_spine_edges(store: MemoryStore) -> int:
    """Drop all TEMPORAL_NEXT edges and rebuild the real spine."""
    conn = store._conn  # noqa: SLF001
    conn.execute("DELETE FROM edges WHERE type = ?", (EDGE_TEMPORAL_NEXT,))
    conn.commit()
    report = backfill_temporal_spine(store)
    return report.n_edges_written


def run_arm_on_store(
    store: MemoryStore,
    conv: LoCoMoConversation,
    evidence_map: dict[str, list[str]],
    arm: str,
    acc: CoverageAccumulator,
    *,
    budget: int,
    l1_limit: int,
    subset_qa: int | None,
) -> None:
    qa_pairs = conv.qa_pairs[:subset_qa] if subset_qa else conv.qa_pairs
    lane_on = arm != ARM_BASELINE
    for qa in qa_pairs:
        gold: set[str] = {
            bid
            for dia_id in qa.evidence
            for bid in evidence_map.get(dia_id, [])
        }
        if not gold:
            continue  # no scorable evidence (e.g. category-5 adversarial)
        result = retrieve_v2(
            store,
            qa.question,
            budget=budget,
            l1_limit=l1_limit,
            include_locked=False,
            use_temporal_spine=lane_on,
        )
        retrieved = {b.id for b in result.beliefs}
        acc.add(qa.category, len(gold & retrieved) / len(gold))


def _lcp_len(a: list[str], b: list[str]) -> int:
    """Length of the longest common prefix of two id lists."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


@dataclass
class RankInvarianceAccumulator:
    """G2 top-rank invariance aggregates (#1064 flip gate).

    The spine lane appends after the ``[locked, l25, l1, hrr]`` core and
    before BFS (see ``retrieval.py`` § temporal-spine lane), so the core
    prefix must stay byte-identical between the lane-off and lane-on arms;
    the lane may only insert its own hits below the core and evict
    BFS-tail items under the token budget. This accumulator measures
    whether that holds empirically. A *top-rank regression* is any
    core-ranked belief the lane displaces or reorders — the thing G2
    forbids. BFS-tail eviction is by-design (BFS is the lowest-priority
    lane and the spine legitimately spends budget above it) and is
    reported separately, not counted as a regression.
    """

    n_questions: int = 0
    head_invariant: int = 0          # core prefix identical, lane on vs off
    top_rank_displacements: int = 0  # questions with a displaced core belief
    core_mismatch: int = 0           # core length differed between arms
    lcp_sum: int = 0
    min_lcp: int | None = None
    tail_eviction_questions: int = 0
    tail_evicted_total: int = 0
    spine_contributed_questions: int = 0
    spine_added_sum: int = 0

    def add(
        self,
        baseline_ids: list[str],
        spine_ids: list[str],
        core_len_base: int,
        core_len_spine: int,
        n_spine_added: int,
    ) -> None:
        self.n_questions += 1
        if core_len_base != core_len_spine:
            self.core_mismatch += 1
        core = core_len_base
        if baseline_ids[:core] == spine_ids[:core]:
            self.head_invariant += 1
        lcp = _lcp_len(baseline_ids, spine_ids)
        self.lcp_sum += lcp
        self.min_lcp = lcp if self.min_lcp is None else min(self.min_lcp, lcp)
        spine_set = set(spine_ids)
        displaced_ranks = [
            i for i, bid in enumerate(baseline_ids) if bid not in spine_set
        ]
        if any(i < core for i in displaced_ranks):
            self.top_rank_displacements += 1
        tail_evicted = [i for i in displaced_ranks if i >= core]
        if tail_evicted:
            self.tail_eviction_questions += 1
            self.tail_evicted_total += len(tail_evicted)
        if n_spine_added > 0:
            self.spine_contributed_questions += 1
            self.spine_added_sum += n_spine_added

    def head_invariant_rate(self) -> float:
        return (
            self.head_invariant / self.n_questions if self.n_questions else 0.0
        )

    def mean_lcp(self) -> float:
        return self.lcp_sum / self.n_questions if self.n_questions else 0.0

    def passed(self) -> bool:
        """G2 top-rank invariance: no core belief displaced or reordered."""
        return (
            self.n_questions > 0
            and self.head_invariant == self.n_questions
            and self.top_rank_displacements == 0
            and self.core_mismatch == 0
        )


def run_rank_invariance_on_store(
    store: MemoryStore,
    conv: LoCoMoConversation,
    acc: RankInvarianceAccumulator,
    *,
    budget: int,
    l1_limit: int,
    subset_qa: int | None,
) -> None:
    """Paired lane-off / lane-on retrieval per question on one store.

    Ingest is arm-independent (the flag only changes retrieval), so both
    arms read the identical belief population and the real spine; the only
    variable is ``use_temporal_spine``. Reads ``last_lane_telemetry()``
    after each call to locate the core boundary
    (``locked + l25 + l1 + hrr_expand``) exactly, rather than inferring it.
    """
    from aelfrice.retrieval import last_lane_telemetry

    qa_pairs = conv.qa_pairs[:subset_qa] if subset_qa else conv.qa_pairs
    for qa in qa_pairs:
        base_result = retrieve_v2(
            store, qa.question, budget=budget, l1_limit=l1_limit,
            include_locked=False, use_temporal_spine=False,
        )
        base_ids = [b.id for b in base_result.beliefs]
        tel_b = last_lane_telemetry()
        core_b = tel_b.locked + tel_b.l25 + tel_b.l1 + tel_b.hrr_expand
        spine_result = retrieve_v2(
            store, qa.question, budget=budget, l1_limit=l1_limit,
            include_locked=False, use_temporal_spine=True,
        )
        spine_ids = [b.id for b in spine_result.beliefs]
        tel_s = last_lane_telemetry()
        core_s = tel_s.locked + tel_s.l25 + tel_s.l1 + tel_s.hrr_expand
        acc.add(base_ids, spine_ids, core_b, core_s, tel_s.temporal_spine)


def main() -> None:
    from benchmarks.locomo_adapter import (
        CATEGORY_NAMES,
        DEFAULT_DATA_PATH,
        load_locomo,
    )

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=DEFAULT_DATA_PATH)
    ap.add_argument("--out", default="/tmp/temporal_spine_ablation.json")
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    ap.add_argument("--l1-limit", type=int, default=DEFAULT_L1_LIMIT)
    ap.add_argument("--shuffle-seed", type=int, default=1064)
    ap.add_argument("--subset-convs", type=int, default=None)
    ap.add_argument("--subset-qa", type=int, default=None)
    ap.add_argument(
        "--rank-invariance",
        action="store_true",
        help=(
            "also run the G2 top-rank invariance pass (paired lane-off/on "
            "retrieval per question; verifies the core prefix is never "
            "displaced or reordered by the lane)"
        ),
    )
    args = ap.parse_args()

    conversations = load_locomo(args.data)
    if args.subset_convs:
        conversations = conversations[: args.subset_convs]

    accs: dict[str, CoverageAccumulator] = {
        arm: CoverageAccumulator() for arm in ARMS
    }
    rank_acc = RankInvarianceAccumulator()
    started = time.time()
    spine_edges_total = 0
    for i, conv in enumerate(conversations):
        store = MemoryStore(":memory:")
        try:
            evidence_map = ingest_with_evidence_map(store, conv)
            spine_report = backfill_temporal_spine(store)
            spine_edges_total += spine_report.n_edges_written
            if args.rank_invariance:
                run_rank_invariance_on_store(
                    store, conv, rank_acc,
                    budget=args.budget, l1_limit=args.l1_limit,
                    subset_qa=args.subset_qa,
                )
            for arm in ARMS:
                if arm == ARM_SHUFFLED:
                    shuffle_spine_edges(store, seed=args.shuffle_seed)
                run_arm_on_store(
                    store, conv, evidence_map, arm, accs[arm],
                    budget=args.budget, l1_limit=args.l1_limit,
                    subset_qa=args.subset_qa,
                )
                if arm == ARM_SHUFFLED:
                    restore_spine_edges(store)
        finally:
            store.close()
        print(
            f"[{i + 1}/{len(conversations)}] {conv.sample_id}: "
            + ", ".join(
                f"{arm}={accs[arm].overall():.3f}" for arm in ARMS
            )
        )

    report: dict[str, object] = {
        "bench": "temporal_spine_ablation",
        "issue": 1064,
        "data": args.data,
        "budget": args.budget,
        "l1_limit": args.l1_limit,
        "shuffle_seed": args.shuffle_seed,
        "n_conversations": len(conversations),
        "spine_edges_built": spine_edges_total,
        "elapsed_seconds": round(time.time() - started, 1),
        "arms": {
            arm: {
                "n_questions": acc.n_questions,
                "coverage_overall": round(acc.overall(), 4),
                "all_evidence_rate": round(acc.all_evidence_rate(), 4),
                "coverage_by_category": {
                    CATEGORY_NAMES.get(cat, str(cat)): round(
                        sum(vals) / len(vals), 4,
                    )
                    for cat, vals in sorted(acc.per_category.items())
                },
            }
            for arm, acc in accs.items()
        },
    }
    if args.rank_invariance:
        report["rank_invariance"] = {
            "n_questions": rank_acc.n_questions,
            "head_invariant": rank_acc.head_invariant,
            "head_invariant_rate": round(rank_acc.head_invariant_rate(), 4),
            "top_rank_displacements": rank_acc.top_rank_displacements,
            "core_length_mismatch": rank_acc.core_mismatch,
            "mean_lcp": round(rank_acc.mean_lcp(), 2),
            "min_lcp": rank_acc.min_lcp,
            "tail_eviction_questions": rank_acc.tail_eviction_questions,
            "tail_evicted_total": rank_acc.tail_evicted_total,
            "spine_contributed_questions": (
                rank_acc.spine_contributed_questions
            ),
            "spine_added_total": rank_acc.spine_added_sum,
            "g2_top_rank_invariance_pass": rank_acc.passed(),
        }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    base = accs[ARM_BASELINE].overall()
    spine = accs[ARM_SPINE].overall()
    shuffled = accs[ARM_SHUFFLED].overall()
    print(f"\nbaseline coverage         : {base:.4f}")
    print(f"+spine coverage           : {spine:.4f}  (Δ {spine - base:+.4f})")
    print(
        f"shuffled-control coverage : {shuffled:.4f}  "
        f"(Δ {shuffled - base:+.4f})"
    )
    if args.rank_invariance:
        print(
            f"\nG2 top-rank invariance ({rank_acc.n_questions} questions):"
        )
        print(
            f"  core prefix invariant     : "
            f"{rank_acc.head_invariant}/{rank_acc.n_questions} "
            f"({rank_acc.head_invariant_rate():.4f})"
        )
        print(
            f"  top-rank displacements    : "
            f"{rank_acc.top_rank_displacements}  (must be 0)"
        )
        print(f"  core-length mismatches    : {rank_acc.core_mismatch}")
        print(
            f"  mean/min common prefix    : "
            f"{rank_acc.mean_lcp():.2f} / {rank_acc.min_lcp}"
        )
        print(
            f"  BFS-tail evictions        : "
            f"{rank_acc.tail_evicted_total} across "
            f"{rank_acc.tail_eviction_questions} q (by-design)"
        )
        print(
            f"  spine contributed         : "
            f"{rank_acc.spine_contributed_questions} q, "
            f"{rank_acc.spine_added_sum} beliefs total"
        )
        print(
            f"  G2 PASS                   : {rank_acc.passed()}"
        )
    print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
