"""Single-coordinate sweep of `posterior_weight` on LongMemEval (#1273).

This is proposal `9`'s own cheapest kill experiment from #1174, descoped
exactly as that proposal writes it: sweep ONE coordinate --
``DEFAULT_POSTERIOR_WEIGHT`` (``scoring.py:50``, 0.5) -- over a four-point
grid, and ask whether the hand-picked default is already the argmax. It is
deliberately *not* the coordinate-ascent harness; that build is gated on
this result.

Why this one can run when its neighbours on #1174 cannot
--------------------------------------------------------
The operator's ruling on #1268 holds proposals whose A/B needs a relevance
gold set, because that instrument's variance currently exceeds the effects
it would measure. This sweep does not use it. The outcome measures come
from :mod:`benchmarks.retrieval_metrics` -- MRR and recall@k over the
*ordered* retrieved list, with relevance decided by whether a normalised
gold surface occurs in a belief. No reader, no judge, no LLM call, so no
judge variance. It also adds no flag: ``posterior_weight`` already resolves
through env / kwarg / TOML / default (``retrieval.resolve_posterior_weight``).

Four traps this umbrella has already paid for, and how they are handled
----------------------------------------------------------------------
1. ``retrieve_v2``, not ``retrieve()`` -- the staged lanes live only in v2.
   Imported the same way the sibling LongMemEval harnesses import it.
2. The budget is pinned to ``SWEEP_BUDGET`` (2000), which is *not*
   ``DEFAULT_TOKEN_BUDGET`` (2400). ``retrieve_with_tiers`` silently
   downgrades an explicit 2400 to ``LEGACY_TOKEN_BUDGET`` when the entity
   index is off (#1271); that equality confounded an arm of #1269. Staying
   off the equality means no arm can be silently re-budgeted.
3. The entity lane is held **fixed** across every arm and its resolved
   value is recorded in the output. Its fire rate on a benchmark corpus is
   a self-ingestion artifact, so varying it would not measure retrieval.
4. The resolver is **env-first**: ``AELFRICE_POSTERIOR_WEIGHT`` outranks the
   explicit kwarg. A harness that passed the weight as a kwarg while an
   ambient env var was set would sweep one value four times. This module
   sweeps *by* the env var and asserts the **resolved** weight per arm
   before each retrieval, so a silent tie is impossible.

Determinism
-----------
Ingestion runs once per question and every arm retrieves against that same
store; ``--verify-no-write`` checks (by file digest) that retrieval leaves
the store byte-identical, so the arms are not ordered-dependent. The
bootstrap uses a fixed seed. Given the same dataset the whole run is
reproducible.

Usage::

    uv run python benchmarks/posterior_weight_sweep.py --subset 20
    uv run python benchmarks/posterior_weight_sweep.py --out results.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Final

_BENCH_DIR: str = str(Path(__file__).resolve().parent.parent)
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)

from aelfrice.retrieval import (  # noqa: E402
    ENV_POSTERIOR_WEIGHT,
    resolve_posterior_weight,
    retrieve_v2 as retrieve,
)
from aelfrice.store import MemoryStore  # noqa: E402
from benchmarks.longmemeval_adapter import (  # noqa: E402
    LongMemEvalQuestion,
    ingest_sessions,
    load_from_file,
    load_from_huggingface,
    parse_questions,
)
from benchmarks.retrieval_metrics import gold_ranks, retrieval_metrics  # noqa: E402

# ---------------------------------------------------------------------------
# Experiment constants -- every one of these is reported in the output so a
# reader never has to infer what was held fixed.
# ---------------------------------------------------------------------------

#: The grid, verbatim from proposal `9`'s kill experiment.
GRID: Final[tuple[float, ...]] = (0.0, 0.25, 0.5, 1.0)

#: The incumbent every arm is compared against -- `DEFAULT_POSTERIOR_WEIGHT`.
BASELINE: Final[float] = 0.5

#: Held off `DEFAULT_TOKEN_BUDGET` (2400) so #1271's silent downgrade to
#: `LEGACY_TOKEN_BUDGET` cannot fire on any arm. Matches the budget the
#: sibling LongMemEval adapter already uses.
SWEEP_BUDGET: Final[int] = 2000

#: Cut-offs reported per arm. Mirrors `retrieval_metrics.DEFAULT_KS`.
KS: Final[tuple[int, ...]] = (1, 5, 10, 20)

#: Bootstrap resamples for the paired CI on the primary metric.
BOOTSTRAP_N: Final[int] = 10_000

#: Fixed so the CI is reproducible; the sampling uncertainty this quantifies
#: is across *questions*, not across reruns -- the pipeline itself is exact.
BOOTSTRAP_SEED: Final[int] = 0

#: Primary metric. Reciprocal rank reads how high the first gold-bearing
#: belief landed, which is what a ranking weight can actually move.
PRIMARY: Final[str] = "reciprocal_rank"


def holdout_half(question_id: str) -> bool:
    """True when a question belongs to the pre-declared holdout half.

    The split rule is fixed **before** the run and is a pure function of the
    question id: the low bit of its SHA-256. It carries no seed and no
    ordering dependence, so it is identical for anyone who reruns this, and
    it cannot be tuned after seeing a result.

    Nothing is *fitted* here -- four values are read off a grid -- so the
    holdout is not there to prevent overfitting. It is there so that a
    headline argmax has to survive being recomputed on a disjoint half
    before anyone acts on it.
    """
    digest: str = hashlib.sha256(question_id.encode("utf-8")).hexdigest()
    return int(digest, 16) & 1 == 1


def _resolved_weight() -> float:
    """The weight retrieval will actually use, given the current environment.

    Called with no explicit kwarg because that is exactly how this harness
    invokes ``retrieve``: the env var is the lever, so the resolved value is
    what the env var produced.
    """
    return resolve_posterior_weight()


def _digest(path: Path) -> str:
    """SHA-256 of a store file, for the no-write assertion."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gold_surfaces(question: LongMemEvalQuestion) -> list[str]:
    """Gold answer surfaces, folded to a list.

    LongMemEval lists several acceptable surfaces of *one* answer for some
    categories; ``retrieval_metrics`` treats matching any of them as a hit,
    matching ``qa_scoring.score_multi_answer``.
    """
    if isinstance(question.answer, list):
        return [str(a) for a in question.answer]
    return [str(question.answer)]


def run_arm(
    store: MemoryStore,
    question: LongMemEvalQuestion,
    weight: float,
    *,
    entity_index: bool | None,
) -> tuple[dict[str, float], list[int], int]:
    """Retrieve for one question under one weight.

    Returns ``(metrics, gold_ranks, n_retrieved)``. Sets the env var and
    asserts the *resolved* weight matches before retrieving -- trap 4.
    """
    os.environ[ENV_POSTERIOR_WEIGHT] = repr(weight)
    got: float = _resolved_weight()
    if got != weight:
        raise SystemExit(
            f"posterior_weight did not take: requested {weight}, resolved {got}. "
            f"The sweep would have measured one value more than once."
        )

    query: str = question.question
    if question.question_date:
        query = f"[As of {question.question_date}] {question.question}"

    result = retrieve(
        store=store,
        query=query,
        budget=SWEEP_BUDGET,
        include_locked=False,
        use_bfs=True,
        use_entity_index=entity_index,
    )
    beliefs: list[str] = [b.content for b in result.beliefs]
    golds: list[str] = _gold_surfaces(question)
    return retrieval_metrics(beliefs, golds), gold_ranks(beliefs, golds), len(beliefs)


def sign_test_p(n_better: int, n_worse: int) -> float:
    """Exact two-sided binomial sign test on the non-tied pairs.

    Reported alongside the bootstrap CI because the two answer different
    questions and can disagree: the sign test asks whether an arm wins *more
    often*, the CI asks whether it wins *by more*. An arm that takes many
    small wins and a few large losses passes one and fails the other, and
    reporting only the flattering one is how a null gets dressed as a result.
    """
    n: int = n_better + n_worse
    if n == 0:
        return 1.0
    k: int = max(n_better, n_worse)
    tail: float = sum(math.comb(n, i) for i in range(k, n + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def paired_bootstrap(
    deltas: list[float], *, n: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for the mean of paired per-question deltas.

    Paired because every arm answers the same questions against the same
    ingested stores; the only thing that varies within a pair is the weight.
    """
    if not deltas:
        return (0.0, 0.0)
    rng: random.Random = random.Random(seed)
    size: int = len(deltas)
    means: list[float] = []
    for _ in range(n):
        means.append(
            statistics.fmean(deltas[rng.randrange(size)] for _ in range(size))
        )
    means.sort()
    lo: float = means[int(0.025 * n)]
    hi: float = means[int(0.975 * n) - 1]
    return (lo, hi)


def summarise(
    per_question: dict[str, dict[float, dict[str, float]]],
    ranks: dict[str, dict[float, list[int]]],
    qids: list[str],
) -> dict[str, object]:
    """Aggregate one question set into per-arm means and paired comparisons.

    ``movable`` is the count of questions whose gold ranking *changed at all*
    between the arm and the baseline. It is reported because a mean delta
    computed over a set where almost nothing moved is not a small effect, it
    is an absent measurement -- and the two look identical in the mean.
    """
    arms: dict[str, object] = {}
    for weight in GRID:
        arms[str(weight)] = {
            "mean_" + m: statistics.fmean(
                per_question[q][weight][m] for q in qids
            )
            for m in [PRIMARY] + [f"recall_at_{k}" for k in KS]
        }

    comparisons: dict[str, object] = {}
    for weight in GRID:
        if weight == BASELINE:
            continue
        deltas: list[float] = [
            per_question[q][weight][PRIMARY] - per_question[q][BASELINE][PRIMARY]
            for q in qids
        ]
        movable: int = sum(
            1 for q in qids if ranks[q][weight] != ranks[q][BASELINE]
        )
        # `movable` counts any change in the gold ranking, including the tail.
        # A gold belief sliding from rank 17 to 18 changes that list and moves
        # no metric. `movable_head` counts the questions where the *first*
        # gold rank moved -- the only movement reciprocal rank can see. When
        # the two diverge, the weight is reordering the parts of the list the
        # metric does not read, which is a different finding from inertness.
        movable_head: int = sum(
            1 for q in qids
            if (ranks[q][weight][:1] or [0]) != (ranks[q][BASELINE][:1] or [0])
        )
        lo, hi = paired_bootstrap(deltas)
        comparisons[str(weight)] = {
            "mean_delta_" + PRIMARY: statistics.fmean(deltas),
            "ci95": [lo, hi],
            "ci_excludes_zero": bool(lo > 0.0 or hi < 0.0),
            "n_better": sum(1 for d in deltas if d > 0),
            "n_worse": sum(1 for d in deltas if d < 0),
            "n_tied": sum(1 for d in deltas if d == 0),
            "sign_test_p": sign_test_p(
                sum(1 for d in deltas if d > 0), sum(1 for d in deltas if d < 0),
            ),
            "movable": movable,
            "movable_head": movable_head,
        }

    best: str = max(
        arms, key=lambda w: arms[w]["mean_" + PRIMARY],  # type: ignore[index]
    )
    return {
        "n_questions": len(qids),
        "arms": arms,
        "vs_baseline": comparisons,
        "argmax": float(best),
        "baseline_is_argmax": float(best) == BASELINE,
    }


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="posterior_weight single-coordinate sweep on LongMemEval (#1273)",
    )
    parser.add_argument(
        "--subset", type=int, default=None,
        help="Limit to the first N questions (smoke runs).",
    )
    parser.add_argument(
        "--data-file", default=None,
        help="Local oracle JSON; defaults to the HuggingFace copy.",
    )
    parser.add_argument("--out", default=None, help="Write the full result JSON here.")
    parser.add_argument(
        "--verify-no-write", action="store_true",
        help="Assert retrieval leaves the store byte-identical (first question only).",
    )
    args: argparse.Namespace = parser.parse_args()

    # Trap 4, belt and braces: clear any ambient value before the run so the
    # first arm cannot inherit one, and record the entity lane's resolved
    # state rather than assuming it.
    os.environ.pop(ENV_POSTERIOR_WEIGHT, None)
    entity_index: bool | None = None

    print("Loading LongMemEval...")
    raw: list[dict[str, object]] = (
        load_from_file(args.data_file) if args.data_file else load_from_huggingface()
    )
    questions: list[LongMemEvalQuestion] = parse_questions(raw)
    if args.subset:
        questions = questions[: args.subset]
    print(f"{len(questions)} questions; grid {GRID}; budget {SWEEP_BUDGET}")

    per_question: dict[str, dict[float, dict[str, float]]] = {}
    ranks: dict[str, dict[float, list[int]]] = {}
    retrieved_counts: dict[str, dict[float, int]] = {}
    t0: float = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="posterior_weight_sweep_") as tmpdir:
        for i, q in enumerate(questions):
            db_path: Path = Path(tmpdir) / f"{q.question_id}.db"
            store: MemoryStore = MemoryStore(str(db_path))
            ingest_sessions(store, q)

            before: str | None = (
                _digest(db_path) if (args.verify_no_write and i == 0) else None
            )

            per_question[q.question_id] = {}
            ranks[q.question_id] = {}
            retrieved_counts[q.question_id] = {}
            for weight in GRID:
                metrics, gold, n_ret = run_arm(
                    store, q, weight, entity_index=entity_index,
                )
                per_question[q.question_id][weight] = metrics
                ranks[q.question_id][weight] = gold
                retrieved_counts[q.question_id][weight] = n_ret

            if before is not None:
                after: str = _digest(db_path)
                if before != after:
                    raise SystemExit(
                        "retrieval mutated the store; arms are order-dependent and "
                        "this sweep is invalid as written. Re-run with a fresh copy "
                        "of the store per arm."
                    )
                print("  no-write check: store byte-identical after all arms")

            elapsed: float = time.monotonic() - t0
            print(
                f"  [{i+1}/{len(questions)}] {q.question_id} "
                f"({q.question_type}) {elapsed:.0f}s"
            )

    # A question whose gold surface never appears in *any* arm's retrieved set
    # is not evidence about ranking -- no reordering of what was retrieved can
    # produce a hit. Reporting the rate against the raw question count instead
    # of the scorable set is what makes rates from different adapters
    # non-comparable; the share is per-adapter.
    scorable: list[str] = [
        qid for qid in per_question
        if any(ranks[qid][w] for w in GRID)
    ]
    unscorable: list[str] = [qid for qid in per_question if qid not in scorable]

    all_ids: list[str] = list(per_question)
    holdout: list[str] = [q for q in scorable if holdout_half(q)]
    kept: list[str] = [q for q in scorable if not holdout_half(q)]

    report: dict[str, object] = {
        "issue": 1273,
        "parent": 1174,
        "proposal": 9,
        "held_fixed": {
            "budget": SWEEP_BUDGET,
            "budget_note": (
                "off DEFAULT_TOKEN_BUDGET=2400 so #1271's silent downgrade to "
                "LEGACY_TOKEN_BUDGET cannot fire"
            ),
            "use_bfs": True,
            "include_locked": False,
            "use_entity_index_requested": entity_index,
            "grid": list(GRID),
            "baseline": BASELINE,
            "primary_metric": PRIMARY,
        },
        "denominators": {
            "questions_run": len(all_ids),
            "scorable": len(scorable),
            "unscorable": len(unscorable),
            "unscorable_share": (
                len(unscorable) / len(all_ids) if all_ids else 0.0
            ),
            "abstention_questions": sum(
                1 for qid in all_ids if qid.endswith("_abs")
            ),
            "abstention_unscorable": sum(
                1 for qid in unscorable if qid.endswith("_abs")
            ),
            "note": (
                "unscorable = gold surface absent from every arm's retrieved set; "
                "no reordering can score it. All rates below are over the scorable "
                "set. The `_abs` abstention variants are unscorable by "
                "construction -- their gold is that no answer is supported -- so "
                "they are reported separately rather than folded into a retrieval "
                "failure rate."
            ),
        },
        "full_scorable": summarise(per_question, ranks, scorable),
        "split_kept": summarise(per_question, ranks, kept),
        "split_holdout": summarise(per_question, ranks, holdout),
        "split_rule": "sha256(question_id) low bit; fixed before the run",
    }

    print()
    print(f"scorable {len(scorable)}/{len(all_ids)} "
          f"(unscorable {len(unscorable)})")
    full: dict[str, object] = report["full_scorable"]  # type: ignore[assignment]
    print(f"{'weight':>8} {'mean MRR':>10} " + " ".join(f"{'r@'+str(k):>7}" for k in KS))
    for weight in GRID:
        arm: dict[str, float] = full["arms"][str(weight)]  # type: ignore[index]
        marker: str = "  <- default" if weight == BASELINE else ""
        print(
            f"{weight:>8} {arm['mean_' + PRIMARY]:>10.4f} "
            + " ".join(f"{arm['mean_recall_at_' + str(k)]:>7.4f}" for k in KS)
            + marker
        )
    print()
    print(f"argmax (full scorable): {full['argmax']}  "
          f"baseline_is_argmax={full['baseline_is_argmax']}")
    for weight, comp in full["vs_baseline"].items():  # type: ignore[index]
        print(
            f"  w={weight}: dMRR {comp['mean_delta_' + PRIMARY]:+.4f} "
            f"CI95 [{comp['ci95'][0]:+.4f}, {comp['ci95'][1]:+.4f}] "
            f"excl0={comp['ci_excludes_zero']} "
            f"better/worse/tied {comp['n_better']}/{comp['n_worse']}/{comp['n_tied']} "
            f"sign-p {comp['sign_test_p']:.4f} "
            f"movable {comp['movable']} (head {comp['movable_head']})"
        )
    print(
        f"holdout argmax: {report['split_holdout']['argmax']}  "  # type: ignore[index]
        f"kept argmax: {report['split_kept']['argmax']}"  # type: ignore[index]
    )

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
