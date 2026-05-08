"""MAB benchmark adapter with the v1.3.0 entity-index (L2.5) flag flipped on.

Thin wrapper over `mab_adapter.py`. Differences:

1. Constructs every per-row `MemoryStore` and explicitly enables L2.5
   by passing `use_entity_index=True` to `retrieve_v2`. The flag is
   default-on at v1.3.0 — the explicit pass keeps the result
   reproducible regardless of the user's `.aelfrice.toml` or
   `AELFRICE_ENTITY_INDEX` env var.
2. Reports an extended results block: per-question L0 / L1 / L2.5
   counts in the `per_question` JSON, plus an aggregate L2.5 hit
   rate in the summary.
3. Same MAB metrics (`exact_match`, `substring_exact_match`, `f1`).
   Pass-through to the same scorer.

Run:
    uv run python benchmarks/mab_entity_index_adapter.py \\
        --split Conflict_Resolution \\
        --source factconsolidation_mh_262k \\
        --rows 5 --subset 5

The default-off comparison is the existing `mab_adapter.py`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from aelfrice.retrieval import retrieve_v2
from aelfrice.store import MemoryStore

# Reuse the public surface from mab_adapter so we don't duplicate
# constants, scoring functions, chunking, or dataset loading. Only
# the per-question loop is replaced (because we need to forward the
# entity-index flag and capture the per-tier counts).
from benchmarks.mab_adapter import (  # type: ignore[import-untyped]
    BASELINES,
    DEFAULT_CHUNK_SIZE,
    MABResult,
    MABRow,
    VALID_SPLITS,
    _count_tokens,
    ingest_context,
    load_mab_split,
    merge_results,
    score_multi_answer,
)


@dataclass
class TierCounts:
    """Per-question L0 / L2.5 / L1 counts surfaced by retrieve_with_tiers."""
    locked: int = 0
    l25: int = 0
    l1: int = 0


@dataclass
class EntityIndexMABResult(MABResult):
    """MABResult extended with aggregate per-tier counts.

    `tier_counts` holds the per-question lists so the JSON output
    can show distribution, not just averages. Included in the
    `per_question` payload.
    """
    tier_counts: list[TierCounts] = field(
        default_factory=lambda: list[TierCounts](),
    )


def query_aelfrice_with_tiers(
    store: MemoryStore,
    question: str,
    budget: int,
) -> tuple[str, TierCounts]:
    """Query aelfrice with the entity-index flag explicitly ON and
    return both the prediction text and the per-tier counts."""
    result = retrieve_v2(
        store=store,
        query=question,
        budget=budget,
        include_locked=False,
        use_hrr=True,
        use_bfs=True,
        use_entity_index=True,
    )
    parts: list[str] = [b.content for b in result.beliefs]
    counts = TierCounts(
        locked=len(result.locked_ids),
        l25=len(result.entity_hits),
        l1=len(result.l1_ids),
    )
    return " ".join(parts), counts


def run_row_with_tiers(
    row: MABRow,
    db_dir: str,
    row_idx: int,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    budget: int = 2400,
    subset: int | None = None,
) -> EntityIndexMABResult:
    """Run the full benchmark pipeline on one dataset row with tier
    counts surfaced.

    Uses a fresh DB per row for isolation. Default budget bumped from
    the v1.0 2000 to v1.3.0's 2400 so L2.5 has its sub-budget
    available.
    """
    db_path: str = f"{db_dir}/mab_eidx_row_{row_idx:04d}.db"
    store: MemoryStore = MemoryStore(db_path)

    result: EntityIndexMABResult = EntityIndexMABResult(label=row.source)

    # Ingest context.
    t0: float = time.monotonic()
    result.ingest_chunks = ingest_context(store, row, chunk_size)
    result.ingest_time_s = time.monotonic() - t0

    questions: list[str] = row.questions
    answers: list[list[str]] = row.answers
    if subset is not None:
        questions = questions[:subset]
        answers = answers[:subset]

    t1: float = time.monotonic()
    for q_idx, (question, answer_list) in enumerate(zip(questions, answers)):
        prediction, counts = query_aelfrice_with_tiers(
            store, question, budget=budget,
        )
        scores: dict[str, float] = score_multi_answer(prediction, answer_list)

        result.total_questions += 1
        for metric, val in scores.items():
            result.scores[metric].append(val)

        result.per_question.append({
            "id": q_idx,
            "row_idx": row_idx,
            "source": row.source,
            "question": question,
            "context": prediction,
            "tier_counts": {
                "locked": counts.locked,
                "l25": counts.l25,
                "l1": counts.l1,
            },
        })
        result.tier_counts.append(counts)
        result.ground_truth.append({
            "id": q_idx,
            "row_idx": row_idx,
            "answers": answer_list,
        })

    result.query_time_s = time.monotonic() - t1
    return result


def _aggregate_tier_counts(
    results: list[EntityIndexMABResult],
) -> tuple[float, float, float, float]:
    """Return (mean_locked, mean_l25, mean_l1, l25_hit_rate) across
    every per-question record. l25_hit_rate is the fraction of
    questions that received at least one L2.5 hit."""
    counts: list[TierCounts] = []
    for r in results:
        counts.extend(r.tier_counts)
    if not counts:
        return 0.0, 0.0, 0.0, 0.0
    n = len(counts)
    total_locked = sum(c.locked for c in counts)
    total_l25 = sum(c.l25 for c in counts)
    total_l1 = sum(c.l1 for c in counts)
    hits = sum(1 for c in counts if c.l25 > 0)
    return (
        total_locked / n,
        total_l25 / n,
        total_l1 / n,
        hits / n,
    )


def print_results(
    result: EntityIndexMABResult,
    *,
    tier_summary: tuple[float, float, float, float] | None = None,
) -> None:
    """Print formatted benchmark results, including the L0/L1/L2.5
    counts surface."""
    print(f"\n{'=' * 60}")
    print(f"MAB+EntityIndex Results: {result.label}")
    print(f"{'=' * 60}")
    print(f"Total questions:         {result.total_questions}")
    print(
        f"Exact match:             "
        f"{result.mean_score('exact_match'):.4f} "
        f"({result.mean_score('exact_match') * 100:.1f}%)"
    )
    print(
        f"Substring exact match:   "
        f"{result.mean_score('substring_exact_match'):.4f} "
        f"({result.mean_score('substring_exact_match') * 100:.1f}%)"
    )
    print(
        f"F1:                      "
        f"{result.mean_score('f1'):.4f} "
        f"({result.mean_score('f1') * 100:.1f}%)"
    )
    print(f"Chunks ingested:         {result.ingest_chunks}")
    print(f"Ingest time:             {result.ingest_time_s:.2f}s")
    print(f"Query time:              {result.query_time_s:.2f}s")
    if result.total_questions > 0:
        print(
            f"Avg query latency:       "
            f"{result.query_time_s / result.total_questions * 1000:.1f}ms"
        )
    if tier_summary is not None:
        m_l0, m_l25, m_l1, hr = tier_summary
        print()
        print(f"Mean L0 (locked) per query:    {m_l0:.2f}")
        print(f"Mean L2.5 per query:           {m_l25:.2f}")
        print(f"Mean L1 per query:             {m_l1:.2f}")
        print(f"L2.5 hit rate (≥1 hit / Q):    {hr * 100:.1f}%")
    print()

    print("Paper baselines (Conflict Resolution):")
    for name, score in BASELINES.items():
        print(f"  {name}: {score}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_BUDGET: Final[int] = 2400


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=(
            "MemoryAgentBench benchmark on aelfrice with the v1.3.0 "
            "entity-index (L2.5) flag explicitly ON."
        ),
    )
    parser.add_argument(
        "--split", default="Conflict_Resolution",
        choices=VALID_SPLITS,
    )
    parser.add_argument("--source", default=None)
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--output", default=None)
    args: argparse.Namespace = parser.parse_args()

    # Defensive: paranoid users may have AELFRICE_ENTITY_INDEX=0 set
    # globally and not realise the adapter forces the flag at the
    # call-site. We do nothing about that — the explicit kwarg in
    # query_aelfrice_with_tiers already overrides the env var per
    # the precedence rules in retrieval.is_entity_index_enabled.
    # Surface a one-line note so the user knows which path is hot.
    if os.environ.get("AELFRICE_ENTITY_INDEX") == "0":
        print(
            "note: AELFRICE_ENTITY_INDEX=0 is set in the env, but this "
            "adapter passes use_entity_index=True explicitly and the "
            "explicit kwarg wins. Reading AELFRICE_ENTITY_INDEX would "
            "only affect callers that don't pass the kwarg."
        )

    print(f"Loading MAB dataset split: {args.split}")
    if args.source:
        print(f"Filtering by source: {args.source}")
    try:
        rows: list[MABRow] = load_mab_split(args.split, source_filter=args.source)
    except FileNotFoundError as exc:
        print(
            f"MAB data not found: {exc}",
            file=sys.stderr,
        )
        sys.exit(2)
    print(f"Loaded {len(rows)} rows")

    if not rows:
        print(
            "No rows matched. Check --split and --source values.",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.rows is not None:
        rows = rows[:args.rows]
        print(f"Using first {len(rows)} rows")

    results: list[EntityIndexMABResult] = []
    with tempfile.TemporaryDirectory(prefix="mab_eidx_") as tmpdir:
        for idx, row in enumerate(rows):
            context_tokens: int = _count_tokens(row.context)
            n_questions: int = len(row.questions)
            if args.subset is not None:
                n_questions = min(n_questions, args.subset)
            print(
                f"\n--- Row {idx}: source={row.source}, "
                f"context={context_tokens} tokens, "
                f"{n_questions} questions ---"
            )

            row_result: EntityIndexMABResult = run_row_with_tiers(
                row, tmpdir, idx,
                chunk_size=args.chunk_size,
                budget=args.budget,
                subset=args.subset,
            )
            results.append(row_result)
            tier_summary = _aggregate_tier_counts([row_result])
            print_results(row_result, tier_summary=tier_summary)

    if len(results) > 1:
        merged_base = merge_results(
            [r for r in results], label=f"{args.split} (all)",
        )
        # MABResult merge doesn't carry tier_counts; reconstruct.
        merged: EntityIndexMABResult = EntityIndexMABResult(
            label=merged_base.label,
            total_questions=merged_base.total_questions,
            scores=merged_base.scores,
            ingest_chunks=merged_base.ingest_chunks,
            ingest_time_s=merged_base.ingest_time_s,
            query_time_s=merged_base.query_time_s,
            per_question=merged_base.per_question,
            ground_truth=merged_base.ground_truth,
        )
        for r in results:
            merged.tier_counts.extend(r.tier_counts)
        merged_summary = _aggregate_tier_counts(results)
        print_results(merged, tier_summary=merged_summary)

    if args.output:
        out_result: EntityIndexMABResult = (
            results[0] if len(results) == 1 else _identity_merge(results)
        )
        m_l0, m_l25, m_l1, hr = _aggregate_tier_counts(results)
        output_data: dict[str, object] = {
            "split": args.split,
            "source_filter": args.source,
            "total_questions": out_result.total_questions,
            "exact_match": round(out_result.mean_score("exact_match"), 4),
            "substring_exact_match": round(
                out_result.mean_score("substring_exact_match"), 4,
            ),
            "f1": round(out_result.mean_score("f1"), 4),
            "tier_means": {
                "locked": round(m_l0, 3),
                "l25": round(m_l25, 3),
                "l1": round(m_l1, 3),
                "l25_hit_rate": round(hr, 4),
            },
            "per_question": out_result.per_question,
        }
        output_path: Path = Path(args.output)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nDetailed results written to {args.output}")


def _identity_merge(
    results: list[EntityIndexMABResult],
) -> EntityIndexMABResult:
    """Local helper: same shape as merge_results but preserves
    tier_counts (which the base merge_results doesn't know about)."""
    out = EntityIndexMABResult(label="ALL")
    for r in results:
        out.total_questions += r.total_questions
        for metric in out.scores:
            out.scores[metric].extend(r.scores.get(metric, []))
        out.ingest_chunks += r.ingest_chunks
        out.ingest_time_s += r.ingest_time_s
        out.query_time_s += r.query_time_s
        out.per_question.extend(r.per_question)
        out.ground_truth.extend(r.ground_truth)
        out.tier_counts.extend(r.tier_counts)
    return out


if __name__ == "__main__":
    main()
