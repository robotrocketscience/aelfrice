"""Sweep top_k and budget for LongMemEval multi-session GT-in-context rate.

Root cause: stores have 247-730 beliefs but top_k=50 caps candidates.
This sweep tests whether increasing top_k improves GT-in-context rate.

Usage:
    uv run python benchmarks/longmemeval_budget_sweep.py
    uv run python benchmarks/longmemeval_budget_sweep.py --subset 20
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path
from typing import Final

_BENCH_DIR: str = str(Path(__file__).resolve().parent)
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)

from longmemeval_adapter import (  # type: ignore[import-untyped]
    LongMemEvalQuestion,
    load_from_huggingface,
    parse_questions,
    ingest_sessions,
)
from aelfrice.retrieval import retrieve_v2 as retrieve  # v1.0.x lab-compat shim
from aelfrice.store import MemoryStore

# (top_k, budget) combinations to sweep
CONFIGS: Final[list[tuple[int, int]]] = [
    (50, 2000),    # current default
    (100, 4000),   # 2x candidates, 2x budget
    (200, 8000),   # 4x candidates, 4x budget
]


def gt_in_context(context: str, answer: str | list[str]) -> bool:
    """Check if any ground truth answer appears in context."""
    ctx_lower: str = context.lower()
    if isinstance(answer, list):
        return any(str(a).lower() in ctx_lower for a in answer)
    return str(answer).lower() in ctx_lower


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="LongMemEval top_k/budget sweep on multi-session questions",
    )
    parser.add_argument(
        "--subset", type=int, default=None,
        help="Limit to first N multi-session questions",
    )
    args: argparse.Namespace = parser.parse_args()

    print("Loading LongMemEval dataset...")
    raw: list[dict[str, object]] = load_from_huggingface()
    questions: list[LongMemEvalQuestion] = parse_questions(raw)
    ms_questions: list[LongMemEvalQuestion] = [
        q for q in questions if q.question_type == "multi-session"
    ]
    print(f"Multi-session questions: {len(ms_questions)}")

    if args.subset is not None:
        ms_questions = ms_questions[:args.subset]
        print(f"Using first {len(ms_questions)}")

    for top_k, budget in CONFIGS:
        print(f"\n=== top_k={top_k}, budget={budget} ===")
        gt_hits: int = 0
        total: int = 0
        total_beliefs: int = 0
        t0: float = time.monotonic()

        with tempfile.TemporaryDirectory(prefix=f"lme_tk{top_k}_") as tmpdir:
            for i, q in enumerate(ms_questions):
                db_path: str = f"{tmpdir}/{q.question_id}.db"
                store: MemoryStore = MemoryStore(db_path)
                ingest_sessions(store, q)

                query_text: str = q.question
                if q.question_date:
                    query_text = f"[As of {q.question_date}] {q.question}"

                result = retrieve(
                    store=store,
                    query=query_text,
                    budget=budget,
                    top_k=top_k,
                    include_locked=False,
                    use_hrr=True,
                    use_bfs=True,
                )
                context: str = " ".join(b.content for b in result.beliefs)
                n_beliefs: int = len(result.beliefs)
                total_beliefs += n_beliefs

                if gt_in_context(context, q.answer):
                    gt_hits += 1
                total += 1

                if (i + 1) % 10 == 0:
                    elapsed: float = time.monotonic() - t0
                    print(
                        f"  [{i+1}/{len(ms_questions)}] "
                        f"GT-in-ctx: {gt_hits}/{total} ({gt_hits/total*100:.0f}%) "
                        f"avg_beliefs: {total_beliefs/total:.0f} "
                        f"elapsed: {elapsed:.0f}s"
                    )

        elapsed = time.monotonic() - t0
        pct: float = gt_hits / total * 100 if total else 0
        avg_b: float = total_beliefs / total if total else 0
        print(
            f"  FINAL: GT-in-ctx: {gt_hits}/{total} ({pct:.1f}%) "
            f"avg_beliefs: {avg_b:.0f} elapsed: {elapsed:.0f}s"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
