"""Sweep temporal_half_life_seconds on StructMemEval temporal-ordering tasks.

Quantifies retrieval accuracy across a range of half-life values on the
StructMemEval ``location`` and ``accounting`` task families, where temporal
ordering changes correctness (current state vs historical state).

Usage:
    uv run python benchmarks/temporal_blend_sweep.py
    uv run python benchmarks/temporal_blend_sweep.py --data /path/to/StructMemEval/benchmark/data
    uv run python benchmarks/temporal_blend_sweep.py --task location --bench small
    uv run python benchmarks/temporal_blend_sweep.py --output benchmarks/results/temporal_half_life_sweep.json

Data prerequisite:
    StructMemEval corpus at /tmp/StructMemEval/benchmark/data (default) or the
    path given by --data. Clone and prepare with:

        git clone https://github.com/Wangt-CN/StructMemEval.git /tmp/StructMemEval

    When the data directory does not exist or yields zero cases, the harness
    falls back to a two-row inline fixture so the sweep can be validated
    structurally without real corpus data. Partial-corpus runs produce a
    result table annotated with the n_queries count so callers can judge
    evidential weight.

Half-life sweep grid:
    Sentinel None = no decay (temporal_sort=False, equivalent to half_life=inf).
    Finite values: 1h, 6h, 24h, 3d, 7d, 14d, 30d.
    Grid order is ascending (fastest decay first, no-decay last) so the
    result table reads left-to-right from aggressive to none.

Output:
    benchmarks/results/temporal_half_life_sweep.json
    Columns: half_life_label, half_life_seconds (null=inf), score, n_queries,
    task, bench, elapsed_sec.

Refs:
    Issue #487  (this sweep)
    PR #483     (temporal_sort kwarg implementation)
    src/aelfrice/retrieval.py::resolve_temporal_half_life
    benchmarks/structmemeval_adapter.py::query_aelfrice
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Sweep grid
# ---------------------------------------------------------------------------

# Each entry is (human-readable label, half_life_seconds | None).
# None means no decay: temporal_sort=False, equivalent to half_life=+inf.
# Order: most aggressive decay first, no-decay last.
SWEEP_GRID: Final[list[tuple[str, float | None]]] = [
    ("1h",  1 * 3600.0),
    ("6h",  6 * 3600.0),
    ("24h", 24 * 3600.0),
    ("3d",  3 * 24 * 3600.0),
    ("7d",  7 * 24 * 3600.0),    # operator-ratified default (#473 A1=A)
    ("14d", 14 * 24 * 3600.0),
    ("30d", 30 * 24 * 3600.0),
    ("inf", None),                # no decay baseline
]

# Tasks where temporal ordering changes correctness.
TEMPORAL_TASKS: Final[list[str]] = ["location", "accounting"]

# ---------------------------------------------------------------------------
# Inline fixture (fallback when corpus data is absent)
# ---------------------------------------------------------------------------

# Two synthetic StructMemEval-shaped cases exercising temporal ordering:
# each case has two sessions (old + recent) and one query asking for the
# current state.  The correct answer appears only in the more recent session.
_FIXTURE_CASES: Final[list[dict]] = [
    {
        "case_id": "fixture_location_01",
        "sessions": [
            {
                "session_id": "s1",
                "topic": "user location old",
                "messages": [
                    {"role": "user",      "content": "I am in Berlin."},
                    {"role": "assistant", "content": "Got it, Berlin."},
                ],
            },
            {
                "session_id": "s2",
                "topic": "user location new",
                "messages": [
                    {"role": "user",      "content": "I moved to Tokyo last week."},
                    {"role": "assistant", "content": "Noted, Tokyo."},
                ],
            },
        ],
        "queries": [
            {
                "question": "Where is the user currently located?",
                "reference_answer": {"text": "Tokyo"},
            },
        ],
    },
    {
        "case_id": "fixture_accounting_01",
        "sessions": [
            {
                "session_id": "s1",
                "topic": "balance old",
                "messages": [
                    {"role": "user",      "content": "My savings account balance is 500 dollars."},
                    {"role": "assistant", "content": "Recorded 500."},
                ],
            },
            {
                "session_id": "s2",
                "topic": "balance updated",
                "messages": [
                    {"role": "user",      "content": "I deposited 300 dollars, balance is now 800 dollars."},
                    {"role": "assistant", "content": "Updated to 800."},
                ],
            },
        ],
        "queries": [
            {
                "question": "What is the current savings account balance?",
                "reference_answer": {"text": "800"},
            },
        ],
    },
]


# ---------------------------------------------------------------------------
# Imports from the existing benchmark adapter (data structures + helpers)
# ---------------------------------------------------------------------------

# Ensure the project root (parent of benchmarks/) is on sys.path so that
# ``from benchmarks.structmemeval_adapter import ...`` resolves when the
# script is invoked directly via ``uv run python benchmarks/temporal_blend_sweep.py``.
_PROJECT_ROOT: str = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Import data structures and helpers from the canonical adapter.
# We do NOT call its query_aelfrice() directly; we need to pass an
# explicit half_life to retrieve_v2 per sweep grid point.
from benchmarks.structmemeval_adapter import (  # type: ignore[import-untyped]
    Case,
    Message,
    Query,
    ReferenceAnswer,
    Session,
    check_state_correctness,
    discover_cases,
    ingest_case,
)
from aelfrice.retrieval import retrieve_v2
from aelfrice.store import MemoryStore


# ---------------------------------------------------------------------------
# Per-grid-point query helper
# ---------------------------------------------------------------------------


def _query_with_half_life(
    store: MemoryStore,
    question: str,
    half_life_seconds: float | None,
    budget: int = 2000,
) -> str:
    """Query aelfrice with an explicit half-life value.

    When half_life_seconds is None (representing +inf / no decay),
    temporal_sort is left False so the decay path is bypassed entirely
    (byte-identical to the pre-#473 ordering for opted-out callers).
    When a finite value is provided, temporal_sort=True applies the
    ``2 ** (-age / half_life)`` multiplicative decay against created_at.
    """
    if half_life_seconds is None:
        result = retrieve_v2(
            store=store,
            query=question,
            budget=budget,
            include_locked=False,
            use_hrr_structural=True,
            use_bfs=True,
            temporal_sort=False,
        )
    else:
        result = retrieve_v2(
            store=store,
            query=question,
            budget=budget,
            include_locked=False,
            use_hrr_structural=True,
            use_bfs=True,
            temporal_sort=True,
            temporal_half_life_seconds=half_life_seconds,
        )
    return " ".join(b.content for b in result.beliefs)


# ---------------------------------------------------------------------------
# Fixture loader
# ---------------------------------------------------------------------------


def _load_fixture_cases(task: str) -> list[Case]:
    """Return the inline fixture cases matching the given task prefix."""
    prefix = f"fixture_{task}_"
    selected = [c for c in _FIXTURE_CASES if c["case_id"].startswith(prefix)]
    if not selected:
        selected = list(_FIXTURE_CASES)  # fall back to all fixture cases

    cases: list[Case] = []
    for rc in selected:
        sessions = [
            Session(
                session_id=str(s["session_id"]),
                topic=str(s["topic"]),
                messages=[
                    Message(role=str(m["role"]), content=str(m["content"]))
                    for m in s["messages"]
                ],
            )
            for s in rc["sessions"]
        ]
        queries = [
            Query(
                question=str(q["question"]),
                reference_answer=ReferenceAnswer(
                    text=str(q["reference_answer"]["text"])
                ),
            )
            for q in rc["queries"]
        ]
        cases.append(
            Case(case_id=str(rc["case_id"]), sessions=sessions, queries=queries)
        )
    return cases


# ---------------------------------------------------------------------------
# Single-grid-point runner
# ---------------------------------------------------------------------------


@dataclass
class GridPointResult:
    half_life_label: str
    half_life_seconds: float | None   # None = inf (no decay)
    task: str
    bench: str | None
    score: float                       # accuracy in [0, 1]
    n_queries: int
    n_correct: int
    n_cases: int
    elapsed_sec: float
    is_fixture: bool = False


def run_grid_point(
    cases: list[Case],
    task: str,
    bench: str | None,
    half_life_label: str,
    half_life_seconds: float | None,
    db_dir: str,
    budget: int = 2000,
) -> GridPointResult:
    """Run all cases for one (task, half_life) combination.

    Each case gets a fresh DB to avoid cross-contamination between grid
    points; the DB name is prefixed with the label so parallel calls
    (if ever added) do not collide.
    """
    t0 = time.monotonic()
    total_queries = 0
    total_correct = 0

    for case in cases:
        db_path = f"{db_dir}/{half_life_label}_{case.case_id}.db"
        store = MemoryStore(db_path)
        ingest_case(store, case)

        for query in case.queries:
            retrieved = _query_with_half_life(
                store, query.question, half_life_seconds, budget=budget,
            )
            if check_state_correctness(retrieved, query.reference_answer):
                total_correct += 1
            total_queries += 1

    elapsed = time.monotonic() - t0
    score = total_correct / total_queries if total_queries > 0 else 0.0
    return GridPointResult(
        half_life_label=half_life_label,
        half_life_seconds=half_life_seconds,
        task=task,
        bench=bench,
        score=score,
        n_queries=total_queries,
        n_correct=total_correct,
        n_cases=len(cases),
        elapsed_sec=round(elapsed, 3),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep temporal_half_life_seconds on StructMemEval temporal tasks",
    )
    parser.add_argument(
        "--data", default="/tmp/StructMemEval/benchmark/data",
        help="Path to StructMemEval data directory (default: /tmp/StructMemEval/benchmark/data)",
    )
    parser.add_argument(
        "--task", default=None,
        choices=["location", "accounting", "all"],
        help="Task to sweep (default: all temporal tasks)",
    )
    parser.add_argument(
        "--bench", default="small",
        choices=["small", "big"],
        help="Bench size for location tasks (default: small)",
    )
    parser.add_argument(
        "--budget", type=int, default=2000,
        help="Token budget for retrieval (default: 2000)",
    )
    parser.add_argument(
        "--output", default=None,
        help=(
            "Output JSON path "
            "(default: benchmarks/results/temporal_half_life_sweep.json)"
        ),
    )
    args = parser.parse_args()

    # Determine output path
    if args.output is None:
        script_dir = Path(__file__).resolve().parent
        results_dir = script_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        output_path = results_dir / "temporal_half_life_sweep.json"
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine task list
    if args.task is None or args.task == "all":
        tasks = list(TEMPORAL_TASKS)
    else:
        tasks = [args.task]

    all_results: list[GridPointResult] = []
    is_fixture_run = False

    with tempfile.TemporaryDirectory(prefix="aelf_hl_sweep_") as tmpdir:
        for task in tasks:
            print(f"\nLoading cases: task={task}, bench={args.bench}")
            cases = discover_cases(args.data, task, args.bench)

            if not cases:
                print(
                    f"  WARNING: no corpus data found for task={task} at {args.data}; "
                    "falling back to inline fixture."
                )
                cases = _load_fixture_cases(task)
                is_fixture_run = True

            print(f"  Cases: {len(cases)}")

            for label, hl_seconds in SWEEP_GRID:
                hl_display = (
                    f"{hl_seconds:.0f}s" if hl_seconds is not None else "inf"
                )
                print(
                    f"  half_life={label} ({hl_display}) ... ",
                    end="",
                    flush=True,
                )
                result = run_grid_point(
                    cases=cases,
                    task=task,
                    bench=args.bench,
                    half_life_label=label,
                    half_life_seconds=hl_seconds,
                    db_dir=tmpdir,
                    budget=args.budget,
                )
                result.is_fixture = is_fixture_run
                all_results.append(result)
                print(
                    f"score={result.score:.4f}  "
                    f"({result.n_correct}/{result.n_queries})  "
                    f"elapsed={result.elapsed_sec:.2f}s"
                )

    # Verdict: does the 7d default stand per task?
    # Fixture-only runs cannot ratify or contradict the default — the
    # two-row fixture has no temporal-sensitivity by construction. Mark
    # fixture verdicts explicitly so a reader cannot mistake a structural
    # smoke pass for operator-quality evidence (#487).
    task_verdicts: dict[str, str] = {}
    for task in tasks:
        task_pts = [r for r in all_results if r.task == task]
        if not task_pts:
            continue
        if is_fixture_run:
            task_verdicts[task] = (
                "fixture_only — harness wires correctly; no operator-quality "
                "evidence. Rerun with real StructMemEval corpus to ratify the "
                "7-day default (clone Wangt-CN/StructMemEval and pass --data)."
            )
            continue
        default_pt = next((r for r in task_pts if r.half_life_label == "7d"), None)
        best_pt = max(task_pts, key=lambda r: r.score)
        if default_pt is None:
            task_verdicts[task] = "inconclusive"
        elif best_pt.score == default_pt.score:
            task_verdicts[task] = "supported"
        elif best_pt.score > default_pt.score + 0.01:
            task_verdicts[task] = (
                f"contradicted (best={best_pt.half_life_label}, "
                f"score={best_pt.score:.4f}; "
                f"7d={default_pt.score:.4f}; "
                f"delta={best_pt.score - default_pt.score:.4f}); "
                "file a follow-up issue to re-tune the default"
            )
        else:
            task_verdicts[task] = (
                f"supported (7d within 0.01 of best={best_pt.half_life_label}; "
                f"delta={best_pt.score - default_pt.score:.4f})"
            )

    # Build output document
    rows = [
        {
            "half_life_label": r.half_life_label,
            "half_life_seconds": r.half_life_seconds,
            "task": r.task,
            "bench": r.bench,
            "score": round(r.score, 6),
            "n_queries": r.n_queries,
            "n_correct": r.n_correct,
            "n_cases": r.n_cases,
            "elapsed_sec": r.elapsed_sec,
            "is_fixture": r.is_fixture,
        }
        for r in all_results
    ]
    output_doc = {
        "_notes": (
            "Sweep of temporal_half_life_seconds on StructMemEval temporal tasks (#487). "
            "half_life_seconds=null means no decay (temporal_sort=False, equivalent to inf). "
            "score=accuracy (fraction of queries where current state is present in retrieval). "
            "is_fixture=true when the inline two-row fixture was used instead of real corpus data."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sweep_grid": [
            {"label": label, "half_life_seconds": hl}
            for label, hl in SWEEP_GRID
        ],
        "tasks": tasks,
        "bench": args.bench if len(tasks) == 1 else None,
        "verdict": task_verdicts,
        "rows": rows,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_doc, f, indent=2)
        f.write("\n")

    print(f"\nResults written to {output_path}")

    # Summary table
    print()
    header = (
        f"{'half_life':>10}  {'task':>14}  {'score':>8}  {'n_queries':>10}  {'fixture':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(
            f"{r.half_life_label:>10}  {r.task:>14}  {r.score:>8.4f}"
            f"  {r.n_queries:>10}  {str(r.is_fixture):>8}"
        )

    print()
    for task, verdict in task_verdicts.items():
        print(f"Verdict [{task}]: {verdict}")


if __name__ == "__main__":
    main()
