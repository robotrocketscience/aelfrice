"""Context-rebuilder eval harness — skeleton.

Backs docs/context_rebuilder.md. NOT runnable as-is — the TODOs
mark integration points that fill in as the rebuilder
implementation lands across v1.2.0–v1.4.0. The shape of this file
is the contract: the spec's acceptance criteria reference these
run modes and metric names.

Run:
    uv run python benchmarks/context-rebuilder/eval_harness.py \\
        --mode threshold-sweep \\
        --corpus benchmarks/context-rebuilder/eval_corpus/ \\
        --out benchmarks/context-rebuilder/results/sweep_$(date +%Y%m%d).json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal


Mode = Literal["threshold-sweep", "budget-sweep", "dynamic", "regression"]


@dataclass(frozen=True)
class TranscriptCase:
    """One replayable transcript fixture."""
    path: Path
    task_type: str  # "debug" | "plan" | "review" | "explore"
    fork_turn: int  # midpoint turn at which to force the clear
    eval_turns: tuple[int, ...]  # turns whose answers we score


@dataclass
class RunResult:
    """One eval run on one transcript at one config."""
    case_path: str
    task_type: str
    config: dict
    fidelity: float  # 0.0 to 1.0
    token_cost_ratio: float
    rebuild_latency_ms: float
    fork_turn: int
    n_eval_turns: int
    failures: list[dict] = field(default_factory=list)


@dataclass
class SweepResult:
    """Aggregated result of a sweep across configs."""
    mode: Mode
    runs: list[RunResult]
    summary: dict = field(default_factory=dict)


def load_corpus(corpus_dir: Path) -> list[TranscriptCase]:
    """Load `eval_corpus/*.jsonl` and the matching `*.meta.json`.

    Each fixture has two files:
      - turns.jsonl — the captured transcript
      - turns.meta.json — task_type, fork_turn, eval_turns
    """
    cases: list[TranscriptCase] = []
    for jsonl in sorted(corpus_dir.glob("*.jsonl")):
        meta_path = jsonl.with_suffix(".meta.json")
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        cases.append(
            TranscriptCase(
                path=jsonl,
                task_type=meta["task_type"],
                fork_turn=meta["fork_turn"],
                eval_turns=tuple(meta["eval_turns"]),
            )
        )
    return cases


def replay_to_fork(case: TranscriptCase) -> "object":
    """Populate a fresh aelfrice store with turns 0..fork_turn-1.

    Returns the store handle. TODO: wire to MemoryStore + ingest_jsonl
    once docs/transcript_ingest.md ships.
    """
    raise NotImplementedError("Wire to ingest_jsonl from transcript_ingest spec")


def run_rebuilder(
    store: object,
    case: TranscriptCase,
    *,
    trigger_threshold: float,
    token_budget: int,
) -> tuple[str, float]:
    """Run the rebuilder against the store at the fork point.

    Returns (rebuilt_context_block, latency_ms). TODO: wire to
    aelfrice.context_rebuilder.rebuild() once context_rebuilder spec
    ships an implementation.
    """
    raise NotImplementedError("Wire to context_rebuilder.rebuild()")


def replay_post_fork(
    rebuilt_context: str,
    case: TranscriptCase,
) -> list[dict]:
    """Replay turns fork_turn..end with rebuilt_context as session start.

    Returns one dict per eval_turn with keys:
      {"turn_idx": int, "expected": str, "actual": str, "matched": bool}.

    TODO: wire to a Claude API client once we decide on the eval
    invocation surface. Initially can be agent-API-driven; later may
    move to a Claude-Code-harness-equivalent for higher fidelity.
    """
    raise NotImplementedError("Wire to model invocation client")


def score_fidelity(replay_results: list[dict]) -> float:
    """Fraction of eval_turns where actual matches expected.

    For string-match turns, exact substring or normalized-token
    match. For open-ended turns, defer to an LLM judge (see
    judges/llm_judge.py — TODO).
    """
    if not replay_results:
        return 0.0
    matched = sum(1 for r in replay_results if r["matched"])
    return matched / len(replay_results)


def measure_token_cost(
    rebuilt_context: str,
    case: TranscriptCase,
) -> float:
    """Rebuild block size / pre-clear context size.

    TODO: needs a tokenizer (tiktoken or model-specific) and the
    pre-clear context size from the captured transcript.
    """
    raise NotImplementedError("Wire to tokenizer + transcript size")


def run_one(
    case: TranscriptCase,
    *,
    trigger_threshold: float,
    token_budget: int,
) -> RunResult:
    """End-to-end one run on one case at one config."""
    store = replay_to_fork(case)
    t0 = time.monotonic()
    rebuilt, latency_ms = run_rebuilder(
        store,
        case,
        trigger_threshold=trigger_threshold,
        token_budget=token_budget,
    )
    replay = replay_post_fork(rebuilt, case)
    fidelity = score_fidelity(replay)
    cost_ratio = measure_token_cost(rebuilt, case)
    return RunResult(
        case_path=str(case.path),
        task_type=case.task_type,
        config={
            "trigger_threshold": trigger_threshold,
            "token_budget": token_budget,
        },
        fidelity=fidelity,
        token_cost_ratio=cost_ratio,
        rebuild_latency_ms=latency_ms,
        fork_turn=case.fork_turn,
        n_eval_turns=len(replay),
        failures=[r for r in replay if not r["matched"]],
    )


def sweep_thresholds(
    cases: list[TranscriptCase],
    *,
    thresholds: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9),
    token_budget: int = 2000,
) -> SweepResult:
    runs = [
        run_one(case, trigger_threshold=t, token_budget=token_budget)
        for case in cases
        for t in thresholds
    ]
    summary = _summarize_by_threshold(runs)
    return SweepResult(mode="threshold-sweep", runs=runs, summary=summary)


def sweep_budgets(
    cases: list[TranscriptCase],
    *,
    budgets: tuple[int, ...] = (500, 1000, 2000, 4000),
    trigger_threshold: float = 0.7,
) -> SweepResult:
    runs = [
        run_one(case, trigger_threshold=trigger_threshold, token_budget=b)
        for case in cases
        for b in budgets
    ]
    summary = _summarize_by_budget(runs)
    return SweepResult(mode="budget-sweep", runs=runs, summary=summary)


def _summarize_by_threshold(runs: list[RunResult]) -> dict:
    """Group runs by threshold and report median fidelity per task type."""
    out: dict = {}
    by_task = _group(runs, key=lambda r: r.task_type)
    for task, task_runs in by_task.items():
        by_t = _group(task_runs, key=lambda r: r.config["trigger_threshold"])
        out[task] = {
            f"threshold={t}": {
                "median_fidelity": statistics.median(r.fidelity for r in rs),
                "median_token_cost_ratio": statistics.median(
                    r.token_cost_ratio for r in rs
                ),
                "p99_latency_ms": _p99([r.rebuild_latency_ms for r in rs]),
                "n_runs": len(rs),
            }
            for t, rs in by_t.items()
        }
    return out


def _summarize_by_budget(runs: list[RunResult]) -> dict:
    out: dict = {}
    by_task = _group(runs, key=lambda r: r.task_type)
    for task, task_runs in by_task.items():
        by_b = _group(task_runs, key=lambda r: r.config["token_budget"])
        out[task] = {
            f"budget={b}": {
                "median_fidelity": statistics.median(r.fidelity for r in rs),
                "p99_latency_ms": _p99([r.rebuild_latency_ms for r in rs]),
                "n_runs": len(rs),
            }
            for b, rs in by_b.items()
        }
    return out


def _group(items, *, key):
    out: dict = {}
    for it in items:
        out.setdefault(key(it), []).append(it)
    return out


def _p99(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    idx = max(0, int(round(0.99 * (len(s) - 1))))
    return s[idx]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("threshold-sweep", "budget-sweep", "dynamic", "regression"), required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    cases = load_corpus(args.corpus)
    if not cases:
        print(f"no cases found in {args.corpus}")
        return 1

    if args.mode == "threshold-sweep":
        result = sweep_thresholds(cases)
    elif args.mode == "budget-sweep":
        result = sweep_budgets(cases)
    elif args.mode == "dynamic":
        raise NotImplementedError("dynamic mode lands once the heuristic exists")
    elif args.mode == "regression":
        raise NotImplementedError("regression mode lands once a v1.0.0 baseline is recorded")
    else:
        raise AssertionError(args.mode)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "mode": result.mode,
        "n_cases": len(cases),
        "summary": result.summary,
        "runs": [asdict(r) for r in result.runs],
    }, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
