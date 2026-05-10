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
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

# Allow `python benchmarks/context-rebuilder/eval_harness.py` to import
# aelfrice and the benchmarks.context_rebuilder package. The hyphenated
# parent directory blocks `python -m`, so a sys.path shim is the
# alternative. Tests already get the repo root from pytest's testpaths
# config and skip this branch as a no-op.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aelfrice.context_rebuilder import (  # noqa: E402
    DEFAULT_N_RECENT_TURNS,
    DEFAULT_REBUILDER_TOKEN_BUDGET,
    RecentTurn,
    rebuild_v14,
)
from aelfrice.ingest import ingest_jsonl  # noqa: E402
from aelfrice.store import MemoryStore  # noqa: E402


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


def replay_to_fork(case: TranscriptCase) -> MemoryStore:
    """Populate a fresh in-memory aelfrice store with turns
    0..fork_turn-1 from `case.path`.

    Returns the store handle. Caller is responsible for closing it.

    Lines are counted by file order — this matches the synthetic-fixture
    convention where each line is one user/assistant turn. Compaction
    markers and tool-result lines are passed through to ingest_jsonl,
    which already skips them; they don't shift the fork boundary
    relative to the underlying transcript file.

    Implementation note: ingest_jsonl reads a file end-to-end, so we
    write the first `fork_turn` lines to a tempfile and ingest that.
    Modifying ingest_jsonl to take a line-count cap would couple the
    production ingest path to harness internals; keeping the truncation
    here is the lighter coupling.
    """
    store = MemoryStore(":memory:")
    if case.fork_turn <= 0:
        return store
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as tmp:
        tmp_path = Path(tmp.name)
        with case.path.open("r", encoding="utf-8") as src:
            for i, line in enumerate(src):
                if i >= case.fork_turn:
                    break
                tmp.write(line)
    try:
        ingest_jsonl(store, tmp_path, source_label="eval-harness")
    finally:
        tmp_path.unlink(missing_ok=True)
    return store


def _recent_turns_pre_fork(
    case: TranscriptCase, n: int = DEFAULT_N_RECENT_TURNS,
) -> list[RecentTurn]:
    """Build the last `n` RecentTurn records from turns 0..fork_turn-1.

    Mirrors the production hook's adapter: each line is a JSON object
    with `role` and `text`, plus an optional `session_id`. Lines that
    don't carry role/text are skipped (compaction markers, malformed).
    """
    turns: list[RecentTurn] = []
    if case.fork_turn <= 0:
        return turns
    with case.path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= case.fork_turn:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            role = obj.get("role")
            text = obj.get("text")
            if not isinstance(role, str) or not isinstance(text, str):
                continue
            sid = obj.get("session_id")
            turns.append(
                RecentTurn(
                    role=role, text=text,
                    session_id=sid if isinstance(sid, str) else None,
                )
            )
    return turns[-n:]


def run_rebuilder(
    store: MemoryStore,
    case: TranscriptCase,
    *,
    trigger_threshold: float,
    token_budget: int,
) -> tuple[str, float]:
    """Run the v1.4 rebuilder against the store at the fork point.

    Returns `(rebuilt_context_block, latency_ms)`. Latency is measured
    via `time.monotonic()` so it stays monotonic by construction.

    `trigger_threshold` is wired to `rebuild_v14`'s per-lane composite-
    score floors (`floor_session` and `floor_l1`). The v1.7 floor knob
    is the closest existing rebuild-time parameter to the harness's
    "threshold-sweep" axis; sweeping it answers "at what relevance
    floor does the rebuilder still pack useful content for this task?"
    L0 locked beliefs are unaffected by the floor — they always pack
    per the documented v1.4 contract.

    `token_budget` is forwarded directly to `rebuild_v14`.
    """
    recent = _recent_turns_pre_fork(case)
    t0 = time.monotonic()
    rebuilt = rebuild_v14(
        recent, store,
        token_budget=token_budget,
        floor_session=trigger_threshold,
        floor_l1=trigger_threshold,
    )
    latency_ms = (time.monotonic() - t0) * 1000.0
    return rebuilt, latency_ms


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
