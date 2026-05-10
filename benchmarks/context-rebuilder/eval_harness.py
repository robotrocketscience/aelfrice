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
from typing import Final, Literal

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
from benchmarks.context_rebuilder.measure import estimate_tokens  # noqa: E402


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


REPLAY_PENDING_REASON: Final[str] = "needs_replay_client"
"""Marker emitted by `replay_post_fork` when no `run_dir` is given (the
v1.2.0 stub default). Surfaced in `failures[].reason` so summary readers
can distinguish "skipped, judge required" from a real fidelity miss."""

PENDING_REPLAY_REASON: Final[str] = "pending_replay"
"""Marker for a row whose request is now persisted to
`<run_dir>/replay_requests.jsonl` but whose response has not yet been
filled in. Distinct from `needs_replay_client`: pending_replay means
"requests written, awaiting host-agent dispatch + replay_responses.jsonl"."""

NEEDS_LLM_JUDGE_REASON: Final[str] = "needs_llm_judge"
"""Marker for a row whose response is filled in but the substring/token
match fails. The open-ended fidelity verdict belongs to commit-3 of
#592 (LLM judge). Substring success short-circuits to matched=True."""

REPLAY_REQUESTS_FILENAME: Final[str] = "replay_requests.jsonl"
REPLAY_RESPONSES_FILENAME: Final[str] = "replay_responses.jsonl"


def _read_user_and_expected(
    case: TranscriptCase,
) -> tuple[dict[int, str], dict[int, str]]:
    """Walk `case.path` once. Return (expected_by_idx, user_turn_by_idx).

    `expected_by_idx[idx]` is the `text` at line `idx` (the eval target).
    `user_turn_by_idx[idx]` is the most recent user-role `text` at-or-
    before line `idx` — the prompt the harness will replay through the
    rebuilt context to produce `actual`.
    """
    expected_by_idx: dict[int, str] = {}
    user_turn_by_idx: dict[int, str] = {}
    if not case.eval_turns:
        return expected_by_idx, user_turn_by_idx
    wanted = set(case.eval_turns)
    last_user_text = ""
    try:
        with case.path.open("r", encoding="utf-8") as f:
            for i, raw in enumerate(f):
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                text = obj.get("text") if isinstance(obj.get("text"), str) else ""
                if obj.get("role") == "user":
                    last_user_text = text
                if i in wanted:
                    expected_by_idx[i] = text
                    user_turn_by_idx[i] = last_user_text
    except OSError:
        return {}, {}
    return expected_by_idx, user_turn_by_idx


def _read_replay_responses(run_dir: Path) -> dict[int, str]:
    """Read `<run_dir>/replay_responses.jsonl`, one row per turn_idx.

    Each row is `{"turn_idx": int, "actual": str}`. Missing file → {};
    malformed lines are skipped silently. The host-agent dispatcher
    (operator-driven, not aelfrice) is the writer; this function is
    the read half of the polymorphic eval-replay split.
    """
    out: dict[int, str] = {}
    path = run_dir / REPLAY_RESPONSES_FILENAME
    if not path.is_file():
        return out
    try:
        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                idx = obj.get("turn_idx")
                actual = obj.get("actual")
                if isinstance(idx, int) and isinstance(actual, str):
                    out[idx] = actual
    except OSError:
        return {}
    return out


def _write_replay_requests(
    run_dir: Path,
    rebuilt_context: str,
    eval_turns: tuple[int, ...],
    expected_by_idx: dict[int, str],
    user_turn_by_idx: dict[int, str],
) -> None:
    """Write `<run_dir>/replay_requests.jsonl`. One row per eval_turn.

    Schema: `{turn_idx, rebuilt_block, user_turn, expected}`. Best-effort
    — `mkdir` and write failures squash to a no-op so the rest of
    `run_one` (latency, token cost) still completes.
    """
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / REPLAY_REQUESTS_FILENAME
        with path.open("w", encoding="utf-8") as f:
            for idx in eval_turns:
                row = {
                    "turn_idx": idx,
                    "rebuilt_block": rebuilt_context,
                    "user_turn": user_turn_by_idx.get(idx, ""),
                    "expected": expected_by_idx.get(idx, ""),
                }
                f.write(json.dumps(row) + "\n")
    except OSError:
        return


def replay_post_fork(
    rebuilt_context: str,
    case: TranscriptCase,
    *,
    run_dir: Path | None = None,
) -> list[dict]:
    """Replay turns fork_turn..end with rebuilt_context as session start.

    Two modes, selected by `run_dir`:

    **No `run_dir` (v1.2.0 stub default).** Returns one placeholder dict
    per `case.eval_turns` with `actual=""`, `matched=False`,
    `reason=REPLAY_PENDING_REASON`. Threshold/budget-sweep modes get
    valid latency + token-cost numbers without dispatching child tasks.

    **With `run_dir` (host-agent eval-replay, #600).** Writes
    `<run_dir>/replay_requests.jsonl` so an operator-driven dispatcher
    (an MCP-enabled host CLI, an operator loop, or a private
    `aelf:replay-eval` skill) can spawn one child task per row,
    prompted with `rebuilt_block + "\\n---\\n" + user_turn`, expecting
    the dispatcher to write `<run_dir>/replay_responses.jsonl`. On the next harness invocation
    this function reads that response file, joins by `turn_idx`, and
    fills `actual`:

      * `actual` empty / row missing  → matched=False, reason=PENDING_REPLAY_REASON.
      * `expected.lower() in actual.lower()` → matched=True (substring half
        of the fidelity verdict).
      * `actual` filled but no substring match → matched=False,
        reason=NEEDS_LLM_JUDGE_REASON (open-ended; commit-3 of #592 LLM
        judge picks this up).

    `expected` is pulled from the captured transcript's `text` field at
    the eval-turn line index (matching the fork_turn convention). The
    aelfrice repo never imports a vendor model SDK on this path; the
    model call is the host's responsibility.
    """
    expected_by_idx, user_turn_by_idx = _read_user_and_expected(case)
    if case.eval_turns and not expected_by_idx and not user_turn_by_idx:
        # _read_user_and_expected returned ({}, {}) only on OSError.
        # Mirror the historical behaviour for unreadable files.
        return []

    responses_by_idx: dict[int, str] = {}
    if run_dir is not None:
        if case.eval_turns:
            _write_replay_requests(
                run_dir,
                rebuilt_context,
                case.eval_turns,
                expected_by_idx,
                user_turn_by_idx,
            )
        responses_by_idx = _read_replay_responses(run_dir)

    out: list[dict] = []
    for idx in case.eval_turns:
        expected = expected_by_idx.get(idx, "")
        actual = responses_by_idx.get(idx, "")
        if run_dir is None:
            reason = REPLAY_PENDING_REASON
            matched = False
        elif not actual:
            reason = PENDING_REPLAY_REASON
            matched = False
        elif expected and expected.lower() in actual.lower():
            reason = ""
            matched = True
        else:
            reason = NEEDS_LLM_JUDGE_REASON
            matched = False
        out.append(
            {
                "turn_idx": idx,
                "expected": expected,
                "actual": actual,
                "matched": matched,
                "reason": reason,
            }
        )
    return out


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


def _pre_clear_text(case: TranscriptCase) -> str:
    """Concatenate the `text` fields of every turn in 0..fork_turn-1.

    This is the harness's stand-in for "the context the agent had open
    just before the clear" — a deterministic, model-agnostic measure
    that doesn't require replaying the transcript through a model.
    """
    parts: list[str] = []
    if case.fork_turn <= 0:
        return ""
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
            if isinstance(obj, dict):
                t = obj.get("text")
                if isinstance(t, str):
                    parts.append(t)
    return "\n".join(parts)


def measure_token_cost(
    rebuilt_context: str,
    case: TranscriptCase,
) -> float:
    """Rebuild-block tokens / pre-clear-context tokens.

    Both sides use `benchmarks.context_rebuilder.measure.estimate_tokens`
    — the 4-chars-per-token heuristic that mirrors the rebuilder's own
    internal estimator (`aelfrice.context_rebuilder._CHARS_PER_TOKEN`).
    Sharing the constant keeps harness measurements aligned with the
    rebuilder's own budget bookkeeping. A real tokenizer (tiktoken /
    sentencepiece) would land alongside the LLM-judge in a follow-up.

    Returns 0.0 when the pre-clear text is empty (no transcript content
    pre-fork) — there's no meaningful ratio to report and division would
    fail; 0.0 documents the corner without raising.
    """
    pre_clear_tokens = estimate_tokens(_pre_clear_text(case))
    if pre_clear_tokens <= 0:
        return 0.0
    rebuilt_tokens = estimate_tokens(rebuilt_context)
    return rebuilt_tokens / pre_clear_tokens


def _run_subdir(
    base: Path | None,
    case: TranscriptCase,
    trigger_threshold: float,
    token_budget: int,
) -> Path | None:
    """Compute a deterministic per-(case, config) subdirectory under `base`.

    Returns None if `base` is None — replay_post_fork falls back to its
    placeholder mode in that case. The slug embeds threshold + budget so
    a sweep produces one request file per (case, threshold, budget) cell
    rather than overwriting a single shared file.
    """
    if base is None:
        return None
    return base / f"{case.path.stem}__t{trigger_threshold}__b{token_budget}"


def run_one(
    case: TranscriptCase,
    *,
    trigger_threshold: float,
    token_budget: int,
    run_dir: Path | None = None,
) -> RunResult:
    """End-to-end one run on one case at one config.

    When `run_dir` is provided, `replay_post_fork` writes a per-run
    `replay_requests.jsonl` under `<run_dir>/<case_stem>__t<threshold>__b<budget>/`
    and joins any matching `replay_responses.jsonl`. Non-None default
    is opt-in; bare callers (the v1.2.0 stub path) get unchanged
    behaviour.
    """
    store = replay_to_fork(case)
    t0 = time.monotonic()
    rebuilt, latency_ms = run_rebuilder(
        store,
        case,
        trigger_threshold=trigger_threshold,
        token_budget=token_budget,
    )
    case_run_dir = _run_subdir(run_dir, case, trigger_threshold, token_budget)
    replay = replay_post_fork(rebuilt, case, run_dir=case_run_dir)
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
    run_dir: Path | None = None,
) -> SweepResult:
    runs = [
        run_one(
            case,
            trigger_threshold=t,
            token_budget=token_budget,
            run_dir=run_dir,
        )
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
    run_dir: Path | None = None,
) -> SweepResult:
    runs = [
        run_one(
            case,
            trigger_threshold=trigger_threshold,
            token_budget=b,
            run_dir=run_dir,
        )
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
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Optional base directory under which replay_post_fork writes "
            "per-(case, config) replay_requests.jsonl and reads "
            "replay_responses.jsonl. Without this flag the harness emits "
            "placeholder rows (reason=needs_replay_client) and skips file "
            "IO. See benchmarks/context-rebuilder/README.md for the "
            "host-agent eval-replay flow (#600)."
        ),
    )
    args = p.parse_args()

    cases = load_corpus(args.corpus)
    if not cases:
        print(f"no cases found in {args.corpus}")
        return 1

    if args.mode == "threshold-sweep":
        result = sweep_thresholds(cases, run_dir=args.run_dir)
    elif args.mode == "budget-sweep":
        result = sweep_budgets(cases, run_dir=args.run_dir)
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
