"""StructMemEval benchmark adapter for aelfrice.

Ingests StructMemEval multi-session conversations into aelfrice,
runs retrieval against state-tracking queries, and reports results.

StructMemEval (Shutova et al., 2026) tests whether memory systems
correctly track evolving state across sessions: location changes,
accounting ledgers, preference updates, and hierarchical knowledge.

Usage:
    uv run python benchmarks/structmemeval_adapter.py [--data PATH] [--task location] [--bench small]
    uv run python benchmarks/structmemeval_adapter.py --retrieve-only results.json --task accounting
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from aelfrice.ingest import ingest_turn
from aelfrice.retrieval import retrieve_v2 as retrieve  # v1.0.x lab-compat shim
from aelfrice.store import MemoryStore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DATA_PATH: Final[str] = "/tmp/StructMemEval/benchmark/data"

TASK_DIRS: Final[dict[str, str]] = {
    "location": "state_machine_location",
    "accounting": "accounting",
    "recommendations": "recommendations",
    "tree": "tree_based",
}

BENCH_SUBDIRS: Final[dict[str, str]] = {
    "small": "small_bench",
    "big": "big_bench",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Message:
    """A single message in a session."""

    role: str
    content: str


@dataclass
class Session:
    """One session within a case."""

    session_id: str
    topic: str
    messages: list[Message]


@dataclass
class ReferenceAnswer:
    """The reference answer for a query, including exclusion notes."""

    text: str


@dataclass
class Query:
    """A query to evaluate against the memory store."""

    question: str
    reference_answer: ReferenceAnswer


@dataclass
class Case:
    """One benchmark case with sessions and queries."""

    case_id: str
    sessions: list[Session]
    queries: list[Query]
    source_path: str = ""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_case(path: Path) -> Case:
    """Load a single case from a JSON file."""
    with path.open("r", encoding="utf-8") as f:
        raw: dict[str, object] = json.load(f)

    case_id: str = str(raw.get("case_id", path.stem))

    # Parse sessions
    raw_sessions: list[dict[str, object]] = raw.get("sessions", [])  # type: ignore[assignment]
    sessions: list[Session] = []
    for rs in raw_sessions:
        raw_messages: list[dict[str, str]] = rs.get("messages", [])  # type: ignore[assignment]
        messages: list[Message] = [
            Message(role=m.get("role", "user"), content=m.get("content", ""))
            for m in raw_messages
        ]
        sessions.append(Session(
            session_id=str(rs.get("session_id", "")),
            topic=str(rs.get("topic", "")),
            messages=messages,
        ))

    # Parse queries
    raw_queries: list[dict[str, object]] = raw.get("queries", [])  # type: ignore[assignment]
    queries: list[Query] = []
    for rq in raw_queries:
        raw_ref: object = rq.get("reference_answer", {})
        if isinstance(raw_ref, list):
            # Accounting tasks: list of dicts with "text" keys
            ref_items: list[object] = list(raw_ref)  # type: ignore[arg-type]
            ref_texts: list[str] = [
                str(r.get("text", ""))  # type: ignore[union-attr]
                for r in ref_items
                if isinstance(r, dict)
            ]
            ref = ReferenceAnswer(text=" | ".join(ref_texts))
        elif isinstance(raw_ref, dict):
            ref = ReferenceAnswer(text=str(raw_ref.get("text", "")))  # type: ignore[union-attr]
        else:
            ref = ReferenceAnswer(text=str(raw_ref))
        queries.append(Query(
            question=str(rq.get("question", "")),
            reference_answer=ref,
        ))

    return Case(
        case_id=case_id,
        sessions=sessions,
        queries=queries,
        source_path=str(path),
    )


def discover_cases(data_dir: str, task: str, bench: str | None = None) -> list[Case]:
    """Discover and load all case files for a given task and bench size."""
    base: Path = Path(data_dir)
    task_dir_name: str = TASK_DIRS.get(task, task)
    task_path: Path = base / task_dir_name

    if not task_path.exists():
        print(f"WARNING: Task directory not found: {task_path}")
        return []

    # For location tasks, select bench subdirectory
    if task == "location" and bench is not None:
        subdir_name: str = BENCH_SUBDIRS.get(bench, bench)
        task_path = task_path / subdir_name
        if not task_path.exists():
            print(f"WARNING: Bench directory not found: {task_path}")
            return []

    # Collect all JSON files recursively
    json_files: list[Path] = sorted(task_path.rglob("*.json"))
    if not json_files:
        print(f"WARNING: No JSON files found in {task_path}")
        return []

    cases: list[Case] = []
    for jf in json_files:
        try:
            case: Case = _load_case(jf)
            cases.append(case)
        except (json.JSONDecodeError, KeyError) as exc:
            print(f"WARNING: Skipping {jf}: {exc}")

    return cases


# ---------------------------------------------------------------------------
# Ingest adapter
# ---------------------------------------------------------------------------


def _synthetic_timestamp(session_idx: int, total_sessions: int) -> str:
    """Generate synthetic narrative timestamps for sessions.

    Spaces sessions 30 days apart so temporal decay can distinguish
    old sessions from recent ones. The last session gets "now" (no decay),
    the first session gets (total_sessions - 1) * 30 days ago.
    """
    from datetime import timedelta
    now: datetime = datetime.now(timezone.utc)
    months_ago: int = total_sessions - 1 - session_idx
    dt: datetime = now - timedelta(days=months_ago * 30)
    return dt.isoformat()


def ingest_case(store: MemoryStore, case: Case) -> int:
    """Ingest all sessions of a case into aelfrice sequentially.

    Sessions are assigned synthetic timestamps 30 days apart so that
    temporal decay can distinguish old sessions from recent ones.
    Returns total messages ingested.
    """
    total_messages: int = 0
    total_sessions: int = len(case.sessions)
    for session_idx, session in enumerate(case.sessions):
        narrative_ts: str = _synthetic_timestamp(session_idx, total_sessions)

        am_session = store.create_session(
            model="structmemeval-benchmark",
            project_context=f"{case.case_id} / {session.session_id}: {session.topic}",
        )

        # Ingest topic marker for session context
        topic_marker: str = f"[Session {session.session_id}: {session.topic}]"
        ingest_turn(
            store=store,
            text=topic_marker,
            source="structmemeval",
            session_id=am_session.id,
            source_id=f"{case.case_id}:{session.session_id}:topic",
            created_at=narrative_ts,
        )

        for idx, msg in enumerate(session.messages):
            role_label: str = "User" if msg.role == "user" else "Assistant"
            text: str = f"[{session.topic}] {role_label}: {msg.content}"
            ingest_turn(
                store=store,
                text=text,
                source="structmemeval",
                session_id=am_session.id,
                source_id=f"{case.case_id}:{session.session_id}:{idx}",
                created_at=narrative_ts,
            )
            total_messages += 1

        store.complete_session(am_session.id)

    return total_messages


# ---------------------------------------------------------------------------
# Query adapter
# ---------------------------------------------------------------------------


def query_aelfrice(store: MemoryStore, question: str, budget: int = 2000) -> str:
    """Query aelfrice and return retrieved belief content.

    Uses temporal_sort=True (#473): post-rerank multiplicative half-life
    decay on `created_at`, so recent beliefs are surfaced over older
    historical states without a hard `created_at DESC` sort. Locked
    (L0) beliefs stay pinned at the head and are not re-ordered.
    """
    result = retrieve(
        store=store,
        query=question,
        budget=budget,
        include_locked=False,
        use_bfs=True,
        temporal_sort=True,
    )
    parts: list[str] = [b.content for b in result.beliefs]
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def check_state_correctness(retrieved: str, reference: ReferenceAnswer) -> bool:
    """Check whether retrieved context contains the current state.

    The reference_answer.text may contain notes about what should NOT
    appear. We check that the expected answer substring is present in
    the retrieved text (case-insensitive).

    Returns True if the current state is found in retrieval.
    """
    ref_lower: str = reference.text.lower()
    retrieved_lower: str = retrieved.lower()

    # The reference text is the expected answer. Check presence.
    # Strip common prefixes like "The user is in" to get the core answer.
    # For a simple heuristic, check if any significant word from the
    # reference appears in the retrieved content.
    ref_words: list[str] = [
        w for w in ref_lower.split()
        if len(w) > 3 and w not in {"the", "that", "this", "with", "from", "should", "their", "about"}
    ]

    if not ref_words:
        return False

    # Count how many significant reference words appear in retrieval
    matches: int = sum(1 for w in ref_words if w in retrieved_lower)
    threshold: float = 0.5
    return (matches / len(ref_words)) >= threshold


# ---------------------------------------------------------------------------
# Task-aware prediction scoring (#899)
#
# `check_state_correctness` above is a single word-overlap heuristic over
# the *whole* reference text. It is invalid for scoring a reader's
# prediction for three independent reasons, one per task:
#
#   * location: the reference embeds an exclusion note ("should NOT
#     mention <previous cities>"). Those distractor tokens land in the
#     word-set, so raw retrieval (which contains every stale city)
#     matches them and scores high, while a concise *correct* answer
#     that omits them falls below the 0.5 threshold and scores 0.
#   * tree: the answer is binary (Yes/No) wrapped in boilerplate
#     ("...indirect colleagues according to their graph relations"). A
#     correct "No" prediction shares almost none of the boilerplate
#     word-set, so polarity is never actually tested.
#   * accounting: the reference is a set of settlement tuples
#     ("Charlie -> Bob : 218.33 EUR | ..."); word overlap on names and
#     bare numbers does not check that the amounts and directions match.
#
# The functions below score a *prediction* (not raw retrieval) with
# task-appropriate logic. `score_prediction` dispatches by task name.
# ---------------------------------------------------------------------------

_LOC_EXCLUSION_MARKERS: Final[tuple[str, ...]] = (
    "should not mention",
    "not mention",
    "do not mention",
)
_STOPWORDS: Final[frozenset[str]] = frozenset({
    "the", "that", "this", "with", "from", "should", "their", "about",
    "your", "since", "currently", "live", "answer", "previous", "cities",
    "city", "activities", "activity", "neighbor", "neighbors", "mention",
    "based", "where", "now", "and", "for", "you",
})


def _significant_words(text: str) -> list[str]:
    return [
        w.strip(".,()") for w in text.lower().split()
        if len(w.strip(".,()")) > 3 and w.strip(".,()") not in _STOPWORDS
    ]


def _split_location_reference(reference_text: str) -> tuple[str, str]:
    """Split a location reference into (expected_portion, exclusion_portion).

    The exclusion portion is everything from the first exclusion marker
    onward; the expected portion is the text before it. When no marker
    is present the whole reference is the expected portion and the
    exclusion portion is empty.
    """
    low: str = reference_text.lower()
    for marker in _LOC_EXCLUSION_MARKERS:
        idx: int = low.find(marker)
        if idx != -1:
            return reference_text[:idx], reference_text[idx:]
    return reference_text, ""


def score_location_prediction(
    prediction: str, reference_text: str, present_threshold: float = 0.5,
) -> bool:
    """Correct iff the prediction reflects the current state AND omits
    the excluded (stale) distractors.

    * expected_present: at least `present_threshold` of the significant
      words in the expected portion appear in the prediction.
    * exclusion_clean: no significant word that appears ONLY in the
      exclusion portion (i.e. a distractor not also part of the current
      answer) appears in the prediction.

    Both conditions must hold. The exclusion-clean check is the fix that
    `check_state_correctness` lacked.
    """
    expected_text, exclusion_text = _split_location_reference(reference_text)
    pred_lower: str = prediction.lower()
    expected_words: list[str] = _significant_words(expected_text)
    if not expected_words:
        return False
    present: int = sum(1 for w in expected_words if w in pred_lower)
    expected_present: bool = (present / len(expected_words)) >= present_threshold
    expected_set: set[str] = set(expected_words)
    exclusion_only: set[str] = {
        w for w in _significant_words(exclusion_text) if w not in expected_set
    }
    leaked: bool = any(w in pred_lower for w in exclusion_only)
    return expected_present and not leaked


def _polarity(text: str) -> str | None:
    """Return 'yes' / 'no' from the first polarity token, else None."""
    for tok in text.lower().replace(",", " ").replace(".", " ").split():
        if tok == "yes":
            return "yes"
        if tok == "no":
            return "no"
    return None


def score_tree_prediction(prediction: str, reference_text: str) -> bool:
    """Correct iff the prediction's Yes/No polarity matches the reference.

    Tree answers are binary connectivity verdicts wrapped in boilerplate;
    only the polarity is load-bearing.
    """
    ref_pol: str | None = _polarity(reference_text)
    pred_pol: str | None = _polarity(prediction)
    return ref_pol is not None and ref_pol == pred_pol


# Settlement transaction parsing for the accounting task. Two forms are
# accepted: the reference's "Payer -> Payee : amount EUR" and the
# question's requested answer format "Payer pays amount to Payee".
_SETTLE_ARROW_RE: Final = re.compile(
    r"([A-Za-z]+)\s*->\s*([A-Za-z]+)\s*:\s*([\d]+(?:\.[\d]+)?)",
)
_SETTLE_PAYS_RE: Final = re.compile(
    r"([A-Za-z]+)\s+pays\s+(?:€|EUR\s*)?([\d]+(?:\.[\d]+)?)\s*(?:€|EUR)?\s+to\s+([A-Za-z]+)",
    re.IGNORECASE,
)
_ACCT_AMOUNT_TOLERANCE: Final[float] = 0.5


def _parse_settlement_transactions(text: str) -> set[tuple[str, str, float]]:
    """Parse settlement transactions into a set of (payer, payee, amount).

    Accepts both the reference arrow form ("Alice -> Bob : 91.00 EUR") and
    the requested answer form ("Alice pays 91 to Bob"). Names are
    lower-cased; amounts are floats. A directed (payer, payee) pair plus
    amount uniquely keys a transaction.
    """
    out: set[tuple[str, str, float]] = set()
    for payer, payee, amt in _SETTLE_ARROW_RE.findall(text):
        out.add((payer.lower(), payee.lower(), float(amt)))
    for payer, amt, payee in _SETTLE_PAYS_RE.findall(text):
        out.add((payer.lower(), payee.lower(), float(amt)))
    return out


def _transaction_sets_match(
    pred: set[tuple[str, str, float]],
    ref: set[tuple[str, str, float]],
    tolerance: float = _ACCT_AMOUNT_TOLERANCE,
) -> bool:
    """True iff pred and ref have the same directed transactions with
    amounts within `tolerance`. Same cardinality required (no extra or
    missing transactions)."""
    if not ref or len(pred) != len(ref):
        return False
    remaining: list[tuple[str, str, float]] = list(pred)
    for r_payer, r_payee, r_amt in ref:
        match_idx: int | None = None
        for i, (p_payer, p_payee, p_amt) in enumerate(remaining):
            if (
                p_payer == r_payer
                and p_payee == r_payee
                and abs(p_amt - r_amt) <= tolerance
            ):
                match_idx = i
                break
        if match_idx is None:
            return False
        remaining.pop(match_idx)
    return True


def score_accounting_prediction(prediction: str, reference_text: str) -> bool:
    """Correct iff the prediction's settlement transactions match ANY of
    the acceptable reference alternatives.

    The reference is a `" | "`-joined list of alternative settlements
    (the benchmark accepts several valid resolutions of ambiguous
    split instructions). Each alternative and the prediction are parsed
    into directed (payer, payee, amount) sets; the prediction is correct
    if its set matches any alternative within an amount tolerance. This
    replaces the word-overlap heuristic, which never checked amounts or
    directions.
    """
    pred_tx: set[tuple[str, str, float]] = _parse_settlement_transactions(
        prediction,
    )
    if not pred_tx:
        return False
    for alternative in reference_text.split(" | "):
        ref_tx = _parse_settlement_transactions(alternative)
        if _transaction_sets_match(pred_tx, ref_tx):
            return True
    return False


def score_prediction(
    prediction: str, reference_text: str, task: str,
) -> bool:
    """Dispatch prediction scoring by task.

    `location`, `tree`, and `accounting` have dedicated scorers; any
    other task falls back to the legacy word-overlap heuristic (a recall
    proxy, not a correctness check).
    """
    if task == "location":
        return score_location_prediction(prediction, reference_text)
    if task == "tree":
        return score_tree_prediction(prediction, reference_text)
    if task == "accounting":
        return score_accounting_prediction(prediction, reference_text)
    return check_state_correctness(
        prediction, ReferenceAnswer(text=reference_text),
    )


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class CaseResult:
    """Result for a single benchmark case."""

    case_id: str = ""
    task: str = ""
    total_queries: int = 0
    correct_state: int = 0
    ingest_messages: int = 0
    ingest_time_s: float = 0.0
    query_time_s: float = 0.0
    per_query: list[dict[str, object]] = field(
        default_factory=lambda: list[dict[str, object]](),
    )

    @property
    def accuracy(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.correct_state / self.total_queries


@dataclass
class TaskResult:
    """Aggregated results for a task."""

    task: str = ""
    cases: list[CaseResult] = field(
        default_factory=lambda: list[CaseResult](),
    )

    @property
    def total_queries(self) -> int:
        return sum(c.total_queries for c in self.cases)

    @property
    def total_correct(self) -> int:
        return sum(c.correct_state for c in self.cases)

    @property
    def accuracy(self) -> float:
        total: int = self.total_queries
        if total == 0:
            return 0.0
        return self.total_correct / total

    @property
    def total_ingest_time(self) -> float:
        return sum(c.ingest_time_s for c in self.cases)

    @property
    def total_query_time(self) -> float:
        return sum(c.query_time_s for c in self.cases)

    @property
    def cases_with_perfect_state(self) -> int:
        return sum(1 for c in self.cases if c.accuracy == 1.0)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_case(
    case: Case,
    task: str,
    db_dir: str,
    budget: int = 2000,
) -> CaseResult:
    """Run the benchmark pipeline on one case with a fresh DB."""
    db_path: str = f"{db_dir}/{case.case_id}.db"
    store: MemoryStore = MemoryStore(db_path)

    result: CaseResult = CaseResult(case_id=case.case_id, task=task)

    # Ingest all sessions
    t0: float = time.monotonic()
    result.ingest_messages = ingest_case(store, case)
    result.ingest_time_s = time.monotonic() - t0

    # Query and evaluate
    t1: float = time.monotonic()
    for query in case.queries:
        retrieved: str = query_aelfrice(store, query.question, budget=budget)
        correct: bool = check_state_correctness(retrieved, query.reference_answer)

        result.total_queries += 1
        if correct:
            result.correct_state += 1

        result.per_query.append({
            "question": query.question,
            "reference_answer": query.reference_answer.text,
            "context": retrieved,
            "correct": correct,
        })

    result.query_time_s = time.monotonic() - t1

    return result


def print_case_result(result: CaseResult) -> None:
    """Print result for a single case."""
    status: str = "PASS" if result.accuracy == 1.0 else "FAIL"
    print(
        f"  [{status}] {result.case_id}: "
        f"{result.correct_state}/{result.total_queries} correct "
        f"({result.accuracy * 100:.0f}%) "
        f"[ingest: {result.ingest_time_s:.2f}s, query: {result.query_time_s:.2f}s]"
    )


def print_task_result(result: TaskResult) -> None:
    """Print aggregated task results with baselines."""
    print(f"\n{'=' * 60}")
    print(f"StructMemEval Results: {result.task}")
    print(f"{'=' * 60}")
    print(f"Cases:             {len(result.cases)}")
    print(f"Total queries:     {result.total_queries}")
    print(f"Correct state:     {result.total_correct}/{result.total_queries} "
          f"({result.accuracy * 100:.1f}%)")
    print(f"Perfect cases:     {result.cases_with_perfect_state}/{len(result.cases)}")
    print(f"Total ingest time: {result.total_ingest_time:.2f}s")
    print(f"Total query time:  {result.total_query_time:.2f}s")
    if result.total_queries > 0:
        avg_ms: float = result.total_query_time / result.total_queries * 1000
        print(f"Avg query latency: {avg_ms:.1f}ms")
    print()

    # Per-case breakdown
    print("Per-case results:")
    for case_result in result.cases:
        print_case_result(case_result)
    print()

    # Reference baselines from paper
    print("Reference baselines (Shutova et al., 2026):")
    print("  Vector retrieval:       degrades sharply with more state transitions")
    print("  Mem0 (without hints):   moderate, gap widens with complexity")
    print("  Mem0 (with hints):      much better across all tasks")
    print("  Key finding:            hint/no-hint gap > framework gap")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Run StructMemEval benchmark on aelfrice",
    )
    parser.add_argument(
        "--data", default=DEFAULT_DATA_PATH,
        help="Path to StructMemEval data directory (default: /tmp/StructMemEval/benchmark/data)",
    )
    parser.add_argument(
        "--task", default="location",
        choices=["location", "accounting", "recommendations", "tree"],
        help="Task type to evaluate (default: location)",
    )
    parser.add_argument(
        "--bench", default="small",
        choices=["small", "big"],
        help="Bench size for location tasks: small (14 cases) or big (42 cases)",
    )
    parser.add_argument(
        "--budget", type=int, default=2000,
        help="Token budget for retrieval (default: 2000)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Write detailed results JSON to this path",
    )
    parser.add_argument(
        "--retrieve-only", default=None, metavar="PATH",
        help="Run retrieval only, write question+context+reference for LLM judge scoring",
    )
    args: argparse.Namespace = parser.parse_args()

    print(f"Loading StructMemEval cases: task={args.task}, bench={args.bench}")
    cases: list[Case] = discover_cases(args.data, args.task, args.bench)
    print(f"Loaded {len(cases)} cases")

    if not cases:
        print(
            "No cases found. Check --data path and --task/--bench options.",
            file=sys.stderr,
        )
        sys.exit(2)

    task_result: TaskResult = TaskResult(task=args.task)

    with tempfile.TemporaryDirectory(prefix="structmemeval_") as tmpdir:
        for case in cases:
            total_msgs: int = sum(len(s.messages) for s in case.sessions)
            print(
                f"\n--- {case.case_id}: {len(case.sessions)} sessions, "
                f"{total_msgs} messages, {len(case.queries)} queries ---"
            )

            case_result: CaseResult = run_case(
                case, args.task, tmpdir, budget=args.budget,
            )
            task_result.cases.append(case_result)

            if not args.retrieve_only:
                print_case_result(case_result)

    # Retrieve-only mode: retrieval (NO answers) + separate GT file
    if args.retrieve_only:
        all_items: list[dict[str, object]] = []
        all_gt: list[dict[str, object]] = []
        for cr in task_result.cases:
            for pq in cr.per_query:
                all_items.append({
                    "case_id": cr.case_id,
                    "task": cr.task,
                    "question": pq["question"],
                    "context": pq["context"],
                })
                all_gt.append({
                    "case_id": cr.case_id,
                    "question": pq["question"],
                    "reference_answer": pq["reference_answer"],
                })
        retrieve_path: Path = Path(args.retrieve_only)
        gt_path: Path = retrieve_path.with_name(
            retrieve_path.stem + "_gt" + retrieve_path.suffix,
        )
        with retrieve_path.open("w", encoding="utf-8") as f:
            json.dump(all_items, f, indent=2)
        with gt_path.open("w", encoding="utf-8") as f:
            json.dump(all_gt, f, indent=2)
        print(f"\nWrote {len(all_items)} retrieval results to {args.retrieve_only}")
        print(f"Wrote {len(all_gt)} ground truth items to {gt_path}")
        print("ISOLATION: retrieval file contains NO ground truth answers")
        return

    # Print full task summary
    print_task_result(task_result)

    # Write detailed output
    if args.output:
        output_data: dict[str, object] = {
            "task": args.task,
            "bench": args.bench,
            "accuracy": round(task_result.accuracy, 4),
            "total_queries": task_result.total_queries,
            "total_correct": task_result.total_correct,
            "perfect_cases": task_result.cases_with_perfect_state,
            "total_cases": len(task_result.cases),
            "per_case": [
                {
                    "case_id": cr.case_id,
                    "accuracy": round(cr.accuracy, 4),
                    "correct": cr.correct_state,
                    "total": cr.total_queries,
                    "per_query": cr.per_query,
                }
                for cr in task_result.cases
            ],
        }
        output_path: Path = Path(args.output)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nDetailed results written to {args.output}")


if __name__ == "__main__":
    main()
