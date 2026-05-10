"""LongMemEval benchmark adapter for aelfrice.

Ingests LongMemEval oracle sessions into aelfrice,
runs retrieval for each question, and outputs results
for LLM-judge scoring.

Paper: LongMemEval (ICLR 2025)
Data:  xiaowu0162/longmemeval-cleaned  (longmemeval_oracle.json)

Usage:
    uv run python benchmarks/longmemeval_adapter.py --retrieve-only results.json
    uv run python benchmarks/longmemeval_adapter.py --question-type temporal-reasoning
    uv run python benchmarks/longmemeval_adapter.py --subset 10
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from aelfrice.ingest import ingest_turn
from aelfrice.retrieval import retrieve_v2 as retrieve  # v1.0.x lab-compat shim
from aelfrice.store import MemoryStore
from benchmarks.qa_scoring import score_multi_answer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUESTION_TYPES: Final[list[str]] = [
    "temporal-reasoning",
    "multi-session",
    "knowledge-update",
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
]

EXPECTED_COUNTS: Final[dict[str, int]] = {
    "temporal-reasoning": 133,
    "multi-session": 133,
    "knowledge-update": 78,
    "single-session-user": 70,
    "single-session-assistant": 56,
    "single-session-preference": 30,
}

HF_DATASET: Final[str] = "xiaowu0162/longmemeval-cleaned"
HF_DATA_FILE: Final[str] = "longmemeval_oracle.json"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SessionTurn:
    """One turn within a haystack session."""

    role: str
    content: str
    has_answer: bool


@dataclass
class LongMemEvalQuestion:
    """One LongMemEval question with its haystack sessions."""

    question_id: str
    question_type: str
    question: str
    answer: str | list[str]
    question_date: str
    haystack_dates: list[str]
    haystack_session_ids: list[str]
    haystack_sessions: list[list[SessionTurn]]
    answer_session_ids: list[str]


_SCORE_KEYS: Final[tuple[str, ...]] = (
    "exact_match",
    "substring_exact_match",
    "f1",
)


@dataclass
class RetrievalResult:
    """Result of retrieval for one question."""

    question_id: str
    question_type: str
    question: str
    question_date: str
    answer: str | list[str]
    retrieved_context: str
    num_beliefs: int
    retrieval_latency_ms: float
    exact_match: float = 0.0
    substring_exact_match: float = 0.0
    f1: float = 0.0


@dataclass
class CategoryStats:
    """Aggregated stats for one question category."""

    question_type: str
    count: int = 0
    total_beliefs: int = 0
    total_latency_ms: float = 0.0
    total_exact_match: float = 0.0
    total_substring_exact_match: float = 0.0
    total_f1: float = 0.0

    @property
    def avg_beliefs(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total_beliefs / self.count

    @property
    def avg_latency_ms(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total_latency_ms / self.count

    @property
    def exact_match(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total_exact_match / self.count

    @property
    def substring_exact_match(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total_substring_exact_match / self.count

    @property
    def f1(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total_f1 / self.count


@dataclass
class BenchmarkResult:
    """Full benchmark results."""

    total_questions: int = 0
    total_beliefs_retrieved: int = 0
    total_latency_ms: float = 0.0
    total_ingest_turns: int = 0
    total_ingest_time_s: float = 0.0
    total_exact_match: float = 0.0
    total_substring_exact_match: float = 0.0
    total_f1: float = 0.0
    category_stats: dict[str, CategoryStats] = field(
        default_factory=lambda: dict[str, CategoryStats](),
    )
    per_question: list[RetrievalResult] = field(
        default_factory=lambda: list[RetrievalResult](),
    )

    @property
    def avg_beliefs(self) -> float:
        if self.total_questions == 0:
            return 0.0
        return self.total_beliefs_retrieved / self.total_questions

    @property
    def avg_latency_ms(self) -> float:
        if self.total_questions == 0:
            return 0.0
        return self.total_latency_ms / self.total_questions

    @property
    def exact_match(self) -> float:
        if self.total_questions == 0:
            return 0.0
        return self.total_exact_match / self.total_questions

    @property
    def substring_exact_match(self) -> float:
        if self.total_questions == 0:
            return 0.0
        return self.total_substring_exact_match / self.total_questions

    @property
    def f1(self) -> float:
        if self.total_questions == 0:
            return 0.0
        return self.total_f1 / self.total_questions


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_from_huggingface() -> list[dict[str, object]]:
    """Load oracle data from HuggingFace datasets."""
    from datasets import load_dataset  # type: ignore[import-untyped]

    ds = load_dataset(  # type: ignore[no-untyped-call]
        HF_DATASET,
        data_files=HF_DATA_FILE,
        split="train",
    )
    raw: list[dict[str, object]] = [dict(row) for row in ds]  # type: ignore[union-attr]
    return raw


def load_from_file(path: str) -> list[dict[str, object]]:
    """Load oracle data from a local JSON file."""
    file_path: Path = Path(path)
    with file_path.open("r", encoding="utf-8") as f:
        raw: list[dict[str, object]] = json.load(f)
    return raw


def parse_questions(raw: list[dict[str, object]]) -> list[LongMemEvalQuestion]:
    """Parse raw records into typed question objects."""
    questions: list[LongMemEvalQuestion] = []
    for row in raw:
        # Parse haystack_sessions: list of sessions, each session is list of turns
        raw_sessions: list[list[dict[str, object]]] = row.get(
            "haystack_sessions", []
        )  # type: ignore[assignment]
        parsed_sessions: list[list[SessionTurn]] = []
        for session in raw_sessions:
            turns: list[SessionTurn] = []
            for turn in session:
                turns.append(SessionTurn(
                    role=str(turn.get("role", "")),
                    content=str(turn.get("content", "")),
                    has_answer=bool(turn.get("has_answer", False)),
                ))
            parsed_sessions.append(turns)

        # Answer can be string or list
        raw_answer: object = row.get("answer", "")
        answer: str | list[str]
        if isinstance(raw_answer, list):
            answer = [str(a) for a in raw_answer]  # type: ignore[union-attr]
        else:
            answer = str(raw_answer)

        questions.append(LongMemEvalQuestion(
            question_id=str(row.get("question_id", "")),
            question_type=str(row.get("question_type", "")),
            question=str(row.get("question", "")),
            answer=answer,
            question_date=str(row.get("question_date", "")),
            haystack_dates=[str(d) for d in row.get("haystack_dates", [])],  # type: ignore[union-attr]
            haystack_session_ids=[str(s) for s in row.get("haystack_session_ids", [])],  # type: ignore[union-attr]
            haystack_sessions=parsed_sessions,
            answer_session_ids=[str(s) for s in row.get("answer_session_ids", [])],  # type: ignore[union-attr]
        ))
    return questions


# ---------------------------------------------------------------------------
# Ingest adapter
# ---------------------------------------------------------------------------


def ingest_sessions(
    store: MemoryStore,
    question: LongMemEvalQuestion,
) -> int:
    """Ingest all haystack sessions for one question into a fresh store.

    Returns total turns ingested.
    """
    total_turns: int = 0
    for idx, session_turns in enumerate(question.haystack_sessions):
        session_id_str: str = ""
        if idx < len(question.haystack_session_ids):
            session_id_str = question.haystack_session_ids[idx]

        session_date: str = ""
        if idx < len(question.haystack_dates):
            session_date = question.haystack_dates[idx]

        # Create aelfrice session for isolation
        am_session = store.create_session(
            model="longmemeval-benchmark",
            project_context=f"q={question.question_id} session={session_id_str}",
        )

        # Ingest date marker for temporal grounding
        if session_date:
            date_marker: str = f"[Session {session_id_str}, date: {session_date}]"
            ingest_turn(
                store=store,
                text=date_marker,
                source="longmemeval",
                session_id=am_session.id,
                created_at=session_date if _is_valid_iso(session_date) else None,
                source_id=f"S{session_id_str}:marker",
            )

        for turn_idx, turn in enumerate(session_turns):
            # Prefix with session date for temporal grounding
            text: str = f"[{session_date}] {turn.role}: {turn.content}"
            ingest_turn(
                store=store,
                text=text,
                source="longmemeval",
                session_id=am_session.id,
                created_at=session_date if _is_valid_iso(session_date) else None,
                source_id=f"S{session_id_str}:{turn_idx}",
            )
            total_turns += 1

        store.complete_session(am_session.id)

    return total_turns


def _is_valid_iso(date_str: str) -> bool:
    """Check if a string looks like a valid ISO date or datetime."""
    if not date_str or len(date_str) < 8:
        return False
    try:
        # Accept YYYY-MM-DD or full ISO datetime
        if "T" in date_str:
            time.strptime(date_str[:19], "%Y-%m-%dT%H:%M:%S")
        else:
            time.strptime(date_str[:10], "%Y-%m-%d")
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Query adapter
# ---------------------------------------------------------------------------


def query_aelfrice(
    store: MemoryStore,
    question: str,
    question_date: str,
    budget: int = 2000,
) -> tuple[str, int]:
    """Query aelfrice for a question. Returns (context, num_beliefs).

    Includes question_date in the query for temporal grounding.
    """
    query_text: str = question
    if question_date:
        query_text = f"[As of {question_date}] {question}"

    result = retrieve(
        store=store,
        query=query_text,
        budget=budget,
        include_locked=False,
        use_bfs=True,
    )
    parts: list[str] = [b.content for b in result.beliefs]
    return " ".join(parts), len(result.beliefs)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_question(
    question: LongMemEvalQuestion,
    db_dir: str,
    budget: int = 2000,
) -> tuple[RetrievalResult, int, float]:
    """Run ingestion and retrieval for one question.

    Returns (result, ingest_turns, ingest_time_s).
    Uses a fresh DB for isolation.
    """
    db_path: str = f"{db_dir}/{question.question_id}.db"
    store: MemoryStore = MemoryStore(db_path)

    # Ingest all haystack sessions
    t0: float = time.monotonic()
    ingest_turns: int = ingest_sessions(store, question)
    ingest_time: float = time.monotonic() - t0

    # Query
    t1: float = time.monotonic()
    context, num_beliefs = query_aelfrice(
        store, question.question, question.question_date, budget=budget,
    )
    latency_ms: float = (time.monotonic() - t1) * 1000.0

    # Score retrieved context against gold answer(s). LongMemEval lists
    # multiple acceptable answer surfaces for some categories — fold
    # via best-of multi-answer per #507.
    gts: list[str] = (
        [str(a) for a in question.answer]
        if isinstance(question.answer, list)
        else [str(question.answer)]
    )
    scores: dict[str, float] = score_multi_answer(context, gts)

    result: RetrievalResult = RetrievalResult(
        question_id=question.question_id,
        question_type=question.question_type,
        question=question.question,
        question_date=question.question_date,
        answer=question.answer,
        retrieved_context=context,
        num_beliefs=num_beliefs,
        retrieval_latency_ms=latency_ms,
        exact_match=scores["exact_match"],
        substring_exact_match=scores["substring_exact_match"],
        f1=scores["f1"],
    )
    return result, ingest_turns, ingest_time


def run_benchmark(
    questions: list[LongMemEvalQuestion],
    budget: int = 2000,
) -> BenchmarkResult:
    """Run the full benchmark over all questions."""
    result: BenchmarkResult = BenchmarkResult()

    with tempfile.TemporaryDirectory(prefix="longmemeval_bench_") as tmpdir:
        for i, q in enumerate(questions):
            print(
                f"  [{i+1}/{len(questions)}] {q.question_id} "
                f"({q.question_type}, {len(q.haystack_sessions)} sessions)"
            )

            qr, ingest_turns, ingest_time = run_question(q, tmpdir, budget=budget)

            result.total_questions += 1
            result.total_beliefs_retrieved += qr.num_beliefs
            result.total_latency_ms += qr.retrieval_latency_ms
            result.total_ingest_turns += ingest_turns
            result.total_ingest_time_s += ingest_time
            result.total_exact_match += qr.exact_match
            result.total_substring_exact_match += qr.substring_exact_match
            result.total_f1 += qr.f1
            result.per_question.append(qr)

            # Per-category stats
            if qr.question_type not in result.category_stats:
                result.category_stats[qr.question_type] = CategoryStats(
                    question_type=qr.question_type,
                )
            cat: CategoryStats = result.category_stats[qr.question_type]
            cat.count += 1
            cat.total_beliefs += qr.num_beliefs
            cat.total_latency_ms += qr.retrieval_latency_ms
            cat.total_exact_match += qr.exact_match
            cat.total_substring_exact_match += qr.substring_exact_match
            cat.total_f1 += qr.f1

    return result


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_results(result: BenchmarkResult) -> None:
    """Print formatted benchmark results to stdout."""
    print(f"\n{'='*65}")
    print("LongMemEval Benchmark Results (retrieval-only)")
    print(f"{'='*65}")
    print(f"Total questions:      {result.total_questions}")
    print(f"Total ingested turns: {result.total_ingest_turns}")
    print(f"Total ingest time:    {result.total_ingest_time_s:.2f}s")
    print(f"Avg beliefs/query:    {result.avg_beliefs:.1f}")
    print(f"Avg query latency:    {result.avg_latency_ms:.1f}ms")
    print()

    print(
        f"Overall correctness (deterministic): "
        f"EM={result.exact_match:.4f}  "
        f"sub_EM={result.substring_exact_match:.4f}  "
        f"F1={result.f1:.4f}"
    )
    print()
    print("Per-category retrieval stats:")
    for qtype in QUESTION_TYPES:
        cat: CategoryStats | None = result.category_stats.get(qtype)
        if cat is None or cat.count == 0:
            continue
        print(
            f"  {qtype:30s}  n={cat.count:3d}  "
            f"avg_beliefs={cat.avg_beliefs:5.1f}  "
            f"avg_latency={cat.avg_latency_ms:6.1f}ms  "
            f"sub_EM={cat.substring_exact_match:.4f}  "
            f"F1={cat.f1:.4f}"
        )
    print()

    # Reference baselines from the paper
    print("Reference baselines (from LongMemEval paper):")
    print("  BM25 session-level Recall@5:        0.634")
    print("  BM25 session-level NDCG@5:          0.516")
    print("  GPT-4o with oracle sessions:        0.924 accuracy")
    print("  GPT-4o with LongMemEval_S pipeline: 0.606 accuracy")
    print(f"{'='*65}")


def write_retrieve_results(result: BenchmarkResult, output_path: str) -> None:
    """Write retrieval results (NO ground truth) and separate GT file."""
    items: list[dict[str, object]] = []
    gt_items: list[dict[str, object]] = []
    for qr in result.per_question:
        # Retrieval file: NO answers, only context for LLM reader
        items.append({
            "question_id": qr.question_id,
            "question_type": qr.question_type,
            "question": qr.question,
            "question_date": qr.question_date,
            "retrieved_context": qr.retrieved_context,
            "num_beliefs": qr.num_beliefs,
            "retrieval_latency_ms": round(qr.retrieval_latency_ms, 2),
        })
        # Ground truth file: answers for scoring AFTER generation
        gt_items.append({
            "question_id": qr.question_id,
            "question_type": qr.question_type,
            "answer": qr.answer,
        })

    path: Path = Path(output_path)
    gt_path: Path = path.with_name(path.stem + "_gt" + path.suffix)
    with path.open("w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)
    with gt_path.open("w", encoding="utf-8") as f:
        json.dump(gt_items, f, indent=2)

    print(f"Wrote {len(items)} retrieval results to {output_path}")
    print(f"Wrote {len(gt_items)} ground truth items to {gt_path}")
    print("ISOLATION: retrieval file contains NO ground truth answers")
    print("Next step: run LLM reader on retrieval file, then score against GT")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Run LongMemEval benchmark on aelfrice",
    )
    parser.add_argument(
        "--data", default=None,
        help="Path to local longmemeval_oracle.json (default: download from HuggingFace)",
    )
    parser.add_argument(
        "--subset", type=int, default=None,
        help="Limit to first N questions (for debugging)",
    )
    parser.add_argument(
        "--budget", type=int, default=2000,
        help="Token budget for retrieval (default: 2000)",
    )
    parser.add_argument(
        "--question-type", default=None, choices=QUESTION_TYPES,
        help="Filter to a single question category",
    )
    parser.add_argument(
        "--retrieve-only", default=None, metavar="PATH",
        help="Write retrieval results to PATH for LLM-judge scoring (primary mode)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Write full results JSON to this path",
    )
    args: argparse.Namespace = parser.parse_args()

    # Load data
    print("Loading LongMemEval oracle dataset...")
    try:
        if args.data is not None:
            raw: list[dict[str, object]] = load_from_file(args.data)
        else:
            print(f"  Downloading from HuggingFace: {HF_DATASET}")
            raw = load_from_huggingface()
    except FileNotFoundError as exc:
        print(
            f"LongMemEval data not found at {args.data}: {exc}",
            file=sys.stderr,
        )
        sys.exit(2)
    print(f"Loaded {len(raw)} questions")

    if not raw:
        print(
            "No LongMemEval questions loaded.",
            file=sys.stderr,
        )
        sys.exit(2)

    questions: list[LongMemEvalQuestion] = parse_questions(raw)

    # Filter by question type
    if args.question_type is not None:
        questions = [q for q in questions if q.question_type == args.question_type]
        print(f"Filtered to {len(questions)} questions of type '{args.question_type}'")

    # Subset
    if args.subset is not None:
        questions = questions[:args.subset]
        print(f"Using first {len(questions)} questions")

    # Print distribution
    type_counts: dict[str, int] = {}
    for q in questions:
        type_counts[q.question_type] = type_counts.get(q.question_type, 0) + 1
    print("\nQuestion distribution:")
    for qtype in QUESTION_TYPES:
        count: int = type_counts.get(qtype, 0)
        if count > 0:
            expected: int = EXPECTED_COUNTS.get(qtype, 0)
            print(f"  {qtype:30s}  {count:3d}  (expected: {expected})")
    print()

    # Run benchmark
    print("Running benchmark...")
    result: BenchmarkResult = run_benchmark(questions, budget=args.budget)

    # Print stats
    print_results(result)

    # Retrieve-only mode: write results for LLM judge
    if args.retrieve_only is not None:
        write_retrieve_results(result, args.retrieve_only)

    # Full output
    if args.output is not None:
        output_data: dict[str, object] = {
            "total_questions": result.total_questions,
            "avg_beliefs_per_query": round(result.avg_beliefs, 2),
            "avg_latency_ms": round(result.avg_latency_ms, 2),
            "total_ingest_turns": result.total_ingest_turns,
            "total_ingest_time_s": round(result.total_ingest_time_s, 2),
            "exact_match": round(result.exact_match, 4),
            "substring_exact_match": round(result.substring_exact_match, 4),
            "f1": round(result.f1, 4),
            "category_stats": {
                qtype: {
                    "count": cat.count,
                    "avg_beliefs": round(cat.avg_beliefs, 2),
                    "avg_latency_ms": round(cat.avg_latency_ms, 2),
                    "exact_match": round(cat.exact_match, 4),
                    "substring_exact_match": round(cat.substring_exact_match, 4),
                    "f1": round(cat.f1, 4),
                }
                for qtype, cat in result.category_stats.items()
            },
            "per_question": [
                {
                    "question_id": qr.question_id,
                    "question_type": qr.question_type,
                    "question": qr.question,
                    "answer": qr.answer,
                    "num_beliefs": qr.num_beliefs,
                    "retrieval_latency_ms": round(qr.retrieval_latency_ms, 2),
                    "exact_match": qr.exact_match,
                    "substring_exact_match": qr.substring_exact_match,
                    "f1": round(qr.f1, 4),
                }
                for qr in result.per_question
            ],
        }
        out_path: Path = Path(args.output)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nDetailed results written to {args.output}")


if __name__ == "__main__":
    main()
