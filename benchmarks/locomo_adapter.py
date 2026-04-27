"""LoCoMo benchmark adapter for aelfrice.

Ingests LoCoMo multi-session conversations into aelfrice,
runs QA evaluation, and reports F1 scores comparable to the
published leaderboard.

Usage:
    uv run python benchmarks/locomo_adapter.py [--data PATH] [--conversations N] [--subset N]
"""
from __future__ import annotations

import argparse
import json
import re
import string
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from nltk.stem import PorterStemmer  # type: ignore[import-untyped]

from aelfrice.ingest import ingest_turn
from aelfrice.retrieval import retrieve_v2 as retrieve  # v1.0.x lab-compat shim
from aelfrice.store import MemoryStore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DATA_PATH: Final[str] = "/tmp/LoCoMo/data/locomo10.json"
CATEGORY_NAMES: Final[dict[int, str]] = {
    1: "multi-hop",
    2: "temporal",
    3: "open-ended",
    4: "single-hop",
    5: "adversarial",
}

_PS: Final[PorterStemmer] = PorterStemmer()


# ---------------------------------------------------------------------------
# Scoring (mirrors LoCoMo evaluation.py exactly)
# ---------------------------------------------------------------------------


def normalize_answer(s: str) -> str:
    """Lowercase, strip punctuation/articles/commas, collapse whitespace."""
    s = s.replace(",", "")

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the|and)\b", " ", text)

    def remove_punc(text: str) -> str:
        exclude: set[str] = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return " ".join(remove_articles(remove_punc(s.lower())).split())


def _stem(word: str) -> str:
    """Porter-stem a word, returning str (nltk stubs are untyped)."""
    result: str = str(_PS.stem(word))  # type: ignore[no-untyped-call]
    return result


def f1_score_single(prediction: str, ground_truth: str) -> float:
    """Token-level F1 with Porter stemming (single answer pair)."""
    pred_tokens: list[str] = [_stem(w) for w in normalize_answer(prediction).split()]
    gt_tokens: list[str] = [_stem(w) for w in normalize_answer(ground_truth).split()]
    if not pred_tokens or not gt_tokens:
        return 0.0
    common: Counter[str] = Counter(pred_tokens) & Counter(gt_tokens)
    num_same: int = sum(common.values())
    if num_same == 0:
        return 0.0
    precision: float = num_same / len(pred_tokens)
    recall: float = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def f1_multi_hop(prediction: str, ground_truth: str) -> float:
    """Multi-hop F1: split on commas, best-match each ground truth sub-answer."""
    predictions: list[str] = [p.strip() for p in prediction.split(",")]
    ground_truths: list[str] = [g.strip() for g in ground_truth.split(",")]
    scores: list[float] = []
    for gt in ground_truths:
        best: float = max(f1_score_single(p, gt) for p in predictions)
        scores.append(best)
    return sum(scores) / len(scores) if scores else 0.0


def score_qa(prediction: str, answer: str, category: int) -> float:
    """Score a single QA pair using the appropriate metric for its category."""
    if category in (2, 4):
        return f1_score_single(prediction, answer)
    if category == 3:
        answer = answer.split(";")[0].strip()
        return f1_score_single(prediction, answer)
    if category == 1:
        return f1_multi_hop(prediction, answer)
    if category == 5:
        lower_pred: str = prediction.lower()
        if "no information available" in lower_pred or "not mentioned" in lower_pred:
            return 1.0
        return 0.0
    msg: str = f"Unknown category: {category}"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass
class LoCoMoConversation:
    """One LoCoMo conversation with sessions and QA pairs."""

    sample_id: str
    speaker_a: str
    speaker_b: str
    sessions: list[LoCoMoSession]
    qa_pairs: list[QAPair]


@dataclass
class LoCoMoSession:
    """One session within a conversation."""

    session_num: int
    date_time: str
    turns: list[Turn]


@dataclass
class Turn:
    """One dialog turn."""

    speaker: str
    dia_id: str
    text: str


@dataclass
class QAPair:
    """One QA evaluation pair."""

    question: str
    answer: str  # empty for category 5
    adversarial_answer: str  # only for category 5
    evidence: list[str]
    category: int


def load_locomo(data_path: str) -> list[LoCoMoConversation]:
    """Load LoCoMo dataset from JSON file."""
    path: Path = Path(data_path)
    with path.open("r", encoding="utf-8") as f:
        raw: list[dict[str, object]] = json.load(f)

    conversations: list[LoCoMoConversation] = []
    for record in raw:
        conv_data: dict[str, object] = record["conversation"]  # type: ignore[assignment]
        speaker_a: str = str(conv_data.get("speaker_a", "A"))
        speaker_b: str = str(conv_data.get("speaker_b", "B"))

        # Extract sessions in order
        sessions: list[LoCoMoSession] = []
        session_num: int = 1
        while True:
            session_key: str = f"session_{session_num}"
            dt_key: str = f"{session_key}_date_time"
            if session_key not in conv_data:
                break
            date_time: str = str(conv_data.get(dt_key, ""))
            raw_turns_val: list[dict[str, str]] = conv_data.get(session_key, [])  # type: ignore[assignment]
            turns: list[Turn] = []
            for t in raw_turns_val:
                turns.append(Turn(
                    speaker=t.get("speaker", ""),
                    dia_id=t.get("dia_id", ""),
                    text=t.get("text", ""),
                ))
            sessions.append(LoCoMoSession(
                session_num=session_num,
                date_time=date_time,
                turns=turns,
            ))
            session_num += 1

        # Extract QA pairs
        raw_qa: list[dict[str, object]] = record.get("qa", [])  # type: ignore[assignment]
        qa_pairs: list[QAPair] = []
        for q in raw_qa:
            qa_pairs.append(QAPair(
                question=str(q.get("question", "")),
                answer=str(q.get("answer", "")),
                adversarial_answer=str(q.get("adversarial_answer", "")),
                evidence=[str(e) for e in q.get("evidence", [])],  # type: ignore[union-attr]
                category=int(q.get("category", 0)),  # type: ignore[arg-type]
            ))

        conversations.append(LoCoMoConversation(
            sample_id=str(record.get("sample_id", "")),
            speaker_a=speaker_a,
            speaker_b=speaker_b,
            sessions=sessions,
            qa_pairs=qa_pairs,
        ))

    return conversations


# ---------------------------------------------------------------------------
# Ingest adapter
# ---------------------------------------------------------------------------


def _parse_locomo_datetime(dt_str: str) -> str:
    """Convert LoCoMo date format to ISO 8601.

    Example: '1:56 pm on 8 May, 2023' -> '2023-05-08T13:56:00+00:00'
    Falls back to current time if parsing fails.
    """
    if not dt_str:
        return datetime.now(timezone.utc).isoformat()
    try:
        # Remove comma before year: "8 May, 2023" -> "8 May 2023"
        cleaned: str = dt_str.replace(",", "")
        # Try: "1:56 pm on 8 May 2023"
        match = re.match(
            r"(\d{1,2}):(\d{2})\s*(am|pm)\s+on\s+(\d{1,2})\s+(\w+)\s+(\d{4})",
            cleaned,
            re.IGNORECASE,
        )
        if match:
            hour: int = int(match.group(1))
            minute: int = int(match.group(2))
            ampm: str = match.group(3).lower()
            day: int = int(match.group(4))
            month_name: str = match.group(5)
            year: int = int(match.group(6))
            if ampm == "pm" and hour != 12:
                hour += 12
            elif ampm == "am" and hour == 12:
                hour = 0
            dt: datetime = datetime.strptime(
                f"{year} {month_name} {day} {hour}:{minute}",
                "%Y %B %d %H:%M",
            ).replace(tzinfo=timezone.utc)
            return dt.isoformat()
    except (ValueError, AttributeError):
        pass
    return datetime.now(timezone.utc).isoformat()


def ingest_conversation(store: MemoryStore, conv: LoCoMoConversation) -> int:
    """Ingest all sessions of a conversation into aelfrice.

    Returns total turns ingested.
    """
    total_turns: int = 0
    for session in conv.sessions:
        # Create a session in the store (observations have FK to sessions)
        am_session = store.create_session(
            model="locomo-benchmark",
            project_context=f"{conv.sample_id} session {session.session_num}",
        )
        # Ingest a session marker with date so temporal queries can resolve
        if session.date_time:
            date_marker: str = f"[Session {session.session_num}, {session.date_time}]"
            ingest_turn(
                store=store,
                text=date_marker,
                source="locomo",
                session_id=am_session.id,
                created_at=_parse_locomo_datetime(session.date_time),
                source_id=f"D{session.session_num}:0",
            )

        for turn in session.turns:
            # Include session date for temporal grounding
            text: str = f"[{session.date_time}] {turn.speaker}: {turn.text}"
            ingest_turn(
                store=store,
                text=text,
                source="locomo",
                session_id=am_session.id,
                created_at=_parse_locomo_datetime(session.date_time),
                source_id=turn.dia_id,
            )
            total_turns += 1
        store.complete_session(am_session.id)
    return total_turns


# ---------------------------------------------------------------------------
# Query adapter
# ---------------------------------------------------------------------------


def _retrieve_context(store: MemoryStore, question: str, budget: int = 2000) -> str:
    """Retrieve relevant beliefs from aelfrice for a question."""
    result = retrieve(
        store=store,
        query=question,
        budget=budget,
        include_locked=False,  # No locked beliefs in benchmark DB
        use_hrr=True,
        use_bfs=True,
    )
    parts: list[str] = [b.content for b in result.beliefs]
    return " ".join(parts)


def query_aelfrice(store: MemoryStore, question: str, budget: int = 2000) -> str:
    """Query aelfrice and return retrieved belief content."""
    return _retrieve_context(store, question, budget)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results."""

    conversation_id: str = ""
    total_qa: int = 0
    total_f1: float = 0.0
    category_scores: dict[int, list[float]] = field(default_factory=lambda: dict[int, list[float]]())
    category_counts: dict[int, int] = field(default_factory=lambda: dict[int, int]())
    ingest_turns: int = 0
    ingest_time_s: float = 0.0
    query_time_s: float = 0.0
    per_question: list[dict[str, object]] = field(default_factory=lambda: list[dict[str, object]]())

    @property
    def overall_f1(self) -> float:
        if self.total_qa == 0:
            return 0.0
        return self.total_f1 / self.total_qa

    def category_f1(self, cat: int) -> float:
        scores: list[float] = self.category_scores.get(cat, [])
        if not scores:
            return 0.0
        return sum(scores) / len(scores)


def merge_results(results: list[BenchmarkResult]) -> BenchmarkResult:
    """Merge per-conversation results into an aggregate."""
    merged: BenchmarkResult = BenchmarkResult(conversation_id="ALL")
    for r in results:
        merged.total_qa += r.total_qa
        merged.total_f1 += r.total_f1
        merged.ingest_turns += r.ingest_turns
        merged.ingest_time_s += r.ingest_time_s
        merged.query_time_s += r.query_time_s
        merged.per_question.extend(r.per_question)
        for cat, scores in r.category_scores.items():
            if cat not in merged.category_scores:
                merged.category_scores[cat] = []
                merged.category_counts[cat] = 0
            merged.category_scores[cat].extend(scores)
            merged.category_counts[cat] = merged.category_counts.get(cat, 0) + r.category_counts.get(cat, 0)
    return merged


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_conversation(
    conv: LoCoMoConversation,
    db_dir: str,
    subset: int | None = None,
    budget: int = 2000,
) -> BenchmarkResult:
    """Run the full benchmark pipeline on one conversation.

    Uses a fresh DB per conversation for isolation.
    """
    db_path: str = f"{db_dir}/{conv.sample_id}.db"
    store: MemoryStore = MemoryStore(db_path)

    result: BenchmarkResult = BenchmarkResult(conversation_id=conv.sample_id)

    # Ingest all sessions
    t0: float = time.monotonic()
    result.ingest_turns = ingest_conversation(store, conv)
    result.ingest_time_s = time.monotonic() - t0

    # Query and score
    qa_pairs: list[QAPair] = conv.qa_pairs
    if subset is not None:
        qa_pairs = qa_pairs[:subset]

    t1: float = time.monotonic()
    for qa in qa_pairs:
        prediction: str = query_aelfrice(store, qa.question, budget=budget)

        # For category 5, check if retrieval found nothing relevant
        # If retrieved content is thin, default to refusal
        if qa.category == 5:
            if len(prediction.split()) < 10:
                prediction = "No information available"

        answer: str = qa.answer if qa.category != 5 else ""
        f1: float = score_qa(prediction, answer, qa.category)

        result.total_qa += 1
        result.total_f1 += f1

        if qa.category not in result.category_scores:
            result.category_scores[qa.category] = []
            result.category_counts[qa.category] = 0
        result.category_scores[qa.category].append(f1)
        result.category_counts[qa.category] += 1

        result.per_question.append({
            "question": qa.question,
            "answer": qa.answer,
            "category": qa.category,
            "category_name": CATEGORY_NAMES.get(qa.category, "unknown"),
            "context": prediction,  # full retrieved context for subagent
            "prediction": prediction[:500],  # truncated for display
            "f1": round(f1, 4),
        })

    result.query_time_s = time.monotonic() - t1

    return result


def print_results(result: BenchmarkResult) -> None:
    """Print formatted benchmark results."""
    print(f"\n{'='*60}")
    print(f"LoCoMo Benchmark Results: {result.conversation_id}")
    print(f"{'='*60}")
    print(f"Total QA pairs:    {result.total_qa}")
    print(f"Overall F1:        {result.overall_f1:.4f} ({result.overall_f1*100:.1f}%)")
    print(f"Ingest turns:      {result.ingest_turns}")
    print(f"Ingest time:       {result.ingest_time_s:.2f}s")
    print(f"Query time:        {result.query_time_s:.2f}s")
    if result.total_qa > 0:
        print(f"Avg query latency: {result.query_time_s / result.total_qa * 1000:.1f}ms")
    print()
    print("Per-category F1:")
    for cat in sorted(result.category_scores.keys()):
        name: str = CATEGORY_NAMES.get(cat, "unknown")
        count: int = result.category_counts.get(cat, 0)
        f1: float = result.category_f1(cat)
        print(f"  {cat}. {name:12s}  {f1:.4f} ({f1*100:.1f}%)  n={count}")
    print()

    # Reference baselines
    print("Reference baselines:")
    print("  Filesystem+grep (Letta):  74.0%")
    print("  EverMemOS (SOTA):         92.3%")
    print(f"{'='*60}")


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Run LoCoMo benchmark on aelfrice",
    )
    parser.add_argument(
        "--data", default=DEFAULT_DATA_PATH,
        help="Path to locomo10.json",
    )
    parser.add_argument(
        "--conversations", type=int, default=None,
        help="Limit to first N conversations (default: all 10)",
    )
    parser.add_argument(
        "--subset", type=int, default=None,
        help="Limit to first N QA pairs per conversation (for debugging)",
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
        help="Run retrieval only, write question+context pairs to PATH for LLM subagent scoring",
    )
    args: argparse.Namespace = parser.parse_args()

    print("Loading LoCoMo dataset...")
    conversations: list[LoCoMoConversation] = load_locomo(args.data)
    print(f"Loaded {len(conversations)} conversations")

    if args.conversations is not None:
        conversations = conversations[:args.conversations]
        print(f"Using first {len(conversations)} conversations")

    results: list[BenchmarkResult] = []

    with tempfile.TemporaryDirectory(prefix="locomo_bench_") as tmpdir:
        for conv in conversations:
            total_turns: int = sum(len(s.turns) for s in conv.sessions)
            total_qa: int = len(conv.qa_pairs)
            if args.subset is not None:
                total_qa = min(total_qa, args.subset)
            print(f"\n--- {conv.sample_id}: {len(conv.sessions)} sessions, "
                  f"{total_turns} turns, {total_qa} QA pairs ---")

            conv_result: BenchmarkResult = run_conversation(
                conv, tmpdir, subset=args.subset, budget=args.budget,
            )
            results.append(conv_result)
            if not args.retrieve_only:
                print_results(conv_result)

    # If retrieve-only, write question+context pairs for subagent scoring
    if args.retrieve_only:
        all_items: list[dict[str, object]] = []
        for r in results:
            all_items.extend(r.per_question)
        retrieve_path: Path = Path(args.retrieve_only)
        with retrieve_path.open("w", encoding="utf-8") as f:
            json.dump(all_items, f, indent=2)
        total_q: int = sum(r.total_qa for r in results)
        print(f"Wrote {total_q} retrieval results to {args.retrieve_only}")
        print("Next step: run locomo_generate.py via subagent to produce answers")
        return

    # Aggregate
    if len(results) > 1:
        merged: BenchmarkResult = merge_results(results)
        print_results(merged)

    # Write detailed output
    if args.output:
        merged_for_output: BenchmarkResult = merge_results(results) if len(results) > 1 else results[0]
        output_data: dict[str, object] = {
            "overall_f1": round(merged_for_output.overall_f1, 4),
            "total_qa": merged_for_output.total_qa,
            "category_f1": {
                str(cat): round(merged_for_output.category_f1(cat), 4)
                for cat in sorted(merged_for_output.category_scores.keys())
            },
            "per_question": merged_for_output.per_question,
        }
        output_path: Path = Path(args.output)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nDetailed results written to {args.output}")


if __name__ == "__main__":
    main()
