"""LoCoMo benchmark adapter for aelfrice.

Ingests LoCoMo multi-session conversations into aelfrice, runs QA
evaluation, and reports F1 alongside reader-independent retrieval
quality.

The two are reported separately on purpose (#1160). `overall_f1` and
`category_f1` are **reader-dependent**: no reader runs in
`aelf bench all`, so the retrieved context is handed to a scorer written
for a model's answer, and the resulting token-F1 moves with the token
budget as much as with the ranking. `retrieval_quality` is
**reader-independent**: MRR and recall@k over the ordered retrieved list,
which a smaller budget can only lower. Category 5 is reported as `n/a`
rather than 0.0 — see `UNSCORABLE_CATEGORIES`.

Usage:
    uv run python benchmarks/locomo_adapter.py [--data PATH] [--conversations N] [--subset N]
"""
from __future__ import annotations

import argparse
import json
import re
import string
import sys
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
from benchmarks.metric_status import (
    NOT_APPLICABLE,
    NOT_APPLICABLE_REASONS_KEY,
)
from benchmarks.retrieval_metrics import mean_metrics, retrieval_metrics

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

#: Categories this adapter cannot score without a reader (#1160).
#:
#: LoCoMo category 5 is adversarial: the correct response is a refusal,
#: and `score_qa` awards the point only when the prediction contains
#: "no information available" or "not mentioned". Nothing in
#: `aelf bench all` can refuse — the prediction is the retrieved context,
#: which never contains those strings. The category therefore scored a
#: hard 0.0 on every run ever recorded, and that zero sat inside
#: `overall_f1` dragging it down and inside a tolerance band where any
#: genuine fix would have registered as a band excursion.
#:
#: Scoring it as 0.0 asserted a measurement that was never taken. It is
#: now reported as `n/a` and excluded from `overall_f1`, which becomes
#: the mean over the categories that were actually scored.
UNSCORABLE_CATEGORIES: Final[frozenset[int]] = frozenset({5})

UNSCORABLE_CATEGORY_REASON: Final[str] = (
    "adversarial: scoring requires a reader that can abstain; "
    "aelf bench all runs no reader, so the retrieved context can never "
    "produce the refusal this category scores for"
)

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
    """Score a single QA pair using the appropriate metric for its category.

    Mirrors LoCoMo's own `evaluation.py`, including the category-5 branch
    — which is why that branch is kept here rather than deleted. It is
    correct for a reader's answer and unreachable from `run_conversation`,
    which routes `UNSCORABLE_CATEGORIES` around scoring entirely because
    `aelf bench all` has no reader to produce a refusal. Wire a reader in
    and this becomes live again unchanged (#1160).
    """
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


def _retrieve_beliefs(
    store: MemoryStore, question: str, budget: int = 2000,
) -> list[str]:
    """Retrieve relevant beliefs, in rank order, without joining them.

    The ordering is what `benchmarks.retrieval_metrics` reads; joining
    first destroys it, which is why the blob scorers cannot tell a
    ranking change from a budget change (#1160).
    """
    result = retrieve(
        store=store,
        query=question,
        budget=budget,
        include_locked=False,  # No locked beliefs in benchmark DB
        use_bfs=True,
    )
    return [b.content for b in result.beliefs]


def _retrieve_context(store: MemoryStore, question: str, budget: int = 2000) -> str:
    """Retrieve relevant beliefs from aelfrice for a question."""
    return " ".join(_retrieve_beliefs(store, question, budget))


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
    # Per-query rank metrics over the retrieved list, one dict per
    # question — excluding the unscorable categories, which have no gold
    # surface to rank against and would contribute a structural 0.0
    # (#1160). `scored_qa` is the matching denominator.
    per_question_retrieval: list[dict[str, float]] = field(
        default_factory=lambda: list[dict[str, float]](),
    )

    @property
    def scored_qa(self) -> int:
        """Questions that contributed to `total_f1`.

        `total_qa` stays the full corpus size — it is a corpus invariant
        the band-check watches for drift — so the two are reported
        separately rather than one being redefined.
        """
        return sum(
            count
            for cat, count in self.category_counts.items()
            if cat not in UNSCORABLE_CATEGORIES
        )

    @property
    def overall_f1(self) -> float:
        """Mean F1 over the scorable categories only.

        Category 5 previously contributed a structural 0.0 to every run
        (see `UNSCORABLE_CATEGORIES`), so this number was a blend of a
        measurement and a placeholder.
        """
        scored: int = self.scored_qa
        if scored == 0:
            return 0.0
        return self.total_f1 / scored

    def category_f1(self, cat: int) -> float:
        scores: list[float] = self.category_scores.get(cat, [])
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def retrieval_quality(self) -> dict[str, float]:
        """Reader-independent MRR / recall@k over the *scorable* questions.

        Excludes `UNSCORABLE_CATEGORIES` for the same reason `overall_f1`
        does. On LoCoMo-10 category 5 is 446 of 1986 questions (22.5%),
        each of which can only score 0.0 because its gold surface is
        empty — including them would cap `mrr` at 0.775 regardless of
        ranking quality, and would make the metric move with corpus
        composition (`--conversations` / `--subset` change the category
        mix), which is exactly the budget-invariance this block promises.
        """
        return mean_metrics(self.per_question_retrieval)


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
        merged.per_question_retrieval.extend(r.per_question_retrieval)
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
        beliefs: list[str] = _retrieve_beliefs(store, qa.question, budget=budget)
        prediction: str = " ".join(beliefs)

        # Rank metrics are reader-independent, but only where a gold
        # surface exists to rank against. Category 5 carries its gold in
        # `adversarial_answer` and leaves `answer` empty, and
        # `is_relevant` correctly refuses an empty gold (`"" in anything`
        # would award a free hit) — so scoring it here appends a
        # structural 0.0 that no ranking can move. That is the same
        # placeholder-as-measurement defect this module exists to remove,
        # and 0.0 would additionally conflate "correctly found nothing to
        # find" with "failed to find what was there" (#1160).
        rank_scores: dict[str, float] = retrieval_metrics(beliefs, [qa.answer])
        if qa.category not in UNSCORABLE_CATEGORIES:
            result.per_question_retrieval.append(rank_scores)

        result.total_qa += 1
        if qa.category not in result.category_scores:
            result.category_scores[qa.category] = []
            result.category_counts[qa.category] = 0
        result.category_counts[qa.category] += 1

        # Unscorable categories are counted and retrieval-measured, but
        # contribute no answer score. Previously category 5 ran through
        # `score_qa` behind a heuristic that promoted thin retrieval to
        # a refusal — unreachable at any realistic budget, so it only
        # ever produced 0.0. Recording that as a score was the defect.
        if qa.category in UNSCORABLE_CATEGORIES:
            f1_display: object = NOT_APPLICABLE
        else:
            f1: float = score_qa(prediction, qa.answer, qa.category)
            result.total_f1 += f1
            result.category_scores[qa.category].append(f1)
            f1_display = round(f1, 4)

        result.per_question.append({
            "question": qa.question,
            "answer": qa.answer,
            "category": qa.category,
            "category_name": CATEGORY_NAMES.get(qa.category, "unknown"),
            "context": prediction,  # full retrieved context for subagent
            "prediction": prediction[:500],  # truncated for display
            "f1": f1_display,
            # Deliberately no rank metrics here. `per_question` doubles
            # as the `--retrieve-only` payload handed to a reader, and
            # every rank metric is computed against the gold answer —
            # putting them in the reader's input would widen the same
            # gold-leak surface `benchmarks/verify_clean.py` exists to
            # police. They live in `per_question_retrieval` instead and
            # surface aggregated under `retrieval_quality`.
        })

    result.query_time_s = time.monotonic() - t1

    return result


def print_results(result: BenchmarkResult) -> None:
    """Print formatted benchmark results."""
    print(f"\n{'='*60}")
    print(f"LoCoMo Benchmark Results: {result.conversation_id}")
    print(f"{'='*60}")
    print(f"Total QA pairs:    {result.total_qa}")
    print(f"Scored QA pairs:   {result.scored_qa}")
    print(f"Overall F1:        {result.overall_f1:.4f} ({result.overall_f1*100:.1f}%)"
          "  [reader-dependent; scorable categories only]")
    print(f"Ingest turns:      {result.ingest_turns}")
    print(f"Ingest time:       {result.ingest_time_s:.2f}s")
    print(f"Query time:        {result.query_time_s:.2f}s")
    if result.total_qa > 0:
        print(f"Avg query latency: {result.query_time_s / result.total_qa * 1000:.1f}ms")
    print()
    print("Per-category F1:")
    for cat in sorted(result.category_counts.keys()):
        name: str = CATEGORY_NAMES.get(cat, "unknown")
        count: int = result.category_counts.get(cat, 0)
        if cat in UNSCORABLE_CATEGORIES:
            print(f"  {cat}. {name:12s}  {NOT_APPLICABLE:>15s}  n={count}"
                  f"  ({UNSCORABLE_CATEGORY_REASON})")
            continue
        f1: float = result.category_f1(cat)
        print(f"  {cat}. {name:12s}  {f1:.4f} ({f1*100:.1f}%)  n={count}")
    print()
    rq: dict[str, float] = result.retrieval_quality()
    print("Retrieval quality (reader-independent, all categories):")
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in rq.items()))
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
    try:
        conversations: list[LoCoMoConversation] = load_locomo(args.data)
    except FileNotFoundError as exc:
        print(
            f"LoCoMo data not found at {args.data}: {exc}",
            file=sys.stderr,
        )
        sys.exit(2)
    print(f"Loaded {len(conversations)} conversations")

    if not conversations:
        print(
            f"No conversations loaded from {args.data}.",
            file=sys.stderr,
        )
        sys.exit(2)

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
            # Reader-dependent: scored by handing the retrieved context
            # to a scorer written for a model's answer, so the value
            # tracks the token budget as much as the ranking (#1160).
            # `overall_f1` is the mean over the scorable categories.
            "overall_f1": round(merged_for_output.overall_f1, 4),
            "total_qa": merged_for_output.total_qa,
            "scored_qa": merged_for_output.scored_qa,
            "category_f1": {
                str(cat): (
                    NOT_APPLICABLE if cat in UNSCORABLE_CATEGORIES
                    else round(merged_for_output.category_f1(cat), 4)
                )
                for cat in sorted(merged_for_output.category_counts.keys())
            },
            # Reader-independent: rank metrics over the retrieved list,
            # covering every question including the unscorable ones.
            "retrieval_quality": merged_for_output.retrieval_quality(),
            NOT_APPLICABLE_REASONS_KEY: {
                f"category_f1.{cat}": (
                    f"{CATEGORY_NAMES.get(cat, 'unknown')} — "
                    f"{UNSCORABLE_CATEGORY_REASON}"
                )
                for cat in sorted(UNSCORABLE_CATEGORIES)
            },
            "per_question": merged_for_output.per_question,
        }
        output_path: Path = Path(args.output)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nDetailed results written to {args.output}")


if __name__ == "__main__":
    main()
