"""MemoryAgentBench (MAB) benchmark adapter for aelfrice.

Loads the HuggingFace ai-hyz/MemoryAgentBench dataset, chunks context
text using NLTK sentence tokenization, ingests chunks into aelfrice
as sequential turns, queries for each question, and scores using
exact_match, substring_exact_match, and F1 metrics per the paper.

Reference: arXiv:2507.05257, ICLR 2026

Usage:
    uv run python benchmarks/mab_adapter.py --split Conflict_Resolution
    uv run python benchmarks/mab_adapter.py --split Conflict_Resolution --source factconsolidation_mh_262k
    uv run python benchmarks/mab_adapter.py --split Conflict_Resolution --retrieve-only /tmp/mab_retrieval.json
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
from pathlib import Path
from typing import Final

import nltk  # type: ignore[import-untyped]
import tiktoken  # type: ignore[import-untyped]
from datasets import load_dataset  # type: ignore[import-untyped]

from aelfrice.ingest import ingest_turn
from aelfrice.retrieval import retrieve
from aelfrice.store import MemoryStore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_DATASET: Final[str] = "ai-hyz/MemoryAgentBench"
VALID_SPLITS: Final[list[str]] = [
    "Accurate_Retrieval",
    "Test_Time_Learning",
    "Long_Range_Understanding",
    "Conflict_Resolution",
]
DEFAULT_CHUNK_SIZE: Final[int] = 4096
TIKTOKEN_MODEL: Final[str] = "gpt-4o"

# FactConsolidation prompt from the paper
FACTCONSOLIDATION_PROMPT: Final[str] = (
    "Pretend you are a knowledge management system. Each fact in the "
    "knowledge pool is provided with a serial number at the beginning, "
    "and the newer fact has larger serial number. You need to solve the "
    "conflicts of facts in the knowledge pool by finding the newest fact "
    "with larger serial number. You need to answer a question based on "
    "this rule. You should give a very concise answer without saying "
    "other words for the question **only** from the knowledge pool you "
    "have memorized rather than the real facts in real world."
)

# Paper baselines: Conflict Resolution split, 262K context
BASELINES: Final[dict[str, str]] = {
    "GPT-4o-mini long context (SH 262K)": "45%",
    "GPT-4o-mini long context (MH 262K)": "5%",
    "o4-mini (SH 6K)": "100%",
    "o4-mini (MH 6K)": "80%",
    "All methods (MH 262K)": "<=7%",
}


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_encoder: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    """Lazy-load tiktoken encoder."""
    global _encoder  # noqa: PLW0603
    if _encoder is None:
        _encoder = tiktoken.encoding_for_model(TIKTOKEN_MODEL)
    return _encoder


def _count_tokens(text: str) -> int:
    """Count tokens using tiktoken."""
    return len(_get_encoder().encode(text))


# ---------------------------------------------------------------------------
# Scoring (from the paper's evaluation code, no Porter stemming)
# ---------------------------------------------------------------------------


def normalize_answer(s: str) -> str:
    """Normalize answer: lowercase, strip punctuation, remove articles, collapse whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def score_exact_match(prediction: str, ground_truth: str) -> float:
    """Normalized string exact match."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def score_substring_exact_match(prediction: str, ground_truth: str) -> float:
    """Ground truth is substring of prediction (after normalization)."""
    return 1.0 if normalize_answer(ground_truth) in normalize_answer(prediction) else 0.0


def score_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 without stemming (per MAB paper)."""
    pred_tokens: list[str] = normalize_answer(prediction).split()
    gt_tokens: list[str] = normalize_answer(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    common: Counter[str] = Counter(pred_tokens) & Counter(gt_tokens)
    num_common: int = sum(common.values())
    if num_common == 0:
        return 0.0
    precision: float = num_common / len(pred_tokens)
    recall: float = num_common / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def score_multi_answer(
    prediction: str,
    ground_truths: list[str],
) -> dict[str, float]:
    """Score a prediction against multiple acceptable ground truth answers.

    Returns best score across all ground truths for each metric.
    """
    best_em: float = 0.0
    best_sem: float = 0.0
    best_f1: float = 0.0
    for gt in ground_truths:
        best_em = max(best_em, score_exact_match(prediction, gt))
        best_sem = max(best_sem, score_substring_exact_match(prediction, gt))
        best_f1 = max(best_f1, score_f1(prediction, gt))
    return {
        "exact_match": best_em,
        "substring_exact_match": best_sem,
        "f1": best_f1,
    }


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class MABRow:
    """One row from the MemoryAgentBench dataset."""

    context: str
    questions: list[str]
    answers: list[list[str]]
    metadata: dict[str, object]

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", "unknown"))


@dataclass
class MABResult:
    """Aggregated benchmark results for one or more rows."""

    label: str = ""
    total_questions: int = 0
    scores: dict[str, list[float]] = field(
        default_factory=lambda: {
            "exact_match": [],
            "substring_exact_match": [],
            "f1": [],
        }
    )
    ingest_chunks: int = 0
    ingest_time_s: float = 0.0
    query_time_s: float = 0.0
    per_question: list[dict[str, object]] = field(
        default_factory=lambda: list[dict[str, object]](),
    )
    ground_truth: list[dict[str, object]] = field(
        default_factory=lambda: list[dict[str, object]](),
    )

    def mean_score(self, metric: str) -> float:
        vals: list[float] = self.scores.get(metric, [])
        if not vals:
            return 0.0
        return sum(vals) / len(vals)


def merge_results(results: list[MABResult], label: str = "ALL") -> MABResult:
    """Merge multiple results into one aggregate."""
    merged: MABResult = MABResult(label=label)
    for r in results:
        merged.total_questions += r.total_questions
        for metric in merged.scores:
            merged.scores[metric].extend(r.scores.get(metric, []))
        merged.ingest_chunks += r.ingest_chunks
        merged.ingest_time_s += r.ingest_time_s
        merged.query_time_s += r.query_time_s
        merged.per_question.extend(r.per_question)
        merged.ground_truth.extend(r.ground_truth)
    return merged


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_context(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> list[str]:
    """Chunk text into segments of approximately chunk_size tokens.

    Uses NLTK sentence tokenization to avoid splitting mid-sentence.
    """
    nltk.download("punkt_tab", quiet=True)  # type: ignore[no-untyped-call]
    sentences: list[str] = nltk.sent_tokenize(text)  # type: ignore[no-untyped-call]

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_tokens: int = 0

    for sentence in sentences:
        sent_tokens: int = _count_tokens(sentence)
        if current_tokens + sent_tokens > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_tokens = 0
        current_chunk.append(sentence)
        current_tokens += sent_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_mab_split(split: str, source_filter: str | None = None) -> list[MABRow]:
    """Load a split from the HuggingFace MemoryAgentBench dataset."""
    if split not in VALID_SPLITS:
        valid: str = ", ".join(VALID_SPLITS)
        msg: str = f"Invalid split '{split}'. Valid splits: {valid}"
        raise ValueError(msg)

    ds = load_dataset(HF_DATASET, split=split)  # type: ignore[no-untyped-call]
    rows: list[MABRow] = []
    for raw_item in ds:  # type: ignore[union-attr]
        item: dict[str, object] = dict(raw_item)  # type: ignore[arg-type]
        raw_meta: object = item.get("metadata", {})
        metadata: dict[str, object]
        if isinstance(raw_meta, str):
            metadata = json.loads(raw_meta)
        elif isinstance(raw_meta, dict):
            metadata = dict(raw_meta)  # type: ignore[arg-type]
        else:
            metadata = {}
        raw_answers: object = item.get("answers", [])
        answers: list[list[str]] = [
            [str(x) for x in a] for a in raw_answers  # type: ignore[union-attr]
        ] if isinstance(raw_answers, list) else []
        row: MABRow = MABRow(
            context=str(item.get("context", "")),
            questions=[str(q) for q in item.get("questions", [])],  # type: ignore[union-attr]
            answers=answers,
            metadata=metadata,
        )
        if source_filter is not None and row.source != source_filter:
            continue
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Ingest adapter
# ---------------------------------------------------------------------------


def ingest_context(store: MemoryStore, row: MABRow, chunk_size: int) -> int:
    """Chunk and ingest context text into aelfrice.

    Each chunk is ingested as a separate turn to simulate multi-turn
    conversation. Returns the number of chunks ingested.
    """
    session = store.create_session(
        model="mab-benchmark",
        project_context=f"MAB {row.source}",
    )
    chunks: list[str] = chunk_context(row.context, chunk_size)
    for i, chunk in enumerate(chunks):
        ingest_turn(
            store=store,
            text=chunk,
            source="mab",
            session_id=session.id,
            source_id=f"chunk_{i:05d}",
        )
    store.complete_session(session.id)
    return len(chunks)


# ---------------------------------------------------------------------------
# Query adapter
# ---------------------------------------------------------------------------


def query_aelfrice(store: MemoryStore, question: str, budget: int = 2000) -> str:
    """Query aelfrice and return retrieved belief content."""
    result = retrieve(
        store=store,
        query=question,
        budget=budget,
        include_locked=False,
        use_hrr=True,
        use_bfs=True,
    )
    parts: list[str] = [b.content for b in result.beliefs]
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_row(
    row: MABRow,
    db_dir: str,
    row_idx: int,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    budget: int = 2000,
    subset: int | None = None,
) -> MABResult:
    """Run the full benchmark pipeline on one dataset row.

    Uses a fresh DB per row for isolation.
    """
    db_path: str = f"{db_dir}/mab_row_{row_idx:04d}.db"
    store: MemoryStore = MemoryStore(db_path)

    result: MABResult = MABResult(label=row.source)

    # Ingest context
    t0: float = time.monotonic()
    result.ingest_chunks = ingest_context(store, row, chunk_size)
    result.ingest_time_s = time.monotonic() - t0

    # Query and score
    questions: list[str] = row.questions
    answers: list[list[str]] = row.answers
    if subset is not None:
        questions = questions[:subset]
        answers = answers[:subset]

    t1: float = time.monotonic()
    for q_idx, (question, answer_list) in enumerate(zip(questions, answers)):
        prediction: str = query_aelfrice(store, question, budget=budget)
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
        })
        result.ground_truth.append({
            "id": q_idx,
            "row_idx": row_idx,
            "answers": answer_list,
        })

    result.query_time_s = time.monotonic() - t1
    return result


def print_results(result: MABResult) -> None:
    """Print formatted benchmark results."""
    print(f"\n{'=' * 60}")
    print(f"MAB Results: {result.label}")
    print(f"{'=' * 60}")
    print(f"Total questions:         {result.total_questions}")
    print(f"Exact match:             {result.mean_score('exact_match'):.4f} ({result.mean_score('exact_match') * 100:.1f}%)")
    print(f"Substring exact match:   {result.mean_score('substring_exact_match'):.4f} ({result.mean_score('substring_exact_match') * 100:.1f}%)")
    print(f"F1:                      {result.mean_score('f1'):.4f} ({result.mean_score('f1') * 100:.1f}%)")
    print(f"Chunks ingested:         {result.ingest_chunks}")
    print(f"Ingest time:             {result.ingest_time_s:.2f}s")
    print(f"Query time:              {result.query_time_s:.2f}s")
    if result.total_questions > 0:
        print(f"Avg query latency:       {result.query_time_s / result.total_questions * 1000:.1f}ms")
    print()

    # Paper baselines
    print("Paper baselines (Conflict Resolution):")
    for name, score in BASELINES.items():
        print(f"  {name}: {score}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Run MemoryAgentBench benchmark on aelfrice",
    )
    parser.add_argument(
        "--split", default="Conflict_Resolution",
        choices=VALID_SPLITS,
        help="Dataset split to evaluate (default: Conflict_Resolution)",
    )
    parser.add_argument(
        "--source", default=None,
        help="Filter by metadata.source (e.g., factconsolidation_mh_262k)",
    )
    parser.add_argument(
        "--rows", type=int, default=None,
        help="Limit to first N rows (default: all)",
    )
    parser.add_argument(
        "--subset", type=int, default=None,
        help="Limit to first N questions per row (for debugging)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help=f"Token chunk size for context ingestion (default: {DEFAULT_CHUNK_SIZE})",
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
        help="Run retrieval only, write question+context pairs to PATH for LLM answer generation",
    )
    args: argparse.Namespace = parser.parse_args()

    print(f"Loading MAB dataset split: {args.split}")
    if args.source:
        print(f"Filtering by source: {args.source}")
    rows: list[MABRow] = load_mab_split(args.split, source_filter=args.source)
    print(f"Loaded {len(rows)} rows")

    if not rows:
        print("No rows matched. Check --split and --source values.")
        return

    if args.rows is not None:
        rows = rows[:args.rows]
        print(f"Using first {len(rows)} rows")

    results: list[MABResult] = []

    with tempfile.TemporaryDirectory(prefix="mab_bench_") as tmpdir:
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

            row_result: MABResult = run_row(
                row, tmpdir, idx,
                chunk_size=args.chunk_size,
                budget=args.budget,
                subset=args.subset,
            )
            results.append(row_result)
            if not args.retrieve_only:
                print_results(row_result)

    # Retrieve-only mode: write question+context (NO answers) for LLM reader
    # Ground truth written to a SEPARATE file for scoring after generation
    if args.retrieve_only:
        all_items: list[dict[str, object]] = []
        all_gt: list[dict[str, object]] = []
        for r in results:
            all_items.extend(r.per_question)
            all_gt.extend(r.ground_truth)
        retrieve_path: Path = Path(args.retrieve_only)
        gt_path: Path = retrieve_path.with_name(
            retrieve_path.stem + "_gt" + retrieve_path.suffix,
        )
        with retrieve_path.open("w", encoding="utf-8") as f:
            json.dump(all_items, f, indent=2)
        with gt_path.open("w", encoding="utf-8") as f:
            json.dump(all_gt, f, indent=2)
        total_q: int = sum(r.total_questions for r in results)
        print(f"Wrote {total_q} retrieval results to {args.retrieve_only}")
        print(f"Wrote {total_q} ground truth items to {gt_path}")
        print("ISOLATION: retrieval file contains NO ground truth answers")
        print("Next step: run LLM reader on retrieval file, then score against GT")
        return

    # Aggregate if multiple rows
    if len(results) > 1:
        merged: MABResult = merge_results(results, label=f"{args.split} (all)")
        print_results(merged)

    # Group by source for per-source breakdown
    source_map: dict[str, list[MABResult]] = {}
    for r in results:
        if r.label not in source_map:
            source_map[r.label] = []
        source_map[r.label].append(r)

    if len(source_map) > 1:
        print(f"\n{'=' * 60}")
        print("Per-source breakdown:")
        print(f"{'=' * 60}")
        for src, src_results in sorted(source_map.items()):
            src_merged: MABResult = merge_results(src_results, label=src)
            sem: float = src_merged.mean_score("substring_exact_match")
            f1: float = src_merged.mean_score("f1")
            print(f"  {src}: SEM={sem * 100:.1f}% F1={f1 * 100:.1f}% n={src_merged.total_questions}")

    # Write detailed output
    if args.output:
        out_result: MABResult = merge_results(results) if len(results) > 1 else results[0]
        output_data: dict[str, object] = {
            "split": args.split,
            "source_filter": args.source,
            "total_questions": out_result.total_questions,
            "exact_match": round(out_result.mean_score("exact_match"), 4),
            "substring_exact_match": round(out_result.mean_score("substring_exact_match"), 4),
            "f1": round(out_result.mean_score("f1"), 4),
            "per_question": out_result.per_question,
        }
        output_path: Path = Path(args.output)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nDetailed results written to {args.output}")


if __name__ == "__main__":
    main()
