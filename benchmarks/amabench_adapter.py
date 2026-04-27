"""AMA-Bench adapter for aelfrice.

Ingests AMA-Bench (Zhao et al., ICLR 2026 Workshop) agentic trajectories
into aelfrice, runs retrieval for QA evaluation, and reports per-domain
and per-QA-type statistics.

Data source: HuggingFace AMA-bench/AMA-bench, split 'test' (208 episodes)

Usage:
    uv run python benchmarks/amabench_adapter.py --retrieve-only results.json
    uv run python benchmarks/amabench_adapter.py --domain Game --max-episodes 10
    uv run python benchmarks/amabench_adapter.py --qa-type A --retrieve-only recall.json
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from datasets import load_dataset  # type: ignore[import-untyped]

from aelfrice.ingest import ingest_turn
from aelfrice.retrieval import retrieve
from aelfrice.store import MemoryStore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_ID: Final[str] = "AMA-bench/AMA-bench"
DATASET_SPLIT: Final[str] = "test"

QA_TYPE_NAMES: Final[dict[str, str]] = {
    "A": "Recall (temporal/sequential)",
    "B": "Causal Inference",
    "C": "State Updating",
    "D": "State Abstraction",
}

# Published baselines (average accuracy %)
BASELINES: Final[dict[str, float]] = {
    "BM25 (Qwen3-32B)": 34.36,
    "HippoRAG2": 44.80,
    "MemoRAG": 46.06,
    "AMA-Agent": 57.22,
    "GPT 5.2 long-context": 72.26,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TrajectoryStep:
    """One action-observation pair from an episode trajectory."""

    turn_idx: int
    action: str
    observation: str


@dataclass
class QAPair:
    """One QA evaluation pair from an episode."""

    question: str
    answer: str
    qa_type: str  # A, B, C, or D
    question_uuid: str


@dataclass
class Episode:
    """One AMA-Bench episode with trajectory and QA pairs."""

    episode_id: str
    task: str
    task_type: str
    domain: str
    success: bool
    num_turns: int
    total_tokens: int
    trajectory: list[TrajectoryStep]
    qa_pairs: list[QAPair]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_amabench(
    domain: str | None = None,
    qa_type: str | None = None,
    max_episodes: int | None = None,
) -> list[Episode]:
    """Load AMA-Bench dataset from HuggingFace.

    Args:
        domain: Filter to a specific domain (e.g. 'Game', 'Text2SQL').
        qa_type: Filter QA pairs to a specific type (A/B/C/D).
        max_episodes: Limit number of episodes loaded.

    Returns:
        List of Episode objects.
    """
    ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)  # type: ignore[no-untyped-call]

    episodes: list[Episode] = []
    for raw_row in ds:  # type: ignore[union-attr]
        row: dict[str, object] = dict(raw_row)  # type: ignore[arg-type]
        ep_domain: str = str(row["domain"])
        if domain is not None and ep_domain != domain:
            continue

        # Parse trajectory
        raw_traj: list[dict[str, object]] = row["trajectory"]  # type: ignore[assignment]
        trajectory: list[TrajectoryStep] = []
        for step in raw_traj:
            trajectory.append(TrajectoryStep(
                turn_idx=int(step["turn_idx"]),  # type: ignore[arg-type]
                action=str(step["action"]),
                observation=str(step["observation"]),
            ))

        # Parse QA pairs, optionally filtering by type
        raw_qa: list[dict[str, object]] = row["qa_pairs"]  # type: ignore[assignment]
        qa_pairs: list[QAPair] = []
        for q in raw_qa:
            q_type: str = str(q["type"])
            if qa_type is not None and q_type != qa_type:
                continue
            qa_pairs.append(QAPair(
                question=str(q["question"]),
                answer=str(q["answer"]),
                qa_type=q_type,
                question_uuid=str(q["question_uuid"]),
            ))

        # Skip episodes with no matching QA pairs after filtering
        if qa_type is not None and not qa_pairs:
            continue

        episodes.append(Episode(
            episode_id=str(row["episode_id"]),
            task=str(row["task"]),
            task_type=str(row["task_type"]),
            domain=ep_domain,
            success=bool(row["success"]),
            num_turns=int(row["num_turns"]),  # type: ignore[arg-type]
            total_tokens=int(row["total_tokens"]),  # type: ignore[arg-type]
            trajectory=trajectory,
            qa_pairs=qa_pairs,
        ))

        if max_episodes is not None and len(episodes) >= max_episodes:
            break

    return episodes


# ---------------------------------------------------------------------------
# Ingest adapter
# ---------------------------------------------------------------------------


def ingest_episode(store: MemoryStore, episode: Episode) -> int:
    """Ingest all trajectory steps of an episode into aelfrice.

    Each step is formatted as:
        [Step {turn_idx}] Action: {action}
        Observation: {observation}

    Returns total turns ingested.
    """
    am_session = store.create_session(
        model="amabench-benchmark",
        project_context=f"{episode.episode_id} ({episode.domain}: {episode.task_type})",
    )

    for step in episode.trajectory:
        text: str = (
            f"[Step {step.turn_idx}] Action: {step.action}\n"
            f"Observation: {step.observation}"
        )
        ingest_turn(
            store=store,
            text=text,
            source="amabench",
            session_id=am_session.id,
            source_id=f"{episode.episode_id}:step{step.turn_idx}",
        )

    store.complete_session(am_session.id)
    return len(episode.trajectory)


# ---------------------------------------------------------------------------
# Query adapter
# ---------------------------------------------------------------------------


def query_aelfrice(store: MemoryStore, question: str, budget: int = 2000) -> str:
    """Retrieve relevant beliefs from aelfrice for a question."""
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
# Results
# ---------------------------------------------------------------------------


@dataclass
class EpisodeResult:
    """Results for one episode."""

    episode_id: str = ""
    domain: str = ""
    total_qa: int = 0
    ingest_turns: int = 0
    ingest_time_s: float = 0.0
    query_time_s: float = 0.0
    per_question: list[dict[str, object]] = field(
        default_factory=lambda: list[dict[str, object]](),
    )
    ground_truth: list[dict[str, object]] = field(
        default_factory=lambda: list[dict[str, object]](),
    )


@dataclass
class AggregateResult:
    """Aggregated benchmark results across episodes."""

    total_episodes: int = 0
    total_qa: int = 0
    total_ingest_turns: int = 0
    total_ingest_time_s: float = 0.0
    total_query_time_s: float = 0.0
    domain_counts: dict[str, int] = field(
        default_factory=lambda: dict[str, int](),
    )
    domain_qa_counts: dict[str, int] = field(
        default_factory=lambda: dict[str, int](),
    )
    type_counts: dict[str, int] = field(
        default_factory=lambda: dict[str, int](),
    )
    per_question: list[dict[str, object]] = field(
        default_factory=lambda: list[dict[str, object]](),
    )
    ground_truth: list[dict[str, object]] = field(
        default_factory=lambda: list[dict[str, object]](),
    )


def aggregate_results(results: list[EpisodeResult]) -> AggregateResult:
    """Merge per-episode results into an aggregate."""
    agg: AggregateResult = AggregateResult()
    for r in results:
        agg.total_episodes += 1
        agg.total_qa += r.total_qa
        agg.total_ingest_turns += r.ingest_turns
        agg.total_ingest_time_s += r.ingest_time_s
        agg.total_query_time_s += r.query_time_s
        agg.per_question.extend(r.per_question)
        agg.ground_truth.extend(r.ground_truth)

        agg.domain_counts[r.domain] = agg.domain_counts.get(r.domain, 0) + 1
        agg.domain_qa_counts[r.domain] = (
            agg.domain_qa_counts.get(r.domain, 0) + r.total_qa
        )

        for q in r.per_question:
            qt: str = str(q["qa_type"])
            agg.type_counts[qt] = agg.type_counts.get(qt, 0) + 1

    return agg


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_episode(
    episode: Episode,
    db_dir: str,
    budget: int = 2000,
) -> EpisodeResult:
    """Run the retrieval pipeline on one episode.

    Uses a fresh DB per episode for isolation.
    """
    db_path: str = f"{db_dir}/{episode.episode_id}.db"
    store: MemoryStore = MemoryStore(db_path)

    result: EpisodeResult = EpisodeResult(
        episode_id=episode.episode_id,
        domain=episode.domain,
    )

    # Stage 1: Ingest trajectory
    t0: float = time.monotonic()
    result.ingest_turns = ingest_episode(store, episode)
    result.ingest_time_s = time.monotonic() - t0

    # Stage 2: Retrieve for each question
    t1: float = time.monotonic()
    for qa in episode.qa_pairs:
        context: str = query_aelfrice(store, qa.question, budget=budget)

        result.total_qa += 1
        result.per_question.append({
            "episode_id": episode.episode_id,
            "domain": episode.domain,
            "task_type": episode.task_type,
            "question": qa.question,
            "qa_type": qa.qa_type,
            "qa_type_name": QA_TYPE_NAMES.get(qa.qa_type, "unknown"),
            "question_uuid": qa.question_uuid,
            "context": context,
        })
        result.ground_truth.append({
            "question_uuid": qa.question_uuid,
            "answer": qa.answer,
            "qa_type": qa.qa_type,
        })

    result.query_time_s = time.monotonic() - t1
    return result


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def print_results(agg: AggregateResult) -> None:
    """Print formatted retrieval statistics."""
    print(f"\n{'=' * 60}")
    print("AMA-Bench Retrieval Results")
    print(f"{'=' * 60}")
    print(f"Episodes processed: {agg.total_episodes}")
    print(f"Total QA pairs:     {agg.total_qa}")
    print(f"Total turns ingested: {agg.total_ingest_turns}")
    print(f"Ingest time:        {agg.total_ingest_time_s:.2f}s")
    print(f"Query time:         {agg.total_query_time_s:.2f}s")
    if agg.total_qa > 0:
        avg_ms: float = agg.total_query_time_s / agg.total_qa * 1000
        print(f"Avg query latency:  {avg_ms:.1f}ms")

    # Per-domain breakdown
    print(f"\n{'- ' * 30}")
    print("Per-domain breakdown:")
    for domain in sorted(agg.domain_counts.keys()):
        ep_count: int = agg.domain_counts[domain]
        qa_count: int = agg.domain_qa_counts.get(domain, 0)
        print(f"  {domain:25s}  episodes={ep_count:3d}  questions={qa_count:4d}")

    # Per-QA-type breakdown
    print(f"\n{'- ' * 30}")
    print("Per-QA-type breakdown:")
    for qt in sorted(agg.type_counts.keys()):
        count: int = agg.type_counts[qt]
        name: str = QA_TYPE_NAMES.get(qt, "unknown")
        print(f"  {qt}: {name:30s}  n={count}")

    # Context length stats
    if agg.per_question:
        lengths: list[int] = [
            len(str(q["context"]).split()) for q in agg.per_question
        ]
        avg_len: float = sum(lengths) / len(lengths)
        min_len: int = min(lengths)
        max_len: int = max(lengths)
        print(f"\n{'- ' * 30}")
        print("Retrieved context stats (word count):")
        print(f"  Mean: {avg_len:.1f}  Min: {min_len}  Max: {max_len}")

    # Reference baselines
    print(f"\n{'- ' * 30}")
    print("Reference baselines (accuracy %, LLM-judged):")
    for name, score in BASELINES.items():
        print(f"  {name:30s}  {score:.2f}%")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Run AMA-Bench benchmark on aelfrice",
    )
    parser.add_argument(
        "--max-episodes", type=int, default=None,
        help="Limit to first N episodes (default: all 208)",
    )
    parser.add_argument(
        "--domain", default=None,
        help="Filter to a specific domain (e.g. Game, Text2SQL, 'Software Engineer')",
    )
    parser.add_argument(
        "--qa-type", default=None, choices=["A", "B", "C", "D"],
        help="Filter QA pairs to a specific type (A=Recall, B=Causal, C=State, D=Abstraction)",
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
        help="Run retrieval only, write question+context pairs to PATH for LLM judge scoring",
    )
    args: argparse.Namespace = parser.parse_args()

    print("Loading AMA-Bench dataset from HuggingFace...")
    episodes: list[Episode] = load_amabench(
        domain=args.domain,
        qa_type=args.qa_type,
        max_episodes=args.max_episodes,
    )
    print(f"Loaded {len(episodes)} episodes")

    if not episodes:
        print("No episodes matched filters. Exiting.")
        return

    # Show domain distribution
    domain_dist: dict[str, int] = {}
    for ep in episodes:
        domain_dist[ep.domain] = domain_dist.get(ep.domain, 0) + 1
    for d in sorted(domain_dist.keys()):
        print(f"  {d}: {domain_dist[d]} episodes")

    results: list[EpisodeResult] = []

    with tempfile.TemporaryDirectory(prefix="amabench_") as tmpdir:
        for i, episode in enumerate(episodes):
            qa_count: int = len(episode.qa_pairs)
            print(
                f"\n--- [{i + 1}/{len(episodes)}] {episode.episode_id}: "
                f"{episode.domain}, {len(episode.trajectory)} steps, "
                f"{qa_count} QA pairs ---"
            )

            ep_result: EpisodeResult = run_episode(
                episode, tmpdir, budget=args.budget,
            )
            results.append(ep_result)

            print(
                f"  Ingested {ep_result.ingest_turns} turns in "
                f"{ep_result.ingest_time_s:.2f}s, "
                f"queried {ep_result.total_qa} questions in "
                f"{ep_result.query_time_s:.2f}s"
            )

    agg: AggregateResult = aggregate_results(results)

    # Retrieve-only: retrieval (NO answers) + separate GT file
    if args.retrieve_only:
        retrieve_path: Path = Path(args.retrieve_only)
        gt_path: Path = retrieve_path.with_name(
            retrieve_path.stem + "_gt" + retrieve_path.suffix,
        )
        with retrieve_path.open("w", encoding="utf-8") as f:
            json.dump(agg.per_question, f, indent=2)
        with gt_path.open("w", encoding="utf-8") as f:
            json.dump(agg.ground_truth, f, indent=2)
        print(f"\nWrote {agg.total_qa} retrieval results to {args.retrieve_only}")
        print(f"Wrote {agg.total_qa} ground truth items to {gt_path}")
        print("ISOLATION: retrieval file contains NO ground truth answers")
        return

    print_results(agg)

    # Write detailed output
    if args.output:
        output_data: dict[str, object] = {
            "total_episodes": agg.total_episodes,
            "total_qa": agg.total_qa,
            "domain_counts": agg.domain_counts,
            "type_counts": agg.type_counts,
            "per_question": agg.per_question,
        }
        output_path: Path = Path(args.output)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nDetailed results written to {args.output}")


if __name__ == "__main__":
    main()
