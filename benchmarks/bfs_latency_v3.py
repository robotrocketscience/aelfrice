"""v3.0 BFS latency bench — gate for #739 default-flip.

Measures retrieval p50/p95/p99 latency with and without BFS on a
≥10k-belief / ≥25k-edge synthetic store, all v3.0 default lanes
engaged.

Gate (per #739 + operator decision in PR #754):

    delta_p50  ≤  5 ms  (bfs_on − bfs_off)
    delta_p95  ≤ 50 ms  (bfs_on − bfs_off)
    max / median ratio ≤ 10x with BFS on (tail sanity)

The original #739 body proposed absolute thresholds (p50 ≤ 25 ms,
p95 ≤ 100 ms) borrowed from the v1.3 acceptance band published in
``docs/bfs_multihop.md:518-525``. Run 1 (commit 62ce0b3) showed the
v3.0 baseline alone already exceeds those absolutes — eight+ minors
of HRR / BM25F / heat / clustering / posterior-rerank work have
moved the baseline. The gate was reframed to a **delta** criterion
in PR #754 so what's measured is what flipping the default actually
costs, not whether v1.3's absolute band still holds.

Corpus shape
------------

200 topics × 50 beliefs = 10,000 beliefs. Each topic has three
named entities (``topic_NN_entity_K``) embedded in belief bodies so
the L2.5 entity-index path has something to hit. Edges follow a
deterministic intra-topic chain (each belief points SUPPORTS-style
to the next two in its topic) plus one cross-topic CITES per topic,
yielding ~25k edges.

The corpus is *not* a real-world distribution. The point is to
exercise the v3.0 default retrieval stack against a store whose
size and edge fan-out are in the same order of magnitude as the
maintainer's production store (~14k beliefs as of 2026-05-13). A
labelled-corpus version would be a strictly better benchmark; this
one is what can ship without a private-data dependency.

Usage
-----

    uv run python benchmarks/bfs_latency_v3.py
    uv run python benchmarks/bfs_latency_v3.py --beliefs 10000 \\
        --output benchmarks/results/bfs_latency_v3/<run-id>.json

Output is a single JSON document per the schema in
``_emit_report``. The default output path embeds the short git
commit so re-runs land beside each other under
``benchmarks/results/bfs_latency_v3/``.

Refs
----

- Issue #739: BFS default-flip gated on this bench.
- ``src/aelfrice/retrieval.py::is_bfs_enabled`` line 1144 — the
  resolver whose default this bench is gating.
- ``src/aelfrice/benchmark.py`` — convention used for the smaller
  v0.9 latency floor and multi-hop accuracy harnesses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Sequence

from aelfrice import __version__ as AELFRICE_VERSION
from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_CITES,
    EDGE_SUPPORTS,
    LOCK_NONE,
    ORIGIN_UNKNOWN,
    Belief,
    Edge,
)
from aelfrice.retrieval import retrieve
from aelfrice.store import MemoryStore

HARNESS_NAME: Final[str] = "bfs_latency_v3"
HARNESS_VERSION: Final[str] = "1"

DEFAULT_BELIEF_COUNT: Final[int] = 10_000
DEFAULT_TOPICS: Final[int] = 200
DEFAULT_ITERATIONS_PER_QUERY: Final[int] = 10
DEFAULT_QUERIES: Final[int] = 30
DEFAULT_WARMUP_ITERATIONS: Final[int] = 3

# Gate thresholds per #739 + PR #754 Option 1 reframe.
# bfs_on − bfs_off, evaluated on the same corpus.
GATE_DELTA_P50_MS: Final[float] = 5.0
GATE_DELTA_P95_MS: Final[float] = 50.0
# Tail-sanity guard on the BFS-on arm — kept from the original
# #739 spec since a pathological tail isn't bounded by a delta.
GATE_MAX_OVER_MEDIAN_RATIO: Final[float] = 10.0

# A handful of stop-style words used by the prompt-shape gate
# (``aelfrice.expansion_gate``) to flag a prompt as "broad" and
# short-circuit BFS. The bench queries are intentionally precise
# (entity-name + verb) so BFS-on actually fires; this list exists
# only as a sanity guard while authoring queries.
_BROAD_HINTS: Final[frozenset[str]] = frozenset({
    "what", "how", "why", "when", "where", "who", "tell", "explain",
})


# --- Corpus generation ------------------------------------------------------


def _belief_content(topic: int, idx: int) -> str:
    """Deterministic belief body referencing two of the topic's three entities.

    Body shape mimics a fact line: ``topic_NN_entity_A interacts_with
    topic_NN_entity_B under condition_K``. The entities serve as L2.5
    anchors; the verb / condition tail keeps BM25 from collapsing all
    50 beliefs in a topic to identical scores.
    """
    a = idx % 3
    b = (idx + 1) % 3
    return (
        f"topic_{topic:03d}_entity_{a} relates_to "
        f"topic_{topic:03d}_entity_{b} under condition_{idx:02d}; "
        f"observation rank_{idx} stable across replays."
    )


def _belief_id(topic: int, idx: int) -> str:
    return f"bench_t{topic:03d}_b{idx:02d}"


def _anchor_for(topic: int, idx: int) -> str:
    """Anchor text for the intra-topic SUPPORTS edge."""
    return f"topic_{topic:03d}_entity_{idx % 3}"


@dataclass(frozen=True)
class CorpusSpec:
    beliefs: int
    edges: int
    topics: int


def seed_corpus(
    store: MemoryStore,
    *,
    belief_count: int = DEFAULT_BELIEF_COUNT,
    topics: int = DEFAULT_TOPICS,
    created_at: str = "2026-05-13T00:00:00Z",
) -> CorpusSpec:
    """Insert a deterministic synthetic corpus into ``store``.

    Per-belief invariants:
      - id pattern ``bench_tNNN_bMM``
      - body contains exactly two ``topic_NNN_entity_K`` references
      - alpha=1, beta=1 (Jeffreys), unlocked, factual

    Edges:
      - SUPPORTS chain within topic: ``bench_tNNN_bMM`` -> ``b(M+1) %
        per_topic`` -> ``b(M+2) % per_topic``. Two SUPPORTS per
        belief.
      - One CITES per topic: from ``b00`` of topic_t to ``b00`` of
        ``(t + 7) % topics``. Cross-topic seed for BFS.

    Returns the realised counts for both beliefs and edges so the
    caller can record them in the report.
    """
    if belief_count <= 0:
        raise ValueError(f"belief_count must be positive: {belief_count!r}")
    if topics <= 0 or belief_count % topics != 0:
        raise ValueError(
            "belief_count must be a positive multiple of topics; "
            f"got belief_count={belief_count}, topics={topics}"
        )
    per_topic = belief_count // topics

    inserted = 0
    for t in range(topics):
        for i in range(per_topic):
            content = _belief_content(t, i)
            store.insert_belief(
                Belief(
                    id=_belief_id(t, i),
                    content=content,
                    content_hash=hashlib.sha256(content.encode()).hexdigest(),
                    alpha=1.0,
                    beta=1.0,
                    type=BELIEF_FACTUAL,
                    lock_level=LOCK_NONE,
                    locked_at=None,
                    demotion_pressure=0,
                    created_at=created_at,
                    last_retrieved_at=None,
                    origin=ORIGIN_UNKNOWN,
                )
            )
            inserted += 1

    edge_count = 0
    for t in range(topics):
        for i in range(per_topic):
            src = _belief_id(t, i)
            for hop in (1, 2):
                dst = _belief_id(t, (i + hop) % per_topic)
                store.insert_edge(
                    Edge(
                        src=src, dst=dst, type=EDGE_SUPPORTS,
                        weight=1.0, anchor_text=_anchor_for(t, i),
                    )
                )
                edge_count += 1
        # cross-topic CITES from topic_t's b00 to topic_(t+7)'s b00.
        store.insert_edge(
            Edge(
                src=_belief_id(t, 0),
                dst=_belief_id((t + 7) % topics, 0),
                type=EDGE_CITES,
                weight=0.5,
                anchor_text=f"topic_{t:03d}_entity_0",
            )
        )
        edge_count += 1

    return CorpusSpec(beliefs=inserted, edges=edge_count, topics=topics)


def synth_queries(
    *,
    count: int = DEFAULT_QUERIES,
    topics: int = DEFAULT_TOPICS,
    belief_count: int = DEFAULT_BELIEF_COUNT,
) -> tuple[str, ...]:
    """Deterministic precise queries that exercise L1+L2.5+BFS together.

    Each query is shaped ``topic_NNN_entity_K relates_to condition_MM``
    — precise enough that the #741 prompt-shape gate runs BFS rather
    than short-circuiting it. Queries are spread across the topic
    space (every ``topics // count`` topics) so the bench does not
    hammer the same FTS slice repeatedly.
    """
    if count <= 0:
        raise ValueError(f"count must be positive: {count!r}")
    if topics <= 0 or count > topics:
        raise ValueError(
            f"count={count} must fit within topics={topics}"
        )
    per_topic = belief_count // topics
    stride = max(1, topics // count)
    out: list[str] = []
    for i in range(count):
        t = (i * stride) % topics
        belief_idx = i % per_topic
        q = (
            f"topic_{t:03d}_entity_{belief_idx % 3} relates_to "
            f"condition_{belief_idx:02d}"
        )
        # Sanity guard so a future author doesn't accidentally hand
        # the gate a broad shape and silently kill the BFS-on arm.
        first_word = q.split(None, 1)[0].lower().strip("?,.")
        if first_word in _BROAD_HINTS:
            raise AssertionError(
                f"query {q!r} starts with broad hint {first_word!r}; "
                f"will be gated out of BFS by aelfrice.expansion_gate."
            )
        out.append(q)
    return tuple(out)


# --- Timing -----------------------------------------------------------------


@dataclass(frozen=True)
class ArmResult:
    label: str
    samples: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    mean_ms: float
    min_ms: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _percentile(sorted_samples: Sequence[float], p: float) -> float:
    """Nearest-rank percentile (1 ≤ p ≤ 100). Sequence must be sorted."""
    if not sorted_samples:
        return 0.0
    if p <= 0:
        return sorted_samples[0]
    if p >= 100:
        return sorted_samples[-1]
    # Nearest-rank: index = ceil(p/100 * n) - 1.
    n = len(sorted_samples)
    rank = max(1, min(n, int((p / 100.0) * n + 0.999999)))
    return sorted_samples[rank - 1]


def time_arm(
    store: MemoryStore,
    queries: Sequence[str],
    *,
    label: str,
    bfs_enabled: bool,
    iterations: int,
    warmup: int = DEFAULT_WARMUP_ITERATIONS,
) -> ArmResult:
    """Run ``retrieve()`` against ``store`` for each query × iterations.

    All non-BFS kwargs are left to retrieve()'s resolver defaults,
    matching the v3.0 install shape (L0+L1+L2.5+HRR-cache off-cold +
    BM25F + heat-kernel off-by-default + posterior-rerank). Only
    ``bfs_enabled`` is toggled between arms.

    ``warmup`` discardable iterations per query absorb HRR / BM25F
    cache warm-up so the percentile distribution reflects steady-
    state retrieval, not first-touch cost.
    """
    if iterations <= 0:
        raise ValueError(f"iterations must be positive: {iterations!r}")
    if warmup < 0:
        raise ValueError(f"warmup must be non-negative: {warmup!r}")

    for q in queries:
        for _ in range(warmup):
            retrieve(store, q, bfs_enabled=bfs_enabled)

    samples_ms: list[float] = []
    for q in queries:
        for _ in range(iterations):
            t0 = time.perf_counter()
            retrieve(store, q, bfs_enabled=bfs_enabled)
            samples_ms.append((time.perf_counter() - t0) * 1000.0)

    samples_sorted = sorted(samples_ms)
    return ArmResult(
        label=label,
        samples=len(samples_ms),
        p50_ms=_percentile(samples_sorted, 50),
        p95_ms=_percentile(samples_sorted, 95),
        p99_ms=_percentile(samples_sorted, 99),
        max_ms=samples_sorted[-1],
        mean_ms=statistics.fmean(samples_ms),
        min_ms=samples_sorted[0],
    )


# --- Gate -------------------------------------------------------------------


@dataclass(frozen=True)
class GateResult:
    delta_p50_ms: float
    delta_p95_ms: float
    delta_p50_pass: bool
    delta_p95_pass: bool
    tail_ratio: float
    tail_ratio_pass: bool
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_gate(bfs_off: ArmResult, bfs_on: ArmResult) -> GateResult:
    """Apply the delta-reframed gate (PR #754 Option 1).

    The cost being measured is what flipping `is_bfs_enabled()`
    default to True adds on top of the same-corpus BFS-off arm —
    not whether the v1.3 absolute band still holds against the
    v3.0 retrieval stack.
    """
    delta_p50 = bfs_on.p50_ms - bfs_off.p50_ms
    delta_p95 = bfs_on.p95_ms - bfs_off.p95_ms
    tail_ratio = (
        bfs_on.max_ms / bfs_on.p50_ms
        if bfs_on.p50_ms > 0 else float("inf")
    )
    delta_p50_pass = delta_p50 <= GATE_DELTA_P50_MS
    delta_p95_pass = delta_p95 <= GATE_DELTA_P95_MS
    tail_pass = tail_ratio <= GATE_MAX_OVER_MEDIAN_RATIO
    return GateResult(
        delta_p50_ms=delta_p50,
        delta_p95_ms=delta_p95,
        delta_p50_pass=delta_p50_pass,
        delta_p95_pass=delta_p95_pass,
        tail_ratio=tail_ratio,
        tail_ratio_pass=tail_pass,
        passed=delta_p50_pass and delta_p95_pass and tail_pass,
    )


# --- Report -----------------------------------------------------------------


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        )
        return out.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def _short_commit(full: str) -> str:
    return full[:12] if full and full != "unknown" else "unknown"


def _emit_report(
    *,
    corpus: CorpusSpec,
    query_count: int,
    iterations_per_query: int,
    bfs_off: ArmResult,
    bfs_on: ArmResult,
    gate: GateResult,
) -> dict[str, object]:
    return {
        "harness": HARNESS_NAME,
        "harness_version": HARNESS_VERSION,
        "aelfrice_version": AELFRICE_VERSION,
        "git_commit": _git_commit(),
        "captured_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ).replace("+00:00", "Z"),
        "corpus": asdict(corpus),
        "query_count": query_count,
        "iterations_per_query": iterations_per_query,
        "arms": {
            "bfs_off": bfs_off.to_dict(),
            "bfs_on": bfs_on.to_dict(),
        },
        "delta_ms": {
            "p50": bfs_on.p50_ms - bfs_off.p50_ms,
            "p95": bfs_on.p95_ms - bfs_off.p95_ms,
            "p99": bfs_on.p99_ms - bfs_off.p99_ms,
            "max": bfs_on.max_ms - bfs_off.max_ms,
        },
        "gate": gate.to_dict(),
        "gate_thresholds": {
            "delta_p50_ms": GATE_DELTA_P50_MS,
            "delta_p95_ms": GATE_DELTA_P95_MS,
            "max_over_median_ratio": GATE_MAX_OVER_MEDIAN_RATIO,
        },
    }


# --- CLI --------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--beliefs", type=int, default=DEFAULT_BELIEF_COUNT)
    ap.add_argument("--topics", type=int, default=DEFAULT_TOPICS)
    ap.add_argument("--queries", type=int, default=DEFAULT_QUERIES)
    ap.add_argument(
        "--iterations", type=int, default=DEFAULT_ITERATIONS_PER_QUERY,
        help="Timing iterations per query (post-warmup).",
    )
    ap.add_argument(
        "--warmup", type=int, default=DEFAULT_WARMUP_ITERATIONS,
        help="Discardable warmup iterations per query.",
    )
    ap.add_argument(
        "--output", type=Path, default=None,
        help=(
            "Output JSON path. Default: benchmarks/results/"
            "bfs_latency_v3/<short-commit>.json"
        ),
    )
    args = ap.parse_args(argv)

    output: Path
    if args.output is None:
        output = (
            Path("benchmarks/results/bfs_latency_v3")
            / f"{_short_commit(_git_commit())}.json"
        )
    else:
        output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="aelf_bfs_latency_v3_") as td:
        db_path = Path(td) / "bench.sqlite"
        store = MemoryStore(str(db_path))

        print(
            f"[bfs_latency_v3] seeding corpus "
            f"({args.beliefs} beliefs / {args.topics} topics)...",
            file=sys.stderr,
        )
        seed_t0 = time.perf_counter()
        corpus = seed_corpus(
            store,
            belief_count=args.beliefs,
            topics=args.topics,
        )
        seed_elapsed = time.perf_counter() - seed_t0
        print(
            f"[bfs_latency_v3] seeded {corpus.beliefs} beliefs / "
            f"{corpus.edges} edges in {seed_elapsed:.1f}s",
            file=sys.stderr,
        )

        queries = synth_queries(
            count=args.queries,
            topics=args.topics,
            belief_count=args.beliefs,
        )

        # bfs_off first so any HRR / BM25F caches built during this
        # arm carry into bfs_on — fair comparison: the user
        # experience after a few warm queries.
        print("[bfs_latency_v3] timing bfs_off arm...", file=sys.stderr)
        bfs_off = time_arm(
            store, queries,
            label="bfs_off",
            bfs_enabled=False,
            iterations=args.iterations,
            warmup=args.warmup,
        )
        print(
            f"[bfs_latency_v3]   bfs_off: p50={bfs_off.p50_ms:.1f}ms "
            f"p95={bfs_off.p95_ms:.1f}ms p99={bfs_off.p99_ms:.1f}ms",
            file=sys.stderr,
        )

        print("[bfs_latency_v3] timing bfs_on arm...", file=sys.stderr)
        bfs_on = time_arm(
            store, queries,
            label="bfs_on",
            bfs_enabled=True,
            iterations=args.iterations,
            warmup=args.warmup,
        )
        print(
            f"[bfs_latency_v3]   bfs_on: p50={bfs_on.p50_ms:.1f}ms "
            f"p95={bfs_on.p95_ms:.1f}ms p99={bfs_on.p99_ms:.1f}ms",
            file=sys.stderr,
        )

        gate = evaluate_gate(bfs_off, bfs_on)
        report = _emit_report(
            corpus=corpus,
            query_count=len(queries),
            iterations_per_query=args.iterations,
            bfs_off=bfs_off,
            bfs_on=bfs_on,
            gate=gate,
        )
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"[bfs_latency_v3] wrote {output} (gate "
            f"{'PASS' if gate.passed else 'FAIL'})",
            file=sys.stderr,
        )
        store.close()
    return 0 if gate.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
