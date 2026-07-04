"""G3 latency bench — temporal-spine default-flip gate (#1064, #739-style).

Measures retrieval p50/p95/p99 latency with the temporal-spine lane off
vs on, on a ≥10k-belief synthetic store carrying a real per-session
``TEMPORAL_NEXT`` spine (built by ``backfill_temporal_spine``). Only
``use_temporal_spine`` differs between the two arms; every other lane is
left at its ``retrieve_v2`` resolver default.

Gate (per #1064 G3, delta-reframed like #739 / PR #754):

    delta_p50  ≤  5 ms  (spine_on − spine_off)
    delta_p95  ≤ 50 ms  (spine_on − spine_off)
    max / median ratio ≤ 10x with the lane on (tail sanity)

Generic BFS measured +1.0 ms p50 / +35.6 ms p95 under the same delta
gate (``bfs_latency_v3.py``); the spine lane is narrower (depth-1
traversal from the top-5 L1 seeds, node budget 32) so it should sit
well inside the band.

Not a null
----------

A latency delta of ~0 is only meaningful if the lane actually did
work. A store whose queries never reach a chained belief would make
the on-arm a no-op and the gate pass vacuously (the ``#981``
trivial-null trap). This bench refuses to report unless the lane fires
on at least one query: ``seed_spine_corpus`` chains every session and
``synth_spine_queries`` targets belief bodies that sit mid-chain, and
``main`` raises if the pre-timing probe records zero spine candidates.

Corpus shape
------------

``sessions`` chains × ``beliefs // sessions`` beliefs each. Every
belief carries a ``session_id`` and a strictly increasing
``created_at`` within its session, so ``backfill_temporal_spine``
links each consecutive pair with one ``TEMPORAL_NEXT`` edge —
``beliefs − sessions`` edges total (the first belief in each chain has
no predecessor). Bodies embed two ``sess_NNNN_entity_K`` anchors for
the L2.5 path plus a per-belief condition tail so BM25 differentiates
the 50 beliefs in a chain.

The corpus is not a real-world distribution; it exists to exercise the
v3.0 default retrieval stack against a store whose size and chain
fan-out are in the maintainer's production order of magnitude (~14k
beliefs). A labelled-corpus version would be strictly better; this one
ships without a private-data dependency — same posture as
``bfs_latency_v3.py``.

Usage
-----

    uv run python benchmarks/temporal_spine_latency.py
    uv run python benchmarks/temporal_spine_latency.py --beliefs 10000 \\
        --output benchmarks/results/temporal_spine_latency/<run-id>.json

Refs
----

- Issue #1064 §"Pre-registered default-ON flip gate" G3.
- ``benchmarks/bfs_latency_v3.py`` — the #739 gate this mirrors.
- ``src/aelfrice/temporal_spine.py::backfill_temporal_spine`` — spine builder.
- ``src/aelfrice/retrieval.py::is_temporal_spine_enabled`` — the resolver
  whose default the flip gate governs.
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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final, Sequence

from aelfrice import __version__ as AELFRICE_VERSION
from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    ORIGIN_UNKNOWN,
    Belief,
)
from aelfrice.retrieval import last_lane_telemetry, retrieve_v2
from aelfrice.store import MemoryStore
from aelfrice.temporal_spine import backfill_temporal_spine

HARNESS_NAME: Final[str] = "temporal_spine_latency"
HARNESS_VERSION: Final[str] = "1"

DEFAULT_BELIEF_COUNT: Final[int] = 10_000
DEFAULT_SESSIONS: Final[int] = 200
DEFAULT_ITERATIONS_PER_QUERY: Final[int] = 10
DEFAULT_QUERIES: Final[int] = 30
DEFAULT_WARMUP_ITERATIONS: Final[int] = 3

# Production operating point (the #1064 G2 "does it survive the trim"
# budget). Latency is measured here, not at the wide dev budget, because
# the flip ships the production hook config.
DEFAULT_BUDGET: Final[int] = 1500
DEFAULT_L1_LIMIT: Final[int] = 50

# Gate thresholds per #1064 G3 (same delta band as #739 / PR #754).
GATE_DELTA_P50_MS: Final[float] = 5.0
GATE_DELTA_P95_MS: Final[float] = 50.0
GATE_MAX_OVER_MEDIAN_RATIO: Final[float] = 10.0

# Corpus epoch. Fixed so the spine chain order (created_at, rowid) and
# every derived id/body are byte-identical across runs.
_CORPUS_EPOCH: Final[datetime] = datetime(
    2026, 5, 13, 0, 0, 0, tzinfo=timezone.utc
)


# --- Corpus generation ------------------------------------------------------


def _session_id(sess: int) -> str:
    return f"sess_{sess:04d}"


def _belief_id(sess: int, idx: int) -> str:
    return f"bench_s{sess:04d}_b{idx:03d}"


def _belief_content(sess: int, idx: int) -> str:
    """Deterministic body with two ``sess_NNNN_entity_K`` anchors + tail.

    Shape: ``sess_NNNN_entity_A relates_to sess_NNNN_entity_B under
    condition_MM; observation rank_MM stable across replays.`` The
    entities anchor the L2.5 path; the condition / rank tail keeps BM25
    from collapsing all beliefs in a chain to identical scores so a
    query resolves to one mid-chain seed.
    """
    a = idx % 3
    b = (idx + 1) % 3
    return (
        f"{_session_id(sess)}_entity_{a} relates_to "
        f"{_session_id(sess)}_entity_{b} under condition_{idx:02d}; "
        f"observation rank_{idx} stable across replays."
    )


def _created_at(sess: int, idx: int) -> str:
    """Strictly increasing within a session; disjoint across sessions.

    Session ``sess`` occupies day ``sess``; belief ``idx`` within it is
    ``idx`` minutes past midnight. The chain builder orders by
    ``(session_id, created_at, rowid)``, so this fixes the chain
    direction deterministically.
    """
    ts = _CORPUS_EPOCH + timedelta(days=sess, minutes=idx)
    return ts.isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class CorpusSpec:
    beliefs: int
    spine_edges: int
    sessions: int
    per_session: int


def seed_spine_corpus(
    store: MemoryStore,
    *,
    belief_count: int = DEFAULT_BELIEF_COUNT,
    sessions: int = DEFAULT_SESSIONS,
) -> CorpusSpec:
    """Insert a deterministic session-chained corpus, then build the spine.

    Beliefs are unlocked, factual, Jeffreys (alpha=beta=1), tagged with
    a ``session_id`` and a per-session strictly-increasing
    ``created_at``. After all inserts, ``backfill_temporal_spine`` links
    each consecutive same-session pair with one ``TEMPORAL_NEXT`` edge.

    Returns the realised belief and spine-edge counts so the caller can
    record them and assert the expected ``beliefs − sessions`` edges.
    """
    if belief_count <= 0:
        raise ValueError(f"belief_count must be positive: {belief_count!r}")
    if sessions <= 0 or belief_count % sessions != 0:
        raise ValueError(
            "belief_count must be a positive multiple of sessions; "
            f"got belief_count={belief_count}, sessions={sessions}"
        )
    per_session = belief_count // sessions

    inserted = 0
    for s in range(sessions):
        sid = _session_id(s)
        for i in range(per_session):
            content = _belief_content(s, i)
            store.insert_belief(
                Belief(
                    id=_belief_id(s, i),
                    content=content,
                    content_hash=hashlib.sha256(content.encode()).hexdigest(),
                    alpha=1.0,
                    beta=1.0,
                    type=BELIEF_FACTUAL,
                    lock_level=LOCK_NONE,
                    locked_at=None,
                    created_at=_created_at(s, i),
                    last_retrieved_at=None,
                    origin=ORIGIN_UNKNOWN,
                    session_id=sid,
                )
            )
            inserted += 1

    report = backfill_temporal_spine(store)
    return CorpusSpec(
        beliefs=inserted,
        spine_edges=report.n_edges_written,
        sessions=sessions,
        per_session=per_session,
    )


def synth_spine_queries(
    *,
    count: int = DEFAULT_QUERIES,
    sessions: int = DEFAULT_SESSIONS,
    per_session: int,
) -> tuple[str, ...]:
    """Precise queries that resolve to a mid-chain belief per session.

    Each query is shaped ``sess_NNNN_entity_K relates_to condition_MM``
    — enough to make L1 rank one belief first, whose chained neighbours
    the spine lane then traverses. Targets are spread across sessions
    (stride ``sessions // count``) and biased to a mid-chain index so
    the lane has predecessors *and* successors to reach.
    """
    if count <= 0:
        raise ValueError(f"count must be positive: {count!r}")
    if sessions <= 0 or count > sessions:
        raise ValueError(f"count={count} must fit within sessions={sessions}")
    if per_session <= 0:
        raise ValueError(f"per_session must be positive: {per_session!r}")
    stride = max(1, sessions // count)
    mid = per_session // 2
    out: list[str] = []
    for i in range(count):
        s = (i * stride) % sessions
        idx = (mid + i) % per_session
        out.append(
            f"{_session_id(s)}_entity_{idx % 3} relates_to "
            f"condition_{idx:02d}"
        )
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
    """Nearest-rank percentile (0 ≤ p ≤ 100). Sequence must be sorted."""
    if not sorted_samples:
        return 0.0
    if p <= 0:
        return sorted_samples[0]
    if p >= 100:
        return sorted_samples[-1]
    n = len(sorted_samples)
    rank = max(1, min(n, int((p / 100.0) * n + 0.999999)))
    return sorted_samples[rank - 1]


def _summarise(label: str, samples_ms: Sequence[float]) -> ArmResult:
    ordered = sorted(samples_ms)
    return ArmResult(
        label=label,
        samples=len(ordered),
        p50_ms=_percentile(ordered, 50),
        p95_ms=_percentile(ordered, 95),
        p99_ms=_percentile(ordered, 99),
        max_ms=ordered[-1] if ordered else 0.0,
        mean_ms=statistics.fmean(samples_ms) if samples_ms else 0.0,
        min_ms=ordered[0] if ordered else 0.0,
    )


def time_arm(
    store: MemoryStore,
    queries: Sequence[str],
    *,
    label: str,
    use_spine: bool,
    budget: int,
    l1_limit: int,
    iterations: int,
    warmup: int = DEFAULT_WARMUP_ITERATIONS,
) -> ArmResult:
    """Time ``retrieve_v2`` per query × iterations; only the lane toggles.

    ``warmup`` discardable iterations per query absorb BM25 / cache
    warm-up so the percentile distribution reflects steady state, not
    first-touch cost.
    """
    if iterations <= 0:
        raise ValueError(f"iterations must be positive: {iterations!r}")
    if warmup < 0:
        raise ValueError(f"warmup must be non-negative: {warmup!r}")

    for q in queries:
        for _ in range(warmup):
            retrieve_v2(
                store, q, budget=budget, l1_limit=l1_limit,
                include_locked=False, use_temporal_spine=use_spine,
            )

    samples_ms: list[float] = []
    for q in queries:
        for _ in range(iterations):
            t0 = time.perf_counter()
            retrieve_v2(
                store, q, budget=budget, l1_limit=l1_limit,
                include_locked=False, use_temporal_spine=use_spine,
            )
            samples_ms.append((time.perf_counter() - t0) * 1000.0)
    return _summarise(label, samples_ms)


@dataclass(frozen=True)
class LaneFireProbe:
    queries_fired: int
    queries_total: int
    candidates_total: int
    survivors_total: int

    @property
    def fired(self) -> bool:
        return self.candidates_total > 0

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "fired": self.fired}


def probe_lane_fires(
    store: MemoryStore,
    queries: Sequence[str],
    *,
    budget: int,
    l1_limit: int,
) -> LaneFireProbe:
    """Run each query once lane-on and tally spine telemetry.

    Guards against a vacuous pass: if the lane never produces a
    candidate the latency delta is measuring nothing. Reads
    ``last_lane_telemetry()`` after each call for the candidate
    (pre-trim) and survivor (packed) spine counts.
    """
    fired = 0
    cand_total = 0
    surv_total = 0
    for q in queries:
        retrieve_v2(
            store, q, budget=budget, l1_limit=l1_limit,
            include_locked=False, use_temporal_spine=True,
        )
        tel = last_lane_telemetry()
        if tel.temporal_spine_candidates > 0:
            fired += 1
        cand_total += tel.temporal_spine_candidates
        surv_total += tel.temporal_spine
    return LaneFireProbe(
        queries_fired=fired,
        queries_total=len(queries),
        candidates_total=cand_total,
        survivors_total=surv_total,
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


def evaluate_gate(spine_off: ArmResult, spine_on: ArmResult) -> GateResult:
    """Delta gate: what turning the lane on costs on the same corpus."""
    delta_p50 = spine_on.p50_ms - spine_off.p50_ms
    delta_p95 = spine_on.p95_ms - spine_off.p95_ms
    tail_ratio = (
        spine_on.max_ms / spine_on.p50_ms
        if spine_on.p50_ms > 0 else float("inf")
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
    budget: int,
    l1_limit: int,
    query_count: int,
    iterations_per_query: int,
    probe: LaneFireProbe,
    spine_off: ArmResult,
    spine_on: ArmResult,
    gate: GateResult,
    captured_at_utc: str,
) -> dict[str, object]:
    return {
        "harness": HARNESS_NAME,
        "harness_version": HARNESS_VERSION,
        "aelfrice_version": AELFRICE_VERSION,
        "git_commit": _git_commit(),
        "captured_at_utc": captured_at_utc,
        "corpus": asdict(corpus),
        "budget": budget,
        "l1_limit": l1_limit,
        "query_count": query_count,
        "iterations_per_query": iterations_per_query,
        "lane_fire_probe": probe.to_dict(),
        "arms": {
            "spine_off": spine_off.to_dict(),
            "spine_on": spine_on.to_dict(),
        },
        "delta_ms": {
            "p50": spine_on.p50_ms - spine_off.p50_ms,
            "p95": spine_on.p95_ms - spine_off.p95_ms,
            "p99": spine_on.p99_ms - spine_off.p99_ms,
            "max": spine_on.max_ms - spine_off.max_ms,
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
    ap.add_argument("--sessions", type=int, default=DEFAULT_SESSIONS)
    ap.add_argument("--queries", type=int, default=DEFAULT_QUERIES)
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    ap.add_argument("--l1-limit", type=int, default=DEFAULT_L1_LIMIT)
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
            "temporal_spine_latency/<short-commit>.json"
        ),
    )
    args = ap.parse_args(argv)

    output: Path = args.output or (
        Path("benchmarks/results/temporal_spine_latency")
        / f"{_short_commit(_git_commit())}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="aelf_spine_latency_") as td:
        db_path = Path(td) / "bench.sqlite"
        store = MemoryStore(str(db_path))

        print(
            f"[{HARNESS_NAME}] seeding corpus "
            f"({args.beliefs} beliefs / {args.sessions} sessions)...",
            file=sys.stderr,
        )
        seed_t0 = time.perf_counter()
        corpus = seed_spine_corpus(
            store, belief_count=args.beliefs, sessions=args.sessions,
        )
        print(
            f"[{HARNESS_NAME}] seeded {corpus.beliefs} beliefs / "
            f"{corpus.spine_edges} spine edges in "
            f"{time.perf_counter() - seed_t0:.1f}s",
            file=sys.stderr,
        )

        queries = synth_spine_queries(
            count=args.queries,
            sessions=args.sessions,
            per_session=corpus.per_session,
        )

        probe = probe_lane_fires(
            store, queries, budget=args.budget, l1_limit=args.l1_limit,
        )
        print(
            f"[{HARNESS_NAME}] lane-fire probe: "
            f"{probe.queries_fired}/{probe.queries_total} queries, "
            f"{probe.candidates_total} candidates / "
            f"{probe.survivors_total} survivors",
            file=sys.stderr,
        )
        if not probe.fired:
            print(
                f"[{HARNESS_NAME}] FATAL: spine lane produced zero "
                f"candidates across all queries — the latency delta would "
                f"be a vacuous null. Refusing to report.",
                file=sys.stderr,
            )
            store.close()
            return 2

        # spine_off first so BM25 caches built here carry into spine_on
        # — the delta then reflects steady-state lane cost, not warm-up.
        print(f"[{HARNESS_NAME}] timing spine_off arm...", file=sys.stderr)
        spine_off = time_arm(
            store, queries, label="spine_off", use_spine=False,
            budget=args.budget, l1_limit=args.l1_limit,
            iterations=args.iterations, warmup=args.warmup,
        )
        print(
            f"[{HARNESS_NAME}]   spine_off: p50={spine_off.p50_ms:.1f}ms "
            f"p95={spine_off.p95_ms:.1f}ms p99={spine_off.p99_ms:.1f}ms",
            file=sys.stderr,
        )

        print(f"[{HARNESS_NAME}] timing spine_on arm...", file=sys.stderr)
        spine_on = time_arm(
            store, queries, label="spine_on", use_spine=True,
            budget=args.budget, l1_limit=args.l1_limit,
            iterations=args.iterations, warmup=args.warmup,
        )
        print(
            f"[{HARNESS_NAME}]   spine_on: p50={spine_on.p50_ms:.1f}ms "
            f"p95={spine_on.p95_ms:.1f}ms p99={spine_on.p99_ms:.1f}ms",
            file=sys.stderr,
        )

        gate = evaluate_gate(spine_off, spine_on)
        captured = (
            datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )
        report = _emit_report(
            corpus=corpus, budget=args.budget, l1_limit=args.l1_limit,
            query_count=len(queries), iterations_per_query=args.iterations,
            probe=probe, spine_off=spine_off, spine_on=spine_on, gate=gate,
            captured_at_utc=captured,
        )
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"[{HARNESS_NAME}] Δp50={gate.delta_p50_ms:+.2f}ms "
            f"Δp95={gate.delta_p95_ms:+.2f}ms  wrote {output} "
            f"(gate {'PASS' if gate.passed else 'FAIL'})",
            file=sys.stderr,
        )
        store.close()
    return 0 if gate.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
