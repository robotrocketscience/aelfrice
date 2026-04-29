"""AC6 perf measurement at N=50k for #151 slice 2.

Builds a 50k-belief in-memory store with a sparse SUPPORTS edge graph,
constructs the eigenbasis cache, then times retrieve() heat-off vs heat-on.

Run: uv run python -m benchmarks.posterior_ranking.ac6_50k
"""
from __future__ import annotations

import random
import statistics
import tempfile
import time
from pathlib import Path

from aelfrice.graph_spectral import GraphEigenbasisCache
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief, Edge
from aelfrice.retrieval import retrieve
from aelfrice.store import MemoryStore

N_BELIEFS = 50_000
N_EDGES = 5_000  # ~10% density (one edge per ten beliefs)
N_QUERIES = 30   # warmup 5, measured 25
WARMUP = 5

WORDS = [
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi",
    "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega",
    "memory", "store", "belief", "vector", "graph", "kernel", "ranking",
    "posterior", "prior", "evidence", "feedback", "weight", "score",
]


def _make_belief(bid: str, content: str) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=0.5,
        beta=0.5,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        demotion_pressure=0,
        created_at="2026-01-01T00:00:00Z",
        last_retrieved_at=None,
    )


def _build_store(n: int, n_edges: int, seed: int = 0) -> MemoryStore:
    rng = random.Random(seed)
    store = MemoryStore(":memory:")
    beliefs = []
    for i in range(n):
        words = rng.sample(WORDS, k=8)
        beliefs.append(_make_belief(f"b{i}", " ".join(words)))
    store.insert_beliefs(beliefs)

    edges = []
    seen: set[tuple[str, str]] = set()
    while len(edges) < n_edges:
        a = rng.randrange(n)
        b = rng.randrange(n)
        if a == b:
            continue
        key = (f"b{a}", f"b{b}")
        if key in seen:
            continue
        seen.add(key)
        edges.append(Edge(src=key[0], dst=key[1], type="SUPPORTS", weight=1.0))
    store.insert_edges(edges)
    return store


def _time_retrieve(
    store: MemoryStore,
    queries: list[str],
    *,
    heat_kernel: bool,
    cache: GraphEigenbasisCache | None,
) -> list[float]:
    times: list[float] = []
    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        retrieve(
            store,
            q,
            l1_limit=10,
            entity_index_enabled=False,
            bfs_enabled=False,
            posterior_weight=None,
            heat_kernel_enabled=heat_kernel,
            eigenbasis_cache=cache,
        )
        t1 = time.perf_counter()
        if i >= WARMUP:
            times.append((t1 - t0) * 1000.0)
    return times


def main() -> None:
    print(f"Building store: {N_BELIEFS} beliefs, {N_EDGES} edges...")
    t0 = time.perf_counter()
    store = _build_store(N_BELIEFS, N_EDGES)
    print(f"  store built in {time.perf_counter() - t0:.2f}s")

    print("Building eigenbasis cache...")
    tmp = tempfile.TemporaryDirectory(prefix="aelf_ac6_")
    cache = GraphEigenbasisCache(store=store, path=Path(tmp.name) / "eb.npz")
    t0 = time.perf_counter()
    cache.build()
    print(f"  cache built in {time.perf_counter() - t0:.2f}s")

    rng = random.Random(42)
    queries = [
        " ".join(rng.sample(WORDS, k=3)) for _ in range(N_QUERIES)
    ]

    print(f"\nTiming retrieve() heat-off (N={N_QUERIES - WARMUP} after warmup={WARMUP})...")
    off = _time_retrieve(store, queries, heat_kernel=False, cache=None)
    print(f"  heat-off: median={statistics.median(off):.3f}ms "
          f"p90={sorted(off)[int(0.9*len(off))]:.3f}ms "
          f"max={max(off):.3f}ms")

    print(f"\nTiming retrieve() heat-on...")
    on = _time_retrieve(store, queries, heat_kernel=True, cache=cache)
    print(f"  heat-on: median={statistics.median(on):.3f}ms "
          f"p90={sorted(on)[int(0.9*len(on))]:.3f}ms "
          f"max={max(on):.3f}ms")

    overhead_med = statistics.median(on) - statistics.median(off)
    print(f"\nHeat-on overhead (median): {overhead_med:+.3f}ms")
    print(f"AC6 budget: heat-off ≤1ms, heat-on ≤10ms")
    off_pass = statistics.median(off) <= 1.0
    on_pass = statistics.median(on) <= 10.0
    print(f"  heat-off pass: {off_pass}")
    print(f"  heat-on  pass: {on_pass}")

    tmp.cleanup()
    store.close()


if __name__ == "__main__":
    main()
