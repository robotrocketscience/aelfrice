"""#1658: measure what the RESOLVES edge moves outside free-text retrieval.

`aelf resolve` writes a `RESOLVES` edge beside each `SUPERSEDES` edge.
Free-text retrieval ignores the edge (see
`tests/test_resolves_from_contradiction_1658.py`), but three readers count
every edge type and so can move:

* `hrr` — the HRR structural index binds every type in `EDGE_TYPES`, so
  `<KIND>:<id>` marker queries see the edge as crosstalk on the winner's
  row. The script probes every `(kind, belief)` pair, leaving out the
  `RESOLVES` kind itself, with and without the edges.
* `wonder_seed` — `aelf wonder` with no argument and no `--seed` picks
  the non-locked belief with the most outgoing edges
  (`cli._wonder_pick_seed`). A contradiction winner gains one outgoing
  edge. The script counts the stores whose picked seed changes.
* `random_walk` — the RW strategy (`wonder.strategies.random_walk`) picks
  uniformly over all outgoing edges. The script counts the stores whose
  phantom set changes under the same RNG seed. The only shipped caller,
  the offline bake-off (`python -m aelfrice.wonder.runner`), builds its
  own corpus with no `CONTRADICTS` edges, so it never sees a `RESOLVES`
  edge; this row describes a store that has been through `aelf resolve`.

Each store is synthetic and in memory. Every belief gets a random origin
from the tie-breaker's precedence classes; edges get random types drawn
from `EDGE_TYPES` without `RESOLVES`. `auto_resolve_all_contradictions`
then settles every `CONTRADICTS` pair, which writes the `RESOLVES` edges.
The "without" side deletes exactly those edges and nothing else. No real
store is opened.

Usage:
    uv run python benchmarks/resolves_edge_effect_1658.py
    uv run python benchmarks/resolves_edge_effect_1658.py --stores 50
    uv run python benchmarks/resolves_edge_effect_1658.py --dry-run

The output is one JSON object on stdout. The exit code is 1 when a store
produced no `RESOLVES` edge, because every comparison would then be
vacuous, and 0 otherwise.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass

from aelfrice.cli import _wonder_pick_seed  # pyright: ignore[reportPrivateUsage]
from aelfrice.contradiction import auto_resolve_all_contradictions
from aelfrice.hrr_index import HRRStructIndex
from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_RESOLVES,
    EDGE_TYPES,
    LOCK_NONE,
    ORIGIN_AGENT_INFERRED,
    ORIGIN_USER_STATED,
    Belief,
    Edge,
)
from aelfrice.store import MemoryStore
from aelfrice.wonder.strategies import random_walk

HRR_SCORE_FLOOR = 0.5
HRR_TOP_K = 10
HRR_INDEX_SEED = 7
_ORIGINS = (ORIGIN_USER_STATED, ORIGIN_AGENT_INFERRED, "unknown")


@dataclass(frozen=True)
class Config:
    beliefs: int
    edges: int
    stores: int
    seed: int
    rw_walks: int


@dataclass
class HrrResult:
    probes: int
    top10_order_changed: int
    above_floor_list_changed: int
    above_floor_set_changed: int
    max_shared_hit_score_delta: float


def _build_store(cfg: Config, rng: random.Random) -> tuple[MemoryStore, int]:
    s = MemoryStore(":memory:")
    ids = [f"b{i:04d}" for i in range(cfg.beliefs)]
    for i, bid in enumerate(ids):
        s.insert_belief(Belief(
            id=bid, content=f"c {bid}", content_hash=f"h{bid}",
            alpha=1.0, beta=1.0, type=BELIEF_FACTUAL, lock_level=LOCK_NONE,
            locked_at=None,
            created_at=f"2026-04-{1 + i % 28:02d}T00:00:00Z",
            last_retrieved_at=None, origin=rng.choice(_ORIGINS),
        ))
    kinds = sorted(EDGE_TYPES - {EDGE_RESOLVES})
    seen: set[tuple[str, str, str]] = set()
    for _ in range(cfg.edges):
        a, b = rng.sample(ids, 2)
        k = rng.choice(kinds)
        if (a, b, k) in seen:
            continue
        seen.add((a, b, k))
        s.insert_edge(Edge(src=a, dst=b, type=k, weight=1.0))
    results = auto_resolve_all_contradictions(s)
    return s, sum(1 for r in results if r.resolves_created)


def _resolves_edges(s: MemoryStore) -> list[Edge]:
    return [e for e in s.iter_all_edges() if e.type == EDGE_RESOLVES]


def _probe_all(s: MemoryStore) -> dict[tuple[str, str], list[tuple[str, float]]]:
    idx = HRRStructIndex()
    idx.build(s, seed=HRR_INDEX_SEED)
    ids = s.list_belief_ids()
    return {
        (k, t): idx.probe(k, t, top_k=HRR_TOP_K)
        for k in sorted(EDGE_TYPES - {EDGE_RESOLVES})
        for t in ids
    }


def _hrr_compare(
    with_p: dict[tuple[str, str], list[tuple[str, float]]],
    without_p: dict[tuple[str, str], list[tuple[str, float]]],
) -> HrrResult:
    out = HrrResult(len(without_p), 0, 0, 0, 0.0)
    for key, p0 in without_p.items():
        p1 = with_p[key]
        if [b for b, _ in p0] != [b for b, _ in p1]:
            out.top10_order_changed += 1
        a0 = [b for b, sc in p0 if sc >= HRR_SCORE_FLOOR]
        a1 = [b for b, sc in p1 if sc >= HRR_SCORE_FLOOR]
        if a0 != a1:
            out.above_floor_list_changed += 1
        if set(a0) != set(a1):
            out.above_floor_set_changed += 1
        d0, d1 = dict(p0), dict(p1)
        for b in set(d0) & set(d1):
            out.max_shared_hit_score_delta = max(
                out.max_shared_hit_score_delta, abs(d0[b] - d1[b]),
            )
    out.max_shared_hit_score_delta = round(out.max_shared_hit_score_delta, 4)
    return out


def _seed_id(s: MemoryStore) -> str | None:
    b = _wonder_pick_seed(s)
    return getattr(b, "id", None)


def _rw(s: MemoryStore, cfg: Config, store_index: int) -> set[tuple[str, ...]]:
    rng = random.Random(cfg.seed * 1000 + store_index)
    return {p.composition for p in random_walk(s, rng=rng, n_walks=cfg.rw_walks)}


def run(cfg: Config) -> tuple[dict[str, object], bool]:
    rng = random.Random(cfg.seed)
    hrr: HrrResult | None = None
    seed_changed = 0
    seed_was_winner = 0
    rw_changed = 0
    jaccards: list[float] = []
    resolves_counts: list[int] = []
    resolves_share: list[float] = []
    vacuous = False
    for i in range(cfg.stores):
        s, n_res = _build_store(cfg, rng)
        edges = _resolves_edges(s)
        resolves_counts.append(len(edges))
        total = sum(1 for _ in s.iter_all_edges())
        resolves_share.append(len(edges) / total if total else 0.0)
        if n_res == 0 or not edges:
            vacuous = True
        with_seed = _seed_id(s)
        with_rw = _rw(s, cfg, i)
        with_probes = _probe_all(s) if i == 0 else None
        winners = {e.src for e in edges}
        for e in edges:
            s.delete_edge(e.src, e.dst, e.type)
        without_seed = _seed_id(s)
        without_rw = _rw(s, cfg, i)
        if with_probes is not None:
            hrr = _hrr_compare(with_probes, _probe_all(s))
        if with_seed != without_seed:
            seed_changed += 1
            if with_seed in winners:
                seed_was_winner += 1
        if with_rw != without_rw:
            rw_changed += 1
        union = with_rw | without_rw
        jaccards.append(len(with_rw & without_rw) / len(union) if union else 1.0)
        s.close()
    report: dict[str, object] = {
        "config": asdict(cfg),
        "resolves_edges_per_store": {
            "min": min(resolves_counts), "max": max(resolves_counts),
        },
        "resolves_share_of_edges_mean": round(
            sum(resolves_share) / len(resolves_share), 4,
        ),
        "hrr_first_store": asdict(hrr) if hrr is not None else None,
        "wonder_seed": {
            "stores": cfg.stores,
            "seed_changed": seed_changed,
            "new_seed_is_a_contradiction_winner": seed_was_winner,
        },
        "random_walk": {
            "stores": cfg.stores,
            "phantom_set_changed": rw_changed,
            "mean_jaccard_with_vs_without": round(
                sum(jaccards) / len(jaccards), 4,
            ),
        },
    }
    return report, vacuous


def _parse(argv: list[str] | None) -> tuple[Config, bool]:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--beliefs", type=int, default=300)
    p.add_argument("--edges", type=int, default=900)
    p.add_argument("--stores", type=int, default=20)
    p.add_argument("--seed", type=int, default=1658)
    p.add_argument("--rw-walks", type=int, default=50)
    p.add_argument(
        "--dry-run", action="store_true",
        help="print the configuration and exit without building a store",
    )
    a = p.parse_args(argv)
    if a.beliefs < 2 or a.edges < 1 or a.stores < 1 or a.rw_walks < 1:
        p.error("--beliefs >= 2, --edges >= 1, --stores >= 1, --rw-walks >= 1")
    cfg = Config(a.beliefs, a.edges, a.stores, a.seed, a.rw_walks)
    return cfg, a.dry_run


def main(argv: list[str] | None = None) -> int:
    cfg, dry_run = _parse(argv)
    if dry_run:
        print(json.dumps({"dry_run": True, "config": asdict(cfg)}, indent=2))
        return 0
    report, vacuous = run(cfg)
    print(json.dumps(report, indent=2))
    if vacuous:
        print("error: a store produced no RESOLVES edge", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
