"""Bench gate for #421 / #387 — POTENTIALLY_STALE rerank demotion.

Per #421 acceptance #3 and #387 acceptance #3: the edge-type-keyed
rerank consumer must demonstrate **≥1pp@k drop on stale-tagged
retrieval** when the rerank pass runs over a labeled corpus where
some retrievable beliefs have at least one ``POTENTIALLY_STALE``
incoming edge.

Metric: stale rate at k = ``count(stale_ids ∩ top_k) / total_stale``,
summed across rows. Compute pre-rerank vs post-rerank; assert the
drop is at least `STALE_DROP_FLOOR` (0.01 = 1pp).

Skips cleanly when ``AELFRICE_CORPUS_ROOT`` is unset (public CI),
when the ``bfs_potentially_stale/`` module dir is missing, or when
the corpus has fewer than ``MIN_ROWS`` non-seed rows (the gate
requires a row floor before rate-difference measurement is
statistically meaningful).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import load_corpus_module

STALE_DROP_FLOOR = 0.01  # +1pp per #421 / #387 acceptance #3
MIN_ROWS = 30  # public-tree floor; lab corpus is expected to exceed this


def _build_store(tmp_path: Path, row: dict, arm: str):
    from aelfrice.models import BELIEF_FACTUAL, Belief, Edge
    from aelfrice.store import MemoryStore

    db_path = tmp_path / f"{row['id']}-{arm}.db"
    store = MemoryStore(str(db_path))
    for b in row["beliefs"]:
        belief = Belief(
            id=b["id"],
            content=b["text"],
            content_hash=f"h_{b['id']}",
            alpha=1.0,
            beta=1.0,
            type=BELIEF_FACTUAL,
            lock_level="none",
            locked_at=None,
            demotion_pressure=0,
            created_at="2026-05-04T00:00:00Z",
            last_retrieved_at=None,
        )
        store.insert_belief(belief)
    for e in row["edges"]:
        store.insert_edge(
            Edge(
                src=e["src"],
                dst=e["dst"],
                type=e["type"],
                weight=float(e["weight"]),
            )
        )
    return store


def _row_top_k_ids(row: dict, store, *, rerank: bool) -> list[str]:
    """Run BFS expansion (POTENTIALLY_STALE is skip-during-BFS per
    `BFS_EDGE_WEIGHTS[POTENTIALLY_STALE] = 0.0`) and either return
    BFS top-k directly (`rerank=False`) or apply the rerank pass
    first and then take top-k (`rerank=True`)."""
    from aelfrice.bfs_multihop import expand_bfs
    from aelfrice.edge_rerank import apply_edge_type_rerank

    seeds = []
    for sid in row["seed_ids"]:
        b = store.get_belief(sid)
        assert b is not None, f"row {row['id']}: seed {sid} not in row beliefs"
        seeds.append(b)
    expansions = expand_bfs(seeds, store)
    if rerank:
        expansions = apply_edge_type_rerank(expansions, store)
    k = int(row["k"])
    return [hop.belief.id for hop in expansions[:k]]


@pytest.mark.bench_gated
def test_potentially_stale_rerank_drops_stale_in_top_k(
    aelfrice_corpus_root: Path,
    tmp_path: Path,
) -> None:
    rows = [
        r
        for r in load_corpus_module(
            aelfrice_corpus_root, "bfs_potentially_stale"
        )
        if not r.get("seed", False)
    ]

    if len(rows) < MIN_ROWS:
        pytest.skip(
            f"bfs_potentially_stale corpus has {len(rows)} non-seed rows; gate "
            f"requires ≥{MIN_ROWS} for stable rate-difference measurement"
        )

    total_stale = sum(len(set(r["stale_ids"])) for r in rows)
    assert total_stale > 0, (
        "corpus has zero stale_ids across all rows; cannot grade "
        "rerank-demotion impact"
    )

    pre_stale_in_top_k = 0
    post_stale_in_top_k = 0
    for row in rows:
        stale = set(row["stale_ids"])
        store = _build_store(tmp_path, row, arm="run")
        try:
            pre_top = set(_row_top_k_ids(row, store, rerank=False))
            post_top = set(_row_top_k_ids(row, store, rerank=True))
            pre_stale_in_top_k += len(pre_top & stale)
            post_stale_in_top_k += len(post_top & stale)
        finally:
            store.close()

    pre_rate = pre_stale_in_top_k / total_stale
    post_rate = post_stale_in_top_k / total_stale
    drop = pre_rate - post_rate

    assert drop >= STALE_DROP_FLOOR, (
        f"POTENTIALLY_STALE rerank drop {drop:+.3f} below "
        f"+{STALE_DROP_FLOOR:.2f} floor (pre={pre_rate:.3f}, "
        f"post={post_rate:.3f}, n_rows={len(rows)}, "
        f"n_stale={total_stale}). Per #421 / #387 acceptance #3, the "
        f"rerank consumer ships only on ≥+1pp drop in stale-tagged "
        f"retrieval; below-floor blocks #387 closure."
    )
