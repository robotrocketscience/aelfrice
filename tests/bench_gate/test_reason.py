"""Bench gate for #389 — `aelf reason` ship gate.

Per operator ratification on issue #389
(https://github.com/robotrocketscience/aelfrice/issues/389#issuecomment-4372792969,
gate decision-ask 4):

    `hit@k` over chain-surfaced beliefs vs `hit@k` over `aelf search`
    baseline, ≥+3pp.

The gate isolates the reasoning-chain contribution by comparing
two arms on the same row: BFS-expanded chain hits (full pipeline)
vs the baseline `search_only_top_k` provided in the row.

Skips cleanly when `AELFRICE_CORPUS_ROOT` is unset, when the
`reasoning/` module dir is empty, or when the corpus has fewer than
`MIN_ROWS` rows.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import load_corpus_module

UPLIFT_FLOOR = 0.03  # +3pp per #389 decision-ask 4
MIN_ROWS = 20


def _build_store(tmp_path: Path, row: dict, idx: int):
    from aelfrice.models import BELIEF_FACTUAL, Belief, Edge
    from aelfrice.store import MemoryStore

    db_path = tmp_path / f"reasoning-{row['id']}-{idx}.db"
    store = MemoryStore(str(db_path))
    for b in row["beliefs"]:
        store.insert_belief(Belief(
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
        ))
    for e in row["edges"]:
        store.insert_edge(
            Edge(src=e["src"], dst=e["dst"], type=e["type"], weight=float(e["weight"]))
        )
    return store


def _chain_hits(row: dict, store, k: int) -> int:
    """Run BM25 → BFS expansion on the row and count hits in top-k."""
    from aelfrice.bfs_multihop import expand_bfs

    seeds = store.search_beliefs(row["query"], limit=3)
    expansions = expand_bfs(seeds, store)
    surfaced_ids = [b.id for b in seeds] + [h.belief.id for h in expansions]
    top_k = set(surfaced_ids[:k])
    expected = set(row["expected_hit_ids"])
    return len(top_k & expected)


def _baseline_hits(row: dict, k: int) -> int:
    """Count how many of `expected_hit_ids` appear in the row's
    `baseline_search_only_top_k` (already ranked by the labeller)."""
    top_k = set(row["baseline_search_only_top_k"][:k])
    expected = set(row["expected_hit_ids"])
    return len(top_k & expected)


@pytest.mark.bench_gated
def test_reason_chain_uplift(
    aelfrice_corpus_root: Path,
    tmp_path: Path,
) -> None:
    rows = [
        r for r in load_corpus_module(aelfrice_corpus_root, "reasoning")
        if not r.get("seed", False)
    ]
    if len(rows) < MIN_ROWS:
        pytest.skip(
            f"reasoning corpus has {len(rows)} non-seed rows; gate requires "
            f"≥{MIN_ROWS} for stable uplift measurement"
        )

    total_targets = sum(len(r["expected_hit_ids"]) for r in rows)
    assert total_targets > 0, "corpus has zero expected_hit_ids; cannot grade"

    chain_total = 0
    baseline_total = 0
    for idx, row in enumerate(rows):
        k = int(row["k"])
        store = _build_store(tmp_path, row, idx)
        try:
            chain_total += _chain_hits(row, store, k)
        finally:
            store.close()
        baseline_total += _baseline_hits(row, k)

    chain_rate = chain_total / total_targets
    baseline_rate = baseline_total / total_targets
    uplift = chain_rate - baseline_rate

    assert uplift >= UPLIFT_FLOOR, (
        f"`aelf reason` chain uplift {uplift:+.3f} below +{UPLIFT_FLOOR:.2f} "
        f"floor (chain={chain_rate:.3f}, baseline={baseline_rate:.3f}, "
        f"n_rows={len(rows)}, n_targets={total_targets}). Per #389 "
        f"decision-ask 4, the surface ships only on ≥+3pp uplift."
    )
