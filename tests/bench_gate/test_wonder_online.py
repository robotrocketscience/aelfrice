"""Bench gate for #389 — `aelf wonder` on-line surface ship gate.

Per operator ratification on issue #389
(https://github.com/robotrocketscience/aelfrice/issues/389#issuecomment-4372792969,
gate decision-ask 8):

    "demonstrable consolidation-candidate surfacing" =
    ≥1 expected candidate in top-10 across ≥60% of rows.

The gate runs the same combined-score pass `_cmd_wonder` runs
(BFS + wonder_consolidation.score) against a labeled per-row graph
and verifies that at least one candidate from `expected_candidate_ids`
appears in the top-10 surfaced for that row.

Skips cleanly when `AELFRICE_CORPUS_ROOT` is unset or `wonder_online/`
is empty.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import load_corpus_module

ROW_HIT_FRACTION_FLOOR = 0.60
TOP_K = 10
MIN_ROWS = 20


def _build_store(tmp_path: Path, row: dict, idx: int):
    from aelfrice.models import BELIEF_FACTUAL, Belief, Edge
    from aelfrice.store import MemoryStore

    db_path = tmp_path / f"wonder-{row['id']}-{idx}.db"
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


def _row_top_k_candidates(row: dict, store, k: int) -> set[str]:
    """Mirror `_cmd_wonder`'s combined scoring + ranking, return top-k ids."""
    from aelfrice import wonder_consolidation
    from aelfrice.bfs_multihop import expand_bfs

    seed_b = store.get_belief(row["seed_id"])
    assert seed_b is not None, f"row {row['id']}: seed_id missing from beliefs"
    hops = expand_bfs([seed_b], store, max_depth=2, total_budget=k * 2)
    scored: list[tuple[float, str]] = []
    for h in hops:
        relatedness = wonder_consolidation.score(seed_b, h.belief)
        combined = h.score * (0.5 + 0.5 * relatedness)
        scored.append((combined, h.belief.id))
    scored.sort(key=lambda r: (-r[0], r[1]))
    return {bid for _, bid in scored[:k]}


@pytest.mark.bench_gated
def test_wonder_online_recall(
    aelfrice_corpus_root: Path,
    tmp_path: Path,
) -> None:
    rows = [
        r for r in load_corpus_module(aelfrice_corpus_root, "wonder_online")
        if not r.get("seed", False)
    ]
    if len(rows) < MIN_ROWS:
        pytest.skip(
            f"wonder_online corpus has {len(rows)} non-seed rows; gate "
            f"requires ≥{MIN_ROWS} for stable recall measurement"
        )

    rows_with_hit = 0
    for idx, row in enumerate(rows):
        store = _build_store(tmp_path, row, idx)
        try:
            top_k = _row_top_k_candidates(row, store, TOP_K)
        finally:
            store.close()
        expected = set(row["expected_candidate_ids"])
        if top_k & expected:
            rows_with_hit += 1

    hit_fraction = rows_with_hit / len(rows)
    assert hit_fraction >= ROW_HIT_FRACTION_FLOOR, (
        f"`aelf wonder` row-recall {hit_fraction:.2%} below "
        f"{ROW_HIT_FRACTION_FLOOR:.0%} floor (rows_with_hit={rows_with_hit}, "
        f"n_rows={len(rows)}, top_k={TOP_K}). Per #389 decision-ask 8, the "
        f"surface ships only when ≥1 expected candidate appears in top-{TOP_K} "
        f"across ≥{ROW_HIT_FRACTION_FLOOR:.0%} of rows."
    )
