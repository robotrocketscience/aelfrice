"""Bench gate for #385 — Track A `IMPLEMENTS` edge type ship gate.

Per #382 Decision A2 (operator ratification 2026-05-04 at
https://github.com/robotrocketscience/aelfrice/issues/382#issuecomment-4372683018),
`IMPLEMENTS` ships only when it demonstrates **≥+5pp BFS multi-hop hit@k
uplift** on the labeled fixture vs. the same fixture run with
`BFS_EDGE_WEIGHTS[IMPLEMENTS]` zeroed (which causes the BFS expander to
skip IMPLEMENTS edges per `bfs_multihop.py:212-213`).

Skips cleanly when `AELFRICE_CORPUS_ROOT` is unset (public CI), when the
`implements_edge/` module dir is missing, or when the corpus has fewer than
`MIN_ROWS` non-seed rows (the gate requires a row floor before uplift
measurement is statistically meaningful).
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from tests.bench_gate.null_model import (
    AblationArms,
    bar_at_least,
    guard_ablation_gate,
)
from tests.conftest import load_corpus_module, require_min_rows

UPLIFT_FLOOR = 0.05  # +5pp per #382 Decision A2 (universal Track A bar)
MIN_ROWS = 30  # public-tree floor; lab corpus is expected to exceed this


def _build_store(tmp_path: Path, row: dict, arm: str):
    """Materialize a row's beliefs + edges into a transient MemoryStore.

    `arm` namespaces the db path so the two arms (with/without IMPLEMENTS)
    don't collide on the same SQLite file.
    """
    from aelfrice.bfs_multihop import expand_bfs  # noqa: F401  (sanity import)
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
            created_at="2026-05-04T00:00:00Z",
            last_retrieved_at=None,
        )
        store.insert_belief(belief)
    for e in row["edges"]:
        store.insert_edge(
            Edge(src=e["src"], dst=e["dst"], type=e["type"], weight=float(e["weight"]))
        )
    return store


def _row_hits(row: dict, store) -> int:
    """Run BFS expansion on a row's seeds and count how many of the
    row's `expected_hit_ids` appear in the top-k expansions."""
    from aelfrice.bfs_multihop import expand_bfs

    seeds = []
    for sid in row["seed_ids"]:
        b = store.get_belief(sid)
        assert b is not None, f"row {row['id']}: seed {sid} not in row beliefs"
        seeds.append(b)
    expansions = expand_bfs(seeds, store)
    k = int(row["k"])
    top_ids = {hop.belief.id for hop in expansions[:k]}
    expected = set(row["expected_hit_ids"])
    return len(top_ids & expected)


@pytest.mark.bench_gated
def test_implements_edge_uplift(
    aelfrice_corpus_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    rows = [
        r for r in load_corpus_module(aelfrice_corpus_root, "implements_edge")
        if not r.get("seed", False)
    ]

    require_min_rows(
        rows,
        module="implements_edge",
        minimum=MIN_ROWS,
        root=aelfrice_corpus_root,
        detail="non-seed rows, for stable uplift measurement",
    )

    from aelfrice.bfs_multihop import BFS_EDGE_WEIGHTS
    from aelfrice.models import EDGE_IMPLEMENTS

    total_targets = sum(len(r["expected_hit_ids"]) for r in rows)
    assert total_targets > 0, "corpus has zero expected_hit_ids; cannot grade"

    def arms() -> AblationArms:
        # Arm 1: full edge weights (IMPLEMENTS at its production weight).
        with_hits = 0
        for row in rows:
            store = _build_store(tmp_path, row, arm="with")
            try:
                with_hits += _row_hits(row, store)
            finally:
                store.close()

        # Arm 2: zero out IMPLEMENTS. The BFS expander treats edge_w == 0.0 as
        # "skip" without marking visited, so this isolates the edge's
        # contribution to reachability.
        monkeypatch.setitem(BFS_EDGE_WEIGHTS, EDGE_IMPLEMENTS, 0.0)
        without_row_hits: list[float] = []
        for row in rows:
            store = _build_store(tmp_path, row, arm="without")
            try:
                without_row_hits.append(float(_row_hits(row, store)))
            finally:
                store.close()

        return AblationArms(
            shipped=with_hits / total_targets,
            ablated=sum(without_row_hits) / total_targets,
            without_row_scores=without_row_hits,
        )

    # #1581: the ablated arm is this gate's declared null model. A corpus
    # on which it reaches no target at all makes the uplift a tautology of
    # BFS_EDGE_WEIGHTS rather than evidence about the edge, so the guard
    # rejects it before the floor below is read as a verdict.
    measured = guard_ablation_gate(
        module="implements_edge",
        rows=rows,
        arms=arms,
        bar=bar_at_least(UPLIFT_FLOOR),
        record_property=record_property,
        gold_key="expected_hit_ids",
        pool_key="beliefs",
    )
    with_rate = measured.shipped
    without_rate = measured.ablated
    uplift = measured.uplift

    assert uplift >= UPLIFT_FLOOR, (
        f"IMPLEMENTS uplift {uplift:+.3f} below +{UPLIFT_FLOOR:.2f} floor "
        f"(with={with_rate:.3f}, without={without_rate:.3f}, n_rows={len(rows)}, "
        f"n_targets={total_targets}). Per #382 Decision A2, edge ships only "
        f"on ≥+5pp uplift; below-floor closes #385 as wontfix."
    )
