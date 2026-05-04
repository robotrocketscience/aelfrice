"""Bench gate for #384 — Track A `TESTS` edge type ship gate.

Per #382 Decision A2 (operator ratification 2026-05-04 at
https://github.com/robotrocketscience/aelfrice/issues/382#issuecomment-4372683018),
`TESTS` ships only when it demonstrates **≥+5pp BFS multi-hop hit@k
uplift** on the labeled fixture vs. the same fixture run with
`BFS_EDGE_WEIGHTS[TESTS]` zeroed (which causes the BFS expander to
skip TESTS edges per `bfs_multihop.py:155-160`).

Skips cleanly when `AELFRICE_CORPUS_ROOT` is unset (public CI), when the
`tests_edge/` module dir is missing, or when the corpus has fewer than
`MIN_ROWS` non-seed rows (the gate requires a row floor before uplift
measurement is statistically meaningful).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import load_corpus_module

UPLIFT_FLOOR = 0.05  # +5pp per #382 Decision A2 (universal Track A bar)
MIN_ROWS = 30  # public-tree floor; lab corpus is expected to exceed this


def _build_store(tmp_path: Path, row: dict, arm: str):
    """Materialize a row's beliefs + edges into a transient MemoryStore.

    `arm` namespaces the db path so the two arms (with/without TESTS)
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
            demotion_pressure=0,
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
def test_tests_edge_uplift(
    aelfrice_corpus_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        r for r in load_corpus_module(aelfrice_corpus_root, "tests_edge")
        if not r.get("seed", False)
    ]

    if len(rows) < MIN_ROWS:
        pytest.skip(
            f"tests_edge corpus has {len(rows)} non-seed rows; gate requires "
            f"≥{MIN_ROWS} for stable uplift measurement"
        )

    from aelfrice.bfs_multihop import BFS_EDGE_WEIGHTS
    from aelfrice.models import EDGE_TESTS

    total_targets = sum(len(r["expected_hit_ids"]) for r in rows)
    assert total_targets > 0, "corpus has zero expected_hit_ids; cannot grade"

    # Arm 1: full edge weights (TESTS at its production weight 0.55).
    with_hits = 0
    for row in rows:
        store = _build_store(tmp_path, row, arm="with")
        try:
            with_hits += _row_hits(row, store)
        finally:
            store.close()

    # Arm 2: zero out TESTS. The BFS expander treats edge_w == 0.0 as
    # "skip" without marking visited, so this isolates the edge's
    # contribution to reachability.
    monkeypatch.setitem(BFS_EDGE_WEIGHTS, EDGE_TESTS, 0.0)
    without_hits = 0
    for row in rows:
        store = _build_store(tmp_path, row, arm="without")
        try:
            without_hits += _row_hits(row, store)
        finally:
            store.close()

    with_rate = with_hits / total_targets
    without_rate = without_hits / total_targets
    uplift = with_rate - without_rate

    assert uplift >= UPLIFT_FLOOR, (
        f"TESTS uplift {uplift:+.3f} below +{UPLIFT_FLOOR:.2f} floor "
        f"(with={with_rate:.3f}, without={without_rate:.3f}, n_rows={len(rows)}, "
        f"n_targets={total_targets}). Per #382 Decision A2, edge ships only "
        f"on ≥+5pp uplift; below-floor closes #384 as wontfix."
    )
