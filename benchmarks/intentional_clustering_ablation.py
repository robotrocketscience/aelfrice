"""#436 intentional clustering — offline multi-fact coverage ablation.

Builds a SYNTHETIC multi-cluster corpus and reports `cluster_coverage@k`
and `recall@k` with the clustering lane OFF vs ON, via the SAME
`run_clustering_uplift()` driver the #436 A2 bench gate uses
(`tests.retrieve_uplift_runner`). It is the public, on-HEAD analogue of the
lab-side `multi_fact/*.jsonl` gate (whose corpus is not shipped), added for
the #1107 Phase-4 production graduation of the lane so the flip carries a
reproducible signal rather than resting only on the v3.0 lab evidence.

Each row is a two-topic query where both topics match, but one topic
(cluster A) out-scores the other on BM25 (denser query-term repetition),
so an OFF pack fills the top-K from cluster A alone. Clustering biases the
top-K toward distinct graph-connected clusters, so it should surface a
cluster-B representative — lifting `cluster_coverage@k` without dropping
`recall@k`. No live-store content — deterministic and CI-safe.

Spec § A2 ship condition: `cluster_coverage_uplift > 0` and
`recall_uplift >= 0` (don't-degrade backstop).

Run: `python benchmarks/intentional_clustering_ablation.py`
"""
from __future__ import annotations

from aelfrice.models import EDGE_DERIVED_FROM
from tests.retrieve_uplift_runner import run_clustering_uplift

N_ROWS = 20
_CLUSTER_SIZE = 3
_BUDGET = 220  # admits both clusters' reps; tight enough to force the choice


def _row(i: int) -> dict:  # type: ignore[type-arg]
    """One two-cluster row. Cluster A ('deploy') out-scores cluster B
    ('prerequisites') on BM25 for the shared query, so an unclustered
    pack takes both top slots from A."""
    a = [f"a{i}_{j}" for j in range(_CLUSTER_SIZE)]
    b = [f"b{i}_{j}" for j in range(_CLUSTER_SIZE)]
    beliefs = (
        [
            {"id": a[j], "content": f"deploy rollout deploy step deploy {j}"}
            for j in range(_CLUSTER_SIZE)
        ]
        + [
            {"id": b[j], "content": f"prerequisites checklist item {j}"}
            for j in range(_CLUSTER_SIZE)
        ]
    )
    # DERIVED_FROM chains make each topic a distinct graph-connected cluster.
    edges = (
        [
            {"src": a[j + 1], "dst": a[j], "type": EDGE_DERIVED_FROM}
            for j in range(_CLUSTER_SIZE - 1)
        ]
        + [
            {"src": b[j + 1], "dst": b[j], "type": EDGE_DERIVED_FROM}
            for j in range(_CLUSTER_SIZE - 1)
        ]
    )
    return {
        "id": f"row{i}",
        "query": "deploy prerequisites",
        "n_clusters_required": 2,
        "expected_belief_ids": [a[0], b[0]],
        "expected_clusters": [a, b],
        "beliefs": beliefs,
        "edges": edges,
    }


def main() -> None:
    rows = [_row(i) for i in range(N_ROWS)]
    res = run_clustering_uplift(rows, budget=_BUDGET, k=2)
    print(f"synthetic corpus: {res.n_rows} two-cluster rows, budget={_BUDGET}, k=2")
    print(
        f"  cluster_coverage@k  OFF={res.cluster_coverage_off:.3f} "
        f"ON={res.cluster_coverage_on:.3f} "
        f"uplift={res.cluster_coverage_uplift:+.3f}"
    )
    print(
        f"  recall@k            OFF={res.mean_recall_off:.3f} "
        f"ON={res.mean_recall_on:.3f} "
        f"uplift={res.recall_uplift:+.3f}"
    )
    ship = res.cluster_coverage_uplift > 0 and res.recall_uplift >= 0
    print(f"  A2 ship condition (coverage uplift > 0, recall uplift >= 0): "
          f"{'PASS' if ship else 'FAIL'}")


if __name__ == "__main__":
    main()
