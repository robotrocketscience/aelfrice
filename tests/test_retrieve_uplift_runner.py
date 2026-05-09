"""Unit tests for the #154 retrieve-uplift harness — corpus-free.

Covers the deterministic primitives so the bench-gate test under
`tests/bench_gate/test_retrieve_uplift.py` (which is corpus-gated and
skips on public CI) doesn't carry the only assertions about the
harness shape. NDCG@k arithmetic, FlagUplift derivation, and the
end-to-end `run_per_flag_uplift` against a one-row synthetic corpus
all run without `AELFRICE_CORPUS_ROOT`.
"""
from __future__ import annotations

from tests.retrieve_uplift_runner import (
    FLAG_KWARGS,
    FlagUplift,
    ndcg_at_k,
    run_per_flag_uplift,
)


def test_ndcg_perfect_ranking_is_one() -> None:
    """Hypothesis: when result_ids equals expected_top_k in order,
    NDCG@k is exactly 1.0. Falsifiable if the metric ever deviates
    from 1.0 on a perfect ordering."""
    expected = ["a", "b", "c", "d"]
    assert ndcg_at_k(expected, expected, k=4) == 1.0


def test_ndcg_empty_expected_is_zero() -> None:
    """Hypothesis: NDCG is 0 when ground truth is empty."""
    assert ndcg_at_k(["a", "b"], [], k=2) == 0.0


def test_ndcg_no_overlap_is_zero() -> None:
    """Hypothesis: result list disjoint from expected_top_k → NDCG=0."""
    assert ndcg_at_k(["x", "y", "z"], ["a", "b", "c"], k=3) == 0.0


def test_ndcg_partial_overlap_between_zero_and_one() -> None:
    """Hypothesis: a partial-overlap ordering yields 0 < NDCG < 1.
    Falsifiable if the metric pegs at the boundaries on partial hits."""
    score = ndcg_at_k(["a", "x", "b"], ["a", "b", "c"], k=3)
    assert 0.0 < score < 1.0


def test_ndcg_position_matters() -> None:
    """Hypothesis: putting the top-relevant belief at position 1
    scores higher than putting it at position 3. Falsifiable if
    position weighting is broken."""
    expected = ["a", "b", "c"]
    front = ndcg_at_k(["a", "x", "y"], expected, k=3)
    back = ndcg_at_k(["x", "y", "a"], expected, k=3)
    assert front > back


def test_flag_uplift_dataclass_uplift_property() -> None:
    """Sanity: uplift = on - off."""
    fu = FlagUplift(flag="x", n_rows=10, mean_ndcg_off=0.4, mean_ndcg_on=0.55)
    assert abs(fu.uplift - 0.15) < 1e-9


def test_clustering_uplift_empty_input() -> None:
    """Empty corpus: zero rows, zero uplift, no exceptions."""
    from tests.retrieve_uplift_runner import (
        ClusteringUplift,
        run_clustering_uplift,
    )
    r = run_clustering_uplift([])
    assert isinstance(r, ClusteringUplift)
    assert r.n_rows == 0
    assert r.cluster_coverage_uplift == 0.0
    assert r.recall_uplift == 0.0


def test_clustering_uplift_shape_contract() -> None:
    """Bench-gate test reads .cluster_coverage_uplift /
    .cluster_coverage_on / .cluster_coverage_off — verify the names
    haven't drifted."""
    from tests.retrieve_uplift_runner import ClusteringUplift
    r = ClusteringUplift(
        n_rows=2,
        mean_recall_off=0.5,
        mean_recall_on=0.75,
        cluster_coverage_off=0.5,
        cluster_coverage_on=1.0,
    )
    assert r.cluster_coverage_uplift == 0.5
    assert r.recall_uplift == 0.25


def test_clustering_uplift_runs_on_synthetic_row() -> None:
    """One row → driver returns valid bounded metrics. Doesn't assert
    a specific uplift sign — that's bench evidence the test can't
    reproduce deterministically across rerank tuning."""
    from tests.retrieve_uplift_runner import run_clustering_uplift
    row = {
        "id": "mf-test-001",
        "query": "deploy and prerequisites",
        "beliefs": [
            {"id": "d1", "content": "deploy: install via pip then run setup"},
            {"id": "d2", "content": "deploy: provisions a sqlite store"},
            {"id": "p1", "content": "prereqs: python 3.13 required"},
            {"id": "p2", "content": "prereqs: writable git directory"},
        ],
        "edges": [
            {"src": "d1", "dst": "d2", "type": "SUPPORTS", "weight": 0.8},
            {"src": "p1", "dst": "p2", "type": "SUPPORTS", "weight": 0.8},
        ],
        "expected_belief_ids": ["d1", "p1"],
        "expected_clusters": [["d1", "d2"], ["p1", "p2"]],
        "n_clusters_required": 2,
    }
    r = run_clustering_uplift([row])
    assert r.n_rows == 1
    for v in (
        r.mean_recall_off, r.mean_recall_on,
        r.cluster_coverage_off, r.cluster_coverage_on,
    ):
        assert 0.0 <= v <= 1.0


def test_clustering_uplift_k_falls_back_to_n_clusters_required() -> None:
    """When k is omitted, the driver uses row['n_clusters_required'].
    Verify by explicitly varying k=1 vs default-K=3 and confirming the
    metric arithmetic reflects the smaller K's narrower window."""
    from tests.retrieve_uplift_runner import run_clustering_uplift
    row = {
        "id": "mf-test-002",
        "query": "alpha beta gamma",
        "beliefs": [
            {"id": "a", "content": "alpha alpha alpha"},
            {"id": "b", "content": "beta beta beta"},
            {"id": "c", "content": "gamma gamma gamma"},
        ],
        "edges": [],
        "expected_belief_ids": ["a", "b", "c"],
        "expected_clusters": [["a"], ["b"], ["c"]],
        "n_clusters_required": 3,
    }
    r_default = run_clustering_uplift([row], budget=10_000)
    r_explicit_k1 = run_clustering_uplift([row], budget=10_000, k=1)
    # K=3 (default from n_clusters_required) admits all three singleton
    # clusters; K=1 admits only the top-ranked one → coverage = 1/3.
    assert r_default.cluster_coverage_off == 1.0
    assert r_default.cluster_coverage_on == 1.0
    assert abs(r_explicit_k1.cluster_coverage_off - (1.0 / 3.0)) < 1e-9
    assert abs(r_explicit_k1.cluster_coverage_on - (1.0 / 3.0)) < 1e-9


def test_doc_linker_uplift_empty_input() -> None:
    from tests.retrieve_uplift_runner import run_doc_linker_uplift

    r = run_doc_linker_uplift([])
    assert r.n_rows == 0
    assert r.mean_ndcg_off == 0.0
    assert r.mean_ndcg_on == 0.0
    assert r.uplift == 0.0


def test_doc_linker_uplift_runs_on_synthetic_row() -> None:
    """OFF/ON arms run end-to-end without raising; metrics are bounded
    and the ON arm did write the row's anchors (verified by re-reading
    via ``get_doc_anchors_batch`` after the driver completes is out of
    scope here — the contract under test is shape + non-negative
    metric, not that the projection influences ranking, which today it
    does not)."""
    from tests.retrieve_uplift_runner import run_doc_linker_uplift

    row = {
        "id": "doc-test-001",
        "query": "memory store",
        "k": 3,
        "beliefs": [
            {"id": "b1", "content": "the memory store persists beliefs"},
            {"id": "b2", "content": "the configuration file lives at /etc"},
            {"id": "b3", "content": "the memory store uses sqlite"},
        ],
        "edges": [],
        "expected_top_k": ["b1", "b3"],
        "anchors": [
            {"belief_id": "b1", "doc_uri": "file:docs/store.md#L1-L40"},
            {"belief_id": "b3", "doc_uri": "file:docs/store.md#L41-L80"},
        ],
    }
    r = run_doc_linker_uplift([row])
    assert r.n_rows == 1
    assert 0.0 <= r.mean_ndcg_off <= 1.0
    assert 0.0 <= r.mean_ndcg_on <= 1.0


def test_doc_linker_uplift_no_anchors_means_zero_uplift() -> None:
    """A row with empty ``anchors`` is the degenerate case: the OFF and
    ON arms see identical store state, so ``uplift`` must be exactly 0
    regardless of what the retriever returns. Falsifiable if the driver
    accidentally treats ``anchors-ON`` differently from ``anchors-OFF``
    when there are no anchors to write."""
    from tests.retrieve_uplift_runner import run_doc_linker_uplift

    row = {
        "id": "doc-test-002",
        "query": "alpha",
        "k": 2,
        "beliefs": [
            {"id": "a", "content": "alpha alpha alpha"},
            {"id": "b", "content": "beta beta beta"},
        ],
        "edges": [],
        "expected_top_k": ["a"],
        "anchors": [],
    }
    r = run_doc_linker_uplift([row])
    assert r.n_rows == 1
    assert r.mean_ndcg_off == r.mean_ndcg_on
    assert r.uplift == 0.0


def test_query_strategy_uplift_empty_input() -> None:
    from tests.retrieve_uplift_runner import run_query_strategy_uplift

    r = run_query_strategy_uplift([])
    assert r.n_rows == 0
    assert r.mean_ndcg_off == 0.0
    assert r.mean_ndcg_on == 0.0
    assert r.uplift == 0.0


def test_query_strategy_uplift_runs_on_synthetic_row() -> None:
    """OFF/ON arms run end-to-end without raising; metrics are bounded.

    The contract under test is shape + non-negative metric, not that
    the rewrite influences ranking on this hand-crafted row — the
    real-corpus uplift is the lab-side gate."""
    from tests.retrieve_uplift_runner import run_query_strategy_uplift

    row = {
        "id": "qs-test-001",
        "query": "MemoryStore persistence",
        "k": 3,
        "beliefs": [
            {"id": "b1", "content": "the memory store persists beliefs"},
            {"id": "b2", "content": "the configuration file lives at /etc"},
            {"id": "b3", "content": "the memory store uses sqlite"},
        ],
        "edges": [],
        "expected_top_k": ["b1", "b3"],
    }
    r = run_query_strategy_uplift([row])
    assert r.n_rows == 1
    assert 0.0 <= r.mean_ndcg_off <= 1.0
    assert 0.0 <= r.mean_ndcg_on <= 1.0


def test_query_strategy_uplift_lowercase_no_extremes_means_zero_uplift() -> None:
    """Degenerate row: query has no capitalised tokens (R1 entity expand
    is a no-op) and the per-store IDF distribution is uniform enough
    that R3 quantile clipping doesn't drop or duplicate any term. In
    that case ``transform_query`` returns the same string for both
    strategies, so the OFF and ON arms see identical retrieval state
    and ``uplift == 0`` by construction.

    Falsifiable if the driver accidentally seeds the store differently
    between arms or threads through a non-deterministic dependency.

    A two-belief synthetic store with identical token counts puts every
    query token at the IDF median, so the default 5%/95% quantile band
    keeps every term at its baseline qf."""
    from tests.retrieve_uplift_runner import run_query_strategy_uplift

    row = {
        "id": "qs-test-002",
        "query": "alpha",
        "k": 2,
        "beliefs": [
            {"id": "a", "content": "alpha"},
            {"id": "b", "content": "beta"},
        ],
        "edges": [],
        "expected_top_k": ["a"],
    }
    r = run_query_strategy_uplift([row])
    assert r.n_rows == 1
    assert r.mean_ndcg_off == r.mean_ndcg_on
    assert r.uplift == 0.0


def test_run_per_flag_uplift_covers_all_flags() -> None:
    """Hypothesis: the harness reports one row per registered flag.
    Falsifiable if a flag is silently dropped."""
    row = {
        "id": "ru-test-001",
        "query": "memory store",
        "k": 3,
        "beliefs": [
            {"id": "b1", "content": "the memory store persists beliefs"},
            {"id": "b2", "content": "the configuration file lives at /etc"},
            {"id": "b3", "content": "the memory store uses sqlite"},
        ],
        "edges": [],
        "expected_top_k": ["b1", "b3"],
    }
    results = run_per_flag_uplift([row])
    flags = {r.flag for r in results}
    assert flags == set(FLAG_KWARGS.keys())
    for r in results:
        assert r.n_rows == 1
        assert 0.0 <= r.mean_ndcg_off <= 1.0
        assert 0.0 <= r.mean_ndcg_on <= 1.0
