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
