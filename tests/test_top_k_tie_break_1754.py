"""Top-k selection keeps the lowest ids among candidates tied at the cutoff (#1754).

`HRRStructIndex.probe` and `seeds_from_bm25` used `np.argpartition`, which
keeps an arbitrary subset of the candidates tied at the k-th score. Ties are
common in the probe: beliefs with the same outgoing edges have identical
struct rows. Rows in both places follow `list_belief_ids()`, which is in
ascending id order, so "lowest index" is "lowest id".

A BLAS matrix-vector product can also score two identical rows one ulp apart,
depending on the row's position and the matrix shape, which splits a tie
before selection sees it. `top_k_rows` re-scores the rows near the cutoff
with one reduction loop for every row.
"""

from __future__ import annotations

import numpy as np
import pytest

import aelfrice.hrr_index as hrr_index_mod
from aelfrice.graph_spectral import seeds_from_bm25
from aelfrice.hrr import top_k_indices, top_k_rows
from aelfrice.hrr_expand import neighbor_rows
from aelfrice.hrr_index import HRRStructIndex
from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_SUPPORTS,
    LOCK_NONE,
    Belief,
    Edge,
)
from aelfrice.store import MemoryStore

# More tied citers than any k below, so every k leaves ties at the cutoff.
_N_TIED = 17


def _mk(bid: str) -> Belief:
    return Belief(
        id=bid,
        content=f"belief {bid}",
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-10-08T00:00:00Z",
        last_retrieved_at=None,
    )


def _citers() -> list[str]:
    return [f"c{i:03d}" for i in range(_N_TIED)]


@pytest.fixture()
def tied_index() -> HRRStructIndex:
    """`_N_TIED` beliefs, each with one SUPPORTS edge to `t`: identical rows."""
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("t"))
    for bid in _citers():
        s.insert_belief(_mk(bid))
        s.insert_edge(Edge(src=bid, dst="t", type=EDGE_SUPPORTS, weight=1.0))
    idx = HRRStructIndex(dim=512, seed=7)
    idx.build(s, seed=7)
    s.close()
    return idx


# --- top_k_indices --------------------------------------------------------


@pytest.mark.parametrize("k", range(1, 9))
def test_ties_at_the_cutoff_keep_the_lowest_indices(k: int) -> None:
    scores = np.array([0.5, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0])
    got = top_k_indices(scores, k).tolist()
    expected = [10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0][:k]
    assert got == expected


def test_order_is_score_desc_then_index_asc() -> None:
    scores = np.array([1.0, 2.0, 1.0, 2.0, 0.0])
    assert top_k_indices(scores, 4).tolist() == [1, 3, 0, 2]


@pytest.mark.parametrize(("k", "expected"), [(0, []), (-3, []), (9, [1, 0, 2])])
def test_k_is_clamped(k: int, expected: list[int]) -> None:
    assert top_k_indices(np.array([1.0, 2.0, 1.0]), k).tolist() == expected


def test_empty_scores() -> None:
    assert top_k_indices(np.empty(0), 3).tolist() == []


# --- top_k_rows: a BLAS-style one-ulp split is undone ----------------------


class _UlpSplitMatrix(np.ndarray):
    """`@` adds one ulp to every 8th row, as a positional BLAS kernel can.

    Any `@`, including one on rows taken out of this matrix, is perturbed,
    so only a re-score that avoids `@` sees exact arithmetic.
    """

    def __matmul__(self, other: np.ndarray) -> np.ndarray:  # type: ignore[override]
        out = np.asarray(self) @ other
        out[::8] = np.nextafter(out[::8], np.inf)
        return out

    def __getitem__(self, key: object) -> np.ndarray:  # type: ignore[override]
        return np.asarray(np.asarray(self)[key]).view(_UlpSplitMatrix)  # type: ignore[index]


class _RowCountingMatrix(np.ndarray):
    """Records how many rows each fancy-index takes out of the matrix."""

    taken: list[int]

    def __matmul__(self, other: np.ndarray) -> np.ndarray:  # type: ignore[override]
        return np.asarray(self) @ other

    def __getitem__(self, key: object) -> np.ndarray:  # type: ignore[override]
        if isinstance(key, np.ndarray):
            self.taken.append(int(key.size))
        return np.asarray(self)[key]  # type: ignore[index]


@pytest.mark.parametrize("k", [1, 3, 7, 12])
def test_top_k_rows_restores_a_tie_split_by_the_product(k: int) -> None:
    rng = np.random.default_rng(1754)
    row = rng.standard_normal(64)
    probe = rng.standard_normal(64)
    plain = np.tile(row, (20, 1))
    split = plain.view(_UlpSplitMatrix)
    fast = split @ probe
    assert len(np.unique(fast)) == 2, "the fake product must split the tie"

    idx, scores = top_k_rows(split, probe, k)
    assert idx.tolist() == list(range(k))
    assert len(np.unique(scores)) == 1


def test_top_k_rows_does_not_rescore_zero_rows() -> None:
    """A sparse store: most rows are zero, and fewer than k score above 0.

    The cutoff is then 0.0 and every zero row is within the margin. They
    must not be copied out and re-scored: at 20k rows that was a full
    copy of the matrix and a 7-10x slower probe.
    """
    rng = np.random.default_rng(5)
    m = np.zeros((2000, 32))
    m[:4] = rng.standard_normal((4, 32))
    probe = rng.standard_normal(32)
    counted = m.view(_RowCountingMatrix)
    counted.taken = []
    idx, scores = top_k_rows(counted, probe, 10)
    assert max(counted.taken, default=0) <= 4
    exact = np.einsum("ij,j->i", m, probe)
    expected = sorted(range(m.shape[0]), key=lambda i: (-exact[i], i))[:10]
    assert idx.tolist() == expected
    np.testing.assert_array_equal(scores, exact[expected])


class _ZeroSplitMatrix(np.ndarray):
    """`@` scores every 8th row exactly 0.0.

    A BLAS kernel can do this to a non-zero row that is orthogonal to the
    probe to within rounding, while scoring its identical twins a few ulps
    away from zero.
    """

    def __matmul__(self, other: np.ndarray) -> np.ndarray:  # type: ignore[override]
        out = np.asarray(self) @ other
        out[::8] = 0.0
        return out

    def __getitem__(self, key: object) -> np.ndarray:  # type: ignore[override]
        return np.asarray(self)[key]  # type: ignore[index]


@pytest.mark.parametrize("sign", [1.0, -1.0])
@pytest.mark.parametrize("k", [1, 3, 7, 12])
def test_top_k_rows_rescores_a_non_zero_row_that_the_product_scores_zero(
    k: int, sign: float,
) -> None:
    rng = np.random.default_rng(1754)
    probe = sign * rng.standard_normal(64)
    row = rng.standard_normal(64)
    row -= (row @ probe) / (probe @ probe) * probe
    plain = np.tile(row, (20, 1))
    exact = np.einsum("ij,j->i", plain, probe)
    assert exact[0] != 0.0, "the rows must score near zero, not exactly zero"

    idx, scores = top_k_rows(plain.view(_ZeroSplitMatrix), probe, k)
    assert idx.tolist() == list(range(k))
    np.testing.assert_array_equal(scores, exact[:k])


class _NegativeZeroMatrix(np.ndarray):
    """`@` scores an all-zero row -0.0, as BLAS can for a negative probe."""

    def __matmul__(self, other: np.ndarray) -> np.ndarray:  # type: ignore[override]
        out = np.asarray(self) @ other
        out[out == 0.0] = -0.0
        return out


def test_top_k_rows_scores_zero_rows_as_positive_zero() -> None:
    rng = np.random.default_rng(9)
    m = np.zeros((10, 4))
    m[[3, 7]] = rng.standard_normal((2, 4))
    _, scores = top_k_rows(m.view(_NegativeZeroMatrix), rng.standard_normal(4), 5)
    zeros = scores[scores == 0.0]
    assert zeros.size >= 3
    assert not np.signbit(zeros).any()


def test_top_k_rows_matches_top_k_indices_on_distinct_scores() -> None:
    rng = np.random.default_rng(3)
    m = rng.standard_normal((500, 32))
    p = rng.standard_normal(32)
    idx, scores = top_k_rows(m, p, 10)
    assert idx.tolist() == top_k_indices(m @ p, 10).tolist()
    np.testing.assert_allclose(scores, (m @ p)[idx], rtol=0, atol=1e-12)


def test_top_k_rows_k_zero() -> None:
    idx, scores = top_k_rows(np.ones((3, 4)), np.ones(4), 0)
    assert idx.size == 0 and scores.size == 0


# --- through the probe and the HRR lane ------------------------------------


@pytest.mark.parametrize("k", range(1, _N_TIED))
def test_probe_keeps_the_lowest_ids_among_tied_candidates(
    tied_index: HRRStructIndex, k: int,
) -> None:
    hits = tied_index.probe(EDGE_SUPPORTS, "t", top_k=k)
    assert [bid for bid, _ in hits] == _citers()[:k]
    assert len({score for _, score in hits}) == 1


@pytest.mark.parametrize("k", [1, 5, 9])
def test_probe_keeps_ties_that_the_matrix_product_splits(
    tied_index: HRRStructIndex, k: int,
) -> None:
    tied_index.struct = tied_index.struct.view(_UlpSplitMatrix)
    hits = tied_index.probe(EDGE_SUPPORTS, "t", top_k=k)
    assert [bid for bid, _ in hits] == _citers()[:k]
    assert len({score for _, score in hits}) == 1


def test_probe_passes_the_cached_zero_row_mask(
    tied_index: HRRStructIndex, monkeypatch: pytest.MonkeyPatch,
) -> None:
    masks: list[object] = []

    def spy(matrix, probe, k, zero_rows=None):  # type: ignore[no-untyped-def]
        masks.append(zero_rows)
        return top_k_rows(matrix, probe, k, zero_rows)

    monkeypatch.setattr(hrr_index_mod, "top_k_rows", spy)
    tied_index.probe(EDGE_SUPPORTS, "t", top_k=3)
    tied_index.probe(EDGE_SUPPORTS, "t", top_k=3)
    assert masks[0] is not None and masks[1] is masks[0]
    # Only the target `t`, sorted after every citer, has no outgoing edges.
    assert np.flatnonzero(masks[0]).tolist() == [_N_TIED]  # type: ignore[arg-type]


def test_build_records_the_zero_row_mask(tied_index: HRRStructIndex) -> None:
    assert tied_index._zero_rows_of is tied_index.struct
    assert tied_index._zero_rows is not None
    np.testing.assert_array_equal(
        tied_index._zero_rows, ~tied_index.struct.any(axis=1),
    )


def test_zero_row_mask_follows_a_replaced_struct(tied_index: HRRStructIndex) -> None:
    assert not tied_index._all_zero_rows().all()
    tied_index.struct = np.zeros_like(tied_index.struct)
    assert tied_index._all_zero_rows().all()


def test_neighbor_rows_selects_and_orders_the_lowest_ids(
    tied_index: HRRStructIndex,
) -> None:
    rows = neighbor_rows(tied_index, "t", kinds=(EDGE_SUPPORTS,), per_probe_k=4)
    assert [(r[0], r[1], r[2]) for r in rows] == [
        (bid, EDGE_SUPPORTS, "reverse") for bid in _citers()[:4]
    ]


# --- seeds_from_bm25 -------------------------------------------------------


@pytest.mark.parametrize("k", range(1, 12))
def test_bm25_seeds_keep_the_lowest_rows_among_tied_scores(k: int) -> None:
    scores = np.array([0.0, 2.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    positive = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11]
    seeds = seeds_from_bm25(scores, top_k=k)
    assert np.flatnonzero(seeds).tolist() == sorted(positive[:k])
    assert seeds.sum() == pytest.approx(1.0)
