"""Load-independent regression gates for the four wall-clock-only paths (#1455).

Five tests sit behind `--run-perf`, which no workflow passes, so the paths
below have had no gate on any pull request. The #1420 §2 disposition is that
this tier is **counting assertions, not wall-clock budgets**: every performance
regression this repository has actually caught was countable — 150 TOML probes
per `retrieve()`, the 275.9 -> 70.5 ms hook import (caught as an import *set*),
`count_active_beliefs` as a full scan — and counts do not flake under
shared-runner load.

That ruling is not an abstract preference. Measured on an unloaded developer
machine, faster than a GitHub-hosted runner, `heat_kernel_score` failed at
10.01 ms against its 10.0 ms budget on one of three in-suite runs and passed
10/10 in isolation. At a 0.1% margin, widening the budget relocates the coin
flip rather than removing it.

So each test here pins the *algebraic shape* that makes the wall-clock budget
achievable, not the wall clock. Each one is paired with the mutation that turns
it red, named in its own docstring, because a gate nobody has seen fail is a
gate nobody has tested.

The four paths, and the asymptotic regression each one guards:

===========================  ==========================================
path                         regression the count catches
===========================  ==========================================
`HRRStructIndex.build`       O(N^2) store access instead of O(N)
`HRRStructIndex.probe`       full sort of N scores instead of top-K
`BM25Index.score`            densifying the sparse term matrix
`heat_kernel_score`          materialising the (N, N) dense kernel
===========================  ==========================================
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from aelfrice import hrr_index as hrr_mod
from aelfrice.bm25 import BM25Index
from aelfrice.graph_spectral import heat_kernel_score
from aelfrice.hrr_index import HRRStructIndex
from aelfrice.models import BELIEF_FACTUAL, EDGE_CITES, LOCK_NONE, Belief, Edge
from aelfrice.store import MemoryStore


def _mk(bid: str, content: str | None = None) -> Belief:
    return Belief(
        id=bid,
        content=content if content is not None else bid,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-04-28T00:00:00Z",
        last_retrieved_at=None,
    )


# --- HRRStructIndex.build: O(N) store calls ------------------------------


def test_build_makes_one_store_call_per_belief_not_n_squared() -> None:
    """`build` must read each belief's edges exactly once.

    The wall-clock budget (5 s at N=10k) is achievable only because `build`
    walks the store linearly: one `list_belief_ids`, then one `edges_from`
    per belief. Any re-read inside the per-belief loop turns the build
    quadratic, which no 5-second budget would survive at scale but which a
    small fixture would hide.

    Mutation that turns this red: call `store.edges_from(bid)` twice in the
    loop body, or move `list_belief_ids()` inside it.
    """
    n = 64
    store = MemoryStore(":memory:")
    for i in range(n):
        store.insert_belief(_mk(f"b{i}"))
    for i in range(n):
        store.insert_edge(
            Edge(src=f"b{i}", dst=f"b{(i + 1) % n}", type=EDGE_CITES, weight=1.0),
        )

    calls = {"edges_from": 0, "list_belief_ids": 0}
    real_edges_from = store.edges_from
    real_list_ids = store.list_belief_ids

    def counting_edges_from(bid: str):  # type: ignore[no-untyped-def]
        calls["edges_from"] += 1
        return real_edges_from(bid)

    def counting_list_ids():  # type: ignore[no-untyped-def]
        calls["list_belief_ids"] += 1
        return real_list_ids()

    store.edges_from = counting_edges_from  # type: ignore[method-assign]
    store.list_belief_ids = counting_list_ids  # type: ignore[method-assign]

    HRRStructIndex(dim=64, seed=0).build(store)

    assert calls["list_belief_ids"] == 1, (
        f"build called list_belief_ids {calls['list_belief_ids']}x; it must "
        "enumerate the store once, not once per belief"
    )
    assert calls["edges_from"] == n, (
        f"build made {calls['edges_from']} edges_from calls for {n} beliefs. "
        "Exactly one per belief is the linear walk the 5s budget assumes; "
        "anything more is quadratic store access."
    )


# --- HRRStructIndex.probe: top-K, not a full sort ------------------------


def test_probe_partitions_for_top_k_instead_of_sorting_all_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`probe` must not sort all N scores to return K of them.

    `argsort` over N=50k is what the 30 ms budget cannot absorb; the
    implementation uses `argpartition` to isolate K candidates and sorts
    only those. This asserts the split directly: `argpartition` is called,
    and every `argsort` call sees at most K elements -- never the full N.

    Mutation that turns this red: replace the `argpartition` branch with
    `order = np.argsort(-scores)`, which is the natural simplification and
    is correct, only slower.
    """
    n, dim, top_k = 512, 64, 10
    rng = np.random.default_rng(0)
    idx = HRRStructIndex(dim=dim, seed=0)
    idx.belief_ids = [f"b{i}" for i in range(n)]
    idx._index = {bid: i for i, bid in enumerate(idx.belief_ids)}
    idx.struct = rng.standard_normal((n, dim)).astype(np.float64) / np.sqrt(dim)
    idx.id_vecs = {"b0": rng.standard_normal(dim).astype(np.float64)}
    idx.role_vecs = {"CONTRADICTS": rng.standard_normal(dim).astype(np.float64)}

    sorted_sizes: list[int] = []
    partition_calls = {"n": 0}
    real_argsort = np.argsort
    real_argpartition = np.argpartition

    def spy_argsort(a, *args, **kwargs):  # type: ignore[no-untyped-def]
        sorted_sizes.append(int(np.asarray(a).size))
        return real_argsort(a, *args, **kwargs)

    def spy_argpartition(a, *args, **kwargs):  # type: ignore[no-untyped-def]
        partition_calls["n"] += 1
        return real_argpartition(a, *args, **kwargs)

    monkeypatch.setattr(hrr_mod.np, "argsort", spy_argsort)
    monkeypatch.setattr(hrr_mod.np, "argpartition", spy_argpartition)

    out = idx.probe("CONTRADICTS", "b0", top_k=top_k)

    assert len(out) == top_k
    assert partition_calls["n"] == 1, (
        "probe did not use argpartition; a top-K probe that sorts every "
        "score is O(N log N) in the term the 30ms budget is spent on"
    )
    assert sorted_sizes, "expected probe to sort the partitioned candidates"
    assert max(sorted_sizes) <= top_k, (
        f"probe sorted an array of {max(sorted_sizes)} elements for a "
        f"top-{top_k} query (N={n}). Only the K partitioned candidates may "
        "be sorted."
    )


# --- BM25Index.score: stays sparse ---------------------------------------


def test_score_never_densifies_the_term_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`score` must stay O(nnz) and never realise the dense term matrix.

    The term matrix is (n_docs x n_terms). At the N=50k the wall-clock test
    used, densifying it is gigabytes -- but at any fixture size small enough
    to run in CI it merely gets slower, so nothing catches it. Refusing the
    conversion outright is the load-independent form of the same assertion.

    Mutation that turns this red: `sat = sp.csr_matrix(...)` -> operate on
    `tf_csr.toarray()`, or any `.todense()` introduced while refactoring the
    saturation transform.
    """
    store = MemoryStore(":memory:")
    for i in range(200):
        store.insert_belief(_mk(f"b{i:04d}", f"token{i % 25} content blob"))
    idx = BM25Index.build(store, anchor_weight=0)

    def refuse(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError(
            "BM25Index.score densified the sparse term matrix. The scoring "
            "path must operate on the CSR data array; a dense (n_docs x "
            "n_terms) intermediate is the regression this gate exists for.",
        )

    monkeypatch.setattr(sp.csr_matrix, "toarray", refuse, raising=False)
    monkeypatch.setattr(sp.csr_matrix, "todense", refuse, raising=False)

    hits = idx.score("token3", top_k=50)

    assert hits, "expected the query to match at least one document"


# --- heat_kernel_score: two matvecs, no dense kernel ---------------------


class _MatmulSpy(np.ndarray):
    """ndarray that records the shape of every matmul it takes part in.

    Subclassing is what makes this reliable: `eigvecs @ x` dispatches to
    `ndarray.__matmul__`, so patching `np.matmul` would not see it.

    `__array_finalize__` is what carries the recorder across `.T` and other
    views. Without it the transpose is a fresh `_MatmulSpy` with no `shapes`
    attribute, and the first matvec -- which is the transposed one -- raises
    instead of being recorded.
    """

    shapes: list[tuple[int, ...]]

    def __array_finalize__(self, obj) -> None:  # type: ignore[no-untyped-def]
        if obj is None:
            return
        self.shapes = getattr(obj, "shapes", [])

    def __matmul__(self, other):  # type: ignore[no-untyped-def]
        out = super().__matmul__(other)
        self.shapes.append(np.shape(out))
        return out


def test_heat_kernel_does_two_matvecs_and_never_forms_the_dense_kernel() -> None:
    """`heat_kernel_score` must apply exp(-tL) through the eigenbasis.

    Its own docstring states the invariant: two matvecs, "to avoid
    materializing the (N, N) dense kernel". At N=50k that kernel is 20 GB,
    so the wall-clock test could never have failed on it -- it would have
    raised MemoryError instead. This asserts the shape directly and at a
    size where the wrong implementation would merely be slow.

    Mutation that turns this red: `return (eigvecs * np.exp(-t * eigvals))
    @ eigvecs.T @ seeds`, which is algebraically identical and forms the
    (N, N) product.
    """
    n, k = 400, 20
    rng = np.random.default_rng(0)
    eigvals = np.sort(rng.uniform(0.0, 2.0, size=k))
    eigvecs = rng.standard_normal((n, k)).astype(np.float64)
    seeds = np.zeros(n, dtype=np.float64)
    seeds[:5] = 1.0 / 5.0

    spy = eigvecs.view(_MatmulSpy)
    spy.shapes = []

    out = heat_kernel_score(eigvals, spy, seeds, t=8.0)

    assert np.shape(out) == (n,)
    assert spy.shapes, "expected heat_kernel_score to use matrix products"
    for shape in spy.shapes:
        assert shape != (n, n), (
            f"heat_kernel_score formed an {shape} intermediate. The (N, N) "
            "kernel is what the eigenbasis formulation exists to avoid; at "
            "N=50k it is 20 GB."
        )
    assert len(spy.shapes) == 2, (
        f"expected exactly 2 matvecs, saw {len(spy.shapes)}: {spy.shapes}. "
        "More products means the projection is being recomputed."
    )
