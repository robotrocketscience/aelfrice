# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeStubs=false, reportOptionalSubscript=false, reportAttributeAccessIssue=false, reportGeneralTypeIssues=false, reportConstantRedefinition=false, reportArgumentType=false
"""Signed normalized Laplacian + heat-kernel authority scoring (#149, #150).

Constructs the signed normalized Laplacian
``L = I - D_abs^(-1/2) W_sym D_abs^(-1/2)``
over the typed edge graph (Kunegis 2010), where ``CONTRADICTS`` /
``SUPERSEDES`` count as negative-weight edges and the supportive edge
families as positive. The top-K=200 eigenpairs of ``L`` are the offline
foundation consumed by the heat-kernel authority signal.

#150 adds the per-query heat kernel ``exp(-tL)`` evaluated through the
precomputed eigenbasis: ``U @ diag(exp(-t*Λ)) @ U.T @ seeds``. Two
matvecs against the ``(n, K)`` eigenbasis, no per-call eigendecomp.
``seeds_from_bm25`` builds the L1-normalized seed vector from BM25 top-K
hits; ``apply_broker_attenuation`` dampens scores by per-belief broker
confidence (``α/(α+β)``); ``combine_log_scores`` is the log-additive
ranking formula ``log(BM25F) + 1.0 * log(heat_safe) + 0.5 * log(post)``.

All construction is offline — no query-time cost. Rebuild is wired to
the same store-mutation invalidation callback that ``RetrievalCache``
subscribes to (extending the registry, not duplicating it).
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Set as AbstractSet
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from aelfrice.models import (
    EDGE_CITES,
    EDGE_CONTRADICTS,
    EDGE_DERIVED_FROM,
    EDGE_IMPLEMENTS,
    EDGE_RELATES_TO,
    EDGE_RESOLVES,
    EDGE_SUPERSEDES,
    EDGE_SUPPORTS,
    EDGE_TEMPORAL_NEXT,
    EDGE_TESTS,
    EDGE_TYPES,
)
from aelfrice.store import MemoryStore

# Edge-type → signed weight mapping for Laplacian construction.
# Distinct from `models.EDGE_TYPE_WEIGHTS` (used for valence
# propagation; different polarity convention). Per #149 spec; exposed
# as a constant so a follow-up sweep issue can override without
# touching call sites.
#
# The table must be **total** over `models.EDGE_TYPES` (#1375). It used
# to list only the six #149 types, and `_build_adjacency` reads it via
# `.get(e.type, 0.0)` — so the four types added after #149 dropped out
# of the Laplacian silently, with no line anywhere saying they had. The
# zeros below are now a stated decision rather than a fall-through, and
# the totality guard underneath fails the import if a new edge type is
# added to `models` without one.
EDGE_LAPLACIAN_WEIGHTS: Final[dict[str, float]] = {
    EDGE_SUPPORTS: 1.0,
    EDGE_CITES: 1.0,
    EDGE_RELATES_TO: 1.0,
    EDGE_DERIVED_FROM: 1.0,
    EDGE_CONTRADICTS: -1.0,
    EDGE_SUPERSEDES: -0.5,
    # --- Deliberate zeros. Each of these is excluded from the spectral
    # graph, not merely unassigned. Changing any of them off 0.0 alters
    # every eigenpair and therefore every heat-kernel authority score,
    # so it is a benchmarked change, not a typo fix.
    #
    # IMPLEMENTS / TESTS: provenance and evidential edges the #149 spec
    # predates. Both are plausible +1.0 members; neither has been run
    # through an authority-scoring gate, and shipping them at a guessed
    # weight would move the ranking on the strength of a guess.
    EDGE_IMPLEMENTS: 0.0,
    EDGE_TESTS: 0.0,
    # TEMPORAL_NEXT: ~88% of all edges on a live store. Chronological
    # adjacency carries no evidential or argumentative content, and at
    # that density a non-zero weight would make the top eigenvectors a
    # picture of ingest order rather than of the belief graph. This is
    # the one zero here that is a positive design choice.
    EDGE_TEMPORAL_NEXT: 0.0,
    # RESOLVES: wonder-lifecycle marker (#548), same rationale as its
    # 0.0 in `EDGE_VALENCE` — resolution intent is not evidence.
    EDGE_RESOLVES: 0.0,
}


def missing_laplacian_weights(
    weights: Mapping[str, float], edge_types: AbstractSet[str],
) -> frozenset[str]:
    """Edge types in `edge_types` with no entry in `weights` (#1375).

    A named function rather than an inline expression so the guard below
    has a distinguishing test: an invariant that only ever runs against
    the shipped table is checked by "it did not raise", which is the same
    observation you would get from a guard that cannot fire at all.
    """
    return frozenset(edge_types) - frozenset(weights)


# Import-time totality guard (#1375). An explicit raise rather than
# `assert` so it survives `python -O` — a silently-dropped edge type is
# precisely the failure mode this exists to prevent, and stripping the
# check under optimisation would restore it.
#
# `POTENTIALLY_STALE` is intentionally absent from both sides: it is a
# marker tag and not a member of `EDGE_TYPES`, so the containment is
# one-directional (weights ⊇ edge types), not equality.
_UNWEIGHTED_EDGE_TYPES: Final[frozenset[str]] = missing_laplacian_weights(
    EDGE_LAPLACIAN_WEIGHTS, EDGE_TYPES
)
if _UNWEIGHTED_EDGE_TYPES:  # pragma: no cover - import-time invariant
    raise RuntimeError(
        "EDGE_LAPLACIAN_WEIGHTS is not total over models.EDGE_TYPES; "
        f"missing: {sorted(_UNWEIGHTED_EDGE_TYPES)}. Add an explicit "
        "entry (0.0 excludes the type from the spectral graph)."
    )

# Top-K eigenpairs retained from the signed Laplacian. K=200 is the
# spec default: at bandwidth t=8 the heat-kernel filter exp(-tL)
# suppresses K>200 modes below 1e-3 contribution.
DEFAULT_K: Final[int] = 200

# Persisted .npz layout version. Bump on incompatible field changes.
_NPZ_VERSION: Final[int] = 1

# Explicit ARPACK iteration budget (#1370 §10). ARPACK's own default is
# `n * 10`; pinning it here keeps the budget a property of this module
# rather than of the installed SciPy, so a SciPy upgrade cannot silently
# change which eigenpairs converge. The floor covers tiny graphs, where
# `10 * n` is a handful of iterations.
ARPACK_MAXITER_FACTOR: Final[int] = 10
ARPACK_MAXITER_FLOOR: Final[int] = 1000


def build_signed_adjacency(
    store: MemoryStore,
    weights: dict[str, float] | None = None,
) -> tuple[sp.csr_matrix, list[str]]:
    """Build the symmetric signed adjacency ``W_sym = W + W.T``.

    Returns ``(W_sym, belief_ids)`` where ``belief_ids`` is the
    canonical row/column ordering used by every downstream consumer
    in this module. Edges referencing ids absent from
    ``list_belief_ids()`` are dropped (defensive: foreign-key
    invariant should prevent this, but a stale graph isn't worth
    crashing the build).
    """
    w_map = EDGE_LAPLACIAN_WEIGHTS if weights is None else weights
    belief_ids = store.list_belief_ids()
    n = len(belief_ids)
    if n == 0:
        return sp.csr_matrix((0, 0), dtype=np.float64), belief_ids

    idx = {bid: i for i, bid in enumerate(belief_ids)}
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for e in store.iter_all_edges():
        if e.src not in idx or e.dst not in idx:
            continue
        w = w_map.get(e.type, 0.0)
        if w == 0.0:
            continue
        rows.append(idx[e.src])
        cols.append(idx[e.dst])
        vals.append(w)

    W = sp.csr_matrix(
        (vals, (rows, cols)), shape=(n, n), dtype=np.float64
    )
    W_sym = (W + W.T).tocsr()
    return W_sym, belief_ids


def build_signed_normalized_laplacian(W: sp.csr_matrix) -> sp.csr_matrix:
    """Construct ``L = I - D_abs^(-1/2) W D_abs^(-1/2)`` (Kunegis 2010).

    ``D_abs[i,i] = sum_j |W[i,j]|`` — the absolute-degree
    normalization is what makes negative weights propagate as
    opposition rather than collapse into the standard normalization.
    Isolated nodes (zero absolute degree) get a zero row/column in
    the normalizer; their Laplacian row reduces to the identity
    contribution, which is the conventional handling.
    """
    n = W.shape[0]
    if n == 0:
        return sp.csr_matrix((0, 0), dtype=np.float64)

    abs_W = W.copy()
    abs_W.data = np.abs(abs_W.data)
    deg = np.asarray(abs_W.sum(axis=1)).ravel()
    with np.errstate(divide="ignore"):
        d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D = sp.diags(d_inv_sqrt)
    L = sp.eye(n, format="csr") - (D @ W @ D)
    # Symmetrize against floating-point asymmetry in the matvec
    # composition. W is symmetric by construction, but D @ W @ D can
    # accumulate ~1e-16 asymmetry that breaks `eigsh`.
    L = ((L + L.T) * 0.5).tocsr()
    return L


def _content_seed(L: "sp.spmatrix | np.ndarray") -> int:
    """Derive a 64-bit RNG seed from the matrix's own bytes.

    Deterministic in the graph, not in the clock or the process: the
    same Laplacian always yields the same seed, and any change to the
    structure or the weights yields a different one.
    """
    h = hashlib.blake2b(digest_size=8)
    if sp.issparse(L):
        csr = L.tocsr()
        h.update(np.asarray(csr.shape, dtype=np.int64).tobytes())
        h.update(np.ascontiguousarray(csr.indptr, dtype=np.int64).tobytes())
        h.update(np.ascontiguousarray(csr.indices, dtype=np.int64).tobytes())
        h.update(np.ascontiguousarray(csr.data, dtype=np.float64).tobytes())
    else:
        arr = np.ascontiguousarray(L, dtype=np.float64)
        h.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
        h.update(arr.tobytes())
    return int.from_bytes(h.digest(), "big")


def deterministic_start_vector(L: "sp.spmatrix | np.ndarray") -> np.ndarray:
    """Content-derived ARPACK start vector ``v0`` for ``L`` (#1370 §10).

    ``eigsh`` with ``v0=None`` lets ARPACK draw its own residual vector
    from an internal RNG whose state advances per call, so consecutive
    solves of the *same* matrix return different (though numerically
    equivalent) eigenvectors — and the heat-kernel authority ranking
    built on them is then not reproducible from the write log.

    A constant vector is not a usable substitute: it is close to an
    eigenvector of the normalized Laplacian, which degenerates the
    Krylov subspace. So the vector is pseudo-random but seeded from the
    matrix content itself.
    """
    n = L.shape[0]
    rng = np.random.default_rng(_content_seed(L))
    return rng.standard_normal(n)


def compute_eigenbasis(
    L: sp.csr_matrix, k: int = DEFAULT_K
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(eigvals, eigvecs)`` — smallest-magnitude eigenpairs.

    ``eigvals`` shape ``(k,)``; ``eigvecs`` shape ``(n, k)``. Sorted
    ascending by eigenvalue. For ``k >= n`` falls back to dense
    ``eigh`` (eigsh requires k < n).

    The sparse path is pinned to be reproducible (#1370 §10, #1157):
    ``v0`` is derived from ``L``'s own bytes, ``tol=0`` asks for machine
    precision rather than an early exit, and ``maxiter`` is explicit
    rather than inherited from ARPACK's default. Two calls on the same
    ``L`` return bit-identical arrays.
    """
    n = L.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.float64), np.zeros((0, 0), dtype=np.float64)
    k_eff = min(k, n - 1) if n > 1 else 1
    if k_eff < 1 or k_eff >= n:
        L_dense = L.toarray() if sp.issparse(L) else np.asarray(L)
        eigvals, eigvecs = np.linalg.eigh(L_dense)
        return eigvals[:k], eigvecs[:, :k]
    eigvals, eigvecs = spla.eigsh(
        L,
        k=k_eff,
        which="SM",
        v0=deterministic_start_vector(L),
        tol=0.0,
        maxiter=max(ARPACK_MAXITER_FACTOR * n, ARPACK_MAXITER_FLOOR),
    )
    order = np.argsort(eigvals)
    return eigvals[order], eigvecs[:, order]


def save_eigenbasis(
    path: str | Path,
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    belief_ids: list[str] | None = None,
) -> None:
    """Round-trippable persistence to ``.npz``. Creates parent dirs."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "version": np.array([_NPZ_VERSION], dtype=np.int32),
        "eigvals": np.asarray(eigvals, dtype=np.float64),
        "eigvecs": np.asarray(eigvecs, dtype=np.float64),
    }
    if belief_ids is not None:
        payload["belief_ids"] = np.asarray(belief_ids, dtype=object)
    np.savez(p, **payload)


def load_eigenbasis(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, list[str] | None]:
    """Inverse of :func:`save_eigenbasis`. ``belief_ids`` is ``None``
    if the archive was written without one."""
    z = np.load(path, allow_pickle=True)
    bids: list[str] | None = None
    if "belief_ids" in z.files:
        bids = [str(x) for x in z["belief_ids"].tolist()]
    return z["eigvals"], z["eigvecs"], bids


# --- #150 heat kernel scoring -------------------------------------------

# Bandwidth `t` for `exp(-tL)`. t=8 is the synthetic-graph optimum at
# K=200 for the authority-scoring task; smaller t localizes scoring to
# immediate neighborhood, larger t smooths globally. Ships as default.
DEFAULT_HEAT_BANDWIDTH: Final[float] = 8.0

# BM25 seed top-K. The heat kernel propagates a sparse seed distribution
# through the eigenbasis; only the highest-BM25 beliefs are seeded.
DEFAULT_BM25_SEED_TOP_K: Final[int] = 25

# Lower clamp for `heat_kernel_safe`. Negative or near-zero scores
# (signed-Laplacian heat can dip below zero through CONTRADICTS edges)
# are floored before the log to avoid `-inf` propagating into the
# log-additive score combination.
HEAT_SCORE_FLOOR: Final[float] = 1e-9

# Default mixing weights for the log-additive ranking formula
# (#150 spec § "Algorithm").
DEFAULT_HEAT_KERNEL_WEIGHT: Final[float] = 1.0
DEFAULT_POSTERIOR_LOG_WEIGHT: Final[float] = 0.5


def heat_kernel_score(
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    seeds: np.ndarray,
    t: float = DEFAULT_HEAT_BANDWIDTH,
) -> np.ndarray:
    """Apply the heat kernel ``exp(-tL)`` to a seed vector via the
    precomputed eigenbasis.

    ``eigvals`` shape ``(K,)``; ``eigvecs`` shape ``(N, K)``; ``seeds``
    shape ``(N,)``. Returns ``(N,)``: per-belief authority scores.

    Equivalent to ``U @ diag(exp(-t*Λ)) @ U.T @ seeds`` but written as
    two matvecs to avoid materializing the ``(N, N)`` dense kernel.
    Cost: ``O(N * K)`` per call; at ``N=50k, K=200`` that is ~7-8 ms in
    BLAS-backed numpy on commodity hardware (#150 AC5).
    """
    if eigvecs.size == 0:
        return np.zeros((eigvecs.shape[0],), dtype=np.float64)
    proj = eigvecs.T @ seeds
    filt = np.exp(-t * eigvals) * proj
    return eigvecs @ filt


def heat_kernel_safe(
    scores: np.ndarray, floor: float = HEAT_SCORE_FLOOR,
) -> np.ndarray:
    """Clamp heat-kernel scores to ``>= floor`` for safe ``log()``.

    Signed-Laplacian heat propagates negative authority through
    ``CONTRADICTS`` edges, so raw scores can be negative or zero on
    disconnected components. The log-additive ranking formula
    needs strictly positive inputs; the floor preserves ranking
    monotonicity (anything ≤ floor maps to the same minimum) while
    avoiding ``-inf``.
    """
    return np.maximum(scores, floor)


def seeds_from_bm25(
    bm25_scores: np.ndarray, top_k: int = DEFAULT_BM25_SEED_TOP_K,
) -> np.ndarray:
    """Build an L1-normalized seed vector from a BM25 score array.

    ``bm25_scores`` shape ``(N,)`` — one BM25 score per belief in the
    eigenbasis row order. Returns a dense ``(N,)`` vector with at most
    ``min(top_k, nnz)`` non-zero entries (the top-`top_k` BM25 scores)
    and L1 norm 1.0. Zero / negative BM25 scores are excluded — only
    strictly positive scores seed the kernel.

    Returns the all-zero vector when no seed has positive score
    (caller should detect and skip the heat-kernel pass; see #150
    spec § "Algorithm").
    """
    n = bm25_scores.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if n == 0 or top_k <= 0:
        return out
    positive = bm25_scores > 0
    if not np.any(positive):
        return out
    # argpartition is O(N); we only need the top_k indices, not a full
    # sort. Tie-breaking is arbitrary among equal scores — matches the
    # spec's "weighted by BM25 score" loose ordering.
    pos_idx = np.flatnonzero(positive)
    k = min(top_k, pos_idx.size)
    top_local = np.argpartition(-bm25_scores[pos_idx], k - 1)[:k]
    top = pos_idx[top_local]
    weights = bm25_scores[top]
    total = float(weights.sum())
    if total <= 0.0:
        return out
    out[top] = weights / total
    return out


def apply_broker_attenuation(
    scores: np.ndarray,
    store: "MemoryStore",
    belief_ids: list[str],
) -> np.ndarray:
    """Multiply each per-belief authority score by the recipient's
    broker confidence ``α / (α + β)``.

    ``scores[i]`` corresponds to ``belief_ids[i]``. Beliefs whose
    ``α + β`` is zero contribute factor 0 (no evidence ⇒ no broker
    confidence). Missing beliefs contribute factor 1.0 (passthrough)
    — defensive for index/store skew during a rebuild.

    The store-side ``propagate_valence`` already applies broker
    attenuation through multi-hop edge traversal. The eigenbasis-
    based heat kernel folds the multi-hop propagation into the
    spectral filter, so per-belief broker scaling on the OUTPUT is
    the right shape — it dampens scores accruing into low-confidence
    targets without re-walking the graph at query time.
    """
    if scores.size == 0:
        return scores
    factors = np.ones_like(scores, dtype=np.float64)
    for i, bid in enumerate(belief_ids):
        b = store.get_belief(bid)
        if b is None:
            continue
        denom = b.alpha + b.beta
        factors[i] = (b.alpha / denom) if denom > 0 else 0.0
    return scores * factors


def combine_log_scores(
    bm25f: float,
    heat: float,
    posterior: float | None = None,
    *,
    heat_weight: float = DEFAULT_HEAT_KERNEL_WEIGHT,
    posterior_weight: float = DEFAULT_POSTERIOR_LOG_WEIGHT,
    heat_floor: float = HEAT_SCORE_FLOOR,
) -> float:
    """Combine BM25F, heat-kernel, and posterior scores log-additively.

    Formula (#150 spec § "Score combination"):
        score = log(BM25F) + heat_weight * log(heat_safe(heat))
                           + posterior_weight * log(posterior_or_1)

    ``posterior=None`` collapses the third term (``log(1) = 0``),
    matching the spec's "defaults to 1.0 when no posterior data
    exists" contract. ``bm25f <= 0`` raises — BM25F is non-negative
    by construction and a non-positive value is a caller bug worth
    surfacing rather than silently flooring.
    """
    if bm25f <= 0.0:
        raise ValueError(f"bm25f must be > 0; got {bm25f}")
    safe_heat = max(heat, heat_floor)
    safe_post = posterior if (posterior is not None and posterior > 0.0) else 1.0
    return (
        float(np.log(bm25f))
        + heat_weight * float(np.log(safe_heat))
        + posterior_weight * float(np.log(safe_post))
    )


# --- Eigenbasis cache ---------------------------------------------------


@dataclass
class GraphEigenbasisCache:
    """In-memory eigenbasis with on-disk persistence and store-mutation
    invalidation.

    Construction registers an invalidation callback on the store; any
    belief or edge mutation invalidates the cached eigenbasis and
    deletes the persisted ``.npz``. ``build()`` is the explicit entry
    point — this class does not eagerly rebuild on invalidation,
    consistent with the offline-only contract (#149 spec).
    """

    store: MemoryStore
    path: Path
    k: int = DEFAULT_K
    eigvals: np.ndarray | None = field(default=None, init=False)
    eigvecs: np.ndarray | None = field(default=None, init=False)
    belief_ids: list[str] | None = field(default=None, init=False)
    _stale: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        self.store.add_invalidation_callback(self._on_store_mutation)

    def build(self) -> None:
        W, ids = build_signed_adjacency(self.store)
        L = build_signed_normalized_laplacian(W)
        ev, vc = compute_eigenbasis(L, k=self.k)
        self.eigvals = ev
        self.eigvecs = vc
        self.belief_ids = ids
        self._stale = False
        save_eigenbasis(self.path, ev, vc, ids)

    def is_stale(self) -> bool:
        return self._stale

    def _on_store_mutation(self) -> None:
        self.eigvals = None
        self.eigvecs = None
        self.belief_ids = None
        self._stale = True
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
