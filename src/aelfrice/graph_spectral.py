# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeStubs=false, reportOptionalSubscript=false, reportAttributeAccessIssue=false, reportGeneralTypeIssues=false, reportConstantRedefinition=false, reportArgumentType=false
"""Signed normalized Laplacian + offline eigenbasis builder (#149).

Constructs the signed normalized Laplacian
``L = I - D_abs^(-1/2) W_sym D_abs^(-1/2)``
over the typed edge graph (Kunegis 2010), where ``CONTRADICTS`` /
``SUPERSEDES`` count as negative-weight edges and the supportive edge
families as positive. The top-K=200 eigenpairs of ``L`` are the offline
foundation consumed by the heat-kernel authority signal (#150). This
module ships only the builder + serialization; no integration into
``retrieve()``.

All construction is offline — no query-time cost. Rebuild is wired to
the same store-mutation invalidation callback that ``RetrievalCache``
subscribes to (extending the registry, not duplicating it).
"""
from __future__ import annotations

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
    EDGE_RELATES_TO,
    EDGE_SUPERSEDES,
    EDGE_SUPPORTS,
)
from aelfrice.store import MemoryStore

# Edge-type → signed weight mapping for Laplacian construction.
# Distinct from `models.EDGE_TYPE_WEIGHTS` (used for valence
# propagation; different polarity convention). Per #149 spec; exposed
# as a constant so a follow-up sweep issue can override without
# touching call sites.
EDGE_LAPLACIAN_WEIGHTS: Final[dict[str, float]] = {
    EDGE_SUPPORTS: 1.0,
    EDGE_CITES: 1.0,
    EDGE_RELATES_TO: 1.0,
    EDGE_DERIVED_FROM: 1.0,
    EDGE_CONTRADICTS: -1.0,
    EDGE_SUPERSEDES: -0.5,
}

# Top-K eigenpairs retained from the signed Laplacian. K=200 is the
# spec default: at bandwidth t=8 the heat-kernel filter exp(-tL)
# suppresses K>200 modes below 1e-3 contribution.
DEFAULT_K: Final[int] = 200

# Persisted .npz layout version. Bump on incompatible field changes.
_NPZ_VERSION: Final[int] = 1


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


def compute_eigenbasis(
    L: sp.csr_matrix, k: int = DEFAULT_K
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(eigvals, eigvecs)`` — smallest-magnitude eigenpairs.

    ``eigvals`` shape ``(k,)``; ``eigvecs`` shape ``(n, k)``. Sorted
    ascending by eigenvalue. For ``k >= n`` falls back to dense
    ``eigh`` (eigsh requires k < n).
    """
    n = L.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.float64), np.zeros((0, 0), dtype=np.float64)
    k_eff = min(k, n - 1) if n > 1 else 1
    if k_eff < 1 or k_eff >= n:
        L_dense = L.toarray() if sp.issparse(L) else np.asarray(L)
        eigvals, eigvecs = np.linalg.eigh(L_dense)
        return eigvals[:k], eigvecs[:, :k]
    eigvals, eigvecs = spla.eigsh(L, k=k_eff, which="SM")
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
