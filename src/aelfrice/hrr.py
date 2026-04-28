"""Holographic Reduced Representations primitives (Plate 1995).

Circular-convolution binding, FFT-correlation unbinding, vector
superposition, and a cleanup-memory utility for nearest-neighbor
recovery via cosine similarity.

This module ships the algebra only. No integration into the
retrieval surface; consumers (e.g. #152's HRR structural-query
lane) build their indices on top of these primitives.

Pure-numpy. Numpy is already a runtime dep at v1.5.0 (BM25F
sparse-matvec work, #148); no dep-policy break.

Reference: Plate, T.A. (1995), *Holographic Reduced
Representations*, IEEE Transactions on Neural Networks 6(3).
"""
from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Vector = npt.NDArray[np.float64]

# Default dimensionality. Capacity per Plate's analysis is roughly
# dim/9 retrievable bound pairs at signal-to-noise threshold; 2048
# accommodates ~227 distinct bindings, well above aelfrice's typical
# ~5 outgoing edges per belief.
DEFAULT_DIM: Final[int] = 2048


# ---------------------------------------------------------------------------
# Core HRR operations
# ---------------------------------------------------------------------------


def random_vector(dim: int, rng: np.random.Generator) -> Vector:
    """Sample a unit vector from N(0, 1/sqrt(n)) and normalise.

    Random vectors drawn this way are approximately orthogonal in
    high dimension — the property HRR algebra relies on. The caller
    supplies the ``Generator`` so determinism is the consumer's
    responsibility (e.g. seed from ``hash(store_path) ^ index``)."""
    v: Vector = rng.normal(0, 1.0 / np.sqrt(dim), size=dim).astype(np.float64)
    norm: float = float(np.linalg.norm(v))
    if norm > 0:
        v = (v / norm).astype(np.float64)
    return v


def bind(a: Vector, b: Vector) -> Vector:
    """Circular convolution of ``a`` and ``b`` (the HRR bind operation).

    Computed via FFT: ``ifft(fft(a) * fft(b))``. Commutative and
    associative; distributive over superposition. Real-valued by
    construction (the imaginary part is FFT round-off; we drop it)."""
    result: Vector = np.real(
        np.fft.ifft(np.fft.fft(a) * np.fft.fft(b))
    ).astype(np.float64)
    return result


def unbind(key: Vector, composite: Vector) -> Vector:
    """Circular correlation: approximate inverse of ``bind(key, .)``.

    ``unbind(k, bind(k, v))`` recovers ``v`` exactly in the noise-free
    single-pair case. In a superposition ``s = sum_i bind(k_i, v_i)``,
    ``unbind(k_j, s)`` recovers ``v_j`` plus orthogonal noise of
    magnitude ``~1/sqrt(dim)`` per bound term."""
    result: Vector = np.real(
        np.fft.ifft(np.conj(np.fft.fft(key)) * np.fft.fft(composite))
    ).astype(np.float64)
    return result


def superpose(vectors: list[Vector]) -> Vector:
    """Vector-sum bundle. Accepts an empty list (returns zero-vector
    of dimension 0; callers must avoid this corner if their pipeline
    cannot tolerate a 0-d array)."""
    if not vectors:
        return np.zeros(0, dtype=np.float64)
    result: Vector = np.sum(np.array(vectors), axis=0).astype(np.float64)
    return result


def cosine_similarity(a: Vector, b: Vector) -> float:
    """Cosine similarity ``a . b / (|a||b|)``. Returns 0.0 for either
    vector having zero norm (Plate 1995 §3 conventions)."""
    na: float = float(np.linalg.norm(a))
    nb: float = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Cleanup memory: nearest-neighbour recovery
# ---------------------------------------------------------------------------


class CleanupMemory:
    """Labeled-vector store with batched cosine-similarity recovery.

    Standard HRR cleanup pattern: after ``unbind`` the recovered
    vector is approximate — find the nearest stored *clean* vector
    and return its label. Recall is monotone in dimension and in the
    superposition's bound-pair count.

    The vector matrix is materialised lazily and invalidated on every
    ``add`` to keep insertion O(1)."""

    def __init__(self) -> None:
        self._labels: list[str] = []
        self._vectors: list[Vector] = []
        self._matrix: Vector | None = None

    def add(self, label: str, vector: Vector) -> None:
        self._labels.append(label)
        self._vectors.append(vector)
        self._matrix = None

    def query(self, probe: Vector, top_k: int = 10) -> list[tuple[str, float]]:
        """Top-K nearest labels by cosine similarity, descending."""
        if not self._labels:
            return []
        if self._matrix is None:
            self._matrix = np.array(self._vectors, dtype=np.float64)
        norms: Vector = np.linalg.norm(self._matrix, axis=1).astype(np.float64)
        norms = np.where(norms > 0, norms, 1.0).astype(np.float64)
        normalized: Vector = (self._matrix / norms[:, np.newaxis]).astype(np.float64)
        probe_norm: float = float(np.linalg.norm(probe))
        if probe_norm == 0:
            return []
        probe_normalized: Vector = (probe / probe_norm).astype(np.float64)
        sims: Vector = (normalized @ probe_normalized).astype(np.float64)
        k: int = min(top_k, len(self._labels))
        # `argpartition` is O(N); `argsort` only the K winners after.
        top_indices: npt.NDArray[np.intp] = np.argpartition(sims, -k)[-k:]
        top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]
        return [(self._labels[int(i)], float(sims[int(i)])) for i in top_indices]

    def size(self) -> int:
        return len(self._labels)
