"""HRR vocabulary bridge (#433).

Closes the vocabulary-gap-recovery claim on the **query side**. A query
token whose surface form does not appear verbatim in the corpus's
canonical vocabulary gets bridged to one or more canonical-entity
tokens before any retrieval lane fires.

Pure linear algebra over the surface-form token universe the corpus
already exposes. No learned components, no LLM call, no embedding
model. Deterministic build from a path-derived seed (mirrors
``HRRStructIndex`` at :mod:`aelfrice.hrr_index`).

Spec: ``docs/feature-hrr-vocab-bridge.md``.

Algorithm (build):

    bridge_vec = sum_{c in canonicals} sum_{s in surface_forms(c)}
                     bind(token_vec[s], canonical_vec[c])

A single ``(dim,)`` superposition encodes every (surface, canonical)
pair the corpus exposes. Cleanup memory holds ``(canonical_token,
canonical_vec)`` so the rewrite step can map a recovered vector back
to a string.

Algorithm (rewrite, per query token ``t``):

    recovered = unbind(token_vec[t], bridge_vec)
    for (canonical, score) in cleanup_memory.query(recovered, top_k):
        if score >= min_score and canonical not in already_appended:
            append canonical

Tokens not seen at build-time short-circuit (no ``token_vec[t]``);
tokens with no canonical above ``noise_floor() = 1/sqrt(dim)`` drop;
tokens that are themselves canonical self-recover (cosine ≈ 1) and
are appended once.

In-memory only at v2.0.0. Persistence is the ``.npz`` round-trip
pattern at :mod:`aelfrice.hrr_index` if a future revision needs it.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np

from aelfrice.bm25 import tokenize as bm25_tokenize
from aelfrice.entity_extractor import extract_entities
from aelfrice.hrr import DEFAULT_DIM, Vector, bind, random_vector, unbind
from aelfrice.models import LOCK_USER
from aelfrice.store import MemoryStore

# XOR salt for the surface-form (token) draw stream. The id stream uses
# the unsalted seed; the role/token stream uses ``seed ^ _TOKEN_SALT``.
# Mirrors the role/id stream split in ``HRRStructIndex.build`` at
# ``hrr_index.py:148-149``. Constant chosen as a 64-bit pattern so it
# fans out evenly under XOR; value is documentation, not crypto.
_TOKEN_SALT: Final[int] = 0xC3C3C3C3C3C3C3C3

# Default top-K canonical rewrites appended per query token.
DEFAULT_TOP_K: Final[int] = 3


def _seed_from_path(path: str | None, salt: int = 0) -> int:
    """Stable 64-bit seed from a store-path string. Same convention as
    ``hrr_index._seed_from_path`` so two builds against the same path
    produce byte-identical bridges."""
    payload = (path or "").encode("utf-8")
    digest = hashlib.md5(payload).digest()
    base = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return (base ^ salt) & 0xFFFFFFFFFFFFFFFF


@dataclass
class _Vocabulary:
    """Surface-form → canonical-token grouping built during ``build()``.

    `canonical_to_surfaces[c]` is the set of lowercased surface-form
    tokens the corpus has used for canonical token ``c``. ``c`` itself
    is always a member of its own surface-form set (the canonical
    self-recovery case).
    """

    canonical_to_surfaces: dict[str, set[str]] = field(default_factory=dict)

    def add(self, canonical: str, surface: str) -> None:
        bucket = self.canonical_to_surfaces.setdefault(canonical, set())
        bucket.add(surface)

    def canonicals(self) -> list[str]:
        return sorted(self.canonical_to_surfaces.keys())

    def surface_forms(self, canonical: str) -> list[str]:
        return sorted(self.canonical_to_surfaces.get(canonical, set()))

    def all_surfaces(self) -> list[str]:
        out: set[str] = set()
        for surfaces in self.canonical_to_surfaces.values():
            out.update(surfaces)
        return sorted(out)


@dataclass
class VocabBridge:
    """In-memory vocabulary bridge over a ``MemoryStore``.

    Build is offline (one walk over beliefs + their incoming
    anchor-text edges); rewrite is one ``unbind`` and one
    cleanup-memory query per query token. Storage cost is dominated
    by the surface-form ``(N_surfaces, dim)`` matrix at ``8 *
    N_surfaces * dim`` bytes; at ``N=10k, dim=2048`` that is ~160 MB.

    Determinism: ``random_vector`` draws are reproducible from
    ``np.random.default_rng(seed)``. Two builds against the same store
    path with an unchanged corpus produce a byte-identical
    ``bridge_vec`` (acceptance #3 in the spec).
    """

    dim: int = DEFAULT_DIM
    seed: int = 0
    canonicals: list[str] = field(default_factory=lambda: [])
    surfaces: list[str] = field(default_factory=lambda: [])
    canonical_vecs: dict[str, Vector] = field(default_factory=lambda: {})
    token_vecs: dict[str, Vector] = field(default_factory=lambda: {})
    bridge_vec: Vector = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64),
    )
    _cleanup_matrix: Vector | None = field(default=None, init=False)
    _cleanup_norms: Vector | None = field(default=None, init=False)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        store: MemoryStore,
        *,
        store_path: str | None = None,
        seed: int | None = None,
    ) -> None:
        """Walk the store and materialise ``bridge_vec`` plus the
        canonical-vec / token-vec tables.

        ``seed`` (explicit) wins over ``store_path`` (derived); if
        neither is supplied, the dataclass field's ``seed`` is used.
        Re-running ``build`` over the same store with the same seed
        produces a byte-identical ``bridge_vec``.
        """
        if seed is None and store_path is not None:
            seed = _seed_from_path(store_path)
        if seed is not None:
            self.seed = seed

        vocab = self._harvest(store)
        self.canonicals = vocab.canonicals()
        self.surfaces = vocab.all_surfaces()

        if not self.canonicals:
            self.canonical_vecs = {}
            self.token_vecs = {}
            self.bridge_vec = np.zeros(self.dim, dtype=np.float64)
            self._cleanup_matrix = None
            self._cleanup_norms = None
            return

        # Two Generators, mirroring the id / role stream split at
        # ``hrr_index.py:148-149``. Adding a new surface form in a
        # later build does not rotate the canonical-vec stream.
        canonical_rng = np.random.default_rng(self.seed)
        token_rng = np.random.default_rng(self.seed ^ _TOKEN_SALT)

        self.canonical_vecs = {
            c: random_vector(self.dim, canonical_rng) for c in self.canonicals
        }
        self.token_vecs = {
            s: random_vector(self.dim, token_rng) for s in self.surfaces
        }

        bridge = np.zeros(self.dim, dtype=np.float64)
        for c in self.canonicals:
            cv = self.canonical_vecs[c]
            for s in vocab.surface_forms(c):
                tv = self.token_vecs[s]
                bridge += bind(tv, cv)
        self.bridge_vec = bridge

        # Cleanup memory: one row per canonical, used by
        # ``rewrite()`` to map a recovered vector back to a string.
        # We materialise the matrix and per-row norms once so each
        # rewrite call avoids the ``CleanupMemory.query`` rebuild.
        matrix = np.array(
            [self.canonical_vecs[c] for c in self.canonicals],
            dtype=np.float64,
        )
        norms = np.linalg.norm(matrix, axis=1).astype(np.float64)
        norms = np.where(norms > 0, norms, 1.0).astype(np.float64)
        self._cleanup_matrix = matrix
        self._cleanup_norms = norms

    def _harvest(self, store: MemoryStore) -> _Vocabulary:
        """Walk the store and build the surface-form → canonical map.

        Three sources, in priority order (per spec § Build):

          1. Anchor-text under ``iter_incoming_anchor_text()`` — same
             field BM25F (#148) consumes.
          2. Belief content tokens — extracted via the entity-index
             lane fallback (the ``beliefs.entity_id`` column does not
             yet exist on the v2.0.0 schema; spec open-question 1
             accepts this fallback).
          3. Lock-asserted statements (``Belief.lock_state == LOCK_USER``)
             — included via source #2 with a single membership pass.

        Sources are unioned, not weighted: the same canonical may
        receive surface forms from any of the three. Belief / edge
        iteration order is the store's; ``surface_forms()`` returns
        sorted output downstream so build-determinism is preserved.
        """
        vocab = _Vocabulary()

        def _ingest(text: str) -> None:
            for ent in extract_entities(text):
                # Single-token canonicals only. Multi-word noun phrases
                # from ``KIND_NOUN_PHRASE`` are dropped — the bridge
                # rewrites query tokens, not query spans, so a phrase
                # canonical can never be self-recovered by a single
                # query token. Identifiers, file paths, URLs, error
                # codes, versions, and branches all pass this filter.
                low = ent.lower.strip()
                if not low or " " in low:
                    continue
                vocab.add(low, low)
            # Plain word tokens from BM25 tokenize as a fallback so
            # corpora without identifier-shaped names still produce a
            # canonical surface. Tokens shorter than 3 chars are
            # dropped — one- and two-letter words bind generically and
            # bloat the bridge with no recoverable signal.
            for tok in bm25_tokenize(text):
                if len(tok) < 3:
                    continue
                vocab.add(tok, tok)

        # Source #1: incoming anchor text on every edge.
        for _dst, anchor_text in store.iter_incoming_anchor_text():
            _ingest(anchor_text)

        # Source #2 + #3: belief content (locked beliefs included).
        for bid in store.list_belief_ids():
            belief = store.get_belief(bid)
            if belief is None:
                continue
            _ingest(belief.content)
            # Source #3 is a flag on top of source #2 in this MVP. The
            # spec lists it as a separate source for traceability;
            # weighting locked surfaces is deferred to a future spec
            # revision (open question 1).
            if belief.lock_level == LOCK_USER:
                pass

        return vocab

    # ------------------------------------------------------------------
    # Rewrite
    # ------------------------------------------------------------------

    def rewrite(
        self,
        query: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        min_score: float | None = None,
    ) -> str:
        """Append canonical-entity rewrites to the query.

        Returns ``query + " " + " ".join(appended)``. The original
        query is preserved verbatim — bridged candidates are
        appended, never substituted, so downstream lanes see no
        regression on already-canonical tokens.

        Tokens unseen at build time short-circuit; tokens that are
        themselves canonical self-recover and are appended once;
        tokens with no canonical above ``min_score`` drop silently.
        """
        if not self.canonicals or self.bridge_vec.size == 0:
            return query
        if top_k <= 0:
            return query
        threshold = min_score if min_score is not None else self.noise_floor()

        appended: list[str] = []
        appended_set: set[str] = set()
        # Canonical tokens already present in the raw query do not
        # need rebridging — skip them so the rewriter is idempotent
        # over its own output.
        already_in_query: set[str] = set(bm25_tokenize(query))

        for token in bm25_tokenize(query):
            tv = self.token_vecs.get(token)
            if tv is None:
                continue
            recovered = unbind(tv, self.bridge_vec)
            for canonical, score in self._cleanup_query(recovered, top_k):
                if score < threshold:
                    continue
                if canonical in appended_set:
                    continue
                if canonical in already_in_query:
                    continue
                appended.append(canonical)
                appended_set.add(canonical)

        if not appended:
            return query
        return f"{query} {' '.join(appended)}"

    def _cleanup_query(
        self, probe: Vector, top_k: int,
    ) -> list[tuple[str, float]]:
        """Top-K canonical labels by cosine similarity to ``probe``.

        Inlined cleanup-memory query — the matrix and norms are
        materialised once at build time so we skip
        ``CleanupMemory.query``'s lazy-rebuild check.
        """
        if self._cleanup_matrix is None or self._cleanup_norms is None:
            return []
        if not self.canonicals:
            return []
        probe_norm = float(np.linalg.norm(probe))
        if probe_norm == 0:
            return []
        normalized = self._cleanup_matrix / self._cleanup_norms[:, np.newaxis]
        sims = (normalized @ (probe / probe_norm)).astype(np.float64)
        n = len(self.canonicals)
        k = min(top_k, n)
        if k <= 0:
            return []
        if k == n:
            order = np.argsort(-sims)
        else:
            top_idx = np.argpartition(-sims, k - 1)[:k]
            order = top_idx[np.argsort(-sims[top_idx])]
        return [
            (self.canonicals[int(i)], float(sims[int(i)])) for i in order
        ]

    def noise_floor(self) -> float:
        """Per-bound-pair orthogonal-noise magnitude (``~1/sqrt(dim)``).

        Same convention as :meth:`HRRStructIndex.noise_floor`. Probe
        scores below this floor carry no signal; the rewrite step uses
        it as the default ``min_score``.
        """
        return 1.0 / float(np.sqrt(self.dim))

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def size(self) -> int:
        """Number of canonical entities in the bridge."""
        return len(self.canonicals)
