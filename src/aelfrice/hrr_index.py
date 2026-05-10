"""HRR structural-query lane (#152).

Per-belief structural encoding using the Plate (1995) bind / probe
algebra in :mod:`aelfrice.hrr`. A belief's outgoing edge structure
is encoded as a superposition of bound role-filler pairs:

    struct[b] = sum_{e in edges_from(b)} bind(role[e.kind], id_vec[e.dst])

A structural query of the form ``"<KIND>:<belief-id>"`` (e.g.
``"CONTRADICTS:b/abc"``) probes the index with
``bind(role[KIND], id_vec[target])`` and ranks beliefs by the
inner product against ``struct[b]``. Beliefs whose structure
contains exactly that bound term score high; orthogonal noise from
other bindings is ``~1/sqrt(dim)`` per term, so at the default
``dim=512`` the top-K is dominated by true structural matches.

Parallel to the textual lane (BM25F + heat kernel), not a competitor.
A query parser (:func:`parse_structural_marker`) routes structural
queries to this lane and falls through to the textual lane otherwise.

Default-ON behind ``use_hrr_structural`` per the #154 composition
tracker, which flipped the default after the #437 reproducibility-
harness gate cleared at 11/11. Wired into
:func:`aelfrice.retrieval.retrieve_v2` as a parallel routing branch
that fires before vocab-bridge rewrite — on a structural-marker hit
the textual lane is bypassed entirely; on miss the call falls through
to BM25F + heat kernel as before. Opt out via env var
``AELFRICE_HRR_STRUCTURAL=0``, the kwarg, or
``[retrieval] use_hrr_structural = false`` in ``.aelfrice.toml``.

Long-running callers should pass an explicit
:class:`HRRStructIndexCache` to amortise the build cost across
queries. The cache subscribes to the store's invalidation registry
so any belief / edge mutation drops the index transparently.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np

from aelfrice.hrr import DEFAULT_DIM, Vector, bind, random_vector
from aelfrice.models import EDGE_TYPES
from aelfrice.store import MemoryStore

# Structural-marker regex: an uppercase-with-underscores edge type
# followed by a colon and a non-empty belief id. Matched
# case-sensitively because ``EDGE_TYPES`` is uppercase by
# convention and accidental ``contradicts:foo`` should fall through
# to the textual lane rather than silently route to a structural
# probe with a typo'd kind.
_STRUCT_MARKER_RE: Final[re.Pattern[str]] = re.compile(
    r"^([A-Z][A-Z_]*):(\S.*)$",
)

# Persisted ``.npz`` layout version. Bump on incompatible changes.
_NPZ_VERSION: Final[int] = 1


def parse_structural_marker(query: str) -> tuple[str, str] | None:
    """Detect a ``"<KIND>:<target_id>"`` structural query.

    Returns ``(kind, target_id)`` when the query matches a supported
    edge kind followed by a non-empty target id; otherwise ``None``
    so the caller routes to the textual lane.

    The kind must be an exact member of :data:`aelfrice.models.EDGE_TYPES`.
    Case-sensitive: ``contradicts:b1`` does not match because edge
    types are uppercase. Whitespace inside the target is preserved
    verbatim except for a leading-trailing strip — belief ids in
    aelfrice are tokens without internal spaces, but a malicious
    query that tries to inject a marker by suffixing one is still
    rejected by the regex's ``\\S.*`` floor.
    """
    m = _STRUCT_MARKER_RE.match(query.strip())
    if m is None:
        return None
    kind = m.group(1)
    target = m.group(2).strip()
    if kind not in EDGE_TYPES:
        return None
    if not target:
        return None
    return kind, target


def _seed_from_path(path: str | None, salt: int = 0) -> int:
    """Stable 64-bit seed from a store path string. ``hash()`` varies
    across Python runs (PYTHONHASHSEED); md5 over UTF-8 bytes is the
    deterministic substitute. ``salt`` lets a caller fan out one
    seed into a family by XOR (``seed ^ k``)."""
    payload = (path or "").encode("utf-8")
    digest = hashlib.md5(payload).digest()
    base = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return (base ^ salt) & 0xFFFFFFFFFFFFFFFF


@dataclass
class HRRStructIndex:
    """In-memory HRR structural index over outgoing edges.

    Build is offline (walks every edge once); query is one matvec
    against the ``(N, dim)`` struct matrix plus one bind. Storage
    cost is dominated by the struct matrix at ``8 * N * dim`` bytes;
    at ``N=50k`` and the default ``dim=512`` that is ~200 MB
    (~800 MB at the ``dim=2048`` escape-hatch value), both within
    the AC8 budget.

    Determinism: ``random_vector`` draws are reproducible from
    ``np.random.default_rng(seed)``. Two builds against the same
    store path produce byte-identical struct matrices when the
    belief / edge set is unchanged.
    """

    dim: int = DEFAULT_DIM
    seed: int = 0
    belief_ids: list[str] = field(default_factory=lambda: [])
    struct: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.float64),
    )
    id_vecs: dict[str, Vector] = field(default_factory=lambda: {})
    role_vecs: dict[str, Vector] = field(default_factory=lambda: {})
    _index: dict[str, int] = field(default_factory=lambda: {}, init=False)

    def build(
        self,
        store: MemoryStore,
        *,
        store_path: str | None = None,
        seed: int | None = None,
    ) -> None:
        """Walk the store and materialise ``struct`` plus the id /
        role vector tables.

        ``seed`` (explicit) wins over ``store_path`` (derived); if
        neither is supplied, the dataclass field's ``seed`` is used.
        Re-running ``build`` over the same store with the same seed
        produces a byte-identical struct matrix (AC2).
        """
        if seed is None and store_path is not None:
            seed = _seed_from_path(store_path)
        if seed is not None:
            self.seed = seed

        bids = store.list_belief_ids()
        self.belief_ids = list(bids)
        self._index = {bid: i for i, bid in enumerate(self.belief_ids)}
        n = len(bids)
        if n == 0:
            self.struct = np.zeros((0, self.dim), dtype=np.float64)
            self.id_vecs = {}
            self.role_vecs = {}
            return

        # One Generator per draw stream — id-vec stream and role-vec
        # stream — so adding a belief in a later build does not
        # rotate the role-vec sequence and invalidate cached probes.
        id_rng = np.random.default_rng(self.seed)
        role_rng = np.random.default_rng(self.seed ^ 0xA5A5A5A5)

        self.id_vecs = {bid: random_vector(self.dim, id_rng) for bid in bids}
        # Role vectors are sorted by edge-type name so the build is
        # deterministic across Python dict-iteration orderings.
        self.role_vecs = {
            kind: random_vector(self.dim, role_rng)
            for kind in sorted(EDGE_TYPES)
        }

        struct = np.zeros((n, self.dim), dtype=np.float64)
        for i, bid in enumerate(bids):
            terms: list[Vector] = []
            for e in store.edges_from(bid):
                rv = self.role_vecs.get(e.type)
                iv = self.id_vecs.get(e.dst)
                if rv is None or iv is None:
                    continue
                terms.append(bind(rv, iv))
            if terms:
                struct[i] = np.sum(np.stack(terms, axis=0), axis=0)
        self.struct = struct

    def probe(
        self, kind: str, target_id: str, top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Return top-K ``(belief_id, score)`` pairs in descending
        score order.

        ``score(b) = struct[b] . bind(role[kind], id_vec[target])``.
        Returns an empty list when the index is empty, the kind is
        unknown, or the target is not a belief in the index. Tie
        ordering is whatever ``argpartition`` decides — ``score``
        ties are vanishingly rare in floating-point HRR.
        """
        if self.struct.size == 0 or top_k <= 0:
            return []
        rv = self.role_vecs.get(kind)
        iv = self.id_vecs.get(target_id)
        if rv is None or iv is None:
            return []
        probe_vec = bind(rv, iv)
        scores = self.struct @ probe_vec
        n = len(self.belief_ids)
        k = min(top_k, n)
        if k <= 0:
            return []
        if k == n:
            order = np.argsort(-scores)
        else:
            top_idx = np.argpartition(-scores, k - 1)[:k]
            order = top_idx[np.argsort(-scores[top_idx])]
        return [
            (self.belief_ids[int(i)], float(scores[int(i)]))
            for i in order
        ]

    def noise_floor(self) -> float:
        """Expected per-bound-pair orthogonal-noise magnitude
        (``~1/sqrt(dim)``). AC5: probe scores below this floor
        carry no signal."""
        return 1.0 / float(np.sqrt(self.dim))

    # --- persistence ------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Round-trippable persistence to ``.npz``."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # id_vecs + role_vecs are stored as parallel name / matrix
        # arrays so callers can reconstruct dicts without pickling
        # arbitrary objects.
        id_names = np.asarray(list(self.id_vecs.keys()), dtype=object)
        id_matrix = (
            np.stack(list(self.id_vecs.values()), axis=0)
            if self.id_vecs
            else np.zeros((0, self.dim), dtype=np.float64)
        )
        role_names = np.asarray(list(self.role_vecs.keys()), dtype=object)
        role_matrix = (
            np.stack(list(self.role_vecs.values()), axis=0)
            if self.role_vecs
            else np.zeros((0, self.dim), dtype=np.float64)
        )
        np.savez(
            p,
            version=np.array([_NPZ_VERSION], dtype=np.int32),
            dim=np.array([self.dim], dtype=np.int64),
            seed=np.array([self.seed], dtype=np.int64),
            belief_ids=np.asarray(self.belief_ids, dtype=object),
            struct=self.struct,
            id_names=id_names,
            id_matrix=id_matrix,
            role_names=role_names,
            role_matrix=role_matrix,
        )

    @classmethod
    def load(cls, path: str | Path) -> "HRRStructIndex":
        """Inverse of :meth:`save`."""
        z = np.load(path, allow_pickle=True)
        idx = cls(dim=int(z["dim"][0]), seed=int(z["seed"][0]))
        idx.belief_ids = [str(x) for x in z["belief_ids"].tolist()]
        idx._index = {bid: i for i, bid in enumerate(idx.belief_ids)}
        idx.struct = z["struct"]
        id_names = [str(x) for x in z["id_names"].tolist()]
        id_matrix = z["id_matrix"]
        idx.id_vecs = {
            n: id_matrix[i] for i, n in enumerate(id_names)
        }
        role_names = [str(x) for x in z["role_names"].tolist()]
        role_matrix = z["role_matrix"]
        idx.role_vecs = {
            n: role_matrix[i] for i, n in enumerate(role_names)
        }
        return idx


@dataclass
class HRRStructIndexCache:
    """Lazy, invalidation-aware wrapper around a single ``HRRStructIndex``.

    Subscribes to the store's invalidation callback registry on
    construction, so any belief / edge mutation drops the cached
    index. The next ``get()`` rebuilds.

    Per-instance: two caches pointing at different stores never share
    state. Thread safety is the caller's responsibility.
    """

    store: MemoryStore
    dim: int = DEFAULT_DIM
    store_path: str | None = None
    seed: int | None = None
    _index: HRRStructIndex | None = field(default=None, init=False, repr=False)
    _subscribed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self._subscribed:
            self.store.add_invalidation_callback(self.invalidate)
            self._subscribed = True

    def get(self) -> HRRStructIndex:
        """Return the current index, building or rebuilding as needed."""
        if self._index is None:
            idx = HRRStructIndex(dim=self.dim)
            idx.build(self.store, store_path=self.store_path, seed=self.seed)
            self._index = idx
        return self._index

    def invalidate(self) -> None:
        """Drop the cached index. Wired to the store mutation hook."""
        self._index = None
