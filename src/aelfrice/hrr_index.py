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
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np

_logger = logging.getLogger(__name__)
_legacy_deprecation_logged = False
_ephemeral_disable_logged: bool = False

from aelfrice.hrr import DEFAULT_DIM, Vector, bind, random_vector
from aelfrice.models import EDGE_TYPES
from aelfrice.store import MemoryStore

# ---------------------------------------------------------------------------
# Build-stats slot (#696): wall-clock duration of the most recent
# HRRStructIndex.build() call in this process.
#
# Module-level dict keeps the read side accessible without changing the
# public API of HRRStructIndex. None means "no build has fired this process".
# ---------------------------------------------------------------------------
_HRR_BUILD_STATS: dict[str, float | None] = {"last_build_seconds": None}


def last_build_seconds() -> float | None:
    """Return wall-clock duration (seconds) of the most recent build, or None.

    Returns ``None`` when no ``HRRStructIndex.build()`` call has completed
    in this process. Used by ``aelf doctor`` and ``aelf status`` (#696).
    """
    return _HRR_BUILD_STATS["last_build_seconds"]


def persist_disk_bytes(persist_dir: Path | None) -> int:
    """Return on-disk size of the persisted index files in bytes.

    Sums ``struct.npy`` + ``meta.npz`` when both exist inside
    ``persist_dir``. Returns ``0`` when ``persist_dir`` is ``None``,
    when the directory does not exist, or when either file is absent.
    ``OSError`` / ``FileNotFoundError`` are caught and mapped to ``0``.
    """
    if persist_dir is None:
        return 0
    try:
        struct_size = os.path.getsize(persist_dir / _STRUCT_FILENAME)
        meta_size = os.path.getsize(persist_dir / _META_FILENAME)
        return struct_size + meta_size
    except (FileNotFoundError, OSError):
        return 0


# Structural-marker regex: an uppercase-with-underscores edge type
# followed by a colon and a non-empty belief id. Matched
# case-sensitively because ``EDGE_TYPES`` is uppercase by
# convention and accidental ``contradicts:foo`` should fall through
# to the textual lane rather than silently route to a structural
# probe with a typo'd kind.
_STRUCT_MARKER_RE: Final[re.Pattern[str]] = re.compile(
    r"^([A-Z][A-Z_]*):(\S.*)$",
)

# Persisted layout version. Bump on incompatible changes.
# v1: bundled ``.npz`` (legacy, v1.7 ship format).
# v2: split format — ``struct.npy`` + ``meta.npz`` in a per-store directory
#     (required for ``mmap_mode='r'``; see ``docs/feature-hrr-integration.md``).
_LAYOUT_VERSION: Final[int] = 2
_NPZ_VERSION: Final[int] = _LAYOUT_VERSION  # back-compat alias

# Split-format filenames inside the persistence directory.
_STRUCT_FILENAME: Final[str] = "struct.npy"
_META_FILENAME: Final[str] = "meta.npz"


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
        _t0 = time.perf_counter()

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
            _HRR_BUILD_STATS["last_build_seconds"] = time.perf_counter() - _t0
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
        _HRR_BUILD_STATS["last_build_seconds"] = time.perf_counter() - _t0

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
        """Round-trippable persistence to a per-store directory.

        ``path`` is a directory; ``save`` writes two files inside it:

        - ``struct.npy`` — the ``(N, dim)`` float64 struct matrix,
          mmap-able via ``np.load(..., mmap_mode='r')``.
        - ``meta.npz`` — the small metadata blob (belief ids, role/id
          vectors, dim, seed, layout version).

        Writes are atomic via temp-file + ``os.replace`` so a reader
        process never observes a partial write.
        """
        d = Path(path)
        d.mkdir(parents=True, exist_ok=True)

        struct_final = d / _STRUCT_FILENAME
        meta_final = d / _META_FILENAME
        struct_tmp = d / (_STRUCT_FILENAME + ".tmp")
        meta_tmp = d / (_META_FILENAME + ".tmp")

        # id_vecs + role_vecs serialize as parallel name / matrix
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

        # np.save / np.savez append ``.npy`` / ``.npz`` to string paths
        # whose extension does not match; pass open file handles so the
        # caller-controlled ``.tmp`` suffix sticks for os.replace.
        with open(struct_tmp, "wb") as f:
            np.save(f, self.struct, allow_pickle=False)
        with open(meta_tmp, "wb") as f:
            np.savez(
                f,
                version=np.array([_LAYOUT_VERSION], dtype=np.int32),
                dim=np.array([self.dim], dtype=np.int64),
                seed=np.array([self.seed], dtype=np.uint64),
                belief_ids=np.asarray(self.belief_ids, dtype=object),
                id_names=id_names,
                id_matrix=id_matrix,
                role_names=role_names,
                role_matrix=role_matrix,
            )
        os.replace(struct_tmp, struct_final)
        os.replace(meta_tmp, meta_final)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        mmap: bool = False,
    ) -> "HRRStructIndex":
        """Inverse of :meth:`save`.

        Accepts both the split-format directory and a legacy bundled
        ``.npz`` file written by v1.7. Legacy loads emit a one-shot
        deprecation log (the file still loads).

        ``mmap=True`` requests ``mmap_mode='r'`` for the struct matrix;
        only honoured by the split format (the legacy ``.npz`` cannot
        be mmap'd per the numpy docs and silently ignores the flag).
        """
        p = Path(path)
        if p.is_dir() and (p / _STRUCT_FILENAME).is_file() and (
            p / _META_FILENAME
        ).is_file():
            return cls._load_split(p, mmap=mmap)
        if p.is_file():
            return cls._load_legacy_npz(p)
        raise FileNotFoundError(
            f"HRRStructIndex.load: no split-format directory or legacy "
            f".npz at {p!s}"
        )

    @classmethod
    def _load_split(cls, d: Path, *, mmap: bool) -> "HRRStructIndex":
        mmap_mode = "r" if mmap else None
        struct = np.load(d / _STRUCT_FILENAME, mmap_mode=mmap_mode)
        meta = np.load(d / _META_FILENAME, allow_pickle=True)
        idx = cls(dim=int(meta["dim"][0]), seed=int(meta["seed"][0]))
        idx.belief_ids = [str(x) for x in meta["belief_ids"].tolist()]
        idx._index = {bid: i for i, bid in enumerate(idx.belief_ids)}
        idx.struct = struct
        id_names = [str(x) for x in meta["id_names"].tolist()]
        id_matrix = meta["id_matrix"]
        idx.id_vecs = {n: id_matrix[i] for i, n in enumerate(id_names)}
        role_names = [str(x) for x in meta["role_names"].tolist()]
        role_matrix = meta["role_matrix"]
        idx.role_vecs = {n: role_matrix[i] for i, n in enumerate(role_names)}
        return idx

    @classmethod
    def _load_legacy_npz(cls, p: Path) -> "HRRStructIndex":
        global _legacy_deprecation_logged
        if not _legacy_deprecation_logged:
            _logger.warning(
                "HRRStructIndex: loaded legacy bundled .npz at %s; "
                "re-save via HRRStructIndex.save(<dir>) to migrate to "
                "the split-format directory layout (struct.npy + "
                "meta.npz). Legacy bundled .npz support will be removed "
                "in a future release.",
                p,
            )
            _legacy_deprecation_logged = True
        z = np.load(p, allow_pickle=True)
        idx = cls(dim=int(z["dim"][0]), seed=int(z["seed"][0]))
        idx.belief_ids = [str(x) for x in z["belief_ids"].tolist()]
        idx._index = {bid: i for i, bid in enumerate(idx.belief_ids)}
        idx.struct = z["struct"]
        id_names = [str(x) for x in z["id_names"].tolist()]
        id_matrix = z["id_matrix"]
        idx.id_vecs = {n: id_matrix[i] for i, n in enumerate(id_names)}
        role_names = [str(x) for x in z["role_names"].tolist()]
        role_matrix = z["role_matrix"]
        idx.role_vecs = {n: role_matrix[i] for i, n in enumerate(role_names)}
        return idx


_PERSIST_DIRNAME: Final[str] = ".hrr_struct_index"
_ENV_PERSIST: Final[str] = "AELFRICE_HRR_PERSIST"
_EPHEMERAL_PATH_PREFIXES: Final[frozenset[str]] = frozenset(
    {"/tmp/", "/var/tmp/", "/dev/shm/", "/run/"}
)


@dataclass
class HRRStructIndexCache:
    """Lazy, invalidation-aware wrapper around a single ``HRRStructIndex``.

    Subscribes to the store's invalidation callback registry on
    construction, so any belief / edge mutation drops the cached
    index. The next ``get()`` rebuilds.

    Persistence (#691, sub-task of #553): when ``store_path`` is set and
    ``AELFRICE_HRR_PERSIST`` is not ``"0"``, the cache also persists the
    built index to ``<store_dir>/.hrr_struct_index/`` and tries to load
    it via ``mmap_mode='r'`` on subsequent process starts. Invalidation
    drops both the in-memory cache and the on-disk blob so the next
    ``get()`` rebuilds from current store state. In-memory stores
    (``store_path=None``) and the explicit opt-out keep pure in-memory
    behavior.

    Per-instance: two caches pointing at different stores never share
    state. Thread safety is the caller's responsibility.
    """

    store: MemoryStore
    dim: int = DEFAULT_DIM
    store_path: str | None = None
    seed: int | None = None
    persist_enabled: bool | None = None
    _index: HRRStructIndex | None = field(default=None, init=False, repr=False)
    _subscribed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self._subscribed:
            self.store.add_invalidation_callback(self.invalidate)
            self._subscribed = True

    def _resolve_persist_dir(
        self, *, log_on_ephemeral: bool = True,
    ) -> Path | None:
        """Resolve the on-disk persist directory, or ``None`` when disabled.

        ``log_on_ephemeral`` controls whether the one-shot ephemeral-path
        WARNING is fired: when ``False`` the warning is suppressed so
        callers like ``resolve_persist_state`` can probe the path without
        triggering the one-shot log.
        """
        global _ephemeral_disable_logged
        if self.store_path is None:
            return None
        # Highest-precedence rung — env var explicit opt-in/opt-out.
        # Overrides both ephemeral auto-disable and persist_enabled.
        env_raw = os.environ.get(_ENV_PERSIST)
        env_force_on = False
        if env_raw is not None:
            norm = env_raw.strip().lower()
            _FALSY = frozenset({"0", "false", "no", "off"})
            _TRUTHY = frozenset({"1", "true", "yes", "on"})
            if norm in _FALSY:
                return None
            if norm in _TRUTHY:
                env_force_on = True
            # Unrecognised value: fall through to subsequent rungs.
        # Ephemeral-path auto-disable. Resolve symlinks so /tmp →
        # /private/tmp on macOS is caught. Trailing-slash check guards
        # against false matches like /tmpfoo/.
        resolved_parent = Path(self.store_path).resolve(strict=False).parent
        path_with_slash = str(resolved_parent).rstrip("/") + "/"
        is_ephemeral = any(
            path_with_slash.startswith(prefix)
            for prefix in _EPHEMERAL_PATH_PREFIXES
        )
        if is_ephemeral and not env_force_on:
            if log_on_ephemeral and not _ephemeral_disable_logged:
                _logger.warning(
                    "aelfrice: HRR persistence disabled on ephemeral path %s;"
                    " set AELFRICE_HRR_PERSIST=1 to force.",
                    resolved_parent,
                )
                _ephemeral_disable_logged = True
            return None
        # persist_enabled field (set at construction from the config-loader;
        # None means "not specified, use default True"). env=truthy already
        # took the early path above.
        if not env_force_on and self.persist_enabled is False:
            return None
        return Path(self.store_path).parent / _PERSIST_DIRNAME

    def resolve_persist_state(self) -> dict[str, object]:
        """Return a doctor-facing snapshot of HRR persist configuration.

        Read-only — does not mutate cache state or fire any one-shot
        WARNING. Returns::

            {
                "enabled": bool,
                "dir": Path | None,
                "on_disk_bytes": int,
                "reason": str | None,  # None when enabled
            }

        ``reason`` is one of ``None`` (enabled), ``"no store path"``,
        ``"AELFRICE_HRR_PERSIST=0"``, ``"ephemeral path"``.
        """
        if self.store_path is None:
            return {
                "enabled": False,
                "dir": None,
                "on_disk_bytes": 0,
                "reason": "no store path",
            }
        if os.environ.get(_ENV_PERSIST, "1") == "0":
            return {
                "enabled": False,
                "dir": None,
                "on_disk_bytes": 0,
                "reason": "AELFRICE_HRR_PERSIST=0",
            }
        persist_dir = self._resolve_persist_dir(log_on_ephemeral=False)
        if persist_dir is None:
            # Determine whether the ephemeral-path predicate fired
            # (env=truthy/unset, persist_enabled not False, so the only
            # remaining reason _resolve_persist_dir returns None is an
            # ephemeral path).  Use the same resolve+prefix logic.
            resolved_parent = Path(self.store_path).resolve(strict=False).parent
            path_with_slash = str(resolved_parent).rstrip("/") + "/"
            reason = (
                "ephemeral path"
                if any(
                    path_with_slash.startswith(prefix)
                    for prefix in _EPHEMERAL_PATH_PREFIXES
                )
                else None
            )
            return {
                "enabled": False,
                "dir": None,
                "on_disk_bytes": 0,
                "reason": reason,
            }
        return {
            "enabled": True,
            "dir": persist_dir,
            "on_disk_bytes": persist_disk_bytes(persist_dir),
            "reason": None,
        }

    def get(self) -> HRRStructIndex:
        """Return the current index, building or rebuilding as needed."""
        if self._index is not None:
            return self._index
        persist_dir = self._resolve_persist_dir()
        if persist_dir is not None and (persist_dir / _STRUCT_FILENAME).is_file():
            try:
                self._index = HRRStructIndex.load(persist_dir, mmap=True)
                return self._index
            except (FileNotFoundError, OSError, KeyError, ValueError) as e:
                _logger.warning(
                    "HRRStructIndexCache: persist load failed at %s: %s; rebuilding",
                    persist_dir,
                    e,
                )
        idx = HRRStructIndex(dim=self.dim)
        idx.build(self.store, store_path=self.store_path, seed=self.seed)
        if persist_dir is not None:
            try:
                idx.save(persist_dir)
            except OSError as e:
                _logger.warning(
                    "HRRStructIndexCache: persist save failed at %s: %s; continuing in-memory",
                    persist_dir,
                    e,
                )
        self._index = idx
        return self._index

    def invalidate(self) -> None:
        """Drop the cached index. Wired to the store mutation hook."""
        self._index = None
        persist_dir = self._resolve_persist_dir()
        if persist_dir is None or not persist_dir.is_dir():
            return
        for filename in (_STRUCT_FILENAME, _META_FILENAME):
            f = persist_dir / filename
            if f.exists():
                try:
                    f.unlink()
                except OSError as e:
                    _logger.warning(
                        "HRRStructIndexCache: persist unlink failed at %s: %s",
                        f,
                        e,
                    )
        try:
            persist_dir.rmdir()
        except OSError:
            # Best-effort cleanup: the persist dir may still hold foreign
            # files (a co-located store metadata file, an editor swap, a
            # concurrent reader's mmap handle on Linux). Leaving the dir
            # in place is harmless — the next get() will repopulate it.
            pass
