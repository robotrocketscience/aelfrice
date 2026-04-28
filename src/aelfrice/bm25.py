"""BM25F sparse-matvec retrieval index (v1.5.0, #148).

Augments standard BM25 over each belief's own content with
**incoming-edge anchor text** as a weighted secondary stream
(BM25F per Robertson 2004), and replaces the per-document Python
scoring loop with a single sparse matvec over a precomputed
``(n_docs × n_terms)`` term-frequency matrix.

Two complementary changes:

1. **Quality lever — anchor-text augmentation.** Each belief's
   indexed document is its own tokens plus the concatenation of
   its incoming edges' anchor-text tokens, replicated by a fixed
   weight ``W`` (default ``W = 3``). The stream-replication form
   is BM25F-compatible because BM25's length-normalisation
   absorbs the inflated token count correctly. Vocab-shifted
   beliefs that would be unreachable by BM25 over their own text
   become recoverable through their citers' anchor text.
2. **Latency lever — sparse matvec.** The index materialises
   ``(n_docs × n_terms)`` scipy.sparse CSR `tf`, the per-document
   document length vector `dl`, and the per-term inverse-document-
   frequency vector `idf`. Per-query cost is one sparse matvec
   plus a length-normalisation broadcast — constant overhead per
   query rather than O(n_docs) Python work.

The standard FTS5 BM25 path in `aelfrice.store.search_beliefs`
remains the default L1 lane at v1.5.0. `BM25Index` lands as an
**opt-in alternative** behind the `bm25f_enabled` flag in
`aelfrice.retrieval`. Default-on flip is gated on a benchmark
follow-up after v1.5.x telemetry — see #154 for the composition
tracker that handles that flip.

The module imports numpy + scipy unconditionally. The v1.5.0
release promotes both to runtime deps; see CHANGELOG and
`pyproject.toml` for the dep-policy break.
"""
from __future__ import annotations

import io
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import scipy.sparse as sp

from aelfrice.models import Belief
from aelfrice.store import MemoryStore

# Default weight for the incoming-anchor token stream, per the #148
# spec. Synthetic-graph evaluation at N=50k under a 15%-vocab-shifted
# regime: rank of vocab-shifted relevant beliefs drops from ~132 to
# ~14 with W=3; clean queries also lift (NDCG@10 ~0.61 -> ~0.77).
DEFAULT_ANCHOR_WEIGHT: Final[int] = 3

# Standard BM25 hyperparameters (Robertson & Walker 1994).
DEFAULT_K1: Final[float] = 1.5
DEFAULT_B: Final[float] = 0.75

# Default top-K for the score() return when the caller omits it.
# Mirrors `aelfrice.retrieval.DEFAULT_L1_LIMIT` so a drop-in swap of
# the FTS5 path for the BM25Index path returns the same result count
# by default.
DEFAULT_TOP_K: Final[int] = 50

# Tokenisation regex — `\w+` over Unicode by default in Python 3,
# matching the FTS5 `unicode61` tokenizer's word-character class
# closely enough for the W=0 equivalence guarantee in
# `tests/test_bm25_index.py::test_w0_equivalence_with_fts5`.
_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"\w+", re.UNICODE)

# Serialisation magic + version. Bumped if the on-disk layout
# changes incompatibly. The format is documented in
# `BM25Index.serialize` / `BM25Index.deserialize`.
_SERIALIZE_MAGIC: Final[bytes] = b"AELFBM25"
_SERIALIZE_VERSION: Final[int] = 1


def tokenize(text: str) -> list[str]:
    """Lowercase + Unicode-word tokenisation.

    Returned tokens are the canonical form used by `BM25Index.build`
    and `BM25Index.score`. Two pieces of text that differ only in
    case or punctuation tokenise identically. Empty / whitespace-only
    input returns ``[]``.
    """
    if not text:
        return []
    return [m.group(0).lower() for m in _TOKEN_PATTERN.finditer(text)]


@dataclass
class BM25Index:
    """Precomputed BM25F sparse term-frequency index.

    Attributes
    ----------
    belief_ids
        Row-aligned list of belief ids. ``belief_ids[i]`` is the
        belief whose tf is row ``i`` of `tf` and whose document
        length is ``dl[i]``.
    vocabulary
        Map from token string to column index in `tf`. Stable across
        builds with the same input (sorted insertion order).
    tf
        Sparse CSR matrix of shape ``(n_docs, n_terms)``. Cell
        ``[i, j]`` is the count of vocabulary term ``j`` in the
        augmented document for belief ``i``.
    dl
        Per-document length vector of shape ``(n_docs,)``. Sum of
        cell values across columns for each row.
    avgdl
        Mean of `dl`. Used in BM25 length normalisation.
    idf
        Per-term inverse-document-frequency vector of shape
        ``(n_terms,)``. Computed as
        ``log(1 + (n_docs - df + 0.5) / (df + 0.5))`` (Robertson 2004
        smoothed form).
    anchor_weight
        Replication factor for each belief's incoming-anchor token
        stream. Stored on the index so `score()` does not need it,
        but `serialize()` round-trips it for diagnostics.
    k1, b
        BM25 hyperparameters used at score time.
    """

    belief_ids: list[str]
    vocabulary: dict[str, int]
    tf: sp.csr_matrix
    dl: np.ndarray
    avgdl: float
    idf: np.ndarray
    anchor_weight: int = DEFAULT_ANCHOR_WEIGHT
    k1: float = DEFAULT_K1
    b: float = DEFAULT_B

    # --- Construction -----------------------------------------------------

    @classmethod
    def build(
        cls,
        store: MemoryStore,
        *,
        anchor_weight: int = DEFAULT_ANCHOR_WEIGHT,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> BM25Index:
        """Construct a fresh index from `store`.

        Walks every belief in `belief_id` ASC order and every edge
        with non-NULL `anchor_text`. The augmented document for
        belief ``b`` is::

            tokens(b.content) + anchor_weight * concat_tokens(incoming_anchors(b))

        Length-normalisation in BM25 absorbs the replicated tokens
        correctly (Robertson 2004 § Stream replication). Setting
        ``anchor_weight = 0`` reproduces standard BM25 over the
        belief's own content — used by the AC3 W=0 equivalence test.

        Deterministic: same store + same `anchor_weight` produces
        the same `belief_ids`, `vocabulary`, `tf`, `dl`, `idf`.
        """
        if anchor_weight < 0:
            raise ValueError("anchor_weight must be >= 0")

        rows = store.list_beliefs_for_indexing()
        belief_ids: list[str] = [bid for bid, _ in rows]
        contents: dict[str, str] = {bid: c for bid, c in rows}
        n_docs: int = len(belief_ids)

        # Group anchor text by destination belief for the W replicas.
        incoming: dict[str, list[str]] = {bid: [] for bid in belief_ids}
        if anchor_weight > 0:
            for dst, anchor in store.iter_incoming_anchor_text():
                if dst in incoming:
                    incoming[dst].append(anchor)

        # First pass: tokenise every augmented document, build the
        # vocabulary in deterministic insertion order (sorted ASC at
        # the end, then re-mapped). The tf entries accumulate in a
        # COO-style triple list; we materialise the CSR at the end.
        vocab: dict[str, int] = {}
        tokens_per_doc: list[list[str]] = []
        for bid in belief_ids:
            content = contents.get(bid, "")
            doc_tokens = tokenize(content)
            for anchor in incoming.get(bid, ()):
                anchor_tokens = tokenize(anchor)
                for _ in range(anchor_weight):
                    doc_tokens.extend(anchor_tokens)
            tokens_per_doc.append(doc_tokens)
            for t in doc_tokens:
                if t not in vocab:
                    vocab[t] = len(vocab)

        # Stable column ordering: sort the vocabulary alphabetically
        # so two builds against the same corpus produce identical
        # column indices regardless of insertion order. The remap
        # rewrites `vocab` to reflect the sorted order.
        sorted_terms: list[str] = sorted(vocab)
        vocab = {t: i for i, t in enumerate(sorted_terms)}
        n_terms: int = len(vocab)

        # Second pass: construct CSR via aggregated COO triples. We
        # accumulate per-document term counts in a dict to fold
        # repeats before handing them to scipy.
        rows_idx: list[int] = []
        cols_idx: list[int] = []
        data: list[int] = []
        dl_list: list[int] = []
        df_counts = np.zeros(n_terms, dtype=np.int64)
        for i, doc_tokens in enumerate(tokens_per_doc):
            counts: dict[int, int] = {}
            for t in doc_tokens:
                j = vocab[t]
                counts[j] = counts.get(j, 0) + 1
            dl_list.append(len(doc_tokens))
            for j, c in counts.items():
                rows_idx.append(i)
                cols_idx.append(j)
                data.append(c)
                df_counts[j] += 1

        if n_docs == 0 or n_terms == 0:
            tf = sp.csr_matrix((max(n_docs, 0), max(n_terms, 0)), dtype=np.float32)
        else:
            tf = sp.csr_matrix(
                (
                    np.asarray(data, dtype=np.float32),
                    (np.asarray(rows_idx, dtype=np.int64),
                     np.asarray(cols_idx, dtype=np.int64)),
                ),
                shape=(n_docs, n_terms),
            )
        dl = np.asarray(dl_list, dtype=np.float32)
        avgdl: float = float(dl.mean()) if n_docs > 0 else 0.0
        # Robertson 2004 smoothed idf. Always non-negative on
        # df <= n_docs, so no clamping needed.
        if n_docs > 0:
            idf = np.log(
                1.0 + (n_docs - df_counts + 0.5) / (df_counts + 0.5)
            ).astype(np.float32)
        else:
            idf = np.zeros(0, dtype=np.float32)

        return cls(
            belief_ids=belief_ids,
            vocabulary=vocab,
            tf=tf,
            dl=dl,
            avgdl=avgdl,
            idf=idf,
            anchor_weight=anchor_weight,
            k1=k1,
            b=b,
        )

    # --- Scoring ----------------------------------------------------------

    def score(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> list[tuple[str, float]]:
        """Return the top-K ``(belief_id, score)`` pairs for `query`.

        Empty / whitespace-only query returns ``[]``. Queries with
        no in-vocabulary terms also return ``[]`` (no document will
        score above 0). Ties on score break by `belief_id` ASC for
        deterministic ordering.
        """
        if not query or not query.strip():
            return []
        if self.tf.shape[0] == 0 or self.tf.shape[1] == 0:
            return []
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        # Build the query indicator * idf vector in dense form
        # (n_terms is small enough that this is faster than the
        # alternative sparse construction).
        q_vec = np.zeros(self.tf.shape[1], dtype=np.float32)
        seen_any = False
        for t in q_tokens:
            j = self.vocabulary.get(t)
            if j is None:
                continue
            q_vec[j] = self.idf[j]
            seen_any = True
        if not seen_any:
            return []

        # BM25 numerator: tf * (k1 + 1)
        # BM25 denominator: tf + k1 * (1 - b + b * dl/avgdl)
        # We compute the sparse scoring in two passes: a TF saturation
        # transform on the nonzero cells, then a sparse matvec with
        # the (idf-weighted) query vector.
        tf_csr = self.tf
        if self.avgdl > 0.0:
            len_norm = (1.0 - self.b) + self.b * (self.dl / self.avgdl)
        else:
            len_norm = np.ones_like(self.dl)
        # tf_sat[i, j] = tf[i, j] * (k1 + 1) / (tf[i, j] + k1 * len_norm[i])
        # Operate on the CSR data array to keep this O(nnz).
        tf_data = tf_csr.data.astype(np.float32, copy=False)
        # Per-cell row index from indptr. expand to nnz length.
        row_idx = np.repeat(
            np.arange(tf_csr.shape[0], dtype=np.int64),
            np.diff(tf_csr.indptr),
        )
        per_cell_norm = (self.k1 * len_norm[row_idx]).astype(np.float32)
        sat_data = tf_data * (self.k1 + 1.0) / (tf_data + per_cell_norm)
        sat = sp.csr_matrix(
            (sat_data, tf_csr.indices.copy(), tf_csr.indptr.copy()),
            shape=tf_csr.shape,
        )
        scores = sat @ q_vec  # shape (n_docs,)

        # Top-K with deterministic tie-break on belief_id ASC.
        nonzero_mask = scores > 0.0
        if not nonzero_mask.any():
            return []
        nonzero_idx = np.flatnonzero(nonzero_mask)
        # argsort by (-score, belief_id) — numpy lexsort sorts on
        # the last key as primary. Negate score so descending.
        ids_arr = np.asarray(self.belief_ids, dtype=object)[nonzero_idx]
        score_arr = scores[nonzero_idx]
        order = np.lexsort((ids_arr, -score_arr))
        picked = nonzero_idx[order][:top_k]
        return [
            (self.belief_ids[int(i)], float(scores[int(i)]))
            for i in picked
        ]

    def score_beliefs(
        self,
        store: MemoryStore,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> list[Belief]:
        """Return the top-K Belief objects for `query`, materialised
        from `store`. Convenience wrapper around `score()`.

        Beliefs whose ids no longer exist in the store (deleted
        between index build and query) are silently skipped — the
        caller should rebuild the index after such mutations via
        the invalidation callback.
        """
        out: list[Belief] = []
        for bid, _ in self.score(query, top_k=top_k):
            b = store.get_belief(bid)
            if b is not None:
                out.append(b)
        return out

    # --- Serialisation ----------------------------------------------------

    def serialize(self) -> bytes:
        """Return a deterministic byte representation of the index.

        Same inputs (store contents + same `anchor_weight`) round-trip
        to identical bytes, satisfying AC7. Format::

            magic              8 bytes  b"AELFBM25"
            version            uint32   _SERIALIZE_VERSION
            anchor_weight      int32
            k1, b, avgdl       float32 x 3
            n_docs, n_terms    uint64 x 2
            belief_ids         length-prefixed UTF-8 strings
            vocabulary terms   length-prefixed UTF-8 strings
            dl                 float32 x n_docs
            idf                float32 x n_terms
            tf.indptr          int64 x (n_docs + 1)
            tf.indices         int64 x nnz
            tf.data            float32 x nnz

        Vocabulary terms are written in column-index order, which
        matches the sorted-ASC order produced by `build()`.
        """
        buf = io.BytesIO()
        buf.write(_SERIALIZE_MAGIC)
        buf.write(np.uint32(_SERIALIZE_VERSION).tobytes())
        buf.write(np.int32(self.anchor_weight).tobytes())
        buf.write(np.float32(self.k1).tobytes())
        buf.write(np.float32(self.b).tobytes())
        buf.write(np.float32(self.avgdl).tobytes())

        n_docs = len(self.belief_ids)
        n_terms = len(self.vocabulary)
        buf.write(np.uint64(n_docs).tobytes())
        buf.write(np.uint64(n_terms).tobytes())

        for bid in self.belief_ids:
            data = bid.encode("utf-8")
            buf.write(np.uint32(len(data)).tobytes())
            buf.write(data)

        # Reverse-lookup vocab to write in column-index order.
        terms_by_index: list[str] = ["" for _ in range(n_terms)]
        for term, idx in self.vocabulary.items():
            terms_by_index[idx] = term
        for term in terms_by_index:
            data = term.encode("utf-8")
            buf.write(np.uint32(len(data)).tobytes())
            buf.write(data)

        buf.write(np.asarray(self.dl, dtype=np.float32).tobytes())
        buf.write(np.asarray(self.idf, dtype=np.float32).tobytes())

        # CSR uses int32 for indptr/indices on small matrices. Coerce
        # to int64 for forward-compat with corpora past 2**31 cells.
        indptr = np.asarray(self.tf.indptr, dtype=np.int64)
        indices = np.asarray(self.tf.indices, dtype=np.int64)
        data_arr = np.asarray(self.tf.data, dtype=np.float32)
        buf.write(indptr.tobytes())
        buf.write(np.uint64(data_arr.size).tobytes())
        buf.write(indices.tobytes())
        buf.write(data_arr.tobytes())

        return buf.getvalue()

    @classmethod
    def deserialize(cls, blob: bytes) -> BM25Index:
        """Inverse of `serialize`. Raises `ValueError` on a corrupt
        or version-mismatched payload.
        """
        view = memoryview(blob)
        if len(view) < len(_SERIALIZE_MAGIC) + 4:
            raise ValueError("BM25Index payload too short")
        if bytes(view[: len(_SERIALIZE_MAGIC)]) != _SERIALIZE_MAGIC:
            raise ValueError("BM25Index payload missing magic header")
        offset = len(_SERIALIZE_MAGIC)

        def _read(dtype: np.dtype, count: int) -> np.ndarray:
            nonlocal offset
            n_bytes = dtype.itemsize * count
            arr = np.frombuffer(view, dtype=dtype, count=count, offset=offset)
            offset += n_bytes
            return arr

        version = int(_read(np.dtype(np.uint32), 1)[0])
        if version != _SERIALIZE_VERSION:
            raise ValueError(
                f"BM25Index version mismatch: payload {version}, "
                f"expected {_SERIALIZE_VERSION}"
            )
        anchor_weight = int(_read(np.dtype(np.int32), 1)[0])
        k1 = float(_read(np.dtype(np.float32), 1)[0])
        b = float(_read(np.dtype(np.float32), 1)[0])
        avgdl = float(_read(np.dtype(np.float32), 1)[0])
        n_docs = int(_read(np.dtype(np.uint64), 1)[0])
        n_terms = int(_read(np.dtype(np.uint64), 1)[0])

        def _read_string() -> str:
            nonlocal offset
            length = int(np.frombuffer(
                view, dtype=np.uint32, count=1, offset=offset,
            )[0])
            offset += 4
            data = bytes(view[offset:offset + length])
            offset += length
            return data.decode("utf-8")

        belief_ids: list[str] = [_read_string() for _ in range(n_docs)]
        terms: list[str] = [_read_string() for _ in range(n_terms)]
        vocab: dict[str, int] = {t: i for i, t in enumerate(terms)}

        dl = np.array(_read(np.dtype(np.float32), n_docs), copy=True)
        idf = np.array(_read(np.dtype(np.float32), n_terms), copy=True)
        indptr = np.array(_read(np.dtype(np.int64), n_docs + 1), copy=True)
        nnz = int(_read(np.dtype(np.uint64), 1)[0])
        indices = np.array(_read(np.dtype(np.int64), nnz), copy=True)
        data_arr = np.array(_read(np.dtype(np.float32), nnz), copy=True)

        if n_docs == 0 or n_terms == 0:
            tf = sp.csr_matrix(
                (max(n_docs, 0), max(n_terms, 0)), dtype=np.float32,
            )
        else:
            tf = sp.csr_matrix(
                (data_arr, indices, indptr),
                shape=(n_docs, n_terms),
            )
        return cls(
            belief_ids=belief_ids,
            vocabulary=vocab,
            tf=tf,
            dl=dl,
            avgdl=avgdl,
            idf=idf,
            anchor_weight=anchor_weight,
            k1=k1,
            b=b,
        )


@dataclass
class BM25IndexCache:
    """Lazy, invalidation-aware wrapper around a single `BM25Index`.

    Subscribes to the store's invalidation callback registry on
    construction, so any belief / edge mutation drops the cached
    index. The next `get()` rebuilds.

    Per-instance: two caches pointing at different stores never share
    state. Thread safety is the caller's responsibility (matches the
    contract of `aelfrice.retrieval.RetrievalCache`).
    """

    store: MemoryStore
    anchor_weight: int = DEFAULT_ANCHOR_WEIGHT
    k1: float = DEFAULT_K1
    b: float = DEFAULT_B
    _index: BM25Index | None = field(default=None, init=False, repr=False)
    _subscribed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self._subscribed:
            self.store.add_invalidation_callback(self.invalidate)
            self._subscribed = True

    def get(self) -> BM25Index:
        """Return the current index, building or rebuilding as needed."""
        if self._index is None:
            self._index = BM25Index.build(
                self.store,
                anchor_weight=self.anchor_weight,
                k1=self.k1,
                b=self.b,
            )
        return self._index

    def invalidate(self) -> None:
        """Drop the cached index. Wired to the store mutation hook."""
        self._index = None


def _expected_augmented_length(
    content_tokens: int,
    incoming_anchor_tokens: int,
    anchor_weight: int,
) -> int:
    """Return the expected length of the augmented document.

    Pure helper for AC2 in the test suite. Kept module-public so the
    relationship between the spec invariant and the build path is
    visible at one site.
    """
    return content_tokens + anchor_weight * incoming_anchor_tokens


# Public API kept narrow so the import surface from `bm25` does not
# accidentally wire scoring internals into downstream code.
__all__ = [
    "DEFAULT_ANCHOR_WEIGHT",
    "DEFAULT_K1",
    "DEFAULT_B",
    "DEFAULT_TOP_K",
    "BM25Index",
    "BM25IndexCache",
    "tokenize",
]


# Backwards-compat with `math` import path used internally for tests.
_ = math  # noqa: F841 (kept for future log-base experiments)

# `Iterable` used only as a forward-compat hook for incremental
# updates (planned for #154); kept imported so a follow-up does not
# need to re-add it.
_ = Iterable  # noqa: F841
