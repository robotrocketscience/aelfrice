# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeStubs=false, reportOptionalSubscript=false, reportAttributeAccessIssue=false, reportGeneralTypeIssues=false
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

`BM25Index` is the **default L1 lane** since v1.7.0: the
`use_bm25f_anchors` flag defaults to True per the #154 bench
evidence (see `resolve_use_bm25f_anchors` in `aelfrice.retrieval`,
precedence step 4). The standard FTS5 BM25 path in
`aelfrice.store.search_beliefs` remains available by disabling the
flag — set `AELFRICE_BM25F=0`, pass `use_bm25f_anchors=False`, or
write `[retrieval] use_bm25f_anchors = false` in `.aelfrice.toml`.

The module imports numpy + scipy unconditionally. The v1.5.0
release promotes both to runtime deps; see CHANGELOG and
`pyproject.toml` for the dep-policy break.
"""
from __future__ import annotations

import hashlib
import io
import os
import re
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Final

import numpy as np
import scipy.sparse as sp
import snowballstemmer

from aelfrice.models import Belief
from aelfrice.store import MemoryStore

# v1.7.0 #154: Porter stemmer for FTS5 parity. snowballstemmer's
# "porter" implementation is the original Porter (1980) algorithm,
# which matches SQLite FTS5's `tokenize='porter unicode61'` behavior
# closely enough that q="banana" against content "bananas" hits in
# both lanes. Constructed once at module load (cheap) and shared
# across `tokenize_stemmed()` calls.
_PORTER_STEMMER = snowballstemmer.stemmer("porter")


@lru_cache(maxsize=65_536)
def _stem(token: str) -> str:
    """LRU-memoised Porter stem.

    snowballstemmer's `stemWord` is pure-Python and slow per-call
    (~10-30 µs); at 10k+ beliefs the per-doc tokenisation dominates
    the BM25Index build. Real corpora have small vocabulary
    relative to total tokens (Zipfian), so a 64K-entry LRU has very
    high hit rate after warm-up. Cache is module-global; reset on
    process exit.
    """
    return _PORTER_STEMMER.stemWord(token)

# Default weight for the incoming-anchor token stream, per the #148
# spec. Synthetic-graph evaluation at N=50k under a 15%-vocab-shifted
# regime: rank of vocab-shifted relevant beliefs drops from ~132 to
# ~14 with W=3; clean queries also lift (NDCG@10 ~0.61 -> ~0.77).
DEFAULT_ANCHOR_WEIGHT: Final[int] = 3

# Standard BM25 hyperparameters (Robertson & Walker 1994).
DEFAULT_K1: Final[float] = 1.5
DEFAULT_B: Final[float] = 0.75

# Per-field length-normalisation strength for the anchor stream (#1180),
# used only when `per_field=True`.
#
# Held equal to `DEFAULT_B` deliberately. The per-field split already
# changes the functional form, so the bench that gates the flip has to
# attribute its result to exactly one change; giving the anchor stream a
# different `b` at the same time would confound "fields were separated"
# with "the anchor stream is normalised more/less strongly". 0.75 is also
# the value implied by the worked example in #1180 — it reproduces that
# issue's stated target figures (uncited 0.5714 / cited 0.7835) exactly.
#
# The constraint that matters is `b_anchor > 0`. At `b_anchor = 0` the
# anchor stream pays no length penalty and its contribution grows
# linearly with citation volume; any positive value bounds it, because
# `dl_anchor` grows in step with `tf_anchor` for a constant-density
# stream. Tuning it away from 0.75 is a separate, separately-benched
# decision.
DEFAULT_B_ANCHOR: Final[float] = DEFAULT_B

# Query-term-frequency saturation (Robertson & Walker 1994's `k3`,
# the query-side analogue of `k1`). `score()` weights each query
# term by ``idf * (k3 + 1) * qf / (k3 + qf)``.
#
# `k3 = 0` collapses that factor to exactly 1.0 for every qf >= 1, so
# the default reproduces the pre-#1166 scoring **byte-for-byte** — the
# old code assigned `q_vec[j] = idf[j]` and thereby discarded qf
# entirely. Keeping 0.0 as the default makes the qf mechanism inert
# until an operator flips it, which is deliberate: three shipped
# components encode their boost as term repetition
# (`query_understanding.entity_expand`, `query_understanding.idf_clip`,
# and `hook._build_conversation_aware_query`) and have therefore been
# no-ops on this lane since it became the default. Turning them on is a
# retrieval-quality change whose constants were tuned against the FTS5
# lane, so the flip is bench-gated separately rather than ridden in on
# this bug fix. Set `AELFRICE_BM25_K3` / `[retrieval] bm25_k3` to opt in;
# k3 ~ 8 gives qf=2 -> 1.8x and qf=3 -> 2.45x, close to the linear
# weighting those components assume.
DEFAULT_K3: Final[float] = 0.0

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
# v2 (#1135): k1/b/avgdl widened float32 -> float64 so a deserialised
# index scores byte-identically to a fresh build. No v1 blobs exist in
# the wild — nothing called serialize() before the sidecar cache.
# v3 (#1166): appends `k3` (float64). Like k1/b it is a scoring-time
# parameter carried on the index, so a sidecar written under one k3
# must not be reused under another. v2 blobs DO exist in the wild;
# `_load_sidecar` rejects a version mismatch and rebuilds, so the
# upgrade costs one rebuild per store and never mis-scores.
# v4 (#1180): appends `per_field` (uint8), `b_anchor` + `avgdl_anchor`
# (float64), and the anchor stream's `dl_anchor` / `tf_anchor` arrays.
# The anchor arrays are empty in legacy (`per_field=False`) blobs, so the
# size cost of the bump is 17 bytes there. Same rebuild-on-mismatch
# posture as v3.
# v5 (#1199): appends the per-document source fingerprints that let
# `update_from` re-tokenise only the documents whose indexed text
# actually changed. 16 bytes per belief; a rebuild without them is
# still correct, just not incremental.
_SERIALIZE_VERSION: Final[int] = 5


def tokenize(text: str) -> list[str]:
    """Lowercase + Unicode-word tokenisation. No stemming.

    Returned tokens are the canonical form used by callers that need
    word-form-preserving tokens (e.g.,
    `aelfrice.relationship_detector` matches against unstemmed
    quantifier tokens like ``"always"`` and ``"rarely"``). Two pieces
    of text that differ only in case or punctuation tokenise
    identically. Empty / whitespace-only input returns ``[]``.

    BM25 indexing uses `tokenize_stemmed()` instead — that's where
    Porter stemming lives so `q="banana"` matches content `"bananas"`
    on the BM25F path (FTS5 already stems).
    """
    if not text:
        return []
    return [m.group(0).lower() for m in _TOKEN_PATTERN.finditer(text)]


def tokenize_stemmed(text: str) -> list[str]:
    """Lowercase + Unicode-word tokenisation + Porter stemming.

    Used by `BM25Index.build` and `BM25Index.score` so the BM25F
    lane has FTS5-equivalent stemming. SQLite FTS5 uses Porter by
    default; without stemming on the BM25F path,
    `q="banana"` against content `"bananas"` would miss matches that
    the legacy FTS5 lane catches. Added at v1.7.0 (#154) when the
    default-on flip was prepared.

    Non-BM25 callers (relationship_detector, scoring helpers, etc.)
    that depend on word-form-preserving tokens should keep using
    `tokenize()`; stemming is BM25-specific.
    """
    if not text:
        return []
    return [
        _stem(m.group(0).lower())
        for m in _TOKEN_PATTERN.finditer(text)
    ]


def _source_fingerprint(text: str) -> int:
    """64-bit digest of one document's source text (#1199).

    Fingerprints what was actually tokenised rather than reusing
    `beliefs.content_hash`: that column is written by callers
    (`classification._content_hash`, `derivation._content_hash`) and is
    not enforced by the store, so keying index validity on it would let
    one writer's wrong hash silently serve stale retrieval results.
    This digest owes nothing to a convention the store does not
    guarantee.

    blake2b truncated to 64 bits. A collision would reuse a stale row;
    at a corpus of 1e5 documents the birthday probability is ~3e-10,
    which is well under the rate at which the sidecar is discarded for
    ordinary reasons.
    """
    return int.from_bytes(
        hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(),
        "little",
    )


def _anchor_fingerprint(anchors: Sequence[str]) -> int:
    """64-bit digest of one belief's incoming anchor texts (#1199).

    Order-independent by construction — the anchors are sorted before
    hashing. Anchor order cannot change the index (it feeds per-term
    counts and a total length, both commutative), so hashing in
    iteration order would invalidate documents whose index is provably
    identical. Each anchor is length-prefixed so `["ab", "c"]` and
    `["a", "bc"]` cannot collide.
    """
    h = hashlib.blake2b(digest_size=8)
    for a in sorted(anchors):
        raw = a.encode("utf-8")
        h.update(len(raw).to_bytes(8, "little"))
        h.update(raw)
    return int.from_bytes(h.digest(), "little")


def _build_stream(
    token_lists: list[list[str]],
    vocab: dict[str, int],
    n_docs: int,
    n_terms: int,
) -> tuple[sp.csr_matrix, np.ndarray, list[set[int]]]:
    """Materialise one field's CSR `tf`, its `dl` vector, and the
    per-document set of column indices it contributes.

    The returned term sets let the caller compute `df` over the
    **union** of streams — a term occurring in both a belief's content
    and its anchor text must count once, not twice, or `idf` would be
    depressed for exactly the terms BM25F is meant to reward
    (Robertson, Zaragoza & Taylor 2004 §3).
    """
    rows_idx: list[int] = []
    cols_idx: list[int] = []
    data: list[int] = []
    dl_list: list[int] = []
    per_doc_terms: list[set[int]] = []
    for i, doc_tokens in enumerate(token_lists):
        counts: dict[int, int] = {}
        for t in doc_tokens:
            j = vocab[t]
            counts[j] = counts.get(j, 0) + 1
        dl_list.append(len(doc_tokens))
        for j, c in counts.items():
            rows_idx.append(i)
            cols_idx.append(j)
            data.append(c)
        per_doc_terms.append(set(counts))

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
    return tf, np.asarray(dl_list, dtype=np.float32), per_doc_terms


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
        ``[i, j]`` is the count of vocabulary term ``j`` in belief
        ``i``'s document. Under `per_field` this is the **content
        stream alone**; otherwise it is the augmented document
        (content plus `anchor_weight` replicas of the anchor text).
    dl
        Per-document length vector of shape ``(n_docs,)``. Sum of
        cell values across columns for each row, so it tracks `tf`:
        content-only under `per_field`, combined otherwise.
    avgdl
        Mean of `dl`. Used in BM25 length normalisation.
    per_field
        True when the index carries content and anchor as two
        separately-normalised BM25F fields (#1180) rather than one
        concatenated document. Selects the `score()` branch.
    tf_anchor, dl_anchor, avgdl_anchor
        The anchor stream's counterparts to `tf` / `dl` / `avgdl`.
        `tf_anchor` holds **unreplicated** raw anchor counts —
        `anchor_weight` acts as a field weight at score time rather
        than as a replication factor. All three are None / 0.0 when
        `per_field` is False.
    b_anchor
        Length-normalisation strength for the anchor stream. Unused
        when `per_field` is False.
    content_fp, anchor_fp
        Per-document `uint64` fingerprints of the source text each row
        was built from, aligned with `belief_ids` (#1199). Not used in
        scoring — they exist so `update_from` can tell which documents
        actually changed and re-tokenise only those. None on an index
        built by a caller that did not supply them, which disables the
        incremental path but leaves scoring untouched.
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
    k3
        Query-term-frequency saturation constant (#1166). ``0.0``
        weights every query term by its idf alone, discarding qf —
        the pre-#1166 behaviour, kept as the default.
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
    k3: float = DEFAULT_K3
    per_field: bool = False
    tf_anchor: sp.csr_matrix | None = None
    dl_anchor: np.ndarray | None = None
    avgdl_anchor: float = 0.0
    b_anchor: float = DEFAULT_B_ANCHOR
    content_fp: np.ndarray | None = None
    anchor_fp: np.ndarray | None = None

    # --- Construction -----------------------------------------------------

    @classmethod
    def build(
        cls,
        store: MemoryStore,
        *,
        anchor_weight: int = DEFAULT_ANCHOR_WEIGHT,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
        k3: float = DEFAULT_K3,
        per_field: bool = False,
        b_anchor: float = DEFAULT_B_ANCHOR,
    ) -> BM25Index:
        """Construct a fresh index from `store`.

        Walks every belief in `belief_id` ASC order and every edge
        with non-NULL `anchor_text`.

        **Legacy (`per_field=False`, the default).** The augmented
        document for belief ``b`` is::

            tokens(b.content) + anchor_weight * concat_tokens(incoming_anchors(b))

        This is the single-field stream-replication approximation.
        Setting ``anchor_weight = 0`` reproduces standard BM25 over the
        belief's own content — used by the AC3 W=0 equivalence test.

        **Per-field (`per_field=True`, #1180).** Content and anchor are
        kept as two streams with their own lengths and their own
        `avgdl`, and `anchor_weight` becomes a *field weight* rather
        than a replication count. Replication is not BM25F: because the
        replicas land in the same `dl`, a belief's own content terms are
        length-penalised in proportion to how much text its citers
        wrote about it. Measured on a synthetic two-belief corpus, a
        belief carrying 90 anchor tokens scores its own twice-occurring
        term 1.79x lower than an otherwise identical uncited belief.

        Deterministic in both modes: same store + same parameters
        produces the same `belief_ids`, `vocabulary`, `tf`, `dl`, `idf`
        (and, under `per_field`, the same `tf_anchor` / `dl_anchor`).
        """
        if anchor_weight < 0:
            raise ValueError("anchor_weight must be >= 0")
        if k3 < 0.0:
            raise ValueError("k3 must be >= 0")
        if b_anchor < 0.0:
            raise ValueError("b_anchor must be >= 0")

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
        anchor_tokens_per_doc: list[list[str]] = []
        # #1199: fingerprint each document's source as it is tokenised,
        # so a later `update_from` can identify the unchanged rows
        # without re-tokenising them.
        content_fp_list: list[int] = []
        anchor_fp_list: list[int] = []
        for bid in belief_ids:
            content = contents.get(bid, "")
            doc_tokens = tokenize_stemmed(content)
            doc_anchors = incoming.get(bid, ())
            content_fp_list.append(_source_fingerprint(content))
            anchor_fp_list.append(_anchor_fingerprint(doc_anchors))
            anchor_token_lists = [
                tokenize_stemmed(anchor) for anchor in doc_anchors
            ]
            if per_field:
                # Two streams. Anchor counts stay unreplicated —
                # `anchor_weight` is applied as a field weight at score
                # time, after each stream has paid its own length
                # normalisation.
                flat_anchor = [t for lst in anchor_token_lists for t in lst]
                anchor_tokens_per_doc.append(flat_anchor)
                tokens_per_doc.append(doc_tokens)
                for t in doc_tokens:
                    if t not in vocab:
                        vocab[t] = len(vocab)
                for t in flat_anchor:
                    if t not in vocab:
                        vocab[t] = len(vocab)
            else:
                for anchor_tokens in anchor_token_lists:
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
        tf, dl, content_terms = _build_stream(
            tokens_per_doc, vocab, n_docs, n_terms,
        )
        tf_anchor: sp.csr_matrix | None = None
        dl_anchor: np.ndarray | None = None
        avgdl_anchor: float = 0.0
        df_counts = np.zeros(n_terms, dtype=np.int64)
        if per_field:
            tf_anchor, dl_anchor, anchor_terms = _build_stream(
                anchor_tokens_per_doc, vocab, n_docs, n_terms,
            )
            avgdl_anchor = float(dl_anchor.mean()) if n_docs > 0 else 0.0
            # df over the union of the two streams (see `_build_stream`).
            for c_terms, a_terms in zip(content_terms, anchor_terms):
                for j in c_terms | a_terms:
                    df_counts[j] += 1
        else:
            for c_terms in content_terms:
                for j in c_terms:
                    df_counts[j] += 1

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
            k3=k3,
            per_field=per_field,
            tf_anchor=tf_anchor,
            dl_anchor=dl_anchor,
            avgdl_anchor=avgdl_anchor,
            b_anchor=b_anchor,
            content_fp=np.asarray(content_fp_list, dtype=np.uint64),
            anchor_fp=np.asarray(anchor_fp_list, dtype=np.uint64),
        )

    # --- Scoring ----------------------------------------------------------

    def _row_index(self, mat: sp.csr_matrix) -> np.ndarray:
        """Per-nonzero row index for `mat`, expanded from its indptr."""
        return np.repeat(
            np.arange(mat.shape[0], dtype=np.int64), np.diff(mat.indptr),
        )

    @staticmethod
    def _length_norm(dl: np.ndarray, avgdl: float, b: float) -> np.ndarray:
        """Robertson's ``B_f(d) = (1 - b) + b * dl_f(d) / avgdl_f``.

        Falls back to all-ones on an empty stream (`avgdl == 0`), which
        makes the stream's contribution its raw `tf` — correct, since an
        empty stream has no nonzero cells to scale.
        """
        if avgdl > 0.0:
            return (1.0 - b) + b * (dl / avgdl)
        return np.ones_like(dl)

    def _saturated_per_field(self) -> sp.csr_matrix:
        """Per-field BM25F saturated weights (#1180).

        Robertson, Zaragoza & Taylor (2004), *Simple BM25 Extension to
        Multiple Weighted Fields*::

            B_f(d)   = (1 - b_f) + b_f * dl_f(d) / avgdl_f
            tf~(t,d) = SUM_f  w_f * tf_f(t,d) / B_f(d)
            weight   = (k1 + 1) * tf~ / (k1 + tf~)

        Length normalisation is applied to each stream's raw `tf`
        *before* the streams are summed, and the saturation denominator
        is the plain constant `k1` — not `tf + k1 * B` as in the
        single-field form, where a mixed `B` would let one stream's
        length decide the other's penalty.

        The `(k1 + 1)` numerator is not in the paper's rank-equivalent
        presentation, which drops it as a constant factor. It is kept
        here because it is the term that makes `anchor_weight = 0`
        collapse onto the legacy single-field scorer exactly rather than
        uniformly 1/(k1+1) of it: with one stream of weight 1,
        ``(k1+1)*(tf/B) / (k1 + tf/B)`` is algebraically
        ``tf*(k1+1) / (tf + k1*B)``. Dropping it would leave every score
        2.5x smaller at the shipped `k1 = 1.5` — rank-neutral, but it
        would break the W=0 equivalence guarantee and shift the
        magnitudes that `scoring.partial_bayesian_score` feeds to
        `log()`.
        """
        content_norm = self._length_norm(self.dl, self.avgdl, self.b)
        weighted = self.tf.copy()
        if weighted.nnz:
            weighted.data = (
                weighted.data / content_norm[self._row_index(weighted)]
            ).astype(np.float32)

        if (
            self.tf_anchor is not None
            and self.dl_anchor is not None
            and self.anchor_weight > 0
            and self.tf_anchor.nnz
        ):
            anchor_norm = self._length_norm(
                self.dl_anchor, self.avgdl_anchor, self.b_anchor,
            )
            a = self.tf_anchor.copy()
            a.data = (
                float(self.anchor_weight) * a.data
                / anchor_norm[self._row_index(a)]
            ).astype(np.float32)
            # CSR addition takes the union of the sparsity patterns and
            # returns canonical (sorted-index) output, so the result is
            # a deterministic function of the two inputs.
            weighted = (weighted + a).tocsr()

        if not weighted.nnz:
            return weighted
        wt = weighted.data
        sat = ((self.k1 + 1.0) * wt / (self.k1 + wt)).astype(np.float32)
        return sp.csr_matrix(
            (sat, weighted.indices.copy(), weighted.indptr.copy()),
            shape=weighted.shape,
        )

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

        A repeated query term contributes ``(k3 + 1) * qf / (k3 + qf)``
        times its idf (#1166). At the default ``k3 = 0`` that factor is
        exactly 1.0, so repetition is ignored and the output matches
        the pre-#1166 lane bit-for-bit; raise `k3` to let callers
        express weight as duplicated tokens.
        """
        if not query or not query.strip():
            return []
        if self.tf.shape[0] == 0 or self.tf.shape[1] == 0:
            return []
        q_tokens = tokenize_stemmed(query)
        if not q_tokens:
            return []
        # Build the query weight vector in dense form (n_terms is small
        # enough that this is faster than the alternative sparse
        # construction).
        #
        # #1166: accumulate query-term frequency instead of assigning.
        # The previous `q_vec[j] = self.idf[j]` overwrote on every
        # repeat, so `score("t") == score("t t t")` and any caller that
        # expressed a boost as a duplicated token was silently ignored.
        qf: dict[int, int] = {}
        for t in q_tokens:
            j = self.vocabulary.get(t)
            if j is None:
                continue
            qf[j] = qf.get(j, 0) + 1
        if not qf:
            return []
        q_vec = np.zeros(self.tf.shape[1], dtype=np.float32)
        for j, count in qf.items():
            # Robertson & Walker (1994) query saturation. At k3 = 0
            # this is idf * 1.0 for every count >= 1, i.e. bit-identical
            # to the pre-#1166 assignment; as k3 -> inf it approaches
            # raw idf * qf.
            q_vec[j] = self.idf[j] * (
                (self.k3 + 1.0) * count / (self.k3 + count)
            )

        # BM25 numerator: tf * (k1 + 1)
        # BM25 denominator: tf + k1 * (1 - b + b * dl/avgdl)
        # We compute the sparse scoring in two passes: a TF saturation
        # transform on the nonzero cells, then a sparse matvec with
        # the (idf-weighted) query vector.
        if self.per_field:
            sat = self._saturated_per_field()
        else:
            tf_csr = self.tf
            len_norm = self._length_norm(self.dl, self.avgdl, self.b)
            # tf_sat[i, j] = tf[i, j] * (k1 + 1) / (tf[i, j] + k1 * len_norm[i])
            # Operate on the CSR data array to keep this O(nnz).
            tf_data = tf_csr.data.astype(np.float32, copy=False)
            # Per-cell row index from indptr. expand to nnz length.
            row_idx = self._row_index(tf_csr)
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
        to identical bytes, satisfying AC7. Format (v3)::

            magic              8 bytes  b"AELFBM25"
            version            uint32   _SERIALIZE_VERSION
            anchor_weight      int32
            k1, b, avgdl       float64 x 3
            k3                 float64
            n_docs, n_terms    uint64 x 2
            belief_ids         length-prefixed UTF-8 strings
            vocabulary terms   length-prefixed UTF-8 strings
            dl                 float32 x n_docs
            idf                float32 x n_terms
            tf.indptr          int64 x (n_docs + 1)
            nnz                uint64   len(tf.data)
            tf.indices         int64 x nnz
            tf.data            float32 x nnz

        Vocabulary terms are written in column-index order, which
        matches the sorted-ASC order produced by `build()`.

        v2 (#1135) widened k1/b/avgdl from float32 to float64: `build()`
        keeps them as Python floats, and the sidecar cache requires a
        deserialised index to score byte-identically to a fresh build —
        a float32 round-trip perturbed the low-order bits of every
        score. dl/idf/tf stay float32 (already float32 in the built
        index, so their round-trip is exact).

        v3 (#1166) appends `k3` after avgdl, for the same reason v2
        widened k1/b: it is a scoring-time parameter carried on the
        index, so a blob written under one k3 must not be reused under
        another. Written float64 so the round-trip is exact.

        v4 (#1180) appends the per-field mode flag, `b_anchor` and
        `avgdl_anchor` (float64, same exactness reason), and — only
        when `per_field` is set — the anchor stream's `dl_anchor`,
        `indptr`, `indices` and `data`, laid out exactly as the content
        stream's are. A legacy blob carries `per_field = 0` and stops
        after `avgdl_anchor`, so the bump costs 17 bytes there::

            per_field          uint8
            b_anchor           float64
            avgdl_anchor       float64
            [if per_field]
              dl_anchor        float32 x n_docs
              tf_anchor.indptr int64 x (n_docs + 1)
              nnz_anchor       uint64
              tf_anchor.indices int64 x nnz_anchor
              tf_anchor.data   float32 x nnz_anchor
        """
        buf = io.BytesIO()
        buf.write(_SERIALIZE_MAGIC)
        buf.write(np.uint32(_SERIALIZE_VERSION).tobytes())
        buf.write(np.int32(self.anchor_weight).tobytes())
        buf.write(np.float64(self.k1).tobytes())
        buf.write(np.float64(self.b).tobytes())
        buf.write(np.float64(self.avgdl).tobytes())
        buf.write(np.float64(self.k3).tobytes())

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

        # v4 (#1180): per-field mode + the anchor stream.
        buf.write(np.uint8(1 if self.per_field else 0).tobytes())
        buf.write(np.float64(self.b_anchor).tobytes())
        buf.write(np.float64(self.avgdl_anchor).tobytes())
        if self.per_field:
            if self.tf_anchor is None or self.dl_anchor is None:
                raise ValueError(
                    "per_field index missing tf_anchor / dl_anchor"
                )
            buf.write(
                np.asarray(self.dl_anchor, dtype=np.float32).tobytes()
            )
            a_indptr = np.asarray(self.tf_anchor.indptr, dtype=np.int64)
            a_indices = np.asarray(self.tf_anchor.indices, dtype=np.int64)
            a_data = np.asarray(self.tf_anchor.data, dtype=np.float32)
            buf.write(a_indptr.tobytes())
            buf.write(np.uint64(a_data.size).tobytes())
            buf.write(a_indices.tobytes())
            buf.write(a_data.tobytes())

        # v5 (#1199): per-document source fingerprints. Flagged rather
        # than mandatory because an index constructed directly (tests,
        # and any future caller that assembles one by hand) has none;
        # such a blob still round-trips and still scores identically,
        # it just cannot seed an incremental update.
        have_fp = self.content_fp is not None and self.anchor_fp is not None
        buf.write(np.uint8(1 if have_fp else 0).tobytes())
        if have_fp:
            buf.write(np.asarray(self.content_fp, dtype=np.uint64).tobytes())
            buf.write(np.asarray(self.anchor_fp, dtype=np.uint64).tobytes())

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
        k1 = float(_read(np.dtype(np.float64), 1)[0])
        b = float(_read(np.dtype(np.float64), 1)[0])
        avgdl = float(_read(np.dtype(np.float64), 1)[0])
        k3 = float(_read(np.dtype(np.float64), 1)[0])
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

        empty = n_docs == 0 or n_terms == 0
        if empty:
            tf = sp.csr_matrix(
                (max(n_docs, 0), max(n_terms, 0)), dtype=np.float32,
            )
        else:
            tf = sp.csr_matrix(
                (data_arr, indices, indptr),
                shape=(n_docs, n_terms),
            )

        per_field = bool(_read(np.dtype(np.uint8), 1)[0])
        b_anchor = float(_read(np.dtype(np.float64), 1)[0])
        avgdl_anchor = float(_read(np.dtype(np.float64), 1)[0])
        tf_anchor: sp.csr_matrix | None = None
        dl_anchor: np.ndarray | None = None
        if per_field:
            dl_anchor = np.array(_read(np.dtype(np.float32), n_docs), copy=True)
            a_indptr = np.array(
                _read(np.dtype(np.int64), n_docs + 1), copy=True,
            )
            a_nnz = int(_read(np.dtype(np.uint64), 1)[0])
            a_indices = np.array(_read(np.dtype(np.int64), a_nnz), copy=True)
            a_data = np.array(_read(np.dtype(np.float32), a_nnz), copy=True)
            if empty:
                tf_anchor = sp.csr_matrix(
                    (max(n_docs, 0), max(n_terms, 0)), dtype=np.float32,
                )
            else:
                tf_anchor = sp.csr_matrix(
                    (a_data, a_indices, a_indptr), shape=(n_docs, n_terms),
                )

        # v5 (#1199): per-document source fingerprints, if the writer
        # had them.
        content_fp: np.ndarray | None = None
        anchor_fp: np.ndarray | None = None
        if bool(_read(np.dtype(np.uint8), 1)[0]):
            content_fp = np.array(
                _read(np.dtype(np.uint64), n_docs), copy=True,
            )
            anchor_fp = np.array(
                _read(np.dtype(np.uint64), n_docs), copy=True,
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
            k3=k3,
            per_field=per_field,
            tf_anchor=tf_anchor,
            dl_anchor=dl_anchor,
            avgdl_anchor=avgdl_anchor,
            b_anchor=b_anchor,
            content_fp=content_fp,
            anchor_fp=anchor_fp,
        )


# Sidecar file framing (#1135). The payload after the header is the
# `BM25Index.serialize()` blob, which carries its own magic + version.
_SIDECAR_MAGIC: Final[bytes] = b"AELFB25S"
_SIDECAR_VERSION: Final[int] = 1
_SIDECAR_SUFFIX: Final[str] = ".bm25f"


def sidecar_path_for(store: MemoryStore) -> Path | None:
    """The persistent-index sidecar path for `store`, or None for
    in-memory stores (nothing to persist against)."""
    db_path = store.db_path
    if db_path == ":memory:":
        return None
    return Path(db_path + _SIDECAR_SUFFIX)


@dataclass
class BM25IndexCache:
    """Lazy, invalidation-aware wrapper around a single `BM25Index`.

    Subscribes to the store's invalidation callback registry on
    construction, so any belief / edge mutation drops the cached
    index. The next `get()` rebuilds.

    #1135: for on-disk stores the built index is also persisted to a
    sidecar file (`<db-path>.bm25f`) stamped with the store's durable
    generation counter and scope id. A fresh process (the
    UserPromptSubmit hook is one per prompt) deserialises the sidecar
    instead of re-tokenising + re-stemming the whole corpus — measured
    584 ms build vs low-ms load at 5k beliefs. Staleness is decided by
    the stamp: any belief/edge content mutation bumps the generation
    in the same transaction (see `MemoryStore._commit_mutation`), so a
    matching stamp proves the blob reflects current content. Loads and
    writes are fail-soft — a missing, corrupt, foreign (scope-id
    mismatch), stale, or parameter-mismatched sidecar falls back to a
    build; an unwritable sidecar is skipped silently.

    Per-instance: two caches pointing at different stores never share
    state. Thread safety is the caller's responsibility (matches the
    contract of `aelfrice.retrieval.RetrievalCache`).
    """

    store: MemoryStore
    anchor_weight: int = DEFAULT_ANCHOR_WEIGHT
    k1: float = DEFAULT_K1
    b: float = DEFAULT_B
    k3: float = DEFAULT_K3
    per_field: bool = False
    b_anchor: float = DEFAULT_B_ANCHOR
    _index: BM25Index | None = field(default=None, init=False, repr=False)
    _generation: int | None = field(default=None, init=False, repr=False)
    _subscribed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self._subscribed:
            self.store.add_invalidation_callback(self.invalidate)
            self._subscribed = True

    def get(self) -> BM25Index:
        """Return the current index; load the sidecar or build as needed."""
        if self._index is not None and self._generation is not None:
            # Revalidate against the durable counter: the in-process
            # invalidation callback only covers own-process mutations,
            # so without this a long-running process (MCP server) would
            # never see a sibling process's writes (the default-on
            # ingest hooks). One indexed point-read per get(); the
            # pre-#1135 behavior was a full rebuild per query.
            if self.store.store_generation() != self._generation:
                self._index = None
        if self._index is None:
            self._index = self._load_sidecar()
        if self._index is None:
            # Read the stamp BEFORE building: a mutation that lands
            # during the build makes the stamp stale, so the next
            # reader rebuilds rather than trusting a torn snapshot.
            generation = self.store.store_generation()
            self._index = BM25Index.build(
                self.store,
                anchor_weight=self.anchor_weight,
                k1=self.k1,
                b=self.b,
                k3=self.k3,
                per_field=self.per_field,
                b_anchor=self.b_anchor,
            )
            self._write_sidecar(self._index, generation)
            self._generation = generation
        return self._index

    def invalidate(self) -> None:
        """Drop the cached index. Wired to the store mutation hook.

        The sidecar file is left in place — its generation stamp no
        longer matches after the mutation, so every reader treats it
        as stale; the next `get()` rebuild overwrites it.
        """
        self._index = None
        self._generation = None

    # --- Sidecar persistence (#1135) ----------------------------------

    def _load_sidecar(self) -> BM25Index | None:
        """Deserialise a valid sidecar, or None on any miss/mismatch."""
        path = sidecar_path_for(self.store)
        if path is None:
            return None
        try:
            blob = path.read_bytes()
            header_len = len(_SIDECAR_MAGIC) + 4 + 8 + 4
            if len(blob) < header_len:
                return None
            if blob[: len(_SIDECAR_MAGIC)] != _SIDECAR_MAGIC:
                return None
            off = len(_SIDECAR_MAGIC)
            version = int(np.frombuffer(blob, np.uint32, 1, off)[0])
            if version != _SIDECAR_VERSION:
                return None
            off += 4
            generation = int(np.frombuffer(blob, np.uint64, 1, off)[0])
            off += 8
            scope_len = int(np.frombuffer(blob, np.uint32, 1, off)[0])
            off += 4
            scope = blob[off:off + scope_len].decode("utf-8")
            off += scope_len
            # Scope id catches a swapped-in different DB at the same
            # path; the generation stamp catches every content
            # mutation on this DB.
            if scope != self.store.local_scope_id:
                return None
            if generation != self.store.store_generation():
                return None
            index = BM25Index.deserialize(blob[off:])
            if index.anchor_weight != self.anchor_weight:
                return None
            # v2 stores k1/b as float64, so the round-trip is exact;
            # compare at full precision so a nearly-equal config never
            # reuses another config's sidecar.
            if index.k1 != self.k1:
                return None
            if index.b != self.b:
                return None
            # v3 (#1166) stores k3 as float64 for the same exactness
            # reason; a blob written with qf saturation off must not be
            # served to a cache configured with it on.
            if index.k3 != self.k3:
                return None
            # v4 (#1180): the field split changes the scoring functional
            # form, not just a constant, so a legacy blob must never be
            # served to a per-field cache or vice versa. `b_anchor` only
            # participates in scoring under per_field, so it is compared
            # only there — otherwise flipping an inert knob would force a
            # pointless rebuild.
            if index.per_field != self.per_field:
                return None
            if self.per_field and index.b_anchor != self.b_anchor:
                return None
            self._generation = generation
            return index
        except Exception:  # noqa: BLE001 — any bad sidecar => rebuild
            return None

    def _write_sidecar(self, index: BM25Index, generation: int) -> None:
        """Atomically persist `index` stamped with `generation`.

        Best-effort: any failure (read-only dir, disk full) is traced
        to stderr and swallowed — persistence is an optimisation, not
        a correctness requirement. `os.replace` of a same-directory
        temp file keeps concurrent readers safe: they see either the
        old blob or the new one, never a torn write.
        """
        path = sidecar_path_for(self.store)
        if path is None:
            return
        try:
            scope = self.store.local_scope_id.encode("utf-8")
            buf = io.BytesIO()
            buf.write(_SIDECAR_MAGIC)
            buf.write(np.uint32(_SIDECAR_VERSION).tobytes())
            buf.write(np.uint64(generation).tobytes())
            buf.write(np.uint32(len(scope)).tobytes())
            buf.write(scope)
            buf.write(index.serialize())
            fd, tmp_name = tempfile.mkstemp(
                prefix=path.name + ".", dir=str(path.parent),
            )
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(buf.getvalue())
                os.replace(tmp_name, str(path))
            except BaseException:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise
        except Exception as exc:  # noqa: BLE001 — persistence is optional
            print(
                f"aelfrice bm25: sidecar write failed (non-fatal): {exc}",
                file=sys.stderr,
            )


__all__ = [
    "DEFAULT_ANCHOR_WEIGHT",
    "DEFAULT_K1",
    "DEFAULT_B",
    "DEFAULT_B_ANCHOR",
    "DEFAULT_K3",
    "DEFAULT_TOP_K",
    "BM25Index",
    "BM25IndexCache",
    "tokenize",
]
