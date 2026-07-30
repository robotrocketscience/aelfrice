"""Per-field BM25F (#1180) — content and anchor as two normalised fields.

The shipped lane concatenates each belief's incoming anchor text into
its own document and normalises by the *combined* length. That is the
single-field stream-replication approximation, not BM25F: because the
replicas land in the same `dl`, a belief's own content terms are
length-penalised in proportion to how much text its citers wrote.

`test_legacy_demotes_cited_belief_below_uncited` is the distinguishing
assert for the whole file — it pins the defect on the legacy path, so
these tests cannot pass vacuously if the per-field branch is later
short-circuited or the flag stops reaching `build()`.

Model: Robertson, Zaragoza & Taylor (2004), *Simple BM25 Extension to
Multiple Weighted Fields* (CIKM '04)::

    B_f(d)   = (1 - b_f) + b_f * dl_f(d) / avgdl_f
    tf~(t,d) = SUM_f  w_f * tf_f(t,d) / B_f(d)
    weight   = (k1 + 1) * tf~ / (k1 + tf~)
"""
from __future__ import annotations

import math

import pytest

from aelfrice.bm25 import (
    DEFAULT_B_ANCHOR,
    DEFAULT_K1,
    BM25Index,
    tokenize_stemmed,
)
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief, Edge
from aelfrice.store import MemoryStore

QUERIES = ("zeta", "zeta gamma0", "alpha3 zeta", "beta7", "gamma1 zeta")


def _mk(bid: str, content: str) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-04-28T00:00:00Z",
        last_retrieved_at=None,
    )


def _corpus(
    anchor_len: int = 200,
    zeta_in_anchor: int = 0,
    n_citers: int = 1,
) -> MemoryStore:
    """`b1` and `b2` are identical in every respect that BM25 can see —
    20 content tokens, the query term `zeta` twice — except that `b2`
    carries incoming anchor text and `b1` does not.

    Any score difference between them is therefore attributable to the
    anchor stream alone.
    """
    s = MemoryStore(":memory:")
    s.insert_belief(
        _mk("b1", "zeta zeta " + " ".join(f"alpha{i}" for i in range(18)))
    )
    s.insert_belief(
        _mk("b2", "zeta zeta " + " ".join(f"beta{i}" for i in range(18)))
    )
    words = ["zeta"] * zeta_in_anchor + [
        f"gamma{i % 40}" for i in range(max(0, anchor_len - zeta_in_anchor))
    ]
    for c in range(n_citers):
        s.insert_belief(_mk(f"citer{c}", f"citation source number {c}"))
        s.insert_edge(Edge(
            src=f"citer{c}", dst="b2", type="cites", weight=1.0,
            anchor_text=" ".join(words),
        ))
    return s


def _ratio(idx: BM25Index, query: str = "zeta") -> float:
    """Cited `b2`'s score over uncited `b1`'s. 1.0 means the anchor
    stream had no effect; below 1.0 means being cited hurt."""
    scored = dict(idx.score(query, top_k=50))
    return scored["b2"] / scored["b1"]


# --- the defect, pinned on the legacy path -------------------------------


def test_legacy_demotes_cited_belief_below_uncited() -> None:
    """Distinguishing assert: on the shipped single-field path, a belief
    whose citers wrote 200 tokens about something else scores well under
    half of an otherwise identical uncited belief.

    Without this, every per-field assertion below could pass on a build
    that silently ignored `per_field`.
    """
    idx = BM25Index.build(_corpus(zeta_in_anchor=0), anchor_weight=3)
    assert _ratio(idx) < 0.5


def test_per_field_does_not_demote_on_irrelevant_anchor_text() -> None:
    """The same corpus under per-field: anchor text that never mentions
    the query term contributes nothing, so the cited belief scores
    exactly what the uncited one does."""
    idx = BM25Index.build(
        _corpus(zeta_in_anchor=0), anchor_weight=3, per_field=True,
    )
    assert _ratio(idx) == pytest.approx(1.0, abs=1e-6)


def test_per_field_rewards_anchor_evidence_monotonically() -> None:
    """Boost rises with the number of query-term occurrences in the
    anchor stream, and never dips below the no-evidence baseline."""
    ratios = [
        _ratio(BM25Index.build(
            _corpus(zeta_in_anchor=z), anchor_weight=3, per_field=True,
        ))
        for z in (0, 1, 5, 20, 100)
    ]
    assert ratios == sorted(ratios)
    assert ratios[0] == pytest.approx(1.0, abs=1e-6)
    assert ratios[-1] > 1.5


# --- AC: w_anchor = 0 recovers standard BM25 over content alone ----------


def test_per_field_at_w0_is_identical_to_legacy_at_w0() -> None:
    """With the anchor field weighted 0, per-field must reproduce
    standard BM25 over content — scores, not merely ordering.

    This holds only because `_saturated_per_field` keeps the `(k1 + 1)`
    numerator. Robertson's rank-equivalent presentation drops it as a
    constant factor; dropping it here would leave every score a factor
    of `1 / (k1 + 1)` below the legacy lane.
    """
    store = _corpus(zeta_in_anchor=5)
    legacy = BM25Index.build(store, anchor_weight=0)
    per_field = BM25Index.build(store, anchor_weight=0, per_field=True)
    for q in QUERIES:
        assert legacy.score(q, top_k=50) == per_field.score(q, top_k=50)


def test_per_field_at_w0_does_not_read_anchor_text_at_all() -> None:
    """At w_anchor = 0 the anchor stream is never tokenised, even on a
    corpus dense with anchor text.

    This is load-bearing rather than an optimisation: anchor-only terms
    would otherwise enter the shared vocabulary and raise `df` under the
    union rule, moving `idf` for every term and breaking the exact
    equality asserted above. `zzqqx` appears only in anchor text, so its
    absence from the vocabulary is the proof.
    """
    store = _corpus(zeta_in_anchor=100)
    store.insert_belief(_mk("c9", "another citing belief"))
    store.insert_edge(Edge(
        src="c9", dst="b2", type="cites", weight=1.0, anchor_text="zzqqx",
    ))
    idx = BM25Index.build(store, anchor_weight=0, per_field=True)
    assert tokenize_stemmed("zzqqx")[0] not in idx.vocabulary
    assert idx.tf_anchor is not None
    assert idx.tf_anchor.nnz == 0
    assert _ratio(idx) == pytest.approx(1.0, abs=1e-6)
    # ...and it does enter once the field is actually weighted.
    on = BM25Index.build(store, anchor_weight=3, per_field=True)
    assert tokenize_stemmed("zzqqx")[0] in on.vocabulary


# --- boundedness ---------------------------------------------------------


def test_per_field_converges_at_constant_anchor_density() -> None:
    """A longer anchor stream at the same term density earns the same
    boost: `dl_anchor` grows in step with `tf_anchor`, so the field's
    normalised contribution is scale-free in citation volume."""
    ratios = [
        _ratio(BM25Index.build(
            _corpus(anchor_len=n, zeta_in_anchor=max(1, n // 10)),
            anchor_weight=3, per_field=True,
        ))
        for n in (40, 160, 640, 2560)
    ]
    assert ratios == sorted(ratios)
    # Successive deltas shrink — the sequence converges rather than
    # growing without bound.
    deltas = [b - a for a, b in zip(ratios, ratios[1:])]
    assert deltas == sorted(deltas, reverse=True)
    assert deltas[-1] < 0.05


def test_per_field_respects_the_saturation_ceiling() -> None:
    """No term can contribute more than `idf * (k1 + 1)`, however dense
    the anchor stream. The outer saturation is applied *after* the
    fields are combined, so no field can escape it."""
    idx = BM25Index.build(
        _corpus(anchor_len=2000, zeta_in_anchor=2000),
        anchor_weight=3, per_field=True,
    )
    col = idx.vocabulary[tokenize_stemmed("zeta")[0]]
    ceiling = float(idx.idf[col]) * (DEFAULT_K1 + 1.0)
    assert dict(idx.score("zeta", top_k=50))["b2"] < ceiling


def test_b_anchor_zero_removes_the_anchor_length_penalty() -> None:
    """`b_anchor = 0` is a permitted ablation setting that disables
    length normalisation on the anchor stream, so the contribution
    tracks raw citation volume instead of density. Pinned so the
    resolver's warning about it stays true."""
    def corpus(b2_anchor_len: int) -> MemoryStore:
        """`b2` carries the same 4 occurrences of the query term either
        tightly or diluted 10x by surrounding anchor prose. Several
        other beliefs are anchored too, so `avgdl_anchor` is set by the
        population rather than by `b2` alone — with a single anchored
        document `dl_anchor / avgdl_anchor` is pinned at `n_docs` and no
        choice of `b_anchor` can express anything.
        """
        s = _corpus(anchor_len=b2_anchor_len, zeta_in_anchor=4)
        for k in range(6):
            s.insert_belief(_mk(f"filler{k}", f"filler belief number {k}"))
            s.insert_belief(_mk(f"fcite{k}", f"filler citer number {k}"))
            s.insert_edge(Edge(
                src=f"fcite{k}", dst=f"filler{k}", type="cites", weight=1.0,
                anchor_text=" ".join(f"delta{i}" for i in range(120)),
            ))
        return s

    kw = {"anchor_weight": 3, "per_field": True}
    normalised = [
        _ratio(BM25Index.build(corpus(n), b_anchor=DEFAULT_B_ANCHOR, **kw))
        for n in (40, 400)
    ]
    unnormalised = [
        _ratio(BM25Index.build(corpus(n), b_anchor=0.0, **kw))
        for n in (40, 400)
    ]
    # Normalised: the diluted stream paid a length penalty for prose
    # carrying no query evidence, so it scores strictly lower.
    assert normalised[1] < normalised[0]
    # Unnormalised: only raw `tf_anchor` is read, so dilution is free.
    assert unnormalised[1] == pytest.approx(unnormalised[0], rel=1e-6)


# --- idf over the union of streams ---------------------------------------


def test_df_counts_a_term_once_across_both_streams() -> None:
    """A term occurring in both a belief's content and its anchor text
    must raise `df` by one, not two — otherwise `idf` is depressed for
    exactly the terms the anchor field exists to reward."""
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("x", "omega unique1"))
    s.insert_belief(_mk("y", "unrelated words here"))
    s.insert_belief(_mk("c", "citing belief"))
    s.insert_edge(Edge(
        src="c", dst="x", type="cites", weight=1.0, anchor_text="omega",
    ))
    idx = BM25Index.build(s, anchor_weight=3, per_field=True)
    col = idx.vocabulary[tokenize_stemmed("omega")[0]]
    n_docs = len(idx.belief_ids)
    expected = math.log(1.0 + (n_docs - 1 + 0.5) / (1 + 0.5))
    assert float(idx.idf[col]) == pytest.approx(expected, abs=1e-5)


# --- construction contract -----------------------------------------------


def test_per_field_defaults_off() -> None:
    """`build()` stays on the legacy single-field path unless asked."""
    idx = BM25Index.build(_corpus(), anchor_weight=3)
    assert idx.per_field is False
    assert idx.tf_anchor is None
    assert idx.dl_anchor is None


def test_legacy_dl_includes_replicas_per_field_dl_does_not() -> None:
    """`dl` tracks whatever `tf` holds: the augmented document on the
    legacy path, the content stream alone under per-field."""
    store = _corpus(anchor_len=30, zeta_in_anchor=2)
    legacy = BM25Index.build(store, anchor_weight=3)
    per_field = BM25Index.build(store, anchor_weight=3, per_field=True)
    row = legacy.belief_ids.index("b2")
    assert legacy.dl[row] == 20 + 3 * 30
    assert per_field.dl[per_field.belief_ids.index("b2")] == 20
    assert per_field.dl_anchor is not None
    assert per_field.dl_anchor[per_field.belief_ids.index("b2")] == 30


def test_negative_b_anchor_rejected() -> None:
    with pytest.raises(ValueError, match="b_anchor must be >= 0"):
        BM25Index.build(_corpus(), anchor_weight=3, b_anchor=-0.1)


# --- serialisation (v4) --------------------------------------------------


@pytest.mark.parametrize("per_field", [False, True])
def test_serialize_roundtrip(per_field: bool) -> None:
    """Both modes round-trip to identical bytes and identical scores."""
    idx = BM25Index.build(
        _corpus(zeta_in_anchor=5, n_citers=3),
        anchor_weight=3, per_field=per_field,
    )
    blob = idx.serialize()
    back = BM25Index.deserialize(blob)
    assert back.serialize() == blob
    assert back.per_field is per_field
    assert back.b_anchor == idx.b_anchor
    assert back.avgdl_anchor == idx.avgdl_anchor
    for q in QUERIES:
        assert back.score(q, top_k=50) == idx.score(q, top_k=50)


def test_serialize_is_deterministic_across_builds() -> None:
    store = _corpus(zeta_in_anchor=5, n_citers=2)
    kw = {"anchor_weight": 3, "per_field": True}
    assert (
        BM25Index.build(store, **kw).serialize()
        == BM25Index.build(store, **kw).serialize()
    )


def test_legacy_blob_is_barely_larger_than_v3() -> None:
    """A legacy index pays only the fixed v4 header for the bump — the
    anchor arrays are omitted entirely rather than written empty."""
    idx = BM25Index.build(_corpus(n_citers=3), anchor_weight=3)
    per_field = BM25Index.build(
        _corpus(n_citers=3), anchor_weight=3, per_field=True,
    )
    assert len(per_field.serialize()) > len(idx.serialize())


def test_deserialize_rejects_older_version() -> None:
    """A v3 sidecar must be refused rather than misread — the trailing
    per-field block would otherwise be absent."""
    import numpy as np

    blob = bytearray(
        BM25Index.build(_corpus(), anchor_weight=3).serialize()
    )
    blob[8:12] = np.uint32(3).tobytes()
    with pytest.raises(ValueError, match="version mismatch"):
        BM25Index.deserialize(bytes(blob))
