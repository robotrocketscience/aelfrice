"""Incremental BM25F index update (#1199).

AC1 measured that 86.2% of retrieval-running prompts rebuild the index
from scratch, and that 96% of that cost is `BM25Index.build()` —
re-tokenising and re-stemming every document to absorb a handful of new
ones. `update_from` re-tokenises only the documents whose indexed text
changed.

The bar for that path is not "close enough": #1135's invariant is that
retrieval output is a deterministic function of store content, so an
incrementally updated index must be *identical* to a full rebuild, field
for field. `test_incremental_matches_a_full_rebuild_under_random_churn`
is the distinguishing assert for this file — it is the property, and the
targeted tests below exist to localise a failure once it fires.
"""
from __future__ import annotations

import random
from dataclasses import replace

import numpy as np
import pytest

from aelfrice.bm25 import (
    BM25Index,
    _anchor_fingerprint,
    _source_fingerprint,
)
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief, Edge
from aelfrice.store import MemoryStore

WORDS = [
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi",
]


def _mk(bid: str, content: str) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}_{_source_fingerprint(content):016x}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-04-28T00:00:00Z",
        last_retrieved_at=None,
    )


def _text(rng: random.Random, n: int | None = None) -> str:
    """Common words plus, half the time, a token unique to this
    document.

    The rare token is load-bearing for the churn test: with a shared
    vocabulary alone, no deletion can ever orphan a term, and an
    implementation that carried the base vocabulary forward without
    pruning would pass. Verified — that mutation escapes the walk on a
    shared-only corpus and is caught with these.
    """
    n = n if n is not None else rng.randint(3, 25)
    words = [rng.choice(WORDS) for _ in range(n)]
    if rng.random() < 0.5:
        words.append(f"rare{rng.randrange(10**9):09d}")
    return " ".join(words)


def _corpus(n: int = 40, seed: int = 0, anchored: float = 0.4) -> MemoryStore:
    rng = random.Random(seed)
    s = MemoryStore(":memory:")
    ids = [f"b{i:04d}" for i in range(n)]
    for bid in ids:
        s.insert_belief(_mk(bid, _text(rng)))
    seen: set[tuple[str, str, str]] = set()
    for i, bid in enumerate(ids):
        if rng.random() < anchored:
            src = ids[(i + 3) % n]
            key = (src, bid, "cites")
            if src != bid and key not in seen:
                seen.add(key)
                s.insert_edge(Edge(
                    src=src, dst=bid, type="cites", weight=1.0,
                    anchor_text=_text(rng, 6),
                ))
    return s


# --- fingerprints ---------------------------------------------------------


def test_source_fingerprint_separates_different_text() -> None:
    assert _source_fingerprint("alpha") != _source_fingerprint("beta")
    assert _source_fingerprint("alpha") == _source_fingerprint("alpha")


def test_anchor_fingerprint_ignores_anchor_order() -> None:
    """Anchor order cannot change the index — it feeds per-term counts
    and a total length, both commutative — so the fingerprint must not
    treat a reordering as a change, or it would invalidate documents
    whose index is provably identical."""
    assert _anchor_fingerprint(["a b", "c d"]) == _anchor_fingerprint(
        ["c d", "a b"]
    )


def test_anchor_fingerprint_length_prefixes_each_anchor() -> None:
    """`["ab", "c"]` and `["a", "bc"]` are different anchor sets and
    must not collide through naive concatenation."""
    assert _anchor_fingerprint(["ab", "c"]) != _anchor_fingerprint(["a", "bc"])


def test_build_stamps_a_fingerprint_per_belief() -> None:
    store = _corpus(12, seed=1)
    idx = BM25Index.build(store)
    assert idx.content_fp is not None
    assert idx.anchor_fp is not None
    assert len(idx.content_fp) == len(idx.belief_ids)
    assert len(idx.anchor_fp) == len(idx.belief_ids)
    assert idx.content_fp.dtype == np.uint64


def test_fingerprints_survive_the_serialisation_round_trip() -> None:
    store = _corpus(12, seed=2)
    idx = BM25Index.build(store)
    back = BM25Index.deserialize(idx.serialize())
    assert np.array_equal(back.content_fp, idx.content_fp)
    assert np.array_equal(back.anchor_fp, idx.anchor_fp)


def test_a_blob_without_fingerprints_still_round_trips() -> None:
    """An index assembled by hand carries no fingerprints. It must still
    serialise, deserialise and score — it simply cannot seed an
    incremental update."""
    store = _corpus(8, seed=3)
    idx = BM25Index.build(store)
    idx.content_fp = None
    idx.anchor_fp = None
    back = BM25Index.deserialize(idx.serialize())
    assert back.content_fp is None
    assert back.anchor_fp is None
    assert back.score("alpha", top_k=5) == idx.score("alpha", top_k=5)


def test_editing_a_belief_moves_only_its_own_fingerprint() -> None:
    store = _corpus(10, seed=4, anchored=0.0)
    before = BM25Index.build(store)
    target = before.belief_ids[3]
    row = store.get_belief(target)
    assert row is not None
    edited = replace(row, content="totally different pi pi pi")
    store.update_belief(edited)
    after = BM25Index.build(store)
    assert before.belief_ids == after.belief_ids
    moved = [
        i for i in range(len(after.belief_ids))
        if before.content_fp[i] != after.content_fp[i]
    ]
    assert moved == [3]


# --- the property: identical to a full rebuild ---------------------------


def _assert_identical(got: BM25Index, want: BM25Index, ctx: str) -> None:
    """Field-for-field equality, not approximate. #1135's contract is
    that retrieval output is a deterministic function of store content,
    so anything short of identity is a correctness regression."""
    assert got.belief_ids == want.belief_ids, f"{ctx}: belief_ids"
    assert got.vocabulary == want.vocabulary, f"{ctx}: vocabulary"
    assert np.array_equal(got.tf.indptr, want.tf.indptr), f"{ctx}: indptr"
    assert np.array_equal(got.tf.indices, want.tf.indices), f"{ctx}: indices"
    assert np.array_equal(got.tf.data, want.tf.data), f"{ctx}: data"
    assert np.array_equal(got.dl, want.dl), f"{ctx}: dl"
    assert got.avgdl == want.avgdl, f"{ctx}: avgdl"
    assert np.array_equal(got.idf, want.idf), f"{ctx}: idf"
    assert got.tf.shape == want.tf.shape, f"{ctx}: shape"
    if want.per_field:
        assert got.tf_anchor is not None
        assert np.array_equal(
            got.tf_anchor.indptr, want.tf_anchor.indptr,
        ), f"{ctx}: anchor indptr"
        assert np.array_equal(
            got.tf_anchor.indices, want.tf_anchor.indices,
        ), f"{ctx}: anchor indices"
        assert np.array_equal(
            got.tf_anchor.data, want.tf_anchor.data,
        ), f"{ctx}: anchor data"
        assert np.array_equal(got.dl_anchor, want.dl_anchor), f"{ctx}: dl_a"
        assert got.avgdl_anchor == want.avgdl_anchor, f"{ctx}: avgdl_a"


def _mutate(store: MemoryStore, rng: random.Random, live: list[str]) -> str:
    """One random store mutation. Returns a label for failure context."""
    choice = rng.random()
    if choice < 0.30 or not live:
        bid = f"n{rng.randrange(10**6):06d}"
        store.insert_belief(_mk(bid, _text(rng)))
        live.append(bid)
        return f"insert {bid}"
    if choice < 0.50:
        bid = rng.choice(live)
        row = store.get_belief(bid)
        if row is None:
            return "edit-noop"
        store.update_belief(replace(row, content=_text(rng)))
        return f"edit {bid}"
    if choice < 0.62:
        bid = rng.choice(live)
        store.soft_delete_belief(bid)
        live.remove(bid)
        return f"soft-delete {bid}"
    if choice < 0.72:
        bid = rng.choice(live)
        store.delete_belief(bid)
        live.remove(bid)
        return f"hard-delete {bid}"
    if choice < 0.88 and len(live) >= 2:
        src, dst = rng.sample(live, 2)
        try:
            store.insert_edge(Edge(
                src=src, dst=dst, type="cites", weight=1.0,
                anchor_text=_text(rng, 5),
            ))
        except Exception:
            return "edge-dup-noop"
        return f"anchor-edge {src}->{dst}"
    # a mutation that bumps the generation without touching indexed text
    bid = rng.choice(live)
    store.bump_posterior(bid, 1.0, 0.0)
    return f"posterior {bid}"


@pytest.mark.parametrize("per_field", [False, True])
def test_incremental_matches_a_full_rebuild_under_random_churn(
    per_field: bool,
) -> None:
    """Distinguishing assert for this file.

    Walks a store through 60 random mutations — inserts, content edits,
    soft-deletes, hard-deletes, anchor-edge churn, and posterior bumps
    that change no indexed text — and after every one asserts the
    incrementally updated index equals a full rebuild exactly.

    Deletions are the case worth having here: they can orphan a
    vocabulary term, and an implementation that carried the base
    vocabulary forward would keep a phantom column, shifting every
    column index after it and every `idf` entry with it.
    """
    rng = random.Random(20260730 + int(per_field))
    store = _corpus(30, seed=7, anchored=0.5)
    live = list(store.list_beliefs_for_indexing_ids()) if hasattr(
        store, "list_beliefs_for_indexing_ids"
    ) else [bid for bid, _ in store.list_beliefs_for_indexing()]

    idx = BM25Index.build(store, per_field=per_field)
    exercised = 0
    for step in range(60):
        label = _mutate(store, rng, live)
        want = BM25Index.build(store, per_field=per_field)
        got = BM25Index.update_from(idx, store, per_field=per_field)
        if got is None:
            idx = want
            continue
        _assert_identical(got, want, f"step {step} ({label})")
        exercised += 1
        idx = got
    assert len(live) > 5, "corpus collapsed; the walk stopped exercising churn"
    # Without this the test could pass by declining every step and
    # comparing a rebuild against itself. It is currently 60/60.
    assert exercised >= 50, f"incremental path ran only {exercised}/60 times"


def test_incremental_is_identical_after_a_no_op_generation_bump() -> None:
    """A posterior bump invalidates the sidecar without changing a
    single indexed token — the case that should cost nothing."""
    store = _corpus(20, seed=8)
    idx = BM25Index.build(store)
    store.bump_posterior(idx.belief_ids[0], 1.0, 0.0)
    got = BM25Index.update_from(idx, store)
    assert got is not None
    _assert_identical(got, BM25Index.build(store), "no-op bump")


def test_a_deleted_belief_drops_its_orphaned_vocabulary_term() -> None:
    """Explicit rather than left to the random walk: a term that only
    one document carried must leave the vocabulary when that document
    does, or `n_terms` diverges from a fresh build."""
    store = _corpus(10, seed=9, anchored=0.0)
    store.insert_belief(_mk("zzz", "quokka quokka"))
    idx = BM25Index.build(store)
    assert "quokka" in idx.vocabulary
    store.delete_belief("zzz")
    got = BM25Index.update_from(idx, store)
    assert got is not None
    assert "quokka" not in got.vocabulary
    _assert_identical(got, BM25Index.build(store), "orphan drop")


def test_update_declines_without_fingerprints() -> None:
    store = _corpus(10, seed=10)
    idx = BM25Index.build(store)
    idx.content_fp = None
    assert BM25Index.update_from(idx, store) is None


def test_update_declines_on_a_different_anchor_weight() -> None:
    """A base built under a different `anchor_weight` describes
    different documents; reusing its rows would silently mis-score."""
    store = _corpus(10, seed=11)
    idx = BM25Index.build(store, anchor_weight=3)
    assert BM25Index.update_from(idx, store, anchor_weight=1) is None


def test_update_declines_past_the_change_ratio() -> None:
    """Past the threshold a full build is cheaper than deciding what to
    keep, so the incremental path stands aside."""
    store = _corpus(10, seed=12, anchored=0.0)
    idx = BM25Index.build(store)
    for bid in list(idx.belief_ids):
        row = store.get_belief(bid)
        assert row is not None
        store.update_belief(replace(row, content="rewritten " + bid))
    assert BM25Index.update_from(idx, store) is None


# --- the cache actually uses it -----------------------------------------


def test_cache_serves_a_stale_sidecar_incrementally(tmp_path) -> None:
    """The wiring, not just the function.

    After a mutation the sidecar's stamp is stale and the pre-#1199
    cache rebuilt from scratch. This asserts the rebuild does not
    happen: `BM25Index.build` is replaced with a bomb, so the only way
    `get()` can return is through `update_from`. The returned index is
    still compared against a real full build for equality.
    """
    from aelfrice.bm25 import BM25IndexCache

    db = str(tmp_path / "m.db")
    store = MemoryStore(db)
    rng = random.Random(77)
    for i in range(25):
        store.insert_belief(_mk(f"b{i:04d}", _text(rng)))

    BM25IndexCache(store).get()          # writes the sidecar
    store.insert_belief(_mk("bNEW", "quokka alpha beta"))
    want = BM25Index.build(store)

    calls: list[int] = []
    real_build = BM25Index.build

    def bomb(*a, **k):
        calls.append(1)
        raise AssertionError("full rebuild taken; incremental path missed")

    BM25Index.build = staticmethod(bomb)  # type: ignore[method-assign]
    try:
        got = BM25IndexCache(store).get()
    finally:
        BM25Index.build = real_build      # type: ignore[method-assign]
    assert calls == []
    _assert_identical(got, want, "cache incremental")


def test_cache_still_builds_when_there_is_no_sidecar(tmp_path) -> None:
    """Negative control for the test above: with no sidecar to update
    from, `get()` must still build — otherwise the previous test would
    pass on a cache that had simply stopped working."""
    from aelfrice.bm25 import BM25IndexCache

    store = MemoryStore(str(tmp_path / "m2.db"))
    rng = random.Random(78)
    for i in range(10):
        store.insert_belief(_mk(f"b{i:04d}", _text(rng)))
    idx = BM25IndexCache(store).get()
    assert len(idx.belief_ids) == 10
