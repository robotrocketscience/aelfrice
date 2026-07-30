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
    n = n if n is not None else rng.randint(3, 25)
    return " ".join(rng.choice(WORDS) for _ in range(n))


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
