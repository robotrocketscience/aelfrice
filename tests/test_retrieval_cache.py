"""Acceptance tests for `RetrievalCache` (docs/lru_query_cache.md).

Eight tests, one per acceptance criterion. All deterministic, in-memory
SQLite, < 100 ms wall clock per test.
"""
from __future__ import annotations

import time

from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_RELATES_TO,
    LOCK_NONE,
    LOCK_USER,
    Belief,
    Edge,
)
from aelfrice.retrieval import (
    DEFAULT_CACHE_CAPACITY,
    RetrievalCache,
    canonicalize_query,
    retrieve,
)
from aelfrice.store import MemoryStore


def _mk(
    bid: str,
    content: str,
    lock_level: str = LOCK_NONE,
    locked_at: str | None = None,
) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=lock_level,
        locked_at=locked_at,
        demotion_pressure=0,
        created_at="2026-04-26T00:00:00Z",
        last_retrieved_at=None,
    )


def _seeded_store() -> MemoryStore:
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("F1", "the kitchen is full of bananas"))
    s.insert_belief(_mk("F2", "the garage is full of tools"))
    s.insert_belief(_mk(
        "L1", "user pinned a fact about cats",
        lock_level=LOCK_USER, locked_at="2026-04-26T01:00:00Z",
    ))
    return s


# --- AC1: cache hit returns identical results to a fresh pipeline run -----

def test_ac1_cache_hit_matches_fresh_pipeline() -> None:
    s = _seeded_store()
    cache = RetrievalCache(s)
    fresh = retrieve(s, "bananas")
    cached_first = cache.retrieve("bananas")  # miss → store
    cached_second = cache.retrieve("bananas")  # hit
    assert [b.id for b in cached_first] == [b.id for b in fresh]
    assert [b.id for b in cached_second] == [b.id for b in fresh]


# --- AC2: cache hit ≤ 50 µs ------------------------------------------------

def test_ac2_cache_hit_under_fifty_microseconds() -> None:
    s = _seeded_store()
    cache = RetrievalCache(s)
    cache.retrieve("bananas")  # warm
    # Best-of-many to dampen scheduler jitter.
    best = float("inf")
    for _ in range(50):
        t0 = time.perf_counter()
        cache.retrieve("bananas")
        elapsed = time.perf_counter() - t0
        if elapsed < best:
            best = elapsed
    assert best <= 50e-6, f"best-of-50 cache hit was {best * 1e6:.1f} us"


# --- AC3: cache miss runs the full pipeline and stores the result --------

def test_ac3_cache_miss_runs_pipeline_and_stores() -> None:
    s = _seeded_store()
    cache = RetrievalCache(s)
    assert len(cache) == 0
    cache.retrieve("bananas")
    assert len(cache) == 1
    cache.retrieve("tools")
    assert len(cache) == 2


# --- AC4: every store mutator invalidates the cache ----------------------

def test_ac4_every_mutator_invalidates_cache() -> None:
    s = _seeded_store()
    cache = RetrievalCache(s)

    # Helper: prime cache, mutate, assert wiped.
    def assert_invalidates(label: str, mutate: callable) -> None:
        cache.retrieve("bananas")
        assert len(cache) >= 1, f"{label}: cache failed to populate"
        mutate()
        assert len(cache) == 0, f"{label}: cache not invalidated"

    new_belief = _mk("F3", "the basement is full of crates")

    assert_invalidates(
        "insert_belief",
        lambda: s.insert_belief(new_belief),
    )

    fetched = s.get_belief("F3")
    assert fetched is not None
    fetched.content = "the basement holds crates and ropes"
    assert_invalidates(
        "update_belief",
        lambda: s.update_belief(fetched),
    )

    assert_invalidates(
        "delete_belief",
        lambda: s.delete_belief("F3"),
    )

    edge = Edge(src="F1", dst="F2", type=EDGE_RELATES_TO, weight=1.0)
    assert_invalidates(
        "insert_edge",
        lambda: s.insert_edge(edge),
    )

    edge2 = Edge(src="F1", dst="F2", type=EDGE_RELATES_TO, weight=0.5)
    assert_invalidates(
        "update_edge",
        lambda: s.update_edge(edge2),
    )

    assert_invalidates(
        "delete_edge",
        lambda: s.delete_edge("F1", "F2", EDGE_RELATES_TO),
    )


# --- AC5: LRU eviction at capacity ---------------------------------------

def test_ac5_lru_eviction_at_capacity() -> None:
    s = _seeded_store()
    cache = RetrievalCache(s, capacity=3)
    cache.retrieve("query_a")
    cache.retrieve("query_b")
    cache.retrieve("query_c")
    assert len(cache) == 3
    # Touch query_a so it becomes most-recently-used; query_b is now LRU.
    cache.retrieve("query_a")
    cache.retrieve("query_d")  # evicts query_b
    assert len(cache) == 3
    # query_b should now miss (re-storing it brings cache to capacity again,
    # which would evict the next LRU — this confirms it was actually gone).
    keys = list(cache._entries.keys())
    canon_b = canonicalize_query("query_b")
    assert all(k[0] != canon_b for k in keys), \
        f"query_b should have been evicted; entries = {keys}"


# --- AC6: punctuation/word-order canonicalization ------------------------

def test_ac6_punctuation_and_word_order_canonicalize() -> None:
    s = _seeded_store()
    cache = RetrievalCache(s)
    cache.retrieve("bananas kitchen")
    assert len(cache) == 1
    # Same tokens, different order + punctuation — must hit existing entry.
    cache.retrieve("KITCHEN, bananas!")
    assert len(cache) == 1, \
        "punctuation + word-order variants should hit the same cache entry"


# --- AC7: distinct token_budget / l1_limit are distinct cache entries ---

def test_ac7_budget_and_limit_are_part_of_key() -> None:
    s = _seeded_store()
    cache = RetrievalCache(s)
    cache.retrieve("bananas", token_budget=2000)
    cache.retrieve("bananas", token_budget=500)
    assert len(cache) == 2, "different token_budget must produce distinct entries"
    cache.retrieve("bananas", token_budget=2000, l1_limit=10)
    assert len(cache) == 3, "different l1_limit must produce distinct entries"


# --- AC8: cache is per-store-instance ------------------------------------

def test_ac8_cache_is_per_instance() -> None:
    s1 = _seeded_store()
    s2 = _seeded_store()
    c1 = RetrievalCache(s1)
    c2 = RetrievalCache(s2)
    c1.retrieve("bananas")
    assert len(c1) == 1
    assert len(c2) == 0, "second cache must not see first cache's entries"
    # Mutating s1 must not affect c2.
    s1.insert_belief(_mk("F9", "extra fact"))
    assert len(c1) == 0, "c1 should have invalidated"
    assert len(c2) == 0, "c2 was already empty; nothing to assert beyond independence"


# --- Sanity: defaults are sensible ---------------------------------------

def test_default_capacity_is_positive() -> None:
    assert DEFAULT_CACHE_CAPACITY >= 1


def test_canonicalize_is_deterministic() -> None:
    assert canonicalize_query("Hello, world!") == "hello world"
    assert canonicalize_query("World hello") == "hello world"
    assert canonicalize_query("  HELLO   WORLD  ") == "hello world"


def test_capacity_must_be_positive() -> None:
    s = _seeded_store()
    try:
        RetrievalCache(s, capacity=0)
    except ValueError:
        pass
    else:
        raise AssertionError("capacity=0 should have raised ValueError")
