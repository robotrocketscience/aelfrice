"""Per-store BM25Index + IDF-quantile cache for the R3 IDF-clip path.

The R3 IDF-clip query rewriter needs a per-store IDF distribution to
derive its low/high cutoffs (#291 ratified scope; synthetic-tuned
constants do not transfer to live stores). Building a `BM25Index`
on every rebuild would be wasteful -- live stores have ~1.6k beliefs
median (lab R5 survey), and the index is deterministic given store
contents.

This module owns a process-wide cache keyed by store identity. The
first call to `get_bm25_and_quantiles(store)` builds a `BM25Index`
from the store, computes the IDF-quantile thresholds, and stores
both. Subsequent calls return the cached tuple.

Invalidation is wired through `MemoryStore.add_invalidation_callback`
(the existing pattern that the retrieval cache + materialised views
use) so any belief / edge mutation drops the stale cache entry. The
cache is a `WeakKeyDictionary`; entries also drop when the store
object is garbage-collected.
"""
from __future__ import annotations

import weakref

from aelfrice.bm25 import BM25Index
from aelfrice.query_understanding.idf_clip import (
    DEFAULT_HIGH_QUANTILE,
    DEFAULT_LOW_QUANTILE,
    compute_idf_quantile_thresholds,
)
from aelfrice.store import MemoryStore

_CacheValue = tuple[BM25Index, tuple[float, float]]
_cache: weakref.WeakKeyDictionary[MemoryStore, _CacheValue] = (
    weakref.WeakKeyDictionary()
)
# Tracks which stores already had an invalidation callback registered,
# so `get_bm25_and_quantiles` does not double-register on every call.
_callback_registered: weakref.WeakSet[MemoryStore] = weakref.WeakSet()


def get_bm25_and_quantiles(
    store: MemoryStore,
    *,
    low_quantile: float = DEFAULT_LOW_QUANTILE,
    high_quantile: float = DEFAULT_HIGH_QUANTILE,
) -> _CacheValue:
    """Return cached `(BM25Index, (low_t, high_t))` for `store`.

    First call: builds a `BM25Index` from `store`, computes IDF
    quantile thresholds, registers a one-time invalidation callback
    on the store, and caches the tuple. Subsequent calls return the
    cached tuple in O(1) until the next store mutation (or until the
    store is garbage-collected).

    Quantile arguments must satisfy
    `0.0 <= low_quantile < high_quantile <= 1.0` (delegated to
    `compute_idf_quantile_thresholds`).
    """
    cached = _cache.get(store)
    if cached is not None:
        return cached
    index = BM25Index.build(store)
    thresholds = compute_idf_quantile_thresholds(
        index.idf, low_quantile, high_quantile,
    )
    value: _CacheValue = (index, thresholds)
    _cache[store] = value
    if store not in _callback_registered:
        store.add_invalidation_callback(lambda s=store: invalidate(s))
        _callback_registered.add(store)
    return value


def invalidate(store: MemoryStore) -> None:
    """Drop the cached entry for `store` if any.

    Called automatically via `MemoryStore`'s invalidation-callback
    chain on belief / edge mutation; can also be called directly
    in tests.
    """
    _cache.pop(store, None)


def _cache_size_for_test() -> int:
    """Test-only: number of live cache entries."""
    return len(_cache)
