# LRU query cache for `aelfrice.retrieval.retrieve()`

**Status:** spec.
**Target milestone:** v1.0.1.
**Dependencies:** stdlib only (`collections.OrderedDict`).
**Risk:** low. Pure addition to `aelfrice/retrieval.py`. No schema
change. No `models.py` change.

## Summary

Add a bounded-size LRU cache in front of `aelfrice.retrieval.retrieve()`
keyed on a canonicalized form of `(query, token_budget, l1_limit)`. On
hit, return the cached result list directly without re-running L0
listing or L1 FTS5 search. On miss, run the full retrieval pipeline and
store the result.

Memoization on canonicalized queries is a standard optimization for
agent workflows where the same logical question is issued repeatedly
across a session.

## Motivation

Agent loops re-issue identical or near-identical queries (re-checking
"what's the database?", retrying after partial failures, exploring the
same neighborhood from different code paths). The current `retrieve()`
runs `list_locked_beliefs()` plus an FTS5 BM25 query on every call —
fast at the scales aelfrice targets, but not free. A cached lookup
shaves the entire pipeline cost on repeat queries.

This is the only sub-millisecond latency band available to the
retrieval surface — every other optimization stays above the FTS5
floor.

The same cache becomes more valuable as later releases extend
`retrieve()` (v1.3.0 partial Bayesian-weighted ranking; v2.0.0 full
feedback-into-retrieval loop): the same hit short-circuits each
addition.

## Design

### Cache key

```python
def canonicalize(query: str) -> str:
    """Lowercase, strip punctuation, sort tokens, rejoin.

    Two queries that differ only in word order or punctuation hit
    the same cache entry. This is correct for FTS5 BM25, which is
    bag-of-words.
    """
```

The full cache key is a tuple `(canonicalized_query, token_budget,
l1_limit)` so a `token_budget=2000` call does not return cached
`token_budget=500` results.

### Cache structure

`OrderedDict[tuple[str, int, int], list[Belief]]` with `move_to_end`
on access. Default capacity: 256 entries.

### Implementation choice

Two options:

- **A (cache on store).** `MemoryStore` owns an internal cache plus a
  `cached_retrieve()` method. Single source of truth for invalidation.
  Cons: store now holds retrieval state.
- **B (cache class subscribes to store).** New `RetrievalCache` in
  `retrieval.py` wraps a store reference and registers an invalidation
  callback. Keeps store as pure storage. Cons: callback plumbing.

This release ships **Option B**. Adds a tiny callback registry to
`MemoryStore` (`add_invalidation_callback(fn)`) and a
`RetrievalCache(store)` class that subscribes its `invalidate()`
method on construction. Free-function `retrieve()` keeps working
unchanged for callers that don't want caching.

### Invalidation

Wipe the entire cache on any state change that could alter retrieval
results. Cheapest correct policy.

Mutators that fire invalidation: `insert_belief`, `update_belief`,
`delete_belief`, `insert_edge`, `update_edge`, `delete_edge`. That
covers lock-state changes (locks flow through `update_belief`) and
posterior updates from `apply_feedback` (also through
`update_belief`).

A finer-grained policy (only invalidate cache entries whose result set
contains the modified belief) is a later optimization. v1.0.1 ships
the wipe-on-write version.

### Forward compatibility with `retrieve_v2`

`retrieve_v2` is a wrapper for academic-suite adapters. It calls
`retrieve()` internally, so it inherits caching once `retrieve()` is
cached — no separate code path needed. The `use_hrr` and `use_bfs`
flags are no-ops at v1.0.x and don't enter the cache key.

## Acceptance criteria

1. Cache hit returns identical result list to a fresh pipeline run for
   the same canonicalized query and key tuple.
2. Cache hit costs ≤ 50 µs on commodity hardware (warm interpreter,
   N=256 cache entries). Verified by a deterministic micro-benchmark.
3. Cache miss runs the full pipeline and stores the result.
4. Each of the six store mutators above invalidates the cache.
   Verified by inserting beliefs → caching a query → mutating store →
   reissuing query → asserting the second call returns updated results,
   not stale cache.
5. Cache eviction is LRU: the least-recently-accessed entry is dropped
   when capacity is exceeded.
6. Two queries that differ only in word order or punctuation hit the
   same cache entry.
7. Two queries that differ in `token_budget` or `l1_limit` use distinct
   cache entries.
8. The cache is per-store-instance. Multiple `MemoryStore` instances
   do not share state.

## Test plan

`tests/test_retrieval_cache.py` covers all 8 acceptance criteria. All
tests deterministic (fixed seeds, in-memory `:memory:` store). Wall
clock per test < 100 ms.

## Out of scope

- Cross-session cache persistence. Cache is in-memory only.
- Distributed cache invalidation. Single-process only.
- Per-belief invalidation (wipe-on-write is sufficient at this
  milestone).
- Cache statistics / hit-rate metrics endpoint. Add when there is a
  consumer that needs them.
- Configurable canonicalization strategy. Single canonicalizer is
  correct for FTS5 BM25; revisit if a non-bag-of-words ranker lands.
