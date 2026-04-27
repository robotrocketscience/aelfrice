"""Two-layer retrieval: L0 locked-beliefs auto-load + L1 FTS5 BM25 keyword search.

Token-budgeted output (default 2000 tokens, ~4 chars/token estimate). L0
beliefs are always present in the output above any L1 result and are never
trimmed by the budget — locks are user-asserted ground truth and must
survive retrieval.

NO HRR, NO BFS multi-hop, NO entity-index in v1.0 (pre-commit #6). Those
land in a later release once the retrieval upgrade R&D is validated against
a real corpus.

A `RetrievalCache` wrapper provides bounded LRU memoization over the free
function `retrieve()`. Cache invalidation is wired through the store's
callback registry (wipe-on-write across the six mutators). Use the
free function for single calls; use `RetrievalCache(store)` for agent
loops that re-issue the same query.
"""
from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Final

from aelfrice.models import LOCK_NONE, Belief
from aelfrice.store import MemoryStore

DEFAULT_TOKEN_BUDGET: Final[int] = 2000
_CHARS_PER_TOKEN: Final[float] = 4.0
DEFAULT_L1_LIMIT: Final[int] = 50
DEFAULT_CACHE_CAPACITY: Final[int] = 256

_CANONICALIZE_PUNCT: Final[re.Pattern[str]] = re.compile(r"[^\w\s]")


def canonicalize_query(query: str) -> str:
    """Return a deterministic key for cache lookup.

    Lowercase, replace punctuation with whitespace, split on whitespace,
    sort tokens, rejoin with single spaces. Two queries that differ only
    in word order or punctuation map to the same key — correct for FTS5
    BM25, which is bag-of-words.
    """
    cleaned = _CANONICALIZE_PUNCT.sub(" ", query.lower()).strip()
    tokens = sorted(cleaned.split())
    return " ".join(tokens)


@dataclass(frozen=True)
class RetrievalResult:
    """Wrapper object for retrieve_v2 callers (academic-suite adapters).

    Public v1.0.x retrieve() returns list[Belief] directly. Lab v2.0.0
    adapters expect `result.beliefs` plus auxiliary diagnostics fields
    that aren't yet computed in public — those are placeholders here so
    adapter code that reads them does not crash.
    """

    beliefs: list[Belief]
    hrr_expansions: list[str] = field(default_factory=lambda: [])
    bfs_chains: list[list[str]] = field(default_factory=lambda: [])


def _estimate_tokens(text: str) -> int:
    """Cheap char-based token estimate. Conservative (rounds up)."""
    if not text:
        return 0
    return int((len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN)


def _belief_tokens(b: Belief) -> int:
    return _estimate_tokens(b.content)


def retrieve(
    store: MemoryStore,
    query: str,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    l1_limit: int = DEFAULT_L1_LIMIT,
) -> list[Belief]:
    """Return L0 locked beliefs first, then L1 FTS5 BM25 results.

    Output is token-budgeted: L1 results are trimmed from the tail until the
    estimated total token count is at or below `token_budget`. L0 beliefs
    are never trimmed — if the locked set alone exceeds the budget, the
    full L0 set is still returned and L1 is empty.

    Dedupe: an L1 hit whose id already appears in L0 is dropped.

    Empty query: returns L0 only (FTS5 has nothing to match against).
    """
    locked: list[Belief] = store.list_locked_beliefs()
    locked_ids: set[str] = {b.id for b in locked}

    l1: list[Belief] = []
    if query.strip():
        raw_l1: list[Belief] = store.search_beliefs(query, limit=l1_limit)
        l1 = [b for b in raw_l1 if b.id not in locked_ids]

    # Token accounting. L0 always survives.
    used: int = sum(_belief_tokens(b) for b in locked)
    out: list[Belief] = list(locked)
    for b in l1:
        cost: int = _belief_tokens(b)
        if used + cost > token_budget:
            break
        out.append(b)
        used += cost
    return out


def retrieve_v2(
    store: MemoryStore,
    query: str,
    budget: int = DEFAULT_TOKEN_BUDGET,
    include_locked: bool = True,
    use_hrr: bool = False,  # noqa: ARG001
    use_bfs: bool = False,  # noqa: ARG001
    l1_limit: int = DEFAULT_L1_LIMIT,
) -> RetrievalResult:
    """Lab-compatible retrieval wrapper for academic-suite adapters.

    Wraps the public `retrieve()` in the signature lab v2.0.0 adapters
    expect:

    - `budget` (lab kwarg) maps to `token_budget` (public kwarg).
    - `include_locked=False` filters out lock_level != LOCK_NONE post-retrieval
      (public always returns L0 first; this wrapper drops them on demand).
    - `use_hrr` and `use_bfs` are accepted but no-op at v1.0.x — the HRR
      vocabulary bridge (P4) and BFS multi-hop chaining (P3) have not yet
      ported. Callers can pass them for forward-compat without conditionals.
    - Returns a `RetrievalResult` wrapper so adapters can read
      `result.beliefs` (and stub diagnostics fields).

    Numbers produced through retrieve_v2 at v1.0.x are the L0+FTS5 baseline.
    They will improve once HRR/BFS port; that is the explicit measurement
    path P3+ takes.
    """
    raw: list[Belief] = retrieve(
        store, query, token_budget=budget, l1_limit=l1_limit,
    )
    if include_locked:
        beliefs = raw
    else:
        beliefs = [b for b in raw if b.lock_level == LOCK_NONE]
    return RetrievalResult(beliefs=beliefs)


class RetrievalCache:
    """Bounded LRU cache wrapping `retrieve()` for an attached store.

    Subscribes to the store's invalidation callback registry on
    construction, so any belief or edge mutation wipes the cache.
    Per-instance: two `RetrievalCache` objects pointing at different
    stores never share state.

    Usage:

        cache = RetrievalCache(store)
        beliefs = cache.retrieve("what database do we use?")
        # Later in the same agent loop — second call is a cache hit.
        beliefs = cache.retrieve("what database do we use?")
    """

    def __init__(
        self,
        store: MemoryStore,
        capacity: int = DEFAULT_CACHE_CAPACITY,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._store = store
        self._capacity = capacity
        self._entries: OrderedDict[
            tuple[str, int, int], list[Belief]
        ] = OrderedDict()
        store.add_invalidation_callback(self.invalidate)

    def retrieve(
        self,
        query: str,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        l1_limit: int = DEFAULT_L1_LIMIT,
    ) -> list[Belief]:
        """Cached `retrieve()`. Identical contract to the free function."""
        key = (canonicalize_query(query), token_budget, l1_limit)
        cached = self._entries.get(key)
        if cached is not None:
            self._entries.move_to_end(key)
            return list(cached)
        result = retrieve(
            self._store, query,
            token_budget=token_budget, l1_limit=l1_limit,
        )
        self._entries[key] = list(result)
        if len(self._entries) > self._capacity:
            self._entries.popitem(last=False)
        return result

    def invalidate(self) -> None:
        """Drop every cached entry. Wired to the store's mutation hook."""
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)
