"""Query-strategy dispatcher: legacy-bm25 (default) vs stack-r1-r3.

The rebuilder calls `transform_query(raw_query, store, strategy)` to
produce the final query string fed to `retrieve()`. Two strategies
exist at v1.7 (#291):

* `legacy-bm25` -- the v1.4-era query string is passed through
  unchanged. Default until #291 PR-3 flips after a clean
  #288 phase-1b operator-week.
* `stack-r1-r3` -- the ratified R1+R3 stack: capitalised-token
  entity expansion, then per-store IDF-quantile clipping, against
  the cached `BM25Index` for the store. Returns the rewritten term
  list joined with spaces (FTS5 MATCH consumes the whitespace-
  separated form; duplicated terms boost their effective query
  frequency the same way the lab campaign measured).

This module owns no state; the per-store BM25Index + quantile cache
lives in `query_understanding.store_cache`.
"""
from __future__ import annotations

from typing import Final

from aelfrice.bm25 import tokenize
from aelfrice.query_understanding.entity_expand import (
    expand_with_capitalised_entities,
)
from aelfrice.query_understanding.idf_clip import (
    clip_with_quantile_thresholds,
)
from aelfrice.query_understanding.store_cache import (
    get_bm25_and_quantiles,
)
from aelfrice.store import MemoryStore

LEGACY_STRATEGY: Final[str] = "legacy-bm25"
STACK_R1_R3_STRATEGY: Final[str] = "stack-r1-r3"
VALID_STRATEGIES: Final[frozenset[str]] = frozenset(
    {LEGACY_STRATEGY, STACK_R1_R3_STRATEGY},
)
DEFAULT_STRATEGY: Final[str] = STACK_R1_R3_STRATEGY


def transform_query(
    raw_query: str,
    store: MemoryStore,
    strategy: str = DEFAULT_STRATEGY,
) -> str:
    """Return a possibly-rewritten query string per `strategy`.

    `legacy-bm25`: returns `raw_query` unchanged.

    `stack-r1-r3`: tokenises `raw_query`, applies R1 entity
    expansion, then R3 IDF-clip against the cached per-store BM25
    IDF distribution, and returns the resulting term list joined
    by whitespace. Empty / whitespace-only input returns `""` for
    any strategy.

    Unknown `strategy` raises `ValueError`.
    """
    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"unknown query_strategy: {strategy!r} "
            f"(valid: {sorted(VALID_STRATEGIES)})"
        )
    if not raw_query.strip():
        return ""
    if strategy == LEGACY_STRATEGY:
        return raw_query
    base_terms = tokenize(raw_query)
    expanded = expand_with_capitalised_entities(raw_query, base_terms)
    index, (low_t, high_t) = get_bm25_and_quantiles(store)
    clipped = clip_with_quantile_thresholds(
        expanded, index.vocabulary, index.idf, low_t, high_t,
    )
    return " ".join(clipped)
