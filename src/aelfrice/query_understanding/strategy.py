"""Query-strategy dispatcher: legacy-bm25 (default) vs stack-r1-r3.

The rebuilder calls `transform_query(raw_query, store, strategy)` to
produce the final query string fed to `retrieve()`. Two strategies
exist at v1.7 (#291):

* `legacy-bm25` -- the v1.4-era query string is passed through
  unchanged. The default again, after #1177 removed the recall
  cliff that made the v3.0 flip away from it pay (#1501; see
  `DEFAULT_STRATEGY`).
* `stack-r1-r3` -- the R1+R3 stack: capitalised-token entity
  expansion, then per-store IDF-quantile clipping, against the
  cached `BM25Index` for the store. Returns the rewritten term list
  joined with spaces (FTS5 MATCH consumes the whitespace-separated
  form; duplicated terms boost their effective query frequency the
  same way the lab campaign measured). Selectable, not default.

This module owns no state; the per-store BM25Index + quantile cache
lives in `query_understanding.store_cache`.
"""
from __future__ import annotations

from typing import Final

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

DEFAULT_STRATEGY: Final[str] = LEGACY_STRATEGY
"""Reverted from `stack-r1-r3` (#1501, ratified 2026-08-13).

The stack did not get worse. The baseline got better, and overtook it.
`4db6744d` (#1177) replaced the conjunctive FTS5 MATCH with a
disjunction over the rarest tokens, which is the change R3 existed to
work around. On the 30-row labelled corpus, across that one commit:

    legacy-bm25  0.3006 -> 0.9553
    stack-r1-r3  0.5858 -> 0.8229
    uplift      +0.2851 -> -0.1324

R3 raised recall by deleting a term so the AND-set would widen. With no
AND-set left to widen, the deletion only removes candidates, and it
removes a lot of them. Measured on a 16,454-belief store: of 1,859
terms entering the clip, 1,284 (69.1%) are dropped.

The boost half cannot compensate, because it cannot fire. A term is
boosted only when its IDF is *strictly* above the 75th percentile, and
on that same store the 75th percentile and the maximum IDF are the same
value (9.3029) — the top quartile is one degenerate point. So the stack
reduces to a term-deleter on any store with that shape.

`stack-r1-r3` stays selectable, and both R-modules stay under test. This
reverses the #718 flip on evidence, so it is left reversible the same
way. `tests/bench_gate/test_query_strategy.py` pins the direction: it
goes red if the default is re-flipped or if the disjunctive MATCH
regresses.
"""


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
    from aelfrice.bm25 import tokenize  # noqa: PLC0415  (#1351: hot-path import)

    base_terms = tokenize(raw_query)
    expanded = expand_with_capitalised_entities(raw_query, base_terms)
    index, (low_t, high_t) = get_bm25_and_quantiles(store)
    clipped = clip_with_quantile_thresholds(
        expanded, index.vocabulary, index.idf, low_t, high_t,
    )
    return " ".join(clipped)
