"""Query understanding stack for the rebuilder (R1 + R3, #291).

Three deterministic transforms a caller stacks before BM25:

1. R1 entity expansion (`expand_with_capitalised_entities`)
2. R3 IDF clip with per-store quantile thresholds
   (`clip_with_quantile_thresholds`, `compute_idf_quantile_thresholds`)
3. PR-2 wiring: `transform_query(raw_query, store, strategy)`
   dispatches between `legacy-bm25` (default) and `stack-r1-r3`.
   The per-store BM25Index + quantile cache lives in
   `store_cache.get_bm25_and_quantiles`.

PR-1 landed the rewriters + per-store quantile helper + unit tests.
PR-2 lands the dispatcher + cache + rebuilder/hook plumbing behind
the `query_strategy = "legacy-bm25"` setting (default unchanged).
The default flip to `stack-r1-r3` lands in PR-3 after a clean
#288 phase-1b operator-week.

The synthetic-tuned IDF constants (1.5, 2.5) from the lab R3.5
campaign do not transfer to live stores (live IDF medians 7.5-9.5
versus synthetic 2.85). Per-store quantiles are the architectural
fix and are computed by `compute_idf_quantile_thresholds` over the
live BM25 IDF distribution.
"""
from aelfrice.query_understanding.entity_expand import (
    DEFAULT_QF_MULTIPLIER,
    expand_with_capitalised_entities,
)
from aelfrice.query_understanding.idf_clip import (
    DEFAULT_BOOST_QF,
    DEFAULT_HIGH_QUANTILE,
    DEFAULT_LOW_QUANTILE,
    clip_with_quantile_thresholds,
    compute_idf_quantile_thresholds,
)
from aelfrice.query_understanding.store_cache import (
    get_bm25_and_quantiles,
    invalidate,
)
from aelfrice.query_understanding.strategy import (
    DEFAULT_STRATEGY,
    LEGACY_STRATEGY,
    STACK_R1_R3_STRATEGY,
    VALID_STRATEGIES,
    transform_query,
)

__all__ = [
    "DEFAULT_BOOST_QF",
    "DEFAULT_HIGH_QUANTILE",
    "DEFAULT_LOW_QUANTILE",
    "DEFAULT_QF_MULTIPLIER",
    "DEFAULT_STRATEGY",
    "LEGACY_STRATEGY",
    "STACK_R1_R3_STRATEGY",
    "VALID_STRATEGIES",
    "clip_with_quantile_thresholds",
    "compute_idf_quantile_thresholds",
    "expand_with_capitalised_entities",
    "get_bm25_and_quantiles",
    "invalidate",
    "transform_query",
]
