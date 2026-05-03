"""R3 IDF-clip query rewriter with per-store quantile thresholds (#291).

Two deterministic transforms over a BM25 term list:

1. `compute_idf_quantile_thresholds(idf, low_q, high_q)` -- derive
   per-store low/high IDF cutoffs from the live IDF distribution.
   The synthetic-tuned (1.5, 2.5) constants from the lab R3.5 audit
   do **not** transfer; live store IDF medians sit at 7.5-9.5
   versus the synthetic 2.85, so absolute constants are inert at
   live scale (R5 prereq survey, #291). Per-store quantiles are
   the architectural fix.

2. `clip_with_quantile_thresholds(terms, vocabulary, idf, low,
   high, *, boost_qf)` -- drop terms whose IDF is strictly below
   `low_threshold`; duplicate (`boost_qf` total copies) terms whose
   IDF is strictly above `high_threshold`. Terms absent from
   `vocabulary` (i.e. unseen by BM25) pass through unchanged --
   they will simply not score, but downstream callers may want
   their slot preserved.

The two transforms are factored apart so the per-store quantile
computation can be cached on `BeliefStore` (PR-1 follow-up: PR-2
wires the cache surface) independent of any individual query.
"""
from __future__ import annotations

from typing import Final

import numpy as np

# 25th / 75th percentile of the live IDF distribution as the default
# low / high cutoffs. These are deliberately liberal -- the lab R5
# survey showed live IDF distributions are heavy-tailed enough that
# tighter quantiles (e.g. 0.1 / 0.9) would drop too aggressively.
# The values are runtime-tunable; the defaults are the ratified
# starting point pending real-corpus calibration via #288 phase-1b.
DEFAULT_LOW_QUANTILE: Final[float] = 0.25
DEFAULT_HIGH_QUANTILE: Final[float] = 0.75

# 2x to mirror the R1 boost convention (integer-qf BM25 cannot encode
# fractional weights; "boost N" means "emit N copies").
DEFAULT_BOOST_QF: Final[int] = 2


def compute_idf_quantile_thresholds(
    idf: np.ndarray,
    low_quantile: float = DEFAULT_LOW_QUANTILE,
    high_quantile: float = DEFAULT_HIGH_QUANTILE,
) -> tuple[float, float]:
    """Return (low_threshold, high_threshold) IDF cutoffs.

    `idf` is the per-vocabulary-term IDF vector from a `BM25Index`.
    Quantiles are computed on that vector directly. An empty
    vector returns `(0.0, 0.0)` as a safe noop -- callers will then
    drop nothing and boost nothing.

    Both quantile arguments must satisfy
    `0.0 <= low_quantile < high_quantile <= 1.0`. `low == high` is
    rejected because it produces a degenerate clip (everything in
    "mid band", nothing dropped or boosted).
    """
    if not (0.0 <= low_quantile < high_quantile <= 1.0):
        raise ValueError(
            f"need 0 <= low ({low_quantile}) < high ({high_quantile}) <= 1"
        )
    if idf.size == 0:
        return (0.0, 0.0)
    low_t = float(np.quantile(idf, low_quantile))
    high_t = float(np.quantile(idf, high_quantile))
    return (low_t, high_t)


def clip_with_quantile_thresholds(
    terms: list[str],
    vocabulary: dict[str, int],
    idf: np.ndarray,
    low_threshold: float,
    high_threshold: float,
    *,
    boost_qf: int = DEFAULT_BOOST_QF,
) -> list[str]:
    """Return `terms` with low-IDF dropped and high-IDF boosted.

    For each term in input order:

    - If absent from `vocabulary`: pass through unchanged.
    - Else read its IDF from `idf[vocabulary[term]]`.
        - IDF strictly less than `low_threshold`: drop.
        - IDF strictly greater than `high_threshold`: emit
          `boost_qf` copies.
        - Otherwise (within the [low, high] band, inclusive): emit
          once.

    `boost_qf` must be >= 1 (1 disables the boost). Out-of-vocab
    terms are kept because the vocabulary may be a strict subset of
    the global term space (e.g. an ephemeral `BM25Index` over a
    per-rebuild belief subset); dropping silently would be
    surprising.
    """
    if boost_qf < 1:
        raise ValueError(f"boost_qf must be >= 1, got {boost_qf}")
    out: list[str] = []
    for term in terms:
        idx = vocabulary.get(term)
        if idx is None:
            out.append(term)
            continue
        term_idf = float(idf[idx])
        if term_idf < low_threshold:
            continue
        if term_idf > high_threshold:
            for _ in range(boost_qf):
                out.append(term)
        else:
            out.append(term)
    return out
