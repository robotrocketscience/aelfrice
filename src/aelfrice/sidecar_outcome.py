"""#1407 — the per-fire BM25 sidecar outcome, in a numpy-free leaf module.

Which of the three sidecar outcomes a `BM25IndexCache.get()` took is written
by `aelfrice.bm25` and read by `aelfrice.hook`. The vocabulary and the
process-level snapshot live here rather than in `bm25` so the hook can reset
and read the field **without importing the numeric stack**.

That separation is load-bearing, not tidiness. Every hook fire is a fresh
process, and #1351 moved numpy / scipy / snowballstemmer off the hook's import
graph precisely because the majority of `UserPromptSubmit` fires are refused by
the prompt-shape gate and never retrieve. The reset has to run above that gate
(the cadence dispatch reaches `BM25IndexCache.get()` and is dispatched there),
so importing the recorder from `aelfrice.bm25` at that point would have pulled
all three back into every gate-skipped fire — reversing #1351 for exactly the
population #1351 exists for.

This module must therefore import nothing outside the standard library, and
nothing from `aelfrice` at all. `aelfrice.bm25` re-exports these names so
existing callers keep working.
"""
from __future__ import annotations

from typing import Final

# Three states, not a boolean: since #1199 a stale sidecar no longer implies a
# full rebuild, and collapsing `incremental` into `full_rebuild` is exactly what
# made #1199's 86.2% and the 8.5% latency proxy look contradictory when they
# were measuring different events.
SIDECAR_FRESH: Final[str] = "fresh"
SIDECAR_INCREMENTAL: Final[str] = "incremental"
SIDECAR_FULL_REBUILD: Final[str] = "full_rebuild"
SIDECAR_OUTCOMES: Final[frozenset[str]] = frozenset(
    {SIDECAR_FRESH, SIDECAR_INCREMENTAL, SIDECAR_FULL_REBUILD}
)

# The costliest outcome any `BM25IndexCache.get()` has recorded in this process
# since the last reset. #1380's cost case is `cold_cost x cold_rate`;
# `cold_cost` is measured (2.89 s cold first-fire at 44,668 beliefs) and
# `cold_rate` was not — the best prior estimate was a latency proxy that cannot
# tell a rebuild from lock contention or a cold page cache.
#
# `None` means no `get()` ran since the last reset — a fire that never built an
# index at all (no L1 lane, gate-skipped). That must stay distinguishable from
# `fresh`, or a fire doing no work is counted as a cache hit and the rate this
# exists to measure is silently inflated.
_LAST_SIDECAR_OUTCOME: str | None = None

# Cost order. A fire is classified by the most expensive thing it paid for, not
# by whichever `get()` happened to return last — see `_record_sidecar_outcome`.
_SIDECAR_COST: Final[dict[str, int]] = {
    SIDECAR_FRESH: 0,
    SIDECAR_INCREMENTAL: 1,
    SIDECAR_FULL_REBUILD: 2,
}


def last_sidecar_outcome() -> str | None:
    """The most expensive sidecar outcome of any `BM25IndexCache.get()` since
    the last `reset_sidecar_outcome()`, or None if none has run (#1407).

    Not "the most recent" — a fire that rebuilt and then hit a warm cache paid
    for the rebuild, and must be counted as one. See `_record_sidecar_outcome`.
    """
    return _LAST_SIDECAR_OUTCOME


def reset_sidecar_outcome() -> None:
    """Clear the outcome snapshot. Called by the hook before retrieval so the
    value read afterwards belongs to this fire and not a previous one."""
    global _LAST_SIDECAR_OUTCOME
    _LAST_SIDECAR_OUTCOME = None


def _record_sidecar_outcome(outcome: str) -> None:
    """Record the outcome of a `get()`, keeping the most expensive one so far.

    Validates against `SIDECAR_OUTCOMES` so the vocabulary is a production
    contract rather than a test-only one: an unrecognised state would otherwise
    reach `hook_audit.jsonl` and be counted as a category nothing knows how to
    aggregate, silently skewing the rate this field exists to measure.

    **Max-wins, not last-write-wins.** A single fire can call `get()` more than
    once — a cadence-driven rebuild followed by the main retrieval is the case
    that exists today. Under last-write-wins that fire recorded `fresh` even
    though it had just paid a full rebuild, and the latency proxy missed it too
    (74 ms), so the one event #1380 is priced on was the one event the field
    failed to count. Keeping the max makes the field answer the question it is
    actually asked: *did this fire pay for a rebuild?* Latent today only
    because cadence is default-off — and this field is meant to outlive that.
    """
    if outcome not in SIDECAR_OUTCOMES:
        raise ValueError(
            f"unknown sidecar outcome {outcome!r}; "
            f"expected one of {sorted(SIDECAR_OUTCOMES)}"
        )
    global _LAST_SIDECAR_OUTCOME
    if (
        _LAST_SIDECAR_OUTCOME is None
        or _SIDECAR_COST[outcome] > _SIDECAR_COST[_LAST_SIDECAR_OUTCOME]
    ):
        _LAST_SIDECAR_OUTCOME = outcome
