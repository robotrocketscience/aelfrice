"""Hook-driven retrieval audit: route the UserPromptSubmit hook through
`aelfrice.retrieval.retrieve()` and record every retrieval to
`feedback_history` so the "feedback updates the math" loop closes for
beliefs surfaced during normal session activity.

This module exists because:

1. The retrieval-side of the feedback loop needs an audit row for every
   belief the hook injects. Without it, `apply_feedback`'s posterior
   moves only on explicit `aelf feedback` calls and never on the
   high-volume implicit signal of "this belief was retrieved and shown
   to the agent". The README's Bayesian-memory claim depends on this
   loop closing.

2. A retrieval is exposure, not endorsement. Since #1086 the hook
   records the exposure event to `feedback_history` (so surfacing
   frequency stays recoverable) but does NOT move the Bayesian posterior
   by default: counting every surfacing as positive evidence let whatever
   recurs float above genuine knowledge. The legacy behaviour — a small
   positive `HOOK_RETRIEVAL_VALENCE = 0.1` per retrieval — is restored by
   setting `AELFRICE_EXPOSURE_UPDATES_POSTERIOR=1` (benchmark A/B and
   rollback).

The module exposes two functions: `search_for_prompt`, the hook's
top-level call (retrieve + record), and `record_retrieval`, the audit
half on its own. Both are best-effort on the write side: a failure to
record does not prevent the retrieved beliefs from being returned.

Non-blocking guarantee: `record_retrieval` swallows write-side failures
with a stderr trace; the caller still gets the retrieval results. The
read side (retrieve) is allowed to raise; `aelfrice.hook` catches at
the outer try/except and degrades to "no memory injected" per the
hook contract.
"""
from __future__ import annotations

import os
import sys
import traceback
from typing import IO, Final, Iterable

from aelfrice.feedback import apply_feedback
from aelfrice.models import Belief
from aelfrice.retrieval import DEFAULT_TOKEN_BUDGET, retrieve
from aelfrice.store import MemoryStore

HOOK_FEEDBACK_SOURCE: Final[str] = "hook"
"""`source` value written into feedback_history for every hook-driven
retrieval row. ARCHITECTURE.md commits this string publicly; downstream
analysis (e.g. `SELECT ... WHERE source = 'hook'`) depends on it."""

ENV_EXPOSURE_UPDATES_POSTERIOR: Final[str] = "AELFRICE_EXPOSURE_UPDATES_POSTERIOR"
"""Opt-in to the legacy behaviour where a hook retrieval moves the
Bayesian posterior. Default is OFF (#1086): a retrieval is exposure, not
endorsement, and counting every surfacing as positive evidence inflated
whatever recurs — measured on a real store, junk (session scaffolding,
fragments) accumulated MORE exposure than genuine knowledge and floated
above it. With the flag unset, hook retrievals are recorded to
feedback_history (so exposure frequency stays recoverable) but leave the
posterior untouched. Set to "1" to restore the pre-#1086 posterior update
(kept for benchmark A/B and rollback). Read per call so a flip is honoured
without restart."""


def _exposure_updates_posterior() -> bool:
    return os.environ.get(ENV_EXPOSURE_UPDATES_POSTERIOR, "0") == "1"


HOOK_RETRIEVAL_VALENCE: Final[float] = 0.1
"""Per-belief positive valence written for each hook-driven retrieval.

Smaller than the conventional ±1.0 because a hook retrieval is implicit
exposure, not explicit user endorsement. Tuned conservatively at v1.0.1
to keep posterior drift slow under high-volume hook activity. v1.x may
introduce source-weighted decay or per-source valence calibration; this
is the simplest defensible value."""


def search_for_prompt(
    store: MemoryStore,
    prompt: str,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    *,
    stderr: IO[str] | None = None,
) -> list[Belief]:
    """Retrieve hits for a hook prompt and record them to feedback_history.

    Wraps `retrieve(store, prompt, token_budget=...)` and, after the
    read, calls `record_retrieval` to write one audit row per returned
    belief.
    """
    # #1016-B: this is a hook injection path whose formatter renders
    # reference-tier locks as a one-line manifest, so budget them at
    # manifest size (frees relevance budget; byte-identical until a lock
    # is demoted to reference).
    hits: list[Belief] = retrieve(
        store, prompt, token_budget=token_budget,
        manifest_reference_locks=True,
    )
    record_retrieval(store, hits, stderr=stderr)
    return hits


def record_retrieval(
    store: MemoryStore,
    beliefs: Iterable[Belief],
    *,
    valence: float = HOOK_RETRIEVAL_VALENCE,
    source: str = HOOK_FEEDBACK_SOURCE,
    stderr: IO[str] | None = None,
) -> int:
    """Write one feedback_history row per belief; return rows written.

    For each belief, calls `apply_feedback(store, belief.id, valence,
    source)` — Beta-Bernoulli posterior update plus an audit row.

    Best-effort: any per-belief failure is logged to `stderr` (defaults
    to `sys.stderr`) and the loop continues. Returns the number of
    successful writes — callers can use this as a smoke metric without
    needing to inspect the audit log.
    """
    serr: IO[str] = stderr if stderr is not None else sys.stderr
    update_posterior: bool = _exposure_updates_posterior()
    written: int = 0
    stamped_ids: list[str] = []
    for b in beliefs:
        try:
            apply_feedback(
                store,
                b.id,
                valence,
                source,
                update_posterior=update_posterior,
            )
            written += 1
            stamped_ids.append(b.id)
        except Exception:  # non-blocking: log and continue
            traceback.print_exc(file=serr)
    # Mirror the audit row to beliefs.last_retrieved_at so downstream
    # consumers (decay moderation, recency-aware ranking, telemetry) get
    # an O(1) read instead of having to join feedback_history. Same
    # best-effort posture as the loop above.
    if stamped_ids:
        try:
            store.stamp_retrieved(stamped_ids)
        except Exception:
            traceback.print_exc(file=serr)
    return written
