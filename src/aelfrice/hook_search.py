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

2. Hook-driven retrievals are *non-corrective* signals. A retrieved
   belief is evidence of relevance, not evidence that any neighbouring
   user-locked belief is wrong. Calling `apply_feedback` with the default
   propagate=True would pressure-walk locked beliefs on every prompt,
   silently auto-demoting them. The hook passes `propagate=False` so the
   posterior moves but the lock graph is left alone.

3. Hook-driven valence is implicit and weaker than explicit user
   feedback. A retrieval is exposure, not endorsement. Use a small
   positive valence (`HOOK_RETRIEVAL_VALENCE = 0.1`) so a thousand
   retrievals over a few months don't dominate the posterior the way a
   handful of explicit user thumbs-up should.

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

import sys
import traceback
from typing import IO, Final, Iterable

from aelfrice.feedback import apply_feedback
from aelfrice.models import Belief
from aelfrice.retrieval import DEFAULT_TOKEN_BUDGET, retrieve
from aelfrice.store import MemoryStore

HOOK_FEEDBACK_SOURCE: Final[str] = "hook"
"""`source` value written into feedback_history for every hook-driven
retrieval row. LIMITATIONS.md commits this string publicly; downstream
analysis (e.g. `SELECT ... WHERE source = 'hook'`) depends on it."""

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

    Wraps `retrieve(store, prompt, token_budget=...)` and, after the read,
    calls `record_retrieval` to write one audit row per returned belief.
    Returns the retrieval result; recording failures are caught and logged
    so the hook's read path is never blocked by a write-side error.
    """
    hits: list[Belief] = retrieve(store, prompt, token_budget=token_budget)
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
    source, propagate=False)`. propagate=False suppresses the
    demotion-pressure walk so implicit hook exposure does not pressure
    user-locked beliefs.

    Best-effort: any per-belief failure is logged to `stderr` (defaults
    to `sys.stderr`) and the loop continues. Returns the number of
    successful writes — callers can use this as a smoke metric without
    needing to inspect the audit log.
    """
    serr: IO[str] = stderr if stderr is not None else sys.stderr
    written: int = 0
    for b in beliefs:
        try:
            apply_feedback(
                store,
                b.id,
                valence,
                source,
                propagate=False,
            )
            written += 1
        except Exception:  # non-blocking: log and continue
            traceback.print_exc(file=serr)
    return written
