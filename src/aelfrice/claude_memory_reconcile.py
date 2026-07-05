"""Shared claude-memory ingest core, extracted from the #985 PostToolUse
mirror so a full-set reconcile sweep (#1089) can reuse the exact same
frontmatter -> origin/prior mapping without drift.

The #985 mirror is write-event-triggered: it ingests a fact file when the
agent writes or edits it in-session, mapping the ``metadata.type``
frontmatter to a belief origin/prior. That per-file logic lived inline in
the hook; it is lifted here as :func:`ingest_memory_text` so the
reconcile sweep and the hook share one code path.

Contract (inherited from #985): one-way and non-authoritative — aelfrice
never writes back to the memory files. Idempotent — belief ids are
content-derived, so a re-run corroborates rather than duplicates.

The heavy derivation/store imports are module-top here (this module is not
a hot path); the PostToolUse hook keeps its lazy-import discipline by
importing this module only once a memory-file write is confirmed.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Final

from aelfrice.classification_core import (
    USER_SOURCE,
    classify_sentence,
    get_source_adjusted_prior,
)
from aelfrice.claude_memory import parse_memory_file
from aelfrice.derivation import DerivationInput, RouteOverrides, derive
from aelfrice.models import (
    BELIEF_FACTUAL,
    CORROBORATION_SOURCE_CLAUDE_MEMORY,
    INGEST_SOURCE_CLAUDE_MEMORY,
    ORIGIN_USER_VALIDATED,
)

if TYPE_CHECKING:
    from aelfrice.store import MemoryStore

# Cap the body we ingest so a pathologically large memory file cannot blow
# the latency budget or the belief content column (mirrors the #985 hook).
_BODY_BYTE_CAP: Final[int] = 16384


def _truncate(text: str) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= _BODY_BYTE_CAP:
        return text
    return encoded[:_BODY_BYTE_CAP].decode("utf-8", errors="ignore")


def ingest_memory_text(store: "MemoryStore", text: str) -> str | None:
    """Parse one memory fact-file's ``text``, derive a belief, and
    ``insert_or_corroborate`` it into ``store``. Returns the belief id on a
    successful ingest, else ``None`` (no frontmatter, empty body, or a
    non-persisting classification).

    This is the single home for the #985 frontmatter -> origin/prior
    mapping (ratified 2026-06-23) so the PostToolUse mirror and the #1089
    reconcile sweep cannot drift:

    - ``metadata.type`` ``user`` / ``feedback`` -> ``origin=user_validated``
      with the undeflated prior, frozen as a ``route_overrides`` decision
      (the driving frontmatter lives in ``raw_meta``, which replay nulls,
      so we freeze the decision rather than recompute it).
    - ``project`` / ``reference`` / absent -> ``route_overrides=None``, which
      lets ``derive()`` run the deterministic classifier path ->
      ``origin=agent_inferred`` with the deflated prior.
    - The mirror NEVER locks: L0 stays reserved for explicit ``aelf lock``.

    The caller owns ``store`` (the sweep opens one and loops; the hook opens
    one per write) so this function neither opens nor closes it.
    """
    parsed = parse_memory_file(text)
    if parsed is None:
        return None  # no frontmatter / empty body -> nothing to mirror

    body = _truncate(parsed.body)

    route_overrides = None
    if parsed.memory_type in ("user", "feedback"):
        result = classify_sentence(body, USER_SOURCE)
        if result.persist:
            belief_type, alpha, beta = (
                result.belief_type,
                result.alpha,
                result.beta,
            )
        else:
            belief_type = BELIEF_FACTUAL
            alpha, beta = get_source_adjusted_prior(BELIEF_FACTUAL, USER_SOURCE)
        route_overrides = RouteOverrides(
            belief_type=belief_type,
            origin=ORIGIN_USER_VALIDATED,
            alpha=alpha,
            beta=beta,
        )

    output = derive(
        DerivationInput(
            raw_text=body,
            source_kind=INGEST_SOURCE_CLAUDE_MEMORY,
            source_path=None,
            route_overrides=route_overrides,
        )
    )
    if output.belief is None:
        return None

    store.insert_or_corroborate(
        output.belief,
        source_type=CORROBORATION_SOURCE_CLAUDE_MEMORY,
    )
    return output.belief.id
