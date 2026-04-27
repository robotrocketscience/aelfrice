"""Two-layer retrieval: L0 locked-beliefs auto-load + L1 FTS5 BM25 keyword search.

Token-budgeted output (default 2000 tokens, ~4 chars/token estimate). L0
beliefs are always present in the output above any L1 result and are never
trimmed by the budget — locks are user-asserted ground truth and must
survive retrieval.

NO HRR, NO BFS multi-hop, NO entity-index in v1.0 (pre-commit #6). Those
land in a later release once the retrieval upgrade R&D is validated against
a real corpus.
"""
from __future__ import annotations

from typing import Final

from aelfrice.models import Belief
from aelfrice.store import Store

DEFAULT_TOKEN_BUDGET: Final[int] = 2000
_CHARS_PER_TOKEN: Final[float] = 4.0
DEFAULT_L1_LIMIT: Final[int] = 50


def _estimate_tokens(text: str) -> int:
    """Cheap char-based token estimate. Conservative (rounds up)."""
    if not text:
        return 0
    return int((len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN)


def _belief_tokens(b: Belief) -> int:
    return _estimate_tokens(b.content)


def retrieve(
    store: Store,
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
