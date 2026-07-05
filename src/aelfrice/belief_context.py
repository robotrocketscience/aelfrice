"""Best-effort recovery of a stored belief's source-turn context (#1081).

The context-loss guard. Passive transcript capture splits a turn on
sentence boundaries, so a scope-qualifying clause can land in a sibling
belief (or an adjacent turn) while a bald claim survives on its own — the
harm reported in #1081 ("all *time* averages, in the coach API" stored
bald as "All averages are not useful."). Nothing is recoverable with
certainty after the fact — the qualifier may never have been captured —
but two substrates already present in the store recover it best-effort:

1. **turns.jsonl join** — the belief's `session_id` plus a content-substring
   match against the retained turn log (`transcript_logger.transcripts_dir`)
   recovers the FULL source turn, whose other sentences carry the scope.
2. **DERIVED_FROM anchor_text** — aelfrice already links consecutive user
   turns (and sub-floor fragments) with `anchor_text` = adjacent context;
   the belief's outbound `DERIVED_FROM` edges surface it.

Both are best-effort. Measured coverage on a real 26k-belief store is
~36% (join) / ~28% (anchor) of transcript beliefs; onboard/scanner
beliefs have no originating turn at all. The consuming agent decides
whether the recovered context changes how it should act on the belief —
semantic judgment stays in the agent, per PHILOSOPHY #605. This module is
read-only: no store mutation, no schema, no non-determinism (pure string
matching over a sorted file walk).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterator

from aelfrice.models import EDGE_DERIVED_FROM
from aelfrice.store import MemoryStore
from aelfrice.transcript_logger import transcripts_dir

# Cap recovered turns so a sentence that recurs across many turns does not
# flood the output; the count of total matches is still reported.
DEFAULT_MAX_TURN_MATCHES: int = 5


@dataclass
class ContextResult:
    """What `recover_context` found for one belief.

    - `turn_matches`: full source-turn texts whose content contained the
      belief's sentence (best-effort join substrate). Truncated to the
      caller's cap; `turn_match_total` is the pre-cap count.
    - `anchor_contexts`: (anchor_text, linked_belief_content) pairs from
      the belief's outbound DERIVED_FROM edges (adjacent-context substrate).
    - `has_session`: whether the belief carried a session_id at all
      (onboard/scanner beliefs do not, so the join can never fire).
    """

    belief_id: str
    content: str
    has_session: bool
    turn_matches: list[str] = field(default_factory=list)
    turn_match_total: int = 0
    anchor_contexts: list[tuple[str, str]] = field(default_factory=list)

    @property
    def recovered(self) -> bool:
        """True when at least one substrate surfaced context."""
        return bool(self.turn_matches or self.anchor_contexts)


def _iter_turn_records() -> Iterator[dict[str, object]]:
    """Yield every turn record across the live turns.jsonl and archives.

    Files are walked in sorted order so output is deterministic. Lines
    that are not JSON objects (compaction markers, malformed) are skipped.
    """
    tdir = transcripts_dir()
    if not tdir.exists():
        return
    for path in sorted(tdir.rglob("*.jsonl")):
        try:
            with path.open("r", encoding="utf-8") as fh:
                for raw in fh:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict):
                        yield obj
        except OSError:
            # A rotated/removed file mid-walk is expected; skip it.
            continue


def recover_context(
    store: MemoryStore,
    belief_id: str,
    *,
    max_turn_matches: int = DEFAULT_MAX_TURN_MATCHES,
) -> ContextResult | None:
    """Best-effort recover the source-turn context for one belief.

    Returns None if the belief does not exist. Otherwise returns a
    `ContextResult`; `recovered` is False when neither substrate matched
    (the honest "context not recoverable" outcome — common for
    onboard/scanner beliefs and for sessions whose turn log has rotated
    away).
    """
    belief = store.get_belief(belief_id)
    if belief is None:
        return None

    content = (belief.content or "").strip()
    session_id = belief.session_id
    has_session = bool(session_id)

    turn_matches: list[str] = []
    turn_total = 0
    if has_session and content:
        seen: set[str] = set()
        for rec in _iter_turn_records():
            if rec.get("session_id") != session_id:
                continue
            text = rec.get("text")
            if not isinstance(text, str) or content not in text:
                continue
            if text in seen:
                continue
            seen.add(text)
            turn_total += 1
            if len(turn_matches) < max_turn_matches:
                turn_matches.append(text)

    anchor_contexts: list[tuple[str, str]] = []
    for edge in store.edges_from(belief_id):
        if edge.type != EDGE_DERIVED_FROM:
            continue
        anchor = edge.anchor_text
        if not anchor:
            continue
        linked = store.get_belief(edge.dst)
        linked_content = linked.content if linked is not None else ""
        anchor_contexts.append((anchor, linked_content or ""))

    return ContextResult(
        belief_id=belief_id,
        content=content,
        has_session=has_session,
        turn_matches=turn_matches,
        turn_match_total=turn_total,
        anchor_contexts=anchor_contexts,
    )
