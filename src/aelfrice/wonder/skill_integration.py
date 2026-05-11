"""Skill-layer ↔ ``wonder_ingest`` adapter (#552).

This module is the contract the published ``/aelf:wonder`` slash command
follows when it runs in ``--axes`` mode:

1. The host runs ``aelf wonder QUERY --axes`` → emits a ``DispatchPayload``
   JSON with ``research_axes`` and ``speculative_anchor_ids``
   (``src/aelfrice/wonder/dispatch.py``).
2. The host spawns one subagent per axis and collects each subagent's
   response into a ``SubagentDocument`` (the per-row shape below).
3. The host serializes those documents as JSONL and pipes them through
   ``aelf wonder --persist-docs FILE`` which converts them to
   ``Phantom`` records and calls ``wonder_ingest``.

Keeping the document → ``Phantom`` translation in a Python module
(instead of inline in the CLI or the markdown) means the contract is
unit-testable without spawning subagents: tests construct
``SubagentDocument`` instances directly and assert the resulting
``Phantom`` list matches the documented shape.

The skill markdown is the production caller; the integration test is
the same caller with a mock subagent fixture. Both rely on this
module's shape.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from aelfrice.models import Phantom


# Score default for subagent-produced documents. The bake-off harness
# scores phantoms by graph-walk weight; subagent-dispatched phantoms
# have no analogous structural score, so we tag them at 1.0 (the same
# value `wonder_ingest` uses for its audit `source_path_hash` formatting).
# Promotion downstream still gates on α-bump from corroborations, so the
# score field is mostly an audit-trail marker for these phantoms.
DEFAULT_SUBAGENT_SCORE: float = 1.0

# Generator label prefix written into each phantom's audit row. The
# axis name is appended so promotion / GC code paths can identify which
# axis of a dispatch run produced which phantom.
GENERATOR_PREFIX: str = "subagent_dispatch"


@dataclass(frozen=True)
class SubagentDocument:
    """One subagent's response from a ``/aelf:wonder --axes`` run.

    ``axis_name`` mirrors the ``ResearchAxis.name`` field from the
    dispatch JSON; ``content`` is the subagent's research document
    (free-form text the subagent returned).
    """

    axis_name: str
    content: str


def documents_to_phantoms(
    documents: list[SubagentDocument],
    anchor_ids: tuple[str, ...],
    *,
    score: float = DEFAULT_SUBAGENT_SCORE,
) -> list[Phantom]:
    """Convert subagent documents to ``Phantom`` records.

    Every returned phantom is anchored to the same ``anchor_ids`` tuple —
    these are the ``speculative_anchor_ids`` the dispatch JSON surfaced
    as the seed beliefs the gap analysis identified. The phantom's
    ``content`` is the subagent document body verbatim; the
    ``generator`` field is ``"subagent_dispatch:<axis_name>"`` so
    downstream audit can tell which axis produced which phantom.

    Returns one ``Phantom`` per input document, in input order. Empty
    input → empty list, no exceptions.
    """
    return [
        Phantom(
            constituent_belief_ids=anchor_ids,
            generator=f"{GENERATOR_PREFIX}:{doc.axis_name}",
            content=doc.content,
            score=score,
        )
        for doc in documents
    ]


def load_documents_jsonl(path: Path) -> tuple[list[SubagentDocument], tuple[str, ...]]:
    """Read a ``--persist-docs`` JSONL file.

    Expected format: each non-empty line is a JSON object with these
    keys:

    * ``axis_name`` (str)
    * ``content`` (str)
    * ``anchor_ids`` (list[str])

    All rows must share the same ``anchor_ids`` (one dispatch run, one
    anchor set). Returns ``(documents, anchor_ids)``.

    Raises ``ValueError`` on malformed rows or mismatched anchor sets so
    the CLI can surface a clean error before touching the store.
    """
    documents: list[SubagentDocument] = []
    anchor_ids: tuple[str, ...] | None = None

    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{path}:{line_no} invalid JSON: {e}"
                ) from None

            for key in ("axis_name", "content", "anchor_ids"):
                if key not in row:
                    raise ValueError(
                        f"{path}:{line_no} missing key {key!r}"
                    )

            row_anchors = tuple(row["anchor_ids"])
            if anchor_ids is None:
                anchor_ids = row_anchors
            elif row_anchors != anchor_ids:
                raise ValueError(
                    f"{path}:{line_no} anchor_ids mismatch; "
                    f"expected {list(anchor_ids)}, got {list(row_anchors)}"
                )

            documents.append(SubagentDocument(
                axis_name=str(row["axis_name"]),
                content=str(row["content"]),
            ))

    if anchor_ids is None:
        return [], ()
    return documents, anchor_ids
