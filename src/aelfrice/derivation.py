"""Pure derivation layer: raw text in, Belief + Edges out, no store I/O.

`derive()` is the single place that turns a classified (or fixed-prior)
text input into a `Belief` dataclass and any accompanying `Edge` objects.
All six v2.0 ingest entry points delegate to this function so the
deterministic part of the pipeline is tested and auditable in one place.

Ingest entry points remain responsible for:
  - store.get_belief() duplicate checks
  - store.record_ingest() write-log entries
  - store.insert_belief() / store.insert_edge() persistence
  - store.record_corroboration() for re-assertions
  - any store.update_belief() for the lock-upgrade path

What `derive()` owns:
  - belief-id derivation
  - content-hash derivation
  - classifier dispatch (regex or fixed-prior)
  - alpha / beta / type / lock_level / origin selection
  - edge list construction (currently empty for most source_kinds;
    the DERIVED_FROM wiring in ingest_jsonl is caller-side because it
    spans consecutive turns and depends on prior-turn state)
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Final

from aelfrice.classification import classify_sentence
from aelfrice.models import (
    BELIEF_FACTUAL,
    INGEST_SOURCE_CLI_REMEMBER,
    INGEST_SOURCE_FILESYSTEM,
    INGEST_SOURCE_FEEDBACK_LOOP_SYNTHESIS,
    INGEST_SOURCE_GIT,
    INGEST_SOURCE_KINDS,
    INGEST_SOURCE_LEGACY_UNKNOWN,
    INGEST_SOURCE_MCP_REMEMBER,
    INGEST_SOURCE_PYTHON_AST,
    LOCK_NONE,
    LOCK_USER,
    ORIGIN_AGENT_INFERRED,
    ORIGIN_USER_STATED,
    Belief,
    Edge,
)

_BELIEF_ID_HEX_LEN: Final[int] = 16

# Source-kinds that go through the regex/LLM classify_sentence path.
_CLASSIFY_SOURCE_KINDS: Final[frozenset[str]] = frozenset({
    INGEST_SOURCE_FILESYSTEM,
    INGEST_SOURCE_GIT,
    INGEST_SOURCE_PYTHON_AST,
    INGEST_SOURCE_FEEDBACK_LOOP_SYNTHESIS,
    INGEST_SOURCE_LEGACY_UNKNOWN,
})

# Source-kinds that create user-locked beliefs with fixed priors.
_LOCK_SOURCE_KINDS: Final[frozenset[str]] = frozenset({
    INGEST_SOURCE_MCP_REMEMBER,
    INGEST_SOURCE_CLI_REMEMBER,
})


@dataclass(frozen=True)
class DerivationInput:
    """All inputs needed to derive a Belief from raw text.

    Fields:
    - raw_text: the sentence / paragraph / phrase to classify.
    - source_kind: one of INGEST_SOURCE_KINDS; controls which derivation
      path is taken (classify-based vs fixed-prior lock).
    - source_path: the provenance label written into the belief-id hash
      (e.g. "doc:README.md:p0", "git:commit:abc1234", "user", "triple").
      When None, the belief id is derived from (source_kind, raw_text).
    - raw_meta: optional caller-side metadata; not consumed by derive()
      itself but threaded through for caller bookkeeping.
    - session_id: written to belief.session_id on the output.
    - ts: ISO-8601 timestamp written to belief.created_at.
    - classifier_version: reserved for future LLM-classifier versioning;
      not used in the regex path. None is always valid.
    - rule_set_hash: reserved for future deterministic rule-set pinning;
      not used by derive() today.
    """

    raw_text: str
    source_kind: str  # one of INGEST_SOURCE_KINDS
    source_path: str | None
    raw_meta: dict | None  # type: ignore[type-arg]
    session_id: str | None
    ts: str
    classifier_version: str | None
    rule_set_hash: str | None


@dataclass(frozen=True)
class DerivationOutput:
    """Result of a derive() call.

    Fields:
    - belief: the derived Belief, or None when classification rejects
      the input (persist=False — questions, empty text).
    - edges: edges to insert alongside the belief. Currently empty for
      all source_kinds; reserved for future intra-turn edge generation.
    - skip_reason: human-readable string when belief is None
      ('persist=False', 'noise', etc.). None when belief is present.
    """

    belief: Belief | None
    edges: list[Edge]
    skip_reason: str | None


def _belief_id(text: str, source: str) -> str:
    """Stable id derived from sha256(source \\x00 text)[:16].

    Matches the scheme used by ingest._belief_id, scanner._derive_belief_id,
    and classification._derive_belief_id so re-ingesting an identical
    (source, text) pair is idempotent across all entry points.
    """
    h = hashlib.sha256(f"{source}\x00{text}".encode("utf-8")).hexdigest()
    return h[:_BELIEF_ID_HEX_LEN]


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def derive(inp: DerivationInput) -> DerivationOutput:
    """Pure function: raw text in, Belief + Edges out, no store I/O.

    Dispatches by `inp.source_kind`:

    - Classify-based (filesystem, git, python_ast, feedback_loop_synthesis,
      legacy_unknown): calls `classify_sentence(raw_text, source_path)`.
      Returns belief=None with skip_reason='persist=False' when the
      classifier rejects the input (questions, empty text). The
      `source_path` parameter is used as the classifier source so
      source-prior deflation applies correctly (non-user sources get
      deflated alpha).

    - Lock-based (mcp_remember, cli_remember): fixed priors
      (alpha=9.0, beta=0.5), type=factual, lock_level=user,
      origin=user_stated. Never rejects — the caller is explicitly
      asserting a user-locked belief.

    The function is pure: same DerivationInput => identical
    DerivationOutput. No global state, no I/O, no randomness.

    Raises ValueError for an unrecognised source_kind.
    """
    if inp.source_kind not in INGEST_SOURCE_KINDS:
        raise ValueError(
            f"unknown source_kind: {inp.source_kind!r}; "
            f"expected one of {sorted(INGEST_SOURCE_KINDS)}"
        )

    # The source label for the belief-id hash and classifier source
    # adjustment. Falls back to source_kind when source_path is absent
    # (e.g. triple-extractor callers that don't thread a path through).
    source_label = inp.source_path if inp.source_path is not None else inp.source_kind
    bid = _belief_id(inp.raw_text, source_label)
    ch = _content_hash(inp.raw_text)

    if inp.source_kind in _LOCK_SOURCE_KINDS:
        # Lock path: fixed high-confidence prior matching the requirement
        # prior (9.0, 0.5). Matches the hardcoded values in mcp_server.tool_lock
        # and cli._cmd_lock — user-locked beliefs carry the same prior as
        # hard requirements.
        alpha: float = 9.0
        beta: float = 0.5
        belief = Belief(
            id=bid,
            content=inp.raw_text,
            content_hash=ch,
            alpha=alpha,
            beta=beta,
            type=BELIEF_FACTUAL,
            lock_level=LOCK_USER,
            locked_at=inp.ts,
            demotion_pressure=0,
            created_at=inp.ts,
            last_retrieved_at=None,
            session_id=inp.session_id,
            origin=ORIGIN_USER_STATED,
        )
        return DerivationOutput(belief=belief, edges=[], skip_reason=None)

    # Classify-based path (filesystem, git, python_ast, etc.)
    result = classify_sentence(inp.raw_text, source_label)
    if not result.persist:
        return DerivationOutput(
            belief=None, edges=[], skip_reason="persist=False"
        )

    belief = Belief(
        id=bid,
        content=inp.raw_text,
        content_hash=ch,
        alpha=result.alpha,
        beta=result.beta,
        type=result.belief_type,
        lock_level=LOCK_NONE,
        locked_at=None,
        demotion_pressure=0,
        created_at=inp.ts,
        last_retrieved_at=None,
        session_id=inp.session_id,
        origin=ORIGIN_AGENT_INFERRED,
    )
    return DerivationOutput(belief=belief, edges=[], skip_reason=None)
