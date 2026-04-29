"""Pure belief-derivation function.

Separates the deterministic "raw text -> Belief + edges" computation
from the store I/O so:

- Every ingest entry point calls `derive()`, then does its own I/O.
- The v2.x replay harness can call `derive()` on rows from `ingest_log`
  without touching a live store.

No `MemoryStore` dependency anywhere in this module.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Final

from aelfrice.classification import classify_sentence, get_source_adjusted_prior
from aelfrice.models import (
    BELIEF_FACTUAL,
    INGEST_SOURCE_CLI_REMEMBER,
    INGEST_SOURCE_GIT,
    INGEST_SOURCE_MCP_REMEMBER,
    LOCK_NONE,
    LOCK_USER,
    ORIGIN_AGENT_INFERRED,
    ORIGIN_USER_STATED,
    Belief,
    Edge,
)

_BELIEF_ID_HEX_LEN: Final[int] = 16

# Source label used by triple-derived beliefs; matches
# `triple_extractor.TRIPLE_BELIEF_SOURCE`. Defined here to avoid a
# circular import (triple_extractor imports from models, not from
# derivation). Both constants must stay in sync.
_TRIPLE_BELIEF_SOURCE: Final[str] = "triple"


# ---------------------------------------------------------------------------
# Input / output dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DerivationInput:
    """Everything `derive()` needs; no store references.

    Fields match the columns written to `ingest_log` by each call site
    so that a replay harness can reconstruct a `DerivationInput` from a
    single log row.

    `source_kind` must be one of `INGEST_SOURCE_KINDS`.
    `ts` is an ISO-8601 timestamp string; callers are responsible for
    supplying it (so tests can inject a stable clock). Empty string
    triggers UTC-now inside `derive()`.
    """

    raw_text: str
    source_kind: str        # one of INGEST_SOURCE_KINDS
    source_path: str | None = None
    raw_meta: dict | None = None  # type: ignore[type-arg]
    session_id: str | None = None
    ts: str = ""            # ISO-8601; empty string -> utc-now
    classifier_version: str | None = None
    rule_set_hash: str | None = None
    # Optional pre-classified type from a host LLM (polymorphic onboard
    # handshake). When set, `derive()` skips `classify_sentence` and uses
    # this type to look up the source-adjusted prior.
    override_belief_type: str | None = None


@dataclass(frozen=True)
class DerivationOutput:
    """Result of `derive()`.

    `belief` is None when the classifier sets `persist=False` (questions,
    empty text, etc.).  Callers should check `belief is not None` before
    writing to the store.

    `edges` is the list of edges to insert after the belief lands.
    Currently always empty; placeholder for v2.x paths that derive
    edges from a single text block.

    `skip_reason` is a short string explaining why `belief` is None, or
    None when the belief will be persisted.
    """

    belief: Belief | None
    edges: list[Edge] = field(default_factory=list)
    skip_reason: str | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _belief_id(text: str, source: str) -> str:
    """Deterministic id from sha256(source + NUL + text)[:16].

    Shared scheme with `ingest._belief_id` and `scanner._derive_belief_id`
    so the same (text, source) pair always resolves to the same id
    regardless of call site.
    """
    h = hashlib.sha256(f"{source}\x00{text}".encode("utf-8")).hexdigest()
    return h[:_BELIEF_ID_HEX_LEN]


def _lock_id(text: str) -> str:
    """Deterministic id for lock/remember call sites.

    Matches `mcp_server._lock_id_for` and `cli._lock_id_for`.
    """
    h = hashlib.sha256(f"lock\x00{text}".encode("utf-8")).hexdigest()
    return h[:_BELIEF_ID_HEX_LEN]


def _triple_belief_id(phrase: str) -> str:
    """Deterministic id for triple-extracted noun-phrase beliefs.

    Matches `triple_extractor._belief_id_for_phrase`. Keyed on
    `_TRIPLE_BELIEF_SOURCE` so the same normalised phrase resolves to
    the same id across all extraction call sites.
    """
    normalized = " ".join(phrase.split()).lower()
    h = hashlib.sha256(
        f"{_TRIPLE_BELIEF_SOURCE}\x00{normalized}".encode("utf-8")
    ).hexdigest()
    return h[:_BELIEF_ID_HEX_LEN]


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _triple_content_hash(phrase: str) -> str:
    """Content hash for triple-derived beliefs (normalised + lower).

    Matches `triple_extractor._content_hash`.
    """
    normalized = " ".join(phrase.split()).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def derive(inp: DerivationInput) -> DerivationOutput:
    """Pure function: raw text in, belief (or skip) out. No I/O.

    Dispatch rules (in evaluation order):

    1. Lock / remember paths (`source_kind` in {mcp_remember,
       cli_remember}): always persist with a USER lock, no classifier.
    2. Triple-extraction path (`source_kind == git`): always persist as
       factual with alpha=1.0 / beta=1.0; id scheme matches
       `triple_extractor._belief_id_for_phrase`.
    3. All other paths (filesystem, python_ast, feedback_loop_synthesis,
       legacy_unknown): run `classify_sentence`; skip when
       `persist=False`.

    The belief `id` is derived deterministically from the input so that
    re-deriving the same input yields the same id — replay equality is
    id-stable.
    """
    ts = inp.ts if inp.ts else _utc_now_iso()
    raw = inp.raw_text

    # 1. Lock / remember paths -------------------------------------------
    if inp.source_kind in (INGEST_SOURCE_MCP_REMEMBER, INGEST_SOURCE_CLI_REMEMBER):
        bid = _lock_id(raw)
        belief = Belief(
            id=bid,
            content=raw,
            content_hash=_content_hash(raw),
            alpha=9.0,
            beta=0.5,
            type=BELIEF_FACTUAL,
            lock_level=LOCK_USER,
            locked_at=ts,
            demotion_pressure=0,
            created_at=ts,
            last_retrieved_at=None,
            session_id=inp.session_id,
            origin=ORIGIN_USER_STATED,
        )
        return DerivationOutput(belief=belief, edges=[])

    # 2. Triple-extraction path (git commit-ingest) -----------------------
    if inp.source_kind == INGEST_SOURCE_GIT:
        normalized = " ".join(raw.split())
        bid = _triple_belief_id(raw)
        belief = Belief(
            id=bid,
            content=normalized,
            content_hash=_triple_content_hash(raw),
            alpha=1.0,
            beta=1.0,
            type=BELIEF_FACTUAL,
            lock_level=LOCK_NONE,
            locked_at=None,
            demotion_pressure=0,
            created_at=ts,
            last_retrieved_at=None,
            session_id=inp.session_id,
            origin=ORIGIN_AGENT_INFERRED,
        )
        return DerivationOutput(belief=belief, edges=[])

    # 3. Classifier paths (filesystem, python_ast, etc.) ------------------
    source = inp.source_path or inp.source_kind

    if inp.override_belief_type is not None:
        # Host-LLM-classified path (polymorphic onboard handshake): the
        # caller has already determined the belief type; skip regex
        # classify_sentence and look up the source-adjusted prior directly.
        alpha, beta = get_source_adjusted_prior(inp.override_belief_type, source)
        bid = _belief_id(raw, source)
        belief = Belief(
            id=bid,
            content=raw,
            content_hash=_content_hash(raw),
            alpha=alpha,
            beta=beta,
            type=inp.override_belief_type,
            lock_level=LOCK_NONE,
            locked_at=None,
            demotion_pressure=0,
            created_at=ts,
            last_retrieved_at=None,
            session_id=inp.session_id,
            origin=ORIGIN_AGENT_INFERRED,
        )
        return DerivationOutput(belief=belief, edges=[])

    result = classify_sentence(raw, source)
    if not result.persist:
        return DerivationOutput(
            belief=None,
            edges=[],
            skip_reason="persist=False",
        )

    bid = _belief_id(raw, source)
    belief = Belief(
        id=bid,
        content=raw,
        content_hash=_content_hash(raw),
        alpha=result.alpha,
        beta=result.beta,
        type=result.belief_type,
        lock_level=LOCK_NONE,
        locked_at=None,
        demotion_pressure=0,
        created_at=ts,
        last_retrieved_at=None,
        session_id=inp.session_id,
        origin=ORIGIN_AGENT_INFERRED,
    )
    return DerivationOutput(belief=belief, edges=[])
