"""Domain dataclasses for aelfrice.

Load-bearing fields only. 4 belief types, 5 edge types, 2 lock levels. No
multi-source tagging, no rigor tier, no bitemporal event_time — those are
deferred to a later release.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# --- Belief types ---
BELIEF_FACTUAL: Final[str] = "factual"
BELIEF_CORRECTION: Final[str] = "correction"
BELIEF_PREFERENCE: Final[str] = "preference"
BELIEF_REQUIREMENT: Final[str] = "requirement"

BELIEF_TYPES: Final[frozenset[str]] = frozenset({
    BELIEF_FACTUAL,
    BELIEF_CORRECTION,
    BELIEF_PREFERENCE,
    BELIEF_REQUIREMENT,
})

# --- Edge types ---
EDGE_SUPPORTS: Final[str] = "SUPPORTS"
EDGE_CITES: Final[str] = "CITES"
EDGE_CONTRADICTS: Final[str] = "CONTRADICTS"
EDGE_SUPERSEDES: Final[str] = "SUPERSEDES"
EDGE_RELATES_TO: Final[str] = "RELATES_TO"
EDGE_DERIVED_FROM: Final[str] = "DERIVED_FROM"

# Edge-type valence multipliers for propagation.
# Positive = propagate same sign; negative = invert; 0.0 = no propagation.
# DERIVED_FROM mirrors CITES (0.5): both indicate B's content depends on A,
# distinct because DERIVED_FROM carries stronger contextual coupling
# (sibling becomes stale if A is superseded).
EDGE_VALENCE: Final[dict[str, float]] = {
    EDGE_SUPPORTS: 1.0,
    EDGE_CITES: 0.5,
    EDGE_CONTRADICTS: -0.5,
    EDGE_SUPERSEDES: 0.0,
    EDGE_RELATES_TO: 0.3,
    EDGE_DERIVED_FROM: 0.5,
}

EDGE_TYPES: Final[frozenset[str]] = frozenset(EDGE_VALENCE.keys())

# --- Lock levels ---
LOCK_NONE: Final[str] = "none"
LOCK_USER: Final[str] = "user"

LOCK_LEVELS: Final[frozenset[str]] = frozenset({LOCK_NONE, LOCK_USER})

# --- Origin (provenance tier, v1.2+) ---
# Wire-format strings; appear in feedback_history.source for promotion
# events. Do not rename without a migration. Tier ordering matches
# contradiction.py precedence numbering.
ORIGIN_USER_STATED: Final[str] = "user_stated"
ORIGIN_USER_CORRECTED: Final[str] = "user_corrected"
ORIGIN_USER_VALIDATED: Final[str] = "user_validated"
ORIGIN_AGENT_INFERRED: Final[str] = "agent_inferred"
ORIGIN_AGENT_REMEMBERED: Final[str] = "agent_remembered"
ORIGIN_DOCUMENT_RECENT: Final[str] = "document_recent"
ORIGIN_UNKNOWN: Final[str] = "unknown"

ORIGINS: Final[frozenset[str]] = frozenset({
    ORIGIN_USER_STATED,
    ORIGIN_USER_CORRECTED,
    ORIGIN_USER_VALIDATED,
    ORIGIN_AGENT_INFERRED,
    ORIGIN_AGENT_REMEMBERED,
    ORIGIN_DOCUMENT_RECENT,
    ORIGIN_UNKNOWN,
})

# --- Corroboration source types (v1.5+) ---
# Wire-format strings. Do not rename without a migration. Distinguishes
# which ingest path produced the corroboration row. Used by T2 (#191)
# for grace-window cross-checking.
CORROBORATION_SOURCE_COMMIT_INGEST: Final[str] = "commit_ingest"
CORROBORATION_SOURCE_TRANSCRIPT_INGEST: Final[str] = "transcript_ingest"
CORROBORATION_SOURCE_MCP_REMEMBER: Final[str] = "mcp_remember"
CORROBORATION_SOURCE_HOOK_INGEST: Final[str] = "hook_ingest"

CORROBORATION_SOURCE_TYPES: Final[frozenset[str]] = frozenset({
    CORROBORATION_SOURCE_COMMIT_INGEST,
    CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
    CORROBORATION_SOURCE_MCP_REMEMBER,
    CORROBORATION_SOURCE_HOOK_INGEST,
})

# v2.0 #205 ingest_log source_kind enum. Wire-format strings; do not
# rename without a migration. Spec: docs/design/write-log-as-truth.md.
INGEST_SOURCE_FILESYSTEM: Final[str] = "filesystem"
INGEST_SOURCE_GIT: Final[str] = "git"
INGEST_SOURCE_PYTHON_AST: Final[str] = "python_ast"
INGEST_SOURCE_MCP_REMEMBER: Final[str] = "mcp_remember"
INGEST_SOURCE_CLI_REMEMBER: Final[str] = "cli_remember"
INGEST_SOURCE_FEEDBACK_LOOP_SYNTHESIS: Final[str] = "feedback_loop_synthesis"
# `legacy_unknown` is reserved for migration: pre-v2.0 beliefs get
# synthesized log rows at their `created_at` timestamp.
INGEST_SOURCE_LEGACY_UNKNOWN: Final[str] = "legacy_unknown"

INGEST_SOURCE_KINDS: Final[frozenset[str]] = frozenset({
    INGEST_SOURCE_FILESYSTEM,
    INGEST_SOURCE_GIT,
    INGEST_SOURCE_PYTHON_AST,
    INGEST_SOURCE_MCP_REMEMBER,
    INGEST_SOURCE_CLI_REMEMBER,
    INGEST_SOURCE_FEEDBACK_LOOP_SYNTHESIS,
    INGEST_SOURCE_LEGACY_UNKNOWN,
})

# --- Onboard-session states ---
# A polymorphic onboard handshake passes through exactly two persisted
# states: `pending` after `start_onboard_session` records the scanner
# candidates, `completed` after the host's classifications have been
# accepted and beliefs inserted. No `aborted` state in v1.0 — abandoned
# sessions are simply garbage-collected later by id and timestamp.
ONBOARD_STATE_PENDING: Final[str] = "pending"
ONBOARD_STATE_COMPLETED: Final[str] = "completed"

ONBOARD_STATES: Final[frozenset[str]] = frozenset({
    ONBOARD_STATE_PENDING,
    ONBOARD_STATE_COMPLETED,
})


@dataclass
class Belief:
    """A unit of memory with Bayesian confidence and lock state.

    Fields: id, content, content_hash, alpha, beta, type, lock_level,
    locked_at, demotion_pressure, created_at, last_retrieved_at,
    session_id, origin.

    `session_id` (v1.2+) tags the belief with the ingest session that
    inserted it. Optional: ingest paths that don't open a session
    leave it None and downstream session-coherent retrieval simply
    does not fire on those rows.

    `origin` (v1.2+) is the provenance tier for the contradiction
    tie-breaker. Defaults to `unknown` for forward compatibility with
    v1.0/v1.1 stores; first-startup migration backfills locked beliefs
    to `user_stated` and correction beliefs to `user_corrected`. New
    inserts should pass an explicit value (scanner: `agent_inferred`;
    lock: `user_stated`).
    """

    id: str
    content: str
    content_hash: str
    alpha: float
    beta: float
    type: str
    lock_level: str
    locked_at: str | None
    demotion_pressure: int
    created_at: str
    last_retrieved_at: str | None
    session_id: str | None = None
    origin: str = ORIGIN_UNKNOWN
    corroboration_count: int = 0


ANCHOR_TEXT_MAX_LEN: Final[int] = 1000
"""Soft cap on edge anchor_text. Real anchor text is short prose
(20-200 chars). Cap protects against pathological writes; callers
truncate with a warning rather than reject."""


@dataclass
class Edge:
    """A typed, weighted directed link between two beliefs.

    `anchor_text` is the citing belief's own phrasing of the
    relationship (e.g. "the WAL discussion" on a CITES edge). Set
    by ingest paths that have access to source prose at edge-
    creation time; left None on programmatic / bulk inserts.
    Truncated to ANCHOR_TEXT_MAX_LEN characters on construction.
    """

    src: str
    dst: str
    type: str
    weight: float
    anchor_text: str | None = None

    def __post_init__(self) -> None:
        if self.anchor_text is not None and len(self.anchor_text) > ANCHOR_TEXT_MAX_LEN:
            self.anchor_text = self.anchor_text[:ANCHOR_TEXT_MAX_LEN]


@dataclass
class OnboardSession:
    """One polymorphic-onboard handshake row.

    `candidates_json` is the JSON-serialized list of scanner candidates
    awaiting host classification. Stored as a blob (rather than a
    side-table) because the candidate list is read/written exactly once
    per session and never queried by field — JSON keeps the schema
    minimal without imposing a second table.
    """

    session_id: str
    repo_path: str
    state: str
    candidates_json: str
    created_at: str
    completed_at: str | None


@dataclass
class FeedbackEvent:
    """One row in the feedback_history audit log.

    Recorded for every successful apply_feedback call so the project's
    feedback regime can be characterized after the fact. Closes the v2.0
    gap where the audit table only logged ignored/superseded events.
    """

    id: int
    belief_id: str
    valence: float
    source: str
    created_at: str


@dataclass
class Session:
    """Ephemeral session identifier produced by MemoryStore.create_session.

    The public v1.0.0 schema does not persist sessions; this dataclass
    exists so academic-suite benchmark adapters that group ingest calls
    by session_id (LongMemEval, LoCoMo, MAB, StructMemEval, AMA-Bench)
    have a stable handle to pass through. Lab v2.0.0 persists this as
    a full sessions table with token/correction/velocity tracking; the
    public shim exposes only what the adapters consume.
    """

    id: str
    started_at: str
    completed_at: str | None
    model: str | None
    project_context: str | None


