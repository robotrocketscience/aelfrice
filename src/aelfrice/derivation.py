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

from aelfrice.classification_core import (
    USER_SOURCE,
    classify_sentence,
    get_source_adjusted_prior,
)
from aelfrice.models import (
    BELIEF_FACTUAL,
    INGEST_SOURCE_CLI_REMEMBER,
    INGEST_SOURCE_GIT,
    INGEST_SOURCE_MCP_REMEMBER,
    INGEST_SOURCE_TRANSCRIPT,
    LOCK_NONE,
    LOCK_USER,
    ORIGIN_AGENT_INFERRED,
    ORIGIN_USER_STATED,
    ORIGIN_USER_TRANSCRIPT,
    Belief,
    Edge,
    retention_class_for_source,
)

# Source label used by the transcript-ingest path
# (`_handle_pre_compact` -> `aelf ingest-transcript`). When combined
# with `raw_meta["role"] == "user"`, `derive()` recognises this as
# user-typed chat content (#888) and applies the undeflated prior +
# ORIGIN_USER_TRANSCRIPT, distinguishing it from scanner-extracted
# document content (origin=agent_inferred, deflated alpha).
_TRANSCRIPT_SOURCE_LABEL: Final[str] = "transcript"

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
class RouteOverrides:
    """LLM-router post-derivation splice (#265 PR-B).

    The scanner's LLM-classify path produces a per-candidate decision
    that the deterministic `derive()` cannot reconstruct from raw text.
    Carrying these fields on the `DerivationInput` lets the worker apply
    them as a post-derivation splice on the classifier-path output: the
    `(type, origin, alpha, beta)` of the produced belief are replaced
    with router-supplied values; everything else (id, content_hash,
    edges) flows through `derive()` unchanged.

    `audit_source` is not a `Belief` field. The worker inspects it
    after insert and emits a `feedback_history` row when set and the
    belief was newly inserted (not corroborated). Mirrors today's
    direct-write scanner audit at `scanner.py:304-310`.

    Frozen at ingest time per the #265 ratification: rebuilds replay
    these values verbatim rather than re-rolling the LLM router.
    """

    belief_type: str
    origin: str
    alpha: float
    beta: float
    audit_source: str | None = None


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
    # Optional LLM-router post-derivation splice (#265 PR-B). Applied
    # only on the classifier path (filesystem / python_ast). On the
    # lock/remember and triple-extraction paths the deterministic
    # output is the contract; route_overrides is silently ignored
    # there.
    route_overrides: RouteOverrides | None = None


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

    Shared scheme with the derivation-worker path used by `ingest.py`
    and `scanner.py` — both route belief creation through
    `derivation_worker.run_worker` -> `derive()` -> `_belief_id`; neither
    module defines its own id helper — so the same (text, source) pair
    always resolves to the same id regardless of call site.
    """
    h = hashlib.sha256(f"{source}\x00{text}".encode("utf-8")).hexdigest()
    return h[:_BELIEF_ID_HEX_LEN]


def _lock_id(text: str) -> str:
    """Deterministic id for lock/remember call sites.

    This id becomes `derived.belief.id`; `mcp_server.py` and `cli.py`
    call `derive()` directly and read that id rather than
    re-deriving it locally — neither module defines its own id
    helper.
    """
    h = hashlib.sha256(f"lock\x00{text}".encode("utf-8")).hexdigest()
    return h[:_BELIEF_ID_HEX_LEN]


def _triple_belief_id(phrase: str) -> str:
    """Deterministic id for triple-extracted noun-phrase beliefs.

    Invoked via `triple_extractor.ingest_triples`, which drives this
    function through the derivation worker (`triple_extractor.py` has
    no local id helper of its own). Keyed on
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

    Produced via the same derivation-worker path as
    `_triple_belief_id`; no local hash helper exists in
    `triple_extractor.py`.
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
       factual with alpha=1.0 / beta=1.0; the id is assigned here,
       invoked from `triple_extractor.ingest_triples` via the
       derivation worker.
    3. Override paths: when `override_belief_type` is set, skip
       `classify_sentence` and look up the source-adjusted prior
       directly (origin=agent_inferred). When `route_overrides` is
       set, skip `classify_sentence` and use the router-supplied
       (type, origin, alpha, beta) verbatim.
    4. All other paths (filesystem, python_ast, feedback_loop_synthesis,
       legacy_unknown), when neither override is set: run
       `classify_sentence`; skip when `persist=False`. Transcript-ingest
       rows that carry `raw_meta["role"]=="user"` (#888) get the
       undeflated USER_SOURCE prior and `origin=user_transcript`; all
       other classifier-path rows get the deflated prior and
       `origin=agent_inferred`.

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
            created_at=ts,
            last_retrieved_at=None,
            session_id=inp.session_id,
            origin=ORIGIN_USER_STATED,
            retention_class=retention_class_for_source(inp.source_kind),
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
            created_at=ts,
            last_retrieved_at=None,
            session_id=inp.session_id,
            origin=ORIGIN_AGENT_INFERRED,
            retention_class=retention_class_for_source(inp.source_kind),
        )
        return DerivationOutput(belief=belief, edges=[])

    # claude-memory write-through mirror (#985) has no dedicated branch
    # here: it flows through the route_overrides path (below) when the mirror
    # hook supplies a frozen (origin=user_validated) decision for a
    # `type: user`/`feedback` file, and otherwise through the deterministic
    # classifier path as origin=agent_inferred. Keeping the per-type origin
    # decision out of the deterministic path (it depends on the file's
    # frontmatter, carried in raw_meta, which replay nulls) is what lets the
    # mirror stay replay-equality-stable — see `hook_claude_memory_mirror`.

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
            created_at=ts,
            last_retrieved_at=None,
            session_id=inp.session_id,
            origin=ORIGIN_AGENT_INFERRED,
            retention_class=retention_class_for_source(inp.source_kind),
        )
        return DerivationOutput(belief=belief, edges=[])

    if inp.route_overrides is not None:
        # LLM-router path: skip the regex classifier entirely. The
        # router has already produced (type, origin, alpha, beta);
        # `derive()`'s job is to mint the deterministic id +
        # content_hash and assemble the Belief shell. `persist=False`
        # routes are filtered upstream (scanner.py) before reaching
        # the worker, so we can assume persist=True here.
        ro = inp.route_overrides
        bid = _belief_id(raw, source)
        belief = Belief(
            id=bid,
            content=raw,
            content_hash=_content_hash(raw),
            alpha=ro.alpha,
            beta=ro.beta,
            type=ro.belief_type,
            lock_level=LOCK_NONE,
            locked_at=None,
            created_at=ts,
            last_retrieved_at=None,
            session_id=inp.session_id,
            origin=ro.origin,
            retention_class=retention_class_for_source(inp.source_kind),
        )
        return DerivationOutput(belief=belief, edges=[])

    # 4. Transcript ingest, role=user (#888). User-typed chat content
    #    enters via the source_kind=transcript path (#1089), and within
    #    that path the user turn is distinguished from agent capture by
    #    raw_meta["role"]=="user". Without this branch, the user's own
    #    statements get the agent-inferred deflated prior (alpha *= 0.2)
    #    that is calibrated for scanner-extracted document text — which
    #    means user-typed facts can sit below uniform-prior distractors
    #    in posterior rerank and never escape the noise floor. Treat
    #    them as USER_SOURCE for the prior; tag with ORIGIN_USER_TRANSCRIPT
    #    so they remain distinct from explicit `aelf lock` intent.
    raw_meta = inp.raw_meta or {}
    is_user_transcript = (
        inp.source_kind == INGEST_SOURCE_TRANSCRIPT
        and source == _TRANSCRIPT_SOURCE_LABEL
        and raw_meta.get("role") == "user"
    )
    classify_source = USER_SOURCE if is_user_transcript else source
    result = classify_sentence(raw, classify_source)
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
        created_at=ts,
        last_retrieved_at=None,
        session_id=inp.session_id,
        origin=(
            ORIGIN_USER_TRANSCRIPT if is_user_transcript
            else ORIGIN_AGENT_INFERRED
        ),
        retention_class=retention_class_for_source(inp.source_kind),
    )
    return DerivationOutput(belief=belief, edges=[])
