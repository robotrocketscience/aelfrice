# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUntypedFunctionDecorator=false, reportUnusedFunction=false
"""MCP server exposing the 12 user-visible tools.

The same surface as the CLI, accessible from any host that speaks the
Model Context Protocol. The handlers are pure Python — they take a
`MemoryStore` plus structured args and return JSON-shaped dicts. The thin
`serve()` shim opens a per-process store at the default path and
registers the pure handlers as FastMCP tools.

Tests import the pure handlers directly; they never need the optional
`fastmcp` dependency. `fastmcp` is listed under
`[project.optional-dependencies].mcp` and is only required when a host
actually starts the server via `serve()`.

Tool surface (all under the `aelf:` namespace at the host):

  aelf:onboard         polymorphic — three input shapes:
                         {path}                          -> start session
                         {session_id, classifications}   -> finish session
                         {}                              -> list pending
  aelf:search          {query, budget?}                  -> hits
  aelf:lock            {statement}                       -> id + action
  aelf:locked          {pressured?}                      -> locked beliefs
  aelf:demote          {belief_id}                       -> demoted bool
  aelf:validate        {belief_id, source?}              -> origin promotion
  aelf:unlock          {belief_id}                       -> lock cleared
  aelf:promote         {belief_id, source?}              -> alias of validate
  aelf:feedback        {belief_id, signal, source?}      -> updated priors
  aelf:confirm         {belief_id, source?, note?}       -> affirmed priors
  aelf:stats           {}                                -> counts
  aelf:health          {}                                -> regime report

The polymorphic onboard is a single MCP tool, not three (per pre-commit
on tool-surface invisibility — the host LLM should not see plumbing
distinctions). Internal dispatch picks the path by which fields the
caller supplied.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Final, Sequence

from aelfrice.classification import (
    HostClassification,
    accept_classifications,
    start_onboard_session,
)
from aelfrice.db_paths import db_path
from aelfrice.derivation import DerivationInput, derive
from aelfrice.derivation_worker import run_worker
from aelfrice.feedback import apply_feedback
from aelfrice.health import (
    REGIME_INSUFFICIENT_DATA,
    assess_health,
    regime_description,
)
from aelfrice.models import (
    CORROBORATION_SOURCE_MCP_REMEMBER,
    INGEST_SOURCE_MCP_REMEMBER,
    LOCK_USER,
    ORIGIN_USER_STATED,
    ORIGIN_USER_VALIDATED,
)
from aelfrice.retrieval import DEFAULT_TOKEN_BUDGET, retrieve
from aelfrice.scanner import scan_repo
from aelfrice.session_resolution import resolve_session_id
from aelfrice.store import MemoryStore

_FEEDBACK_VALENCES: Final[dict[str, float]] = {"used": 1.0, "harmful": -1.0}


# Server-level overview shown to host LLMs at registration time. Concise
# on purpose — hosts that surface the instructions field treat it as a
# hint, not a manual; per-tool docstrings carry the detail.
_SERVER_INSTRUCTIONS: Final[str] = """\
aelfrice exposes a local belief store: a small SQLite-backed memory of
locked rules, validated facts, and decay-managed agent inferences for
the current project. Tools fall into three groups:

- READ (search, locked, stats, health): retrieve or summarize beliefs
  before acting. Cheap, idempotent, no host approval needed.
- WRITE (lock, validate, promote, unlock, feedback, confirm, onboard):
  introduce or refine beliefs based on user signals. Idempotent where
  marked; otherwise expect each call to shift posterior or audit state.
- TIER (demote): the only destructively-flagged tool. Drops a lock or
  devalidates a belief one tier; reversible only by re-locking with
  fresh evidence.

When unsure what already exists, call aelf_search before aelf_lock.
When the user explicitly asserts a non-negotiable rule, prefer aelf_lock
over aelf_confirm. All tools operate against the LOCAL store only — no
network egress, no external APIs.
"""


# --- Helpers -----------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _open_default_store() -> MemoryStore:
    p = db_path()
    if str(p) != ":memory:":
        _ensure_parent_dir(p)
    return MemoryStore(str(p))


# --- Pure tool handlers (test target) ----------------------------------
#
# Each `tool_*` is a pure function over (store, args) -> dict. Tests
# invoke them directly. The serve() registration layer wraps them with
# a per-call store opened from the default path.


def tool_onboard(
    store: MemoryStore,
    *,
    path: str | None = None,
    session_id: str | None = None,
    classifications: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Polymorphic onboard. Three shapes:

    - `path` set       : start an onboard session, return sentences
    - `session_id` set : accept host classifications, finish the session
    - neither set      : status — list pending sessions
    """
    if path is not None:
        return _onboard_start(store, path)
    if session_id is not None:
        return _onboard_accept(store, session_id, classifications or [])
    return _onboard_status(store)


def _onboard_start(store: MemoryStore, repo_path: str) -> dict[str, Any]:
    # CI / no-host-agent environments use scan_repo for sync onboarding;
    # the polymorphic state machine is for hosts that can classify in
    # their own context. The MCP server always has a host (that's what
    # MCP is for), so this branch always uses the state machine path.
    result = start_onboard_session(
        store, Path(repo_path), now=_utc_now_iso()
    )
    return {
        "kind": "onboard.session_started",
        "session_id": result.session_id,
        "n_already_present": result.n_already_present,
        "sentences": [
            {"index": s.index, "text": s.text, "source": s.source}
            for s in result.sentences
        ],
    }


def _onboard_accept(
    store: MemoryStore,
    session_id: str,
    classifications: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    parsed: list[HostClassification] = []
    for c in classifications:
        parsed.append(HostClassification(
            index=int(c["index"]),
            belief_type=str(c["belief_type"]),
            persist=bool(c["persist"]),
        ))
    outcome = accept_classifications(
        store, session_id, parsed, now=_utc_now_iso()
    )
    return {
        "kind": "onboard.session_completed",
        "session_id": outcome.session_id,
        "inserted": outcome.inserted,
        "skipped_non_persisting": outcome.skipped_non_persisting,
        "skipped_existing": outcome.skipped_existing,
        "skipped_unclassified": outcome.skipped_unclassified,
    }


def _onboard_status(store: MemoryStore) -> dict[str, Any]:
    pending = store.list_pending_onboard_sessions()
    return {
        "kind": "onboard.status",
        "n_pending": len(pending),
        "pending_session_ids": [s.session_id for s in pending],
    }


def tool_onboard_sync(store: MemoryStore, *, path: str) -> dict[str, Any]:
    """Synchronous fallback that runs the regex-classifier pipeline
    instead of the host handshake. Exposed for hosts that explicitly
    want the no-LLM onboard path (CI runs, scripted setup). Not a
    user-visible MCP tool — the public surface is `tool_onboard`.
    """
    result = scan_repo(store, Path(path), now=_utc_now_iso())
    return {
        "kind": "onboard.sync_completed",
        "inserted": result.inserted,
        "skipped_existing": result.skipped_existing,
        "skipped_non_persisting": result.skipped_non_persisting,
        "total_candidates": result.total_candidates,
    }


def tool_search(
    store: MemoryStore, *, query: str, budget: int = DEFAULT_TOKEN_BUDGET,
) -> dict[str, Any]:
    hits = retrieve(store, query, token_budget=budget)
    return {
        "kind": "search.results",
        "n_hits": len(hits),
        "hits": [
            {
                "id": h.id,
                "content": h.content,
                "lock_level": h.lock_level,
                "type": h.type,
            }
            for h in hits
        ],
    }


def tool_lock(
    store: MemoryStore,
    *,
    statement: str,
    session_id: str | None = None,
) -> dict[str, Any]:
    now = _utc_now_iso()
    sid = resolve_session_id(session_id, surface_name="mcp aelf_lock")
    # #264 slice 2: route through the derivation worker. The worker
    # handles record_ingest -> derive -> insert_or_corroborate; the
    # entry point only owns the re-lock semantic on an existing
    # lock-id belief (worker would otherwise corroborate without
    # touching lock_level / locked_at).
    derived = derive(DerivationInput(
        raw_text=statement,
        source_kind=INGEST_SOURCE_MCP_REMEMBER,
        ts=now,
        session_id=sid,
    ))
    # mcp_remember always produces a belief; if it didn't, derivation
    # broke (empty input post-strip, classifier dropped all spans, etc).
    # Surface as a structured error rather than asserting — a crashing
    # MCP tool is far worse host-UX than a kind we can read and grep.
    if derived.belief is None:
        return {
            "kind": "lock.error",
            "id": "",
            "action": "error",
            "error": (
                "derivation produced no belief from the supplied "
                "statement (likely empty after normalization)"
            ),
        }
    lock_bid = derived.belief.id
    pre_existing_at_lock_id = store.get_belief(lock_bid) is not None
    ids_before: set[str] = set(store.list_belief_ids())
    log_id = store.record_ingest(
        source_kind=INGEST_SOURCE_MCP_REMEMBER,
        raw_text=statement,
        session_id=sid,
        ts=now,
        raw_meta={"call_site": CORROBORATION_SOURCE_MCP_REMEMBER},
    )
    run_worker(store)
    entry = store.get_ingest_log_entry(log_id)
    derived_ids = entry.get("derived_belief_ids") if entry is not None else None
    if not isinstance(derived_ids, list) or not derived_ids:
        # mcp_remember always derives a belief; an empty list here means
        # something is broken downstream. Surface as a kind we can grep.
        return {"kind": "lock.error", "id": "", "action": "error"}
    actual_id = str(derived_ids[0])
    if pre_existing_at_lock_id and actual_id == lock_bid:
        # Re-lock of an existing lock-id belief: apply lock-upgrade
        # (worker's insert_or_corroborate just records corroboration).
        existing = store.get_belief(actual_id)
        if existing is not None:
            existing.lock_level = LOCK_USER
            existing.locked_at = now
            existing.demotion_pressure = 0
            existing.origin = ORIGIN_USER_STATED
            store.update_belief(existing)
        return {"kind": "lock.upgraded", "id": actual_id, "action": "upgraded"}
    if actual_id in ids_before:
        # content_hash collision with a different-source belief: the
        # worker corroborated it; the original behavior returned
        # `corroborated` without applying lock semantics, preserved here.
        return {"kind": "lock.corroborated", "id": actual_id, "action": "corroborated"}
    return {"kind": "lock.created", "id": actual_id, "action": "locked"}


def tool_locked(
    store: MemoryStore, *, pressured: bool = False,
) -> dict[str, Any]:
    locked = store.list_locked_beliefs()
    if pressured:
        locked = [b for b in locked if b.demotion_pressure > 0]
    return {
        "kind": "locked.list",
        "n": len(locked),
        "locked": [
            {
                "id": b.id,
                "content": b.content,
                "demotion_pressure": b.demotion_pressure,
                "locked_at": b.locked_at,
            }
            for b in locked
        ],
    }


def tool_demote(store: MemoryStore, *, belief_id: str) -> dict[str, Any]:
    belief = store.get_belief(belief_id)
    if belief is None:
        return {
            "kind": "demote.not_found",
            "id": belief_id,
            "demoted": False,
            "error": "belief not found",
        }
    if belief.lock_level == LOCK_USER:
        from aelfrice.promotion import unlock
        unlock(store, belief_id)
        return {"kind": "demote.demoted", "id": belief_id, "demoted": True}
    if belief.origin == ORIGIN_USER_VALIDATED:
        from aelfrice.promotion import devalidate
        devalidate(store, belief_id)
        return {
            "kind": "demote.devalidated",
            "id": belief_id,
            "demoted": True,
            "tier": "user_validated_to_agent_inferred",
        }
    return {
        "kind": "demote.not_locked",
        "id": belief_id,
        "demoted": False,
    }


def tool_validate(
    store: MemoryStore,
    *,
    belief_id: str,
    source: str = "user_validated",
) -> dict[str, Any]:
    """Promote agent_inferred -> user_validated. v1.2.0.

    `source` becomes the audit-row source suffix: "promotion:<source>".
    Defaults to "user_validated" so the audit row reads
    "promotion:user_validated" — the canonical wire-format string.
    """
    from aelfrice.promotion import promote

    label = (
        f"promotion:{source}"
        if source != "user_validated"
        else "promotion:user_validated"
    )
    try:
        result = promote(store, belief_id, source_label=label)
    except ValueError as e:
        return {
            "kind": "validate.error",
            "id": belief_id,
            "error": str(e),
        }
    if result.already_validated:
        return {
            "kind": "validate.already",
            "id": belief_id,
            "prior_origin": result.prior_origin,
            "new_origin": result.new_origin,
        }
    return {
        "kind": "validate.promoted",
        "id": belief_id,
        "prior_origin": result.prior_origin,
        "new_origin": result.new_origin,
        "audit_event_id": result.audit_event_id,
    }


def tool_unlock(store: MemoryStore, *, belief_id: str) -> dict[str, Any]:
    """Drop a user-lock without touching origin.

    Idempotent: calling on an already-unlocked belief returns
    unlocked=False (no-op). Returns unlocked=True when the lock
    was actually cleared. Always writes a lock:unlock audit row on
    the active path.
    """
    from aelfrice.promotion import unlock

    try:
        result = unlock(store, belief_id)
    except ValueError as e:
        return {
            "kind": "unlock.not_found",
            "id": belief_id,
            "unlocked": False,
            "error": str(e),
        }
    if result.already_unlocked:
        return {
            "kind": "unlock.already",
            "id": belief_id,
            "unlocked": False,
        }
    return {
        "kind": "unlock.unlocked",
        "id": belief_id,
        "unlocked": True,
        "audit_event_id": result.audit_event_id,
    }


def tool_promote(
    store: MemoryStore,
    *,
    belief_id: str,
    source: str = "user_validated",
) -> dict[str, Any]:
    """Promote agent_inferred -> user_validated. Alias of tool_validate.

    Exists as a first-class tool so MCP callers can use the more
    intuitive name. Identical semantics and return shape to
    tool_validate.
    """
    return tool_validate(store, belief_id=belief_id, source=source)


def tool_feedback(
    store: MemoryStore,
    *,
    belief_id: str,
    signal: str,
    source: str = "user",
) -> dict[str, Any]:
    valence = _FEEDBACK_VALENCES.get(signal)
    if valence is None:
        return {
            "kind": "feedback.bad_signal",
            "id": belief_id,
            "error": f"signal must be 'used' or 'harmful', got: {signal}",
        }
    try:
        result = apply_feedback(
            store=store, belief_id=belief_id, valence=valence, source=source,
        )
    except ValueError as exc:
        return {
            "kind": "feedback.unknown_belief",
            "id": belief_id,
            "error": str(exc),
        }
    return {
        "kind": "feedback.applied",
        "id": belief_id,
        "signal": signal,
        "prior_alpha": result.prior_alpha,
        "new_alpha": result.new_alpha,
        "prior_beta": result.prior_beta,
        "new_beta": result.new_beta,
        "pressured_locks": result.pressured_locks,
        "demoted_locks": result.demoted_locks,
    }


_CONFIRM_VALENCE: Final[float] = 1.0
_CONFIRM_SOURCE_DEFAULT: Final[str] = "user_confirmed"


def tool_confirm(
    store: MemoryStore,
    *,
    belief_id: str,
    source: str = _CONFIRM_SOURCE_DEFAULT,
    note: str = "",
) -> dict[str, Any]:
    """Affirm a belief: apply a unit positive valence via apply_feedback.

    Records source as `user_confirmed` by default so confirm events are
    distinguishable from generic `used` feedback in the history table.
    `note` is an optional free-text annotation that appears in the return
    payload for the caller's context; it is not persisted to the store.

    The positive signal increments alpha, raising posterior_mean, and
    activates the demotion-pressure walk on any contradicting user-locked
    belief (same propagate=True default as apply_feedback).
    """
    try:
        result = apply_feedback(
            store=store,
            belief_id=belief_id,
            valence=_CONFIRM_VALENCE,
            source=source,
        )
    except ValueError as exc:
        return {
            "kind": "confirm.unknown_belief",
            "id": belief_id,
            "error": str(exc),
        }
    payload: dict[str, Any] = {
        "kind": "confirm.applied",
        "id": belief_id,
        "source": source,
        "prior_alpha": result.prior_alpha,
        "new_alpha": result.new_alpha,
        "prior_beta": result.prior_beta,
        "new_beta": result.new_beta,
        "pressured_locks": result.pressured_locks,
        "demoted_locks": result.demoted_locks,
    }
    if note:
        payload["note"] = note
    return payload


def tool_stats(store: MemoryStore) -> dict[str, Any]:
    n_edges = store.count_edges()
    return {
        "kind": "stats.snapshot",
        "beliefs": store.count_beliefs(),
        # `edges` is the v1.0 key. v1.1.0 adds `threads` as the
        # forward-compatible alias and emits both for one minor; v1.2.0
        # drops `edges`. Clients should migrate now.
        "edges": n_edges,
        "threads": n_edges,
        "locked": store.count_locked(),
        "feedback_events": store.count_feedback_events(),
        "onboard_sessions_total": store.count_onboard_sessions(),
    }


def tool_health(store: MemoryStore) -> dict[str, Any]:
    report = assess_health(store)
    payload: dict[str, Any] = {
        "kind": "health.report",
        "regime": report.regime,
        "description": regime_description(report.regime),
    }
    if report.regime != REGIME_INSUFFICIENT_DATA:
        payload["classification_confidence"] = report.classification_confidence
        payload["features"] = {
            "n_beliefs": report.features.n_beliefs,
            "confidence_mean": report.features.confidence_mean,
            "confidence_median": report.features.confidence_median,
            "mass_mean": report.features.mass_mean,
            "lock_per_1000": report.features.lock_per_1000,
            # `edge_per_belief` is the v1.0 key. v1.1.0 adds
            # `thread_per_belief` and emits both; v1.2.0 drops the edge form.
            "edge_per_belief": report.features.edge_per_belief,
            "thread_per_belief": report.features.edge_per_belief,
        }
    return payload


# --- FastMCP registration (optional) -----------------------------------
#
# The fastmcp library is an optional [mcp] extra. Importing it lazily
# inside `serve()` lets the rest of this module — and the test suite —
# work without it installed.


def serve() -> None:
    """Start a FastMCP server with the 12 tools registered.

    Requires the `[mcp]` extra: `pip install aelfrice[mcp]`. Raises
    `RuntimeError` with an actionable message if `fastmcp` is missing.

    fastmcp itself ships no type stubs, so its surface is cast to `Any`
    here. Strict pyright treats the rest of this function uniformly
    rather than emitting one warning per `mcp.tool` decorator.
    """
    try:
        from fastmcp import FastMCP as _FastMCPCls  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "fastmcp is not installed. Install with: pip install aelfrice[mcp]"
        ) from exc

    # pydantic is a transitive dep of fastmcp — import here, not at
    # module top, to keep `aelfrice.mcp_server` importable when the
    # [mcp] extra is absent (the test suite relies on this).
    from pydantic import Field  # type: ignore[import-not-found]

    # Reusable Field constraints. Defined inline so they share scope
    # with the lazily-imported `Field` symbol; promoting them to module
    # level would force pydantic at import time.
    _BeliefId = Annotated[
        str,
        Field(
            description=(
                "Stable hash-prefix belief ID returned by aelf_search, "
                "aelf_lock, or aelf_locked."
            ),
            min_length=1,
            max_length=64,
        ),
    ]
    _SourceLabel = Annotated[
        str,
        Field(
            description=(
                "Audit-row source suffix. Override only when a "
                "non-canonical source is appropriate."
            ),
            max_length=128,
        ),
    ]

    _FastMCP: Any = _FastMCPCls
    mcp: Any = _FastMCP(
        name="aelfrice",
        instructions=_SERVER_INSTRUCTIONS,
    )

    @mcp.tool(
        annotations={
            "title": "Onboard project into the belief store",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False,
        },
    )
    def aelf_onboard(
        path: Annotated[
            str | None,
            Field(
                default=None,
                description=(
                    "Absolute filesystem path to the project root. Set to "
                    "start an onboard session; leave None for the other "
                    "two shapes."
                ),
                max_length=4096,
            ),
        ] = None,
        session_id: Annotated[
            str | None,
            Field(
                default=None,
                description=(
                    "Session ID returned by a prior path-shape call. Set "
                    "with `classifications` to finalize."
                ),
                max_length=128,
            ),
        ] = None,
        classifications: Annotated[
            list[dict[str, Any]] | None,
            Field(
                default=None,
                description=(
                    "Host verdicts: each item {index: int, belief_type: "
                    "str, persist: bool}. Required when session_id is set."
                ),
            ),
        ] = None,
    ) -> dict[str, Any]:
        """Polymorphic ingest entrypoint for a project's belief corpus.

        Three input shapes drive the same tool. The host LLM picks shape
        by which fields it supplies; pass nothing to inspect state.

        Args:
            path: Absolute filesystem path to the project root. When set,
                starts an onboard session and returns the candidate
                sentences for the host to classify (e.g. "/Users/me/proj").
            session_id: An ID returned by a prior path-shape call. When
                set, finalizes that session by accepting the host's
                classifications.
            classifications: List of host verdicts to accept; required
                when session_id is set. Each item: {"index": int,
                "belief_type": str, "persist": bool}.

        Returns: dict with discriminating `kind`:
          - "onboard.session_started": {session_id, n_already_present,
            sentences:[{index, text, source}]}
          - "onboard.session_completed": {session_id, inserted,
            skipped_non_persisting, skipped_existing, skipped_unclassified}
          - "onboard.status": {n_pending, pending_session_ids}
        """
        store = _open_default_store()
        try:
            return tool_onboard(
                store, path=path, session_id=session_id,
                classifications=classifications,
            )
        finally:
            store.close()

    @mcp.tool(
        annotations={
            "title": "Search beliefs by query",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    )
    def aelf_search(
        query: Annotated[
            str,
            Field(
                description=(
                    "Search string. Whitespace-separated terms; SQLite "
                    "FTS5 syntax honored (NEAR, quoted phrases). "
                    "Examples: 'release process', 'auth NEAR token'."
                ),
                min_length=1,
                max_length=500,
            ),
        ],
        budget: Annotated[
            int,
            Field(
                description=(
                    "Soft token budget for the response. Lower values "
                    "trim hits aggressively."
                ),
                ge=1,
                le=100_000,
            ),
        ] = DEFAULT_TOKEN_BUDGET,
    ) -> dict[str, Any]:
        """Retrieve beliefs matching a free-text query, ranked by BM25.

        Locked beliefs (L0) auto-load above BM25 hits (L1). Use this to
        recall constraints, prior decisions, or facts the agent should
        know before acting. Read-only; never mutates the store.

        Args:
            query: Search string. Whitespace-separated terms; SQLite
                FTS5 syntax is honored (e.g. "auth NEAR token", quoted
                phrases). Examples: "release process", "uv tool upgrade".
            budget: Soft token budget for the response. Defaults to the
                module's DEFAULT_TOKEN_BUDGET; lower values trim hits.

        Returns: {"kind": "search.results", "n_hits": int,
                  "hits": [{"id": str, "content": str,
                            "lock_level": str, "type": str}, ...]}
        """
        store = _open_default_store()
        try:
            return tool_search(store, query=query, budget=budget)
        finally:
            store.close()

    @mcp.tool(
        annotations={
            "title": "Lock a belief as ground truth",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    )
    def aelf_lock(
        statement: Annotated[
            str,
            Field(
                description=(
                    "The free-text claim to lock as ground truth. "
                    "Treated verbatim; no rewriting. Example: 'All "
                    "commits must be signed'."
                ),
                min_length=1,
                max_length=2000,
            ),
        ],
    ) -> dict[str, Any]:
        """Lock a statement as user-asserted ground truth (L0).

        Use when the user has explicitly stated a non-negotiable rule,
        constraint, or fact that future sessions must respect. Re-locking
        the same content refreshes the lock without creating a duplicate.
        Mutates: creates or upgrades a belief.

        Args:
            statement: The free-text claim to lock. Treated verbatim;
                no rewriting. Example: "All commits must be signed".

        Returns: {"kind": one of [lock.created, lock.upgraded,
                                  lock.corroborated, lock.error],
                  "id": belief_id, "action": str}
        """
        store = _open_default_store()
        try:
            return tool_lock(store, statement=statement)
        finally:
            store.close()

    @mcp.tool(
        annotations={
            "title": "List user-locked beliefs",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    )
    def aelf_locked(
        pressured: Annotated[
            bool,
            Field(
                description=(
                    "If True, return only locks whose demotion_pressure "
                    "> 0 (challenged by contradicting evidence). "
                    "Default False returns all locks."
                ),
            ),
        ] = False,
    ) -> dict[str, Any]:
        """List all user-locked (L0) beliefs in the store.

        Use to show the user their current ground-truth set, or to find
        candidates for unlock/demote. Read-only.

        Args:
            pressured: If True, return only locks whose demotion_pressure
                is greater than zero (i.e. ones being challenged by
                contradicting evidence). Default False returns all locks.

        Returns: {"kind": "locked.list", "n": int,
                  "locked": [{"id": str, "content": str,
                              "demotion_pressure": int,
                              "locked_at": str}, ...]}
        """
        store = _open_default_store()
        try:
            return tool_locked(store, pressured=pressured)
        finally:
            store.close()

    @mcp.tool(
        annotations={
            "title": "Demote a belief one tier",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": False,
            "openWorldHint": False,
        },
    )
    def aelf_demote(belief_id: _BeliefId) -> dict[str, Any]:
        """Demote a belief one tier — drop a lock OR devalidate.

        For a user-locked (L0) belief: clears the lock to L1.
        For a user_validated belief: drops origin to agent_inferred.
        For other beliefs: no-op. Mutates origin/lock fields.

        Args:
            belief_id: Stable hash-prefix ID returned by aelf_search,
                aelf_lock, or aelf_locked.

        Returns: {"kind": one of [demote.demoted, demote.devalidated,
                                  demote.not_locked, demote.not_found],
                  "id": belief_id, "demoted": bool,
                  "tier": str (when devalidated),
                  "error": str (when not_found)}
        """
        store = _open_default_store()
        try:
            return tool_demote(store, belief_id=belief_id)
        finally:
            store.close()

    @mcp.tool(
        annotations={
            "title": "Validate (promote) an agent-inferred belief",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    )
    def aelf_validate(
        belief_id: _BeliefId,
        source: _SourceLabel = "user_validated",
    ) -> dict[str, Any]:
        """Promote agent_inferred → user_validated (no lock applied).

        Use when the user explicitly confirms an agent-inferred claim is
        correct, but does not want to lock it as ground truth. Writes
        an audit row 'promotion:<source>'. Mutates origin field.

        Args:
            belief_id: ID of the agent_inferred belief to promote.
            source: Audit-row source suffix; defaults to "user_validated".
                Override only when a non-canonical source is appropriate
                (e.g. an automated promotion pipeline).

        Returns: {"kind": one of [validate.promoted, validate.already,
                                  validate.error],
                  "id": str, "prior_origin": str, "new_origin": str,
                  "audit_event_id": int (when promoted),
                  "error": str (when error)}
        """
        store = _open_default_store()
        try:
            return tool_validate(
                store, belief_id=belief_id, source=source,
            )
        finally:
            store.close()

    @mcp.tool(
        annotations={
            "title": "Unlock a belief (clears L0 lock)",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    )
    def aelf_unlock(belief_id: _BeliefId) -> dict[str, Any]:
        """Drop a user-lock without changing the belief's origin.

        Idempotent: calling on an already-unlocked belief returns
        unlocked=False. Always writes a 'lock:unlock' audit row when the
        lock was actually cleared. Mutates lock_level from L0 to L1.

        Args:
            belief_id: ID of the locked belief to unlock.

        Returns: {"kind": one of [unlock.unlocked, unlock.already,
                                  unlock.not_found],
                  "id": str, "unlocked": bool,
                  "audit_event_id": int (when unlocked),
                  "error": str (when not_found)}
        """
        store = _open_default_store()
        try:
            return tool_unlock(store, belief_id=belief_id)
        finally:
            store.close()

    @mcp.tool(
        annotations={
            "title": "Promote (validate) an agent-inferred belief",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    )
    def aelf_promote(
        belief_id: _BeliefId,
        source: _SourceLabel = "user_validated",
    ) -> dict[str, Any]:
        """Alias of aelf_validate. Identical semantics and return shape.

        Exposed under both names so callers can use whichever verb reads
        more naturally for their use case ('promote' for tier transitions,
        'validate' for verification flows).

        Args, Returns: see aelf_validate.
        """
        store = _open_default_store()
        try:
            return tool_promote(
                store, belief_id=belief_id, source=source,
            )
        finally:
            store.close()

    @mcp.tool(
        annotations={
            "title": "Record feedback on a belief",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False,
        },
    )
    def aelf_feedback(
        belief_id: _BeliefId,
        signal: Annotated[
            str,
            Field(
                description=(
                    "Either 'used' (positive valence, +1) or 'harmful' "
                    "(negative valence, -1). Other values return a "
                    "bad_signal error without mutating."
                ),
                pattern=r"^(used|harmful)$",
            ),
        ],
        source: _SourceLabel = "user",
    ) -> dict[str, Any]:
        """Record positive or negative feedback on a belief's usefulness.

        Updates the Beta-Bernoulli posterior (alpha for 'used', beta for
        'harmful'). Negative feedback also walks the contradiction graph
        and increments demotion_pressure on supporting locks. Mutates
        posterior + audit + (potentially) lock pressure.

        Args:
            belief_id: Target belief ID.
            signal: Either "used" (positive valence, +1) or "harmful"
                (negative valence, -1). Other values return a bad_signal
                error without mutating.
            source: Free-text label for the feedback origin. Defaults to
                "user". Used for audit and provenance.

        Returns: {"kind": one of [feedback.applied, feedback.bad_signal,
                                  feedback.unknown_belief],
                  "id": str, "signal": str,
                  "prior_alpha": float, "new_alpha": float,
                  "prior_beta": float, "new_beta": float,
                  "pressured_locks": list[str], "demoted_locks": list[str],
                  "error": str (on error variants)}
        """
        store = _open_default_store()
        try:
            return tool_feedback(
                store, belief_id=belief_id, signal=signal, source=source,
            )
        finally:
            store.close()

    @mcp.tool(
        annotations={
            "title": "Confirm a belief (positive valence)",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False,
        },
    )
    def aelf_confirm(
        belief_id: _BeliefId,
        source: _SourceLabel = _CONFIRM_SOURCE_DEFAULT,
        note: Annotated[
            str,
            Field(
                description=(
                    "Optional free-text annotation. Returned on the "
                    "response payload but NOT persisted."
                ),
                max_length=2000,
            ),
        ] = "",
    ) -> dict[str, Any]:
        """Affirm an existing belief without locking it (bumps posterior).

        Use when the user reviews a belief and says it's correct, but the
        commitment is softer than aelf_lock would imply. Records source
        as 'user_confirmed' by default so confirms are distinguishable
        from generic 'used' feedback in the audit table. Mutates
        posterior and may pressure contradicting locks.

        Args:
            belief_id: Target belief ID.
            source: Audit source label. Default 'user_confirmed'.
            note: Optional free-text annotation. Returned on the response
                payload for the caller's context; NOT persisted.

        Returns: {"kind": one of [confirm.applied, confirm.unknown_belief],
                  "id": str, "source": str,
                  "prior_alpha": float, "new_alpha": float,
                  "prior_beta": float, "new_beta": float,
                  "pressured_locks": list[str], "demoted_locks": list[str],
                  "note": str (when supplied),
                  "error": str (when unknown_belief)}
        """
        store = _open_default_store()
        try:
            return tool_confirm(
                store, belief_id=belief_id, source=source, note=note,
            )
        finally:
            store.close()

    @mcp.tool(
        annotations={
            "title": "Snapshot belief-store counts",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    )
    def aelf_stats() -> dict[str, Any]:
        """Return summary counts for the local belief store.

        Cheap snapshot — no graph walk, no scoring. Use to verify the
        store is populated, to gauge corpus size, or as a heartbeat.
        Read-only.

        Returns: {"kind": "stats.snapshot",
                  "beliefs": int,
                  "edges": int,    # v1.0 key (deprecated, removed v1.2)
                  "threads": int,  # v1.1 key (forward-compatible alias)
                  "locked": int,
                  "feedback_events": int,
                  "onboard_sessions_total": int}
        """
        store = _open_default_store()
        try:
            return tool_stats(store)
        finally:
            store.close()

    @mcp.tool(
        annotations={
            "title": "Classify store operating regime",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    )
    def aelf_health() -> dict[str, Any]:
        """Classify the store's current operating regime + describe it.

        Runs the regime classifier over a small feature set (corpus size,
        confidence stats, lock density, edge density). Use as a coarse
        diagnostic before taking weighty mutating actions, or to drive
        UX nudges (e.g. 'too few locks for this stage'). Read-only.

        Returns: {"kind": "health.report",
                  "regime": str,
                  "description": str,
                  "classification_confidence": float (omitted when
                       insufficient data),
                  "features": {n_beliefs, confidence_mean,
                       confidence_median, mass_mean, lock_per_1000,
                       edge_per_belief, thread_per_belief}
                       (omitted when insufficient data)}
        """
        store = _open_default_store()
        try:
            return tool_health(store)
        finally:
            store.close()

    # The decorator rebinds each name on `mcp`; the local references
    # below silence pyright's reportUnusedFunction for the bindings
    # themselves. They serve no runtime purpose.
    _registered = (
        aelf_onboard, aelf_search, aelf_lock, aelf_locked,
        aelf_demote, aelf_validate, aelf_unlock, aelf_promote,
        aelf_feedback, aelf_confirm, aelf_stats, aelf_health,
    )
    del _registered

    mcp.run()


if __name__ == "__main__":  # pragma: no cover — exercised by `aelf mcp`
    # `python -m aelfrice.mcp_server` is the fallback entry point for
    # hosts that prefer module invocation over the `aelf mcp` console
    # script. Both routes call serve() identically.
    serve()
