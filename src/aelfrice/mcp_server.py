# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUntypedFunctionDecorator=false, reportUnusedFunction=false
"""MCP server exposing the 8 user-visible tools.

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
  aelf:feedback        {belief_id, signal, source?}      -> updated priors
  aelf:stats           {}                                -> counts
  aelf:health          {}                                -> regime report

The polymorphic onboard is a single MCP tool, not three (per pre-commit
on tool-surface invisibility — the host LLM should not see plumbing
distinctions). Internal dispatch picks the path by which fields the
caller supplied.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Sequence

from aelfrice.classification import (
    HostClassification,
    accept_classifications,
    start_onboard_session,
)
from aelfrice.cli import db_path
from aelfrice.feedback import apply_feedback
from aelfrice.health import (
    REGIME_INSUFFICIENT_DATA,
    assess_health,
    regime_description,
)
from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    LOCK_USER,
    Belief,
)
from aelfrice.retrieval import DEFAULT_TOKEN_BUDGET, retrieve
from aelfrice.scanner import scan_repo
from aelfrice.store import MemoryStore

_FEEDBACK_VALENCES: Final[dict[str, float]] = {"used": 1.0, "harmful": -1.0}
_LOCK_ID_LEN: Final[int] = 16


# --- Helpers -----------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _lock_id_for(content: str) -> str:
    return hashlib.sha256(f"lock\x00{content}".encode("utf-8")).hexdigest()[:_LOCK_ID_LEN]


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


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


def tool_lock(store: MemoryStore, *, statement: str) -> dict[str, Any]:
    bid = _lock_id_for(statement)
    existing = store.get_belief(bid)
    now = _utc_now_iso()
    if existing is None:
        store.insert_belief(Belief(
            id=bid,
            content=statement,
            content_hash=_content_hash(statement),
            alpha=9.0,
            beta=0.5,
            type=BELIEF_FACTUAL,
            lock_level=LOCK_USER,
            locked_at=now,
            demotion_pressure=0,
            created_at=now,
            last_retrieved_at=None,
        ))
        return {"kind": "lock.created", "id": bid, "action": "locked"}
    existing.lock_level = LOCK_USER
    existing.locked_at = now
    existing.demotion_pressure = 0
    store.update_belief(existing)
    return {"kind": "lock.upgraded", "id": bid, "action": "upgraded"}


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
    if belief.lock_level == LOCK_NONE:
        return {
            "kind": "demote.not_locked",
            "id": belief_id,
            "demoted": False,
        }
    belief.lock_level = LOCK_NONE
    belief.locked_at = None
    belief.demotion_pressure = 0
    store.update_belief(belief)
    return {"kind": "demote.demoted", "id": belief_id, "demoted": True}


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


def tool_stats(store: MemoryStore) -> dict[str, Any]:
    return {
        "kind": "stats.snapshot",
        "beliefs": store.count_beliefs(),
        "edges": store.count_edges(),
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
            "edge_per_belief": report.features.edge_per_belief,
        }
    return payload


# --- FastMCP registration (optional) -----------------------------------
#
# The fastmcp library is an optional [mcp] extra. Importing it lazily
# inside `serve()` lets the rest of this module — and the test suite —
# work without it installed.


def serve() -> None:
    """Start a FastMCP server with the 8 tools registered.

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

    _FastMCP: Any = _FastMCPCls
    mcp: Any = _FastMCP(name="aelfrice")

    @mcp.tool()
    def aelf_onboard(
        path: str | None = None,
        session_id: str | None = None,
        classifications: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        store = _open_default_store()
        try:
            return tool_onboard(
                store, path=path, session_id=session_id,
                classifications=classifications,
            )
        finally:
            store.close()

    @mcp.tool()
    def aelf_search(
        query: str, budget: int = DEFAULT_TOKEN_BUDGET,
    ) -> dict[str, Any]:
        store = _open_default_store()
        try:
            return tool_search(store, query=query, budget=budget)
        finally:
            store.close()

    @mcp.tool()
    def aelf_lock(statement: str) -> dict[str, Any]:
        store = _open_default_store()
        try:
            return tool_lock(store, statement=statement)
        finally:
            store.close()

    @mcp.tool()
    def aelf_locked(pressured: bool = False) -> dict[str, Any]:
        store = _open_default_store()
        try:
            return tool_locked(store, pressured=pressured)
        finally:
            store.close()

    @mcp.tool()
    def aelf_demote(belief_id: str) -> dict[str, Any]:
        store = _open_default_store()
        try:
            return tool_demote(store, belief_id=belief_id)
        finally:
            store.close()

    @mcp.tool()
    def aelf_feedback(
        belief_id: str, signal: str, source: str = "user",
    ) -> dict[str, Any]:
        store = _open_default_store()
        try:
            return tool_feedback(
                store, belief_id=belief_id, signal=signal, source=source,
            )
        finally:
            store.close()

    @mcp.tool()
    def aelf_stats() -> dict[str, Any]:
        store = _open_default_store()
        try:
            return tool_stats(store)
        finally:
            store.close()

    @mcp.tool()
    def aelf_health() -> dict[str, Any]:
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
        aelf_demote, aelf_feedback, aelf_stats, aelf_health,
    )
    del _registered

    mcp.run()
