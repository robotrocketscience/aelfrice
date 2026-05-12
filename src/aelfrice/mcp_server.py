# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUntypedFunctionDecorator=false, reportUnusedFunction=false
"""MCP server exposing the 14 user-visible tools.

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
  aelf:wonder_persist  {query, budget?, depth?, top?, seed?} -> insert summary
  aelf:wonder_gc       {ttl_days?, dry_run?}            -> gc summary

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
from aelfrice.federation import ForeignBeliefError
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
from aelfrice.wonder.dispatch import build_dispatch_payload

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


# --- Response formatting ------------------------------------------------
#
# Read-only tools accept `response_format = "json" | "markdown"`. JSON
# (default) returns the structured dict the LLM can read natively.
# Markdown returns a wrapped dict {kind: "<orig>.markdown", format,
# text} where `text` is a human-readable rendering — useful when the
# host surface displays raw tool output to the user without LLM
# rephrasing.

_RESPONSE_FORMAT_JSON: Final[str] = "json"
_RESPONSE_FORMAT_MARKDOWN: Final[str] = "markdown"
_RESPONSE_FORMATS: Final[frozenset[str]] = frozenset(
    {_RESPONSE_FORMAT_JSON, _RESPONSE_FORMAT_MARKDOWN}
)


def _wrap_markdown(json_payload: dict[str, Any], text: str) -> dict[str, Any]:
    """Wrap a markdown rendering with the JSON payload's metadata.

    Always emits a stable shape so callers can branch on format:
      {"kind": "<orig>.markdown", "format": "markdown", "text": str}
    """
    orig_kind = json_payload.get("kind", "unknown")
    return {
        "kind": f"{orig_kind}.markdown",
        "format": "markdown",
        "text": text,
    }


def _render_search_markdown(payload: dict[str, Any]) -> str:
    hits = payload.get("hits", [])
    if not hits:
        return f"# Search results\n\nNo hits ({payload.get('n_hits', 0)})."
    lines = [f"# Search results — {payload['n_hits']} hits", ""]
    for h in hits:
        lines.append(f"## {h['id']} ({h.get('lock_level', 'L?')}, {h.get('type', '?')})")
        lines.append(h.get("content", "").strip() or "(empty content)")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_locked_markdown(payload: dict[str, Any]) -> str:
    locked = payload.get("locked", [])
    total = payload.get("total", payload.get("n", 0))
    n = payload.get("n", 0)
    offset = payload.get("offset", 0)
    has_more = payload.get("has_more", False)
    header = (
        f"# Locked beliefs — page {offset // max(n, 1) + 1 if n else 1}, "
        f"{n} of {total} shown"
    )
    if not locked:
        return f"{header}\n\nNo locks found at offset {offset}.\n"
    lines = [header, ""]
    for b in locked:
        pressure = b.get("demotion_pressure", 0)
        suffix = f" (pressure={pressure})" if pressure > 0 else ""
        lines.append(f"- **{b['id']}**{suffix}: {b.get('content', '').strip()}")
    if has_more:
        lines.append("")
        lines.append(
            f"…more available; pass `offset={payload.get('next_offset')}` to continue."
        )
    return "\n".join(lines).rstrip() + "\n"


def _render_stats_markdown(payload: dict[str, Any]) -> str:
    return (
        "# Aelfrice store snapshot\n\n"
        f"- Beliefs: {payload.get('beliefs', 0)}\n"
        f"- Threads: {payload.get('threads', payload.get('edges', 0))}\n"
        f"- Locked: {payload.get('locked', 0)}\n"
        f"- Feedback events: {payload.get('feedback_events', 0)}\n"
        f"- Onboard sessions (total): "
        f"{payload.get('onboard_sessions_total', 0)}\n"
    )


def _render_health_markdown(payload: dict[str, Any]) -> str:
    regime = payload.get("regime", "unknown")
    desc = payload.get("description", "")
    lines = [f"# Store regime: **{regime}**", "", desc, ""]
    if "features" in payload:
        f = payload["features"]
        lines.extend([
            "## Features",
            f"- n_beliefs: {f.get('n_beliefs', 0)}",
            f"- confidence_mean: {f.get('confidence_mean', 0):.3f}",
            f"- confidence_median: {f.get('confidence_median', 0):.3f}",
            f"- mass_mean: {f.get('mass_mean', 0):.3f}",
            f"- lock_per_1000: {f.get('lock_per_1000', 0):.2f}",
            f"- thread_per_belief: "
            f"{f.get('thread_per_belief', f.get('edge_per_belief', 0)):.3f}",
        ])
    if "classification_confidence" in payload:
        lines.append(
            f"\n_Classifier confidence: "
            f"{payload['classification_confidence']:.3f}_"
        )
    return "\n".join(lines).rstrip() + "\n"


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
    store: MemoryStore,
    *,
    query: str,
    budget: int = DEFAULT_TOKEN_BUDGET,
    response_format: str = _RESPONSE_FORMAT_JSON,
) -> dict[str, Any]:
    hits = retrieve(store, query, token_budget=budget)
    payload: dict[str, Any] = {
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
    if response_format == _RESPONSE_FORMAT_MARKDOWN:
        return _wrap_markdown(payload, _render_search_markdown(payload))
    return payload


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


_LOCKED_DEFAULT_LIMIT: Final[int] = 50
_LOCKED_MAX_LIMIT: Final[int] = 500


def tool_locked(
    store: MemoryStore,
    *,
    pressured: bool = False,
    limit: int = _LOCKED_DEFAULT_LIMIT,
    offset: int = 0,
    response_format: str = _RESPONSE_FORMAT_JSON,
) -> dict[str, Any]:
    """List user-locked beliefs with stable cursor pagination.

    `limit` is clamped to [1, _LOCKED_MAX_LIMIT]; `offset` is clamped at
    zero on the low end and unbounded on the high end (returning empty
    when past the end). The full unpaginated count is returned as
    `total` so callers know whether to keep paging.
    """
    locked = store.list_locked_beliefs()
    if pressured:
        locked = [b for b in locked if b.demotion_pressure > 0]
    total = len(locked)
    safe_offset = max(0, offset)
    safe_limit = max(1, min(limit, _LOCKED_MAX_LIMIT))
    page = locked[safe_offset : safe_offset + safe_limit]
    next_offset = safe_offset + len(page)
    has_more = next_offset < total
    payload: dict[str, Any] = {
        "kind": "locked.list",
        "n": len(page),
        "total": total,
        "offset": safe_offset,
        "has_more": has_more,
        "next_offset": next_offset if has_more else None,
        "locked": [
            {
                "id": b.id,
                "content": b.content,
                "demotion_pressure": b.demotion_pressure,
                "locked_at": b.locked_at,
            }
            for b in page
        ],
    }
    if response_format == _RESPONSE_FORMAT_MARKDOWN:
        return _wrap_markdown(payload, _render_locked_markdown(payload))
    return payload


def _tool_apply_scope_change(
    store: MemoryStore,
    belief_id: str,
    to_scope: str,
) -> dict[str, Any]:
    """Flip a belief's scope field and write a zero-valence audit row.

    Shared by tool_demote and tool_validate/tool_promote.
    Returns a result dict with 'kind', 'id', 'scope_updated', and optionally
    'prior_scope', 'new_scope', 'audit_event_id', 'owning_scope', or 'error'.
    """
    from aelfrice.models import validate_belief_scope

    try:
        validate_belief_scope(to_scope)
    except ValueError as e:
        return {
            "kind": "scope.invalid",
            "id": belief_id,
            "scope_updated": False,
            "error": str(e),
        }

    try:
        store.assert_local_ownership(belief_id)
    except ForeignBeliefError as e:
        return {
            "kind": "scope.foreign_belief",
            "id": belief_id,
            "scope_updated": False,
            "owning_scope": e.owning_scope,
            "error": str(e),
        }

    belief = store.get_belief(belief_id)
    if belief is None:
        return {
            "kind": "scope.not_found",
            "id": belief_id,
            "scope_updated": False,
            "error": "belief not found",
        }

    old_scope = belief.scope
    if old_scope == to_scope:
        return {
            "kind": "scope.unchanged",
            "id": belief_id,
            "scope_updated": False,
            "prior_scope": old_scope,
            "new_scope": to_scope,
        }

    belief.scope = to_scope
    store.update_belief(belief)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    audit_id = store.insert_feedback_event(
        belief_id=belief_id,
        valence=0.0,
        source=f"scope:{old_scope}->{to_scope}",
        created_at=ts,
    )
    return {
        "kind": "scope.updated",
        "id": belief_id,
        "scope_updated": True,
        "prior_scope": old_scope,
        "new_scope": to_scope,
        "audit_event_id": audit_id,
    }


def tool_demote(
    store: MemoryStore,
    *,
    belief_id: str,
    to_scope: str | None = None,
) -> dict[str, Any]:
    # --to-scope: orthogonal scope flip; run before tier demotion.
    if to_scope is not None:
        return _tool_apply_scope_change(store, belief_id, to_scope)

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
    to_scope: str | None = None,
) -> dict[str, Any]:
    """Promote agent_inferred -> user_validated. v1.2.0.

    `source` becomes the audit-row source suffix: "promotion:<source>".
    Defaults to "user_validated" so the audit row reads
    "promotion:user_validated" — the canonical wire-format string.

    `to_scope` is an optional scope flip (#689). When supplied, the
    belief's scope field is updated to the given value and a zero-valence
    audit row tagged 'scope:<old>-><new>' is written. Orthogonal to the
    origin flip: both may occur in one call when the belief is
    agent_inferred; only scope changes when the belief is already at the
    user_validated tier (the promote path is idempotent and returns
    'validate.already' without writing a redundant audit row).
    """
    from aelfrice.promotion import promote

    # --to-scope: apply orthogonal scope change first.
    if to_scope is not None:
        scope_result = _tool_apply_scope_change(store, belief_id, to_scope)
        if scope_result["kind"] in (
            "scope.invalid", "scope.foreign_belief", "scope.not_found",
        ):
            return scope_result

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
        # When to_scope drove the call and origin was already validated,
        # include the scope result so the caller sees what changed.
        if to_scope is not None:
            return {
                "kind": "validate.already",
                "id": belief_id,
                "prior_origin": result.prior_origin,
                "new_origin": result.new_origin,
                "scope": scope_result,  # type: ignore[possibly-undefined]
            }
        return {
            "kind": "validate.already",
            "id": belief_id,
            "prior_origin": result.prior_origin,
            "new_origin": result.new_origin,
        }
    out: dict[str, Any] = {
        "kind": "validate.promoted",
        "id": belief_id,
        "prior_origin": result.prior_origin,
        "new_origin": result.new_origin,
        "audit_event_id": result.audit_event_id,
    }
    if to_scope is not None:
        out["scope"] = scope_result  # type: ignore[possibly-undefined]
    return out


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
    except ForeignBeliefError as e:
        # #655: distinguish cross-scope mutation from generic "not found"
        # so MCP callers can branch (e.g. route the user to the owning
        # scope's CLI for the unlock).
        return {
            "kind": "unlock.foreign_belief",
            "id": belief_id,
            "unlocked": False,
            "owning_scope": e.owning_scope,
            "error": str(e),
        }
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
    to_scope: str | None = None,
) -> dict[str, Any]:
    """Promote agent_inferred -> user_validated. Alias of tool_validate.

    Exists as a first-class tool so MCP callers can use the more
    intuitive name. Identical semantics and return shape to
    tool_validate. Accepts optional to_scope (#689).
    """
    return tool_validate(store, belief_id=belief_id, source=source, to_scope=to_scope)


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
    except ForeignBeliefError as exc:
        return {
            "kind": "feedback.foreign_belief",
            "id": belief_id,
            "owning_scope": exc.owning_scope,
            "error": str(exc),
        }
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


def tool_stats(
    store: MemoryStore,
    *,
    response_format: str = _RESPONSE_FORMAT_JSON,
) -> dict[str, Any]:
    n_edges = store.count_edges()
    payload: dict[str, Any] = {
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
    if response_format == _RESPONSE_FORMAT_MARKDOWN:
        return _wrap_markdown(payload, _render_stats_markdown(payload))
    return payload


def tool_health(
    store: MemoryStore,
    *,
    response_format: str = _RESPONSE_FORMAT_JSON,
) -> dict[str, Any]:
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
    if response_format == _RESPONSE_FORMAT_MARKDOWN:
        return _wrap_markdown(payload, _render_health_markdown(payload))
    return payload


def tool_wonder(
    store: MemoryStore,
    *,
    query: str,
    budget: int = 24,
    depth: int = 2,
    agent_count: int = 4,
) -> dict[str, Any]:
    """Research-agent dispatch wonder: gap analysis + research axes.

    Wraps :func:`aelfrice.wonder.dispatch.build_dispatch_payload`. The
    JSON-only return shape is what the skill layer consumes to fan out
    parallel research agents (one per axis); the resulting research
    documents are intended to flow back through ``wonder_ingest``
    (track C of umbrella #542).

    Read-only: no writes to the store, no side-effects. Determinism
    matches :func:`aelfrice.retrieval.retrieve` for the same store
    snapshot and query.
    """
    payload = build_dispatch_payload(
        store, query, budget=budget, depth=depth, agent_count=agent_count,
    )
    out: dict[str, Any] = {"kind": "wonder.axes"}
    out.update(payload.to_dict())
    return out


def tool_wonder_persist(
    store: MemoryStore,
    *,
    query: str,
    budget: int = 24,
    depth: int = 2,
    top: int = 10,
    seed: str | None = None,
) -> dict[str, Any]:
    """Persist BFS phantom candidates to the store via wonder_ingest.

    Runs the same default-mode pipeline as ``aelf wonder --persist``:
    seed selection → BFS expansion → wonder_consolidation scoring →
    top-N phantom construction → ``wonder_ingest``. Returns the insert
    summary so callers know how many beliefs were written.

    Idempotent on the constituent-pair key: a second call with the
    same seed belief produces ``inserted=0, skipped=N``.

    Side-effects: writes ``type='speculative'`` beliefs with
    ``RELATES_TO`` edges and a ``wonder_ingest`` corroboration row.
    """
    from aelfrice.bfs_multihop import expand_bfs
    from aelfrice import wonder_consolidation
    from aelfrice.models import LOCK_USER, Phantom
    from aelfrice.wonder.lifecycle import wonder_ingest

    # Seed selection.
    if seed is not None:
        seed_b = store.get_belief(seed)
        if seed_b is None:
            return {"kind": "wonder.persist", "error": f"seed not found: {seed}"}
    else:
        # Deterministic seed: highest-degree non-locked belief, tie-broken by id asc.
        best_id: str | None = None
        best_degree = -1
        for bid in store.list_belief_ids():
            b = store.get_belief(bid)
            if b is None or b.lock_level == LOCK_USER:
                continue
            degree = len(store.edges_from(bid))
            if degree > best_degree or (
                degree == best_degree
                and best_id is not None
                and bid < best_id
            ):
                best_degree = degree
                best_id = bid
        seed_b = store.get_belief(best_id) if best_id is not None else None

    if seed_b is None:
        return {
            "kind": "wonder.persist",
            "inserted": 0,
            "skipped": 0,
            "edges_created": 0,
            "note": "no eligible seeds",
        }

    hops = expand_bfs([seed_b], store, max_depth=depth, total_budget=top * 2)

    candidates: list[tuple] = []
    for h in hops:
        relatedness = wonder_consolidation.score(seed_b, h.belief)
        combined = h.score * (0.5 + 0.5 * relatedness)
        candidates.append((combined, h))
    candidates.sort(key=lambda r: (-r[0], r[1].belief.id))
    candidates = candidates[:top]

    phantoms = [
        Phantom(
            constituent_belief_ids=(seed_b.id, h.belief.id),
            generator="bfs+wonder_consolidation",
            content=f"{seed_b.content} ⟷ {h.belief.content}",
            score=combined,
        )
        for combined, h in candidates
    ]

    result = wonder_ingest(store, phantoms)
    return {
        "kind": "wonder.persist",
        "inserted": result.inserted,
        "skipped": result.skipped,
        "edges_created": result.edges_created,
    }


def tool_wonder_gc(
    store: MemoryStore,
    *,
    ttl_days: int = 14,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Soft-delete stale speculative beliefs via wonder_gc.

    Wraps :func:`aelfrice.wonder.lifecycle.wonder_gc`. Candidates must
    be ``type='speculative'`` + ``origin=ORIGIN_SPECULATIVE``, still
    active, older than ``ttl_days`` days, with unchanged Bayesian
    priors and no feedback / RESOLVES edges.

    When ``dry_run=True`` no beliefs are mutated. The second non-dry-run
    call finds zero new candidates (idempotent).
    """
    from aelfrice.wonder.lifecycle import wonder_gc

    result = wonder_gc(store, ttl_days=ttl_days, dry_run=dry_run)
    return {
        "kind": "wonder.gc",
        "scanned": result.scanned,
        "deleted": result.deleted,
        "surviving": result.surviving,
    }


# --- FastMCP registration (optional) -----------------------------------
#
# The fastmcp library is an optional [mcp] extra. Importing it lazily
# inside `serve()` lets the rest of this module — and the test suite —
# work without it installed.


def serve() -> None:
    """Start a FastMCP server with the 14 tools registered.

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
    _ResponseFormat = Annotated[
        str,
        Field(
            description=(
                "'json' (default) returns the structured payload the "
                "LLM reads natively. 'markdown' returns a wrapped dict "
                "{kind, format, text} suitable for direct human display."
            ),
            pattern=r"^(json|markdown)$",
        ),
    ]
    _ScopeValue = Annotated[
        str,
        Field(
            description=(
                "Visibility scope for the belief. One of 'project' "
                "(local-only, default), 'global' (any dependent peer), "
                "or 'shared:<name>' where <name> matches [a-z0-9_-]+. "
                "Validated against BELIEF_SCOPE_RE at write time."
            ),
            pattern=r"^project$|^global$|^shared:[a-z0-9_-]+$",
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
        response_format: _ResponseFormat = _RESPONSE_FORMAT_JSON,
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
            return tool_search(
                store,
                query=query,
                budget=budget,
                response_format=response_format,
            )
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
        limit: Annotated[
            int,
            Field(
                description=(
                    "Maximum number of locks to return in this page. "
                    "Clamped to [1, 500]."
                ),
                ge=1,
                le=_LOCKED_MAX_LIMIT,
            ),
        ] = _LOCKED_DEFAULT_LIMIT,
        offset: Annotated[
            int,
            Field(
                description=(
                    "Number of locks to skip for pagination. Use the "
                    "previous response's `next_offset` to keep paging."
                ),
                ge=0,
            ),
        ] = 0,
        response_format: _ResponseFormat = _RESPONSE_FORMAT_JSON,
    ) -> dict[str, Any]:
        """List user-locked (L0) beliefs with cursor pagination.

        Use to show the user their current ground-truth set, to find
        candidates for unlock/demote, or to walk a large lock corpus
        page by page. Read-only.

        Args:
            pressured: If True, return only locks whose demotion_pressure
                is greater than zero (i.e. ones being challenged by
                contradicting evidence). Default False returns all locks.
            limit: Page size. Default 50, max 500.
            offset: Pagination offset. Pass `next_offset` from the prior
                response to advance.

        Returns: {"kind": "locked.list",
                  "n": int (count in this page),
                  "total": int (count across all pages),
                  "offset": int (echoed back),
                  "has_more": bool,
                  "next_offset": int | None,
                  "locked": [{"id": str, "content": str,
                              "demotion_pressure": int,
                              "locked_at": str}, ...]}
        """
        store = _open_default_store()
        try:
            return tool_locked(
                store,
                pressured=pressured,
                limit=limit,
                offset=offset,
                response_format=response_format,
            )
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
    def aelf_demote(
        belief_id: _BeliefId,
        to_scope: _ScopeValue | None = None,
    ) -> dict[str, Any]:
        """Demote a belief one tier — drop a lock OR devalidate.

        For a user-locked (L0) belief: clears the lock to L1.
        For a user_validated belief: drops origin to agent_inferred.
        For other beliefs: no-op. Mutates origin/lock fields.

        When to_scope is supplied, only the scope field is changed
        (orthogonal to tier demotion). Writes a zero-valence audit row
        tagged 'scope:<old>-><new>' to feedback_history. Rejected for
        foreign belief ids.

        Args:
            belief_id: Stable hash-prefix ID returned by aelf_search,
                aelf_lock, or aelf_locked.
            to_scope: Optional target scope ('project', 'global', or
                'shared:<name>'). When supplied, scope is flipped instead
                of performing tier demotion.

        Returns (no to_scope): {"kind": one of [demote.demoted,
                  demote.devalidated, demote.not_locked, demote.not_found],
                  "id": belief_id, "demoted": bool,
                  "tier": str (when devalidated),
                  "error": str (when not_found)}
        Returns (with to_scope): {"kind": one of [scope.updated,
                  scope.unchanged, scope.invalid, scope.foreign_belief,
                  scope.not_found], "id": str, "scope_updated": bool,
                  "prior_scope": str, "new_scope": str,
                  "audit_event_id": int (when updated)}
        """
        store = _open_default_store()
        try:
            return tool_demote(store, belief_id=belief_id, to_scope=to_scope)
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
        to_scope: _ScopeValue | None = None,
    ) -> dict[str, Any]:
        """Alias of aelf_validate. Identical semantics and return shape.

        Exposed under both names so callers can use whichever verb reads
        more naturally for their use case ('promote' for tier transitions,
        'validate' for verification flows).

        When to_scope is supplied, the belief's scope field is also
        updated. Orthogonal: both origin flip and scope flip may occur in
        one call when the belief is agent_inferred; only scope changes
        when the belief is already user_validated.

        Args:
            belief_id: see aelf_validate.
            source: see aelf_validate.
            to_scope: Optional target scope ('project', 'global', or
                'shared:<name>'). When supplied, a zero-valence audit row
                tagged 'scope:<old>-><new>' is written alongside any
                origin-flip audit row.

        Returns: see aelf_validate. When to_scope is supplied, the
            payload includes an additional 'scope' key containing the
            scope-change result dict.
        """
        store = _open_default_store()
        try:
            return tool_promote(
                store, belief_id=belief_id, source=source, to_scope=to_scope,
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
    def aelf_stats(
        response_format: _ResponseFormat = _RESPONSE_FORMAT_JSON,
    ) -> dict[str, Any]:
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
            return tool_stats(store, response_format=response_format)
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
    def aelf_health(
        response_format: _ResponseFormat = _RESPONSE_FORMAT_JSON,
    ) -> dict[str, Any]:
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
            return tool_health(store, response_format=response_format)
        finally:
            store.close()

    @mcp.tool(
        annotations={
            "title": "Wonder: gap analysis + research axes",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    )
    def aelf_wonder(
        query: Annotated[
            str,
            Field(
                description=(
                    "Question or topic to analyse. The store is queried "
                    "for known beliefs; gaps and unresolved contradictions "
                    "are surfaced as research axes for parallel "
                    "research-agent dispatch."
                ),
                min_length=1,
                max_length=500,
            ),
        ],
        budget: Annotated[
            int,
            Field(
                description=(
                    "Maximum candidates pulled from retrieve(). Mapped to "
                    "the L1 limit; bounds the candidate-set size for the "
                    "uncertainty / contradiction sweeps."
                ),
                ge=1,
                le=200,
            ),
        ] = 24,
        depth: Annotated[
            int,
            Field(
                description=(
                    "BFS expansion depth for retrieve(). 0 disables BFS "
                    "(direct hits only); higher values widen the candidate "
                    "graph at the cost of more entries to weigh."
                ),
                ge=0,
                le=4,
            ),
        ] = 2,
        agent_count: Annotated[
            int,
            Field(
                description=(
                    "Hint for how many research agents the skill layer plans "
                    "to fan out. Returned axes are capped at 6 regardless."
                ),
                ge=1,
                le=8,
            ),
        ] = 4,
    ) -> dict[str, Any]:
        """Surface gap analysis + research axes for a query.

        Read-only. Returns:

          - `kind`: "wonder.axes"
          - `gap_analysis`: {query, known_beliefs, high_uncertainty_beliefs,
              unresolved_contradicts_pairs, query_term_coverage, gaps}
          - `research_axes`: list of {name, description, search_hints,
              gap_context} — 2 to 6 entries
          - `agent_count`: int (echoed)
          - `speculative_anchor_ids`: list[str] — the candidate-belief ids,
              passed through for downstream `wonder_ingest` to use as
              `RELATES_TO` targets when persisting research outputs

        The skill layer (E4 of #542) is responsible for spawning agents
        from the axes and routing their outputs through `wonder_ingest`.
        """
        store = _open_default_store()
        try:
            return tool_wonder(
                store, query=query, budget=budget,
                depth=depth, agent_count=agent_count,
            )
        finally:
            store.close()

    @mcp.tool(
        annotations={
            "title": "Wonder persist: ingest BFS phantoms to store",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    )
    def aelf_wonder_persist(
        query: Annotated[
            str,
            Field(
                description=(
                    "Ignored in BFS mode — kept for API symmetry with "
                    "aelf_wonder. Pass the topic you are exploring; "
                    "the actual seed is selected deterministically from "
                    "the store (highest-degree non-locked belief) unless "
                    "seed is provided."
                ),
                min_length=1,
                max_length=500,
            ),
        ],
        budget: Annotated[
            int,
            Field(
                description="BFS expansion budget (total nodes, default 24).",
                ge=1,
                le=200,
            ),
        ] = 24,
        depth: Annotated[
            int,
            Field(
                description="BFS expansion depth (default 2).",
                ge=0,
                le=4,
            ),
        ] = 2,
        top: Annotated[
            int,
            Field(
                description=(
                    "Maximum phantom candidates to build and persist "
                    "(default 10)."
                ),
                ge=1,
                le=50,
            ),
        ] = 10,
        seed: Annotated[
            str | None,
            Field(
                description=(
                    "Explicit seed belief ID. When omitted the "
                    "highest-degree non-locked belief is used."
                ),
            ),
        ] = None,
    ) -> dict[str, Any]:
        """Persist BFS phantom candidates to the store.

        Runs the default-mode pipeline (BFS + wonder_consolidation
        scoring) then calls wonder_ingest to write the top-N candidates
        as ``type='speculative'`` beliefs with ``RELATES_TO`` edges and
        a ``wonder_ingest`` corroboration row. Idempotent on the
        constituent-pair key.

        Destructive (writes beliefs). Returns:

          - `kind`: "wonder.persist"
          - `inserted`: int — new beliefs written
          - `skipped`: int — already-present duplicates
          - `edges_created`: int — RELATES_TO edges written
          - `note`: str (only when no eligible seed)
        """
        store = _open_default_store()
        try:
            return tool_wonder_persist(
                store,
                query=query,
                budget=budget,
                depth=depth,
                top=top,
                seed=seed,
            )
        finally:
            store.close()

    @mcp.tool(
        annotations={
            "title": "Wonder GC: soft-delete stale speculative beliefs",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    )
    def aelf_wonder_gc(
        ttl_days: Annotated[
            int,
            Field(
                description=(
                    "Age threshold in days. Speculative beliefs older "
                    "than this with unchanged priors and no feedback are "
                    "GC candidates (default 14)."
                ),
                ge=1,
                le=365,
            ),
        ] = 14,
        dry_run: Annotated[
            bool,
            Field(
                description=(
                    "When True, reports candidates without mutating the "
                    "store. Use this to preview what would be deleted."
                ),
            ),
        ] = False,
    ) -> dict[str, Any]:
        """Soft-delete stale speculative beliefs.

        Calls wonder_gc(store, ttl_days, dry_run). Destructive when
        ``dry_run=False`` (sets ``valid_to`` on matching beliefs).
        Second call finds zero new candidates (idempotent). Returns:

          - `kind`: "wonder.gc"
          - `scanned`: int — candidates found
          - `deleted`: int — beliefs soft-deleted (0 when dry_run=True)
          - `surviving`: int — candidates not deleted
        """
        store = _open_default_store()
        try:
            return tool_wonder_gc(store, ttl_days=ttl_days, dry_run=dry_run)
        finally:
            store.close()

    # The decorator rebinds each name on `mcp`; the local references
    # below silence pyright's reportUnusedFunction for the bindings
    # themselves. They serve no runtime purpose.
    _registered = (
        aelf_onboard, aelf_search, aelf_lock, aelf_locked,
        aelf_demote, aelf_validate, aelf_unlock, aelf_promote,
        aelf_feedback, aelf_confirm, aelf_stats, aelf_health,
        aelf_wonder, aelf_wonder_persist, aelf_wonder_gc,
    )
    del _registered

    mcp.run()


if __name__ == "__main__":  # pragma: no cover — exercised by `aelf mcp`
    # `python -m aelfrice.mcp_server` is the fallback entry point for
    # hosts that prefer module invocation over the `aelf mcp` console
    # script. Both routes call serve() identically.
    serve()
