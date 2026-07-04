"""PreToolUse hook that carries belief-store context into dispatched
subagent prompts (#1068).

Dispatched subagents receive none of the parent
session's injection lanes — no SessionStart baseline, no per-prompt
retrieval, no locked constraints. This hook closes that gap: on an
Agent dispatch it rewrites ``tool_input.prompt`` through the PreToolUse
``updatedInput`` channel, prepending a bounded, tagged block containing
the L0 locked beliefs plus hits relevant to the worker's prompt.

Hook contract (probed live on the target harness, 2026-07-03 — see
issue #1068 for the probe record):

- payload: ``tool_name == "Agent"`` (``"Task"`` on older harnesses);
  ``tool_input`` carries ``prompt`` plus dispatch metadata, snake_case.
- Emitting ``{"hookSpecificOutput": {"hookEventName": "PreToolUse",
  "updatedInput": {...}}}`` replaces the tool input before dispatch.
  ``permissionDecision`` is deliberately NOT emitted — the probe
  confirmed ``updatedInput`` applies without it, and emitting
  ``"allow"`` would silently bypass stricter permission modes.
- ``SubagentStart`` cannot serve this lane: its payload carries no
  prompt text, so it cannot do query-aware retrieval.

Behavior contract:

- All failure modes return exit 0 with **no stdout** — the harness
  proceeds with the original tool input. Dispatch may never feel
  broken (same non-blocking posture as ``hook_search_tool``).
- Idempotent: a prompt already carrying the worker-context open tag
  (nested dispatch, retried tool call) passes through unchanged.
- Deterministic: L0 lock listing + the existing ``retrieve()`` path
  only; no new ranking machinery (#605).

Latency: the block is built from one ``retrieve()`` call at a reduced
auxiliary budget (same convention as the Grep|Glob lane); dispatch
itself — spawning a whole worker session — dwarfs the hook cost.

Local-only: brain-graph reads never cross the git boundary or any
network boundary.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from typing import IO, Final, cast

AGENT_TOOL_NAMES: Final[tuple[str, ...]] = ("Agent", "Task")
"""Exact tool names treated as a subagent dispatch. "Agent" is the
current harness name; "Task" is the legacy name on older harnesses.
Exact membership (not substring) so TaskCreate/TaskUpdate/etc. never
match even if the settings matcher over-matches."""

WORKER_CONTEXT_OPEN_TAG: Final[str] = "<aelfrice-worker-context>"
WORKER_CONTEXT_CLOSE_TAG: Final[str] = "</aelfrice-worker-context>"

INJECTED_TOKEN_BUDGET: Final[int] = 600
"""Token budget for retrieve() — same reduced auxiliary allowance as the
Grep|Glob search-tool lane. The worker's own task prompt is the primary
content; memory context must not crowd it out."""

INJECTED_L1_LIMIT: Final[int] = 10
"""L1 result cap, mirroring the Grep|Glob lane."""

QUERY_CHAR_CAP: Final[int] = 2000
"""Cap on how much of the worker prompt feeds the retrieval query.
Bounds FTS5 query-construction cost on very long dispatch prompts;
the opening of a task prompt carries its topical signal."""

ENV_AGENT_CONTEXT: Final[str] = "AELFRICE_AGENT_CONTEXT"
"""Kill switch: set to 0/false/no/off to disable injection entirely
(byte-identical passthrough). Unset or any other value = enabled."""

_DISABLED_VALUES: Final[frozenset[str]] = frozenset({"0", "false", "no", "off"})


def _is_disabled() -> bool:
    return os.environ.get(ENV_AGENT_CONTEXT, "").strip().lower() in _DISABLED_VALUES


def _read_payload(stdin: IO[str]) -> dict[str, object] | None:
    raw = stdin.read()
    if not raw.strip():
        return None
    try:
        parsed = json.loads(raw)  # pyright: ignore[reportAny]
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return cast(dict[str, object], parsed)


def _extract_dispatch(
    payload: dict[str, object],
) -> tuple[dict[str, object], str] | None:
    """Return ``(tool_input, prompt)`` for an Agent dispatch, else None.

    None means "not our tool call / nothing to inject into" — the caller
    passes through silently.
    """
    if payload.get("tool_name") not in AGENT_TOOL_NAMES:
        return None
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return None
    ti = cast(dict[str, object], tool_input)
    prompt = ti.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return None
    return ti, prompt


def _build_block(hits: list[object]) -> str:
    """Render hits as the tagged worker-context block.

    Reuses the UserPromptSubmit formatters (`_split_belief_lines` /
    `_manifest_block_lines` and the trust-tier framing header) so the
    worker sees the same belief shape — including #1016-B reference-lock
    manifest bounding and #1037 envelope escaping — as a parent session.
    """
    from aelfrice.hook import (  # noqa: PLC0415
        _FRAMING_HEADER,
        _manifest_block_lines,
        _split_belief_lines,
    )
    belief_lines, manifest_lines = _split_belief_lines(hits)  # type: ignore[arg-type]
    lines: list[str] = [WORKER_CONTEXT_OPEN_TAG, _FRAMING_HEADER]
    lines.extend(belief_lines)
    lines.extend(_manifest_block_lines(manifest_lines))
    lines.append(WORKER_CONTEXT_CLOSE_TAG)
    return "\n".join(lines)


def _emit(
    stdout: IO[str], tool_input: dict[str, object], new_prompt: str,
) -> None:
    """Write the updatedInput payload replacing only the prompt field.

    No ``permissionDecision``: the probe confirmed ``updatedInput``
    applies without one, and this hook must not alter the user's
    permission flow for the dispatch.
    """
    updated: dict[str, object] = dict(tool_input)
    updated["prompt"] = new_prompt
    payload = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "updatedInput": updated,
        }
    }
    stdout.write(json.dumps(payload))


def _db_path_accepts_cwd(db_path_fn: object) -> bool:
    """Best-effort detection: does db_path() accept a cwd kw?

    Same posture as hook_search_tool: detect either signature without a
    hard dependency on the later API.
    """
    import inspect  # noqa: PLC0415

    try:
        sig = inspect.signature(db_path_fn)  # pyright: ignore[reportArgumentType]
    except (TypeError, ValueError):
        return False
    return "cwd" in sig.parameters


def _do_inject(
    payload: dict[str, object],
    stdout: IO[str],
    stderr: IO[str] | None = None,
) -> None:
    """Core hook body. Returns silently (passthrough) on any miss.

    Lazy imports keep the cold-start path light: retrieval / store
    import cost is paid only on an actual Agent dispatch with a
    resolvable store.
    """
    if _is_disabled():
        return
    dispatch = _extract_dispatch(payload)
    if dispatch is None:
        return
    tool_input, prompt = dispatch
    if WORKER_CONTEXT_OPEN_TAG in prompt:
        # Already injected (nested dispatch or retried call) — never
        # stack a second block.
        return

    cwd_obj = payload.get("cwd")
    cwd = cwd_obj if isinstance(cwd_obj, str) else None
    session_obj = payload.get("session_id")
    session_id = session_obj if isinstance(session_obj, str) else None

    # Guard against stale installs missing a runtime dep (issue #236).
    try:
        from aelfrice.db_paths import db_path  # noqa: PLC0415
        from aelfrice.retrieval import retrieve  # noqa: PLC0415
        from aelfrice.store import MemoryStore  # noqa: PLC0415
    except ImportError as _ie:
        missing = getattr(_ie, "name", None) or str(_ie)
        import sys as _sys  # noqa: PLC0415
        print(
            f"aelf-agent-context-hook: install incomplete (missing "
            f"{missing}); skipping",
            file=_sys.stderr,
        )
        return

    p = db_path(cwd=cwd) if _db_path_accepts_cwd(db_path) else db_path()
    if str(p) != ":memory:" and not p.exists():
        # No store: passthrough. Unlike the search-tool lane there is no
        # value in a "no beliefs" sentinel — the worker cannot skip its
        # dispatch, so an empty block would be pure noise.
        return

    store = MemoryStore(str(p))
    try:
        hits = retrieve(
            store,
            prompt[:QUERY_CHAR_CAP],
            token_budget=INJECTED_TOKEN_BUDGET,
            l1_limit=INJECTED_L1_LIMIT,
            # #1016-B: reference-tier locks render as a bounded manifest
            # line in this lane too, so budget them at manifest size.
            manifest_reference_locks=True,
        )
        # Same post-retrieve filters as the UserPromptSubmit lane: the
        # worker inherits the parent session's project-context scoping
        # (#858) and session scope-outs (#856). Intra-package private
        # reuse, same precedent as hook_search_tool -> retrieval._lock_topic.
        from aelfrice.hook import (  # noqa: PLC0415
            _filter_by_project_context,
            _filter_session_exclusions,
        )
        hits = _filter_by_project_context(hits)
        hits = _filter_session_exclusions(hits, session_id)
        if not hits:
            return
        # Close the feedback loop for this lane like the UPS lane does:
        # exposure-valence audit rows, best-effort.
        from aelfrice.hook_search import record_retrieval  # noqa: PLC0415
        record_retrieval(store, hits, stderr=stderr)
    finally:
        store.close()

    block = _build_block(list(hits))
    _emit(stdout, tool_input, f"{block}\n\n{prompt}")


def main(
    *,
    stdin: IO[str] | None = None,
    stdout: IO[str] | None = None,
    stderr: IO[str] | None = None,
) -> int:
    """Hook entry point. Always returns 0 (non-blocking contract)."""
    sin = stdin if stdin is not None else sys.stdin
    sout = stdout if stdout is not None else sys.stdout
    serr = stderr if stderr is not None else sys.stderr
    try:
        payload = _read_payload(sin)
        if payload is None:
            return 0
        _do_inject(payload, sout, stderr=serr)
    except Exception:  # non-blocking: surface but never raise
        traceback.print_exc(file=serr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
