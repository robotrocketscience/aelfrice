"""PreToolUse hook that runs `aelf search` against the per-project belief
store before a `Grep` or `Glob` tool call fires, and emits the results as
`additionalContext` so the agent sees them and can decide to skip / refine
the tool call or use the tool to fill in gaps.

Hook contract (Claude Code PreToolUse):
- payload includes `tool_name`, `tool_input`, `cwd`, plus the standard
  event fields. We act only when:
    * tool_name in {"Grep", "Glob"}
    * tool_input.pattern is a string with at least one extractable token
- All failure modes return exit 0 silently. The hook may NEVER cause a
  `Grep` or `Glob` to feel broken.

Latency budget per docs/search_tool_hook.md:
    median <= 50 ms, p95 <= 200 ms

Tactics:
- Lazy imports of retrieval / store (cold-start dominates).
- Skip on empty token sets up front.
- Cap query at 5 tokens; require 3+ chars per token.
- Lower retrieval budget (token_budget=600, l1_limit=10) than the
  user-facing default — this is auxiliary context.

Local-only: brain-graph reads never cross the git boundary or any
network boundary.
"""
from __future__ import annotations

import json
import re
import sys
import traceback
from typing import IO, Final, cast

QUERY_TOKEN_LIMIT: Final[int] = 5
"""Maximum number of tokens joined into the FTS5 OR query.
Bounds query complexity; longer patterns rarely add useful signal."""

MIN_TOKEN_LEN: Final[int] = 3
"""Minimum token length to be considered a query term.
Filters single-letter regex anchors (\\b, \\d) and 2-char noise."""

INJECTED_TOKEN_BUDGET: Final[int] = 600
"""Token budget for retrieve() — half the user-facing default. Auxiliary
context should not crowd out the user's primary turn budget."""

INJECTED_L1_LIMIT: Final[int] = 10
"""L1 result cap. Lower than retrieval default to bound injection size."""

PER_LINE_CHAR_CAP: Final[int] = 200
"""Truncate each emitted belief line to this many chars. Keeps the
injection bounded even when individual beliefs have long content."""

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(
    rf"[A-Za-z][A-Za-z0-9_-]{{{MIN_TOKEN_LEN - 1},}}"
)


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


def _is_search_tool_call(payload: dict[str, object]) -> bool:
    tool_name = payload.get("tool_name")
    return tool_name in ("Grep", "Glob")


def _extract_query(payload: dict[str, object]) -> str | None:
    """Lift the search query out of tool_input.pattern.

    Both Grep and Glob use the field name `pattern`. Strips regex/glob
    metacharacters by extracting only alphanumeric word tokens, then
    joins the first QUERY_TOKEN_LIMIT with FTS5 ` OR ` to form a query.
    Returns None when no usable tokens are present (pure-glob patterns,
    single-character regexes, etc.) — caller treats None as "skip".
    """
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return None
    raw = cast(dict[str, object], tool_input).get("pattern")
    if not isinstance(raw, str) or not raw.strip():
        return None
    tokens = _TOKEN_RE.findall(raw)
    if not tokens:
        return None
    return " OR ".join(tokens[:QUERY_TOKEN_LIMIT])


def _format_results(
    query: str, beliefs: list[object], locked_ids: set[str]
) -> str:
    """Render retrieve() output as a flat text block for additionalContext.

    Format: "[L0] {id-prefix}: {content}" for locked, "[L1] ..." for
    BM25-ranked. One belief per line, truncated to PER_LINE_CHAR_CAP.
    """
    lines: list[str] = []
    for b in beliefs:
        bid = getattr(b, "id", "") or ""
        content = getattr(b, "content", "") or ""
        if not bid or not content:
            continue
        tier = "L0" if bid in locked_ids else "L1"
        prefix = bid[:16]
        line = f"[{tier}] {prefix}: {content}".replace("\n", " ")
        if len(line) > PER_LINE_CHAR_CAP:
            line = line[: PER_LINE_CHAR_CAP - 3] + "..."
        lines.append(line)
    if not lines:
        return (
            f'<aelfrice-search query="{query}">'
            f"no matching beliefs in store; the tool result will fill the gap"
            f"</aelfrice-search>"
        )
    body = "\n".join(lines)
    return (
        f'<aelfrice-search query="{query}">aelf search ran on this query before '
        f"the tool fires; results:\n{body}\n"
        f"If this answers the question, you may skip the tool call. Otherwise "
        f"use the tool to fill gaps.</aelfrice-search>"
    )


def _emit(stdout: IO[str], context: str) -> None:
    payload = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "additionalContext": context,
        }
    }
    stdout.write(json.dumps(payload))


def _do_search(
    payload: dict[str, object], stdout: IO[str]
) -> None:
    """Core hook body. Returns silently on any non-budget failure.

    Lazy imports keep the cold-start path light: the hook does not pay
    for `aelfrice.store` / `aelfrice.retrieval` import cost on tool
    calls that aren't `Grep` / `Glob`, or on patterns that have no
    extractable tokens.
    """
    if not _is_search_tool_call(payload):
        return
    query = _extract_query(payload)
    if query is None:
        return

    cwd_obj = payload.get("cwd")
    cwd = cwd_obj if isinstance(cwd_obj, str) else None

    # Lazy imports: cold-start cost is paid only when we actually search.
    from aelfrice.cli import db_path  # noqa: PLC0415  # pyright: ignore[reportPrivateUsage]
    from aelfrice.retrieval import retrieve  # noqa: PLC0415
    from aelfrice.store import MemoryStore  # noqa: PLC0415

    p = db_path(cwd=cwd) if _db_path_accepts_cwd(db_path) else db_path()
    if str(p) != ":memory:" and not p.exists():
        # Empty / not-yet-onboarded store — explicit sentinel so the agent
        # learns the check ran.
        _emit(stdout, _format_results(query, [], set()))
        return

    store = MemoryStore(str(p))
    try:
        locked = store.list_locked_beliefs()
        locked_ids = {b.id for b in locked}
        beliefs = retrieve(
            store,
            query,
            token_budget=INJECTED_TOKEN_BUDGET,
            l1_limit=INJECTED_L1_LIMIT,
        )
    finally:
        store.close()

    _emit(stdout, _format_results(query, beliefs, locked_ids))


def _db_path_accepts_cwd(db_path_fn: object) -> bool:
    """Best-effort detection: does aelfrice.cli.db_path() accept a cwd kw?

    v1.1.0 db_path() reads cwd from os.getcwd(); a later patch may add a
    cwd parameter for callers that need to scope to a specific worktree.
    The hook detects either signature without a hard dependency on the
    later API.
    """
    import inspect  # noqa: PLC0415

    try:
        sig = inspect.signature(db_path_fn)  # pyright: ignore[reportArgumentType]
    except (TypeError, ValueError):
        return False
    return "cwd" in sig.parameters


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
        _do_search(payload, sout)
    except Exception:  # non-blocking: surface but never raise
        traceback.print_exc(file=serr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
