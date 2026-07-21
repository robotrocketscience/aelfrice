"""PostToolUse hook that mirrors a claude-memory write into the
aelfrice belief graph (#985).

The upstream auto-memory tool ships on by default, writing one-fact
markdown files (``name`` / ``description`` / ``metadata.type``
frontmatter) into ``~/.claude/projects/<encoded>/memory/`` plus a
``MEMORY.md`` index. Without this hook the two stores drift: a fact
written to claude-memory is not in the belief graph unless it is
separately ``aelf lock``'d by hand (``/aelf:audit-claude-memory`` can
*detect* the divergence after the fact, but nothing keeps them in sync
at write time).

This hook is a **one-way mirror** (claude-memory -> aelfrice graph):
on each successful ``Write`` / ``Edit`` whose path is a per-memory fact
file, parse the written frontmatter and ingest the body as a belief.
aelfrice is never authoritative over the memory files.

Mapping (ratified 2026-06-23, #985):
- ``metadata.type`` ``user`` / ``feedback`` -> ``origin=user_validated``,
  undeflated prior.
- ``project`` / ``reference`` / absent -> ``origin=agent_inferred``,
  deflated prior.
- The mirror NEVER locks (L0 stays reserved for explicit ``aelf lock``).

Consent-gated: gated on :func:`claude_memory.is_mirror_enabled` (env
``AELFRICE_MIRROR_CLAUDE_MEMORY`` > explicit kwarg > ``[memory]
mirror_claude_memory`` in ``.aelfrice.toml`` > the #1089 per-project
consent sentinel written by the one-shot reconcile at first ``aelf
setup`` (present ⇒ on) > default **False**). On a set-up project the
sentinel makes the mirror effectively on; an explicit env ``0`` / TOML
``false`` outranks the sentinel and is the opt-out. When the resolved
value is off the hook returns after three cheap checks (tool name, path
shape, flag) and never imports the store.

Hook contract (host PostToolUse):
- payload includes ``tool_name``, ``tool_input``, ``tool_response``,
  ``cwd``. We act only when:
    * ``tool_name`` in {``Write``, ``Edit``, ``MultiEdit``}
    * ``tool_input.file_path`` is a claude-memory fact file
      (``.../.claude/projects/<encoded>/memory/<name>.md``, not
      ``MEMORY.md``)
    * ``tool_response`` is not flagged as an error / interrupted
    * the mirror flag is enabled
- All failure modes return exit 0 silently. The hook may NEVER cause a
  ``Write`` / ``Edit`` to feel broken.

Idempotency: the belief id is content-derived, so a byte-identical
re-write corroborates the existing belief instead of inserting a
duplicate (``insert_or_corroborate``). An edit that changes the body
mints a new belief; supersession of the stale row is left to the
consuming agent, consistent with the narrow-surface PHILOSOPHY (#605).

Local-only: brain-graph writes never cross any network boundary.
"""
from __future__ import annotations

import json
import sys
import traceback
from typing import IO, Final, cast

# Tools whose successful invocation can land a memory fact file on disk.
_WRITE_TOOLS: Final[frozenset[str]] = frozenset({"Write", "Edit", "MultiEdit"})

def _read_payload(stdin: IO[str]) -> dict[str, object] | None:
    raw = stdin.read()
    if not raw.strip():
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return cast("dict[str, object]", parsed)


def _file_path_for_memory_write(payload: dict[str, object]) -> str | None:
    """Return the written file path when ``payload`` is a successful
    Write/Edit of a claude-memory fact file, else None.

    Cheap-reject ordering (cheapest first) keeps the hot path — every
    Write/Edit in every session — close to free when the write is not a
    memory file or the mirror is disabled.
    """
    if payload.get("tool_name") not in _WRITE_TOOLS:
        return None
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return None
    path = cast("dict[str, object]", tool_input).get("file_path")
    if not isinstance(path, str) or not path:
        return None

    # Lazy import: only pay for claude_memory once the tool name matched.
    from aelfrice.claude_memory import is_memory_fact_path  # noqa: PLC0415

    if not is_memory_fact_path(path):
        return None

    tool_response = payload.get("tool_response")
    if isinstance(tool_response, dict):
        resp = cast("dict[str, object]", tool_response)
        if resp.get("isError") is True or resp.get("interrupted") is True:
            return None
    return path


def _do_mirror(payload: dict[str, object]) -> None:
    """Core hook body. Returns silently on any failure."""
    path = _file_path_for_memory_write(payload)
    if path is None:
        return

    from aelfrice.claude_memory import is_mirror_enabled  # noqa: PLC0415

    if not is_mirror_enabled():
        return

    # The mirror is on and the path is a memory fact file: read the
    # written content from disk (authoritative over the tool_input, which
    # for Edit carries only the patch, not the full body).
    from pathlib import Path  # noqa: PLC0415

    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return

    # Lazy imports: the store / derivation cost (pulled in transitively by
    # `claude_memory_reconcile`) is paid only when we actually ingest a
    # memory fact — the hot path (non-memory writes / mirror off) returned
    # above without touching it. `ingest_memory_text` is the single home for
    # the #985 frontmatter -> origin/prior mapping, shared with the #1089
    # reconcile sweep.
    from aelfrice.claude_memory_reconcile import (  # noqa: PLC0415
        ingest_memory_text,
    )
    from aelfrice.db_paths import db_path  # noqa: PLC0415
    from aelfrice.store import MemoryStore  # noqa: PLC0415

    p = db_path()
    if str(p) != ":memory:":
        p.parent.mkdir(parents=True, exist_ok=True)

    store = MemoryStore(str(p))
    try:
        ingest_memory_text(store, text)
    finally:
        store.close()


def main(
    *,
    stdin: IO[str] | None = None,
    stderr: IO[str] | None = None,
) -> int:
    """Hook entry point. Always returns 0 (non-blocking contract)."""
    sin = stdin if stdin is not None else sys.stdin
    serr = stderr if stderr is not None else sys.stderr
    try:
        payload = _read_payload(sin)
        if payload is None:
            return 0
        _do_mirror(payload)
    except Exception:  # non-blocking: surface but never raise
        traceback.print_exc(file=serr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
