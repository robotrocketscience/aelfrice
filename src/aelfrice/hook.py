"""Claude Code hook entry-points for aelfrice.

This module exposes the script-side half of the v0.7.0 wiring: the
process Claude Code spawns when a `UserPromptSubmit` hook fires. It
reads the JSON event payload from stdin, pulls the user's prompt out
of it, runs aelfrice retrieval against that prompt, and writes the
formatted hits to stdout. Claude Code injects stdout as additional
context above the user's message.

Non-blocking contract: the hook must never fail in a way that
prevents the user's prompt from reaching the model. Every failure
mode (empty payload, malformed JSON, missing prompt field, retrieval
error) returns exit 0 and emits no stdout. Internal exceptions are
written to stderr (Claude Code captures and surfaces these in the
hook log) but do not bubble up.

Output format: a single XML-tag-delimited block. The tag delimiters
are stable; the contents inside are the same per-belief lines the
`aelf search` CLI prints, so a future change to the retrieval format
flows here automatically.
"""
from __future__ import annotations

import json
import sys
import traceback
from typing import IO, Final, cast

from aelfrice.cli import db_path
from aelfrice.models import LOCK_USER, Belief
from aelfrice.retrieval import retrieve
from aelfrice.store import MemoryStore

DEFAULT_HOOK_TOKEN_BUDGET: Final[int] = 1500
"""Conservative default budget for hook-injected context.

Below the CLI default (2000) to leave headroom for the user's
prompt and other concurrent UserPromptSubmit hooks competing for
the same context window.
"""

OPEN_TAG: Final[str] = "<aelfrice-memory>"
CLOSE_TAG: Final[str] = "</aelfrice-memory>"
_PROMPT_KEY: Final[str] = "prompt"


def user_prompt_submit(
    *,
    stdin: IO[str] | None = None,
    stdout: IO[str] | None = None,
    stderr: IO[str] | None = None,
    token_budget: int | None = None,
) -> int:
    """Run the UserPromptSubmit hook. Always returns 0.

    Reads a Claude Code UserPromptSubmit JSON payload from `stdin`,
    runs retrieval against the `prompt` field, and writes the
    formatted output to `stdout`. Streams default to the process
    `sys.stdin`/`sys.stdout`/`sys.stderr`.
    """
    sin = stdin if stdin is not None else sys.stdin
    sout = stdout if stdout is not None else sys.stdout
    serr = stderr if stderr is not None else sys.stderr
    try:
        raw = sin.read()
        prompt = _extract_prompt(raw)
        if prompt is None:
            return 0
        budget = (
            token_budget
            if token_budget is not None
            else DEFAULT_HOOK_TOKEN_BUDGET
        )
        body = _retrieve_and_format(prompt, budget)
        if body:
            sout.write(body)
    except Exception:  # non-blocking: surface but do not fail
        traceback.print_exc(file=serr)
    return 0


def _extract_prompt(raw: str) -> str | None:
    if not raw.strip():
        return None
    try:
        payload = json.loads(raw)  # pyright: ignore[reportAny]
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    payload_typed = cast(dict[str, object], payload)
    prompt = payload_typed.get(_PROMPT_KEY)
    if not isinstance(prompt, str):
        return None
    if not prompt.strip():
        return None
    return prompt


def _retrieve_and_format(prompt: str, token_budget: int) -> str:
    store = _open_store()
    try:
        hits = retrieve(store, prompt, token_budget=token_budget)
    finally:
        store.close()
    if not hits:
        return ""
    return _format_hits(hits)


def _format_hits(hits: list[Belief]) -> str:
    lines: list[str] = [OPEN_TAG]
    for h in hits:
        prefix = "[locked]" if h.lock_level == LOCK_USER else "        "
        lines.append(f"{prefix} {h.id}: {h.content}")
    lines.append(CLOSE_TAG)
    lines.append("")
    return "\n".join(lines)


def _open_store() -> MemoryStore:
    p = db_path()
    if str(p) != ":memory:":
        p.parent.mkdir(parents=True, exist_ok=True)
    return MemoryStore(str(p))


def main() -> int:
    """Entry point for `python -m aelfrice.hook`."""
    return user_prompt_submit()


if __name__ == "__main__":
    sys.exit(main())
