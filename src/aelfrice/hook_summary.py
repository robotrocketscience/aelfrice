"""The opt-in summary line a hook shows the user (#1812).

Codex adds a hook's stdout to the model's developer context and doesn't
render it in its terminal UI, so a Codex user can't tell that aelfrice ran.
A hook can show the user text through the `systemMessage` output field,
which Codex surfaces "as a warning in the UI or event stream"
(https://developers.openai.com/codex/hooks, "Common output fields"). JSON
stdout can carry that field together with
`hookSpecificOutput.additionalContext`, the text the model reads.

**Default off.** `AELFRICE_HOOK_SUMMARY_LINE` (env, decisive) then
`[hook] summary_line` in the nearest `.aelfrice.toml`, then off: the
precedence `aelfrice.provenance_render` uses for `[hook]
provenance_render`. With it off, the SessionStart hook writes the bytes it
wrote before this module existed.

A leaf apart from `config_discovery`, and imported only on the Codex path,
so the hook's import graph is unchanged for every other host.
"""
from __future__ import annotations

import json
import os
import sys
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import IO, Any, Final, cast

from aelfrice.config_discovery import discover_config

SECTION: Final[str] = "hook"
SUMMARY_LINE_KEY: Final[str] = "summary_line"
ENV_HOOK_SUMMARY_LINE: Final[str] = "AELFRICE_HOOK_SUMMARY_LINE"

_ENV_TRUTHY: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})
_ENV_FALSY: Final[frozenset[str]] = frozenset({"0", "false", "no", "off"})


def _env_override(env: Mapping[str, str] | None) -> bool | None:
    """True/False if the env var holds a recognised value, else None."""
    src = env if env is not None else os.environ
    raw = src.get(ENV_HOOK_SUMMARY_LINE)
    if raw is None:
        return None
    norm = raw.strip().lower()
    if norm in _ENV_TRUTHY:
        return True
    if norm in _ENV_FALSY:
        return False
    return None


def _read_toml(start: Path | None, stderr: IO[str]) -> bool | None:
    """`[hook] summary_line` from the nearest `.aelfrice.toml`, else None.

    None on a missing file, section, or key, on malformed TOML, and on a
    value that isn't a bool. Never raises: the hook must not break on it.
    """
    candidate = discover_config(start)
    if candidate is None:
        return None
    try:
        parsed: dict[str, Any] = tomllib.loads(
            candidate.read_bytes().decode("utf-8", errors="replace"),
        )
    except (OSError, tomllib.TOMLDecodeError) as exc:
        print(
            f"aelfrice summary_line: cannot read {candidate}: {exc}",
            file=stderr,
        )
        return None
    section_obj: Any = parsed.get(SECTION, {})
    if not isinstance(section_obj, dict):
        return None
    val: Any = cast(dict[str, Any], section_obj).get(SUMMARY_LINE_KEY)
    return val if isinstance(val, bool) else None


def is_summary_line_enabled(
    *,
    start: Path | None = None,
    env: Mapping[str, str] | None = None,
    stderr: IO[str] | None = None,
) -> bool:
    """Resolve the opt-in. Precedence: env > TOML > off.

    `start` is the folder the config walk begins in. The SessionStart hook
    passes the payload's `cwd`, the project the session runs in, because
    the hook process's own cwd is the host's choice.
    """
    override = _env_override(env)
    if override is not None:
        return override
    serr = stderr if stderr is not None else sys.stderr
    from_toml = _read_toml(start, serr)
    return from_toml if from_toml is not None else False


def summary_message(n_beliefs: int, n_locked: int) -> str:
    """The line the user sees, for example `aelfrice: 37 beliefs injected
    (37 locked)`."""
    if n_beliefs <= 0:
        return "aelfrice: ran, nothing injected"
    noun = "belief" if n_beliefs == 1 else "beliefs"
    return f"aelfrice: {n_beliefs} {noun} injected ({n_locked} locked)"


def codex_hook_json(
    event: str, context: str, *, n_beliefs: int, n_locked: int,
) -> str:
    """JSON stdout for a Codex hook: the context plus the summary line.

    `context` is the text the hook prints as plain stdout with the opt-in
    off, carried unchanged as `hookSpecificOutput.additionalContext`.
    Empty context omits `hookSpecificOutput`, so the model gets nothing
    new and the user still sees that the hook ran.
    """
    out: dict[str, object] = {}
    if context:
        out["hookSpecificOutput"] = {
            "hookEventName": event,
            "additionalContext": context,
        }
    out["systemMessage"] = summary_message(n_beliefs, n_locked)
    return json.dumps(out) + "\n"


__all__ = [
    "ENV_HOOK_SUMMARY_LINE",
    "SECTION",
    "SUMMARY_LINE_KEY",
    "codex_hook_json",
    "is_summary_line_enabled",
    "summary_message",
]
