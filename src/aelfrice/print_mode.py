"""Recognize a headless host session, and whether to capture it (#1634).

An evaluation harness drives the host headlessly (`claude -p`, or an
Agent SDK app), one scripted prompt per session. Captured, each run's
prompt is stored as if a user had typed it, and one scripted run becomes N
"independent" sessions of corroboration -- which is how a single afternoon
of evaluation runs put 67 beliefs into `aelf core` on one store (#1635).

The host names how a session was started, as its *entrypoint*:

* in the session log, the `entrypoint` field of its user, assistant,
  attachment, and system records;
* in a hook's environment, `CLAUDE_CODE_ENTRYPOINT`.

The headless entrypoints are `sdk-cli` (`claude -p`), `sdk-ts`, and
`sdk-py` (the Agent SDKs). Interactive sessions carry others -- `cli`,
`claude-vscode`, `remote`, `claude-desktop`.

The entrypoint is the one signal, on both paths. The host's
`CLAUDE_CODE_SESSION_ATTENDED=0` is not used: it is also set for
background, daemon, and teammate-agent sessions, which are a user's own
work.

**It is the host's label, and not a guarantee.** A process started
inside a host session can inherit that session's label. On a
non-interactive start (`-p`, or output that is not a terminal), the host
turns an inherited `cli` into `sdk-cli`, but keeps an IDE, desktop, or
`sdk-*` label. So `claude -p` run from a terminal session is skipped, but
`claude -p` run inside an IDE or desktop session can keep that session's
label and be captured, and an interactive session started from an SDK
process can keep `sdk-*` and be skipped. The skip is right for the common
cases -- a harness running `claude -p`, an SDK app -- and a user whose
setup differs sets the override.

Both capture paths skip a headless session by default: `ingest_jsonl`
drops a session-log record with a headless entrypoint, and the transcript
logger records neither the prompt nor the reply when its hook runs under
one. Capture is restorable
for anyone who drives aelfrice through the SDK on purpose:

  1. `AELFRICE_CAPTURE_PRINT_MODE` env var (truthy / falsy);
  2. `[ingest] capture_print_mode` in `.aelfrice.toml`;
  3. default: False (skip).
"""
from __future__ import annotations

import os
import sys
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import IO, Any, Final, cast

from aelfrice.config_discovery import discover_config

ENV_CAPTURE_PRINT_MODE: Final[str] = "AELFRICE_CAPTURE_PRINT_MODE"
SECTION: Final[str] = "ingest"
CAPTURE_KEY: Final[str] = "capture_print_mode"

HEADLESS_ENTRYPOINTS: Final[frozenset[str]] = frozenset(
    {"sdk-cli", "sdk-ts", "sdk-py"}
)
"""Entrypoints the host assigns to a headless start (`-p`, the SDKs)."""

ENV_HOST_ENTRYPOINT: Final[str] = "CLAUDE_CODE_ENTRYPOINT"
"""The entrypoint, as the host sets it in a hook's environment."""

_ENV_TRUTHY: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})
_ENV_FALSY: Final[frozenset[str]] = frozenset({"0", "false", "no", "off"})


def _env_override(env: Mapping[str, str]) -> bool | None:
    raw = env.get(ENV_CAPTURE_PRINT_MODE)
    if raw is None:
        return None
    norm = raw.strip().lower()
    if norm in _ENV_TRUTHY:
        return True
    if norm in _ENV_FALSY:
        return False
    return None


def _read_toml(start: Path | None) -> bool | None:
    """`[ingest] capture_print_mode` from the nearest `.aelfrice.toml`.

    None on a missing file, section, or key, on an unreadable directory
    or file, on TOML that does not parse (nesting too deep to parse
    included), and on a non-bool value. Bytes that are not UTF-8 are
    replaced before parsing, so a stray byte in a comment does not hide
    the key. Never raises:
    both callers run inside hooks.
    """
    serr: IO[str] = sys.stderr
    try:
        candidate = discover_config(start)
    except (OSError, ValueError) as exc:  # ValueError: a NUL in the path
        print(f"aelfrice print_mode: cannot look for .aelfrice.toml: {exc}",
              file=serr)
        return None
    if candidate is None:
        return None
    try:
        parsed: dict[str, Any] = tomllib.loads(
            candidate.read_bytes().decode("utf-8", errors="replace"),
        )
    except (OSError, tomllib.TOMLDecodeError, RecursionError) as exc:
        print(
            f"aelfrice print_mode: cannot read {CAPTURE_KEY} in "
            f"{candidate}: {exc}",
            file=serr,
        )
        return None
    section: Any = parsed.get(SECTION, {})
    if not isinstance(section, dict):
        return None
    val: Any = cast("dict[str, Any]", section).get(CAPTURE_KEY)
    return val if isinstance(val, bool) else None


def is_print_mode_capture_enabled(
    *,
    start: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> bool:
    """Whether headless sessions are captured. Default False.

    The environment variable wins over the TOML key. `env` defaults to
    `os.environ`; pass a mapping to keep a caller off the ambient
    environment.
    """
    env_map = env if env is not None else os.environ
    override = _env_override(env_map)
    if override is not None:
        return override
    toml_value = _read_toml(start)
    if toml_value is not None:
        return toml_value
    return False


def is_print_mode_record(obj: Mapping[str, object]) -> bool:
    """True for a host session-log record from a headless session.

    A non-string `entrypoint` -- a list, say -- is not headless, and never
    raises: `ingest_jsonl` calls this on every record it reads.
    """
    entrypoint = obj.get("entrypoint")
    return isinstance(entrypoint, str) and entrypoint in HEADLESS_ENTRYPOINTS


def is_headless_hook_env(env: Mapping[str, str] | None = None) -> bool:
    """True when the current hook runs in a headless host session."""
    env_map = env if env is not None else os.environ
    return env_map.get(ENV_HOST_ENTRYPOINT) in HEADLESS_ENTRYPOINTS
