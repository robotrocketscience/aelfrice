"""Gemini CLI names and paths shared by the hook adapter and the installer (#1812).

A leaf module on purpose: `aelfrice.hook_gemini` runs the claude-host
hook handlers, and `aelfrice.host_gemini` is called from the CLI, which
those handlers import. Holding the shared names here keeps both modules
off an import cycle through `aelfrice.hook` and `aelfrice.cli`.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Final, Mapping

SESSION_START_EVENT: Final[str] = "SessionStart"
BEFORE_AGENT_EVENT: Final[str] = "BeforeAgent"
AFTER_AGENT_EVENT: Final[str] = "AfterAgent"
PRE_COMPRESS_EVENT: Final[str] = "PreCompress"
AFTER_TOOL_EVENT: Final[str] = "AfterTool"

SAVE_MEMORY_TOOL: Final[str] = "save_memory"

GEMINI_CLI_HOME_ENV: Final[str] = "GEMINI_CLI_HOME"
GEMINI_DIRNAME: Final[str] = ".gemini"
GLOBAL_CONTEXT_FILENAME: Final[str] = "GEMINI.md"


class GeminiHomeError(RuntimeError):
    """`$GEMINI_CLI_HOME` is set to something that is not a directory."""


def resolve_gemini_dir(env: Mapping[str, str] | None = None) -> Path:
    """The directory Gemini keeps its user settings in.

    Gemini resolves `~` to `$GEMINI_CLI_HOME` when that variable is set and
    to the user's home otherwise, and keeps its user files in `.gemini`
    below it. Late-bound, like `host_codex.resolve_codex_home`, so a
    changed `HOME` is seen. A set value that does not name an existing
    directory raises rather than falling back to the home directory: the
    running Gemini would not read the fallback, so a setup that wrote
    there would report success for files nothing loads.
    """
    source = env if env is not None else os.environ
    raw = source.get(GEMINI_CLI_HOME_ENV, "")
    if not raw:
        return Path.home() / GEMINI_DIRNAME
    base = Path(raw).expanduser().absolute()
    if not base.is_dir():
        raise GeminiHomeError(
            f"${GEMINI_CLI_HOME_ENV} is set to {raw!r}, which is not an "
            "existing directory; create it, fix the value, or unset it"
        )
    return base / GEMINI_DIRNAME
