"""Statusline prefix-snippet emitter for Claude Code (and any
shell-driven status bar).

Design (mirrors GSD's gsd-statusline.js):

* Reads the update-check cache only -- NEVER makes a network call.
  Statuslines are invoked frequently; any latency here is visible.
* Emits a *prefix snippet*, not a full statusline. When composed onto
  an existing statusline command via 'original ; aelf statusline
  2>/dev/null' the user's existing bar is unchanged unless an update
  is pending.
* When an update IS pending: emits an orange ANSI-coloured one-liner
  ending in ' | ' so it looks like a leading badge. Empty otherwise.
* Color: truecolor #FFA500 with 256-color (208) and basic-yellow (33)
  fallbacks. NO_COLOR env disables all colour.
"""
from __future__ import annotations

import os
from typing import Final

from aelfrice.lifecycle import (
    UpdateStatus,
    format_update_banner,
    installed_version,
    is_newer,
    read_cache,
)

# CSS orange #FFA500 in truecolor, 256-color 208 (#FF8700), basic 33.
ANSI_RESET: Final[str] = "\x1b[0m"
ANSI_TRUECOLOR_ORANGE: Final[str] = "\x1b[38;2;255;165;0m"
ANSI_256_ORANGE: Final[str] = "\x1b[38;5;208m"
ANSI_BASIC_YELLOW: Final[str] = "\x1b[33m"
SEPARATOR: Final[str] = " │ "
ICON: Final[str] = "⬆"


def _no_color(env: dict[str, str] | None = None) -> bool:
    """Honour the NO_COLOR convention (https://no-color.org).

    Any value (including empty string) for NO_COLOR disables color.
    """
    src = os.environ if env is None else env
    return "NO_COLOR" in src


def _pick_color(
    env: dict[str, str] | None = None,
) -> str:
    """Choose the most-supported orange escape for this terminal.

    Truecolor support is signalled by COLORTERM in {'truecolor',
    '24bit'}. Otherwise we assume 256-color (~all modern terminals).
    Most-basic fallback is yellow (33) which has been universal since
    the 80s. NO_COLOR returns the empty string.
    """
    src = os.environ if env is None else env
    if _no_color(src):
        return ""
    colorterm = src.get("COLORTERM", "").strip().lower()
    if colorterm in {"truecolor", "24bit"}:
        return ANSI_TRUECOLOR_ORANGE
    term = src.get("TERM", "").strip().lower()
    if "256" in term or term in {"xterm", "screen", "tmux"}:
        return ANSI_256_ORANGE
    return ANSI_BASIC_YELLOW


def format_snippet(
    status: UpdateStatus,
    *,
    env: dict[str, str] | None = None,
    installed: str | None = None,
) -> str:
    """Return the statusline snippet for the given cache status.

    Empty when no update is pending. When an update is pending,
    returns 'COLOR_ON⬆ aelfrice X.Y.Z available, run: aelf upgrade
    COLOR_OFF │ '. The trailing ' │ ' is the GSD-style separator
    so the snippet composes naturally as a leading badge.
    """
    if not status.update_available or not status.latest:
        return ""
    running = installed if installed is not None else installed_version()
    if running and not is_newer(status.latest, running):
        return ""
    color_on = _pick_color(env)
    color_off = ANSI_RESET if color_on else ""
    body = format_update_banner(status.latest)
    return f"{color_on}{body}{color_off}{SEPARATOR}"


def render() -> str:
    """Read the cache and return the statusline snippet.

    Convenience wrapper: this is what the CLI subcommand prints.
    """
    return format_snippet(read_cache())
