"""Session-id inference for ingest entry points.

Per the phantom-prereqs T3 contract (#192), every ingest entry point
that takes an explicit `session_id` parameter must, when the caller
omits it, attempt one inference step before falling back to NULL.

This module is the single source of that step: read the
``AELF_SESSION_ID`` environment variable, warn-once-per-surface to
stderr if unset, and return the resolved value (which may be ``None``).

The two surfaces that consult this helper today are:

  * ``aelfrice.ingest.ingest_turn`` — library-direct ingest path.
  * ``aelfrice.cli._cmd_lock`` — the ``aelf lock`` CLI subcommand.
  * ``aelfrice.mcp_server.tool_lock`` — the MCP ``aelf_lock`` tool.

Surfaces with their own per-call session context (commit-ingest hook,
JSONL replay, onboard scanner/classification) do **not** call this
helper: they have a structurally-correct session_id source already.

Per the R0 ratification (Q5.a) the warn channel is
``print(..., file=sys.stderr)``, matching the existing config-warn
convention. No structured logger; no new dependency.
"""

from __future__ import annotations

import os
import sys

_ENV_VAR = "AELF_SESSION_ID"

# Surfaces that have already emitted their warn line in this process.
# We warn at most once per surface to avoid spamming long-lived
# library callers (e.g. a benchmark that calls ingest_turn in a loop
# without a session_id and without the env var set).
_WARNED: set[str] = set()


def resolve_session_id(
    explicit: str | None,
    *,
    surface_name: str,
) -> str | None:
    """Resolve the session_id for an ingest call site.

    Order:

      1. ``explicit`` if non-empty — caller-passed value wins.
      2. ``AELF_SESSION_ID`` env var if set and non-empty.
      3. ``None`` and a one-shot stderr warn keyed on surface_name.

    The returned ``None`` is propagated to the store layer where it
    becomes a ``NULL`` ``session_id`` column on the inserted row.

    ``surface_name`` is included in the warn line so operators can
    tell which entry point lost the attribution. It is also the key
    used to dedupe warns within a process: a single surface emits its
    warn at most once per process lifetime.
    """
    if explicit:
        return explicit
    env = os.environ.get(_ENV_VAR)
    if env:
        return env
    if surface_name not in _WARNED:
        _WARNED.add(surface_name)
        print(
            f"aelfrice: {surface_name}: no session_id and ${_ENV_VAR} unset; "
            f"writing NULL session attribution",
            file=sys.stderr,
        )
    return None


def _reset_warned_for_tests() -> None:
    """Test-only hook: clear the per-surface warn dedupe set."""
    _WARNED.clear()
