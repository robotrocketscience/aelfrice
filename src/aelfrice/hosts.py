"""The agent hosts aelfrice installs into, and the default one (#1812).

`aelf setup`, `unsetup`, `uninstall`, and `doctor` take `--host`, and the
code behind them branches on its value. Before #1812 each parser spelled
the choices and the default itself, and each branch compared against a
string literal, so adding a host meant finding every copy. This module
holds the one copy. A host is added by naming it here and giving each
per-host table in the CLI and the hooks an entry for it.

A leaf module on purpose: it imports only `typing`, so the hook entry
points can read it without loading the installer or the store.
"""
from __future__ import annotations

from typing import Final, cast

CLAUDE_HOST: Final[str] = "claude"
CODEX_HOST: Final[str] = "codex"

DEFAULT_HOST: Final[str] = CLAUDE_HOST
"""The host every `--host` option defaults to. Changing it changes what a
bare `aelf setup` installs, so it stays `claude`."""

SUPPORTED_HOSTS: Final[tuple[str, ...]] = (CLAUDE_HOST, CODEX_HOST)
"""Every value `--host` accepts, in the order help text lists them."""


def host_of(args: object) -> str:
    """The host an argparse namespace names, or the default when it has none.

    Internal callers build namespaces by hand, and some omit `host`; those
    have always meant the default host.
    """
    return cast(str, getattr(args, "host", DEFAULT_HOST))
