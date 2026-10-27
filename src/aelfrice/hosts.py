"""The agent hosts aelfrice installs into, and the default one (#1812).

`aelf setup`, `unsetup`, `uninstall`, and `doctor` take `--host`, and the
code behind them branches on its value. Before #1812 each parser spelled
the choices and the default itself, and each branch compared against a
string literal, so adding a host meant finding every copy. This module
holds the one copy. A host is added by naming it here and giving each
per-host table in the CLI and the hooks an entry for it.

A leaf module on purpose: it imports only `os` and `typing`, so the hook
entry points can read it without loading the installer or the store.
"""
from __future__ import annotations

import os
from typing import Final, Mapping, cast

CLAUDE_HOST: Final[str] = "claude"
CODEX_HOST: Final[str] = "codex"
GEMINI_HOST: Final[str] = "gemini"

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


HOST_ENV: Final[str] = "AELFRICE_HOST"
"""The environment variable naming the host a hook process runs under.

The Gemini hook adapter sets it in its own process before it calls the
shared hook handlers, so a handler that must act differently on that
host can tell. The value is set in the process and not in the hook
command string: Gemini runs a hook command under PowerShell on Windows,
where a leading `NAME=value` assignment is not valid syntax."""

_KNOWN_HOSTS: Final[frozenset[str]] = frozenset(
    {CLAUDE_HOST, CODEX_HOST, GEMINI_HOST},
)


def host_from_env(env: Mapping[str, str] | None = None) -> str:
    """The host `HOST_ENV` names, or the default when it names none.

    An unset, empty, or unknown value reads as the default host, which is
    what every hook process meant before the variable existed. Reads
    `os.environ` when `env` is None.
    """
    source = env if env is not None else os.environ
    value = source.get(HOST_ENV, "")
    return value if value in _KNOWN_HOSTS else DEFAULT_HOST
