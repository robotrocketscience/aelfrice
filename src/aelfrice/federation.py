"""Read-only federation primitives (#655).

Per #661 ratification of Option B (read-only federation), the local DB
is the sole writer for its own beliefs. Peers are opened read-only and
their FTS5 results are UNIONed into retrieval. Mutations targeting
foreign belief IDs raise `ForeignBeliefError` at the API surface.

This module provides three things:

1. ``ForeignBeliefError`` — structured error type carrying the foreign
   belief id and the scope name that owns it.
2. ``PeerDep`` and ``load_peer_deps`` — parser for ``knowledge_deps.json``,
   the committed file at the project root that lists peer DB paths.
3. ``open_peer_connection`` — read-only ``sqlite3`` handle on a peer DB
   via ``file:...?mode=ro`` URI.

The committed file uses this shape::

    {
      "version": 1,
      "deps": [
        {"name": "global",      "path": "~/.aelfrice/shared/global/memory.db"},
        {"name": "team-shared", "path": "/path/to/team/aelfrice.db"}
      ]
    }

Path resolution:

* absolute paths are honoured as-is;
* ``~`` expands to ``$HOME``;
* relative paths resolve from the directory of ``knowledge_deps.json``.

Missing peer files are reported via ``PeerDep.reachable=False`` rather
than raising — federation is opportunistic, not fail-closed.
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Final

KNOWLEDGE_DEPS_FILENAME: Final[str] = "knowledge_deps.json"
ENV_KNOWLEDGE_DEPS: Final[str] = "AELFRICE_KNOWLEDGE_DEPS"
SUPPORTED_VERSION: Final[int] = 1


class ForeignBeliefError(ValueError):
    """Raised when a mutation targets a belief owned by a foreign scope.

    Carries ``belief_id`` and ``owning_scope`` so structured error
    responses (MCP tool calls, CLI exit codes) can surface them. Subclass
    of ``ValueError`` so existing ``except ValueError`` blocks in CLI
    surfaces keep working — they will print the message and exit 1.
    """

    def __init__(self, belief_id: str, owning_scope: str) -> None:
        super().__init__(
            f"belief {belief_id!r} is owned by foreign scope "
            f"{owning_scope!r}; mutations require local ownership"
        )
        self.belief_id = belief_id
        self.owning_scope = owning_scope


@dataclass(frozen=True)
class PeerDep:
    """One peer entry from ``knowledge_deps.json``.

    ``name`` is the human-readable scope label used in retrieval output
    annotations (``[scope:<name>]``). ``path`` is the resolved absolute
    path to the peer's SQLite DB file. ``reachable`` is ``True`` when
    the file exists at load time; ``False`` produces a warning + skip
    rather than an open failure.
    """

    name: str
    path: Path
    reachable: bool


def _git_toplevel() -> Path | None:
    """Absolute path of the current git working tree's top-level, or None.

    Mirrors ``db_paths._git_common_dir`` but uses ``--show-toplevel`` —
    ``knowledge_deps.json`` is a committed file, so it belongs in the
    working tree, not under ``.git/``.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    raw = result.stdout.strip()
    return Path(raw).resolve() if raw else None


def resolve_knowledge_deps_path() -> Path | None:
    """Return the path to ``knowledge_deps.json``, or None when unconfigured.

    Resolution order:

    1. ``$AELFRICE_KNOWLEDGE_DEPS`` env override (honoured everywhere).
    2. ``<git-toplevel>/knowledge_deps.json`` when cwd is in a git tree.
    3. ``None`` — caller treats this as "no peers".
    """
    override = os.environ.get(ENV_KNOWLEDGE_DEPS)
    if override:
        return Path(override).expanduser()
    top = _git_toplevel()
    if top is None:
        return None
    return top / KNOWLEDGE_DEPS_FILENAME


def _resolve_dep_path(raw: str, base: Path) -> Path:
    """Resolve a single peer path string against ``base`` (the deps-file dir)."""
    expanded = Path(raw).expanduser()
    if not expanded.is_absolute():
        expanded = (base / expanded).resolve()
    return expanded


def load_peer_deps(deps_path: Path | None = None) -> list[PeerDep]:
    """Parse ``knowledge_deps.json`` and return resolved peer entries.

    ``deps_path=None`` uses ``resolve_knowledge_deps_path()``. A missing
    file (or no configured path) returns ``[]`` — federation is
    opt-in. Malformed JSON, an unsupported ``version``, or invalid dep
    entries raise ``ValueError`` since the config is under user control
    and the user should see the parse error.
    """
    if deps_path is None:
        deps_path = resolve_knowledge_deps_path()
    if deps_path is None or not deps_path.exists():
        return []
    try:
        raw_text = deps_path.read_text("utf-8")
    except OSError as e:
        raise ValueError(f"{deps_path}: cannot read ({e})") from e
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"{deps_path}: invalid JSON ({e.msg})") from e
    if not isinstance(data, dict):
        raise ValueError(f"{deps_path}: top-level must be a JSON object")
    version = data.get("version")
    if version != SUPPORTED_VERSION:
        raise ValueError(
            f"{deps_path}: unsupported version {version!r}; "
            f"expected {SUPPORTED_VERSION}"
        )
    raw_deps = data.get("deps")
    if raw_deps is None:
        return []
    if not isinstance(raw_deps, list):
        raise ValueError(f"{deps_path}: 'deps' must be a list")
    base = deps_path.parent
    out: list[PeerDep] = []
    seen_names: set[str] = set()
    for entry in raw_deps:
        if not isinstance(entry, dict):
            raise ValueError(f"{deps_path}: each dep must be an object")
        name = entry.get("name")
        path_str = entry.get("path")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{deps_path}: dep missing non-empty 'name'")
        if not isinstance(path_str, str) or not path_str:
            raise ValueError(
                f"{deps_path}: dep {name!r} missing non-empty 'path'"
            )
        if name in seen_names:
            raise ValueError(f"{deps_path}: duplicate dep name {name!r}")
        seen_names.add(name)
        resolved = _resolve_dep_path(path_str, base)
        out.append(
            PeerDep(name=name, path=resolved, reachable=resolved.exists())
        )
    return out


def open_peer_connection(path: Path) -> sqlite3.Connection:
    """Open a peer DB file read-only as a fresh ``sqlite3.Connection``.

    Uses the ``file:...?mode=ro&immutable=1`` URI form: read-only at the
    OS level (mode=ro) plus a promise that the file will not change
    during the connection's lifetime (immutable=1) so SQLite skips
    locking. Any attempt to mutate the peer through this handle raises
    ``sqlite3.OperationalError: attempt to write a readonly database``.

    Caller owns ``close()``.
    """
    uri = f"file:{path.resolve()}?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn
