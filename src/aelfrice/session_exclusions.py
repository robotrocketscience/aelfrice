"""Session-scoped retrieval exclusion patterns (#856).

A user invokes ``aelf scope-out <pattern>`` (slash form
``/aelf:scope-out <pattern>``) to add a substring pattern to the active
session's exclusion list. The UserPromptSubmit memory hook reads the list
on every fire and drops any retrieved belief whose content contains a
listed pattern (case-insensitive literal substring).

Storage: ``<git-common-dir>/aelfrice/session_exclusions.json``, sibling
of ``session_first_prompt.json``. Format::

    {"session_id": "<sid>", "patterns": ["substr1", "substr2"]}

Session boundary: the file is keyed by ``session_id``. When the hook
reads with a different ``current_session_id`` (i.e. a new session has
started) ``load_exclusions`` returns ``[]``. The next ``save_exclusions``
overwrites the file with the new session's id. No GC: the file is a few
hundred bytes at most.

Matching: literal substring, case-insensitive. Regex is intentionally
NOT supported. Aelfrice's retrieval property is deterministic
stdlib-only (locked belief: "avoid embeddings + non-determinism in
retrieval"); substring stays well inside that line. A future ``--regex``
flag can extend the surface if a concrete need surfaces.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Final

EXCLUSIONS_FILENAME: Final[str] = "session_exclusions.json"
SESSION_STATE_FILENAME: Final[str] = "session_first_prompt.json"


def exclusions_path(state_dir: Path) -> Path:
    return state_dir / EXCLUSIONS_FILENAME


def session_state_path(state_dir: Path) -> Path:
    return state_dir / SESSION_STATE_FILENAME


def read_current_session_id(state_dir: Path) -> str | None:
    """Return the session_id stored in session_first_prompt.json, or None.

    Used by the CLI to discover the active session_id without taking it
    on argv. The hook writes this file on every first prompt of a new
    session, so it is current by the time any slash command shells out.
    """
    path = session_state_path(state_dir)
    try:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        sid = data.get("session_id")
        return sid if isinstance(sid, str) and sid.strip() else None
    except Exception:
        return None


def load_exclusions(
    path: Path, current_session_id: str | None
) -> list[str]:
    """Return the active exclusion patterns for ``current_session_id``.

    Returns ``[]`` when:
    - ``current_session_id`` is None or blank
    - the file does not exist
    - the stored ``session_id`` does not match ``current_session_id``
    - the file is malformed JSON or unreadable

    Fail-soft: never raises.
    """
    if not current_session_id or not current_session_id.strip():
        return []
    try:
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return []
        if data.get("session_id") != current_session_id:
            return []
        patterns = data.get("patterns", [])
        if not isinstance(patterns, list):
            return []
        return [p for p in patterns if isinstance(p, str) and p]
    except Exception:
        return []


def save_exclusions(
    path: Path, session_id: str, patterns: list[str]
) -> None:
    """Atomically write the exclusion file. Fail-soft."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        deduped = list(dict.fromkeys(p for p in patterns if p))
        payload = json.dumps({
            "session_id": session_id,
            "patterns": deduped,
        })
        fd, tmp_name = tempfile.mkstemp(
            prefix=path.name + ".",
            suffix=".tmp",
            dir=str(path.parent),
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
    except Exception:
        # Fail-soft per docstring contract: save_exclusions must not
        # raise. Hook retrieval reads this file every turn; a write
        # failure (disk full, perms, ENOSPC) cannot break the session.
        # The user can re-issue `aelf scope-out` if the persisted set
        # is stale.
        pass


def add_exclusion(
    path: Path, session_id: str, pattern: str
) -> list[str]:
    """Append ``pattern`` for ``session_id``; return the updated list.

    Idempotent: a pattern already present is not re-added.
    """
    current = load_exclusions(path, session_id)
    if pattern in current:
        return current
    updated = current + [pattern]
    save_exclusions(path, session_id, updated)
    return updated


def clear_exclusions(path: Path, session_id: str) -> None:
    """Empty the patterns list, keeping the session_id key."""
    save_exclusions(path, session_id, [])


def is_excluded(content: str, patterns: list[str]) -> bool:
    """Return True iff ``content`` matches any pattern (case-insensitive substring)."""
    if not patterns:
        return False
    lowered = content.lower()
    return any(p.lower() in lowered for p in patterns if p)
