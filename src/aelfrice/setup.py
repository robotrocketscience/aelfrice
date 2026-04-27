"""Claude Code wiring: idempotent install/uninstall of UserPromptSubmit hooks.

This module mutates a Claude Code `settings.json` file (project- or
user-scoped) to add or remove an entry under
`hooks.UserPromptSubmit`, the hook event Claude Code fires every
time a user submits a prompt. Hook stdout is injected as additional
context for the model — that is the channel through which retrieval
is delivered.

The functions here only manipulate JSON; the actual hook script /
command (which calls aelfrice retrieval and prints context) is
supplied by the caller as a `command` string. That keeps this module
purely about "wiring", with no opinion about *what* the hook does.

Schema reference (a single UserPromptSubmit entry, taken from the
Claude Code settings.json contract):

    {
      "hooks": {
        "UserPromptSubmit": [
          {
            "hooks": [
              {
                "type": "command",
                "command": "aelf retrieve --hook",
                "timeout": 5,
                "statusMessage": "Searching aelfrice..."
              }
            ]
          }
        ]
      }
    }

Idempotency contract: two install calls with the same `command`
string produce a settings.json with exactly one matching entry.
Uninstall is keyed off the same `command` string and is a no-op if
no match exists.

Atomicity contract: the on-disk settings.json either reflects the
prior state or the post-mutation state — never a partial write. We
write to a sibling tempfile and `os.replace` it into place.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, cast, overload

SettingsScope = Literal["user", "project"]

USER_SETTINGS_PATH: Final[Path] = Path.home() / ".claude" / "settings.json"
PROJECT_SETTINGS_RELPATH: Final[Path] = Path(".claude") / "settings.json"

_HOOKS_KEY: Final[str] = "hooks"
_EVENT_KEY: Final[str] = "UserPromptSubmit"
_INNER_HOOKS_KEY: Final[str] = "hooks"
_TYPE_KEY: Final[str] = "type"
_COMMAND_KEY: Final[str] = "command"
_TIMEOUT_KEY: Final[str] = "timeout"
_STATUS_MESSAGE_KEY: Final[str] = "statusMessage"
_HOOK_TYPE_COMMAND: Final[str] = "command"


@dataclass(frozen=True)
class InstallResult:
    """Outcome of `install_user_prompt_submit_hook`.

    `installed` is True when the entry was newly written, False when
    an entry with the same command was already present (idempotent
    no-op). `path` is the settings.json path that was inspected.
    """
    path: Path
    installed: bool
    already_present: bool


@dataclass(frozen=True)
class UninstallResult:
    """Outcome of `uninstall_user_prompt_submit_hook`.

    `removed` counts how many entries with the matching command
    were stripped (typically 0 or 1). `path` is the settings.json
    path that was inspected.
    """
    path: Path
    removed: int


def default_settings_path(
    scope: SettingsScope, project_root: Path | None = None
) -> Path:
    """Resolve the conventional Claude Code settings.json path.

    `scope="user"` returns `~/.claude/settings.json`.
    `scope="project"` returns `<project_root>/.claude/settings.json`.
    `project_root` defaults to the current working directory.
    """
    if scope == "user":
        return USER_SETTINGS_PATH
    root = project_root if project_root is not None else Path.cwd()
    return root / PROJECT_SETTINGS_RELPATH


def install_user_prompt_submit_hook(
    settings_path: Path,
    *,
    command: str,
    timeout: int | None = None,
    status_message: str | None = None,
) -> InstallResult:
    """Add a UserPromptSubmit hook entry running `command`.

    Creates `settings_path` (and parent dirs) if missing. Preserves
    every other key in the file. Idempotent: a second call with the
    same `command` is a no-op and returns `already_present=True`.
    """
    if not command:
        raise ValueError("command must be a non-empty string")
    data = _load_settings(settings_path)
    user_prompt_submit = _get_user_prompt_submit_list(data, create=True)
    if _find_entry_index(user_prompt_submit, command) is not None:
        return InstallResult(
            path=settings_path, installed=False, already_present=True
        )
    user_prompt_submit.append(
        _build_entry(
            command=command, timeout=timeout, status_message=status_message
        )
    )
    _atomic_write(settings_path, data)
    return InstallResult(
        path=settings_path, installed=True, already_present=False
    )


def uninstall_user_prompt_submit_hook(
    settings_path: Path, *, command: str
) -> UninstallResult:
    """Remove all UserPromptSubmit entries running exactly `command`.

    Returns `removed=0` if the file does not exist, has no matching
    entry, or has no `hooks.UserPromptSubmit` block at all.
    """
    if not command:
        raise ValueError("command must be a non-empty string")
    if not settings_path.exists():
        return UninstallResult(path=settings_path, removed=0)
    data = _load_settings(settings_path)
    user_prompt_submit = _get_user_prompt_submit_list(data, create=False)
    if user_prompt_submit is None:
        return UninstallResult(path=settings_path, removed=0)
    before = len(user_prompt_submit)
    kept = [
        entry for entry in user_prompt_submit
        if not _entry_matches(entry, command)
    ]
    removed = before - len(kept)
    if removed == 0:
        return UninstallResult(path=settings_path, removed=0)
    user_prompt_submit[:] = kept
    _atomic_write(settings_path, data)
    return UninstallResult(path=settings_path, removed=removed)


# --- internal helpers ---------------------------------------------------


def _load_settings(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(
            f"settings file must contain a JSON object at top level: {path}"
        )
    return cast(dict[str, object], parsed)


@overload
def _get_user_prompt_submit_list(
    data: dict[str, object], *, create: Literal[True]
) -> list[dict[str, object]]: ...


@overload
def _get_user_prompt_submit_list(
    data: dict[str, object], *, create: Literal[False]
) -> list[dict[str, object]] | None: ...


def _get_user_prompt_submit_list(
    data: dict[str, object], *, create: bool
) -> list[dict[str, object]] | None:
    hooks_obj = data.get(_HOOKS_KEY)
    if hooks_obj is None:
        if not create:
            return None
        hooks_obj = {}
        data[_HOOKS_KEY] = hooks_obj
    if not isinstance(hooks_obj, dict):
        raise ValueError(f"'{_HOOKS_KEY}' must be an object")
    hooks_dict = cast(dict[str, object], hooks_obj)
    event_list = hooks_dict.get(_EVENT_KEY)
    if event_list is None:
        if not create:
            return None
        event_list = []
        hooks_dict[_EVENT_KEY] = event_list
    if not isinstance(event_list, list):
        raise ValueError(f"'{_HOOKS_KEY}.{_EVENT_KEY}' must be a list")
    return cast(list[dict[str, object]], event_list)


def _build_entry(
    *, command: str, timeout: int | None, status_message: str | None
) -> dict[str, object]:
    inner: dict[str, object] = {
        _TYPE_KEY: _HOOK_TYPE_COMMAND,
        _COMMAND_KEY: command,
    }
    if timeout is not None:
        inner[_TIMEOUT_KEY] = timeout
    if status_message is not None:
        inner[_STATUS_MESSAGE_KEY] = status_message
    return {_INNER_HOOKS_KEY: [inner]}


def _find_entry_index(
    entries: list[dict[str, object]], command: str
) -> int | None:
    for i, entry in enumerate(entries):
        if _entry_matches(entry, command):
            return i
    return None


def _entry_matches(entry: dict[str, object], command: str) -> bool:
    inner = entry.get(_INNER_HOOKS_KEY)
    if not isinstance(inner, list):
        return False
    for hook in cast(list[object], inner):
        if not isinstance(hook, dict):
            continue
        hook_dict = cast(dict[str, object], hook)
        if (
            hook_dict.get(_TYPE_KEY) == _HOOK_TYPE_COMMAND
            and hook_dict.get(_COMMAND_KEY) == command
        ):
            return True
    return False


def _atomic_write(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(serialized)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
