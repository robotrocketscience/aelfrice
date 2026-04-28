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
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, cast, overload

SettingsScope = Literal["user", "project"]

USER_SETTINGS_PATH: Final[Path] = Path.home() / ".claude" / "settings.json"
PROJECT_SETTINGS_RELPATH: Final[Path] = Path(".claude") / "settings.json"
_HOOK_SCRIPT_NAME: Final[str] = "aelf-hook"
_PRE_COMPACT_HOOK_SCRIPT_NAME: Final[str] = "aelf-pre-compact-hook"
# Legacy / pre-pipx shim locations we will silently clean up if they
# point at a deleted target. Keep this list narrow: only paths the
# package itself ever wrote, never user-managed binaries.
_DANGLING_SHIM_CANDIDATES: Final[tuple[Path, ...]] = (
    Path("/usr/local/bin/aelf"),
    Path("/usr/local/bin/aelf-hook"),
)

_HOOKS_KEY: Final[str] = "hooks"
_EVENT_KEY: Final[str] = "UserPromptSubmit"
_PRE_COMPACT_EVENT_KEY: Final[str] = "PreCompact"
_INNER_HOOKS_KEY: Final[str] = "hooks"
_TYPE_KEY: Final[str] = "type"
_COMMAND_KEY: Final[str] = "command"
_TIMEOUT_KEY: Final[str] = "timeout"
_STATUS_MESSAGE_KEY: Final[str] = "statusMessage"
_HOOK_TYPE_COMMAND: Final[str] = "command"
_STATUSLINE_KEY: Final[str] = "statusLine"
STATUSLINE_COMMAND: Final[str] = "aelf statusline"
STATUSLINE_SUFFIX: Final[str] = " ; aelf statusline 2>/dev/null"
# Shell metacharacters that signal a "complex" existing statusline command
# we do NOT want to mutate. If any of these are already in the user's
# command we leave it alone and ask them to compose manually.
_COMPLEX_SHELL_TOKENS: Final[tuple[str, ...]] = (
    "|", "<<", "&&", "||", "`", "\\",
)


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


def _venv_bin_dir() -> Path:
    """Return the directory holding entry-point scripts for the active interpreter.

    Uses `sys.prefix`, which is the venv root for an active virtualenv
    (and the system root otherwise). `sys._base_executable` points at
    the *base* interpreter even inside a venv, so it is unsuitable.
    """
    return Path(sys.prefix) / "bin"


def _executable_in_dir(directory: Path, name: str) -> Path | None:
    """Return `directory/name` if it exists and is executable, else None."""
    candidate = directory / name
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return candidate
    return None


def resolve_hook_command(scope: SettingsScope) -> str:
    """Pick the absolute `aelf-hook` path appropriate for `scope`.

    Project scope: prefer the entry point next to `sys.executable` so
    project-scope settings always pin to that project's venv. This is
    the routing primitive — `~/projects/aelfrice` and
    `~/projects/aelfrice-lab` each get their own venv's `aelf-hook`.

    User scope: prefer `shutil.which("aelf-hook")` so the hook resolves
    to whatever the user has on $PATH (typically a pipx-installed
    `~/.local/bin/aelf-hook`). Fallback chain ensures we never write a
    bare `aelf-hook` that depends on a venv being active later.

    Last-resort fallback is the bare name; tested but emitted only when
    no executable is found anywhere on the system, in which case the
    user has bigger problems and `aelf doctor` will flag it.
    """
    return _resolve_script(_HOOK_SCRIPT_NAME, scope)


def resolve_pre_compact_hook_command(scope: SettingsScope) -> str:
    """Pick the absolute `aelf-pre-compact-hook` path for `scope`.

    Same resolution rules as `resolve_hook_command`; different script
    name. Used by `aelf setup --rebuilder`.
    """
    return _resolve_script(_PRE_COMPACT_HOOK_SCRIPT_NAME, scope)


def _resolve_script(script_name: str, scope: SettingsScope) -> str:
    venv_bin = _venv_bin_dir()
    venv_hook = _executable_in_dir(venv_bin, script_name)
    path_hook_str = shutil.which(script_name)
    path_hook = Path(path_hook_str) if path_hook_str else None
    if scope == "project":
        chosen = venv_hook or path_hook
    else:
        chosen = path_hook or venv_hook
    if chosen is None:
        return script_name
    return str(chosen)


def detect_default_scope(cwd: Path | None = None) -> SettingsScope:
    """Pick `project` if the active interpreter is the venv at `cwd`, else `user`.

    The strict test: `sys.prefix` (the active venv root) must equal
    `<cwd>/.venv`. We deliberately do not `.resolve()` `sys.executable`
    because on uv-managed venvs the `python` shim resolves through to
    the *base* interpreter, which would defeat the comparison. Comparing
    venv roots is the correct primitive.
    """
    project_root = cwd if cwd is not None else Path.cwd()
    venv_dir = project_root / ".venv"
    if not venv_dir.exists():
        return "user"
    try:
        venv_resolved = venv_dir.resolve()
        prefix_resolved = Path(sys.prefix).resolve()
    except OSError:
        return "user"
    if prefix_resolved != venv_resolved:
        return "user"
    return "project"


@dataclass(frozen=True)
class DanglingShimCleanup:
    """Outcome of `clean_dangling_shims`.

    `removed` lists the absolute paths that were unlinked (each was a
    symlink whose target no longer existed). Empty when nothing to do.
    `skipped` lists paths we declined to touch (real files, symlinks
    whose target still exists, or non-symlinks owned by another tool).
    """
    removed: tuple[Path, ...]
    skipped: tuple[Path, ...]


def clean_dangling_shims(
    candidates: tuple[Path, ...] | None = None,
) -> DanglingShimCleanup:
    """Silently remove dangling symlinks the package itself ever wrote.

    Only acts on paths in `_DANGLING_SHIM_CANDIDATES` (or the override
    list for tests). A path is removed iff it is a symlink AND its
    target does not exist. Real files and live symlinks are skipped —
    we never destroy a working install.
    """
    targets = candidates if candidates is not None else _DANGLING_SHIM_CANDIDATES
    removed: list[Path] = []
    skipped: list[Path] = []
    for path in targets:
        try:
            is_symlink = path.is_symlink()
        except OSError:
            skipped.append(path)
            continue
        if not is_symlink:
            if path.exists():
                skipped.append(path)
            continue
        try:
            target_exists = path.resolve(strict=True).exists()
        except (OSError, RuntimeError):
            target_exists = False
        if target_exists:
            skipped.append(path)
            continue
        try:
            path.unlink()
        except OSError:
            skipped.append(path)
            continue
        removed.append(path)
    return DanglingShimCleanup(
        removed=tuple(removed), skipped=tuple(skipped)
    )


def install_user_prompt_submit_hook(
    settings_path: Path,
    *,
    command: str,
    timeout: int | None = None,
    status_message: str | None = None,
) -> InstallResult:
    """Add a UserPromptSubmit hook entry running `command`."""
    return _install_event_hook(
        settings_path,
        event_key=_EVENT_KEY,
        command=command,
        timeout=timeout,
        status_message=status_message,
    )


def install_pre_compact_hook(
    settings_path: Path,
    *,
    command: str,
    timeout: int | None = None,
    status_message: str | None = None,
) -> InstallResult:
    """Add a PreCompact hook entry running `command`.

    Mirrors install_user_prompt_submit_hook for the PreCompact event,
    which Claude Code fires before its default context compaction. The
    aelfrice context-rebuilder lives behind this event; the command
    string typically resolves to `aelf-pre-compact-hook`.
    """
    return _install_event_hook(
        settings_path,
        event_key=_PRE_COMPACT_EVENT_KEY,
        command=command,
        timeout=timeout,
        status_message=status_message,
    )


def _install_event_hook(
    settings_path: Path,
    *,
    event_key: str,
    command: str,
    timeout: int | None,
    status_message: str | None,
) -> InstallResult:
    """Shared install logic for any hook event.

    Creates `settings_path` (and parent dirs) if missing. Preserves
    every other key in the file. Idempotent: a second call with the
    same `command` is a no-op and returns `already_present=True`.
    """
    if not command:
        raise ValueError("command must be a non-empty string")
    data = _load_settings(settings_path)
    entries = _get_event_list(data, event_key, create=True)
    if _find_entry_index(entries, command) is not None:
        return InstallResult(
            path=settings_path, installed=False, already_present=True
        )
    entries.append(
        _build_entry(
            command=command, timeout=timeout, status_message=status_message
        )
    )
    _atomic_write(settings_path, data)
    return InstallResult(
        path=settings_path, installed=True, already_present=False
    )


def uninstall_user_prompt_submit_hook(
    settings_path: Path,
    *,
    command: str | None = None,
    command_basename: str | None = None,
) -> UninstallResult:
    """Remove UserPromptSubmit entries matching `command` or `command_basename`."""
    return _uninstall_event_hook(
        settings_path,
        event_key=_EVENT_KEY,
        command=command,
        command_basename=command_basename,
    )


def uninstall_pre_compact_hook(
    settings_path: Path,
    *,
    command: str | None = None,
    command_basename: str | None = None,
) -> UninstallResult:
    """Remove PreCompact entries matching `command` or `command_basename`."""
    return _uninstall_event_hook(
        settings_path,
        event_key=_PRE_COMPACT_EVENT_KEY,
        command=command,
        command_basename=command_basename,
    )


def _uninstall_event_hook(
    settings_path: Path,
    *,
    event_key: str,
    command: str | None,
    command_basename: str | None,
) -> UninstallResult:
    """Shared uninstall logic for any hook event.

    Pass exactly one of:
      * `command` -- exact-string match of the stored hook command.
      * `command_basename` -- match every hook whose stored command,
        treated as a path, has this basename. Lets a bare-name install
        and an absolute-path install be cleaned up by the same call.

    Returns `removed=0` if the file does not exist, has no matching
    entry, or has no `hooks.<event_key>` block at all.
    """
    if command is None and command_basename is None:
        raise ValueError("provide command or command_basename")
    if command is not None and command_basename is not None:
        raise ValueError("command and command_basename are mutually exclusive")
    if command is not None and not command:
        raise ValueError("command must be a non-empty string")
    if command_basename is not None and not command_basename:
        raise ValueError("command_basename must be a non-empty string")
    if not settings_path.exists():
        return UninstallResult(path=settings_path, removed=0)
    data = _load_settings(settings_path)
    entries = _get_event_list(data, event_key, create=False)
    if entries is None:
        return UninstallResult(path=settings_path, removed=0)
    before = len(entries)
    if command is not None:
        kept = [
            entry for entry in entries
            if not _entry_matches(entry, command)
        ]
    else:
        assert command_basename is not None
        kept = [
            entry for entry in entries
            if not _entry_matches_basename(entry, command_basename)
        ]
    removed = before - len(kept)
    if removed == 0:
        return UninstallResult(path=settings_path, removed=0)
    entries[:] = kept
    _atomic_write(settings_path, data)
    return UninstallResult(path=settings_path, removed=removed)


# --- Transcript-ingest hooks (v1.2+) ------------------------------------

# The four hook events the transcript-logger wires onto. All four use
# the no-matcher `{"hooks": [{type, command}]}` entry shape (Stop and
# the compaction events do not take a matcher).
TRANSCRIPT_INGEST_EVENTS: Final[tuple[str, ...]] = (
    "UserPromptSubmit", "Stop", "PreCompact", "PostCompact",
)
TRANSCRIPT_LOGGER_SCRIPT_NAME: Final[str] = "aelf-transcript-logger"
SESSION_START_EVENT_KEY: Final[str] = "SessionStart"
SESSION_START_HOOK_SCRIPT_NAME: Final[str] = "aelf-session-start-hook"


def resolve_transcript_logger_command(scope: SettingsScope) -> str:
    """Pick the absolute aelf-transcript-logger path for `scope`.

    Same routing primitive as `resolve_hook_command`: project scope
    pins to the venv next to sys.executable; user scope prefers
    $PATH (typically a pipx install).
    """
    venv_bin = _venv_bin_dir()
    venv_hook = _executable_in_dir(venv_bin, TRANSCRIPT_LOGGER_SCRIPT_NAME)
    path_hook_str = shutil.which(TRANSCRIPT_LOGGER_SCRIPT_NAME)
    path_hook = Path(path_hook_str) if path_hook_str else None
    if scope == "project":
        chosen = venv_hook or path_hook
    else:
        chosen = path_hook or venv_hook
    if chosen is None:
        return TRANSCRIPT_LOGGER_SCRIPT_NAME
    return str(chosen)


@dataclass(frozen=True)
class TranscriptIngestInstallResult:
    """Per-event outcome of `install_transcript_ingest_hooks`.

    `installed` lists events where a fresh entry was added. `already`
    lists events where an entry with the same command was already
    present (idempotent no-op).
    """
    path: Path
    installed: tuple[str, ...]
    already: tuple[str, ...]


@dataclass(frozen=True)
class TranscriptIngestUninstallResult:
    """Outcome of `uninstall_transcript_ingest_hooks`.

    `removed` maps each event to how many entries with the matching
    command (or basename) were stripped.
    """
    path: Path
    removed: dict[str, int]


def install_transcript_ingest_hooks(
    settings_path: Path, *, command: str, timeout: int | None = None,
) -> TranscriptIngestInstallResult:
    """Wire the four transcript-logger events to `command`. Idempotent.

    Each event gets its own entry under `hooks.<EventName>`. The same
    `command` is reused across all four events; the logger dispatches
    internally on the `hook_event_name` field of the JSON payload.
    """
    if not command:
        raise ValueError("command must be a non-empty string")
    data = _load_settings(settings_path)
    installed: list[str] = []
    already: list[str] = []
    for event in TRANSCRIPT_INGEST_EVENTS:
        entries = _get_event_list(data, event, create=True)
        if _find_entry_index(entries, command) is not None:
            already.append(event)
            continue
        entries.append(_build_entry(
            command=command, timeout=timeout, status_message=None,
        ))
        installed.append(event)
    if installed:
        _atomic_write(settings_path, data)
    return TranscriptIngestInstallResult(
        path=settings_path,
        installed=tuple(installed),
        already=tuple(already),
    )


def uninstall_transcript_ingest_hooks(
    settings_path: Path, *,
    command: str | None = None,
    command_basename: str | None = None,
) -> TranscriptIngestUninstallResult:
    """Strip transcript-logger entries from all four events. Idempotent.

    Pass exactly one of `command` (exact match) or `command_basename`
    (basename match — lets a bare-name install and an absolute-path
    install be cleaned up by the same call).
    """
    if command is None and command_basename is None:
        raise ValueError("provide command or command_basename")
    if command is not None and command_basename is not None:
        raise ValueError("command and command_basename are mutually exclusive")
    removed: dict[str, int] = {}
    if not settings_path.exists():
        return TranscriptIngestUninstallResult(path=settings_path, removed=removed)
    data = _load_settings(settings_path)
    any_removed = False
    for event in TRANSCRIPT_INGEST_EVENTS:
        entries = _get_event_list(data, event, create=False)
        if entries is None:
            continue
        before = len(entries)
        if command is not None:
            kept = [e for e in entries if not _entry_matches(e, command)]
        else:
            assert command_basename is not None
            kept = [
                e for e in entries
                if not _entry_matches_basename(e, command_basename)
            ]
        n_removed = before - len(kept)
        if n_removed:
            entries[:] = kept
            removed[event] = n_removed
            any_removed = True
    if any_removed:
        _atomic_write(settings_path, data)
    return TranscriptIngestUninstallResult(path=settings_path, removed=removed)


# --- Commit-ingest hook (v1.2+) -----------------------------------------

COMMIT_INGEST_EVENT: Final[str] = "PostToolUse"
COMMIT_INGEST_MATCHER: Final[str] = "Bash"
COMMIT_INGEST_SCRIPT_NAME: Final[str] = "aelf-commit-ingest"


def resolve_commit_ingest_command(scope: SettingsScope) -> str:
    """Pick the absolute aelf-commit-ingest path for `scope`.

    Same routing primitive as `resolve_hook_command`: project scope
    pins to the venv next to sys.executable; user scope prefers
    $PATH (typically a pipx install).
    """
    venv_bin = _venv_bin_dir()
    venv_hook = _executable_in_dir(venv_bin, COMMIT_INGEST_SCRIPT_NAME)
    path_hook_str = shutil.which(COMMIT_INGEST_SCRIPT_NAME)
    path_hook = Path(path_hook_str) if path_hook_str else None
    if scope == "project":
        chosen = venv_hook or path_hook
    else:
        chosen = path_hook or venv_hook
    if chosen is None:
        return COMMIT_INGEST_SCRIPT_NAME
    return str(chosen)


def install_commit_ingest_hook(
    settings_path: Path, *, command: str, timeout: int | None = None,
) -> InstallResult:
    """Add a PostToolUse:matcher=Bash hook entry running `command`.

    Idempotent against the same `command`. Coexists with other Bash-
    matcher PostToolUse entries — appending only after confirming no
    matching entry already exists for the same command.
    """
    if not command:
        raise ValueError("command must be a non-empty string")
    data = _load_settings(settings_path)
    entries = _get_event_list(data, COMMIT_INGEST_EVENT, create=True)
    if _find_entry_index(entries, command) is not None:
        return InstallResult(
            path=settings_path, installed=False, already_present=True,
        )
    entries.append(_build_entry(
        command=command, timeout=timeout, status_message=None,
        matcher=COMMIT_INGEST_MATCHER,
    ))
    _atomic_write(settings_path, data)
    return InstallResult(
        path=settings_path, installed=True, already_present=False,
    )


def uninstall_commit_ingest_hook(
    settings_path: Path, *,
    command: str | None = None,
    command_basename: str | None = None,
) -> UninstallResult:
    """Strip PostToolUse entries matching `command` or `command_basename`.

    Pass exactly one of the two. Other PostToolUse entries (other
    matchers, other tools) are left alone.
    """
    if command is None and command_basename is None:
        raise ValueError("provide command or command_basename")
    if command is not None and command_basename is not None:
        raise ValueError("command and command_basename are mutually exclusive")
    if not settings_path.exists():
        return UninstallResult(path=settings_path, removed=0)
    data = _load_settings(settings_path)
    entries = _get_event_list(data, COMMIT_INGEST_EVENT, create=False)
    if entries is None:
        return UninstallResult(path=settings_path, removed=0)
    before = len(entries)
    if command is not None:
        kept = [e for e in entries if not _entry_matches(e, command)]
    else:
        assert command_basename is not None
        kept = [e for e in entries if not _entry_matches_basename(e, command_basename)]
    removed = before - len(kept)
    if removed == 0:
        return UninstallResult(path=settings_path, removed=0)
    entries[:] = kept
    _atomic_write(settings_path, data)
    return UninstallResult(path=settings_path, removed=removed)


# --- Search-tool wiring -----------------------------------------------


SEARCH_TOOL_EVENT: Final[str] = "PreToolUse"
SEARCH_TOOL_MATCHER: Final[str] = "Grep|Glob"
SEARCH_TOOL_SCRIPT_NAME: Final[str] = "aelf-search-tool-hook"

SEARCH_TOOL_BASH_MATCHER: Final[str] = "Bash"
# The Bash matcher reuses the same entry-point script as the Grep|Glob hook;
# both matchers route into aelfrice.hook_search_tool:main which dispatches
# on tool_name internally.
SEARCH_TOOL_BASH_SCRIPT_NAME: Final[str] = "aelf-search-tool-hook"


def resolve_search_tool_command(scope: SettingsScope) -> str:
    """Pick the absolute aelf-search-tool-hook path for `scope`.

    Same routing primitive as resolve_commit_ingest_command: project
    scope pins to the venv next to sys.executable; user scope prefers
    $PATH (typically a pipx install).
    """
    venv_bin = _venv_bin_dir()
    venv_hook = _executable_in_dir(venv_bin, SEARCH_TOOL_SCRIPT_NAME)
    path_hook_str = shutil.which(SEARCH_TOOL_SCRIPT_NAME)
    path_hook = Path(path_hook_str) if path_hook_str else None
    if scope == "project":
        chosen = venv_hook or path_hook
    else:
        chosen = path_hook or venv_hook
    if chosen is None:
        return SEARCH_TOOL_SCRIPT_NAME
    return str(chosen)


def install_search_tool_hook(
    settings_path: Path, *, command: str, timeout: int | None = None,
) -> InstallResult:
    """Add a PreToolUse:matcher=Grep|Glob hook entry running `command`.

    Idempotent against the same `command`. Coexists with other PreToolUse
    entries — appending only after confirming no matching entry already
    exists for the same command.
    """
    if not command:
        raise ValueError("command must be a non-empty string")
    data = _load_settings(settings_path)
    entries = _get_event_list(data, SEARCH_TOOL_EVENT, create=True)
    if _find_entry_index(entries, command) is not None:
        return InstallResult(
            path=settings_path, installed=False, already_present=True,
        )
    entries.append(_build_entry(
        command=command, timeout=timeout, status_message=None,
        matcher=SEARCH_TOOL_MATCHER,
    ))
    _atomic_write(settings_path, data)
    return InstallResult(
        path=settings_path, installed=True, already_present=False,
    )


def uninstall_search_tool_hook(
    settings_path: Path, *,
    command: str | None = None,
    command_basename: str | None = None,
) -> UninstallResult:
    """Strip PreToolUse entries matching `command` or `command_basename`.

    Pass exactly one of the two. Other PreToolUse entries (other matchers,
    other tools) are left alone.
    """
    if command is None and command_basename is None:
        raise ValueError("provide command or command_basename")
    if command is not None and command_basename is not None:
        raise ValueError("command and command_basename are mutually exclusive")
    if not settings_path.exists():
        return UninstallResult(path=settings_path, removed=0)
    data = _load_settings(settings_path)
    entries = _get_event_list(data, SEARCH_TOOL_EVENT, create=False)
    if entries is None:
        return UninstallResult(path=settings_path, removed=0)
    before = len(entries)
    if command is not None:
        kept = [e for e in entries if not _entry_matches(e, command)]
    else:
        assert command_basename is not None
        kept = [e for e in entries if not _entry_matches_basename(e, command_basename)]
    removed = before - len(kept)
    if removed == 0:
        return UninstallResult(path=settings_path, removed=0)
    entries[:] = kept
    _atomic_write(settings_path, data)
    return UninstallResult(path=settings_path, removed=removed)


def resolve_search_tool_bash_command(scope: SettingsScope) -> str:
    """Pick the absolute aelf-search-tool-hook path for the Bash matcher.

    The Bash-matcher PreToolUse hook reuses the same script as the
    Grep|Glob hook (aelf-search-tool-hook). Both matchers dispatch
    internally on tool_name. Resolution rules mirror resolve_search_tool_command.
    """
    return _resolve_script(SEARCH_TOOL_BASH_SCRIPT_NAME, scope)


def install_search_tool_bash_hook(
    settings_path: Path, *, command: str, timeout: int | None = None,
) -> InstallResult:
    """Add a PreToolUse:matcher=Bash hook entry running `command`.

    Idempotent: a second call with the same `command` and a Bash matcher
    already present is a no-op. Coexists with the Grep|Glob search-tool
    entry — both hooks may share the same command string (the script
    dispatches on tool_name internally) and will be stored as separate
    entries distinguished by their `matcher` field.

    The Bash matcher uses a separate hook entry (different matcher field)
    from the Grep|Glob matcher so they can be installed / removed
    independently per the v1.5.0 spec (§ AC7).
    """
    if not command:
        raise ValueError("command must be a non-empty string")
    data = _load_settings(settings_path)
    entries = _get_event_list(data, SEARCH_TOOL_EVENT, create=True)
    # Idempotency check: look for an existing entry with the same command
    # AND matcher="Bash". An entry with the same command under Grep|Glob
    # is a different hook and must not be conflated.
    already = any(
        e.get("matcher") == SEARCH_TOOL_BASH_MATCHER
        and _entry_matches(e, command)
        for e in entries
    )
    if already:
        return InstallResult(
            path=settings_path, installed=False, already_present=True,
        )
    entries.append(_build_entry(
        command=command, timeout=timeout, status_message=None,
        matcher=SEARCH_TOOL_BASH_MATCHER,
    ))
    _atomic_write(settings_path, data)
    return InstallResult(
        path=settings_path, installed=True, already_present=False,
    )


def uninstall_search_tool_bash_hook(
    settings_path: Path, *,
    command: str | None = None,
    command_basename: str | None = None,
) -> UninstallResult:
    """Strip PreToolUse Bash-matcher entries matching `command` or `command_basename`.

    Pass exactly one of the two. Other PreToolUse entries (Grep|Glob or
    other matchers) are left alone. Since the Bash and Grep|Glob hooks
    share the same script name, uninstall_search_tool_bash_hook only
    removes entries whose matcher field is "Bash" — entries with
    matcher="Grep|Glob" are unaffected.
    """
    if command is None and command_basename is None:
        raise ValueError("provide command or command_basename")
    if command is not None and command_basename is not None:
        raise ValueError("command and command_basename are mutually exclusive")
    if not settings_path.exists():
        return UninstallResult(path=settings_path, removed=0)
    data = _load_settings(settings_path)
    entries = _get_event_list(data, SEARCH_TOOL_EVENT, create=False)
    if entries is None:
        return UninstallResult(path=settings_path, removed=0)
    before = len(entries)
    if command is not None:
        kept = [
            e for e in entries
            if not (
                e.get("matcher") == SEARCH_TOOL_BASH_MATCHER
                and _entry_matches(e, command)
            )
        ]
    else:
        assert command_basename is not None
        kept = [
            e for e in entries
            if not (
                e.get("matcher") == SEARCH_TOOL_BASH_MATCHER
                and _entry_matches_basename(e, command_basename)
            )
        ]
    removed = before - len(kept)
    if removed == 0:
        return UninstallResult(path=settings_path, removed=0)
    entries[:] = kept
    _atomic_write(settings_path, data)
    return UninstallResult(path=settings_path, removed=removed)


# --- SessionStart wiring -----------------------------------------------


def resolve_session_start_hook_command(scope: SettingsScope) -> str:
    """Pick the absolute aelf-session-start-hook path for `scope`.

    Same routing primitive as resolve_hook_command and
    resolve_transcript_logger_command: project scope pins to the venv
    next to sys.executable; user scope prefers $PATH.
    """
    venv_bin = _venv_bin_dir()
    venv_hook = _executable_in_dir(venv_bin, SESSION_START_HOOK_SCRIPT_NAME)
    path_hook_str = shutil.which(SESSION_START_HOOK_SCRIPT_NAME)
    path_hook = Path(path_hook_str) if path_hook_str else None
    if scope == "project":
        chosen = venv_hook or path_hook
    else:
        chosen = path_hook or venv_hook
    if chosen is None:
        return SESSION_START_HOOK_SCRIPT_NAME
    return str(chosen)


def install_session_start_hook(
    settings_path: Path,
    *,
    command: str,
    timeout: int | None = None,
    status_message: str | None = None,
) -> InstallResult:
    """Add a SessionStart hook entry running `command`. Idempotent.

    SessionStart fires once per Claude Code session, before any user
    message. The aelfrice handler injects L0 locked beliefs as the
    session's baseline context.

    Coexists with the UserPromptSubmit and transcript-ingest hooks
    independently; the three events live under separate keys in
    settings.json's hooks block and never disturb each other.
    """
    if not command:
        raise ValueError("command must be a non-empty string")
    data = _load_settings(settings_path)
    entries = _get_event_list(data, SESSION_START_EVENT_KEY, create=True)
    if _find_entry_index(entries, command) is not None:
        return InstallResult(
            path=settings_path, installed=False, already_present=True
        )
    entries.append(
        _build_entry(
            command=command, timeout=timeout, status_message=status_message
        )
    )
    _atomic_write(settings_path, data)
    return InstallResult(
        path=settings_path, installed=True, already_present=False
    )


def uninstall_session_start_hook(
    settings_path: Path,
    *,
    command: str | None = None,
    command_basename: str | None = None,
) -> UninstallResult:
    """Remove SessionStart entries matching `command` or `command_basename`.

    Same exact/basename match semantics as
    uninstall_user_prompt_submit_hook. Returns removed=0 if the file
    does not exist or has no matching entry.
    """
    if command is None and command_basename is None:
        raise ValueError("provide command or command_basename")
    if command is not None and command_basename is not None:
        raise ValueError("command and command_basename are mutually exclusive")
    if command is not None and not command:
        raise ValueError("command must be a non-empty string")
    if command_basename is not None and not command_basename:
        raise ValueError("command_basename must be a non-empty string")
    if not settings_path.exists():
        return UninstallResult(path=settings_path, removed=0)
    data = _load_settings(settings_path)
    entries = _get_event_list(data, SESSION_START_EVENT_KEY, create=False)
    if entries is None:
        return UninstallResult(path=settings_path, removed=0)
    before = len(entries)
    if command is not None:
        kept = [e for e in entries if not _entry_matches(e, command)]
    else:
        assert command_basename is not None
        kept = [
            e for e in entries
            if not _entry_matches_basename(e, command_basename)
        ]
    removed = before - len(kept)
    if removed == 0:
        return UninstallResult(path=settings_path, removed=0)
    entries[:] = kept
    _atomic_write(settings_path, data)
    return UninstallResult(path=settings_path, removed=removed)


# --- Statusline auto-wiring ---------------------------------------------


@dataclass(frozen=True)
class StatuslineInstallResult:
    """Outcome of `install_statusline`.

    `mode` describes what we did:
      'installed'   - settings had no statusLine; we wrote a fresh one.
      'composed'    - settings had a simple existing statusLine; we
                      appended our snippet to its command.
      'already'     - settings already had our statusline (idempotent).
      'skipped'     - existing statusLine was complex (shell
                      metacharacters); we left it alone. The caller
                      should print the documented manual-compose hint.
    """
    path: Path
    mode: Literal["installed", "composed", "already", "skipped"]
    existing_command: str | None = None


@dataclass(frozen=True)
class StatuslineUninstallResult:
    """Outcome of `uninstall_statusline`.

    `mode` describes what we did:
      'removed'   - statusLine was just `aelf statusline`; field deleted.
      'unwrapped' - statusLine was a composed `<orig> ; aelf statusline ...`;
                    we restored the original command.
      'absent'    - no statusLine matched ours; no-op.
    """
    path: Path
    mode: Literal["removed", "unwrapped", "absent"]


def _looks_complex(command: str) -> bool:
    """True iff `command` contains shell tokens we refuse to wrap."""
    return any(tok in command for tok in _COMPLEX_SHELL_TOKENS)


def _has_our_statusline(command: str) -> bool:
    """True iff `command` already invokes our statusline snippet."""
    return STATUSLINE_COMMAND in command


def install_statusline(
    settings_path: Path, *, statusline_command: str = STATUSLINE_COMMAND
) -> StatuslineInstallResult:
    """Add or compose `aelf statusline` into Claude Code's statusLine.

    Deterministic rules:
      * No `statusLine` set -> write fresh `{type, command}`.
      * Already `aelf statusline` -> idempotent no-op.
      * Existing simple command -> append ' ; aelf statusline 2>/dev/null'.
      * Existing complex command -> skip with a 'skipped' result.
    """
    data = _load_settings(settings_path)
    existing = data.get(_STATUSLINE_KEY)
    if existing is None:
        data[_STATUSLINE_KEY] = {
            _TYPE_KEY: _HOOK_TYPE_COMMAND,
            _COMMAND_KEY: statusline_command,
        }
        _atomic_write(settings_path, data)
        return StatuslineInstallResult(
            path=settings_path, mode="installed", existing_command=None
        )
    if not isinstance(existing, dict):
        # Malformed -- leave alone, signal as skipped so caller can warn.
        return StatuslineInstallResult(
            path=settings_path, mode="skipped",
            existing_command=str(existing),
        )
    existing_dict = cast(dict[str, object], existing)
    cmd_obj = existing_dict.get(_COMMAND_KEY)
    if not isinstance(cmd_obj, str):
        return StatuslineInstallResult(
            path=settings_path, mode="skipped", existing_command=None
        )
    if _has_our_statusline(cmd_obj):
        return StatuslineInstallResult(
            path=settings_path, mode="already", existing_command=cmd_obj
        )
    if _looks_complex(cmd_obj):
        return StatuslineInstallResult(
            path=settings_path, mode="skipped", existing_command=cmd_obj
        )
    existing_dict[_COMMAND_KEY] = cmd_obj + STATUSLINE_SUFFIX
    _atomic_write(settings_path, data)
    return StatuslineInstallResult(
        path=settings_path, mode="composed", existing_command=cmd_obj
    )


def uninstall_statusline(
    settings_path: Path, *, statusline_command: str = STATUSLINE_COMMAND
) -> StatuslineUninstallResult:
    """Reverse `install_statusline` surgically.

    Recognises both the standalone form (drop the field) and the
    composed form (strip our suffix, restore the original command).
    Leaves anything else alone.
    """
    if not settings_path.exists():
        return StatuslineUninstallResult(path=settings_path, mode="absent")
    data = _load_settings(settings_path)
    existing = data.get(_STATUSLINE_KEY)
    if not isinstance(existing, dict):
        return StatuslineUninstallResult(path=settings_path, mode="absent")
    existing_dict = cast(dict[str, object], existing)
    cmd_obj = existing_dict.get(_COMMAND_KEY)
    if not isinstance(cmd_obj, str):
        return StatuslineUninstallResult(path=settings_path, mode="absent")
    if cmd_obj == statusline_command:
        del data[_STATUSLINE_KEY]
        _atomic_write(settings_path, data)
        return StatuslineUninstallResult(path=settings_path, mode="removed")
    if cmd_obj.endswith(STATUSLINE_SUFFIX):
        existing_dict[_COMMAND_KEY] = cmd_obj[: -len(STATUSLINE_SUFFIX)]
        _atomic_write(settings_path, data)
        return StatuslineUninstallResult(path=settings_path, mode="unwrapped")
    return StatuslineUninstallResult(path=settings_path, mode="absent")


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
def _get_event_list(
    data: dict[str, object], event: str, *, create: Literal[True]
) -> list[dict[str, object]]: ...


@overload
def _get_event_list(
    data: dict[str, object], event: str, *, create: Literal[False]
) -> list[dict[str, object]] | None: ...


def _get_event_list(
    data: dict[str, object], event: str, *, create: bool
) -> list[dict[str, object]] | None:
    """Return `data['hooks'][event]` as a list, optionally creating it.

    Generic over event name so the same machinery can wire
    UserPromptSubmit, Stop, PreCompact, PostCompact, PreToolUse,
    PostToolUse, etc.
    """
    hooks_obj = data.get(_HOOKS_KEY)
    if hooks_obj is None:
        if not create:
            return None
        hooks_obj = {}
        data[_HOOKS_KEY] = hooks_obj
    if not isinstance(hooks_obj, dict):
        raise ValueError(f"'{_HOOKS_KEY}' must be an object")
    hooks_dict = cast(dict[str, object], hooks_obj)
    event_list = hooks_dict.get(event)
    if event_list is None:
        if not create:
            return None
        event_list = []
        hooks_dict[event] = event_list
    if not isinstance(event_list, list):
        raise ValueError(f"'{_HOOKS_KEY}.{event}' must be a list")
    return cast(list[dict[str, object]], event_list)




def _build_entry(
    *,
    command: str,
    timeout: int | None,
    status_message: str | None,
    matcher: str | None = None,
) -> dict[str, object]:
    inner: dict[str, object] = {
        _TYPE_KEY: _HOOK_TYPE_COMMAND,
        _COMMAND_KEY: command,
    }
    if timeout is not None:
        inner[_TIMEOUT_KEY] = timeout
    if status_message is not None:
        inner[_STATUS_MESSAGE_KEY] = status_message
    entry: dict[str, object] = {_INNER_HOOKS_KEY: [inner]}
    if matcher is not None:
        # PreToolUse / PostToolUse entries gate the inner hooks on a
        # tool-name matcher. UserPromptSubmit / Stop / PreCompact /
        # PostCompact entries omit the field entirely.
        entry["matcher"] = matcher
    return entry


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


def _entry_matches_basename(
    entry: dict[str, object], basename: str
) -> bool:
    """True iff any inner command's first whitespace-stripped path token has `basename`."""
    inner = entry.get(_INNER_HOOKS_KEY)
    if not isinstance(inner, list):
        return False
    for hook in cast(list[object], inner):
        if not isinstance(hook, dict):
            continue
        hook_dict = cast(dict[str, object], hook)
        if hook_dict.get(_TYPE_KEY) != _HOOK_TYPE_COMMAND:
            continue
        cmd = hook_dict.get(_COMMAND_KEY)
        if not isinstance(cmd, str):
            continue
        # First whitespace token is the program; basename match against it.
        first = cmd.strip().split(maxsplit=1)[0] if cmd.strip() else ""
        if first and Path(first).name == basename:
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
