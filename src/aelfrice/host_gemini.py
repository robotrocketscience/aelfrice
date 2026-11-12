"""Gemini CLI host target for setup, doctor, and unsetup (#1812).

Writes, checks, and removes two things under Gemini's user directory,
`~/.gemini` (or `$GEMINI_CLI_HOME/.gemini`, see
`hook_gemini.resolve_gemini_dir`):

- **Hooks in `settings.json`.** One handler per Gemini event aelfrice uses,
  each running `aelf-gemini-hook` (see `aelfrice.hook_gemini` for what it
  runs). The command is the absolute path resolved on this machine when
  setup runs, quoted for the shell Gemini runs it under, and it goes into
  the user settings file only, so no machine path is written into a
  project. Gemini reads hook `timeout` in milliseconds. Each handler carries a `name`, which Gemini's `/hooks
  disable` takes.
- **Custom commands as TOML.** Every bundled `/aelf:*` slash command is
  generated into `commands/aelf/<name>.toml`, which Gemini invokes as
  `/aelf:<name>`. Nothing generated is kept in the repository.

The posture follows the Codex host (#1052). The settings file is shared
with the user and other tools, so the merge is per handler: setup replaces
only handlers whose command is `aelf-gemini-hook` and keeps every other
key, group, and handler. A file that is not a JSON object, or that holds a
non-object `hooks` or a non-list value at an event setup writes, is
refused unless `--force` is given. Gemini settings may hold comments,
which this JSON reader cannot parse, so such a file is refused too.
Writes take an advisory lock kept in the temporary directory, so no lock
file is left in Gemini's directory, re-check a content hash before
committing, and replace the file atomically.
Unsetup keeps the settings file, as `{}` when nothing else is left.
Generated commands carry a marker line, and only files with the marker
are replaced or removed.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, cast

from aelfrice.gemini_paths import (
    AFTER_AGENT_EVENT,
    BEFORE_AGENT_EVENT,
    PRE_COMPRESS_EVENT,
    SESSION_START_EVENT,
    resolve_gemini_dir,
)
from aelfrice.launcher import (
    command_program_keys,
    command_tokens,
    launcher_key,
    owned_keys,
    program_exists,
    which_on_path,
)
from aelfrice.setup import (
    GEMINI_HOOK_SCRIPT_NAME,
    bundled_slash_files,
    resolve_gemini_hook_command,
)

SETTINGS_FILENAME: Final[str] = "settings.json"
COMMANDS_DIRNAME: Final[str] = "commands"
COMMAND_NAMESPACE: Final[str] = "aelf"
COMMAND_SUFFIX: Final[str] = ".toml"

#: Gemini event -> (hook name, matcher, timeout in milliseconds). A
#: lifecycle matcher of "*" matches every trigger. Each timeout is the sum
#: of the claude-host timeouts, in seconds, of the handlers the event runs
#: in sequence. No tool event is wired.
HOOK_SPECS: Final[Mapping[str, tuple[str, str, int]]] = {
    SESSION_START_EVENT: ("aelfrice-session-start", "*", 15_000),
    BEFORE_AGENT_EVENT: ("aelfrice-prompt", "*", 45_000),
    AFTER_AGENT_EVENT: ("aelfrice-turn-end", "*", 60_000),
    PRE_COMPRESS_EVENT: ("aelfrice-pre-compress", "*", 30_000),
}

_OWNED_BASENAMES: Final[frozenset[str]] = frozenset({GEMINI_HOOK_SCRIPT_NAME})


def gemini_settings_path(gemini_dir: Path | None = None) -> Path:
    gdir = gemini_dir if gemini_dir is not None else resolve_gemini_dir()
    return gdir / SETTINGS_FILENAME


def gemini_commands_dir(gemini_dir: Path | None = None) -> Path:
    gdir = gemini_dir if gemini_dir is not None else resolve_gemini_dir()
    return gdir / COMMANDS_DIRNAME / COMMAND_NAMESPACE


# --- the hook command as Gemini's shell reads it ----------------------------------

#: A path made only of these characters is one bare word to PowerShell.
_POWERSHELL_BARE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9_.:\\/~+-]+")

#: PowerShell runs a quoted path only after its call operator.
_POWERSHELL_CALL: Final[str] = "& "


def _is_windows(windows: bool | None) -> bool:
    return os.name == "nt" if windows is None else windows


def gemini_hook_command(program: str, *, windows: bool | None = None) -> str:
    """`program` as a hook command for the shell Gemini runs it under.

    Gemini runs a hook's command string under `bash -c` on macOS and Linux
    and under PowerShell on Windows, so a path with a space, written bare,
    runs only the part before the space. A path the shell would split is
    quoted for that shell: in single quotes on POSIX, and in single quotes
    after the call operator on Windows. Any other path is written as is.
    """
    if not _is_windows(windows):
        return shlex.quote(program)
    if _POWERSHELL_BARE.fullmatch(program):
        return program
    return _POWERSHELL_CALL + "'" + program.replace("'", "''") + "'"


def shell_program(command: str, *, windows: bool | None = None) -> str:
    """The program Gemini's shell runs for `command`; '' when there is none.

    Reads `gemini_hook_command` back. It does not rejoin an unquoted path
    that a space splits, since the shell does not either.
    """
    stripped = command.strip()
    if not _is_windows(windows):
        try:
            tokens = shlex.split(stripped)
        except ValueError:
            return ""
        return tokens[0] if tokens else ""
    opener = _POWERSHELL_CALL + "'"
    if stripped.startswith(opener) and stripped.endswith("'") and len(stripped) > len(opener):
        return stripped[len(opener):-1].replace("''", "'")
    tokens = command_tokens(stripped, windows=True)
    return tokens[0] if tokens else ""


def desired_gemini_hooks(
    command: str | None = None, *, windows: bool | None = None,
) -> dict[str, list[dict[str, object]]]:
    """The aelfrice hook set in Gemini's settings shape, keyed by event.

    `command` is written as given. Without it, the hook path is resolved
    on this machine and quoted for Gemini's shell.
    """
    cmd = command if command is not None else gemini_hook_command(
        resolve_gemini_hook_command("user"), windows=windows,
    )
    return {
        event: [{
            "matcher": matcher,
            "hooks": [{
                "type": "command",
                "command": cmd,
                "name": name,
                "timeout": timeout_ms,
            }],
        }]
        for event, (name, matcher, timeout_ms) in HOOK_SPECS.items()
    }


# --- ownership --------------------------------------------------------------


def _handler_command(handler: object) -> str:
    if not isinstance(handler, dict):
        return ""
    cmd = cast(dict[str, object], handler).get("command")
    return cmd if isinstance(cmd, str) else ""


def _handler_is_owned(handler: object, *, windows: bool | None = None) -> bool:
    """True iff `handler` runs `aelf-gemini-hook`.

    Every reading of the command is offered (#1482), including the program
    a quoted command names, but membership is the closed owned set, so a
    foreign handler is never claimed.
    """
    cmd = _handler_command(handler)
    if not cmd.strip():
        return False
    owned = owned_keys(_OWNED_BASENAMES, windows=windows)
    keys = command_program_keys(cmd, windows=windows)
    program = shell_program(cmd, windows=windows)
    if program:
        keys.append(launcher_key(program, windows=windows))
    return any(k in owned for k in keys)


def _without_owned_handlers(
    group: object, *, windows: bool | None = None,
) -> object | None:
    """Drop our handlers from a group; None when no handler survives.

    Ownership is per handler, as on the codex host (#1412): a group that
    also holds a foreign handler keeps that handler and every other key.
    """
    if not isinstance(group, dict):
        return group
    gd = cast(dict[str, object], group)
    handlers = gd.get("hooks")
    if not isinstance(handlers, list):
        return gd
    listed = cast(list[object], handlers)
    kept = [h for h in listed if not _handler_is_owned(h, windows=windows)]
    if len(kept) == len(listed):
        return gd
    if not kept:
        return None
    survivor = dict(gd)
    survivor["hooks"] = kept
    return survivor


def _strip_owned(
    groups: list[object], *, windows: bool | None = None,
) -> list[object]:
    return [
        survivor
        for survivor in (_without_owned_handlers(g, windows=windows) for g in groups)
        if survivor is not None
    ]


# --- settings.json as a transaction ------------------------------------------

_LOCK_TIMEOUT: Final[float] = 10.0
_COMMIT_ATTEMPTS: Final[int] = 3

#: A key missing from the document, distinct from a key holding `null`.
_ABSENT: Final[object] = object()


@dataclass
class GeminiSettingsResult:
    path: Path
    changed: bool
    events: list[str] = field(default_factory=list[str])
    error: str | None = None


def _json_type_name(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _serialize(doc: dict[str, object], original: str | None) -> str:
    """`doc` as JSON, laid out like `original`, the text it was read from.

    Keeps the indent of the first indented line (none for a one-line
    file), the line ending, the final newline or its absence, and whether
    non-ASCII characters are escaped. A file in a standard JSON layout so
    gets its bytes back when unsetup undoes setup. A new file gets two
    spaces and a final newline.
    """
    if original is None:
        return json.dumps(doc, indent=2, ensure_ascii=False) + "\n"
    indented = re.search(r"\n([ \t]+)\S", original)
    text = json.dumps(
        doc,
        indent=indented.group(1) if indented else None,
        ensure_ascii=original.isascii(),
    )
    newline = "\r\n" if "\r\n" in original else "\n"
    if newline != "\n":
        text = text.replace("\n", newline)
    return text + (newline if original.endswith("\n") else "")


def _plan_install(
    text: str | None, *, force: bool, path: Path, command: str | None,
    windows: bool | None = None,
) -> tuple[str | None, GeminiSettingsResult]:
    """Merge the aelfrice hook set into one snapshot. Touches no file."""
    def refuse(msg: str) -> tuple[str | None, GeminiSettingsResult]:
        return None, GeminiSettingsResult(path=path, changed=False, error=msg)

    doc: dict[str, object] = {}
    if text is not None:
        try:
            parsed: object = json.loads(text)
        except json.JSONDecodeError as exc:
            if not force:
                return refuse(
                    f"existing settings.json is not plain JSON ({exc}); a "
                    "file with comments can't be merged. Fix it, or re-run "
                    "with --force to replace it"
                )
            parsed = {}
        if isinstance(parsed, dict):
            doc = cast(dict[str, object], parsed)
        elif not force:
            return refuse(
                "existing settings.json is not a JSON object; re-run with "
                "--force to replace it"
            )

    hooks_obj = doc.get("hooks", _ABSENT)
    if isinstance(hooks_obj, dict):
        hooks_map = cast(dict[str, object], hooks_obj)
    elif hooks_obj is _ABSENT or force:
        hooks_map = {}
    else:
        return refuse(
            "existing settings.json has a non-object `hooks` value "
            f"({_json_type_name(hooks_obj)}); fix it, or re-run with "
            "--force to replace it"
        )

    desired = desired_gemini_hooks(command, windows=windows)
    if not force:
        for event in desired:
            current = hooks_map.get(event, _ABSENT)
            if current is not _ABSENT and not isinstance(current, list):
                return refuse(
                    f"existing settings.json has a non-list value for the "
                    f"`{event}` event ({_json_type_name(current)}); fix it, "
                    "or re-run with --force to replace it"
                )

    before = json.dumps(hooks_map, sort_keys=True)
    for event, groups in desired.items():
        current = hooks_map.get(event)
        current_list = cast(list[object], current) if isinstance(current, list) else []
        hooks_map[event] = _strip_owned(current_list, windows=windows) + cast(
            list[object], groups,
        )
    doc["hooks"] = hooks_map
    changed = text is None or before != json.dumps(hooks_map, sort_keys=True)
    result = GeminiSettingsResult(
        path=path, changed=changed, events=sorted(desired),
    )
    if not changed:
        return None, result
    return _serialize(doc, text), result


def _plan_remove(
    text: str | None, *, path: Path, windows: bool | None = None,
) -> tuple[str | None, GeminiSettingsResult]:
    """Strip aelfrice handlers from one snapshot. Touches no file.

    An unparseable file is reported and left alone. An event that loses
    its last group is dropped, and so is a `hooks` object that this
    leaves empty. The file itself is kept, as `{}` when nothing else is
    left: nothing records whether setup created it or the user had it.
    """
    if text is None:
        return None, GeminiSettingsResult(path=path, changed=False)
    try:
        parsed: object = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, GeminiSettingsResult(
            path=path, changed=False,
            error=f"settings.json is not plain JSON ({exc}); not modified",
        )
    if not isinstance(parsed, dict):
        return None, GeminiSettingsResult(
            path=path, changed=False,
            error="settings.json is not a JSON object; not modified",
        )
    doc = cast(dict[str, object], parsed)
    hooks_obj = doc.get("hooks")
    if not isinstance(hooks_obj, dict):
        return None, GeminiSettingsResult(path=path, changed=False)
    hooks_map = cast(dict[str, object], hooks_obj)
    removed: list[str] = []
    for event in list(hooks_map):
        groups = hooks_map[event]
        if not isinstance(groups, list):
            continue
        group_list = cast(list[object], groups)
        kept = _strip_owned(group_list, windows=windows)
        if kept == group_list:
            continue
        removed.append(event)
        if kept:
            hooks_map[event] = kept
        else:
            del hooks_map[event]
    if not removed:
        return None, GeminiSettingsResult(path=path, changed=False)
    if not hooks_map:
        del doc["hooks"]
    return _serialize(doc, text), GeminiSettingsResult(
        path=path, changed=True, events=sorted(removed),
    )


def _read_snapshot(path: Path) -> tuple[str | None, str]:
    """`(text, sha256)` from one read; `(None, "")` when absent.

    Raises `OSError` when the path can't be read, for example a directory,
    a file without read permission, or a parent that is not a directory,
    and `UnicodeDecodeError` when the bytes are not UTF-8.
    """
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None, ""
    return raw.decode("utf-8"), hashlib.sha256(raw).hexdigest()


def _fingerprint(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return ""


def _atomic_replace(path: Path, text: str, expected: str) -> bool:
    """Replace `path` with `text` iff it still hashes to `expected`."""
    if _fingerprint(path) != expected:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.chmod(tmp_path, path.stat().st_mode & 0o777)
        except OSError:
            # First install, or a platform that won't apply the mode; the
            # 0600 default is the restrictive side.
            pass
        os.replace(tmp_path, path)
    except BaseException:
        try:
            tmp_path.unlink()
        except OSError:
            # Keep the original failure; a stray temp file is the smaller
            # problem.
            pass
        raise
    return True


_Plan = Callable[[str | None], tuple[str | None, GeminiSettingsResult]]


def settings_lock_anchor(settings_path: Path) -> Path:
    """The path whose `.lock` sibling serializes writes to `settings_path`.

    It is in the temporary directory, named for a hash of the resolved
    settings path, so every aelfrice process that writes one settings
    file takes one lock, and neither Gemini's directory nor aelfrice's
    keeps a lock file after unsetup.
    """
    key = hashlib.sha256(
        str(settings_path.resolve()).encode("utf-8"),
    ).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f"aelfrice-gemini-settings-{key}"


def _commit(path: Path, plan: _Plan) -> GeminiSettingsResult:
    """Run `plan` against `path` under the lock and commit atomically."""
    from aelfrice.session_ring import FileLockTimeout, exclusive_file_lock

    try:
        with exclusive_file_lock(
            settings_lock_anchor(path), timeout=_LOCK_TIMEOUT,
        ):
            for _ in range(_COMMIT_ATTEMPTS):
                try:
                    text, fingerprint = _read_snapshot(path)
                except (OSError, UnicodeDecodeError) as exc:
                    return GeminiSettingsResult(
                        path=path, changed=False,
                        error=f"could not read {path} ({exc}); nothing was "
                              "changed",
                    )
                serialized, result = plan(text)
                if serialized is None:
                    return result
                try:
                    if _atomic_replace(path, serialized, fingerprint):
                        return result
                except OSError as exc:
                    return GeminiSettingsResult(
                        path=path, changed=False,
                        error=f"could not write {path} ({exc}); the "
                              "existing file is unchanged",
                    )
    except FileLockTimeout as exc:
        return GeminiSettingsResult(
            path=path, changed=False,
            error=(
                f"another aelfrice process is writing {path} ({exc}); "
                "nothing was changed. Re-run the command."
            ),
        )
    return GeminiSettingsResult(
        path=path, changed=False,
        error=(
            f"{path} was modified by another process during each of "
            f"{_COMMIT_ATTEMPTS} attempts; nothing was written. Re-run the "
            "command."
        ),
    )


def install_gemini_hooks(
    settings_path: Path,
    *,
    force: bool = False,
    command: str | None = None,
    windows: bool | None = None,
) -> GeminiSettingsResult:
    """Write the aelfrice hook set into `settings_path`, merge-aware."""
    return _commit(
        settings_path,
        lambda text: _plan_install(
            text, force=force, path=settings_path, command=command,
            windows=windows,
        ),
    )


def remove_gemini_hooks(
    settings_path: Path, *, windows: bool | None = None,
) -> GeminiSettingsResult:
    """Remove every aelfrice handler from `settings_path`.

    A missing file is not opened, locked, or created: unsetup must not
    create the configuration it removes from.
    """
    if not settings_path.is_file():
        return GeminiSettingsResult(path=settings_path, changed=False)
    return _commit(
        settings_path,
        lambda text: _plan_remove(text, path=settings_path, windows=windows),
    )


# --- custom commands -------------------------------------------------------------

#: The first line of every generated command file. Only files that carry it
#: are replaced or removed, so a user's own `aelf` command is left alone.
COMMAND_MARKER: Final[str] = "AELFRICE-GEMINI-COMMAND"

_HOST_MANAGEMENT_COMMANDS: Final[frozenset[str]] = frozenset(
    {"setup", "doctor", "uninstall", "upgrade"},
)

_SETUP_DESCRIPTION: Final[str] = (
    "Install the aelfrice hooks in Gemini's user settings.json "
    "(~/.gemini, or $GEMINI_CLI_HOME/.gemini) and the /aelf:* commands "
    "under its commands/aelf/ directory."
)

#: The #1429 option rule for this host. It lives here, not in the shared
#: `setup.md`, which also builds the claude command and the codex skill.
_SETUP_OPTION_NOTE: Final[str] = (
    "On this host, `aelf setup --host gemini` accepts only `--force`, and "
    "`aelf unsetup --host gemini` accepts no option but `--host`. Both "
    "refuse every other option, including the flags the steps below list "
    "for other hosts, with exit 2 and change nothing."
)

_HOST_MANAGEMENT_NOTE: Final[str] = (
    "On this host, every `aelf setup`, `aelf doctor`, `aelf unsetup`, or "
    "`aelf uninstall` command in the steps below must carry `--host "
    "gemini`, for example `aelf setup --host gemini` or `aelf uninstall "
    "<flags> --host gemini`. Without it the command acts on another "
    "host's configuration and leaves this one unchanged."
)


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split a slash-command file into its scalar keys and its body."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    close = next(
        (i for i in range(1, len(lines)) if lines[i].strip() == "---"), None,
    )
    if close is None:
        return {}, text
    front: dict[str, str] = {}
    for raw in lines[1:close]:
        if raw[:1] in (" ", "\t") or ":" not in raw:
            continue
        key, _, value = raw.partition(":")
        front[key.strip()] = value.strip()
    return front, "\n".join(lines[close + 1:]).lstrip("\n")


def _toml_string(value: str) -> str:
    """`value` as a TOML string, multi-line literal where that is valid."""
    literal_ok = "'''" not in value and all(
        ch in "\n\t" or ord(ch) >= 0x20 and ord(ch) != 0x7F for ch in value
    )
    if literal_ok and "\n" in value:
        # The newline after the opening quotes is not part of the value.
        return "'''\n" + value + "'''"
    # JSON string escapes are a subset of TOML basic-string escapes.
    return json.dumps(value, ensure_ascii=False)


def gemini_command_from_slash(filename: str, text: str) -> tuple[str, str]:
    """One bundled slash command as `(file name, TOML text)`.

    The body keeps its steps. `$ARGUMENTS` becomes Gemini's `{{args}}`,
    `uv run aelf` becomes `aelf` because the command runs the installed CLI
    (the codex host does the same, #1413), and host-management commands
    get a note that steers them to `--host gemini`. Deterministic: the same
    input gives the same bytes.
    """
    stem = filename.removesuffix(".md")
    front, body = _parse_frontmatter(text)
    slash_name = front.get("name") or f"{COMMAND_NAMESPACE}:{stem}"
    description = front.get("description", "")
    if stem == "setup":
        description = _SETUP_DESCRIPTION
    preamble = [
        f"This is the Gemini CLI port of the `/{slash_name}` command.",
    ]
    hint = front.get("argument-hint")
    if hint:
        preamble.append(f"Arguments: {hint}")
    if "Task" in text:
        # The source lists the `Task` tool, so its steps fan work out.
        preamble.append(
            "Where the steps hand work to the host's `Task` tool or to "
            "helper agents, use Gemini's sub-agents for the same work, or "
            "do it in sequence; the `aelf` commands are unchanged."
        )
    if stem in _HOST_MANAGEMENT_COMMANDS:
        preamble.append(_HOST_MANAGEMENT_NOTE)
    if stem == "setup":
        preamble.append(_SETUP_OPTION_NOTE)
    preamble.append(
        "Run each `aelf ...` command with your shell tool and show its "
        "output to the user. Run `aelf` directly, as written."
    )
    prompt_body = (
        body.rstrip("\n")
        .replace("uv run aelf ", "aelf ")
        .replace("$ARGUMENTS", "{{args}}")
    )
    prompt = "\n".join(preamble) + "\n\n" + prompt_body + "\n"
    lines = [
        f"# {COMMAND_MARKER}: generated from src/aelfrice/slash_commands/"
        f"{filename} by `aelf setup --host gemini`. Edit the source file, "
        "not this copy.",
        f"description = {_toml_string(description)}",
        f"prompt = {_toml_string(prompt)}",
        "",
    ]
    return stem + COMMAND_SUFFIX, "\n".join(lines)


def bundled_gemini_commands() -> dict[str, str]:
    """Map file name -> TOML text for every bundled slash command."""
    out: dict[str, str] = {}
    for filename, text in bundled_slash_files().items():
        name, toml_text = gemini_command_from_slash(filename, text)
        out[name] = toml_text
    return out


def _is_owned_command(path: Path) -> bool:
    if path.suffix != COMMAND_SUFFIX or not path.is_file():
        return False
    try:
        with path.open(encoding="utf-8", errors="replace") as f:
            first = f.readline()
    except OSError:
        return False
    return first.startswith(f"# {COMMAND_MARKER}:")


@dataclass(frozen=True)
class GeminiCommandsResult:
    dest_dir: Path
    written: tuple[str, ...] = ()
    already: tuple[str, ...] = ()
    pruned: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    failed: tuple[str, ...] = ()


def _orphans(target: Path, bundle: Mapping[str, str]) -> list[Path]:
    if not target.is_dir():
        return []
    return [
        child for child in sorted(target.glob(f"*{COMMAND_SUFFIX}"))
        if child.name not in bundle and _is_owned_command(child)
    ]


def _write_atomic(dest: Path, text: str) -> None:
    fd, tmp_name = tempfile.mkstemp(
        prefix=dest.name + ".", suffix=".tmp", dir=str(dest.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, dest)
    except BaseException:
        try:
            tmp_path.unlink()
        except OSError:
            # Keep the original failure.
            pass
        raise


def install_gemini_commands(gemini_dir: Path | None = None) -> GeminiCommandsResult:
    """Write every bundled command under `commands/aelf/`.

    Idempotent, and orphan-pruning for files with the marker. A file of
    the same name without the marker is someone else's and is skipped.
    """
    target = gemini_commands_dir(gemini_dir)
    bundle = bundled_gemini_commands()
    written: list[str] = []
    already: list[str] = []
    skipped: list[str] = []
    target.mkdir(parents=True, exist_ok=True)
    for name, text in sorted(bundle.items()):
        dest = target / name
        if dest.is_file():
            try:
                if dest.read_text(encoding="utf-8") == text:
                    already.append(name)
                    continue
            except OSError:
                # Unreadable: ownership can't be read either, so the marker
                # check below fails closed.
                pass
            if not _is_owned_command(dest):
                skipped.append(name)
                continue
        _write_atomic(dest, text)
        written.append(name)
    pruned: list[str] = []
    failed: list[str] = []
    for child in _orphans(target, bundle):
        _remove_command(child, pruned, failed)
    return GeminiCommandsResult(
        dest_dir=target, written=tuple(written), already=tuple(already),
        pruned=tuple(pruned), skipped=tuple(skipped), failed=tuple(failed),
    )


def _remove_command(child: Path, pruned: list[str], failed: list[str]) -> None:
    try:
        child.unlink()
    except OSError as exc:
        failed.append(f"{child.name}: could not remove ({exc})")
        return
    pruned.append(child.name)


def remove_gemini_commands(gemini_dir: Path | None = None) -> GeminiCommandsResult:
    """Remove every generated command; then `commands/aelf/` if it is empty,
    and `commands/` if that leaves it empty."""
    gdir = gemini_dir if gemini_dir is not None else resolve_gemini_dir()
    target = gemini_commands_dir(gdir)
    pruned: list[str] = []
    failed: list[str] = []
    if target.is_dir():
        for child in sorted(target.glob(f"*{COMMAND_SUFFIX}")):
            if _is_owned_command(child):
                _remove_command(child, pruned, failed)
        if pruned:
            for directory in (target, gdir / COMMANDS_DIRNAME):
                try:
                    directory.rmdir()
                except OSError:
                    # Not empty: other files in it are not ours to remove.
                    break
    return GeminiCommandsResult(
        dest_dir=target, pruned=tuple(pruned), failed=tuple(failed),
    )


def installed_gemini_commands(gemini_dir: Path | None = None) -> list[str]:
    """Names of the generated command files present."""
    target = gemini_commands_dir(gemini_dir)
    if not target.is_dir():
        return []
    return [
        child.name for child in sorted(target.glob(f"*{COMMAND_SUFFIX}"))
        if _is_owned_command(child)
    ]


def modified_gemini_commands(gemini_dir: Path | None = None) -> list[str]:
    """Generated command files whose bytes differ from what this build writes."""
    target = gemini_commands_dir(gemini_dir)
    modified: list[str] = []
    for name, expected in sorted(bundled_gemini_commands().items()):
        dest = target / name
        if not _is_owned_command(dest):
            continue
        try:
            actual = dest.read_text(encoding="utf-8")
        except OSError:  # pragma: no cover - raced with the marker read
            actual = ""
        if actual != expected:
            modified.append(name)
    return modified


def orphaned_gemini_commands(gemini_dir: Path | None = None) -> list[str]:
    """Generated command files for commands this build no longer ships."""
    target = gemini_commands_dir(gemini_dir)
    return [p.name for p in _orphans(target, bundled_gemini_commands())]


# --- doctor -------------------------------------------------------------------------


@dataclass
class GeminiDoctorReport:
    """The Gemini host scan; the CLI renders it and picks the exit code."""

    gemini_dir: Path
    gemini_dir_present: bool
    settings_present: bool = False
    settings_valid: bool = False
    parse_error: str | None = None
    #: The unparseable file still names `aelf-gemini-hook`, so wiring may
    #: be installed in it, and its hooks can't be checked.
    unparseable_names_hook: bool = False
    owned_handler_count: int = 0
    missing_events: list[str] = field(default_factory=list[str])
    stale_commands: list[str] = field(default_factory=list[str])
    #: Bare hook commands that don't resolve on PATH.
    unresolved_commands: list[str] = field(default_factory=list[str])
    unquoted_commands: list[str] = field(default_factory=list[str])
    hooks_turned_off_by: list[str] = field(default_factory=list[str])
    disabled_hook_names: list[str] = field(default_factory=list[str])
    warnings: list[str] = field(default_factory=list[str])

    def faults(self) -> list[str]:
        """Ways installed wiring is broken, each a reason to exit 1.

        Gated on aelfrice handlers being present, as on the codex host
        (#1430): a machine that never ran `aelf setup --host gemini` is a
        normal machine. A settings.json that can't be parsed counts as
        installed wiring only when its text names the hook script.
        """
        reasons: list[str] = []
        if self.unparseable_names_hook:
            reasons.append(
                f"settings.json can't be read as JSON ({self.parse_error}) "
                f"but names {GEMINI_HOOK_SCRIPT_NAME}, so aelfrice can't "
                "check the aelfrice hooks in it; make it plain JSON again",
            )
        if not self.owned_handler_count:
            return reasons
        if self.stale_commands:
            reasons.append(
                "hook command(s) configured but not present on disk: "
                + ", ".join(self.stale_commands),
            )
        if self.unresolved_commands:
            reasons.append(
                "hook command(s) not on PATH, so Gemini can't run "
                "them: " + ", ".join(self.unresolved_commands)
                + f"; put the directory that holds {GEMINI_HOOK_SCRIPT_NAME} "
                "on PATH, or re-run `aelf setup --host gemini` from an "
                "install outside a .claude/worktrees venv",
            )
        if self.unquoted_commands:
            reasons.append(
                "hook command(s) hold a path with a space that is not "
                "quoted, so Gemini's shell runs only the part before the "
                "space: " + ", ".join(self.unquoted_commands)
                + "; re-run `aelf setup --host gemini`",
            )
        if self.missing_events:
            reasons.append(
                "aelfrice hooks are partially installed; missing events: "
                + ", ".join(self.missing_events),
            )
        if self.hooks_turned_off_by:
            reasons.append(
                "aelfrice hooks are installed but Gemini hooks are turned "
                "off by " + ", ".join(self.hooks_turned_off_by)
                + " in settings.json, so none of them run",
            )
        return reasons


def _setting(doc: Mapping[str, object], section: str, key: str) -> object:
    block = doc.get(section)
    if not isinstance(block, dict):
        return None
    return cast(dict[str, object], block).get(key)


def _check_command(
    report: GeminiDoctorReport, cmd: str, *, windows: bool | None,
) -> None:
    """File `cmd` as stale, unresolved, or unquoted when Gemini's shell
    can't run it."""
    if cmd in (
        *report.stale_commands, *report.unresolved_commands,
        *report.unquoted_commands,
    ):
        return
    program = shell_program(cmd, windows=windows)
    if not program:
        return
    exe = Path(program)
    if not exe.is_absolute():
        # Setup writes the bare name when it resolves no path (#928), and
        # the shell then looks the name up on PATH.
        if which_on_path(program) is None:
            report.unresolved_commands.append(cmd)
        return
    if exe.is_file():
        return
    if program_exists(cmd, windows=windows):
        # The file is there, at a path the shell splits at a space.
        report.unquoted_commands.append(cmd)
    else:
        report.stale_commands.append(cmd)


def doctor_gemini(
    gemini_dir: Path | None = None, *, windows: bool | None = None,
) -> GeminiDoctorReport:
    """Scan the Gemini user settings. Read-only; reports, never raises."""
    gdir = gemini_dir if gemini_dir is not None else resolve_gemini_dir()
    report = GeminiDoctorReport(gemini_dir=gdir, gemini_dir_present=gdir.is_dir())
    if not report.gemini_dir_present:
        report.warnings.append(f"{gdir} not found; is Gemini CLI installed?")
        return report
    path = gemini_settings_path(gdir)
    report.settings_present = path.is_file()
    doc: dict[str, object] = {}
    if report.settings_present:
        text = ""
        try:
            raw = path.read_bytes()
            # Kept readable for the hook-name check below when the bytes
            # are not UTF-8.
            text = raw.decode("utf-8", errors="replace")
            parsed: object = json.loads(raw.decode("utf-8"))
        except (OSError, ValueError) as exc:
            # ValueError covers JSONDecodeError and UnicodeDecodeError.
            report.parse_error = str(exc)
        else:
            if isinstance(parsed, dict):
                report.settings_valid = True
                doc = cast(dict[str, object], parsed)
            else:
                report.parse_error = "top-level JSON is not an object"
        report.unparseable_names_hook = (
            not report.settings_valid and GEMINI_HOOK_SCRIPT_NAME in text
        )
        if not report.settings_valid:
            report.warnings.append(
                f"{path} could not be read as JSON ({report.parse_error}); "
                "aelfrice can't check its hooks",
            )
    hooks_obj = doc.get("hooks")
    hooks_map = cast(dict[str, object], hooks_obj) if isinstance(hooks_obj, dict) else {}
    covered: set[str] = set()
    for event, groups in hooks_map.items():
        if not isinstance(groups, list):
            continue
        for group in cast(list[object], groups):
            if not isinstance(group, dict):
                continue
            handlers = cast(dict[str, object], group).get("hooks")
            if not isinstance(handlers, list):
                continue
            for handler in cast(list[object], handlers):
                if not _handler_is_owned(handler, windows=windows):
                    continue
                report.owned_handler_count += 1
                covered.add(event)
                _check_command(
                    report, _handler_command(handler), windows=windows,
                )
    report.missing_events = sorted(set(HOOK_SPECS) - covered)
    # The one hooks switch Gemini 0.68.0 documents
    # (docs/reference/configuration.md, `hooksConfig.enabled`).
    if _setting(doc, "hooksConfig", "enabled") is False:
        report.hooks_turned_off_by.append("hooksConfig.enabled")
    disabled = _setting(doc, "hooksConfig", "disabled")
    if isinstance(disabled, list):
        # Gemini documents the entries as hook names or commands
        # (docs/reference/configuration.md, `hooksConfig.disabled`).
        ours = {name for name, _, _ in HOOK_SPECS.values()}
        report.disabled_hook_names = sorted(
            entry for entry in cast(list[object], disabled)
            if isinstance(entry, str) and (
                entry in ours
                or _handler_is_owned({"command": entry}, windows=windows)
            )
        )
    for cmd in report.stale_commands:
        report.warnings.append(f"hook command not found on disk: {cmd}")
    for cmd in report.unresolved_commands:
        report.warnings.append(f"hook command not found on PATH: {cmd}")
    if report.disabled_hook_names:
        report.warnings.append(
            "aelfrice hooks turned off in hooksConfig.disabled: "
            + ", ".join(report.disabled_hook_names),
        )
    if not report.owned_handler_count and report.settings_valid:
        report.warnings.append(
            f"no aelfrice hooks in {path} (run `aelf setup --host gemini`)",
        )
    elif not report.settings_present:
        report.warnings.append(
            f"{path} does not exist; no aelfrice hooks are installed for "
            "the Gemini host (run `aelf setup --host gemini`)",
        )
    return report
