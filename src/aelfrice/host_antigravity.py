"""Antigravity CLI (agy) host target for `aelf setup` / `unsetup` / `doctor`.

Writes one aelfrice-owned entry, ``"aelfrice"``, into agy's global hooks
file ``~/.gemini/config/hooks.json``, and the ``/aelf-*`` skills into
``~/.gemini/config/skills/``. Every handler runs `aelf-agy-hook`
(`aelfrice.agy_hook`), which translates agy's payloads and runs the
existing hooks. The event table and the gaps come from
``docs/design/antigravity_hook_surface.md`` (agy 1.5.7).

Design constraints:

- **One owned key.** agy merges every top-level key of hooks.json as a
  named hook, so setup replaces only ``"aelfrice"`` and keeps every other
  key as is. Unsetup removes only that key.
- **Unknown event names are dropped silently by agy**, so `doctor` fails
  on any event under the aelfrice key that agy does not fire.
- **Timeouts are seconds** (agy's unit), summed over the roles one event
  runs, because the dispatcher runs them one after another.
- **Global scope only.** agy also reads ``<workspace>/.agents/hooks.json``,
  but the hook process runs in that file's folder and the project comes
  from the payload, so one global entry serves every project.

Hooks with no agy equivalent are listed with the reason in
`AGY_UNSUPPORTED`.
"""
from __future__ import annotations

import json
import os
import shlex
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from aelfrice import agy_hook
from aelfrice.launcher import command_tokens
from aelfrice.setup import SettingsScope, _resolve_script, read_settings, write_settings  # pyright: ignore[reportPrivateUsage]

AGY_HOOK_SCRIPT_NAME: Final[str] = "aelf-agy-hook"
OWNED_HOOK_KEY: Final[str] = "aelfrice"
HOOKS_FILENAME: Final[str] = "hooks.json"
SKILLS_DIRNAME: Final[str] = "skills"

EVENT_SESSION_START: Final[str] = "SessionStart"
EVENT_PRE_INVOCATION: Final[str] = "PreInvocation"
EVENT_PRE_TOOL: Final[str] = "PreToolUse"
EVENT_POST_TOOL: Final[str] = "PostToolUse"
EVENT_STOP: Final[str] = "Stop"
AGY_EVENTS: Final[frozenset[str]] = frozenset({
    EVENT_SESSION_START, EVENT_PRE_INVOCATION, EVENT_PRE_TOOL, EVENT_POST_TOOL,
    "PostInvocation", EVENT_STOP,
})
TOOL_EVENTS: Final[frozenset[str]] = frozenset({EVENT_PRE_TOOL, EVENT_POST_TOOL})

# Roles each agy event runs (see `aelfrice.agy_hook`).
EVENT_ROLES: Final[dict[str, tuple[str, ...]]] = {
    EVENT_SESSION_START: (agy_hook.ROLE_SESSION_START,),
    EVENT_PRE_INVOCATION: (
        agy_hook.ROLE_SESSION_START, agy_hook.ROLE_TRANSCRIPT,
        agy_hook.ROLE_USER_PROMPT, agy_hook.ROLE_SEARCH_BASH,
    ),
    EVENT_PRE_TOOL: (
        agy_hook.ROLE_PRE_ISSUE, agy_hook.ROLE_SEARCH_BASH, agy_hook.ROLE_AGENT_CONTEXT,
    ),
    EVENT_POST_TOOL: (agy_hook.ROLE_COMMIT,),
    EVENT_STOP: (agy_hook.ROLE_TRANSCRIPT, agy_hook.ROLE_STOP),
}
EVENT_ORDER: Final[tuple[str, ...]] = (
    EVENT_SESSION_START, EVENT_PRE_INVOCATION, EVENT_PRE_TOOL, EVENT_POST_TOOL, EVENT_STOP,
)

DEFAULT_ROLE_TIMEOUT_S: Final[int] = 15
MAX_EVENT_TIMEOUT_S: Final[int] = 120

AGY_UNSUPPORTED: Final[dict[str, str]] = {
    "PreCompact/PostCompact (transcript rotation, --rebuilder)":
        "agy fires no compaction event",
    "statusline": "agy has no statusline command hook",
    "claude-memory-mirror": "agy keeps no per-project auto-memory directory",
    "search_tool (Grep|Glob|WebSearch|WebFetch)":
        "agy 1.5.7 searches through run_command, which search_tool_bash covers",
    "PreToolUse context at tool time":
        "nothing a tool hook prints reaches the model; search context waits "
        "for the next model call",
    "SessionStart on resume":
        "agy -c fires no SessionStart; the first PreInvocation of a resumed "
        "conversation supplies the baseline",
}

# Slash commands with no agy counterpart.
_EXCLUDED_SLASH: Final[frozenset[str]] = frozenset({
    "audit-claude-memory.md",
    "reconcile-claude-memory.md",
})
_SKILL_MARKER: Final[str] = "AELFRICE-ANTIGRAVITY-SKILL"
_SKILL_PREFIX: Final[str] = "aelf-"
_SKILL_FILENAME: Final[str] = "SKILL.md"
_HOST_MANAGEMENT_SKILLS: Final[frozenset[str]] = frozenset({
    "aelf-setup", "aelf-doctor", "aelf-uninstall", "aelf-upgrade",
})
_HOST_MANAGEMENT_NOTE: Final[str] = (
    "On this host, every `aelf setup`, `aelf doctor`, `aelf unsetup`, or "
    "`aelf uninstall` command in the steps below must use the "
    "`--host antigravity` form, for example `aelf doctor --host antigravity`. "
    "The bare form targets other hosts."
)
_SETUP_DESCRIPTION: Final[str] = (
    "Install the aelfrice hooks in ~/.gemini/config/hooks.json and the "
    "/aelf-* skills under ~/.gemini/config/skills/ for Antigravity CLI."
)


def agy_config_dir() -> Path:
    """agy's global customization folder, ``~/.gemini/config``."""
    return Path.home() / ".gemini" / "config"


def agy_hooks_path() -> Path:
    return agy_config_dir() / HOOKS_FILENAME


def agy_skills_dir() -> Path:
    return agy_config_dir() / SKILLS_DIRNAME


def resolve_agy_hook_command(scope: SettingsScope = "user") -> str:
    return _resolve_script(AGY_HOOK_SCRIPT_NAME, scope)


def _event_timeout(roles: tuple[str, ...], timeouts_s: dict[str, int]) -> int:
    total = sum(timeouts_s.get(r, DEFAULT_ROLE_TIMEOUT_S) for r in roles)
    return max(DEFAULT_ROLE_TIMEOUT_S, min(total, MAX_EVENT_TIMEOUT_S))


def _pre_tool_matcher(active: tuple[str, ...]) -> str:
    tools: list[str] = []
    if agy_hook.ROLE_PRE_ISSUE in active or agy_hook.ROLE_SEARCH_BASH in active:
        tools.append("run_command")
    if agy_hook.ROLE_AGENT_CONTEXT in active:
        tools.append("invoke_subagent")
    return "|".join(tools)


def desired_agy_hooks(
    command: str,
    *,
    timeouts_s: dict[str, int] | None = None,
    skip: frozenset[str] = frozenset(),
) -> dict[str, object]:
    """The aelfrice entry for hooks.json, keyed by agy event.

    `command` is the resolved `aelf-agy-hook` path. Roles in `skip` are
    passed to it as ``--skip``; an event whose roles are all skipped is
    left out.
    """
    budgets = timeouts_s if timeouts_s is not None else {}
    skipped = sorted(skip & agy_hook.ALL_ROLES)
    base = shlex.quote(command)
    full = f"{base} --skip {','.join(skipped)}" if skipped else base
    out: dict[str, object] = {}
    for event in EVENT_ORDER:
        active = tuple(r for r in EVENT_ROLES[event] if r not in skip)
        if not active:
            continue
        handler: dict[str, object] = {
            "type": "command",
            "command": full,
            "timeout": _event_timeout(active, budgets),
        }
        if event == EVENT_PRE_TOOL:
            out[event] = [{"matcher": _pre_tool_matcher(active), "hooks": [handler]}]
        elif event == EVENT_POST_TOOL:
            out[event] = [{"matcher": "run_command", "hooks": [handler]}]
        else:
            out[event] = [handler]
    return out


@dataclass(frozen=True)
class AgyHooksResult:
    """Outcome of `install_agy_hooks` / `uninstall_agy_hooks`."""
    path: Path
    changed: bool
    events: tuple[str, ...]


def _read(path: Path) -> dict[str, object]:
    return read_settings(path) if path.exists() else {}


def install_agy_hooks(
    path: Path,
    command: str,
    *,
    timeouts_s: dict[str, int] | None = None,
    skip: frozenset[str] = frozenset(),
) -> AgyHooksResult:
    """Write the aelfrice entry into `path`, keeping every other key.

    Joins an open `setup.settings_transaction` for the same path. Writes
    nothing when the file already holds the same entry.
    """
    data = _read(path)
    desired = desired_agy_hooks(command, timeouts_s=timeouts_s, skip=skip)
    changed = data.get(OWNED_HOOK_KEY) != desired
    if changed:
        write_settings(path, {**data, OWNED_HOOK_KEY: desired})
    return AgyHooksResult(path=path, changed=changed, events=tuple(desired))


def uninstall_agy_hooks(path: Path) -> AgyHooksResult:
    """Remove the aelfrice entry from `path`; other keys stay.

    A missing file or a file without the entry is a no-op. A file left
    with no keys is written as ``{}`` rather than deleted, so a file the
    user created stays where they put it.
    """
    if not path.exists():
        return AgyHooksResult(path=path, changed=False, events=())
    data = _read(path)
    entry = data.get(OWNED_HOOK_KEY)
    if entry is None:
        return AgyHooksResult(path=path, changed=False, events=())
    events = tuple(cast(dict[str, object], entry)) if isinstance(entry, dict) else ()
    write_settings(path, {k: v for k, v in data.items() if k != OWNED_HOOK_KEY})
    return AgyHooksResult(path=path, changed=True, events=events)


# --- skills -----------------------------------------------------------------


def antigravity_skill_from_slash(filename: str, text: str) -> tuple[str, str]:
    """One bundled slash command as an agy skill: (skill name, SKILL.md text).

    Deterministic. The description is written as a JSON string, which is
    a valid YAML double-quoted scalar, because several descriptions hold
    ``": "`` that a plain YAML scalar would reject.
    """
    from aelfrice.host_codex import (  # noqa: PLC0415
        _direct_cli_invocations,  # pyright: ignore[reportPrivateUsage]
        _parse_slash_frontmatter,  # pyright: ignore[reportPrivateUsage]
    )

    front, body = _parse_slash_frontmatter(text)
    raw_name = front.get("name") or (_SKILL_PREFIX + filename.removesuffix(".md"))
    skill_name = raw_name.replace(":", "-")
    description = front.get("description", "")
    if skill_name == "aelf-setup":
        description = _SETUP_DESCRIPTION
    notes: list[str] = [
        f"<!-- {_SKILL_MARKER}: generated from src/aelfrice/slash_commands/"
        f"{filename}. Edit the source file, not this copy. -->",
        "",
        f"This is the Antigravity port of the `/{raw_name}` slash command; "
        f"invoke it as `/{skill_name}`.",
    ]
    hint = front.get("argument-hint")
    if hint:
        notes.append(f"Arguments: {hint}")
    if "$ARGUMENTS" in body:
        notes.append(
            "Where the steps below reference `$ARGUMENTS`, substitute the text "
            f"the user typed after `/{skill_name}`."
        )
    if "Task" in body and "subagent" in body:
        notes.append(
            "Where the steps mention the host's `Task` tool or subagents, use "
            "Antigravity's own subagents for the same work; the CLI calls are "
            "unchanged."
        )
    if skill_name in _HOST_MANAGEMENT_SKILLS:
        notes.append(_HOST_MANAGEMENT_NOTE)
    notes.append(
        "Run each `aelf ...` command in your shell, exactly as written, and "
        "show its output to the user."
    )
    lines = [
        "---",
        f"name: {skill_name}",
        f"description: {json.dumps(description)}",
        "---",
        *notes,
        "",
        _direct_cli_invocations(body.rstrip("\n")),
        "",
    ]
    return skill_name, "\n".join(lines)


def bundled_agy_skills() -> dict[str, str]:
    """skill name -> SKILL.md text for every bundled command agy supports."""
    from aelfrice.setup import bundled_slash_files  # noqa: PLC0415

    out: dict[str, str] = {}
    for filename, text in bundled_slash_files().items():
        if filename in _EXCLUDED_SLASH:
            continue
        name, skill = antigravity_skill_from_slash(filename, text)
        out[name] = skill
    return out


def _is_owned_skill_dir(skill_dir: Path) -> bool:
    """An ``aelf-*`` dir whose SKILL.md carries the marker; nothing else is ours."""
    if not skill_dir.name.startswith(_SKILL_PREFIX) or not skill_dir.is_dir():
        return False
    skill_md = skill_dir / _SKILL_FILENAME
    try:
        return skill_md.is_file() and _SKILL_MARKER in skill_md.read_text(encoding="utf-8")
    except OSError:
        return False


@dataclass(frozen=True)
class AgySkillsResult:
    """Outcome of `install_agy_skills` / `remove_agy_skills`."""
    dest_dir: Path
    written: tuple[str, ...] = ()
    already: tuple[str, ...] = ()
    pruned: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    failed: tuple[str, ...] = ()


def _write_atomic(path: Path, text: str) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _remove_skill_dir(child: Path, pruned: list[str], failed: list[str]) -> None:
    try:
        (child / _SKILL_FILENAME).unlink()
        child.rmdir()
    except OSError as exc:
        failed.append(f"{child.name}: {exc}")
        return
    pruned.append(child.name)


def install_agy_skills(dest_dir: Path | None = None) -> AgySkillsResult:
    """Write each bundled skill to ``<dest>/aelf-<cmd>/SKILL.md``.

    Idempotent. Overwrites and prunes only marker-carrying ``aelf-*``
    dirs; a user's own ``aelf-*`` skill is reported under `skipped`.
    """
    target = dest_dir if dest_dir is not None else agy_skills_dir()
    bundle = bundled_agy_skills()
    written: list[str] = []
    already: list[str] = []
    skipped: list[str] = []
    pruned: list[str] = []
    failed: list[str] = []
    target.mkdir(parents=True, exist_ok=True)
    for name, text in sorted(bundle.items()):
        skill_dir = target / name
        dest = skill_dir / _SKILL_FILENAME
        if dest.is_file():
            try:
                if dest.read_text(encoding="utf-8") == text:
                    already.append(name)
                    continue
            except OSError:
                pass
            if not _is_owned_skill_dir(skill_dir):
                skipped.append(name)
                continue
        skill_dir.mkdir(parents=True, exist_ok=True)
        _write_atomic(dest, text)
        written.append(name)
    for child in sorted(target.glob(f"{_SKILL_PREFIX}*")):
        if child.name not in bundle and _is_owned_skill_dir(child):
            _remove_skill_dir(child, pruned, failed)
    return AgySkillsResult(
        dest_dir=target, written=tuple(written), already=tuple(already),
        pruned=tuple(pruned), skipped=tuple(skipped), failed=tuple(failed),
    )


def remove_agy_skills(dest_dir: Path | None = None) -> AgySkillsResult:
    """Remove every marker-carrying ``aelf-*`` skill under `dest_dir`."""
    target = dest_dir if dest_dir is not None else agy_skills_dir()
    pruned: list[str] = []
    failed: list[str] = []
    if target.is_dir():
        for child in sorted(target.glob(f"{_SKILL_PREFIX}*")):
            if _is_owned_skill_dir(child):
                _remove_skill_dir(child, pruned, failed)
    return AgySkillsResult(dest_dir=target, pruned=tuple(pruned), failed=tuple(failed))


# --- doctor -----------------------------------------------------------------


@dataclass(frozen=True)
class AgyDoctorResult:
    lines: tuple[str, ...]
    broken: bool


def _program(command: str) -> str | None:
    tokens = command_tokens(command)
    return tokens[0] if tokens else None


def _check_handler(event: str, handler: object, lines: list[str]) -> bool:
    """Append findings for one handler; return True when it is broken."""
    if not isinstance(handler, dict):
        lines.append(f"  BROKEN {event}: handler is not an object")
        return True
    h = cast(dict[str, object], handler)
    command = h.get("command")
    if not isinstance(command, str) or not command.strip():
        lines.append(f"  BROKEN {event}: handler has no command")
        return True
    program = _program(command)
    if program is None or not os.path.isfile(program) or not os.access(program, os.X_OK):
        lines.append(f"  BROKEN {event}: {program!r} is not an executable file")
        return True
    timeout = h.get("timeout")
    if isinstance(timeout, int) and not isinstance(timeout, bool) and timeout >= 1000:
        lines.append(f"  BROKEN {event}: timeout {timeout} looks like milliseconds; agy reads seconds")
        return True
    lines.append(f"  ok {event}: {command}")
    return False


def _handlers_of(event: str, groups: object, lines: list[str]) -> tuple[list[object], bool]:
    if not isinstance(groups, list):
        lines.append(f"  BROKEN {event}: value is not a list")
        return [], True
    items = cast(list[object], groups)
    if event not in TOOL_EVENTS:
        return items, False
    handlers: list[object] = []
    broken = False
    for group in items:
        g = cast(dict[str, object], group) if isinstance(group, dict) else {}
        matcher = g.get("matcher")
        inner = g.get("hooks")
        if not isinstance(matcher, str) or not isinstance(inner, list):
            lines.append(f"  BROKEN {event}: group needs a matcher and a hooks list")
            broken = True
            continue
        if "run_command" not in matcher.split("|") and event == EVENT_POST_TOOL:
            lines.append(f"  BROKEN {event}: matcher {matcher!r} misses run_command")
            broken = True
        handlers.extend(cast(list[object], inner))
    return handlers, broken


def doctor_agy(path: Path | None = None, skills_dir: Path | None = None) -> AgyDoctorResult:
    """Check the aelfrice entry in agy's hooks.json and the skills.

    Broken: no entry, ``enabled: false``, an event name agy never fires,
    a handler whose program is missing, or a timeout in milliseconds.
    Every `AGY_UNSUPPORTED` row is listed without failing.
    """
    hooks_path = path if path is not None else agy_hooks_path()
    lines: list[str] = [f"antigravity hooks: {hooks_path}"]
    broken = False
    data: dict[str, object] = {}
    try:
        data = _read(hooks_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        lines.append(f"  BROKEN: cannot read {hooks_path}: {exc}")
        broken = True
    entry = data.get(OWNED_HOOK_KEY)
    if not isinstance(entry, dict):
        lines.append("  BROKEN: no aelfrice entry; run `aelf setup --host antigravity`")
        broken = True
    else:
        spec = cast(dict[str, object], entry)
        if spec.get("enabled") is False:
            lines.append("  BROKEN: the aelfrice entry has \"enabled\": false")
            broken = True
        for event, groups in spec.items():
            if event == "enabled":
                continue
            if event not in AGY_EVENTS:
                lines.append(f"  BROKEN {event}: agy never fires this event")
                broken = True
                continue
            handlers, group_broken = _handlers_of(event, groups, lines)
            broken = broken or group_broken
            for handler in handlers:
                broken = _check_handler(event, handler, lines) or broken
        for event in EVENT_ORDER:
            if event not in spec:
                lines.append(f"  note {event}: not installed (skipped at setup)")
    target = skills_dir if skills_dir is not None else agy_skills_dir()
    bundle = bundled_agy_skills()
    owned = [n for n in bundle if _is_owned_skill_dir(target / n)]
    lines.append(f"antigravity skills: {len(owned)}/{len(bundle)} in {target}")
    for name, text in sorted(bundle.items()):
        f = target / name / _SKILL_FILENAME
        try:
            if f.is_file() and f.read_text(encoding="utf-8") != text:
                lines.append(f"  stale skill {name}: rerun `aelf setup --host antigravity`")
        except OSError:
            continue
    for hook, reason in AGY_UNSUPPORTED.items():
        lines.append(f"unsupported on antigravity: {hook} — {reason}")
    return AgyDoctorResult(lines=tuple(lines), broken=broken)
