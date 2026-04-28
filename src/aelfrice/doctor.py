"""Diagnose Claude Code settings.json hook & statusline commands.

`aelf doctor` runs this module against the user-scope settings.json and
(when present) the project-scope settings.json under cwd. It walks
every command field that Claude Code is going to spawn -- across all
hook events plus the top-level statusLine -- and asks one question per
command: when Claude Code spawns this, will the OS find an executable?

The check is deliberately lossy: we extract the first whitespace token,
treat it as a program path or name, and verify either that the
absolute path exists and is executable OR that the bare name resolves
via $PATH. Shell-pipe constructs are skipped only when we cannot
identify a script path -- a `bash /abs/path.sh ...` wrapper is
inspected by extracting the script path even if `||`, `;`, etc.
appear later (issue #113: a stale `bash <missing>.sh 2>/dev/null
|| true` hook was silently skipped instead of flagged broken).

In addition to existence checks the report surfaces two soft
warnings:

* commands that wrap a script in the silent-failure pattern
  (`2>/dev/null || true`), which hides infrastructure failures from
  the user (issue #114);
* recent entries in `~/.aelfrice/logs/hook-failures.log`, the file
  hook bash wrappers should redirect stderr into instead of dropping
  it on the floor.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal, cast

from aelfrice.setup import PROJECT_SETTINGS_RELPATH, USER_SETTINGS_PATH


def _load_settings_json(path: Path) -> dict[str, object]:
    """Read settings.json. Empty / nonexistent files are treated as {}."""
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

Scope = Literal["user", "project"]

_HOOKS_KEY: Final[str] = "hooks"
_STATUSLINE_KEY: Final[str] = "statusLine"
_INNER_HOOKS_KEY: Final[str] = "hooks"
_TYPE_KEY: Final[str] = "type"
_COMMAND_KEY: Final[str] = "command"
_HOOK_TYPE_COMMAND: Final[str] = "command"

# Tokens that signal "this is a shell expression, do not try to verify
# the first program statically." We record these as 'skipped' UNLESS
# the command starts with a known interpreter and a script path can be
# extracted (handled separately).
_SHELL_INDICATORS: Final[tuple[str, ...]] = ("|", "&&", "||", "`", "$(", ";")

# Interpreters whose first non-flag argument is the script we should
# verify -- catches `bash /path/to/script.sh`, `sh ./tool.sh`, etc.
_SCRIPT_INTERPRETERS: Final[frozenset[str]] = frozenset(
    {"bash", "sh", "zsh", "python", "python3"}
)

# Substring marker for the silent-failure pattern: `bash foo
# 2>/dev/null || true` swallows every category of script breakage.
# We surface a soft warning when a hook command contains this, even
# if the underlying script resolves -- the pattern itself is the
# anti-feature (issue #114).
_SILENT_FAILURE_MARKER: Final[str] = "2>/dev/null || true"

# Where setup-installed bash hook wrappers should append stderr.
# `aelf doctor` reads (but does not create) this path.
HOOK_FAILURES_LOG: Final[Path] = (
    Path.home() / ".aelfrice" / "logs" / "hook-failures.log"
)
# How many trailing lines of the hook-failures log to surface.
_HOOK_FAILURES_TAIL: Final[int] = 10


@dataclass(frozen=True)
class CommandFinding:
    """One hook/statusline command we inspected.

    `location` is a human-readable JSON path like "hooks.PreToolUse[0].hooks[0]"
    or "statusLine". `command` is the raw command string. `program` is the
    first whitespace token (the executable). `status` is one of:
      * 'ok'      - program resolves to an existing executable.
      * 'broken'  - program does not resolve.
      * 'skipped' - command contains shell metacharacters; cannot statically verify.

    `silent_failure` is True when the command contains the
    `2>/dev/null || true` wrapper that hides script breakage from the
    user. Reported even when status is 'ok' (the wrapper itself is the
    anti-feature; issue #114).
    """
    settings_path: Path
    location: str
    command: str
    program: str
    status: Literal["ok", "broken", "skipped"]
    detail: str = ""
    silent_failure: bool = False


@dataclass
class DoctorReport:
    """Aggregate result of scanning one or more settings.json files."""
    scopes_scanned: list[tuple[Scope, Path]] = field(
        default_factory=lambda: cast(list[tuple[Scope, Path]], [])
    )
    findings: list[CommandFinding] = field(
        default_factory=lambda: cast(list[CommandFinding], [])
    )
    hook_failures_log: Path | None = None
    hook_failures_tail: tuple[str, ...] = ()
    # Slash commands installed under ~/.claude/commands/aelf/ that
    # name a subcommand the running `aelf` CLI does not implement
    # (issue #115 acceptance: surface the gap so a user on a branch
    # without a feature still sees the slash file but knows it'll
    # error out).
    orphan_slash_commands: list[str] = field(
        default_factory=lambda: cast(list[str], [])
    )

    @property
    def broken(self) -> list[CommandFinding]:
        return [f for f in self.findings if f.status == "broken"]

    @property
    def ok_count(self) -> int:
        return sum(1 for f in self.findings if f.status == "ok")

    @property
    def skipped_count(self) -> int:
        return sum(1 for f in self.findings if f.status == "skipped")

    @property
    def silent_failure(self) -> list[CommandFinding]:
        return [f for f in self.findings if f.silent_failure]


SLASH_COMMANDS_DIR_DEFAULT: Final[Path] = (
    Path.home() / ".claude" / "commands" / "aelf"
)


def diagnose(
    *,
    user_settings: Path | None = None,
    project_root: Path | None = None,
    hook_failures_log: Path | None = None,
    slash_commands_dir: Path | None = None,
    known_cli_subcommands: frozenset[str] | None = None,
) -> DoctorReport:
    """Walk user and project settings.json, return a DoctorReport.

    Defaults: user_settings -> ~/.claude/settings.json,
    project_root -> Path.cwd() (only scanned if .claude/settings.json
    exists there). When the file at `hook_failures_log` (default
    `~/.aelfrice/logs/hook-failures.log`) exists and is non-empty,
    the last few lines are surfaced in the report. When
    `known_cli_subcommands` is provided, doctor additionally checks
    the slash-commands directory (default `~/.claude/commands/aelf/`)
    for files naming subcommands the running CLI does not implement
    (issue #115).
    """
    user_path = user_settings if user_settings is not None else USER_SETTINGS_PATH
    project_path = (
        project_root if project_root is not None else Path.cwd()
    ) / PROJECT_SETTINGS_RELPATH
    report = DoctorReport()
    if user_path.exists():
        report.scopes_scanned.append(("user", user_path))
        report.findings.extend(_scan_settings(user_path))
    if project_path.exists():
        report.scopes_scanned.append(("project", project_path))
        report.findings.extend(_scan_settings(project_path))
    log_path = (
        hook_failures_log if hook_failures_log is not None else HOOK_FAILURES_LOG
    )
    report.hook_failures_log = log_path
    report.hook_failures_tail = _tail_log(log_path, _HOOK_FAILURES_TAIL)
    if known_cli_subcommands is not None:
        slash_dir = (
            slash_commands_dir if slash_commands_dir is not None
            else SLASH_COMMANDS_DIR_DEFAULT
        )
        report.orphan_slash_commands = _scan_orphan_slash_commands(
            slash_dir, known_cli_subcommands,
        )
    return report


def _tail_log(path: Path, n: int) -> tuple[str, ...]:
    """Return the trailing `n` non-empty lines of `path` (empty if missing).

    Read errors swallow to empty -- doctor is diagnostic, not authoritative.
    """
    try:
        if not path.exists():
            return ()
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ()
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    return tuple(lines[-n:])


def _scan_orphan_slash_commands(
    slash_dir: Path, known: frozenset[str],
) -> list[str]:
    """Return slash-command basenames whose CLI subcommand is missing.

    Ignores `aelf-*` files that wrap meta commands (none ship today
    but reserved). Returns sorted basenames so the CLI report stable.
    """
    if not slash_dir.is_dir():
        return []
    orphans: list[str] = []
    for md in sorted(slash_dir.glob("*.md")):
        sub = md.stem  # `ingest-transcript.md` -> `ingest-transcript`
        if sub in known:
            continue
        orphans.append(sub)
    return orphans


def _scan_settings(path: Path) -> list[CommandFinding]:
    """Yield findings for every hook command + the statusline in `path`."""
    try:
        data = _load_settings_json(path)
    except (ValueError, OSError):
        return [CommandFinding(
            settings_path=path, location="<root>", command="",
            program="", status="broken",
            detail="settings.json could not be parsed",
        )]
    findings: list[CommandFinding] = []
    findings.extend(_scan_hooks(path, data))
    findings.extend(_scan_statusline(path, data))
    return findings


def _scan_hooks(
    path: Path, data: dict[str, object]
) -> list[CommandFinding]:
    out: list[CommandFinding] = []
    hooks_obj = data.get(_HOOKS_KEY)
    if not isinstance(hooks_obj, dict):
        return out
    hooks_dict = cast(dict[str, object], hooks_obj)
    for event_name, event_list in hooks_dict.items():
        if not isinstance(event_list, list):
            continue
        for i, entry in enumerate(cast(list[object], event_list)):
            if not isinstance(entry, dict):
                continue
            entry_dict = cast(dict[str, object], entry)
            inner = entry_dict.get(_INNER_HOOKS_KEY)
            if not isinstance(inner, list):
                continue
            for j, hook in enumerate(cast(list[object], inner)):
                if not isinstance(hook, dict):
                    continue
                hook_dict = cast(dict[str, object], hook)
                if hook_dict.get(_TYPE_KEY) != _HOOK_TYPE_COMMAND:
                    continue
                cmd = hook_dict.get(_COMMAND_KEY)
                if not isinstance(cmd, str):
                    continue
                location = (
                    f"hooks.{event_name}[{i}].hooks[{j}]"
                )
                out.append(_inspect_command(path, location, cmd))
    return out


def _scan_statusline(
    path: Path, data: dict[str, object]
) -> list[CommandFinding]:
    sl = data.get(_STATUSLINE_KEY)
    if not isinstance(sl, dict):
        return []
    sl_dict = cast(dict[str, object], sl)
    cmd = sl_dict.get(_COMMAND_KEY)
    if not isinstance(cmd, str):
        return []
    return [_inspect_command(path, _STATUSLINE_KEY, cmd)]


def _inspect_command(
    settings_path: Path, location: str, command: str
) -> CommandFinding:
    """Categorise a single command string."""
    stripped = command.strip()
    silent = _SILENT_FAILURE_MARKER in stripped
    if not stripped:
        return CommandFinding(
            settings_path=settings_path, location=location,
            command=command, program="",
            status="broken",
            detail="empty command string",
            silent_failure=silent,
        )
    try:
        tokens = shlex.split(stripped)
    except ValueError:
        # Unparseable as shell tokens -- only safe to skip when no
        # interpreter+script is recognisable from a token prefix.
        return CommandFinding(
            settings_path=settings_path, location=location,
            command=command, program="",
            status="broken",
            detail="command is not parseable as shell tokens",
            silent_failure=silent,
        )
    if not tokens:
        return CommandFinding(
            settings_path=settings_path, location=location,
            command=command, program="",
            status="broken",
            detail="no program token",
            silent_failure=silent,
        )
    program = tokens[0]
    interpreter_basename = Path(program).name
    has_shell_meta = any(tok in stripped for tok in _SHELL_INDICATORS)
    if interpreter_basename in _SCRIPT_INTERPRETERS:
        # `bash /abs/path.sh 2>/dev/null || true` -- check the script
        # path even when shell metas appear later. The script vanishing
        # is the failure mode we care about (issue #113); the wrapper
        # is just noise around it.
        for tok in tokens[1:]:
            if tok.startswith("-"):
                continue
            if "/" in tok and not _is_shell_meta_token(tok):
                finding = _check_path(
                    settings_path, location, command, tok
                )
                if silent:
                    finding = _with_silent_failure(finding)
                return finding
            if _is_shell_meta_token(tok):
                # First non-flag token is shell-meta (e.g. `bash &&
                # foo`). Fall through to the generic skip.
                break
            break  # first non-flag, non-path argument: stop scanning
    if has_shell_meta:
        return CommandFinding(
            settings_path=settings_path, location=location,
            command=command, program="",
            status="skipped",
            detail="contains shell metacharacters; not statically checked",
            silent_failure=silent,
        )
    if "/" in program:
        finding = _check_path(settings_path, location, command, program)
        return _with_silent_failure(finding) if silent else finding
    # Bare name -- $PATH lookup.
    resolved = shutil.which(program)
    if resolved is None:
        return CommandFinding(
            settings_path=settings_path, location=location,
            command=command, program=program, status="broken",
            detail=f"{program!r} not on $PATH",
            silent_failure=silent,
        )
    return CommandFinding(
        settings_path=settings_path, location=location,
        command=command, program=program, status="ok",
        silent_failure=silent,
    )


def _is_shell_meta_token(tok: str) -> bool:
    return tok in {"|", "||", "&&", ";", "&", "`"}


def _with_silent_failure(f: CommandFinding) -> CommandFinding:
    """Return a copy of `f` with silent_failure=True (frozen dataclass)."""
    return CommandFinding(
        settings_path=f.settings_path, location=f.location,
        command=f.command, program=f.program, status=f.status,
        detail=f.detail, silent_failure=True,
    )


def _check_path(
    settings_path: Path, location: str, command: str, program: str
) -> CommandFinding:
    """Existence + executable-bit check for a path-shaped program token."""
    prog_path = Path(program)
    if prog_path.is_file() and os.access(prog_path, os.X_OK):
        return CommandFinding(
            settings_path=settings_path, location=location,
            command=command, program=program, status="ok",
        )
    return CommandFinding(
        settings_path=settings_path, location=location,
        command=command, program=program, status="broken",
        detail=(
            "path does not exist"
            if not prog_path.exists()
            else "exists but is not executable"
        ),
    )


def format_report(report: DoctorReport) -> str:
    """Render a DoctorReport as a human-readable string for the CLI."""
    lines: list[str] = []
    if not report.scopes_scanned:
        lines.append(
            "no settings.json found at user or project scope -- nothing to check"
        )
        return "\n".join(lines)
    for scope, path in report.scopes_scanned:
        lines.append(f"scanned {scope}: {path}")
    lines.append("")
    lines.append(
        f"summary: {report.ok_count} ok, "
        f"{len(report.broken)} broken, {report.skipped_count} skipped"
    )
    if report.broken:
        lines.append("")
        lines.append("broken commands:")
        for f in report.broken:
            lines.append(
                f"  - {f.settings_path}:{f.location}"
            )
            lines.append(f"      command:  {f.command}")
            lines.append(f"      program:  {f.program or '(empty)'}")
            lines.append(f"      issue:    {f.detail}")
        lines.append("")
        lines.append(
            "fix: run 'aelf setup' from the project venv to rewrite the "
            "hook command, or edit the affected settings.json by hand."
        )
    if report.silent_failure:
        lines.append("")
        lines.append(
            "silent-failure pattern (`2>/dev/null || true`) hides script "
            "breakage from you:"
        )
        for f in report.silent_failure:
            lines.append(
                f"  - {f.settings_path}:{f.location}"
            )
            lines.append(f"      command:  {f.command}")
        lines.append(
            "fix: rewrite the hook command to redirect stderr to "
            f"{HOOK_FAILURES_LOG} (use `>>` not `>`), or remove the "
            "wrapper entirely if the script is meant to surface errors."
        )
    if report.hook_failures_tail:
        lines.append("")
        lines.append(
            f"recent hook failures ({report.hook_failures_log}):"
        )
        for entry in report.hook_failures_tail:
            lines.append(f"  {entry}")
    if report.orphan_slash_commands:
        lines.append("")
        lines.append(
            "slash commands installed but missing from the active CLI "
            "(running them will print 'invalid choice' errors):"
        )
        for sub in report.orphan_slash_commands:
            lines.append(f"  - /aelf:{sub}  (no `aelf {sub}` subcommand)")
        lines.append(
            "fix: upgrade aelfrice (`aelf upgrade`) so the slash file's "
            "feature is available, or remove the stale slash file."
        )
    return "\n".join(lines)
