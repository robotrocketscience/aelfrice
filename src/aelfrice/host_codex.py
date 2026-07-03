"""Codex host target for setup / doctor / uninstall (#1052).

Writes, validates, and removes the aelfrice hook set in Codex's
``~/.codex/hooks.json``. Design constraints established in #1054/#1052
triage:

- **hooks.json only.** The documented ``{"hooks": {"<Event>": [...]}}``
  shape is the stable public surface. The per-hook trust ledger
  (``[hooks.state]`` tables in ``config.toml``, keyed positionally with
  a ``trusted_hash`` over a canonical TOML serialization) is explicitly
  marked for replacement in the Codex source — we never write it.
  Setup instead prints approval guidance: the user runs ``/hooks``
  inside a Codex session to trust the new entries. Until approved (and
  until the ``codex_hooks`` feature flag is on), Codex silently skips
  the hooks — doctor surfaces both conditions.
- **Merge-aware and idempotent.** Entries whose command basename is one
  of ours are replaced wholesale on every setup run; everything else in
  the file is preserved byte-for-byte at the JSON level. An unparseable
  hooks.json is never overwritten without ``force`` — a real-world
  ``~/.codex/hooks.json`` has been observed holding truncated JSON, and
  clobbering user content on a parse error is worse than refusing.
- **Portable hook subset.** Only host-agnostic hooks are installed:
  retrieval injection (UserPromptSubmit), the transcript logger
  (UserPromptSubmit / Stop / PreCompact / PostCompact), session-start
  baseline injection (SessionStart, all sources — ``compact`` included,
  which is the rebuild-at-compaction channel per #1054), and the stop
  lock-prompt. Tool-matcher hooks (PreToolUse/PostToolUse) assume the
  Claude tool namespace and are tracked separately in #1055.
"""
from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, cast

from aelfrice.setup import (
    SettingsScope,
    resolve_hook_command,
    resolve_session_start_hook_command,
    resolve_stop_hook_command,
    resolve_transcript_logger_command,
)

CODEX_DIR: Final[Path] = Path.home() / ".codex"
CODEX_HOOKS_FILENAME: Final[str] = "hooks.json"
CODEX_CONFIG_FILENAME: Final[str] = "config.toml"

# SessionStart matcher covering every source, so the baseline block fires
# on fresh sessions and the rebuild block fires post-compaction (#1054:
# SessionStart(source=="compact") is Codex's only compaction-injection
# channel — PreCompact/PostCompact stdout is ignored by the harness).
_SESSION_START_MATCHER: Final[str] = "startup|resume|clear|compact"

# Basenames owned by aelfrice; setup replaces exactly these on re-run and
# uninstall removes exactly these. Mirrors the manifest-ownership posture
# of auto_install (additive within owned basenames).
_OWNED_BASENAMES: Final[frozenset[str]] = frozenset({
    "aelf-hook",
    "aelf-transcript-logger",
    "aelf-session-start-hook",
    "aelf-stop-hook",
})


def codex_hooks_path(codex_dir: Path | None = None) -> Path:
    return (codex_dir if codex_dir is not None else CODEX_DIR) / CODEX_HOOKS_FILENAME


def codex_config_path(codex_dir: Path | None = None) -> Path:
    return (codex_dir if codex_dir is not None else CODEX_DIR) / CODEX_CONFIG_FILENAME


def _handler(command: str, *, timeout: int | None = None) -> dict[str, object]:
    entry: dict[str, object] = {"type": "command", "command": command}
    if timeout is not None:
        entry["timeout"] = timeout
    return entry


def desired_codex_hooks(scope: SettingsScope = "user") -> dict[str, list[dict[str, object]]]:
    """The aelfrice hook set in Codex hooks.json shape, keyed by event.

    Commands resolve to absolute paths via the same resolvers the Claude
    installers use, so both hosts pin identical executables.
    """
    hook_cmd = resolve_hook_command(scope)
    logger_cmd = resolve_transcript_logger_command(scope)
    session_cmd = resolve_session_start_hook_command(scope)
    stop_cmd = resolve_stop_hook_command(scope)
    return {
        "UserPromptSubmit": [
            {"hooks": [_handler(hook_cmd), _handler(logger_cmd)]},
        ],
        "Stop": [
            {"hooks": [_handler(logger_cmd), _handler(stop_cmd)]},
        ],
        "PreCompact": [
            {"hooks": [_handler(logger_cmd)]},
        ],
        "PostCompact": [
            {"hooks": [_handler(logger_cmd)]},
        ],
        "SessionStart": [
            {
                "matcher": _SESSION_START_MATCHER,
                "hooks": [_handler(session_cmd)],
            },
        ],
    }


def _command_basename(handler: object) -> str:
    """Basename of a handler's command's first token, '' on shape miss."""
    if not isinstance(handler, dict):
        return ""
    hd = cast(dict[str, object], handler)
    cmd = hd.get("command")
    if not isinstance(cmd, str) or not cmd.strip():
        return ""
    return Path(cmd.split()[0]).name


def _group_is_owned(group: object) -> bool:
    """A matcher group is aelfrice's iff every handler in it is ours.

    Mixed groups (user handler + aelfrice handler in one group) are left
    untouched — we never edit inside someone else's group.
    """
    if not isinstance(group, dict):
        return False
    gd = cast(dict[str, object], group)
    handlers = gd.get("hooks")
    if not isinstance(handlers, list) or not handlers:
        return False
    return all(
        _command_basename(h) in _OWNED_BASENAMES
        for h in cast(list[object], handlers)
    )


@dataclass
class CodexInstallResult:
    path: Path
    changed: bool
    installed_events: list[str] = field(default_factory=list[str])
    guidance: list[str] = field(default_factory=list[str])
    error: str | None = None


def install_codex_hooks(
    hooks_path: Path,
    *,
    scope: SettingsScope = "user",
    force: bool = False,
) -> CodexInstallResult:
    """Write the aelfrice hook set into ``hooks_path``, merge-aware.

    Refuses (with ``error`` set) when the existing file is unparseable
    and ``force`` is False; ``force`` replaces the broken file with a
    fresh aelfrice-only document.
    """
    existing: dict[str, object] = {}
    if hooks_path.is_file():
        try:
            parsed = json.loads(hooks_path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                existing = cast(dict[str, object], parsed)
            elif not force:
                return CodexInstallResult(
                    path=hooks_path, changed=False,
                    error="existing hooks.json is not a JSON object; "
                          "re-run with --force to replace it",
                )
        except json.JSONDecodeError as exc:
            if not force:
                return CodexInstallResult(
                    path=hooks_path, changed=False,
                    error=f"existing hooks.json is invalid JSON ({exc}); "
                          "re-run with --force to replace it",
                )
            existing = {}

    hooks_obj = existing.get("hooks")
    hooks_map: dict[str, object] = (
        cast(dict[str, object], hooks_obj) if isinstance(hooks_obj, dict) else {}
    )

    desired = desired_codex_hooks(scope)
    before = json.dumps({"hooks": hooks_map}, sort_keys=True)
    for event, groups in desired.items():
        current = hooks_map.get(event)
        current_list = (
            cast(list[object], current) if isinstance(current, list) else []
        )
        kept = [g for g in current_list if not _group_is_owned(g)]
        hooks_map[event] = kept + cast(list[object], groups)
    existing["hooks"] = hooks_map
    after = json.dumps({"hooks": hooks_map}, sort_keys=True)

    changed = before != after or not hooks_path.is_file()
    if changed:
        hooks_path.parent.mkdir(parents=True, exist_ok=True)
        hooks_path.write_text(
            json.dumps(existing, indent=2) + "\n", encoding="utf-8",
        )
    return CodexInstallResult(
        path=hooks_path,
        changed=changed,
        installed_events=sorted(desired.keys()),
        guidance=[
            "Codex runs hooks only after per-hook trust approval: open a "
            "Codex session and run /hooks to approve the new entries.",
            "The codex_hooks feature flag must be enabled "
            "(codex features enable codex_hooks) — it is off by default "
            "while the hooks surface is under development upstream.",
        ],
    )


def remove_codex_hooks(hooks_path: Path) -> CodexInstallResult:
    """Remove aelfrice-owned matcher groups; drop emptied events.

    A missing or unparseable file is reported, not modified — uninstall
    never destroys content it cannot positively identify as ours.
    """
    if not hooks_path.is_file():
        return CodexInstallResult(path=hooks_path, changed=False)
    try:
        parsed = json.loads(hooks_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return CodexInstallResult(
            path=hooks_path, changed=False,
            error=f"hooks.json is invalid JSON ({exc}); not modified",
        )
    if not isinstance(parsed, dict):
        return CodexInstallResult(
            path=hooks_path, changed=False,
            error="hooks.json is not a JSON object; not modified",
        )
    doc = cast(dict[str, object], parsed)
    hooks_obj = doc.get("hooks")
    if not isinstance(hooks_obj, dict):
        return CodexInstallResult(path=hooks_path, changed=False)
    hooks_map = cast(dict[str, object], hooks_obj)

    changed = False
    removed_events: list[str] = []
    for event in list(hooks_map.keys()):
        groups = hooks_map[event]
        if not isinstance(groups, list):
            continue
        kept = [g for g in cast(list[object], groups) if not _group_is_owned(g)]
        if len(kept) != len(cast(list[object], groups)):
            changed = True
            removed_events.append(event)
            if kept:
                hooks_map[event] = kept
            else:
                del hooks_map[event]
    if changed:
        hooks_path.write_text(
            json.dumps(doc, indent=2) + "\n", encoding="utf-8",
        )
    return CodexInstallResult(
        path=hooks_path, changed=changed, installed_events=sorted(removed_events),
    )


@dataclass
class CodexDoctorReport:
    """Structured result of the Codex host scan; render at the CLI."""

    codex_dir_present: bool
    hooks_file_present: bool = False
    hooks_file_valid: bool = False
    parse_error: str | None = None
    owned_handler_count: int = 0
    missing_events: list[str] = field(default_factory=list[str])
    stale_commands: list[str] = field(default_factory=list[str])
    feature_flag_on: bool | None = None
    trusted_state_entries: int = 0
    warnings: list[str] = field(default_factory=list[str])


def doctor_codex(codex_dir: Path | None = None) -> CodexDoctorReport:
    """Scan the Codex host: hooks.json shape, coverage, flag, trust.

    Read-only. Reports rather than raises; the CLI decides exit codes.
    """
    cdir = codex_dir if codex_dir is not None else CODEX_DIR
    report = CodexDoctorReport(codex_dir_present=cdir.is_dir())
    if not report.codex_dir_present:
        report.warnings.append(f"{cdir} not found — Codex not installed?")
        return report

    hooks_path = codex_hooks_path(cdir)
    report.hooks_file_present = hooks_path.is_file()
    hooks_map: dict[str, object] = {}
    if report.hooks_file_present:
        try:
            parsed = json.loads(hooks_path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                report.hooks_file_valid = True
                obj = cast(dict[str, object], parsed).get("hooks")
                if isinstance(obj, dict):
                    hooks_map = cast(dict[str, object], obj)
            else:
                report.parse_error = "top-level JSON is not an object"
        except json.JSONDecodeError as exc:
            report.parse_error = str(exc)
        if not report.hooks_file_valid:
            report.warnings.append(
                f"{hooks_path} is unreadable as hooks config "
                f"({report.parse_error}); Codex will ignore or reject it",
            )

    expected_events = set(desired_codex_hooks().keys())
    covered: set[str] = set()
    for event, groups in hooks_map.items():
        if not isinstance(groups, list):
            continue
        for group in cast(list[object], groups):
            if not _group_is_owned(group):
                continue
            covered.add(event)
            gd = cast(dict[str, object], group)
            for handler in cast(list[object], gd.get("hooks", [])):
                report.owned_handler_count += 1
                hd = cast(dict[str, object], handler)
                cmd = hd.get("command")
                if isinstance(cmd, str):
                    exe = Path(cmd.split()[0])
                    if exe.is_absolute() and not exe.exists():
                        report.stale_commands.append(cmd)
    report.missing_events = sorted(expected_events - covered)
    for cmd in report.stale_commands:
        report.warnings.append(f"hook command not found on disk: {cmd}")
    if report.owned_handler_count and report.missing_events:
        report.warnings.append(
            "aelfrice hook coverage incomplete; missing events: "
            + ", ".join(report.missing_events),
        )

    config_path = codex_config_path(cdir)
    if config_path.is_file():
        try:
            cfg = tomllib.loads(config_path.read_text(encoding="utf-8"))
        except (tomllib.TOMLDecodeError, OSError) as exc:
            report.warnings.append(f"could not parse {config_path}: {exc}")
            cfg = {}
        features = cfg.get("features")
        if isinstance(features, dict):
            flag = cast(dict[str, object], features).get("codex_hooks")
            report.feature_flag_on = flag is True
        else:
            report.feature_flag_on = False
        hooks_cfg = cfg.get("hooks")
        if isinstance(hooks_cfg, dict):
            state = cast(dict[str, object], hooks_cfg).get("state")
            if isinstance(state, dict):
                for entry in cast(dict[str, object], state).values():
                    if (
                        isinstance(entry, dict)
                        and cast(dict[str, object], entry).get("trusted_hash")
                    ):
                        report.trusted_state_entries += 1
    if report.feature_flag_on is False:
        report.warnings.append(
            "codex_hooks feature flag is off — Codex will not run any "
            "hooks (enable: codex features enable codex_hooks)",
        )
    if (
        report.owned_handler_count
        and report.trusted_state_entries < report.owned_handler_count
    ):
        report.warnings.append(
            f"{report.owned_handler_count} aelfrice handler(s) configured "
            f"but only {report.trusted_state_entries} trusted "
            "[hooks.state] entr(ies) exist — untrusted hooks are "
            "silently skipped; run /hooks in a Codex session to approve",
        )
    return report
