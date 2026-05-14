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

`aelf doctor --classify-orphans` (issue #206) finds beliefs whose
`type` was never resolved (type = 'unknown') AND that have never
received any feedback (alpha + beta <= 2 — the untouched prior), then
re-classifies them through the same Haiku batch path used by
`aelf onboard --llm-classify`.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import re
import shlex
import shutil
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, cast

from aelfrice.setup import PROJECT_SETTINGS_RELPATH, USER_SETTINGS_PATH

if TYPE_CHECKING:
    from aelfrice.store import MemoryStore


# ---------------------------------------------------------------------------
# Search-tool telemetry section (v1.5.0 #155 AC8)
# ---------------------------------------------------------------------------

SEARCH_TOOL_TELEMETRY_SUBPATH: Final[str] = (
    "aelfrice/telemetry/search_tool_hook.jsonl"
)


@dataclass(frozen=True)
class SearchToolTelemetryStats:
    """Rolling statistics derived from the Bash matcher telemetry file.

    `fire_count` is the total number of records in the ring buffer.
    `p50_ms` and `p95_ms` are the 50th- and 95th-percentile latency
    in milliseconds over the buffer. `noise_rate` is the fraction of
    fires that returned zero L0 + L1 results (0.0 – 1.0).
    """
    fire_count: int
    p50_ms: float
    p95_ms: float
    noise_rate: float


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Nearest-rank percentile on a sorted list. Returns 0.0 for empty."""
    if not sorted_values:
        return 0.0
    idx = max(0, int(len(sorted_values) * pct / 100.0) - 1)
    return sorted_values[idx]


def diagnose_search_tool_telemetry(
    telemetry_path: Path,
) -> SearchToolTelemetryStats | None:
    """Read the Bash matcher telemetry ring buffer and return rolling stats.

    Returns `None` when the file does not exist or is empty (caller
    prints the "no fires recorded" sentinel). Raises `ValueError` when
    the file exists but contains malformed JSON (real corruption).
    """
    from aelfrice.hook_search_tool import read_telemetry  # noqa: PLC0415

    records = read_telemetry(telemetry_path)  # propagates ValueError on corruption
    if not records:
        return None

    latencies: list[float] = sorted(
        float(r.get("latency_ms", 0.0)) for r in records
    )
    noise_count = sum(
        1 for r in records
        if int(r.get("injected_l0", 0)) == 0 and int(r.get("injected_l1", 0)) == 0
    )
    return SearchToolTelemetryStats(
        fire_count=len(records),
        p50_ms=_percentile(latencies, 50),
        p95_ms=_percentile(latencies, 95),
        noise_rate=noise_count / len(records),
    )


# ---------------------------------------------------------------------------
# UserPromptSubmit telemetry section (#218 AC4)
# ---------------------------------------------------------------------------

USER_PROMPT_SUBMIT_TELEMETRY_SUBPATH: Final[str] = (
    "aelfrice/telemetry/user_prompt_submit.jsonl"
)


@dataclass(frozen=True)
class UserPromptSubmitTelemetryStats:
    """Rolling statistics derived from the UserPromptSubmit telemetry file.

    `fire_count` is the total number of records in the ring buffer.
    `p50_chars` and `p95_chars` are the 50th- and 95th-percentile
    injection size in characters. `median_collapse_rate` is the median
    ratio of n_returned / n_unique_content_hashes across all records
    (1.0 means no duplicates seen).
    """

    fire_count: int
    p50_chars: float
    p95_chars: float
    median_collapse_rate: float


def diagnose_user_prompt_submit_telemetry(
    telemetry_path: Path,
) -> UserPromptSubmitTelemetryStats | None:
    """Read the UserPromptSubmit telemetry ring buffer and return rolling stats.

    Returns `None` when the file does not exist or is empty. Raises
    `ValueError` when the file exists but contains malformed JSON.
    """
    from aelfrice.hook import read_user_prompt_submit_telemetry  # noqa: PLC0415

    records = read_user_prompt_submit_telemetry(telemetry_path)
    if not records:
        return None

    chars: list[float] = sorted(
        float(r.get("total_chars", 0)) for r in records
    )
    collapse_rates: list[float] = []
    for r in records:
        n_ret = int(r.get("n_returned", 0))
        n_uniq = int(r.get("n_unique_content_hashes", 0))
        if n_uniq > 0:
            collapse_rates.append(n_ret / n_uniq)
        else:
            collapse_rates.append(1.0)
    collapse_rates.sort()

    return UserPromptSubmitTelemetryStats(
        fire_count=len(records),
        p50_chars=_percentile(chars, 50),
        p95_chars=_percentile(chars, 95),
        median_collapse_rate=_percentile(collapse_rates, 50),
    )


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

# Root directory under which per-project aelfrice state lives.
# Each sub-directory is a project-id slug; memory.db sits directly inside.
# Override via `aelfrice_projects_dir` kwarg on `diagnose()` (tests use this).
_AELFRICE_PROJECTS_DIR: Final[Path] = (
    Path.home() / ".aelfrice" / "projects"
)


@dataclass(frozen=True)
class LegacySchemaDB:
    """One per-project DB detected as pre-v1.x (no `origin` column).

    `path`      — absolute path to the memory.db file.
    `row_count` — number of rows in the `beliefs` table.
    `idle_days` — whole days since the file was last modified (mtime).
    """
    path: Path
    row_count: int
    idle_days: int


@dataclass(frozen=True)
class DormantDB:
    """One per-project DB detected as idle for >= the dormancy threshold (#594).

    `path`        — absolute path to the memory.db file.
    `row_count`   — number of rows in the `beliefs` table (0 when the
                    table is missing or empty; both still count as
                    pruneable when the file itself is dormant).
    `idle_days`   — whole days since the file was last modified (mtime).
    `size_bytes`  — file size of the memory.db, surfaced so the user
                    knows what storage they reclaim by pruning.
    """
    path: Path
    row_count: int
    idle_days: int
    size_bytes: int


@dataclass(frozen=True)
class MigratedDB:
    """One per-project DB auto-migrated in place to the modern schema (#593).

    `path`        — final path of the now-modern-schema DB.
    `backup_path` — sibling preserving the original pre-v1.x file.
    `row_count`   — beliefs inserted into the modern-schema DB.
    `duration_ms` — wall-clock time for the migration.
    """
    path: Path
    backup_path: Path
    row_count: int
    duration_ms: int


@dataclass(frozen=True)
class FailedMigrateDB:
    """One per-project DB that auto-migrate could not action (#593).

    `path`   — the legacy DB path; the file is untouched after failure.
    `reason` — short label (exception class name) for the failure mode.
    """
    path: Path
    reason: str


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
    # v1.5.0 #155 AC8: search_tool_hook telemetry stats.
    # None  → telemetry section not requested or telemetry file absent.
    # SearchToolTelemetryStats → computed stats from the ring buffer.
    search_tool_telemetry: SearchToolTelemetryStats | None = None
    # Path to the telemetry file that was (or would be) read.
    search_tool_telemetry_path: Path | None = None
    # True when the file existed but was malformed (ValueError from read).
    search_tool_telemetry_corrupt: bool = False
    # #218 AC4: user_prompt_submit_hook telemetry stats.
    user_prompt_submit_telemetry: UserPromptSubmitTelemetryStats | None = None
    user_prompt_submit_telemetry_path: Path | None = None
    user_prompt_submit_telemetry_corrupt: bool = False
    # Runtime deps declared in pyproject.toml that are not importable
    # in the current environment (issue #236: stale uv tool env).
    missing_runtime_deps: list[str] = field(
        default_factory=lambda: cast(list[str], [])
    )
    # Default-on auto-capture hook basenames (since v2.1, #529) that
    # are absent from every scanned settings.json. Pre-v2.1 installs
    # that ran `aelf setup` before the default flip end up here
    # (#557): they have retrieval-only wiring and need a re-run.
    missing_auto_capture_hooks: list[str] = field(
        default_factory=lambda: cast(list[str], [])
    )
    # Per-project DBs under ~/.aelfrice/projects/*/memory.db that use
    # the pre-v1.x schema (no `origin` column on the `beliefs` table)
    # and have at least one row. These DBs cannot participate in the
    # v2.x lifecycle (agent_remembered, user_validated, calibrated
    # weights, aelf:promote) without `aelf migrate` (#589). The field
    # holds the pre-migration detection set; after `diagnose()` runs
    # the auto-migrate pass (#593), check `migrated_dbs` for success
    # outcomes and `failed_migrate_dbs` for the residual nag.
    legacy_schema_dbs: list[LegacySchemaDB] = field(
        default_factory=lambda: cast(list[LegacySchemaDB], [])
    )
    # Per-project DBs that the auto-migrate pass brought forward to
    # the modern schema this run (#593). Each entry preserves the
    # backup path so the operator can recover the pre-migration file
    # if anything looks off after the fact.
    migrated_dbs: list[MigratedDB] = field(
        default_factory=lambda: cast(list[MigratedDB], [])
    )
    # Per-project DBs detected as legacy but where the auto-migrate
    # pass raised (e.g. backup target already exists; sqlite read
    # errors). The legacy file is untouched on failure; the operator
    # can investigate manually and re-run.
    failed_migrate_dbs: list[FailedMigrateDB] = field(
        default_factory=lambda: cast(list[FailedMigrateDB], [])
    )
    # HRR persistence state (#696). Keys: enabled (bool), dir (Path|None),
    # on_disk_bytes (int), reason (str|None), last_build_seconds (float|None).
    # None when the HRR persist probe was not requested (no store_path).
    hrr_persist_state: dict[str, object] | None = None

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
    search_tool_telemetry_path: Path | None = None,
    user_prompt_submit_telemetry_path: Path | None = None,
    aelfrice_projects_dir: Path | None = None,
    hrr_store_path: str | None = None,
    hrr_dim: int = 512,
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
    (issue #115). When `search_tool_telemetry_path` is provided (or
    derivable from the project root's git-common-dir), the
    search_tool_hook telemetry section is populated. Similarly for
    `user_prompt_submit_telemetry_path` (#218 AC4).
    `aelfrice_projects_dir` overrides the default scan root for
    per-project DBs (`~/.aelfrice/projects`); useful in tests (#589).
    `hrr_store_path` enables the HRR persist-state block (#696) — pass
    the resolved DB path so doctor can probe `_resolve_persist_dir`.
    `hrr_dim` is the HRR dimension (default 512, matches DEFAULT_DIM).
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
    report.missing_runtime_deps = _check_runtime_deps()
    report.missing_auto_capture_hooks = _check_auto_capture_hooks(report.findings)
    # #589: scan per-project DBs for pre-v1.x schema (no `origin` column).
    _proj_dir = (
        aelfrice_projects_dir
        if aelfrice_projects_dir is not None
        else _AELFRICE_PROJECTS_DIR
    )
    report.legacy_schema_dbs = _check_legacy_schema_dbs(projects_dir=_proj_dir)
    # #593: auto-migrate any detected legacy DBs in place. Operator
    # decision was "no prompt, no banner" — silent migration with a
    # `.pre-v1x.bak` backup hop. Failures degrade to the residual
    # `failed_migrate_dbs` nag.
    (
        report.migrated_dbs,
        report.failed_migrate_dbs,
    ) = _auto_migrate_legacy_dbs(report.legacy_schema_dbs)
    if known_cli_subcommands is not None:
        slash_dir = (
            slash_commands_dir if slash_commands_dir is not None
            else SLASH_COMMANDS_DIR_DEFAULT
        )
        report.orphan_slash_commands = _scan_orphan_slash_commands(
            slash_dir, known_cli_subcommands,
        )
    # v1.5.0 #155 AC8: populate search_tool_hook telemetry section.
    tel_path = search_tool_telemetry_path
    if tel_path is None:
        resolved_root = project_root if project_root is not None else Path.cwd()
        tel_path = _derive_telemetry_path(
            resolved_root, SEARCH_TOOL_TELEMETRY_SUBPATH,
        )
    if tel_path is not None:
        report.search_tool_telemetry_path = tel_path
        try:
            report.search_tool_telemetry = diagnose_search_tool_telemetry(tel_path)
        except ValueError:
            report.search_tool_telemetry_corrupt = True
    # #218 AC4: populate user_prompt_submit_hook telemetry section.
    ups_tel_path = user_prompt_submit_telemetry_path
    if ups_tel_path is None:
        resolved_root = project_root if project_root is not None else Path.cwd()
        ups_tel_path = _derive_telemetry_path(
            resolved_root, USER_PROMPT_SUBMIT_TELEMETRY_SUBPATH,
        )
    if ups_tel_path is not None:
        report.user_prompt_submit_telemetry_path = ups_tel_path
        try:
            report.user_prompt_submit_telemetry = (
                diagnose_user_prompt_submit_telemetry(ups_tel_path)
            )
        except ValueError:
            report.user_prompt_submit_telemetry_corrupt = True
    # #696: HRR persist-state block — populate when hrr_store_path is
    # provided. Uses a transient HRRStructIndexCache instance (read-only,
    # does not trigger any build or WARNING log).
    if hrr_store_path is not None:
        report.hrr_persist_state = _diagnose_hrr_persist(
            hrr_store_path, hrr_dim,
        )
    return report


def _diagnose_hrr_persist(
    store_path: str,
    dim: int,
) -> dict[str, object]:
    """Return the HRR persist-state dict for the given store path (#696).

    Constructs a transient ``HRRStructIndexCache`` against an in-memory
    store (no build, no I/O except the persist-dir size check) and calls
    ``resolve_persist_state()``. Merges ``last_build_seconds`` from the
    module-level slot.

    Imported lazily to avoid importing numpy at doctor import time.
    """
    try:
        from aelfrice.hrr_index import (  # noqa: PLC0415
            HRRStructIndexCache,
            last_build_seconds as _last_build_seconds,
        )
        from aelfrice.store import MemoryStore as _MemoryStore  # noqa: PLC0415
        _store = _MemoryStore(":memory:")
        try:
            cache = HRRStructIndexCache(
                store=_store, dim=dim, store_path=store_path,
            )
            state: dict[str, object] = dict(cache.resolve_persist_state())
            state["last_build_seconds"] = _last_build_seconds()
        finally:
            _store.close()
    except Exception:  # noqa: BLE001 — doctor never crashes on this
        state = {
            "enabled": False,
            "dir": None,
            "on_disk_bytes": 0,
            "reason": "error",
            "last_build_seconds": None,
        }
    return state


def _derive_telemetry_path(
    project_root: Path,
    subpath: str = SEARCH_TOOL_TELEMETRY_SUBPATH,
) -> Path | None:
    """Best-effort: locate a telemetry file under the project's git-common-dir.

    `subpath` is appended to the git-common-dir (e.g.
    `aelfrice/telemetry/search_tool_hook.jsonl`). Falls back to None if
    the git invocation fails or the project is not in a git repo.
    """
    try:
        import subprocess  # noqa: PLC0415
        result = subprocess.run(
            ["git", "-C", str(project_root),
             "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True, text=True, check=False, timeout=5,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        git_common = Path(result.stdout.strip()).resolve()
        return git_common / subpath
    except Exception:
        return None


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


def _check_runtime_deps() -> list[str]:
    """Return the names of declared runtime deps that are not importable.

    Reads the installed package metadata for 'aelfrice' via
    importlib.metadata, parses each PEP 508 requirement to extract the
    top-level distribution name, converts it to an import name (hyphens
    to underscores), and tries importing it. Returns sorted list of
    missing import names.

    Swallows all errors so doctor never crashes on unusual envs.
    """
    try:
        reqs = importlib.metadata.requires("aelfrice") or []
    except importlib.metadata.PackageNotFoundError:
        return []
    missing: list[str] = []
    # Only check unconditional (non-extra) deps.
    _extra_re = re.compile(r'extra\s*==', re.IGNORECASE)
    for req_str in reqs:
        # Skip extras / optional deps (lines with '; extra ==' marker).
        if _extra_re.search(req_str):
            continue
        # Extract the distribution name: first token before any version
        # specifier or environment marker.
        dist_name = re.split(r'[\s;>=<!(\[]', req_str)[0].strip()
        if not dist_name:
            continue
        # Normalise dist name to import name: hyphens -> underscores.
        import_name = dist_name.replace("-", "_")
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(dist_name)
    return sorted(missing)


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


# ---------------------------------------------------------------------------
# Prune dead `aelf-*` hook entries from settings.json (#781).
# ---------------------------------------------------------------------------

# Basename prefix that marks an entry as one this package installed.
# Custom hooks (user-installed shell scripts, third-party integrations)
# are left strictly alone.
_AELF_HOOK_BASENAME_PREFIX: Final[str] = "aelf-"


@dataclass(frozen=True)
class HookPruneResult:
    """Outcome of `prune_broken_aelf_hooks`.

    `removed_per_event` maps each hook event name (e.g. `"PreToolUse"`)
    to the number of parent entries removed from it. Events with no
    removals are absent.

    `total_removed` is the sum across events — zero when the file is
    already clean.
    """

    settings_path: Path
    removed_per_event: dict[str, int]
    total_removed: int


def prune_broken_aelf_hooks(
    settings_path: Path, *, dry_run: bool = False,
) -> HookPruneResult:
    """Drop hook entries whose `aelf-*` program no longer resolves.

    Walks every `hooks.<event>[i].hooks[j]` command in `settings_path`.
    An entry is removed when ALL of:

    * its inner command's first whitespace token has a basename
      starting with `aelf-` (so `aelf-hook`, `aelf-stop-hook`, etc. are
      in scope; bare shell scripts, `bash`, custom integrations are
      not);
    * `_inspect_command` classifies it `status="broken"` (the program
      path / `$PATH` lookup fails).

    Custom shell hooks (`gh-pii-guard.sh`, `conversation-logger.sh`,
    etc.) and statuslines are never touched. Skipped (shell-meta)
    commands are not pruned — the predicate is conservative.

    When `dry_run=True`, the file is not rewritten; the result still
    reports what *would* have been removed. Missing files return a
    zero-removal result.
    """
    if not settings_path.exists():
        return HookPruneResult(
            settings_path=settings_path, removed_per_event={}, total_removed=0,
        )
    try:
        data = _load_settings_json(settings_path)
    except (ValueError, OSError):
        return HookPruneResult(
            settings_path=settings_path, removed_per_event={}, total_removed=0,
        )
    hooks_obj = data.get(_HOOKS_KEY)
    if not isinstance(hooks_obj, dict):
        return HookPruneResult(
            settings_path=settings_path, removed_per_event={}, total_removed=0,
        )
    hooks_dict = cast(dict[str, object], hooks_obj)
    removed_per_event: dict[str, int] = {}
    for event_name, event_list in list(hooks_dict.items()):
        if not isinstance(event_list, list):
            continue
        entries = cast(list[object], event_list)
        kept: list[object] = []
        n_removed = 0
        for entry in entries:
            if _entry_is_broken_aelf_hook(settings_path, entry):
                n_removed += 1
                continue
            kept.append(entry)
        if n_removed:
            removed_per_event[event_name] = n_removed
            event_list[:] = kept
    total = sum(removed_per_event.values())
    if total and not dry_run:
        _atomic_rewrite_settings(settings_path, data)
    return HookPruneResult(
        settings_path=settings_path,
        removed_per_event=removed_per_event,
        total_removed=total,
    )


def _entry_is_broken_aelf_hook(
    settings_path: Path, entry: object,
) -> bool:
    """True iff `entry` is an `aelf-*`-basename hook whose program is broken.

    The settings entry shape is::

        {"hooks": [{"type": "command", "command": "<cmd>"}, ...]}

    An entry is considered broken when ANY of its inner command hooks
    have an `aelf-*` basename and resolve to `status="broken"`. One
    broken inner command condemns the whole parent entry — Claude
    Code's hook runner spawns each inner command per fire, so leaving
    a half-broken parent in place would still emit ENOENT per call.
    """
    if not isinstance(entry, dict):
        return False
    entry_dict = cast(dict[str, object], entry)
    inner = entry_dict.get(_INNER_HOOKS_KEY)
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
        stripped = cmd.strip()
        if not stripped:
            continue
        first = stripped.split(maxsplit=1)[0]
        if not Path(first).name.startswith(_AELF_HOOK_BASENAME_PREFIX):
            continue
        finding = _inspect_command(settings_path, "<prune>", cmd)
        if finding.status == "broken":
            return True
    return False


_HOOK_TYPE_COMMAND: Final[str] = "command"


def _atomic_rewrite_settings(path: Path, data: dict[str, object]) -> None:
    """Atomically replace `path` with the new JSON. Mirrors setup._atomic_write.

    Pulled into doctor so the prune helper can rewrite without taking
    an import dependency on a private setup symbol. The format matches
    setup's writer byte-for-byte (indent=2, ensure_ascii=False, trailing
    newline) so a setup→prune→setup round trip stays diff-stable.
    """
    import os
    import tempfile

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


def format_report(report: DoctorReport) -> str:
    """Render a DoctorReport as a human-readable string for the CLI."""
    lines: list[str] = []
    if not report.scopes_scanned:
        lines.append(
            "no settings.json found at user or project scope -- nothing to check"
        )
        # Still render the telemetry sections if paths are available.
        _format_telemetry_section(report, lines)
        _format_user_prompt_submit_telemetry_section(report, lines)
        # #236: render the missing-dep block too — the install-broken
        # case is exactly when settings.json may be absent.
        _format_missing_runtime_deps_section(report, lines)
        # #696: HRR block is independent of settings.json scan.
        _format_hrr_section(report, lines)
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
    _format_telemetry_section(report, lines)
    _format_user_prompt_submit_telemetry_section(report, lines)
    _format_missing_runtime_deps_section(report, lines)
    _format_missing_auto_capture_section(report, lines)
    _format_legacy_schema_section(report, lines)
    _format_hrr_section(report, lines)
    return "\n".join(lines)


# Default-on auto-capture hook basenames the v2.1 `aelf setup` wires
# (#529 commit 3c27f45). Mirrors `setup.TRANSCRIPT_LOGGER_SCRIPT_NAME`,
# `setup.COMMIT_INGEST_SCRIPT_NAME`, `setup.SESSION_START_HOOK_SCRIPT_NAME`.
# Duplicated here to keep the doctor module dependency-free of setup's
# install primitives; if the names drift, the test in
# tests/test_doctor.py::test_auto_capture_basenames_match_setup catches it.
_AUTO_CAPTURE_HOOK_BASENAMES: Final[tuple[str, ...]] = (
    "aelf-transcript-logger",
    "aelf-commit-ingest",
    "aelf-session-start-hook",
    # #582: session-end correction-lock prompt. Default-on since the
    # Stop hook landed.
    "aelf-stop-hook",
)


def _check_auto_capture_hooks(
    findings: list[CommandFinding],
) -> list[str]:
    """Return basenames of v2.1 default-on hooks absent from all scanned scopes.

    A hook is "present" if any finding's command string contains the
    basename. Substring match is intentional: the command may be a bare
    basename (PATH-resolved pipx install) or an absolute path
    (project venv); both should count as installed (#557).
    """
    missing: list[str] = []
    for basename in _AUTO_CAPTURE_HOOK_BASENAMES:
        if not any(basename in f.command for f in findings):
            missing.append(basename)
    return missing


def _format_missing_auto_capture_section(
    report: DoctorReport, lines: list[str],
) -> None:
    """Append the v2.1 auto-capture nag block (#557) to `lines`.

    Quiet when no settings.json was scanned (the install-broken case
    is already covered by the no-scopes-scanned message at line 1 of
    format_report) or when every default-on hook is present.
    """
    if not report.scopes_scanned:
        return
    if not report.missing_auto_capture_hooks:
        return
    lines.append("")
    lines.append(
        "auto-capture hooks not installed (v2.1+, #529). missing: "
        + ", ".join(report.missing_auto_capture_hooks)
    )
    lines.append(
        "fix: re-run 'aelf setup' to wire transcript-ingest, "
        "commit-ingest, session-start, and stop-hook (default-on "
        "since v2.1 / #582). to opt out per-hook: "
        "`aelf setup --no-transcript-ingest --no-commit-ingest "
        "--no-session-start --no-stop-hook`."
    )


def _check_legacy_schema_dbs(
    *,
    projects_dir: Path | None = None,
) -> list[LegacySchemaDB]:
    """Return per-project DBs that are on the pre-v1.x schema (no `origin`).

    Scans `~/.aelfrice/projects/*/memory.db` (or `projects_dir` override).
    A DB is flagged when ALL of the following hold:
      1. The `beliefs` table exists.
      2. No column named `origin` is present.
      3. `SELECT COUNT(*) FROM beliefs` returns > 0.

    Opens each DB read-only (`file:...?mode=ro`) to avoid accidental writes.
    Any DB that raises a connection or query error is silently skipped —
    doctor is diagnostic only.
    """
    root = projects_dir if projects_dir is not None else _AELFRICE_PROJECTS_DIR
    if not root.is_dir():
        return []

    results: list[LegacySchemaDB] = []
    now = time.time()

    for db_path in sorted(root.glob("*/memory.db")):
        try:
            uri = f"file:{db_path}?mode=ro"
            con = sqlite3.connect(uri, uri=True)
        except Exception:  # noqa: BLE001
            continue
        try:
            cur = con.execute("PRAGMA table_info(beliefs)")
            columns = cur.fetchall()
            if not columns:
                # No beliefs table — skip (uninitialised or unrelated DB).
                continue
            col_names = {row[1] for row in columns}  # row[1] is the column name
            if "origin" in col_names:
                # Modern schema — skip.
                continue
            # Legacy schema. Skip empty DBs — uninteresting.
            (row_count,) = con.execute("SELECT COUNT(*) FROM beliefs").fetchone()
            if row_count == 0:
                continue
            mtime = db_path.stat().st_mtime
            idle_days = int((now - mtime) / 86400)
            results.append(
                LegacySchemaDB(
                    path=db_path,
                    row_count=int(row_count),
                    idle_days=idle_days,
                )
            )
        except Exception:  # noqa: BLE001
            pass
        finally:
            con.close()

    return results


# Default minimum idle period before a per-project DB is treated as
# dormant by `_check_dormant_dbs` (#594). Conservative: a project a
# user touches monthly stays out of the prune list. Override via the
# `idle_days` kwarg on the scanner (CLI: `--idle-days N`).
DORMANT_IDLE_DAYS_DEFAULT: Final[int] = 30


def _check_dormant_dbs(
    *,
    projects_dir: Path | None = None,
    idle_days: int = DORMANT_IDLE_DAYS_DEFAULT,
) -> list[DormantDB]:
    """Return per-project DBs whose mtime is at least `idle_days` days old.

    Scans `~/.aelfrice/projects/*/memory.db` (or `projects_dir`
    override). A DB is flagged when ALL of the following hold:
      1. The memory.db file exists and stat() succeeds.
      2. (now - mtime) / 86400 >= `idle_days`.

    Schema is irrelevant — both legacy and modern DBs can be dormant.
    Empty DBs (zero beliefs, or no `beliefs` table at all) are still
    flagged when dormant; an idle empty DB has zero forward value and
    is the cleanest prune target.

    Opens each DB read-only (`file:...?mode=ro`) for the row-count
    probe; any DB whose row-count probe fails (corrupt, no `beliefs`
    table) is reported with `row_count=0` rather than skipped — the
    file is still a pruneable artefact even when the schema is unreadable.
    """
    root = projects_dir if projects_dir is not None else _AELFRICE_PROJECTS_DIR
    if not root.is_dir():
        return []

    results: list[DormantDB] = []
    now = time.time()
    threshold_seconds = idle_days * 86400

    for db_path in sorted(root.glob("*/memory.db")):
        try:
            stat = db_path.stat()
        except OSError:
            continue
        age_seconds = now - stat.st_mtime
        if age_seconds < threshold_seconds:
            continue

        row_count = 0
        try:
            uri = f"file:{db_path}?mode=ro"
            con = sqlite3.connect(uri, uri=True)
        except Exception:  # noqa: BLE001
            con = None
        if con is not None:
            try:
                cur = con.execute("PRAGMA table_info(beliefs)")
                if cur.fetchall():
                    (rc,) = con.execute(
                        "SELECT COUNT(*) FROM beliefs"
                    ).fetchone()
                    row_count = int(rc)
            except Exception:  # noqa: BLE001
                pass
            finally:
                con.close()

        results.append(
            DormantDB(
                path=db_path,
                row_count=row_count,
                idle_days=int(age_seconds / 86400),
                size_bytes=int(stat.st_size),
            )
        )

    return results


def _auto_migrate_legacy_dbs(
    legacy_dbs: list[LegacySchemaDB],
) -> tuple[list[MigratedDB], list[FailedMigrateDB]]:
    """Run `migrate_in_place` on every detected legacy DB (#593).

    Silent per the operator decision: no per-DB prompt, no banner.
    Successes populate `MigratedDB` rows for the format pass to render
    as one-line summaries. Failures populate `FailedMigrateDB` rows
    that fall back to the residual nag (the original #589 flow).

    Each migration is independent — a failure on DB N doesn't stop
    the pass on DB N+1.
    """
    if not legacy_dbs:
        return [], []

    # Local import keeps the doctor → migrate dep one-way and lets
    # tests stub the migrate path independently.
    from aelfrice.migrate import migrate_in_place

    migrated: list[MigratedDB] = []
    failed: list[FailedMigrateDB] = []
    for entry in legacy_dbs:
        try:
            report = migrate_in_place(entry.path)
        except Exception as exc:  # noqa: BLE001  # silent per #593 contract
            failed.append(
                FailedMigrateDB(path=entry.path, reason=type(exc).__name__)
            )
            continue
        migrated.append(
            MigratedDB(
                path=report.db_path,
                backup_path=report.backup_path,
                row_count=report.counts.inserted_beliefs,
                duration_ms=report.duration_ms,
            )
        )
    return migrated, failed


def _format_legacy_schema_section(
    report: DoctorReport, lines: list[str],
) -> None:
    """Append legacy-schema migration outcomes to `lines` (#589, #593).

    Quiet when no legacy DBs were detected this run (parity with #557).
    Otherwise renders:
      * one summary line per successfully auto-migrated DB (`migrated_dbs`),
      * a residual nag block for DBs that auto-migrate could not action
        (`failed_migrate_dbs`).

    The pre-#593 detection-only nag is gone — `aelf doctor` now actions
    the migration in place rather than asking the operator to run a
    follow-up command.
    """
    for entry in report.migrated_dbs:
        lines.append("")
        lines.append(
            f"migrated {entry.path}: {entry.row_count:,} beliefs, "
            f"{entry.duration_ms}ms (backup at {entry.backup_path})"
        )
    if report.failed_migrate_dbs:
        lines.append("")
        lines.append(
            "legacy-schema auto-migrate FAILED for the following DB(s):"
        )
        for entry in report.failed_migrate_dbs:
            lines.append(f"  {entry.path} ({entry.reason})")
        lines.append(
            "fix: investigate manually with "
            "`aelf migrate --from <path> --apply`; the legacy file is "
            "untouched after a failed auto-migrate."
        )


def _format_missing_runtime_deps_section(
    report: DoctorReport, lines: list[str],
) -> None:
    """Append the [FAIL] missing-runtime-dep block (#236) to `lines`."""
    if not report.missing_runtime_deps:
        return
    lines.append("")
    for dep in report.missing_runtime_deps:
        lines.append(f"[FAIL] missing runtime dep: {dep}")
    lines.append(
        "fix: reinstall aelfrice to pull in all declared deps: "
        "`uv tool upgrade aelfrice` or `pip install --upgrade aelfrice`"
    )


def _format_hrr_section(
    report: DoctorReport, lines: list[str],
) -> None:
    """Append the HRR persist-state block to `lines` (#696).

    Quiet when ``hrr_persist_state`` was not populated (no store probe
    requested). Renders three rows under an ``HRR`` header:
    ``persist_enabled``, ``on_disk_bytes``, ``last_build_seconds``.
    """
    if report.hrr_persist_state is None:
        return
    state = report.hrr_persist_state
    lines.append("")
    lines.append("HRR")
    enabled = bool(state.get("enabled", False))
    reason = state.get("reason")
    if enabled:
        enabled_str = "true"
    else:
        reason_suffix = f" ({reason})" if reason else ""
        enabled_str = f"false{reason_suffix}"
    lines.append(f"  persist_enabled:      {enabled_str}")
    on_disk = int(state.get("on_disk_bytes", 0))  # type: ignore[arg-type]
    lines.append(f"  on_disk_bytes:        {on_disk}")
    lbs = state.get("last_build_seconds")
    if lbs is None:
        lbs_str = "n/a"
    else:
        lbs_str = f"{float(lbs):.3f}"  # type: ignore[arg-type]
    lines.append(f"  last_build_seconds:   {lbs_str}")


def _format_telemetry_section(report: DoctorReport, lines: list[str]) -> None:
    """Append the search_tool_hook telemetry block to `lines` (in-place)."""
    if report.search_tool_telemetry_path is None:
        return
    lines.append("")
    lines.append("search_tool_hook telemetry:")
    lines.append(f"  file: {report.search_tool_telemetry_path}")
    if report.search_tool_telemetry_corrupt:
        lines.append(
            "  status: CORRUPT — file exists but contains malformed JSON; "
            "delete it to reset the ring buffer."
        )
    elif report.search_tool_telemetry is None:
        lines.append("  no fires recorded")
    else:
        st = report.search_tool_telemetry
        lines.append(f"  fires: {st.fire_count}")
        lines.append(f"  latency p50: {st.p50_ms:.1f} ms")
        lines.append(f"  latency p95: {st.p95_ms:.1f} ms")
        lines.append(
            f"  noise rate:  {st.noise_rate:.1%} "
            f"(fires with zero L0+L1 hits)"
        )


def _format_user_prompt_submit_telemetry_section(
    report: DoctorReport, lines: list[str],
) -> None:
    """Append the user_prompt_submit_hook telemetry block to `lines` (#218 AC4)."""
    if report.user_prompt_submit_telemetry_path is None:
        return
    lines.append("")
    lines.append("user_prompt_submit_hook telemetry:")
    lines.append(f"  file: {report.user_prompt_submit_telemetry_path}")
    if report.user_prompt_submit_telemetry_corrupt:
        lines.append(
            "  status: CORRUPT — file exists but contains malformed JSON; "
            "delete it to reset the ring buffer."
        )
    elif report.user_prompt_submit_telemetry is None:
        lines.append("  no fires recorded")
    else:
        st = report.user_prompt_submit_telemetry
        lines.append(f"  fires: {st.fire_count}")
        lines.append(f"  injection size p50: {st.p50_chars:.0f} chars")
        lines.append(f"  injection size p95: {st.p95_chars:.0f} chars")
        lines.append(
            f"  dedup collapse rate (median): {st.median_collapse_rate:.2f}x "
            f"(n_returned / n_unique_hashes; 1.0 = no duplicates)"
        )


# ---------------------------------------------------------------------------
# classify-orphans pass (issue #206)
# ---------------------------------------------------------------------------

# Approximate Haiku pricing as of 2025-Q1.  Used only for the cost
# estimate in the CLI report; not invoiced, not sent to Anthropic.
_HAIKU_INPUT_COST_PER_TOKEN: Final[float] = 0.80 / 1_000_000   # $0.80 / MTok
_HAIKU_OUTPUT_COST_PER_TOKEN: Final[float] = 4.00 / 1_000_000  # $4.00 / MTok


@dataclass
class OrphanRunReport:
    """Summary of one `classify_orphans` pass.

    `orphans_found` is the raw count before `max_n` is applied.
    `classified` is the number updated in the store.
    `skipped` is orphans whose LLM result was invalid or non-persisting.
    `dry_run` flags whether any DB writes occurred.
    `type_dist_before` and `type_dist_after` are {type: count} snapshots.
    `telemetry` is the raw Haiku token accounting.
    """

    orphans_found: int = 0
    classified: int = 0
    skipped: int = 0
    dry_run: bool = False
    type_dist_before: dict[str, int] = field(default_factory=dict)
    type_dist_after: dict[str, int] = field(default_factory=dict)
    # BatchTelemetry is imported lazily below; store raw token counts here.
    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0
    fallbacks: int = 0
    model: str = ""


def classify_orphans(
    store: "MemoryStore",
    *,
    api_key: str,
    model: str,
    max_tokens: int,
    max_n: int | None = None,
    dry_run: bool = False,
    sdk_module: object = None,
) -> OrphanRunReport:
    """Find un-typed low-confidence beliefs and re-classify via Haiku batch.

    Orphan definition (both signals required):
      - type = 'unknown' OR type IS NULL  (never successfully typed)
      - alpha + beta <= 2                  (no feedback ever applied)

    The function reuses `aelfrice.llm_classifier.classify_batch` — the
    same path `aelf onboard --llm-classify` uses.  No new LLM client is
    introduced.

    When `dry_run=True` the orphan set is found and counted but neither
    network calls nor DB writes are made.

    `max_n` caps classifications per run. Recommended: 500 per run when
    the store is large; None (the default) processes all orphans.
    """
    from aelfrice.llm_classifier import (  # local to avoid circular imports
        BatchTelemetry,
        CandidateInput,
        classify_batch,
    )
    from aelfrice.models import ORIGIN_AGENT_INFERRED

    report = OrphanRunReport(dry_run=dry_run, model=model)
    report.type_dist_before = store.count_beliefs_by_type()

    # Count total orphans before applying max_n.
    all_orphans = store.find_orphan_beliefs()
    report.orphans_found = len(all_orphans)

    # Apply max_n cap for the actual processing set.
    to_process = all_orphans[:max_n] if max_n is not None else all_orphans

    if dry_run or not to_process:
        # Dry-run: report pre-snapshot only; no network, no writes.
        return report

    # Build classifier inputs from the orphan beliefs.  Use the belief id
    # as the source tag so parse failures are traceable.
    inputs = [
        CandidateInput(index=i, text=b.content, source=b.id)
        for i, b in enumerate(to_process)
    ]

    batch = classify_batch(
        inputs,
        api_key=api_key,
        model=model,
        max_tokens=max_tokens,
        sdk_module=sdk_module,  # type: ignore[arg-type]
    )

    # Accumulate telemetry.
    tel: BatchTelemetry = batch.telemetry
    report.input_tokens = tel.input_tokens
    report.output_tokens = tel.output_tokens
    report.requests = tel.requests
    report.fallbacks = tel.fallbacks

    # Auth failure: propagate so CLI can exit 1.
    if batch.auth_error is not None:
        from aelfrice.llm_classifier import LLMAuthError  # noqa: PLC0415
        raise LLMAuthError(batch.auth_error)

    # Token cap: propagate so CLI can exit 1.
    if batch.token_cap_exceeded:
        from aelfrice.llm_classifier import LLMTokenCapExceeded  # noqa: PLC0415
        raise LLMTokenCapExceeded(
            consumed=batch.token_cap_consumed,
            cap=max_tokens,
        )

    # When the whole batch fell back to regex, skip_all — the regex
    # fallback path is designed for scan_repo's fresh-insert flow, not
    # for updating existing beliefs with already-set prior weights.
    if batch.fallback_used:
        report.skipped = len(to_process)
        report.type_dist_after = store.count_beliefs_by_type()
        return report

    # Apply valid classifications back to the store.
    by_index: dict[int, object] = {c.index: c for c in batch.classifications}
    for i, belief in enumerate(to_process):
        cls_any = by_index.get(i)
        if cls_any is None:
            report.skipped += 1
            continue
        from aelfrice.llm_classifier import CandidateClassification  # noqa: PLC0415
        cls: CandidateClassification = cls_any  # type: ignore[assignment]
        if not cls.persist:
            report.skipped += 1
            continue
        # Update only the type and origin fields; leave alpha/beta, lock,
        # demotion_pressure, and timestamps intact so prior work is not lost.
        updated = _belief_with_type(belief, cls.belief_type, ORIGIN_AGENT_INFERRED)
        store.update_belief(updated)
        report.classified += 1

    report.type_dist_after = store.count_beliefs_by_type()
    return report


def _belief_with_type(
    b: "object", new_type: str, new_origin: str
) -> "object":
    """Return a copy of `b` (a Belief) with `type` and `origin` replaced.

    Uses dataclasses.replace-style reconstruction so we don't depend on
    Belief being mutable.  The import is local to keep the doctor module
    importable without triggering the full models chain at top level.
    """
    from aelfrice.models import Belief  # noqa: PLC0415
    belief: Belief = b  # type: ignore[assignment]
    return Belief(
        id=belief.id,
        content=belief.content,
        content_hash=belief.content_hash,
        alpha=belief.alpha,
        beta=belief.beta,
        type=new_type,
        lock_level=belief.lock_level,
        locked_at=belief.locked_at,
        demotion_pressure=belief.demotion_pressure,
        created_at=belief.created_at,
        last_retrieved_at=belief.last_retrieved_at,
        session_id=belief.session_id,
        origin=new_origin,
    )


def format_orphan_report(report: OrphanRunReport) -> str:
    """Render an OrphanRunReport as a human-readable string for the CLI."""
    lines: list[str] = []
    prefix = "[dry-run] " if report.dry_run else ""
    lines.append(f"{prefix}classify-orphans: {report.orphans_found} orphan(s) found")
    if report.dry_run:
        lines.append(
            "(dry-run: no LLM calls made, no DB writes performed)"
        )
    else:
        lines.append(
            f"  classified: {report.classified}  "
            f"skipped: {report.skipped}"
        )
    lines.append("")
    lines.append("type distribution before:")
    _append_type_dist(lines, report.type_dist_before)
    if not report.dry_run:
        lines.append("")
        lines.append("type distribution after:")
        _append_type_dist(lines, report.type_dist_after)
        lines.append("")
        total_tokens = report.input_tokens + report.output_tokens
        cost = (
            report.input_tokens * _HAIKU_INPUT_COST_PER_TOKEN
            + report.output_tokens * _HAIKU_OUTPUT_COST_PER_TOKEN
        )
        lines.append(
            f"tokens: input={report.input_tokens} "
            f"output={report.output_tokens} "
            f"total={total_tokens} "
            f"requests={report.requests} "
            f"fallbacks={report.fallbacks}"
        )
        lines.append(f"estimated cost: ${cost:.4f} (model={report.model})")
    return "\n".join(lines)


def _append_type_dist(lines: list[str], dist: dict[str, int]) -> None:
    if not dist:
        lines.append("  (empty)")
        return
    for t, n in sorted(dist.items()):
        lines.append(f"  {t}: {n}")


# ---------------------------------------------------------------------------
# gc-orphan-feedback pass (issue #223)
# ---------------------------------------------------------------------------


@dataclass
class OrphanFeedbackReport:
    """Summary of one `gc_orphan_feedback` pass.

    `orphans_found` is the number of `feedback_history` rows whose
    `belief_id` no longer resolves in `beliefs`. `deleted` is the
    number actually removed (zero on dry-run).
    """

    orphans_found: int = 0
    deleted: int = 0
    dry_run: bool = True


def gc_orphan_feedback(
    store: "MemoryStore",
    *,
    dry_run: bool = True,
) -> OrphanFeedbackReport:
    """Identify and (with `dry_run=False`) delete `feedback_history`
    rows whose `belief_id` no longer resolves in `beliefs`. Issue #223.

    Pre-#283 re-ingest could leave a feedback row pointing at a
    deleted belief: the same content_hash got a fresh belief_id, the
    old row was dropped, but feedback_history kept the dangling
    reference. The mechanism is plugged going forward by the UNIQUE
    `content_hash` constraint plus `insert_or_corroborate`; this
    pass cleans the residue.

    Recovery is not attempted: feedback_history stores only
    `belief_id`, not `content_hash`, so the original target's content
    is unrecoverable. The pass deletes rather than re-links.

    `dry_run=True` (the default) counts without modifying the store.
    """
    report = OrphanFeedbackReport(dry_run=dry_run)
    report.orphans_found = store.count_orphan_feedback_events()
    if dry_run or report.orphans_found == 0:
        return report
    report.deleted = store.delete_orphan_feedback_events()
    return report


def format_orphan_feedback_report(report: OrphanFeedbackReport) -> str:
    """Human-readable rendering of `gc_orphan_feedback` output."""
    lines: list[str] = []
    lines.append(
        f"orphan feedback rows: {report.orphans_found}"
    )
    if report.dry_run:
        if report.orphans_found == 0:
            lines.append("nothing to do.")
        else:
            lines.append(
                "dry-run; re-run with --apply to delete these rows."
            )
    else:
        lines.append(f"deleted: {report.deleted}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# promote-retention pass (issue #290 phase-3)
# ---------------------------------------------------------------------------

# Promotion thresholds from docs/belief_retention_class.md §4.
# A snapshot belief is promoted to ``fact`` once it has been
# corroborated at least N times across at least M distinct sessions
# with no inbound CONTRADICTS edge. Constants are module-level so
# tests can reference the canonical values.
PROMOTE_RETENTION_MIN_CORROBORATIONS: Final[int] = 3
PROMOTE_RETENTION_MIN_SESSIONS: Final[int] = 2

# Wire-format source string for the synthetic feedback_history row
# written on promotion. Operators auditing feedback_history grep on
# this; do not rename without a migration.
FEEDBACK_SOURCE_RETENTION_PROMOTION: Final[str] = "retention_promotion"


@dataclass
class PromotionRunReport:
    """Summary of one `promote_retention` pass.

    `candidates_found` is the count of snapshot beliefs meeting the
    corroboration / distinct-session / no-CONTRADICTS rule, before
    `max_n` is applied. `promoted` is the number actually flipped to
    `fact`. `dry_run` flags whether DB writes occurred.
    """

    candidates_found: int = 0
    promoted: int = 0
    dry_run: bool = False


def promote_retention(
    store: "MemoryStore",
    *,
    dry_run: bool = False,
    max_n: int | None = None,
    min_corroborations: int = PROMOTE_RETENTION_MIN_CORROBORATIONS,
    min_sessions: int = PROMOTE_RETENTION_MIN_SESSIONS,
) -> PromotionRunReport:
    """Promote snapshot beliefs to ``fact`` once corroborated enough.

    Per docs/belief_retention_class.md §4 a snapshot is promoted when
    it has been re-asserted ``min_corroborations`` times across
    ``min_sessions`` distinct sessions with no inbound CONTRADICTS
    edge. Promotion writes two things per belief:

      1. ``UPDATE beliefs SET retention_class = 'fact'`` via
         ``store.set_retention_class``.
      2. A synthetic ``feedback_history`` row with
         ``source = 'retention_promotion'`` and ``valence = 0.0``.
         The neutral valence keeps the Bayesian alpha/beta untouched;
         the row exists for audit trail only.

    ``dry_run=True`` returns the candidate count without mutating.
    ``max_n`` caps how many candidates are promoted per run.
    """
    from datetime import datetime, timezone

    report = PromotionRunReport(dry_run=dry_run)
    candidates = store.find_promotable_snapshots(
        min_corroborations=min_corroborations,
        min_sessions=min_sessions,
    )
    report.candidates_found = len(candidates)

    if dry_run or not candidates:
        return report

    to_promote = candidates[:max_n] if max_n is not None else candidates
    ts = datetime.now(timezone.utc).isoformat()
    for belief in to_promote:
        store.set_retention_class(belief.id, "fact")
        store.insert_feedback_event(
            belief.id,
            valence=0.0,
            source=FEEDBACK_SOURCE_RETENTION_PROMOTION,
            created_at=ts,
        )
        report.promoted += 1
    return report


def format_promotion_report(report: PromotionRunReport) -> str:
    """Render a `PromotionRunReport` for the CLI."""
    lines: list[str] = []
    prefix = "[dry-run] " if report.dry_run else ""
    lines.append(
        f"{prefix}promote-retention: {report.candidates_found} "
        f"snapshot(s) eligible for promotion"
    )
    if report.dry_run:
        if report.candidates_found == 0:
            lines.append("nothing to do.")
        else:
            lines.append(
                "dry-run; re-run without --dry-run to promote these beliefs."
            )
    else:
        lines.append(f"  promoted: {report.promoted}")
    return "\n".join(lines)
